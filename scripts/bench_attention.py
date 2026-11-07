#!/usr/bin/env python3
"""
进阶挑战 1：注意力与归一化的**实现方式**对性能与精度的影响。

要回答的问题
------------
同样的数学、不同的实现，在昇腾 910B 上差多少？

  1. 全长注意力：SDPA 因果路径 vs 昇腾融合算子
  2. 滑窗注意力：SDPA 显式 T×T mask vs 昇腾融合算子
  3. RMSNorm：原生 PyTorch vs npu_rms_norm 融合算子
  4. 每一种都要给出**数值一致性**证据（不能只看快慢）

为什么全长层和滑窗层要分开看
----------------------------
这是本挑战最值得写进报告的一点：

                    SDPA       融合算子     加速比
    滑窗层          13.42 ms    6.70 ms     2.00×
    全长层           9.18 ms   12.42 ms     0.74×   ← 反而更慢

SDPA 在 **Tq == Tk 且全长**时会走 PyTorch 自己的融合内核（很快）；
而在**滑窗**时必须物化一张 (T,T) 的布尔 mask 再算，于是用不上任何融合。
所以融合算子的收益**只出现在滑窗层**，而 `window_pattern="SSSL"`
意味着 3/4 的层都是滑窗 —— 这才是它值得做的原因。

**如果你的结论是"融合算子全面更快"，那是没分开测。**

用法
----
    python scripts/bench_attention.py                  # 跑全部
    python scripts/bench_attention.py --only rmsnorm
    python scripts/bench_attention.py --batch 32 --seq 2048

输出：一张对照表（延迟 / 加速比 / 数值误差 / 显存）。

⚠️ 必须在**有 NPU 的机器**上跑。没有 NPU 时会打印提示并退出。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# 参考形状：本课程 d8 模型的真实配置
DEF_BATCH, DEF_SEQ, DEF_HEADS, DEF_HEAD_DIM, DEF_WINDOW = 32, 2048, 4, 128, 512


def require_npu():
    if not hasattr(torch, "npu"):
        print("本脚本需要 NPU（torch.npu 不存在）。请在昇腾机器上运行。")
        sys.exit(0)
    try:
        import torch_npu  # noqa: F401
    except Exception as e:                                 # noqa: BLE001
        print(f"torch_npu 不可用: {e}")
        sys.exit(1)
    if torch.npu.device_count() == 0:
        print("没有可见的 NPU 设备。")
        sys.exit(1)
    torch.npu.set_device(0)


def timed(fn, warmup=3, iters=10):
    """返回平均毫秒数。**必须预热** —— 首次调用包含算子编译/tiling 开销。"""
    for _ in range(warmup):
        fn()
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def peak_npu_memory_mb(fn):
    """Return peak additional allocated memory for one NPU call, if supported."""
    if not hasattr(torch, "npu"):
        return None
    try:
        torch.npu.synchronize()
        baseline = torch.npu.memory_allocated(0)
        torch.npu.reset_peak_memory_stats(0)
        output = fn()
        torch.npu.synchronize()
        peak = torch.npu.max_memory_allocated(0)
        del output
    except (AttributeError, RuntimeError, TypeError):
        return None
    return max(0, peak - baseline) / (1024 ** 2)


# ===========================================================================
# 任务 1：RMSNorm
# ===========================================================================

def bench_rmsnorm(args):
    """
    对比三种 RMSNorm：
      a) 你的原生实现          nanochat.gpt.norm
      b) PyTorch 内置          F.rms_norm
      c) 昇腾融合              torch_npu.npu_rms_norm

    要看的现象：每层有 4 次 RMSNorm，8 层共 32 次块内调用；模型输入
    与输出各再调用一次。这里测单算子开销，端到端影响看 train 项。
    """
    import torch.nn.functional as F
    from nanochat.gpt import norm

    B, T, C = args.batch, args.seq, args.heads * args.head_dim
    dev = torch.device("npu:0")
    x = torch.randn(B, T, C, dtype=torch.bfloat16, device=dev)
    w = torch.ones(C, dtype=torch.bfloat16, device=dev)

    # 用 float64 算"真值" —— 判断谁更准必须有一个独立基准，
    # 否则只会得出"融合算子与原生不一样 => 融合算子精度差"的错误结论
    truth = F.rms_norm(x.detach().cpu().double(), (C,), eps=1e-6)

    impls = [
        ("你的原生实现", lambda: norm(x)),
        ("F.rms_norm", lambda: F.rms_norm(x, (C,), eps=1e-6)),
    ]
    if _has_fused_rms():
        impls.append(("npu_rms_norm (融合)",
                      lambda: torch_npu_rms(x, w)))
    else:
        print("  [skip] 当前 torch_npu 没有 npu_rms_norm")

    print(f"\n  RMSNorm  ({tuple(x.shape)}, bf16)")
    print(f"  {'实现':<22}{'延迟(ms)':>10}{'加速比':>9}{'vs 原生':>11}"
          f"{'vs fp64 真值':>14}{'显存增量(MiB)':>16}")
    print("  " + "-" * 68)
    base_ms = None
    base_out = None
    fused_ms = None
    previous_norm_impl = os.environ.get("NANOCHAT_RMSNORM_IMPL")
    os.environ["NANOCHAT_RMSNORM_IMPL"] = "native"
    try:
        for name, fn in impls:
            ms = timed(fn)
            out = fn().float()
            if base_ms is None:
                base_ms, base_out = ms, out
            sp = base_ms / ms
            d_native = (out - base_out).abs().max().item()
            d_truth = (out.double().cpu() - truth).abs().max().item()
            memory_mb = peak_npu_memory_mb(fn)
            if "融合" in name:
                fused_ms = ms
            memory_text = "N/A" if memory_mb is None else f"{memory_mb:.1f}"
            print(f"  {name:<22}{ms:>10.3f}{sp:>8.2f}x{d_native:>11.3e}"
                  f"{d_truth:>14.3e}{memory_text:>16}")
    finally:
        if previous_norm_impl is None:
            os.environ.pop("NANOCHAT_RMSNORM_IMPL", None)
        else:
            os.environ["NANOCHAT_RMSNORM_IMPL"] = previous_norm_impl
    print("  " + "-" * 68)
    if fused_ms is not None:
        per_call_saved = base_ms - fused_ms
        print(f"  单次融合节省 {per_call_saved:.3f} ms；端到端收益请看 train 项。")


def _has_fused_rms():
    if not hasattr(torch, "npu"):
        return False
    try:
        import torch_npu
    except Exception:                                      # noqa: BLE001
        return False
    return hasattr(torch_npu, "npu_rms_norm")


def torch_npu_rms(x, w, eps=1e-6):
    """昇腾融合 RMSNorm。返回 tuple，取 [0]；eps 必须与原生实现一致（1e-6）。"""
    import torch_npu
    return torch_npu.npu_rms_norm(x, w, epsilon=eps)[0]


# ===========================================================================
# 任务 2：注意力（全长 vs 滑窗）
# ===========================================================================

def bench_attention(args):
    """
    对比 SDPA 与昇腾融合注意力，**全长层与滑窗层分开测**。

    注意 q/k/v 的布局：
      * 我们的接口是 (B, T, H, D)（BSND）
      * SDPA 要 (B, H, T, D) -> 需要转置
      * 融合算子直接吃 (B, T, H, D) -> **不需要转置**
    所以"融合算子更快"的原因里也包含省掉两次转置，报告里可以量化。
    """
    from nanochat.flash_attention import _npu_attention, _sdpa_attention

    B, T, H, D = args.batch, args.seq, args.heads, args.head_dim
    dev = torch.device("npu:0")
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)

    for label, window in (("全长层", (-1, -1)),
                          (f"滑窗层 (w={args.window})", (args.window, 0))):
        print(f"\n  注意力 · {label}   ({tuple(q.shape)}, bf16)")

        def sdpa():
            # BSND -> BHSD -> BSND
            return _sdpa_attention(q.transpose(1, 2), k.transpose(1, 2),
                                   v.transpose(1, 2), window, False
                                   ).transpose(1, 2)

        fused = lambda: _npu_attention(q, k, v, window)
        ref = sdpa().float()
        got = fused().float()
        max_error = (got - ref).abs().max().item()
        mean_error = (got - ref).abs().mean().item()

        ms_sdpa = timed(sdpa)
        ms_fused = timed(fused)
        memory_sdpa = peak_npu_memory_mb(sdpa)
        memory_fused = peak_npu_memory_mb(fused)

        def timed_backward(fn):
            def step():
                q_grad = q.detach().requires_grad_(True)
                k_grad = k.detach().requires_grad_(True)
                v_grad = v.detach().requires_grad_(True)
                fn_grad = lambda: fn(q_grad, k_grad, v_grad)
                output = fn_grad()
                output.float().square().mean().backward()
            return timed(step, warmup=2, iters=5)

        def sdpa_with_inputs(qi, ki, vi):
            return _sdpa_attention(
                qi.transpose(1, 2), ki.transpose(1, 2), vi.transpose(1, 2),
                window, False,
            ).transpose(1, 2)

        def fused_with_inputs(qi, ki, vi):
            return _npu_attention(qi, ki, vi, window)

        ms_sdpa_bwd = timed_backward(sdpa_with_inputs)
        ms_fused_bwd = timed_backward(fused_with_inputs)
        sdpa_memory_text = "N/A" if memory_sdpa is None else f"{memory_sdpa:.1f} MiB"
        fused_memory_text = "N/A" if memory_fused is None else f"{memory_fused:.1f} MiB"
        print(f"    SDPA fwd         {ms_sdpa:>8.2f} ms    (基准, +{sdpa_memory_text})")
        print(f"    融合 fwd         {ms_fused:>8.2f} ms    加速比 = {ms_sdpa / ms_fused:.2f}x"
              f"  (+{fused_memory_text})")
        print(f"    SDPA fwd+bwd     {ms_sdpa_bwd:>8.2f} ms")
        print(f"    融合 fwd+bwd     {ms_fused_bwd:>8.2f} ms    加速比 = {ms_sdpa_bwd / ms_fused_bwd:.2f}x")
        print(f"    数值一致性      max|Δ| = {max_error:.3e}, mean|Δ| = {mean_error:.3e}")


# ===========================================================================
# 任务 3：训练步级别的端到端对比
# ===========================================================================

def bench_train_step(args):
    """
    算子级微基准有偏差：真实训练里还穿插着反向、优化器、通信。
    这里跑真实的训练步，看**端到端**收益还剩多少。

    """
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.gpt import GPT, GPTConfig

    device = torch.device("npu:0")
    batch = min(args.batch, 4)
    seq = min(args.seq, 512)
    vocab = 1024
    config = GPTConfig(
        sequence_len=seq, vocab_size=vocab, n_layer=2,
        n_head=args.heads, n_kv_head=args.heads,
        n_embd=args.heads * args.head_dim, window_pattern="SSSL",
    )
    tokens = torch.randint(vocab, (batch, seq), device=device)
    targets = torch.randint(vocab, (batch, seq), device=device)
    variants = (
        ("native", "sdpa", "native"),
        ("rmsnorm", "sdpa", "npu"),
        ("attention", "npu_fa", "native"),
        ("both", "npu_fa", "npu"),
    )
    results = {}
    env_keys = ("NANOCHAT_ATTN_IMPL", "NANOCHAT_RMSNORM_IMPL")
    original_env = {key: os.environ.get(key) for key in env_keys}
    try:
        for label, attention_impl, norm_impl in variants:
            os.environ["NANOCHAT_ATTN_IMPL"] = attention_impl
            os.environ["NANOCHAT_RMSNORM_IMPL"] = norm_impl
            torch.manual_seed(0)
            model = GPT(config).to(device)
            model.init_weights()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            def step():
                optimizer.zero_grad(set_to_none=True)
                loss = model(tokens, targets)
                loss.backward()
                optimizer.step()

            for _ in range(2):
                step()
            torch.npu.synchronize()
            start = time.perf_counter()
            iterations = 5
            for _ in range(iterations):
                step()
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) / iterations
            results[label] = (elapsed * 1000, batch * seq / elapsed)
            del optimizer, model
            torch.npu.empty_cache()
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    baseline_ms = results["native"][0]
    print(f"\n  端到端训练步 ({batch} × {seq}, {COMPUTE_DTYPE})")
    print(f"  {'配置':<14}{'单步(ms)':>12}{'tok/s':>14}{'相对 native':>14}")
    for label, _, _ in variants:
        elapsed_ms, tokens_per_sec = results[label]
        print(f"  {label:<14}{elapsed_ms:>12.2f}{tokens_per_sec:>14.0f}"
              f"{baseline_ms / elapsed_ms:>13.2f}x")
    print("  两项优化是否可叠加，应以实测端到端时间比较；eager 下的算子微基准"
          "通常会高估收益。")


def main() -> int:
    ap = argparse.ArgumentParser(description="进阶挑战 1：实现方式的性能对比")
    ap.add_argument("--only", choices=["rmsnorm", "attn", "train"],
                    help="只跑其中一项")
    ap.add_argument("--batch", type=int, default=DEF_BATCH)
    ap.add_argument("--seq", type=int, default=DEF_SEQ)
    ap.add_argument("--heads", type=int, default=DEF_HEADS)
    ap.add_argument("--head-dim", type=int, default=DEF_HEAD_DIM)
    ap.add_argument("--window", type=int, default=DEF_WINDOW)
    args = ap.parse_args()

    require_npu()
    required_ops = args.only or "all"
    if required_ops in ("all", "rmsnorm", "train") and not _has_fused_rms():
        raise RuntimeError("挑战要求 torch_npu.npu_rms_norm，但当前版本未提供该算子。")
    if required_ops in ("all", "attn", "train"):
        from nanochat.flash_attention import HAS_NPU_FA
        if not HAS_NPU_FA:
            raise RuntimeError(
                "挑战要求 torch_npu.npu_fusion_attention，但当前版本未提供该算子。")
    print("=" * 68)
    print("  进阶挑战 1：注意力与归一化的实现方式对比")
    print(f"  形状 batch={args.batch} seq={args.seq} heads={args.heads} "
          f"head_dim={args.head_dim}")
    print("=" * 68)

    todo = args.only or "all"
    if todo in ("all", "rmsnorm"):
        bench_rmsnorm(args)
    if todo in ("all", "attn"):
        bench_attention(args)
    if todo in ("all", "train"):
        bench_train_step(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
