#!/usr/bin/env python3
"""
进阶挑战 1：比较注意力与 RMSNorm 不同实现的性能和数值误差。

分别测量全长注意力、滑窗注意力和 RMSNorm；可选地比较四种配置下
微型 GPT 的训练步耗时。结果受 CANN、torch_npu 和硬件版本影响，
应以实际运行数据为准。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


DEF_BATCH = 32
DEF_SEQ = 2048
DEF_HEADS = 4
DEF_HEAD_DIM = 128
DEF_WINDOW = 512


def require_npu():
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"torch_npu 不可用: {exc}")
        sys.exit(1)

    if not hasattr(torch, "npu"):
        print("本脚本需要 NPU（torch.npu 不存在）。请在昇腾机器上运行。")
        sys.exit(1)

    if torch.npu.device_count() == 0:
        print("没有可见的 NPU 设备。")
        sys.exit(1)

    torch.npu.set_device(0)


def timed(fn, warmup=3, iters=10):
    """预热后测量平均毫秒数。"""
    for _ in range(warmup):
        result = fn()
        del result

    torch.npu.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        result = fn()
        del result

    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def peak_npu_memory_mb(fn):
    """返回单次调用的额外峰值显存；当前 torch_npu 不支持时返回 None。"""
    allocated = getattr(torch.npu, "memory_allocated", None)
    peak_allocated = getattr(torch.npu, "max_memory_allocated", None)
    reset_peak = getattr(torch.npu, "reset_peak_memory_stats", None)

    if not all(callable(api) for api in (allocated, peak_allocated, reset_peak)):
        return None

    torch.npu.synchronize()
    baseline = allocated(0)
    reset_peak(0)
    output = fn()
    torch.npu.synchronize()
    peak = peak_allocated(0)
    del output

    return max(0, peak - baseline) / (1024 ** 2)


# ===========================================================================
# 任务 1：RMSNorm
# ===========================================================================

def bench_rmsnorm(args):
    """
    对比 nanochat 原生 norm、PyTorch F.rms_norm 和 torch_npu 融合算子。
    """
    import torch.nn.functional as F
    import nanochat.gpt as gpt_module

    norm = gpt_module.norm
    batch, seq = args.batch, args.seq
    channels = args.heads * args.head_dim
    device = torch.device("npu:0")

    x = torch.randn(
        batch, seq, channels, dtype=torch.bfloat16, device=device
    )
    weight = torch.ones(channels, dtype=torch.bfloat16, device=device)

    # 在 CPU 上计算 float64 参考值，避免 NPU double 路径降级。
    truth = F.rms_norm(
        x.detach().cpu().double(), (channels,), eps=1e-6
    )

    implementations = [
        ("原生 norm", lambda: norm(x)),
        ("F.rms_norm", lambda: F.rms_norm(x, (channels,), eps=1e-6)),
    ]

    if _has_fused_rms():
        implementations.append(
            ("npu_rms_norm 融合", lambda: torch_npu_rms(x, weight))
        )
    else:
        print("  [skip] 当前 torch_npu 没有 npu_rms_norm")

    print(f"\n  RMSNorm  ({tuple(x.shape)}, bf16)")
    print(
        f"  {'实现':<22}{'延迟(ms)':>10}{'加速比':>9}"
        f"{'vs 原生':>11}{'vs fp64 真值':>14}{'显存增量(MiB)':>16}"
    )
    print("  " + "-" * 82)

    base_ms = None
    base_output = None
    fused_ms = None
    previous_env = os.environ.get("NANOCHAT_RMSNORM_IMPL")
    previous_state = dict(gpt_module._NPU_RMS_STATE)

    # 同一进程可能已缓存过别的 RMSNorm 实现，显式切回原生并清缓存。
    os.environ["NANOCHAT_RMSNORM_IMPL"] = "native"
    gpt_module._NPU_RMS_STATE.update(resolved=False, fn=None)

    try:
        for name, fn in implementations:
            elapsed_ms = timed(fn)
            output = fn().float()

            if base_ms is None:
                base_ms = elapsed_ms
                base_output = output

            speedup = base_ms / elapsed_ms
            native_error = (output - base_output).abs().max().item()
            truth_error = (
                output.double().cpu() - truth
            ).abs().max().item()
            memory_mb = peak_npu_memory_mb(fn)

            if "融合" in name:
                fused_ms = elapsed_ms

            memory_text = "N/A" if memory_mb is None else f"{memory_mb:.1f}"
            print(
                f"  {name:<22}{elapsed_ms:>10.3f}{speedup:>8.2f}x"
                f"{native_error:>11.3e}{truth_error:>14.3e}"
                f"{memory_text:>16}"
            )
    finally:
        if previous_env is None:
            os.environ.pop("NANOCHAT_RMSNORM_IMPL", None)
        else:
            os.environ["NANOCHAT_RMSNORM_IMPL"] = previous_env

        gpt_module._NPU_RMS_STATE.update(previous_state)

    print("  " + "-" * 82)
    if fused_ms is not None:
        print(
            f"  单次融合节省 {base_ms - fused_ms:.3f} ms；"
            "端到端收益请看 train 项。"
        )


def _has_fused_rms():
    try:
        import torch_npu
    except Exception:  # noqa: BLE001
        return False

    return hasattr(torch_npu, "npu_rms_norm")


def torch_npu_rms(x, weight, eps=1e-6):
    """调用昇腾融合 RMSNorm；算子返回 tuple，取第一个元素。"""
    import torch_npu

    return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]


# ===========================================================================
# 任务 2：注意力（全长 vs 滑窗）
# ===========================================================================

def bench_attention(args):
    """比较 SDPA 与昇腾融合注意力的前向及前向反向性能。"""
    from nanochat.flash_attention import _npu_attention, _sdpa_attention

    batch, seq = args.batch, args.seq
    heads, head_dim = args.heads, args.head_dim
    device = torch.device("npu:0")

    torch.manual_seed(0)
    q = torch.randn(
        batch, seq, heads, head_dim,
        dtype=torch.bfloat16, device=device,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    cases = (
        ("全长层", (-1, -1)),
        (f"滑窗层 (w={args.window})", (args.window, 0)),
    )

    for label, window in cases:
        print(f"\n  注意力 · {label}   ({tuple(q.shape)}, bf16)")

        def sdpa():
            # SDPA 接收 BHSD；项目输入布局为 BSND。
            return _sdpa_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                window,
                False,
            ).transpose(1, 2)

        def fused():
            # 昇腾融合算子直接接收 BSND。
            return _npu_attention(q, k, v, window)

        def sdpa_with_inputs(qi, ki, vi):
            return _sdpa_attention(
                qi.transpose(1, 2),
                ki.transpose(1, 2),
                vi.transpose(1, 2),
                window,
                False,
            ).transpose(1, 2)

        def fused_with_inputs(qi, ki, vi):
            return _npu_attention(qi, ki, vi, window)

        # 数值比较以 SDPA 为参考。
        reference = sdpa().float()
        candidate = fused().float()
        max_error = (candidate - reference).abs().max().item()
        mean_error = (candidate - reference).abs().mean().item()
        del reference, candidate

        sdpa_ms = timed(sdpa)
        fused_ms = timed(fused)
        sdpa_memory = peak_npu_memory_mb(sdpa)
        fused_memory = peak_npu_memory_mb(fused)

        def timed_backward(attention_fn):
            def step():
                qi = q.detach().requires_grad_(True)
                ki = k.detach().requires_grad_(True)
                vi = v.detach().requires_grad_(True)
                output = attention_fn(qi, ki, vi)
                output.float().square().mean().backward()

            return timed(step, warmup=2, iters=5)

        sdpa_backward_ms = timed_backward(sdpa_with_inputs)
        fused_backward_ms = timed_backward(fused_with_inputs)

        sdpa_memory_text = (
            "N/A" if sdpa_memory is None else f"{sdpa_memory:.1f} MiB"
        )
        fused_memory_text = (
            "N/A" if fused_memory is None else f"{fused_memory:.1f} MiB"
        )

        print(
            f"    SDPA fwd         {sdpa_ms:>8.2f} ms"
            f"    (基准, +{sdpa_memory_text})"
        )
        print(
            f"    融合 fwd         {fused_ms:>8.2f} ms"
            f"    加速比 = {sdpa_ms / fused_ms:.2f}x"
            f"  (+{fused_memory_text})"
        )
        print(f"    SDPA fwd+bwd     {sdpa_backward_ms:>8.2f} ms")
        print(
            f"    融合 fwd+bwd     {fused_backward_ms:>8.2f} ms"
            f"    加速比 = {sdpa_backward_ms / fused_backward_ms:.2f}x"
        )
        print(
            f"    数值一致性      max|Δ| = {max_error:.3e}, "
            f"mean|Δ| = {mean_error:.3e}"
        )


# ===========================================================================
# 任务 3：训练步级别的端到端对比
# ===========================================================================

def bench_train_step(args):
    """比较四种算子组合下微型 GPT 的训练步性能。"""
    from nanochat.common import COMPUTE_DTYPE
    import nanochat.gpt as gpt_module

    GPT = gpt_module.GPT
    GPTConfig = gpt_module.GPTConfig

    device = torch.device("npu:0")
    batch = min(args.batch, 4)
    seq = min(args.seq, 512)
    vocab = 1024

    config = GPTConfig(
        sequence_len=seq,
        vocab_size=vocab,
        n_layer=2,
        n_head=args.heads,
        n_kv_head=args.heads,
        n_embd=args.heads * args.head_dim,
        window_pattern="SSSL",
    )

    tokens = torch.randint(vocab, (batch, seq), device=device)
    targets = torch.randint(vocab, (batch, seq), device=device)

    variants = (
        ("native", "sdpa", "native"),
        ("rmsnorm", "sdpa", "npu"),
        ("attention", "npu_fa", "native"),
        ("both", "npu_fa", "npu"),
    )
    env_keys = ("NANOCHAT_ATTN_IMPL", "NANOCHAT_RMSNORM_IMPL")
    original_env = {key: os.environ.get(key) for key in env_keys}
    original_rms_state = dict(gpt_module._NPU_RMS_STATE)
    results = {}

    try:
        for label, attention_impl, norm_impl in variants:
            os.environ["NANOCHAT_ATTN_IMPL"] = attention_impl
            os.environ["NANOCHAT_RMSNORM_IMPL"] = norm_impl

            # gpt.norm 缓存首次解析到的实现；每个变体开始前清除缓存。
            gpt_module._NPU_RMS_STATE.update(resolved=False, fn=None)

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

            del step, optimizer, model
            torch.npu.empty_cache()
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        gpt_module._NPU_RMS_STATE.update(original_rms_state)

    baseline_ms = results["native"][0]
    print(f"\n  端到端训练步 ({batch} × {seq}, {COMPUTE_DTYPE})")
    print(f"  {'配置':<14}{'单步(ms)':>12}{'tok/s':>14}{'相对 native':>14}")

    for label, _, _ in variants:
        elapsed_ms, tokens_per_sec = results[label]
        print(
            f"  {label:<14}{elapsed_ms:>12.2f}"
            f"{tokens_per_sec:>14.0f}"
            f"{baseline_ms / elapsed_ms:>13.2f}x"
        )

    print(
        "  两项优化是否可叠加，以端到端实测时间为准；"
        "eager 下的算子微基准通常会高估收益。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="进阶挑战 1：实现方式的性能对比"
    )
    parser.add_argument(
        "--only",
        choices=["rmsnorm", "attn", "train"],
        help="只运行指定项目",
    )
    parser.add_argument("--batch", type=int, default=DEF_BATCH)
    parser.add_argument("--seq", type=int, default=DEF_SEQ)
    parser.add_argument("--heads", type=int, default=DEF_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEF_HEAD_DIM)
    parser.add_argument("--window", type=int, default=DEF_WINDOW)
    args = parser.parse_args()

    if min(args.batch, args.seq, args.heads, args.head_dim) <= 0:
        parser.error("batch、seq、heads 和 head-dim 必须为正整数")

    if args.only in (None, "attn") and not 0 <= args.window < args.seq:
        parser.error("注意力测试要求 0 <= window < seq")

    require_npu()

    required_ops = args.only or "all"
    if required_ops in ("all", "rmsnorm", "train") and not _has_fused_rms():
        raise RuntimeError(
            "挑战要求 torch_npu.npu_rms_norm，但当前版本未提供该算子。"
        )

    if required_ops in ("all", "attn", "train"):
        from nanochat.flash_attention import HAS_NPU_FA

        if not HAS_NPU_FA:
            raise RuntimeError(
                "挑战要求 torch_npu.npu_fusion_attention，"
                "但当前版本未提供该算子。"
            )

    print("=" * 68)
    print("  进阶挑战 1：注意力与归一化的实现方式对比")
    print(
        f"  形状 batch={args.batch} seq={args.seq} "
        f"heads={args.heads} head_dim={args.head_dim}"
    )
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