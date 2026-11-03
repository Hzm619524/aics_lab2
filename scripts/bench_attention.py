#!/usr/bin/env python3
"""
进阶挑战 1：注意力与归一化的**实现方式**对性能与精度的影响。

要回答的问题
------------
同样的数学、不同的实现，在昇腾 910B 上差多少？

  1. 全长注意力：SDPA（构造 T×T mask）vs 昇腾融合算子
  2. 滑窗注意力：同上
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


# ===========================================================================
# 任务 1：RMSNorm
# ===========================================================================

def bench_rmsnorm(args):
    """
    对比三种 RMSNorm：
      a) 你的原生实现          nanochat.gpt.norm
      b) PyTorch 内置          F.rms_norm
      c) 昇腾融合              torch_npu.npu_rms_norm

    要看的现象：本模型每层 4 次 RMSNorm、8 层共 **32 次/前向**，
    eager 模式下每次都是若干个逐元素 kernel —— 所以它的绝对收益
    可能不比换注意力小。
    """
    import torch.nn.functional as F
    from nanochat.gpt import norm

    B, T, C = args.batch, args.seq, 768
    dev = torch.device("npu:0")
    x = torch.randn(B, T, C, dtype=torch.bfloat16, device=dev)
    w = torch.ones(C, dtype=torch.bfloat16, device=dev)

    # 用 float64 算"真值" —— 判断谁更准必须有一个独立基准，
    # 否则只会得出"融合算子与原生不一样 => 融合算子精度差"的错误结论
    truth = F.rms_norm(x.double(), (C,), eps=1e-6)

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
          f"{'vs fp64 真值':>14}")
    print("  " + "-" * 68)
    base_ms = None
    base_out = None
    for name, fn in impls:
        ms = timed(fn)
        out = fn().float()
        if base_ms is None:
            base_ms, base_out = ms, out
        sp = base_ms / ms
        d_native = (out - base_out).abs().max().item()
        d_truth = (out.double() - truth).abs().max().item()
        print(f"  {name:<22}{ms:>10.3f}{sp:>8.2f}x{d_native:>11.3e}"
              f"{d_truth:>14.3e}")
    print("  " + "-" * 68)

    # TODO(学生): 回答下面三个问题（报告里要有数据支撑）：
    #   1) 融合算子的加速比是多少？与你预估的一致吗？
    #   2) 融合算子与你的实现在数值上"不一样"，这是精度损失吗？
    #      看最后一列 —— 比较**各自相对 fp64 真值**的偏差，再下结论。
    #      实测结论：两者相对真值的偏差量级相同，差异只是 bf16 的固有舍入。
    #      （另注：F.rms_norm 在 torch 2.12 下不传 eps 时的默认值实测是 1e-6，
    #        不是文档写的 finfo(dtype).eps；bf16 的 finfo.eps=0.0078 会让
    #        结果差 3e-2 —— 差的是 eps，不是算子。）
    #   3) 按"每前向 32 次"折算，端到端能省多少？


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
    from nanochat.flash_attention import _sdpa_attention

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

        ms_sdpa = timed(sdpa)
        ref = sdpa().float()

        # TODO(学生): 用你在进阶挑战 1b 里实现的 _npu_attention 测同一组形状，
        #             然后填出下面这张表。要点：
        #   * 数值必须与 SDPA **参考实现**对比（不是与"上一次跑的结果"比）
        #   * 滑窗必须用 band 模式；用因果模式加窗口参数会被忽略
        #   * 报告 max|Δ|，并解释它是否只是 bf16 累加顺序造成的
        #   * 分别给出 fwd 与 fwd+bwd（训练要的是后者）
        print(f"    SDPA            {ms_sdpa:>8.2f} ms    (基准)")
        print(f"    融合算子        {'待实现':>8}      加速比 = ?")
        print(f"    数值一致性      max|Δ| = ?")

    raise NotImplementedError("任务 2：补全注意力对照表（含数值一致性验证）")


# ===========================================================================
# 任务 3：训练步级别的端到端对比
# ===========================================================================

def bench_train_step(args):
    """
    算子级微基准有偏差：真实训练里还穿插着反向、优化器、通信。
    这里跑真实的训练步，看**端到端**收益还剩多少。

    TODO(学生):
      1) 构造一个小 GPT（例如 d8 的宽度但 2 层就够），
         分别在以下配置下跑若干个训练步，记录 tok/s 与单步耗时：
            A) 全原生：NANOCHAT_ATTN_IMPL=sdpa, NANOCHAT_RMSNORM_IMPL=native
            B) 只换 RMSNorm
            C) 只换注意力
            D) 都换
      2) 用矩阵形式报告，并回答"1+1 是否等于 2"（两个优化的收益能否叠加）。
      3) 注意：eager 模式下算子级微基准的收益通常**会被放大**，
         因为 eager 的 kernel 启动开销占比更高。报告里要说明这一点。
    """
    raise NotImplementedError("任务 3：端到端训练步对比")


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
