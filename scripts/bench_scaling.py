#!/usr/bin/env python3
"""
进阶挑战 2：不同**规模**的模型 —— 从"能跑通"到"scaling law"。

要回答的问题
------------
1. 参数量、算力、验证损失之间是什么关系？能不能拟合出一条 power law？
2. **给定固定的算力预算，应该把预算花在"更大的模型"还是"更多的数据"上？**
   这是 Chinchilla 的核心问题，也是本挑战最有价值的产出。
3. 小模型（d4/d6）到底能不能用来做代码调试？它给出的结论能外推吗？

为什么这个挑战值得做
--------------------
"小模型先跑通、再上大模型"是工程常识，但**什么时候小模型的结论不可信**
并不显然。本挑战要求你把这个边界测出来 —— 这是研究报告里最容易出彩的部分。

用法
----
    # 1) 先看每个规模要多少算力（不训练，纯估算，几秒出结果）
    python scripts/bench_scaling.py plan

    # 2) 训练一个规模（需要 NPU；小规模几分钟就能出结果）
    python scripts/bench_scaling.py train --depth 4 --epochs 1

    # 3) 汇总已训练的所有规模，拟合 scaling law
    python scripts/bench_scaling.py fit

    # 4) 一键跑完全部规模（耗时长，建议后台运行）
    python scripts/bench_scaling.py sweep --depths 4 6 8 --epochs 1

结果写在 `scaling_results.jsonl`（每行一个规模），`fit` 会读它。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "scaling_results.jsonl")

# 昇腾 910B bf16 的理论峰值（TFLOPS）。实测 MFU 通常 15~25%。
PEAK_TFLOPS = 376.0


# ===========================================================================
# plan：不训练，先算清楚每个规模要多少算力
# ===========================================================================

def cmd_plan(args):
    """
    用模型自己的 estimate_flops() 算每个规模的预算。

    这一节**不需要 NPU**，也不需要训练 —— 先把账算清楚，
    再决定跑哪些规模，比盲目开跑重要得多。
    """
    from nanochat.gpt import GPTConfig

    data_bytes = _corpus_bytes()
    print(f"  语料：{data_bytes / 2**20:.1f} MiB"
          f"（{args.tokens_per_byte:.2f} token/字节的假设下 ≈ "
          f"{data_bytes * args.tokens_per_byte / 1e6:.2f} M token/epoch）")
    print()
    print(f"  {'depth':>6}{'params':>12}{'FLOPs/token':>15}"
          f"{'tok:param':>11}{'1 epoch 步数':>14}{'单卡估时':>11}")
    print("  " + "-" * 70)
    for d in args.depths:
        cfg = _config_for_depth(d)
        # 参数量的解析式在 src/budget.py 里；这里用一个极小的模型实例直接数
        from nanochat.gpt import GPT
        import torch
        with torch.device("meta"):
            model = GPT(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        flops = model.estimate_flops()
        tokens_per_epoch = data_bytes * args.tokens_per_byte
        steps = int(tokens_per_epoch * args.epochs // args.total_batch)
        tflops = flops * steps * args.total_batch / 1e12
        # 单位：TFLOPs / (TFLOPS) = 秒，再 /3600 得小时
        hours = tflops / (PEAK_TFLOPS * args.mfu) / 3600.0
        print(f"  {d:>6}{n_params / 1e6:>11.2f}M{flops / 1e9:>14.4f}G"
              f"{tokens_per_epoch / n_params:>11.1f}{steps:>14}"
              f"{hours:>10.2f}h")
    print("  " + "-" * 70)
    print(f"  假设：峰值 {PEAK_TFLOPS} TFLOPS × MFU {args.mfu:.0%}；"
          f"total_batch={args.total_batch}")

    # TODO(学生):
    #   1) 上面的"单卡估时"用的是假设的 MFU。**跑一个规模实测 tok/s**，
    #      反推真实 MFU，再用它重算整张表。报告里必须写明
    #      "估算值"与"实测值"分别是多少（这是实验报告的基本要求）。
    #   2) 参数量随 depth 大致按 d^2 增长、FLOPs/token 也按 d^2 增长，
    #      而 tok:param 是模型自己算出来的。请你解释：
    #      **为什么"固定算力下最优的模型大小"不等于"最大的模型"？**
    #   3) 挑出你能在时间预算内跑完的 3 个规模，说明选择理由。


# ===========================================================================
# train：训练一个规模
# ===========================================================================

def cmd_train(args):
    depth = args.depth
    tag = f"scale-d{depth}"
    cmd = [
        sys.executable, "-m", "scripts.base_train", "--",
        f"--depth={depth}",
        f"--max-seq-len={args.seq}",
        f"--device-batch-size={args.device_batch}",
        f"--total-batch-size={args.total_batch}",
        f"--num-iterations={args.iters}",
        f"--eval-every={args.eval_every}",
        "--sample-every=-1",
        f"--model-tag={tag}",
        f"--run={tag}",
    ]
    if args.data_dir:
        os.environ["NANOCHAT_DATA_DIR"] = args.data_dir
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode:
        return r.returncode

    # 训练完立刻评测，把结果写进结果文件
    bpb = _eval_bpb(tag, args)
    _append_result({
        "depth": depth, "model_tag": tag, "iters": args.iters,
        "seq": args.seq, "total_batch": args.total_batch,
        "val_bpb": bpb,
    })
    print(f"  已记录 {tag}: val_bpb={bpb}")
    return 0


def _eval_bpb(tag, args):
    """跑 base_eval 拿 val_bpb。**必须用 BPB 而不是 loss** —— 见下方说明。"""
    env = dict(os.environ)
    cmd = [sys.executable, "-m", "scripts.base_eval", "--",
           "--eval=bpb", f"--model-tag={tag}",
           f"--device-batch-size={args.device_batch}",
           "--split-tokens=2000000"]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    print(r.stdout[-2000:])
    m = re.search(r"val[^\n]*?bpb[^\n]*?([0-9]+\.[0-9]+)", r.stdout, re.I)
    if not m:
        m = re.search(r"([0-9]+\.[0-9]+)", r.stdout.split("val")[-1]
                      if "val" in r.stdout else "")
    return float(m.group(1)) if m else float("nan")


def _append_result(rec):
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ===========================================================================
# fit：拟合 scaling law
# ===========================================================================

def cmd_fit(args):
    """
    读 scaling_results.jsonl，拟合 L = a * N^(-alpha)。

    做法：对 log(L) 与 log(N) 做最小二乘（一次多项式拟合），
    斜率就是 -alpha。

    ⚠️ **只有 2 个点拟合不出可信的幂律。** 至少 3 个，最好 4 个以上，
       且规模要拉开（例如 d4/d6/d8/d12 而不是 d7/d8）。
       如果你的点都在同一个数量级，拟合出来的 alpha 没有意义 ——
       报告里要如实说明这一点（**这比硬拟合一个漂亮的数字更重要**）。
    """
    if not os.path.exists(RESULTS):
        print(f"  找不到 {RESULTS}，先跑 train 或 sweep。")
        return 1
    recs = [json.loads(l) for l in open(RESULTS, encoding="utf-8")
            if l.strip()]
    recs = [r for r in recs if r.get("val_bpb") == r.get("val_bpb")]  # 去 NaN
    if len(recs) < 2:
        print(f"  只有 {len(recs)} 个有效结果，至少要 2 个才能拟合。")
        return 1

    import math
    print(f"  {'depth':>6}{'params':>12}{'val_bpb':>10}")
    print("  " + "-" * 30)
    for r in sorted(recs, key=lambda x: x["depth"]):
        n = 12 * r["depth"] ** 2 + 300   # 粗略参数计数，仅用于排序展示
        print(f"  {r['depth']:>6}{'':>12}{r['val_bpb']:>10.4f}")
    print("  " + "-" * 30)

    # TODO(学生): 完成下面的事
    #   1) 用**真实的参数量**（不是上面的粗略式子）做横轴，
    #      用 numpy 对 (log N, log L) 做最小二乘，求出 alpha 与 R^2
    #   2) 画出 log-log 图（matplotlib 可选；没有环境就用文本表格）
    #   3) 计算"固定算力预算下的最优规模"：
    #        给定 C = 6*N*D（D 是 token 数），把 L(N, D) 的经验式
    #        在约束下最小化，给出 N_opt 与 D_opt
    #   4) 用 2 个规模拟合一次、去掉最大规模再拟合一次，
    #      **比较两次的 alpha** —— 这个差值就是"小模型能不能外推"的量化答案
    #
    # 提示：本仓库 src/budget.py 里有参数量的解析模型（model_dim_for_depth 等），
    #       可以直接 import 复用，不必自己推。
    raise NotImplementedError("完成 scaling law 拟合与最优化分析")


# ===========================================================================
# sweep
# ===========================================================================

def cmd_sweep(args):
    for d in args.depths:
        print(f"\n{'=' * 68}\n  训练 depth={d}\n{'=' * 68}")
        args.depth = d
        r = cmd_train(args)
        if r:
            print(f"  depth={d} 失败（返回码 {r}），继续下一个")
    print("\n  全部完成，现在跑 `fit`：")
    print(f"    {sys.executable} {os.path.relpath(__file__)} fit")
    return 0


# ===========================================================================

def _config_for_depth(depth: int, seq: int = 2048, vocab: int = 16384):
    """
    构造 depth 对应的模型配置。

    宽度公式与 nanochat 一致：model_dim = ceil(64 * depth / 128) * 128，
    与 src/budget.py 的 model_dim_for_depth 保持一致。
    如果你改了宽度公式，报告里要说明，否则与预算表对不上。
    """
    from nanochat.gpt import GPTConfig
    n_embd = int(-(-(64 * depth) // 128) * 128)
    return GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=depth,
                     n_head=4, n_kv_head=4, n_embd=n_embd,
                     window_pattern="SSSL")


def _corpus_bytes():
    """语料字节数。默认用自带的公版语料，也可以用 NANOCHAT_DATA_DIR 指向别的。"""
    d = os.environ.get("NANOCHAT_DATA_DIR") or os.path.join(REPO, "data",
                                                            "corpus")
    total = 0
    for root, _dirs, files in os.walk(d):
        for fn in files:
            if fn.endswith((".txt", ".jsonl")):
                total += os.path.getsize(os.path.join(root, fn))
    return total or 1


def main() -> int:
    ap = argparse.ArgumentParser(description="进阶挑战 2：模型规模对比与 scaling law")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--depths", type=int, nargs="*",
                       default=[4, 6, 8])
        p.add_argument("--epochs", type=float, default=1.0)
        p.add_argument("--seq", type=int, default=2048)
        p.add_argument("--total-batch", type=int, default=32768,
                       help="起手语料小，默认用较小的 batch 以保证步数足够")
        p.add_argument("--device-batch", type=int, default=16)
        p.add_argument("--mfu", type=float, default=0.20)
        p.add_argument("--tokens-per-byte", type=float, default=0.25)
        p.add_argument("--data-dir", default=None)

    p = sub.add_parser("plan", help="只估算算力，不训练")
    common(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("train", help="训练一个规模")
    common(p)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=50)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("sweep", help="依次训练多个规模")
    common(p)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=50)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("fit", help="汇总并拟合")
    p.set_defaults(func=cmd_fit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
