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
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "scaling_results.jsonl")

# 昇腾 910B bf16 的理论峰值（TFLOPS）。实测 MFU 通常 15~25%。
PEAK_TFLOPS = 376.0
TRAINING_ASPECT_RATIO = 64
TRAINING_HEAD_DIM = 128


def _format_duration(hours):
    seconds = hours * 3600
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{hours:.2f}h"


# ===========================================================================
# plan：不训练，先算清楚每个规模要多少算力
# ===========================================================================

def cmd_plan(args):
    """
    用模型自己的 estimate_flops() 算每个规模的预算。

    这一节**不需要 NPU**，也不需要训练 —— 先把账算清楚，
    再决定跑哪些规模，比盲目开跑重要得多。
    """
    if args.tokens_per_byte <= 0 or args.epochs <= 0:
        print("--tokens-per-byte 与 --epochs 必须大于 0。")
        return 1
    if args.total_batch <= 0 or args.mfu <= 0 or args.mfu > 1:
        print("--total-batch 必须为正数，--mfu 必须在 (0, 1] 内。")
        return 1
    if not args.depths or any(depth <= 0 for depth in args.depths):
        print("--depths 至少需要一个正整数。")
        return 1
    if args.measured_tok_per_sec is not None and args.measured_tok_per_sec <= 0:
        print("--measured-tok-per-sec 必须大于 0。")
        return 1
    data_dir = _resolve_data_dir(args.data_dir)
    try:
        data_bytes = _corpus_bytes(data_dir, require_train_data=True)
    except (OSError, ValueError) as error:
        print(f"  无法估算语料：{error}")
        return 1
    parquet_files = _parquet_files(data_dir)
    source = ("训练集 Parquet 的 text 列" if parquet_files else
              "训练文本文件（最后一个文件保留作验证集）")
    print(f"  数据：{data_dir}（{source}）")
    print(f"  语料：{data_bytes / 2**20:.1f} MiB"
          f"（{args.tokens_per_byte:.2f} token/字节的假设下 ≈ "
          f"{data_bytes * args.tokens_per_byte / 1e6:.2f} M token/epoch）")
    print()
    print(f"  {'depth':>6}{'params':>12}{'FLOPs/token':>15}"
          f"{'tok:param':>11}{'训练步数':>14}{'单卡估时':>11}")
    print("  " + "-" * 70)
    for d in args.depths:
        model, n_params = _meta_model_stats(d, args.seq)
        flops = model.estimate_flops()
        tokens_per_epoch = data_bytes * args.tokens_per_byte
        steps = max(1, math.ceil(tokens_per_epoch * args.epochs /
                                 args.total_batch))
        tflops = flops * steps * args.total_batch / 1e12
        # 单位：TFLOPs / (TFLOPS) = 秒，再 /3600 得小时
        hours = tflops / (PEAK_TFLOPS * args.mfu) / 3600.0
        print(f"  {d:>6}{n_params / 1e6:>11.2f}M{flops / 1e9:>14.4f}G"
              f"{tokens_per_epoch / n_params:>11.3f}{steps:>14}"
              f"{_format_duration(hours):>11}")
    print("  " + "-" * 70)
    print(f"  假设：峰值 {PEAK_TFLOPS} TFLOPS × MFU {args.mfu:.0%}；"
          f"total_batch={args.total_batch}")

    if args.measured_tok_per_sec:
        ref_model, _ = _meta_model_stats(args.depths[0], args.seq)
        measured_mfu = (args.measured_tok_per_sec * ref_model.estimate_flops()
                        / (PEAK_TFLOPS * 1e12))
        print(f"  实测吞吐 {args.measured_tok_per_sec:,.0f} tok/s 对应 MFU "
              f"{measured_mfu:.2%}（depth={args.depths[0]}），可用"
              " --mfu 重新估算其他规模。")
    else:
        print("  训练后将日志中的 tok/sec 填入 --measured-tok-per-sec，"
              "据此换算实测 MFU。")
    return 0


# ===========================================================================
# train：训练一个规模
# ===========================================================================

def cmd_train(args):
    depth = args.depth
    tag = f"scale-d{depth}"
    data_dir = _resolve_data_dir(args.data_dir)
    parquet_files = _parquet_files(data_dir)
    text_files = _text_data_files(data_dir)
    if ((parquet_files and len(parquet_files) < 2) or
            (not parquet_files and len(text_files) < 2)):
        print(f"  {data_dir} 至少需要两个 Parquet shard 或文本文件，"
              "最后一个文件作为验证集。")
        return 1
    env = dict(os.environ)
    env["NANOCHAT_DATA_DIR"] = data_dir
    if args.iters is None:
        if args.tokens_per_byte <= 0 or args.epochs <= 0:
            print("--tokens-per-byte 与 --epochs 必须大于 0。")
            return 1
        if args.total_batch <= 0:
            print("--total-batch 必须为正数。")
            return 1
        try:
            token_count = (_corpus_bytes(data_dir, require_train_data=True)
                           * args.tokens_per_byte * args.epochs)
        except (OSError, ValueError) as error:
            print(f"  无法估算训练 token 数：{error}")
            return 1
        iters = max(1, math.ceil(token_count / args.total_batch))
    else:
        iters = args.iters
        if iters <= 0:
            print("--iters 必须为正数。")
            return 1
    model_meta, n_params = _meta_model_stats(
        depth, args.seq, _tokenizer_vocab_size())
    vocab_size = model_meta.config.vocab_size
    cmd = [
        sys.executable, "-m", "scripts.base_train", "--",
        f"--depth={depth}",
        f"--aspect-ratio={TRAINING_ASPECT_RATIO}",
        f"--head-dim={TRAINING_HEAD_DIM}",
        f"--max-seq-len={args.seq}",
        "--window-pattern=SSSL",
        f"--device-batch-size={args.device_batch}",
        f"--total-batch-size={args.total_batch}",
        f"--num-iterations={iters}",
        f"--eval-every={args.eval_every}",
        "--sample-every=-1",
        f"--model-tag={tag}",
        f"--run={tag}",
    ]
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO, env=env)
    if r.returncode:
        return r.returncode

    # 训练完立刻评测，把结果写进结果文件
    bpb = _eval_bpb(tag, args, env)
    _append_result({
        "depth": depth, "model_tag": tag, "iters": iters,
        "seq": args.seq, "total_batch": args.total_batch,
        "params": n_params, "vocab_size": vocab_size,
        "n_embd": model_meta.config.n_embd,
        "n_head": model_meta.config.n_head,
        "tokens_trained": iters * args.total_batch,
        "val_bpb": bpb,
    })
    print(f"  已记录 {tag}: val_bpb={bpb}")
    return 0


def _eval_bpb(tag, args, env=None):
    """跑 base_eval 拿 val_bpb。**必须用 BPB 而不是 loss** —— 见下方说明。"""
    env = dict(os.environ) if env is None else dict(env)
    cmd = [sys.executable, "-m", "scripts.base_eval", "--",
           "--eval=bpb", f"--model-tag={tag}",
           f"--device-batch-size={args.device_batch}",
           "--split-tokens=2000000"]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
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

    latest = {}
    for record in recs:
        depth = int(record["depth"])
        if depth not in latest or record.get("timestamp", 0) >= latest[depth].get("timestamp", 0):
            latest[depth] = record
    fit_records = []
    print(f"  {'depth':>6}{'params':>14}{'val_bpb':>10}{'tokens':>14}")
    print("  " + "-" * 48)
    for record in sorted(latest.values(), key=lambda row: row["depth"]):
        params = int(_params_for_record(record))
        tokens = int(record.get("tokens_trained") or
                     record.get("iters", 0) * record.get("total_batch", 0))
        fit_records.append((int(record["depth"]), params,
                            float(record["val_bpb"]), tokens))
        print(f"  {record['depth']:>6}{params:>14,}{record['val_bpb']:>10.4f}"
              f"{tokens:>14,}")
    print("  " + "-" * 48)
    if len(fit_records) < 3:
        print("  Scaling Law 拟合至少需要 3 个不同模型规模。")
        return 1
    if len({params for _, params, _, _ in fit_records}) < 3:
        print("  至少需要 3 个不同参数规模才能拟合。")
        return 1
    if any(params <= 0 or loss <= 0 for _, params, loss, _ in fit_records):
        print("  参数量和 val_bpb 必须为正数。")
        return 1
    if any(not math.isfinite(loss) for _, _, loss, _ in fit_records):
        print("  val_bpb 必须为有限值。")
        return 1

    import numpy as np

    def fit_power_law(rows):
        log_params = np.log([row[1] for row in rows])
        log_loss = np.log([row[2] for row in rows])
        slope, intercept = np.polyfit(log_params, log_loss, 1)
        predicted = slope * log_params + intercept
        residual = float(np.square(log_loss - predicted).sum())
        total = float(np.square(log_loss - log_loss.mean()).sum())
        r_squared = 1.0 if total == 0 else 1.0 - residual / total
        return -float(slope), float(np.exp(intercept)), r_squared

    alpha, coefficient, r_squared = fit_power_law(fit_records)
    print(f"  拟合 val_bpb = {coefficient:.6g} × N^(-{alpha:.4f}); "
          f"R²={r_squared:.4f}")
    reduced_alpha, _, reduced_r2 = fit_power_law(fit_records[:-1])
    print(f"  去掉最大规模后 alpha={reduced_alpha:.4f}，"
          f"R²={reduced_r2:.4f}，Δalpha={reduced_alpha - alpha:+.4f}")

    observed_budgets = [6 * row[1] * row[3] for row in fit_records if row[3] > 0]
    compute_budget = (args.compute_budget if args.compute_budget is not None else
                      max(observed_budgets) if observed_budgets else 0)
    if compute_budget <= 0:
        print("  无训练 token 数，无法估算固定算力下的 N_opt / D_opt。")
        return 1
    ratio = args.target_token_param_ratio
    if ratio <= 0:
        print("--target-token-param-ratio 必须大于 0。")
        return 1
    optimal_params = math.sqrt(compute_budget / (6 * ratio))
    optimal_tokens = ratio * optimal_params
    print(f"  固定预算 C={compute_budget:.4e} FLOPs，采用假设 D/N={ratio:g}：")
    print(f"  N_opt≈{optimal_params:,.0f} params，D_opt≈{optimal_tokens:,.0f} tokens")
    print("  注：仅拟合 L(N) 无法从本批固定数据量实验识别数据指数；"
          "此处最优点显式采用给定的 Chinchilla token:param 假设。")


# ===========================================================================
# sweep
# ===========================================================================

def cmd_sweep(args):
    failed_depths = []
    for d in args.depths:
        print(f"\n{'=' * 68}\n  训练 depth={d}\n{'=' * 68}")
        args.depth = d
        r = cmd_train(args)
        if r:
            print(f"  depth={d} 失败（返回码 {r}），继续下一个")
            failed_depths.append(d)
    print("\n  全部完成，现在跑 `fit`：")
    print(f"    {sys.executable} {os.path.relpath(__file__)} fit")
    return 1 if failed_depths else 0


# ===========================================================================

def _config_for_depth(depth: int, seq: int = 2048, vocab: int | None = None):
    """
    构造 depth 对应的模型配置。

    宽度公式与 nanochat 一致：model_dim = ceil(aspect_ratio * depth /
    head_dim) * head_dim，当前训练参数是 64 / 128，
    与 scripts.base_train.build_model_meta 保持一致。
    如果你改了宽度公式，报告里要说明，否则与预算表对不上。
    """
    from nanochat.gpt import GPTConfig
    vocab = _tokenizer_vocab_size() if vocab is None else vocab
    n_embd = int(-(-(TRAINING_ASPECT_RATIO * depth) // TRAINING_HEAD_DIM)
                 * TRAINING_HEAD_DIM)
    n_head = n_embd // TRAINING_HEAD_DIM
    return GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=depth,
                     n_head=n_head, n_kv_head=n_head, n_embd=n_embd,
                     window_pattern="SSSL")


def _meta_model_stats(depth: int, seq: int, vocab: int | None = None):
    """Return the exact configured model and parameter count without allocating weights."""
    import torch
    from nanochat.gpt import GPT

    with torch.device("meta"):
        model = GPT(_config_for_depth(depth, seq, vocab))
    return model, model.num_scaling_params()["total"]


def _params_for_record(record):
    model, params = _meta_model_stats(
        int(record["depth"]), int(record.get("seq", 2048)),
        int(record["vocab_size"]) if "vocab_size" in record else
        _tokenizer_vocab_size(),
    )
    if "n_embd" in record and "n_head" in record and (
            model.config.n_embd != int(record["n_embd"]) or
            model.config.n_head != int(record["n_head"])):
        import torch
        from nanochat.gpt import GPT, GPTConfig
        config = GPTConfig(
            sequence_len=int(record.get("seq", 2048)),
            vocab_size=int(record["vocab_size"]) if "vocab_size" in record else
            _tokenizer_vocab_size(),
            n_layer=int(record["depth"]), n_head=int(record["n_head"]),
            n_kv_head=int(record["n_head"]), n_embd=int(record["n_embd"]),
            window_pattern="SSSL",
        )
        with torch.device("meta"):
            model = GPT(config)
        params = model.num_scaling_params()["total"]
    return params


def _tokenizer_vocab_size():
    """Read the same tokenizer used for training, including non-default vocab sizes."""
    from nanochat.tokenizer import RustBPETokenizer

    base_dir = os.environ.get("NANOCHAT_BASE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "nanochat")
    candidates = [os.path.join(base_dir, "tokenizer"),
                  os.path.join(REPO, "tokenizer")]
    for tokenizer_dir in candidates:
        if os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.pkl")):
            return RustBPETokenizer.from_directory(tokenizer_dir).get_vocab_size()
    raise FileNotFoundError(
        "tokenizer/tokenizer.pkl not found in NANOCHAT_BASE_DIR or repository")


def _resolve_data_dir(data_dir=None):
    if data_dir:
        return os.path.abspath(os.path.expanduser(data_dir))
    if os.environ.get("NANOCHAT_DATA_DIR"):
        return os.path.abspath(os.path.expanduser(
            os.environ["NANOCHAT_DATA_DIR"]))
    base_dir = os.environ.get("NANOCHAT_BASE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "nanochat")
    candidates = [os.path.join(base_dir, "base_data_climbmix"),
                  os.path.join(base_dir, "base_data"),
                  os.path.join(REPO, "data", "corpus")]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.listdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[-1])


def _parquet_files(data_dir):
    if not os.path.isdir(data_dir):
        return []
    return sorted(os.path.join(data_dir, name) for name in os.listdir(data_dir)
                  if name.endswith(".parquet") and not name.endswith(".tmp"))


def _text_data_files(data_dir):
    extensions = (".txt", ".text", ".jsonl")
    files = []
    for root, _dirs, names in os.walk(data_dir):
        for name in names:
            if name.lower().endswith(extensions):
                files.append(os.path.join(root, name))
    return sorted(files)


def _corpus_bytes(data_dir=None, require_train_data=False):
    """Count training text bytes, excluding the last validation file or shard."""
    data_dir = _resolve_data_dir(data_dir)
    all_parquet_files = _parquet_files(data_dir)
    parquet_files = all_parquet_files[:-1] if len(all_parquet_files) >= 2 else []
    if all_parquet_files and not parquet_files:
        raise ValueError(
            f"{data_dir} has only one Parquet shard; base_train needs train and validation shards")
    if parquet_files:
        import pyarrow.parquet as pq

        total = 0
        for path in parquet_files:
            parquet = pq.ParquetFile(path)
            if "text" not in parquet.schema_arrow.names:
                raise ValueError(f"{path} has no text column")
            for row_group in range(parquet.num_row_groups):
                texts = parquet.read_row_group(
                    row_group, columns=["text"]).column("text").to_pylist()
                total += sum(len(text.encode("utf-8")) for text in texts
                             if isinstance(text, str))
        if total <= 0:
            raise ValueError(f"{data_dir} contains no training text")
        return total
    text_files = _text_data_files(data_dir)
    if len(text_files) >= 2:
        total = 0
        for path in text_files[:-1]:
            if path.lower().endswith(".jsonl"):
                with open(path, encoding="utf-8-sig") as source:
                    for line_number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        text = record.get("text") if isinstance(record, dict) else None
                        if not isinstance(text, str):
                            raise ValueError(
                                f"{path}:{line_number} must contain a string 'text' field")
                        total += len(text.encode("utf-8"))
            else:
                total += os.path.getsize(path)
        if total > 0:
            return total
        raise ValueError(f"{data_dir} contains no training text")
    if require_train_data:
        raise ValueError(f"{data_dir} has no train/validation data file pair")

    total = 0
    for root, _dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith((".txt", ".text", ".jsonl")):
                total += os.path.getsize(os.path.join(root, filename))
    if total <= 0:
        raise ValueError(f"{data_dir} contains no text or training Parquet data")
    return total


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
    p.add_argument("--measured-tok-per-sec", type=float, default=None,
                   help="实测训练吞吐，用于换算参考规模的 MFU")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("train", help="训练一个规模")
    common(p)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--iters", type=int, default=None,
                   help="固定训练步数；默认按 --epochs 和语料 token 数计算")
    p.add_argument("--eval-every", type=int, default=50)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("sweep", help="依次训练多个规模")
    common(p)
    p.add_argument("--iters", type=int, default=None,
                   help="固定训练步数；默认按 --epochs 和语料 token 数计算")
    p.add_argument("--eval-every", type=int, default=50)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("fit", help="汇总并拟合")
    p.add_argument("--compute-budget", type=float, default=None,
                   help="固定 FLOPs 预算；默认使用已训练记录中的最大预算")
    p.add_argument("--target-token-param-ratio", type=float, default=20.0,
                   help="固定算力下用于估算最优点的 D/N 假设")
    p.set_defaults(func=cmd_fit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
