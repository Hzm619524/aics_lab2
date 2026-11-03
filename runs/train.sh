#!/usr/bin/env bash
# 训练入口。用法：
#   bash runs/train.sh              # 默认 d6
#   bash runs/train.sh 4            # depth=4（最小，几十秒）
#   DEPTH=8 ITERS=2000 bash runs/train.sh
set -euo pipefail

DEPTH="${1:-${DEPTH:-6}}"
ITERS="${ITERS:-300}"
SEQ="${SEQ:-512}"                 # 起手语料短，用 512 就够，显存也友好
DEVICE_BATCH="${DEVICE_BATCH:-16}"
TOTAL_BATCH="${TOTAL_BATCH:-32768}"
EVAL_EVERY="${EVAL_EVERY:-50}"
TAG="${TAG:-d${DEPTH}-zh}"
DATA_DIR="${NANOCHAT_DATA_DIR:-$(pwd)/data/corpus}"

# 昇腾上必须关闭 torch.compile：TorchInductor 没有 npu 调度器，
# 一编译就崩在 `torch/_inductor/codegen/common.py: assert scheduling_ctor`。
# 代码里已按设备自动关闭，这里再显式钉一遍，让日志里能看到这个决定。
export NANOCHAT_DISABLE_COMPILE=1
export NANOCHAT_ENFORCE_EAGER=1
export NANOCHAT_DATA_DIR="$DATA_DIR"
export OMP_NUM_THREADS=1

echo "=============================================================="
echo " depth=$DEPTH  seq=$SEQ  iters=$ITERS"
echo " device_batch=$DEVICE_BATCH  total_batch=$TOTAL_BATCH  tag=$TAG"
echo " data=$DATA_DIR"
echo "=============================================================="

# 1) 自检：不要带着坏代码去排队等卡
echo "[1/3] 自检"
python scripts/selfcheck.py || {
    echo
    echo "  自检未通过，先修好再训练。见 docs/01_任务清单.md"
    exit 1
}

# 2) 分词器（只需训练一次）
if [[ ! -f "${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/tokenizer/tokenizer.pkl" ]]; then
    echo
    echo "[2/3] 训练分词器"
    python -m scripts.tok_train -- --vocab-size=16384
else
    echo
    echo "[2/3] 分词器已存在，跳过"
fi

# 3) 预训练
echo
echo "[3/3] 预训练"
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth="$DEPTH" \
    --max-seq-len="$SEQ" \
    --device-batch-size="$DEVICE_BATCH" \
    --total-batch-size="$TOTAL_BATCH" \
    --num-iterations="$ITERS" \
    --eval-every="$EVAL_EVERY" \
    --sample-every=-1 \
    --save-every=200 \
    --model-tag="$TAG" \
    --run="$TAG"

cat <<EOF

==============================================================
 训练完成
 续写：
   python -m scripts.chat_cli -i base -g $TAG -p "那少年提剑立在崖边，"
 进阶挑战 1（实现方式对比）：
   python scripts/bench_attention.py
 进阶挑战 2（模型规模对比）：
   python scripts/bench_scaling.py plan --depths 2 4 6 8
==============================================================
EOF
