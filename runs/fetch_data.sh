#!/usr/bin/env bash
# 用途：一条命令从南大云盘分享链接**下载并解压**课程数据包（Stage A/B/C 语料）。
#
# 谁该跑这个：拿到仓库、准备开始训练的人（通常是 NPU 服务器上）。
# 本脚本**不需要 NPU**，也不需要 torch —— 纯下载 + 解压。
#
# 用法：
#     bash runs/fetch_data.sh                 # 下载全部，解压到 NANOCHAT_BASE_DIR
#     bash runs/fetch_data.sh --list          # 先看看云盘上有什么
#     BOX_ONLY=wuxia  bash runs/fetch_data.sh # 只下 Stage C 的小包
#     DATA_DEST=/data/pkg bash runs/fetch_data.sh
#
# 可调环境变量：
#     BOX_URL        分享链接（默认课程链接）
#     DATA_DEST      归档落盘目录（默认 <repo>/data/packed）
#     EXTRACT_TO     解压目录（默认 $NANOCHAT_BASE_DIR，没设则同 DATA_DEST）
#     BOX_ONLY       只处理名字含该子串的文件（逗号分隔，可多个）
#     BOX_EXCLUDE    跳过名字含该子串的文件（逗号分隔）
#     PYTHON         指定解释器（默认优先用仓库 .venv，其次 python3）
#
# 其余参数原样透传给 scripts/fetch_box_data.py，例如：
#     bash runs/fetch_data.sh --list
#     bash runs/fetch_data.sh --no-extract
#
# 预期输出：
#     逐个文件的下载进度，随后每个归档一行 "✓ 解压 → xxx/"
#     末尾汇总 "下载 N/N 个文件，…  解压完成，产物在 /path"
# 常见失败：
#     403/404     分享链接过期 → 用浏览器打开 BOX_URL 确认还能访问
#     连接超时    校网外需要代理 → export https_proxy=http://<host>:<port>
#     SHA-256 不符 云盘文件被重新上传 → 核对来源后再决定是否 --no-verify
# 中断了怎么办：重跑同一条命令，已下完的会跳过，下了一半的会**断点续传**。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH_PY="$REPO_ROOT/scripts/fetch_box_data.py"

BOX_URL="${BOX_URL:-https://box.nju.edu.cn/d/2b845e514e754429bb86/}"
DATA_DEST="${DATA_DEST:-$REPO_ROOT/data/packed}"
EXTRACT_TO="${EXTRACT_TO:-${NANOCHAT_BASE_DIR:-$DATA_DEST}}"

# 挑解释器：脚本是纯标准库的，只要有 python3.11+ 就能跑；
# 但仓库自己的 .venv 更省心（版本确定），所以优先用。
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "错误：找不到 python3。请装 Python 3.11+，或用 PYTHON=/path/to/python 指定。" >&2
    exit 1
fi

if [[ ! -f "$FETCH_PY" ]]; then
    echo "错误：找不到 $FETCH_PY —— 仓库不完整？" >&2
    exit 1
fi

ARGS=(--url "$BOX_URL" --dest "$DATA_DEST" --extract-to "$EXTRACT_TO")
if [[ -n "${BOX_ONLY:-}" ]]; then
    IFS=',' read -r -a only <<<"$BOX_ONLY"
    for p in "${only[@]}"; do ARGS+=(--only "$p"); done
fi
if [[ -n "${BOX_EXCLUDE:-}" ]]; then
    IFS=',' read -r -a skip <<<"$BOX_EXCLUDE"
    for p in "${skip[@]}"; do ARGS+=(--exclude "$p"); done
fi

echo "  解释器  : $PY"
exec "$PY" "$FETCH_PY" "${ARGS[@]}" "$@"
