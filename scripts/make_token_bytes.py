#!/usr/bin/env python3
"""
从已训练好的分词器生成 `token_bytes.pt`。

为什么需要它
------------
评测用的是 **BPB（bits per byte）** 而不是 loss：

    loss 是"每个 token 多少比特"，换个词表就没法比；
    BPB 是"每个字节多少比特"，与词表大小无关，跨阶段、跨配置都能直接比。

算 BPB 需要知道**每个 token 对应多少字节**，所以要预先把这张表算出来存好。
`scripts/tok_train.py` 会在训练分词器时顺便生成它；本脚本用于
**已经有词表、但缺这张表**的情况（例如用了随包提供的词表）。

用法
----
    python scripts/make_token_bytes.py                  # 默认读 $NANOCHAT_BASE_DIR/tokenizer
    python scripts/make_token_bytes.py --tokenizer-dir tokenizer
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 token_bytes.pt（BPB 评测用）")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="含 tokenizer.pkl 的目录（默认 $NANOCHAT_BASE_DIR/tokenizer）")
    args = ap.parse_args()

    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer

    tok_dir = args.tokenizer_dir or os.path.join(get_base_dir(), "tokenizer")
    pkl = os.path.join(tok_dir, "tokenizer.pkl")
    if not os.path.exists(pkl):
        print(f"错误: 找不到 {pkl}\n"
              f"  请先把 tokenizer.pkl 放到该目录（随包的词表在 tokenizer/ 下），\n"
              f"  或用 --tokenizer-dir 指定目录。", file=sys.stderr)
        return 1

    tok = RustBPETokenizer.from_directory(tok_dir)
    vocab_size = tok.get_vocab_size()
    special = set(tok.encode_special(s) for s in tok.get_special_tokens())

    # 特殊 token 记为 0 字节（不计入 BPB）；
    # 其余用**原始字节**数，不要先 decode 成字符串 ——
    # 那样会破坏不是合法 UTF-8 的 token（字节 >= 0x80）
    token_bytes = [
        0 if tid in special else len(tok.decode_single_token_bytes(tid))
        for tid in range(vocab_size)
    ]
    out = os.path.join(tok_dir, "token_bytes.pt")
    torch.save(torch.tensor(token_bytes, dtype=torch.int32, device="cpu"), out)

    t = torch.tensor(token_bytes)
    print(f"  ✓ 已写出 {out}")
    print(f"    {vocab_size} 个 token，其中 {int((t > 0).sum())} 个非零")
    print(f"    平均 {t.float().mean():.3f} 字节/token"
          f"（词表平均；实际语料上的均值会更低，因为高频 token 更短）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
