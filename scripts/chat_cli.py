"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""

import argparse
import sys

from nanochat.common import (
    compute_init,
    compute_cleanup,
    autodetect_device_type,
)
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model


def normalize_token_ids(values):
    """Convert Python ints or scalar tensors to plain integer token IDs."""
    result = []
    for value in values:
        if hasattr(value, "item"):
            result.append(int(value.item()))
        else:
            result.append(int(value))
    return result


parser = argparse.ArgumentParser(description="Chat with the model")
parser.add_argument(
    "-i",
    "--source",
    type=str,
    default="sft",
    choices=["base", "sft", "rl"],
    help="Source of the model: base(续写)|sft|rl。"
         "base 走基座模型续写（不加对话模板）",
)
parser.add_argument(
    "-g",
    "--model-tag",
    type=str,
    default=None,
    help="Model tag to load",
)
parser.add_argument(
    "-s",
    "--step",
    type=int,
    default=None,
    help="Step to load",
)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="",
    help="Prompt the model, get a single response back",
)
parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    default=0.6,
    help="Temperature for generation",
)
parser.add_argument(
    "-k",
    "--top-k",
    type=int,
    default=50,
    help="Top-k sampling parameter",
)
parser.add_argument(
    "--device-type",
    type=str,
    default="",
    choices=["cuda", "cpu", "mps", "npu"],
    help="Device type: cuda|cpu|mps|npu (empty = autodetect)",
)
args = parser.parse_args()


device_type = (
    autodetect_device_type()
    if args.device_type == ""
    else args.device_type
)

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(
    device_type
)

model, tokenizer, meta = load_model(
    args.source,
    device,
    phase="eval",
    model_tag=args.model_tag,
    step=args.step,
)

bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

engine = Engine(model, tokenizer)


# -----------------------------------------------------------------------------
# 基座模型续写模式
# -----------------------------------------------------------------------------
if args.source == "base":
    print("\nNanoChat 续写模式（基座模型，不加对话模板）")
    print("-" * 50)
    print("输入一段上文，模型往下续写；'quit'/'exit' 退出")
    print("-" * 50)

    try:
        while True:
            if args.prompt:
                prompt = args.prompt
            else:
                try:
                    prompt = input("\n上文: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

            if prompt.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if not prompt:
                continue

            # 基座模型的文档边界使用 BOS。
            # 必须在调用生成前保存长度，因为 Engine 可能修改输入列表。
            tokens = tokenizer(prompt, prepend="<|bos|>")
            prompt_ids = normalize_token_ids(tokens)
            prompt_length = len(prompt_ids)

            # 使用副本，避免 Engine 修改原始提示词列表。
            generation_input = prompt_ids.copy()

            samples, _ = engine.generate_batch(
                generation_input,
                num_samples=1,
                max_tokens=256,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            sample_ids = normalize_token_ids(samples[0])

            # 兼容两种返回格式：
            # 1. [prompt tokens + generated tokens]
            # 2. [generated tokens]
            if (
                len(sample_ids) >= prompt_length
                and sample_ids[:prompt_length] == prompt_ids
            ):
                new_tokens = sample_ids[prompt_length:]
            else:
                new_tokens = sample_ids

            print("\n续写: ", end="", flush=True)
            print(tokenizer.decode(new_tokens))

            if args.prompt:
                break
    finally:
        compute_cleanup()

    sys.exit(0)


# -----------------------------------------------------------------------------
# SFT / RL 对话模式
# -----------------------------------------------------------------------------
print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]

try:
    while True:
        if args.prompt:
            user_input = args.prompt
        else:
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)

        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": 256,
            "temperature": args.temperature,
            "top_k": args.top_k,
        }

        response_tokens = []
        generation_input = conversation_tokens.copy()

        print("\nAssistant: ", end="", flush=True)

        for token_column, token_masks in engine.generate(
            generation_input,
            **generate_kwargs,
        ):
            token_id = normalize_token_ids([token_column[0]])[0]
            response_tokens.append(token_id)

            token_text = tokenizer.decode([token_id])
            print(token_text, end="", flush=True)

        print()

        if response_tokens and response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)

        conversation_tokens.extend(response_tokens)

        if args.prompt:
            break
finally:
    compute_cleanup()