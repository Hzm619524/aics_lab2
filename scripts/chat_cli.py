"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""
import argparse
import torch
from nanochat.common import compute_init, compute_cleanup, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft",
                    choices=['base', 'sft', 'rl'],
                    help="Source of the model: base(续写)|sft|rl。"
                         "base 走基座模型续写（不加对话模板）")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--device-type', type=str, default='',
                    choices=['cuda', 'cpu', 'mps', 'npu'],
                    help='Device type: cuda|cpu|mps|npu (empty = autodetect)')
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

# ---------------------------------------------------------------------------
# 基座模型续写模式（`-i base`）
# ---------------------------------------------------------------------------
# 与对话模式**必须区分**：基座模型没有经过对话模板训练，套上 <|user_start|>
# 之类的特殊 token 只会得到很差的结果 —— 那不是模型没训好，是接口用错了。
# 学生经常在这里误判"训练失败"（见 docs/stage_d_infer_eval.md D1）。
if args.source == "base":
    print("\nNanoChat 续写模式（基座模型，不加对话模板）")
    print("-" * 50)
    print("输入一段上文，模型往下续写；'quit'/'exit' 退出")
    print("-" * 50)
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
        # prepend "<|bos|>"：基座模型的文档边界就是 BOS
        tokens = tokenizer(prompt, prepend="<|bos|>")
        samples, _ = engine.generate_batch(
            tokens, num_samples=1, max_tokens=256,
            temperature=args.temperature, top_k=args.top_k)
        # generate_batch 返回的是"含 prompt 的完整序列"，只解码新增部分
        new_tokens = samples[0][len(tokens):]
        print("\n续写: ", end="", flush=True)
        print(tokenizer.decode(new_tokens))
        if args.prompt:
            break
    compute_cleanup()
    import sys as _sys
    _sys.exit(0)

print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # Add User message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # pop the batch dimension (num_samples=1)
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
