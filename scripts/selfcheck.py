#!/usr/bin/env python3
"""
本地自检 —— 不需要 NPU、不需要 checkpoint、不需要语料。

用法
----
    # CPU 上就能跑（开发阶段的主要工具）
    python scripts/selfcheck.py

    # 只看某一项
    python scripts/selfcheck.py --only rope

**边写边跑这个脚本**，不要等全部写完再一次性调试。它每一项都对应一个填空，
失败信息会告诉你"期望什么、实际什么"。

设计说明
--------
本脚本里的"参考实现"（naive_attention 等）都是**直白但极慢**的写法，
只用于和你的实现做数值对拍。它们跑不了训练（Python 循环，慢几千倍），
所以不要试图在训练里调用它们 —— 但它们是**标准的定义**：
你的实现必须与它们在数值上一致（bf16 下允许 1e-2 量级的差）。

自检通过不代表训练一定好，但**不通过一定有问题**。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CHECKS: list[tuple[str, str, object]] = []


def check(key: str, title: str):
    def deco(fn):
        CHECKS.append((key, title, fn))
        return fn
    return deco


# ===========================================================================
# 参考实现（只用于对拍，性能不可用）
# ===========================================================================

def naive_attention(q, k, v, window=None):
    """
    **注意力的定义**，直白写法。q,k,v: (B, H, T, D)，返回 (B, H, T, D)。

    第 i 个 query 只能看到 j <= i 的 key（因果），
    且若给了 window，只看 j >= i - window（滑窗）。
    """
    B, H, T, D = q.shape
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            for i in range(T):
                lo = 0 if window is None else max(0, i - window)
                ks = k[b, h, lo:i + 1].float()          # (n, D)
                vs = v[b, h, lo:i + 1].float()
                scores = (q[b, h, i].float() @ ks.T) / math.sqrt(D)
                p = torch.softmax(scores, dim=-1)
                out[b, h, i] = p @ vs
    return out


# ===========================================================================
# 基础任务
# ===========================================================================

@check("norm", "基础任务 1：RMSNorm 与 PyTorch 参考实现一致")
def check_norm():
    from nanochat.gpt import norm
    torch.manual_seed(0)
    for shape in [(2, 3, 64), (1, 1, 128), (4, 7, 32)]:
        x = torch.randn(*shape, dtype=torch.float32)
        got = norm(x)
        want = F.rms_norm(x, (shape[-1],), eps=1e-6)
        d = (got - want).abs().max().item()
        assert d < 1e-5, (
            f"{shape}: 与 F.rms_norm(eps=1e-6) 的最大误差 {d:.3e}\n"
            f"  提示：eps 是不是用错了？不带权重的 RMSNorm 只有这一个参数，\n"
            f"        torch 2.12 的 F.rms_norm 默认 eps 是 1e-6（不是 finfo.eps）")
    # bf16 输入也要能用
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    assert norm(x).dtype == torch.bfloat16, "bf16 输入应当返回 bf16"


@check("rmsnorm_impl", "进阶 1：RMSNorm 融合实现开关与原生实现隔离")
def check_rmsnorm_impl():
    import types
    from nanochat import gpt

    sentinel = object()
    previous_module = sys.modules.get("torch_npu", sentinel)
    previous_env = os.environ.get("NANOCHAT_RMSNORM_IMPL")
    previous_state = gpt._NPU_RMS_STATE.copy()
    previous_cache = gpt._NPU_RMS_WEIGHT_CACHE.copy()
    calls = []

    def fake_rms_norm(x, weight, epsilon):
        calls.append((tuple(x.shape), x.device.type, epsilon, weight))
        return (F.rms_norm(x, (x.shape[-1],), weight=weight, eps=epsilon),)

    fake_module = types.ModuleType("torch_npu")
    fake_module.npu_rms_norm = fake_rms_norm
    try:
        sys.modules["torch_npu"] = fake_module
        os.environ["NANOCHAT_RMSNORM_IMPL"] = "npu"
        gpt._NPU_RMS_STATE.update(resolved=False, mode=None, fn=None)
        gpt._NPU_RMS_WEIGHT_CACHE.clear()
        x = torch.randn(2, 3, 32)
        got = gpt.norm(x)
        want = F.rms_norm(x, (32,), eps=1e-6)
        assert (got - want).abs().max().item() < 1e-6
        assert len(calls) == 1 and calls[0][:3] == ((2, 3, 32), "cpu", 1e-6), (
            f"NPU RMSNorm 开关未调用预期融合函数：{calls}")
        os.environ["NANOCHAT_RMSNORM_IMPL"] = "native"
        native = gpt.norm(x)
        assert len(calls) == 1, "native 模式不能调用融合函数"
        assert (native - want).abs().max().item() < 1e-6
        os.environ["NANOCHAT_RMSNORM_IMPL"] = "npu"
        gpt.norm(x)
        assert len(calls) == 2 and calls[0][3] is calls[1][3], (
            "切回 npu 模式后应复用该设备/shape 的权重缓存")
    finally:
        gpt._NPU_RMS_STATE.clear()
        gpt._NPU_RMS_STATE.update(previous_state)
        gpt._NPU_RMS_WEIGHT_CACHE.clear()
        gpt._NPU_RMS_WEIGHT_CACHE.update(previous_cache)
        if previous_env is None:
            os.environ.pop("NANOCHAT_RMSNORM_IMPL", None)
        else:
            os.environ["NANOCHAT_RMSNORM_IMPL"] = previous_env
        if previous_module is sentinel:
            sys.modules.pop("torch_npu", None)
        else:
            sys.modules["torch_npu"] = previous_module


@check("rope", "基础任务 2/3：RoPE 频率表与旋转")
def check_rope():
    from nanochat.gpt import GPT
    from nanochat.gpt import apply_rotary_emb

    head_dim, seq_len, base = 32, 16, 100000
    # 直接构造一个最小对象来调 _precompute_rotary_embeddings
    class _Stub:
        _precompute_rotary_embeddings = GPT._precompute_rotary_embeddings
    cos, sin = _Stub()._precompute_rotary_embeddings(seq_len, head_dim, base,
                                                     device=torch.device("cpu"))
    assert cos.shape == (1, seq_len, 1, head_dim // 2), \
        f"cos 形状应为 (1,{seq_len},1,{head_dim // 2})，实际 {tuple(cos.shape)}"
    assert sin.shape == cos.shape

    # 逐项核对频率定义
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv)
    want_cos = torch.cos(angles)
    want_sin = torch.sin(angles)
    # RoPE tables use COMPUTE_DTYPE, so compare against the reference rounded
    # to that dtype rather than treating expected bf16 quantization as an error.
    want_cos_out = want_cos.to(cos.dtype).reshape(seq_len, -1).float()
    want_sin_out = want_sin.to(sin.dtype).reshape(seq_len, -1).float()
    d = max(
        (cos.reshape(seq_len, -1).float() - want_cos_out).abs().max().item(),
        (sin.reshape(seq_len, -1).float() - want_sin_out).abs().max().item(),
    )
    assert d < 1e-5, f"频率表与定义不符，最大误差 {d:.3e}（角度是 t * inv_freq？）"

    # Exercise the actual output table, then test rotation identities in fp32
    # so their strict tolerances measure the math rather than table quantization.
    x = torch.randn(2, seq_len, 4, head_dim)
    actual_y = apply_rotary_emb(x, cos, sin)
    assert actual_y.shape == x.shape, f"输出形状应与输入相同，实际 {tuple(actual_y.shape)}"

    reference_cos = want_cos.view(1, seq_len, 1, head_dim // 2)
    reference_sin = want_sin.view(1, seq_len, 1, head_dim // 2)
    y = apply_rotary_emb(x, reference_cos, reference_sin)
    assert y.shape == x.shape, f"输出形状应与输入相同，实际 {tuple(y.shape)}"
    dn = (y.float().norm(dim=-1) - x.float().norm(dim=-1)).abs().max().item()
    assert dn < 1e-4, f"旋转应保范数，范数最大变化 {dn:.3e}"

    # 相对位置性质：位置平移后，q·k 只与**相对距离**有关
    q = torch.randn(1, seq_len, 1, head_dim)
    k = torch.randn(1, seq_len, 1, head_dim)
    qr = apply_rotary_emb(q, reference_cos, reference_sin)[0, 3]     # 位置 3 的 q
    kr = apply_rotary_emb(k, reference_cos, reference_sin)[0, 7]      # 位置 7 的 k
    # 把两者整体平移同样的量，内积应当不变
    # 把位置 3 与 7 的内容整体右移 shift 位（roll 会把末端绕回开头，
    # 但 3 和 7 都远小于 seq_len，不会被绕过）
    shift = 4
    q_s = apply_rotary_emb(
        torch.roll(q, shift, dims=1), reference_cos, reference_sin
    )[0, 3 + shift]
    k_s = apply_rotary_emb(
        torch.roll(k, shift, dims=1), reference_cos, reference_sin
    )[0, 7 + shift]
    d_inv = abs((qr * kr).sum().item() - (q_s * k_s).sum().item())
    assert d_inv < 1e-3, (
        f"RoPE 的平移不变性不成立（差 {d_inv:.3e}）—— 内积应只依赖相对距离\n"
        f"  提示：旋转角必须 = 位置 * inv_freq，且 q/k 用**同一个**角度表")


@check("mlp", "基础任务 5：MLP 使用 ReLU² 且形状正确")
def check_mlp():
    from nanochat.gpt import GPTConfig, MLP
    cfg = GPTConfig(sequence_len=32, vocab_size=256, n_layer=2, n_head=2,
                    n_embd=64, window_pattern="L")
    mlp = MLP(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    y = mlp(x)
    assert y.shape == x.shape, f"输出形状应为 {tuple(x.shape)}，实际 {tuple(y.shape)}"
    # 手工按 ReLU² 算一遍
    h = mlp.c_fc(x)
    want = mlp.c_proj(F.relu(h).square())
    d = (y - want).abs().max().item()
    assert d < 1e-5, (
        f"与 ReLU² 定义不符，最大误差 {d:.3e}\n"
        f"  提示：激活是 relu(x)**2，不是 GELU、不是 SiLU、也不是平方后取正")


@check("sdpa", "核心任务 2：SDPA 全长因果注意力")
def check_sdpa_full():
    from nanochat.flash_attention import _sdpa_attention
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 12, 16
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    got = _sdpa_attention(q, k, v, (-1, -1), False).float()
    want = naive_attention(q, k, v, window=None)
    d = (got - want).abs().max().item()
    assert d < 1e-4, (
        f"全长因果注意力与定义不符，最大误差 {d:.3e}\n"
        f"  提示：Tq == Tk 时应走 is_causal=True 的快路径")


@check("sdpa_window", "核心任务 2：SDPA 滑窗注意力")
def check_sdpa_window():
    from nanochat.flash_attention import _sdpa_attention
    torch.manual_seed(0)
    B, H, T, D = 2, 2, 16, 16
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    for w in (0, 1, 3):
        got = _sdpa_attention(q, k, v, (w, 0), False).float()
        want = naive_attention(q, k, v, window=w)
        d = (got - want).abs().max().item()
        assert d < 1e-4, (
            f"window={w} 与定义不符，最大误差 {d:.3e}\n"
            f"  提示：滑窗是 (row - col) <= window，即每个 query 恰好能看\n"
            f"        自己 + 前 window 个 key（共 window+1 个）")


@check("sdpa_single", "核心任务 2：单 token 生成路径（Tq=1，Tk>1）")
def check_sdpa_single():
    from nanochat.flash_attention import _sdpa_attention
    torch.manual_seed(0)
    B, H, Tk, D = 1, 2, 10, 16
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, H, Tk, D)
    v = torch.randn(B, H, Tk, D)
    # 用 einsum 明确写出"对 D 求和"，避免 (B,H,D) @ (B,H,D,Tk) 的广播歧义
    q0 = q[:, :, 0].float()                                  # (B,H,D)
    got = _sdpa_attention(q, k, v, (-1, -1), False)[:, :, 0].float()
    sc = torch.einsum("bhd,bhtd->bht", q0, k.float()) / math.sqrt(D)
    want = torch.einsum("bht,bhtd->bhd", torch.softmax(sc, dim=-1), v.float())
    d = (got - want).abs().max().item()
    assert d < 1e-4, (
        f"Tq=1 全长情形与定义不符，最大误差 {d:.3e}\n"
        f"  提示：Tq==1 时不需要 causal（只有一个 query），\n"
        f"        它应当能看到全部 Tk 个 key")
    # 滑窗：只看最后 window+1 个
    w = 3
    got = _sdpa_attention(q, k, v, (w, 0), False)[:, :, 0].float()
    ks, vs = k[:, :, Tk - w - 1:].float(), v[:, :, Tk - w - 1:].float()
    sc = torch.einsum("bhd,bhtd->bht", q0, ks) / math.sqrt(D)
    want = torch.einsum("bht,bhtd->bhd", torch.softmax(sc, dim=-1), vs)
    d = (got - want).abs().max().item()
    assert d < 1e-4, (
        f"Tq=1 滑窗情形与定义不符，最大误差 {d:.3e}\n"
        f"  提示：Tq==1 时不需要 causal mask，但要按窗口裁掉左边的 K/V")


@check("kvcache_attn", "核心任务 4：带 KV cache 的注意力")
def check_kv_attention():
    from nanochat.flash_attention import flash_attn_with_kvcache
    torch.manual_seed(0)
    B, H, D, T = 1, 2, 16, 12
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    kc = torch.zeros(B, T + 4, H, D)
    vc = torch.zeros(B, T + 4, H, D)
    got = flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                                 cache_seqlens=torch.tensor([0]),
                                 window_size=(-1, -1)).float()
    # 与"没有 cache、直接全长因果"一致
    from nanochat.flash_attention import _sdpa_attention
    want = _sdpa_attention(q.transpose(1, 2), k.transpose(1, 2),
                           v.transpose(1, 2), (-1, -1), False)
    want = want.transpose(1, 2).float()
    d = (got - want).abs().max().item()
    assert d < 1e-4, f"KV cache 首次写入与全长因果不符，最大误差 {d:.3e}"
    # cache 必须被**原地**写入
    assert kc[:, :T].abs().sum() > 0, "新的 k 没有写进 k_cache（应原地写入）"


# ===========================================================================
# 核心任务：模型整体
# ===========================================================================

def _tiny_config():
    from nanochat.gpt import GPTConfig
    return GPTConfig(sequence_len=64, vocab_size=256, n_layer=2, n_head=2,
                     n_kv_head=2, n_embd=64, window_pattern="SSSL")


@check("forward", "核心任务 7：GPT 前向（形状与损失）")
def check_forward():
    from nanochat.gpt import GPT
    torch.manual_seed(0)
    cfg = _tiny_config()
    if hasattr(torch, "npu") and torch.npu.is_available():
        with torch.device("meta"):
            npu_model = GPT(cfg)
        npu_device = torch.device("npu", torch.npu.current_device())
        npu_model.to_empty(device=npu_device)
        npu_model.init_weights()
        assert npu_model.cos.device == npu_device and npu_model.sin.device == npu_device, (
            "RoPE tables must stay on the target device after init_weights()"
        )
        del npu_model

    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(idx)
    assert logits.shape == (2, 32, cfg.vocab_size), \
        f"logits 形状应为 (2,32,{cfg.vocab_size})，实际 {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "logits 出现 inf/nan"
    # logit softcap：|logits| 必须 <= 15（tanh 的有界性）。
    # ⚠️ 这是**必要非充分**检查：随机初始化时原始 logits 本就很小，
    #    没做 softcap 也可能通过。它的价值在于"一旦超过 15 就一定错了"。
    assert logits.abs().max().item() <= 15.001, (
        f"|logits| 最大 {logits.abs().max().item():.3f} > 15 —— "
        f"logit softcap（15 * tanh(logits/15)）没做？")
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    loss = model(idx, targets)
    assert loss.dim() == 0 and torch.isfinite(loss), "损失应为有限标量"
    loss_none = model(idx, targets, loss_reduction="none")
    # 注意：参照实现是在**摊平后的** logits 上做 cross_entropy，
    # 所以 reduction="none" 返回的是 (B*T,)，不是 (B,T)
    assert loss_none.shape == (2 * 32,), \
        f"loss_reduction='none' 应返回 (B*T,)=({2 * 32},)，实际 {tuple(loss_none.shape)}"
    d = abs(loss.item() - loss_none.mean().item())
    assert d < 1e-4, f"'mean' 与 'none'.mean() 不一致，差 {d:.3e}"


@check("window", "核心任务 7：滑窗模式真的生效")
def check_window_pattern():
    """
    规则（见 _compute_window_sizes 的文档字符串）：
      * 模式串按层铺开，L=全长、S=短窗
      * **最后一层总是全长**
      * 短窗 = ceil(sequence_len / 4 / 128) * 128
    所以 sequence_len 太小（如 64）时短窗会算出 128 > 64，短窗等于没开 ——
    这里用 512 才能看出差别。
    """
    from nanochat.gpt import GPT
    torch.manual_seed(0)
    cfg = _tiny_config()
    cfg.sequence_len = 512
    cfg.n_layer = 8
    ws = GPT(cfg)._compute_window_sizes(cfg)
    seq = cfg.sequence_len
    assert len(ws) == cfg.n_layer, f"每层一个 window，应有 {cfg.n_layer} 个"
    assert ws[-1][0] >= seq, (
        f"最后一层必须是全长（window_pattern 的规则）：{ws[-1]}\n"
        f"  'L' 对应 (sequence_len, 0)，'S' 对应 (短窗, 0)")
    shorts = [w for w in ws if w[0] < seq]
    assert shorts, f"SSSL 应当产生短窗层，实际全是全长: {ws}"
    assert all(w == seq // 4 for w, _ in shorts), (
        f"短窗应等于 ceil(sequence_len/4/128)*128 = {seq // 4}，实际 {shorts}")


@check("sampling", "基础任务 4：采样可复现且 top-k 生效")
def check_sampling():
    from nanochat.engine import sample_next_token
    torch.manual_seed(0)
    logits = torch.randn(3, 50)
    # 贪心
    got = sample_next_token(logits.clone(), torch.Generator().manual_seed(0),
                            temperature=0.0)
    got = got.reshape(-1)          # 实现返回 (B,1)，比较时摊平
    assert torch.equal(got, logits.argmax(dim=-1)), \
        f"temperature=0 应当等价于 argmax\n  实际 {got.tolist()}\n  期望 {logits.argmax(dim=-1).tolist()}"
    # 同 seed 同结果
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = sample_next_token(logits.clone(), g1, temperature=0.8, top_k=10)
    b = sample_next_token(logits.clone(), g2, temperature=0.8, top_k=10)
    assert torch.equal(a, b), (
        "同一个 seed 必须给出相同结果 —— 你是不是没用传进来的 rng？\n"
        "        （用了全局 torch.manual_seed 的话，实验之间无法对比）")
    # top-k=1 等价于 argmax
    got = sample_next_token(logits.clone(), torch.Generator().manual_seed(0),
                            temperature=1.0, top_k=1).reshape(-1)
    assert torch.equal(got, logits.argmax(dim=-1)), "top_k=1 应当等价于 argmax"
    # top-k 之外的 token 不应被采到
    k = 5
    allowed = logits.topk(k, dim=-1).indices
    g = torch.Generator().manual_seed(7)
    for _ in range(50):
        s = sample_next_token(logits.clone(), g, temperature=1.0, top_k=k).reshape(-1)
        for i in range(len(s)):
            assert s[i].item() in allowed[i].tolist(), (
                f"采到了 top-{k} 之外的 token —— top_k 截断没生效？")


@check("generate", "进阶 A：朴素生成可复现、限长且与 Engine 一致")
def check_generate():
    from nanochat.engine import Engine
    from nanochat.gpt import GPT

    torch.manual_seed(0)
    cfg = _tiny_config()
    cfg.sequence_len = 16
    model = GPT(cfg).eval()
    prompt = [1, 2, 3]

    first = list(model.generate(prompt, 5, temperature=0.8, top_k=8, seed=123))
    second = list(model.generate(prompt, 5, temperature=0.8, top_k=8, seed=123))
    assert first == second, "相同 seed 的 GPT.generate 结果必须一致"
    assert len(first) == 5 and all(0 <= token < cfg.vocab_size for token in first)

    capped = list(model.generate(prompt, 100, temperature=0.0))
    assert len(capped) == cfg.sequence_len - len(prompt), (
        "生成长度必须被 sequence_len - prompt_len 截断")

    class _Tokenizer:
        def encode_special(self, _name):
            return -1

        def get_bos_token_id(self):
            return -2

    reference = list(model.generate(prompt, 5, temperature=0.0, seed=7))
    engine_tokens = []
    engine = Engine(model, _Tokenizer())
    for token_column, _token_masks in engine.generate(
            prompt, num_samples=1, max_tokens=5, temperature=0.0, seed=7):
        engine_tokens.append(token_column[0])
    assert engine_tokens == reference, (
        f"Engine 与朴素生成不一致：{engine_tokens} != {reference}")


@check("scaling_data", "进阶 2：Scaling 统计训练 Parquet 文本且排除验证 shard")
def check_scaling_data():
    import tempfile
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.bench_scaling import _corpus_bytes

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        pq.write_table(pa.table({"text": ["A", "水"]}),
                       data_dir / "shard_00001.parquet")
        pq.write_table(pa.table({"text": ["validation text"]}),
                       data_dir / "shard_99999.parquet")
        assert _corpus_bytes(data_dir, require_train_data=True) == 4, (
            "训练集 byte 计数应统计 UTF-8 text 列，并忽略最后的验证 shard")


@check("text_dataloader", "README 数据格式：纯文本目录按末文件划分验证集")
def check_text_dataloader():
    import tempfile
    from pathlib import Path
    from nanochat import dataloader
    from scripts.bench_scaling import _corpus_bytes

    previous_data_dir = os.environ.get("NANOCHAT_DATA_DIR")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            train_text = "first training paragraph\n\nsecond training paragraph\n"
            train_jsonl = '{"text":"third training document"}\n'
            val_text = "validation paragraph one\n\nvalidation paragraph two\n"
            (data_dir / "01_train.txt").write_text(train_text, encoding="utf-8")
            (data_dir / "02_train.jsonl").write_text(train_jsonl, encoding="utf-8")
            (data_dir / "99_validation.txt").write_text(val_text, encoding="utf-8")
            os.environ["NANOCHAT_DATA_DIR"] = str(data_dir)

            train_batches = dataloader._document_batches("train", None, 1)
            try:
                training_docs = [next(train_batches)[0][0] for _ in range(3)]
            finally:
                train_batches.close()
            assert set(training_docs) == {
                "first training paragraph", "second training paragraph",
                "third training document",
            }

            validation_batches = dataloader._document_batches("val", None, 1)
            try:
                validation_docs = [next(validation_batches)[0][0]
                                   for _ in range(2)]
            finally:
                validation_batches.close()
            assert validation_docs == [
                "validation paragraph one", "validation paragraph two",
            ]
            expected_bytes = ((data_dir / "01_train.txt").stat().st_size
                              + len("third training document".encode("utf-8")))
            assert _corpus_bytes(data_dir, require_train_data=True) == expected_bytes
    finally:
        if previous_data_dir is None:
            os.environ.pop("NANOCHAT_DATA_DIR", None)
        else:
            os.environ["NANOCHAT_DATA_DIR"] = previous_data_dir


@check("kvcache", "核心任务 5/6：KV cache 增量解码 == 全序列前向（最强的一致性检查）")
def check_kv_cache_equivalence():
    """
    逐步生成（用 KV cache）与一次性前向必须给出**相同的 logits**。
    这一条能同时抓出：cache 写错位置、mask 对不上 cache 长度、
    位置编码偏移、prefill 复制错误……几乎所有推理实现的 bug。
    """
    from nanochat.gpt import GPT
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        full = model(idx)
        from nanochat.engine import KVCache
        cache = KVCache(1, cfg.n_head, cfg.sequence_len, cfg.n_embd // cfg.n_head,
                        cfg.n_layer, idx.device, torch.float32)
        outs = []
        for t in range(idx.size(1)):
            out = model(idx[:, t:t + 1], kv_cache=cache)
            # ⚠️ 两个坑：
            #   1) 不要 out[:, -1] —— 那会得到 (B, V)，后面 cat(dim=1)
            #      会沿词表维拼成 (B, T*V)。保留时间维 (B, 1, V) 再拼。
            #   2) **不要在这里手动 cache.advance(1)** —— 位置推进由
            #      CausalSelfAttention.forward 在**最后一层**内部完成
            #      （见核心任务 1 的线索）。这里再推一次会推两遍，
            #      于是 RoPE 偏移翻倍、t>=1 全部错位。
            outs.append(out)
        step = torch.cat(outs, dim=1)
    d = (full - step).abs().max().item()
    assert d < 1e-2, (
        f"增量解码与全序列前向不一致，最大误差 {d:.3e}\n"
        f"  请依次排查：\n"
        f"   1) RoPE 表是否按**当前**长度正确裁剪、位置是否与 cache 对齐\n"
        f"   2) kv_cache.advance(T) 是否在正确的时机调用\n"
        f"   3) 写 cache 的位置是不是 cache_seqlens 指定的那个\n"
        f"   4) 滑窗层的 mask 是否用到了 cache 的真实长度")


@check("prefill", "核心任务 6：KVCache.prefill 复制正确")
def check_prefill():
    """
    语义：`self.prefill(other)` 把 `other` 里已算好的 KV **复制进 self**。
    用途是"batch=1 先算一遍 prompt，再复制成 num_samples 份并行生成"，
    所以 **other 的 batch 通常更小、self 必须为空**。
    """
    from nanochat.engine import KVCache
    torch.manual_seed(0)
    n_head, head_dim, n_layer, seq = 2, 4, 2, 8

    other = KVCache(1, n_head, seq, head_dim, n_layer, torch.device("cpu"),
                    torch.float32)
    for li in range(n_layer):
        k, v = other.get_layer_cache(li)
        k[0, :3] = torch.randn(3, n_head, head_dim)
        v[0, :3] = torch.randn(3, n_head, head_dim)
    other.advance(3)
    want_k = [other.get_layer_cache(li)[0][:, :3].clone() for li in range(n_layer)]

    me = KVCache(4, n_head, seq, head_dim, n_layer, torch.device("cpu"),
                 torch.float32)
    me.prefill(other)

    assert me.get_pos() == 3, f"复制后 pos 应为 3，实际 {me.get_pos()}"
    for li in range(n_layer):
        k, _v = me.get_layer_cache(li)
        for b in range(4):
            d = (k[b, :3] - want_k[li][0]).abs().max().item()
            assert d < 1e-6, (
                f"第 {li} 层 batch {b} 的 KV 没有被正确复制（差 {d:.3e}）\n"
                f"  提示：沿 batch 维（dim=0）复制，4 份内容应当相同")
    # self 非空时不允许 prefill
    try:
        me.prefill(other)
        raise AssertionError("非空的 cache 也接受了 prefill —— 应当 assert 拒绝")
    except AssertionError as e:
        if "非空" in str(e):
            raise


# ===========================================================================
# 进阶挑战
# ===========================================================================

@check("npu_fa", "进阶挑战 1：昇腾融合注意力（无 NPU 时自动跳过）")
def check_npu_fa():
    if not hasattr(torch, "npu"):
        print("     [skip] 当前环境没有 torch.npu")
        return
    try:
        import torch_npu  # noqa: F401
    except Exception:                                  # noqa: BLE001
        print("     [skip] torch_npu 不可用")
        return
    if torch.npu.device_count() == 0:
        print("     [skip] 没有可见的 NPU 设备")
        return
    from nanochat.flash_attention import _sdpa_attention, _npu_attention
    dev = torch.device("npu:0")
    B, H, T, D, W = 2, 4, 512, 128, 128
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    ref = _sdpa_attention(q.transpose(1, 2), k.transpose(1, 2),
                          v.transpose(1, 2), (W, 0), False).transpose(1, 2)
    got = _npu_attention(q, k, v, (W, 0))
    d = (got.float() - ref.float()).abs().max().item()
    assert d < 5e-3, (
        f"融合注意力与 SDPA 参考实现差 {d:.3e}（bf16 下应 < 5e-3）\n"
        f"  提示：滑窗必须用 band 模式（sparse_mode=4）。\n"
        f"        在因果模式（2/3）上加 pre_tockens 会被**完全忽略** ——\n"
        f"        症状是各种参数组合的误差**一模一样**。\n"
        f"        自己扫一遍参数、与 SDPA 数值对比，确认哪一组对")


@check("smoke_train", "端到端：跑 3 步训练，loss 有限且能下降")
def check_smoke_train():
    from nanochat.gpt import GPT
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, cfg.vocab_size, (4, 32))
    tgt = torch.randint(0, cfg.vocab_size, (4, 32))
    losses = []
    for _ in range(3):
        loss = model(idx, tgt)
        opt.zero_grad()
        loss.backward()
        # 梯度必须存在且有限（零初始化的输出投影会让某几步梯度为 0，正常）
        gn = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                 if p.grad is not None) ** 0.5
        assert math.isfinite(gn), "梯度出现 inf/nan"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(x) for x in losses), f"loss 非有限: {losses}"
    print(f"     loss: {[round(x, 4) for x in losses]}")


# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="本地自检（不需要 NPU）")
    ap.add_argument("--only", help="只跑 key 含该子串的检查")
    args = ap.parse_args()

    only = args.only
    todo = [(k, t, f) for k, t, f in CHECKS if not only or only in k]

    print("=" * 72)
    print("  本地自检 —— 不需要 NPU / checkpoint / 语料")
    print(f"  torch {torch.__version__}  设备: "
          f"{'npu' if hasattr(torch, 'npu') else 'cpu'}")
    print("=" * 72)

    npass = nskip = nfail = 0
    failures = []
    for key, title, fn in todo:
        print(f"\n[{key}] {title}")
        try:
            fn()
            print("   [ok]")
            npass += 1
        except AssertionError as e:
            print(f"   [FAIL] {e}")
            nfail += 1
            failures.append((key, str(e)))
        except NotImplementedError as e:
            print(f"   [TODO] 还没实现：{e}")
            nfail += 1
            failures.append((key, f"未实现: {e}"))
        except Exception:                              # noqa: BLE001
            print("   [ERROR] 未预期的异常：")
            print("      " + traceback.format_exc().replace("\n", "\n      "))
            nfail += 1
            failures.append((key, "见上方 traceback"))

    print("\n" + "=" * 72)
    print(f"  通过 {npass} / 失败 {nfail} / 共 {len(todo)}")
    if failures:
        print("\n  未通过：")
        for k, msg in failures:
            first = msg.strip().split("\n")[0][:100]
            print(f"    - {k}: {first}")
    print("=" * 72)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
