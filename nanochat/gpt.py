"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

import os
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


# ---------------------------------------------------------------------------
# RMSNorm：原生 vs 昇腾融合
# ---------------------------------------------------------------------------
#
# 本模型每层有 4 次 RMSNorm（attn 前、mlp 前，以及 QK norm 的两次），
# 8 层 → **32 次/前向**。eager 模式下每次是若干个逐元素 kernel，
# 而注意力只占总 FLOPs 约 18%（见 docs/01_budget.md 第 3 节）——
# 所以换融合 RMSNorm 的收益可能不比换融合注意力小。
#
# ⚠️ eps 必须对上。实测（torch 2.12.0）：
#     F.rms_norm(x, (n,)) 不传 eps 时的默认值**实际是 1e-6**
#     （文档写的是 finfo(dtype).eps，但实测手算 eps=1e-6 与它逐位一致；
#       而 bf16 的 finfo.eps=0.0078 会差 3e-2 —— 差的是 eps 不是算子）
#
# 默认 "native"（保持原行为）。改 NANOCHAT_RMSNORM_IMPL=npu 才走融合算子。
# 是否可用先用 scripts/check_npu_attention.py --bench 验证。
_RMS_EPS = 1e-6
_NPU_RMS_STATE: dict = {"resolved": False, "fn": None}
_NPU_RMS_WEIGHT_CACHE: dict = {}


def _resolve_npu_rms_norm():
    """惰性解析（不在 import 时决定）：gpt.py 被导入时环境变量未必就绪。"""
    if _NPU_RMS_STATE["resolved"]:
        return _NPU_RMS_STATE["fn"]
    mode = os.environ.get("NANOCHAT_RMSNORM_IMPL", "native").strip().lower()
    fn = None
    if mode == "npu":
        try:
            import torch_npu
        except Exception as e:                          # noqa: BLE001
            raise RuntimeError(
                f"NANOCHAT_RMSNORM_IMPL=npu 但无法 import torch_npu：{e}\n"
                f"  确认：python -c \"import torch_npu; print(torch_npu.npu_rms_norm)\"") from e
        fn = getattr(torch_npu, "npu_rms_norm", None)
        if fn is None:
            raise RuntimeError(
                "NANOCHAT_RMSNORM_IMPL=npu 但 torch_npu.npu_rms_norm 不存在。")
    elif mode != "native":
        raise ValueError(
            f"NANOCHAT_RMSNORM_IMPL 只能是 native / npu，收到 {mode!r}")
    _NPU_RMS_STATE.update(resolved=True, fn=fn)
    return fn


def norm(x):
    # ---- 基础任务 1：RMSNorm ------------------------------------------------
    # 把最后一维做均方根归一化：
    #     y = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
    #
    # 注意（写错不会报错，只会训得不对）：
    #   * 这里**没有可学习权重**（不带 gamma），所以不需要 nn.Parameter
    #   * eps 必须与 PyTorch 的默认取值一致。实测 torch 2.12 下
    #     `F.rms_norm(x, shape)` 不传 eps 时的默认值是 **1e-6**，
    #     *不是* finfo(dtype).eps（bf16 下后者是 0.0078，结果会差 3e-2）
    #   * 输入可能是 bf16；先转 float32 算再转回去更稳
    #
    # 一行就能写完，但请自己实现 —— 进阶挑战 1 要拿它和融合算子对比性能。
    raise NotImplementedError("基础任务 1：实现 RMSNorm")
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included).

    Set NANOCHAT_DISABLE_VE=1 to turn value embeddings off entirely.
    On small models VE can own more than half of all parameters while costing almost no
    FLOPs (it is an nn.Embedding lookup), so this course disables it by default and keeps
    the switch around as an ablation knob. See docs/01_budget.md section 2.1.
    """
    if os.environ.get("NANOCHAT_DISABLE_VE", "0") == "1":
        return False
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # ---- 基础任务 2：RoPE 旋转位置编码 --------------------------------------
    # 输入 x: (B, T, H, D)，cos/sin: (1, T, 1, D/2)，返回同形状。
    #
    # 做法：把最后一维**两两配对**，每对 (x1, x2) 按该位置的角 theta 旋转：
    #     y1 = x1 * cos - x2 * sin
    #     y2 = x1 * sin + x2 * cos
    #
    # 提示：
    #   * x[..., 0::2] 是每对的第一个元素，x[..., 1::2] 是第二个
    #     （也可以 view 成 (..., D//2, 2) 再拆；注意 cos/sin 要能广播上）
    #   * 频率表由 _precompute_rotary_embeddings 生成（它也留空了）
    #   * 为什么 RoPE 能编码**相对**位置？见 docs/03_思考题.md 第 2 题
    #   * 注意不要就地修改输入（autograd 会不高兴）
    raise NotImplementedError("基础任务 2：实现 RoPE")
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # ---- 核心任务 1：因果自注意力 -------------------------------------------
        # 参数：
        #   x           (B, T, n_embd)
        #   ve          (B, T, n_embd) 或 None —— value embedding（ResFormer 的 value residual）
        #   cos_sin     (cos, sin)，来自 _precompute_rotary_embeddings
        #   window_size (left, right)；left < 0 表示不限制 —— 这就是**滑窗注意力**
        #   kv_cache    KVCache 或 None（训练时 None，推理时不是）
        #
        # **顺序很重要**（下面每一步的顺序都是实测过的，调换会训得不对）：
        #
        #   1) 投影并整形：
        #        q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)
        #        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        #        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        #      **布局是 (B,T,H,D)** —— 这是 FA3 与昇腾融合算子共用的原生布局，
        #      刻意不做转置（见 docs/02_进阶挑战.md 的挑战 1）
        #
        #   2) value residual（ve 不是 None 时）：
        #        ve   = ve.view(B, T, self.n_kv_head, self.head_dim)
        #        gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
        #        v    = v + gate.unsqueeze(-1) * ve
        #      注意 gate 是 (B,T,n_kv_head)，取值 (0,3)，**逐 head 生效**，
        #      所以要 unsqueeze 一维才能广播到 (B,T,H,D) 的 v 上。
        #      ve_gate 只在 has_ve(layer_idx, n_layer) 为真时才存在（否则是 None）
        #
        #   3) RoPE：q, k = apply_rotary_emb(q, *cos_sin), apply_rotary_emb(k, *cos_sin)
        #
        #   4) **QK norm**：q, k = norm(q), norm(k)
        #      ⚠️ 是在 RoPE **之后**做，不是之前。这是本版本相对经典 GPT 的改动
        #
        #   5) q = q * 1.2 且 **k = k * 1.2**（两个都要乘，别只乘 q）
        #      作用是把缩放分摊到 Q 和 K 上，让注意力更尖锐
        #
        #   6) 注意力本体，两种情形：
        #        kv_cache is None（训练）：
        #            y = flash_attn.flash_attn_func(q, k, v, causal=True,
        #                                           window_size=window_size)
        #        kv_cache 不为 None（推理）：
        #            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
        #            y = flash_attn.flash_attn_with_kvcache(
        #                    q, k_cache, v_cache, k=k, v=v,
        #                    cache_seqlens=kv_cache.cache_seqlens,
        #                    causal=True, window_size=window_size)
        #            # 位置推进放在**最后一层**做（不是每层都做，也不是在 GPT.forward 里做）
        #            if self.layer_idx == kv_cache.n_layers - 1:
        #                kv_cache.advance(T)
        #
        #   7) 合头并投影回去：y = self.c_proj(y.contiguous().view(B, T, -1))
        #
        # 线索：flash_attn 是本仓库自己的封装（nanochat/flash_attention.py），
        #       它的基础实现（SDPA）也是你要写的 —— 两个任务在这里汇合。
        raise NotImplementedError("核心任务 1：实现 CausalSelfAttention.forward")
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        # ---- 基础任务 5：MLP 前向（ReLU² 激活）---------------------------------
        # 结构简单，但激活函数**不是** GELU / SwiGLU，而是：
        #     x = self.c_fc(x)        # (B, T, 4C)
        #     x = relu(x) ** 2        # ← ReLU 平方
        #     x = self.c_proj(x)      # (B, T, C)
        #
        # 为什么用 ReLU²？见 docs/03_思考题.md 第 3 题。
        # 提示：它比 GELU 少一次指数运算，且在 bf16 下数值行为更简单。
        raise NotImplementedError("基础任务 5：实现 MLP 前向")
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # ---- 基础任务 6：Transformer Block（pre-norm 残差）---------------------
        # 输入 x: (B, T, C)，返回同形状。经典 pre-norm，两层残差：
        #     x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        #     x = x + self.mlp(norm(x))
        #
        # 注意参数要按顺序往下传全（ve / cos_sin / window_size / kv_cache 一个都不能丢）。
        # 提示：norm 是你写的（基础任务 1），attn 与 mlp 也在本文件里。
        raise NotImplementedError("基础任务 6：实现 Block 前向")
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # ---- 基础任务 3：预计算 RoPE 频率表 -------------------------------------
        # 返回 (cos, sin)，形状都是 (1, seq_len, 1, head_dim // 2)，
        # dtype 用 COMPUTE_DTYPE（模块顶部已 import）。
        #
        # 定义：
        #     inv_freq[i] = 1.0 / (base ** (2i / head_dim)),  i = 0 .. head_dim/2 - 1
        #     angles[t, i] = t * inv_freq[i]
        #     cos = cos(angles)、sin = sin(angles)，再 reshape 成 (1, T, 1, D/2)
        #
        # 提示：
        #   * `base` 默认 100000（见签名）。它越大，低频衰减越慢、能编码的
        #     位置范围越远 —— 调它是个可以做的实验（docs/03_思考题.md 第 2 题）
        #   * device 参数别忘，否则表建在 CPU 上、前向时设备不匹配
        #   * 频率建议用 float32 算，最后再转 COMPUTE_DTYPE（数值更稳）
        raise NotImplementedError("基础任务 3：实现 RoPE 频率表")
    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # ---- 核心任务 8：GPT.forward（模型主干）--------------------------------
        # 这是把所有部件串起来的地方。**smear 那一段已经给你**（它是本版本特有的
        # 机制，靠"标准 GPT"的知识猜不出来），其余全部由你自己写。
        #
        # 1) 取 RoPE 表
        #    * 表已缓存在 self.cos / self.sin，形状 (1, seq_len, 1, head_dim/2)
        #    * **必须按当前长度裁剪**：cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        #    * T0 = 这段 token 在序列里的起始位置：
        #          训练（kv_cache is None）时 T0 = 0
        #          带 KV cache 推理时 T0 = kv_cache.get_pos()
        #      ⚠️ T0 搞错会让整个 RoPE 位置偏移，而且**不报错**，只是结果变差。
        #         唯一能抓到它的是 scripts/selfcheck.py 的 kvcache 项。
        #    * 顺手保留原来的三条 assert（T 不超表长 / 设备一致 / dtype 一致），
        #      它们是这类静默错误的早期防线
        #
        # 2) 词嵌入 + 归一化
        #      x = self.transformer.wte(idx)      # (B, T, n_embd)
        #      x = x.to(COMPUTE_DTYPE)            # 精度路径，别省
        #      x = norm(x)                        # 复用基础任务 1
        #
        # 3) **smear（这一段直接抄，两个分支都不能省）**
        #    作用：把上一个 token 的嵌入按门控混进当前位置，提供廉价的 bigram 信息。
        #    ⚠️ 注意 `:24` 是 self.smear_gate 的输入维度（这里写死），
        #       gate 形状 (B,T,1)，靠广播作用到整个 n_embd 上。
        #
        #        # Smear: 把上一个 token 的嵌入混入当前位置
        #        if kv_cache is None:
        #            # 训练 / 朴素生成：整段都在手上，直接切片
        #            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
        #                self.smear_gate(x[:, 1:, :24]))
        #            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        #        else:
        #            # KV cache 推理：从 cache 读上一个嵌入，同时存下当前这个
        #            x_pre_smear = kv_cache.prev_embedding
        #            kv_cache.prev_embedding = x[:, -1:, :]
        #            if T > 1:
        #                # prefill：与训练相同，位置 1 起才做 smear
        #                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
        #                    self.smear_gate(x[:, 1:, :24]))
        #                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        #            elif x_pre_smear is not None:
        #                # 单 token 解码：用 cache 里的上一个嵌入
        #                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
        #                    self.smear_gate(x[:, :, :24]))
        #                x = x + gate * x_pre_smear
        #
        # 4) 主干循环（自己写）
        #      x0 = x                     # 记下最初的嵌入，后面每层都要回灌
        #      逐层遍历 self.transformer.h，每层做三件事：
        #        a) **混合残差**：x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        #           resid_lambdas / x0_lambdas 是每层的可学习标量，让模型自己决定
        #           "保留多少当前表示、回灌多少最初嵌入"
        #        b) **取该层的 value embedding**（只有部分层有）：
        #             ve = self.value_embeds[str(i)](idx).to(x.dtype) \
        #                  if str(i) in self.value_embeds else None
        #        c) **调用 block，参数一个都不能少**：
        #             x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        #           ⚠️ window_sizes[i] 漏传 -> 所有层都变全长注意力，不报错但变差
        #
        # 5) backout：在**中间层**存一份残差，最后减掉，用于在投影 logits 前
        #    去掉低层特征
        #      backout_layer = self.config.n_layer // 2
        #      循环里：if i == backout_layer: x_backout = x
        #      循环后：if x_backout is not None:
        #                  x = x - self.backout_lambda.to(x.dtype) * x_backout
        #
        # 6) 最后一次 norm：x = norm(x)
        #
        # 7) logits —— **三步都不能少**
        #      logits = self.lm_head(x)                     # (B,T,padded_vocab) 很大
        #      logits = logits[..., :self.config.vocab_size]  # ① 切掉 padding
        #      logits = logits.float()                        # ② 转 fp32
        #      logits = 15 * torch.tanh(logits / 15)          # ③ logit softcap
        #    ① 漏了 -> 形状与词表对不上；② 漏了 -> bf16 下 tanh 精度不够；
        #    ③ 漏了 -> logits 无界，bf16 训练容易发散
        #
        # 8) 返回
        #      targets is not None 时算损失并返回，否则直接返回 logits。
        #      损失用 F.cross_entropy，注意两点：
        #        * **ignore_index=-1**（-1 表示"这个位置不算损失"）
        #        * 先在**摊平后**的 logits 上算：logits.view(-1, logits.size(-1))
        #          于是 reduction="none" 返回的是 (B*T,)，**不是** (B,T)
        #
        # 自检会同时检查：logits 形状、|logits| <= 15、loss 有限、
        # "mean" 与 "none".mean() 一致、以及增量解码与全序列前向一致。
        raise NotImplementedError("核心任务 8：实现 GPT.forward")
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        # ---- 进阶任务：GPT.generate（不依赖 KV cache 的朴素采样）---------------
        # 这是 Engine.generate 的"慢但直白"版本，用来**对照验证**。
        # 每步都把整段序列重新前向一遍（不复用 KV cache），所以很慢，
        # 但实现简单、容易确认正确性。**建议先写这个，再写 Engine.generate。**
        #
        # 循环 max_tokens 次：
        #   logits = self(idx_cond)[:, -1, :]      # 只取最后一个位置
        #   logits = logits / temperature；top_k 截断（k 超过词表大小时要 clamp）
        #   probs = torch.softmax(logits, dim=-1)
        #   idx_next = torch.multinomial(probs, 1, generator=rng)
        #   idx = torch.cat([idx, idx_next], dim=1)
        #
        # 提示：
        #   * 采样逻辑与 engine.sample_next_token 相同，可以 import 复用
        #   * rng 必须由 seed 构造（torch.Generator(device=...) + manual_seed(seed)），
        #     否则同一 seed 不可复现，实验之间无法对比
        #   * 这个函数在 __init__ 里被标记为 @torch.inference_mode()，别把梯度带进来
        raise NotImplementedError("进阶任务：实现朴素采样式生成")