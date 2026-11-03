"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on incompatible CUDA GPUs, MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import math
import os

import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on CUDA GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are currently compiled for Hopper (sm90), Ada (sm89) and Ampere (sm80/sm86)
        # Blackwell (sm100) needs SDPA fallback until FA3 is recompiled or FA4 is released
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel, has_kernel
        # The varunneal kernel obtains better results for H100/Hopper
        if major == 9:
            hf_kernel = "varunneal/flash-attention-3"
            return get_kernel(hf_kernel).flash_attn_interface
        else:
            hf_kernel = "kernels-community/flash-attn3"
            if has_kernel(hf_kernel):
                return get_kernel(hf_kernel).flash_attn_interface
            else:
                return None

    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    # ---- 核心任务 2：用 PyTorch SDPA 实现注意力（**基础实现，必做**）--------
    # 输入 q,k,v 形状 (B, H, T, D)（是 BHSD！和 flash_attn_func 的 BSND 不同，
    # 调用方会负责转置）；window_size=(left, right)，left<0 表示不限制。
    #
    # **三种情形都要处理**（本函数的难点与考点）：
    #
    #   ① 全长、且 Tq == Tk：
    #          F.scaled_dot_product_attention(q, k, v, is_causal=True,
    #                                         enable_gqa=enable_gqa)
    #      这是最快的路径（PyTorch 会走融合内核）。**务必单独判断**，
    #      不要图省事一律构造 mask —— 实测构造 mask 会让全长层慢约 46%
    #
    #   ② 单 token 生成（Tq == 1）：只有一个 query，不需要 causal，
    #      但要按滑窗**裁掉左边的 K/V**：
    #          start = max(0, Tk - (window + 1))
    #          k = k[:, :, start:, :]；v = v[:, :, start:, :]
    #      然后 F.scaled_dot_product_attention(..., is_causal=False, ...)
    #
    #   ③ 其余（滑窗训练、chunk 推理）：**必须构造显式 bool mask**。
    #      Tq != Tk 时 is_causal=True 对不上 cache 位置，只能自己造：
    #          row = (Tk - Tq) + torch.arange(Tq).unsqueeze(1)
    #          col = torch.arange(Tk).unsqueeze(0)
    #          mask = col <= row
    #          滑窗再加： mask = mask & ((row - col) <= window)
    #      然后 F.scaled_dot_product_attention(..., attn_mask=mask, ...)
    #
    # 提示：
    #   * 这里的 bool mask **True 表示保留**；昇腾融合算子的 mask 语义相反
    #     （True 表示屏蔽）—— 进阶挑战 1 里这是个坑
    #   * enable_gqa 由调用方算好传进来（本模型 n_kv_head == n_head，实际是 False）
    #   * window 取 window_size[0]；本实现只支持左窗（右窗恒为 0）
    raise NotImplementedError("核心任务 2：用 SDPA 实现注意力")
# =============================================================================
# NPU 融合注意力（torch_npu.npu_fusion_attention）
# =============================================================================
#
# 为什么需要它
# ------------
# FA3 是 Hopper 专用，昇腾上走不到；SDPA 回退路径在**滑窗**层会构造一张
# (T, T) 的布尔 mask 并交给 SDPA —— 于是这些层用不上任何融合内核，
# 而是要物化完整的注意力分数矩阵。nanochat 的 window_pattern="SSSL"
# 意味着 **3/4 的层都走这条慢路径**。
#
# 昇腾原生提供 `torch_npu.npu_fusion_attention`，它是 flash-attention 式的
# 融合算子，而且有两个对我们特别合适的性质：
#   1. 输入布局 `BSND` = (batch, seq, head, dim) —— **与 nanochat 的
#      (B, T, H, D) 完全一致**，不需要转置；
#   2. 通过 `pre_tockens` / `next_tockens` **原生支持滑窗**，
#      不需要构造 (T, T) mask。
#
# ⚠️ 默认不启用，原因有两个（都不是保守，是真实风险）
# ------------------------------------------------------------------
# 1. **`sparse_mode` 选错会静默算错注意力。** 昇腾的 causal mask 有两种对齐：
#    2 = top-left（对应 SDPA 的 is_causal=True，也是训练要的），
#    3 = bottom-right（Tq == Tk 时与 2 等价，但语义不同）。
#    选错不会报错，只会让 val_bpb 略差一点。
# 2. **`pre_tockens` 的语义（含不含当前 token）官方文档表述不完整**，
#    需要实测确定。
#
# 还有一个**可能直接不适用**的风险：部分昇腾融合注意力算子要求
# head_num 满足对齐（常见是 16 的倍数），而本项目的 d8 是 **n_head=4**。
# 若如此，这个算子在本模型上根本用不了 —— 见 scripts/check_npu_attention.py
# 的第一项检查（它会用我们的真实形状先试一次，失败会明确报出来）。
#
# 因此启用流程是"先验证、再启用"：
#     python scripts/check_npu_attention.py --bench     # 在 NPU 上跑
#     export NANOCHAT_ATTN_IMPL=npu_fa
#     export NANOCHAT_NPU_FA_SPARSE_MODE=2              # 用脚本报出的值
#     export NANOCHAT_NPU_FA_PRE_TOKENS=513
# 脚本会给出它实测通过的**确切环境变量组合**。

def _load_npu_fusion_attention():
    """返回 torch_npu.npu_fusion_attention，不可用时返回 None。"""
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return None
    fn = getattr(torch_npu, "npu_fusion_attention", None)
    return fn


_npu_fa = _load_npu_fusion_attention()
HAS_NPU_FA = _npu_fa is not None


def attn_impl() -> str:
    """
    选择注意力实现：
      sdpa   （默认）现有的 SDPA 回退路径，行为不变
      npu_fa  昇腾融合注意力，需要先在真机上验证过参数
      auto   有 npu_fa 就用它，否则 sdpa（**仅在验证过参数后再用**）
    """
    env = os.environ.get("NANOCHAT_ATTN_IMPL", "sdpa").strip().lower()
    if env not in ("sdpa", "npu_fa", "auto"):
        raise ValueError(
            f"NANOCHAT_ATTN_IMPL 只能是 sdpa / npu_fa / auto，收到 {env!r}")
    if env == "npu_fa" and not HAS_NPU_FA:
        raise RuntimeError(
            "NANOCHAT_ATTN_IMPL=npu_fa 但 torch_npu.npu_fusion_attention 不可用。\n"
            "  确认：python -c \"import torch_npu; print(torch_npu.npu_fusion_attention)\"")
    if env == "auto":
        return "npu_fa" if HAS_NPU_FA else "sdpa"
    return env


def npu_fa_config() -> dict:
    """从环境变量读融合注意力的参数（默认值按 SDPA 的语义推）。"""
    return {
        # 2 = top-left causal，与 SDPA 的 is_causal=True 语义一致
        "sparse_mode": int(os.environ.get("NANOCHAT_NPU_FA_SPARSE_MODE", "2")),
        # pre_tockens 为 None 时按 window+1 推（SDPA 的 mask 是 i-j <= window，
        # 即含自己在内共 window+1 个 key）；实测语义不同的话用环境变量覆盖
        "pre_tokens": os.environ.get("NANOCHAT_NPU_FA_PRE_TOKENS"),
        "next_tokens": int(os.environ.get("NANOCHAT_NPU_FA_NEXT_TOKENS", "0")),
        "window_offset": int(os.environ.get("NANOCHAT_NPU_FA_WINDOW_OFFSET", "1")),
    }


_NPU_MASK_CACHE: dict = {}


NPU_FA_MASK_SIZE = int(os.environ.get("NANOCHAT_NPU_FA_MASK_SIZE", "2048"))


def _npu_causal_mask(seq_len: int, device):
    """
    上三角为 True 的 bool mask（被屏蔽的位置）。

    两个**实测**得来的约定，都不要凭常理猜：

    1. 昇腾文档明确：**不传 atten_mask 时 pre_tockens / next_tockens 会被忽略**，
       所以即使用滑窗也必须传这张 mask。
    2. **mask 的尺寸是固定的 2048×2048，不是 (T, T)。**
       T < 2048 时传 (T, T) 会直接报：
           "In the attenmask compression, please set the
            atten_mask_shape to [2048,2048]"
           [FUNC:AnalyzeOptionalInput]
           [FILE:flash_attention_score_tiling_general.cpp][LINE:1720]
       HF 的官方集成里同样是硬编码 2048 —— 那不是随手写的，是算子的要求。

    因此这里统一构造 `NPU_FA_MASK_SIZE × NPU_FA_MASK_SIZE`（默认 2048）。
    **这意味着本路径要求 `max_seq_len <= 2048`**；超过就得调大
    `NANOCHAT_NPU_FA_MASK_SIZE`（若算子支持）。
    """
    # ---- 进阶挑战 1a：构造昇腾融合注意力要的 mask ---------------------------
    # 返回上三角为 True 的 bool mask（True = **屏蔽**，与 SDPA 相反！）。
    #
    # ⚠️ 有一个**只能靠实测发现**的约定：
    #     mask 尺寸是**固定的 2048×2048**，不是 (T, T)。
    #     T < 2048 时传 (T,T) 会直接报：
    #         "In the attenmask compression, please set the
    #          atten_mask_shape to [2048,2048]"
    #         [FILE:flash_attention_score_tiling_general.cpp][LINE:1720]
    #     这也是 HuggingFace 的昇腾集成把 mask 硬编码成
    #     `torch.ones([2048, 2048])` 的原因 —— 不是随手写的。
    #
    # 副作用：这条路径要求 max_seq_len <= NPU_FA_MASK_SIZE（默认 2048）。
    #         超了要给**可操作的**报错，而不是让算子去骂人。
    #
    # 另外：昇腾文档明确"不传 atten_mask 时 pre_tockens/next_tockens 会被忽略"，
    #       所以即使只想要滑窗也必须传 mask。
    #
    # 提示：结果按 (device, NPU_FA_MASK_SIZE) 缓存，别每层都重建。
    raise NotImplementedError("进阶挑战 1a：构造融合注意力的 mask")
def _should_use_npu_fa(window_size, seq_len: int) -> bool:
    """
    这一层该不该用融合算子。

    实测全长层用融合算子反而慢（0.74×），所以默认只在**有滑窗**的层用它。
    想强制所有层都用（例如你测出来全长层也更快）就设
    NANOCHAT_NPU_FA_FULL_LAYERS=1。
    """
    if os.environ.get("NANOCHAT_NPU_FA_FULL_LAYERS", "0").strip() in (
            "1", "true", "yes", "on"):
        return True
    window = window_size[0]
    return window >= 0 and window < seq_len


def _npu_attention(q, k, v, window_size):
    """
    q, k, v 均为 (B, T, H, D) —— 与 npu_fusion_attention 的 BSND 布局一致。
    """
    # ---- 进阶挑战 1b：接入昇腾融合注意力 ------------------------------------
    # 目标：用 torch_npu.npu_fusion_attention 替换滑窗层的 SDPA 实现。
    #
    # 为什么值得做（实测数据见 docs/02_进阶挑战.md）：
    #     滑窗层   SDPA 13.42 ms  ->  融合  6.70 ms   2.00×
    #     全长层   SDPA  9.18 ms  ->  融合 12.42 ms   0.74×（**更慢**）
    #   所以只对滑窗层切过去 —— 这也是 _should_use_npu_fa 存在的原因。
    #
    # 关键事实（已实测确认，可直接用）：
    #   * 布局 "BSND" = (batch, seq, head, dim)，**与 q/k/v 形状完全一致**，
    #     不需要任何转置；返回 tuple，取 [0]
    #   * scale 传 1.0 / sqrt(D)
    #   * **滑窗必须用 band 模式 sparse_mode=4**，不是在因果模式（2/3）上
    #     加 pre_tockens —— 在 2/3 上加窗口参数会被**完全忽略**
    #   * band 模式下 pre_tockens 表示"能看前面多少个 token"：
    #     SDPA 的滑窗是 (row - col) <= window（含自己共 window+1 个 key），
    #     实测 pre_tockens=window、next_tockens=0 与 SDPA 一致
    #     （max|Δ| = 1.95e-03，bf16 累加顺序差异）
    #   * 全长层用 sparse_mode=2（对应 SDPA 的 is_causal=True）
    #
    # **但不要照抄上面的结论就直接交。** 题目要求你：
    #   1) 自己写参数扫描，把 sparse_mode × pre_tockens × next_tockens 扫一遍，
    #      与 SDPA 参考实现做**数值比对**，自己确认哪一组对；
    #   2) 解释为什么"各种组合误差一模一样"能推出"参数根本没生效"；
    #   3) 实测并报告加速比（scripts/bench_attention.py）。
    #
    # 配置从 npu_fa_config() 读（环境变量优先），便于不改代码就换参数。
    raise NotImplementedError("进阶挑战 1b：接入 npu_fusion_attention")
# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    # ---- 核心任务 3：把 BSND 布局接到两种注意力实现上 -----------------------
    # 输入 q,k,v 是 (B, T, H, D)；SDPA 要 (B, H, T, D)，走 SDPA 时需要转置两次
    # （去、回）。这个接口是刻意这么定的：
    #   保持 (B,T,H,D) -> 与昇腾融合算子的 BSND 布局**天然一致**，
    #                     走融合算子时一次转置都不需要
    #
    # 分发逻辑：
    #   if USE_FA3:                  # Hopper 上才有，本项目拿不到
    #       return _fa3.flash_attn_func(q, k, v, causal=causal,
    #                                   window_size=window_size)
    #   _impl = attn_impl()
    #   if _impl == "npu_fa" and _should_use_npu_fa(window_size, q.size(1)):
    #       return _npu_attention(q, k, v, window_size)
    #   # 否则 SDPA：转置 -> _sdpa_attention -> 转回
    #
    # 提示：_should_use_npu_fa 已经写好，它**只对真的有滑窗的层**返回 True ——
    #       实测全长层用融合算子反而慢（0.74×），原因见 docs/02_进阶挑战.md 的挑战 1。
    raise NotImplementedError("核心任务 3：flash_attn_func 的转置与分发")
def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    # ---- 核心任务 4：带 KV cache 的推理注意力 -------------------------------
    # 推理时长度会变（Tq=1 时 Tk=cache 长度），**不能**走 is_causal=True，
    # 要走 _sdpa_attention 的显式 mask 分支。
    #
    # 步骤：
    #   1) 把新的 k/v **原地**写进 cache（与 FA3 行为一致）：
    #          pos = cache_seqlens[0].item()
    #          k_cache[:, pos:pos+T_new, :, :] = k
    #          v_cache[:, pos:pos+T_new, :, :] = v
    #   2) 取 cache 的前 end_pos = pos + T_new 个位置作为完整 K/V
    #   3) 转成 (B,H,T,D) 调 _sdpa_attention，再转回 (B,T,H,D)
    #
    # 提示：滑窗推理只需最后 window+1 个 key —— 这正是短窗层能省 KV 读取的原因，
    #       也是 docs/02_进阶挑战.md 的挑战 2 里分析推理效率的切入点。
    raise NotImplementedError("核心任务 4：带 KV cache 的推理注意力")
# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
