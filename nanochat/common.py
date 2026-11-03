"""
Common utilities for nanochat.
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    if hasattr(torch, "npu") and torch.npu.is_available():
        # Ascend NPU has native bf16 tensor support, and bf16 does not require a GradScaler.
        # NOTE: without this branch the NPU falls through to fp32 below, which is a SILENT
        # ~3-4x slowdown with double the memory. NANOCHAT_DTYPE=bfloat16 also works.
        return torch.bfloat16, "auto-detected: Ascend NPU (native bf16)"
    # Note: MPS on recent macOS also handles bf16 fine, opt in via NANOCHAT_DTYPE=bfloat16
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # prefer to use CUDA if available, then Ascend NPU, then MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device_type = "npu"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


# 哪些设备**不能**用 torch.compile。
#
# 昇腾 NPU 实测：TorchInductor 没有 "npu" 设备的调度器，一编译就抛
#     torch._inductor.exc.InductorError: LoweringException: AssertionError:
#     File ".../torch/_inductor/codegen/common.py", line 452, in get_backend_features
#         assert scheduling_ctor
# 触发点是 nanochat/gpt.py 里 norm() 调用的 F.rms_norm -> aten.mean.dim。
# CPU/MPS 同理（inductor 支持但收益低、编译时间长），本项目一律走 eager。
_NO_COMPILE_DEVICES = ("npu", "mps", "cpu")

_COMPILE_ENABLED_CACHE: dict[str, bool] = {}


def compile_enabled(device_type: str | None = None) -> bool:
    """
    是否启用 `torch.compile`。

    ⚠️ **昇腾上必须返回 False。** 这个项目曾经在 `configs/model_d8.yaml` 里
    写着 `enforce_eager: true`，但**没有任何代码读那个文件** ——
    于是 `torch.compile` 照常生效，训练在第一次 eval 时就崩在 inductor 的
    断言上（`assert scheduling_ctor`）。配置项看起来是真的、实际是装饰，
    是最难查的一类问题。

    规则（不需要任何配置就会生效）：
      * 设备是 npu / mps / cpu -> 关闭（npu 上 inductor 没有调度器，
        实测一编译就崩；cpu/mps 能编译但收益低、编译时间长）
      * 设备是 cuda -> 开启
      * `NANOCHAT_DISABLE_COMPILE=1` 无条件关闭（便于排查）；
        `=0` 无条件开启（仅用于实验）
      * `NANOCHAT_ENFORCE_EAGER` 是 `NANOCHAT_DISABLE_COMPILE` 的**别名**
        （语义相反）。4 个 `runs/npu_stage_*.sh` 一直在导出它，但历史上
        没有任何代码读它 —— 又一个"看着生效、其实没有"的开关。

    环境变量**每次调用都重新读**，只有"按设备推断的默认值"才缓存
    （缓存是为了避免重复调用 autodetect 打印多遍横幅）。
    """
    def _truthy(v: str) -> bool:
        return v.strip().lower() in ("1", "true", "yes", "on")

    env = os.environ.get("NANOCHAT_DISABLE_COMPILE")
    if env is not None:
        enabled = not _truthy(env)
        print0(f"torch.compile: {'启用' if enabled else '关闭'}"
               f"（环境变量 NANOCHAT_DISABLE_COMPILE={env}）")
        return enabled
    env_alias = os.environ.get("NANOCHAT_ENFORCE_EAGER")
    if env_alias is not None:
        enabled = not _truthy(env_alias)
        print0(f"torch.compile: {'启用' if enabled else '关闭'}"
               f"（环境变量 NANOCHAT_ENFORCE_EAGER={env_alias}）")
        return enabled

    if device_type is None:
        device_type = autodetect_device_type()
    if device_type in _COMPILE_ENABLED_CACHE:
        return _COMPILE_ENABLED_CACHE[device_type]

    enabled = device_type not in _NO_COMPILE_DEVICES
    if enabled:
        reason = f"设备 {device_type} 支持 torch.compile"
    else:
        reason = (f"设备 {device_type} 上 TorchInductor 不可用或不划算"
                  + ("（昇腾没有 npu 调度器，一编译就 assert scheduling_ctor）"
                     if device_type == "npu" else ""))
    print0(f"torch.compile: {'启用' if enabled else '关闭'}（{reason}）")
    _COMPILE_ENABLED_CACHE[device_type] = enabled
    return enabled


def maybe_compile(fn, device_type: str | None = None):
    """
    给函数套上 `torch.compile`，但在不支持编译的设备上原样返回。

    **必须是"第一次调用时"才决定**，不能在装饰器求值（import）时就决定：
    `nanochat/optim.py` 里的 Muon/AdamW 融合步进函数是在模块导入时加装饰器的，
    那时候 `torch_npu` 还没被 import，`torch.npu.is_available()` 还是 False，
    设备探测会误判成 cpu。
    """
    cached = None

    def wrapper(*args, **kwargs):
        nonlocal cached
        if cached is None:
            cached = (torch.compile(fn, dynamic=False, fullgraph=True)
                      if compile_enabled(device_type) else fn)
        return cached(*args, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapper


def compute_init(device_type="cuda"): # cuda|cpu|mps|npu
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu", "npu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
    if device_type == "npu":
        # importing torch_npu registers the "npu" device and patches torch
        import torch_npu  # noqa: F401
        assert torch.npu.is_available(), "Your PyTorch installation is not configured for Ascend NPU but device_type is 'npu'"

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    if device_type == "npu":
        torch.npu.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls, see https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    elif is_ddp_requested and device_type == "npu":
        device = torch.device("npu", ddp_local_rank)
        torch.npu.set_device(device)  # make "npu" default to this device
        dist.init_process_group(backend="hccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')

def get_peak_bandwidth(device_name: str) -> float:
    """Peak HBM/GDDR memory bandwidth in bytes/sec. The decode phase of inference
    is memory-bandwidth-bound, so this is the roofline for tokens/sec (see MBU)."""
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_BANDWIDTH_TABLE = (
        # NVIDIA Blackwell (HBM3e)
        (["gb200"], 8.0e12),
        (["grace blackwell"], 8.0e12),
        (["b200"], 8.0e12),
        (["b100"], 8.0e12),
        # NVIDIA Hopper
        (["h200"], 4.8e12),
        (["h100", "nvl"], 3.9e12),
        (["h100", "pcie"], 2.0e12),
        (["h100"], 3.35e12), # SXM
        (["h800", "pcie"], 2.0e12),
        (["h800"], 3.35e12), # SXM
        # NVIDIA Ampere data center (A100 80GB; the 40GB variant is 1.6e12)
        (["a100"], 2.0e12),
        (["a800"], 2.0e12),
        (["a40"], 696e9),
        (["a30"], 933e9),
        # NVIDIA Ada data center
        (["l40s"], 864e9),
        (["l40-s"], 864e9),
        (["l40 s"], 864e9),
        (["l4"], 300e9),
        # AMD CDNA accelerators
        (["mi355"], 8.0e12),
        (["mi325"], 6.0e12),
        (["mi300x"], 5.3e12),
        (["mi300a"], 5.3e12),
        (["mi250x"], 3.28e12),
        (["mi250"], 3.28e12),
        # Consumer RTX
        (["5090"], 1.79e12),
        (["4090"], 1.01e12),
        (["3090"], 936e9),
    )
    for patterns, bandwidth in _PEAK_BANDWIDTH_TABLE:
        if all(p in name for p in patterns):
            return bandwidth

    # Unknown GPU - return inf so MBU shows as 0% rather than a wrong guess
    logger.warning(f"Peak bandwidth undefined for: {device_name}, MBU will show as 0%")
    return float('inf')
