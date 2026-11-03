# 智能计算系统实验操作手册

## 实验二 nanoChat on Ascend NPU

在昇腾 NPU 上，从零训练一个中文 GPT。

本作业基于 [karpathy/nanochat](https://github.com/karpathy/nanochat) 改造而来，
已适配华为昇腾 NPU 与 torch-npu。

---

## 你会做什么

代码里留了 **16 处空洞**（标着 `NotImplementedError` 与任务编号）。
每处都配有线索：数学定义、张量形状、以及**最容易错在哪里**。
你要把它们填出来，然后完成两个进阶挑战。

| 编号 | 任务 | 文件 | 难度 |
|---|---|---|---|
| 基础 1 | RMSNorm | `nanochat/gpt.py` | 一行 |
| 基础 2 | RoPE 旋转位置编码 | `nanochat/gpt.py` | 短 |
| 基础 3 | RoPE 频率表 | `nanochat/gpt.py` | 短 |
| 基础 4 | 采样（temperature / top-k） | `nanochat/engine.py` | 短 |
| 基础 5 | MLP（ReLU²） | `nanochat/gpt.py` | 极短 |
| 基础 6 | Transformer Block | `nanochat/gpt.py` | 极短 |
| 核心 1 | **因果自注意力** | `nanochat/gpt.py` | 难 |
| 核心 2 | **SDPA 注意力（三路分支）** | `nanochat/flash_attention.py` | 难 |
| 核心 3 | 布局转置与分发 | `nanochat/flash_attention.py` | 中 |
| 核心 4 | 带 KV cache 的推理注意力 | `nanochat/flash_attention.py` | 中 |
| 核心 5 | **流式生成主循环** | `nanochat/engine.py` | 难 |
| 核心 6 | KV cache 批量复制 | `nanochat/engine.py` | 短 |
| 核心 8 | **GPT.forward（模型主干）** | `nanochat/gpt.py` | 难 |
| 进阶 A | 朴素采样（对照用） | `nanochat/gpt.py` | 中 |
| 进阶 B | **昇腾融合注意力** | `nanochat/flash_attention.py` | 进阶 |
| 进阶 C | **融合 mask** | `nanochat/flash_attention.py` | 进阶 |

另外两个**进阶挑战**（见 `docs/02_进阶挑战.md`）：

1. **实现方式对比**：SDPA vs 昇腾融合算子 vs 融合 RMSNorm，
   要求给出**数值一致性证据**与**实测加速比**，并解释为什么全长层反而更慢。
2. **模型规模对比**：训练不同深度的模型，做 scaling law 分析，
   并回答"固定算力该投给更大的模型还是更多的数据"。

**金庸风格微调不在本作业范围内 —— 那部分完全由你自己设计并实现。**

---

## 快速开始

```bash
# 1) 环境（有 NPU 的机器）
#    先确认能 import torch 与 torch_npu
python -c "import torch, torch_npu; print(torch.npu.device_count())"

# 2) 自检 —— 边写边跑，不要等全写完再调试
python scripts/selfcheck.py

# 3) 只跑某一项
python scripts/selfcheck.py --only sdpa
```

`scripts/selfcheck.py` **不需要 NPU**：CPU 上就能跑（用 CPU 版 torch）。
它每一项都对应一个填空，失败信息会告诉你期望什么、实际什么。
**通过它再上机器训练**，能省下大量排队时间。

没有 NPU 时的开发流程建议：

```bash
# CPU 上把逻辑写对、自检全过
python scripts/selfcheck.py
# 再上 NPU 训练
bash runs/train.sh
```

---

## 运行训练

```bash
bash runs/train.sh                 # 默认 d6，小规模快速验证
bash runs/train.sh 4               # depth=4，更小更快
DEPTH=8 ITERS=2000 bash runs/train.sh
```

推理与续写：

```bash
python -m scripts.chat_cli -i base -g d6-zh -p "那少年提剑立在崖边，"
```

---

## 数据

### 一、随包的起步语料

`data/corpus/` 有 8 部**公版**明清小说（水滸傳、三國演義、西遊記、施公案、
彭公案、老殘遊記、何典、西遊補），约 11 MiB。

它够跑通整条流水线、够看出"模型变大验证损失下降"的趋势；
**不够**做严肃的 scaling 研究 —— 见 `docs/02_进阶挑战.md`。

### 二、课程数据包（南大云盘）

规模更大的语料在课程云盘上，**匿名可下载**：

```
https://box.nju.edu.cn/d/2b845e514e754429bb86/
```

一条命令下载并解压：

```bash
bash runs/fetch_data.sh --list          # 先看云盘上有什么
bash runs/fetch_data.sh                 # 全部下载并解压
BOX_ONLY=climbmix bash runs/fetch_data.sh   # 只下预训练语料
```

| 云盘上的文件 | 内容 | 用途 |
|---|---|---|
| `base_data_climbmix.tar` | 中文预训练语料（约 1.25 GiB parquet） | 正式预训练 |
| `data_wuxia_jinyong.tar` | 金庸单源（约 12.7 MiB） | **你自己设计微调时用** |
| `data_wuxia_mixed.tar` | 金庸 + 32 部公版古典（约 41 MiB） | 同上 |
| `wuxia_sft.tar.gz` | 对话/续写 SFT 数据 | 同上 |

解压位置由 `NANOCHAT_BASE_DIR` 决定，之后把训练指向解压出来的目录：

```bash
export NANOCHAT_BASE_DIR=/your/outputs/nanochat
export NANOCHAT_DATA_DIR=$NANOCHAT_BASE_DIR/base_data_climbmix
```

可调环境变量（`BOX_URL` / `DATA_DEST` / `EXTRACT_TO` / `BOX_ONLY`）与常见失败
（403 链接过期、校网外需要代理、断点续传）都写在 `runs/fetch_data.sh` 开头的注释里。

> ⚠️ **合规**：金庸作品著作权保护期至 2068 年。云盘上的武侠包仅供本课程学习使用，

### 三、自己准备数据

训练脚本接受含 `text` 字段的 parquet，或纯文本目录。指向它即可：

```bash
export NANOCHAT_DATA_DIR=/path/to/your/data
```

文件名排序后**最后一个**会被当作验证集，所以请只留一个 val 分片。

---

## 分词器

本作业**随包提供**课程训练好的 16k 中文 BPE 词表：

```
tokenizer/tokenizer.pkl
```

放到 `NANOCHAT_BASE_DIR/tokenizer/` 下即可被训练与评测脚本自动找到：

```bash
mkdir -p $NANOCHAT_BASE_DIR/tokenizer
cp tokenizer/tokenizer.pkl $NANOCHAT_BASE_DIR/tokenizer/
```

评测用的是 **BPB（bits per byte）** 而不是 loss —— 它与你自己的词表无关，
所以跨配置可比。算 BPB 还需要一张"每个 token 多少字节"的表，用一条命令生成：

```bash
python scripts/make_token_bytes.py
```

也可以自己训词表（约十几分钟到一个小时）：

```bash
python -m scripts.tok_train -- --vocab-size=16384
```

---

## 目录

```
nanochat/          框架代码（gpt.py / flash_attention.py / engine.py 有空洞）
scripts/           入口：训练、评测、自检、两个进阶挑战
runs/              训练脚本 train.sh + 云盘数据下载 fetch_data.sh
docs/              任务清单、进阶挑战与思考题
data/corpus/       公版起步语料（约 11 MiB，8 部明清小说）
tokenizer/         随包的 16k 中文词表
configs/           模型配置示例（纯文档，不参与运行）
```

**`nanochat/` 里没被标 `NotImplementedError` 的部分不要重写** ——
那是工程脚手架（设备探测、精度选择、DDP、checkpoint 存取），
重写一遍没有收获，写错还会浪费大量时间。

---

## 重要提示

1. **昇腾上必须关闭 `torch.compile`**（TorchInductor 没有 npu 调度器，
   一编译就崩在 `assert scheduling_ctor`）。代码里已按设备自动关闭，
   不要把守卫去掉。
2. **`data/corpus/` 是公有领域的明清小说**（水滸傳、三國演義、西遊記…）。
   约 11 MiB / 2.7 M token —— 够跑通流水线，**不够**做严肃的 scaling 研究。
   要做得更大请自行扩充语料。
3. **金庸作品仍在著作权保护期内**（保护期至 2068 年）。
   **不要把原文提交到任何公开仓库**。
