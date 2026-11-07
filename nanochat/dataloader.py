"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import json
import os

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import DATA_DIR, base_dir, list_parquet_files


def _text_data_files(data_dir):
    extensions = (".txt", ".text", ".jsonl")
    paths = []
    for root, _dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(extensions):
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def _read_text_documents(path):
    if path.lower().endswith(".jsonl"):
        with open(path, encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text") if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(
                        f"{path}:{line_number} must contain a string 'text' field")
                if text.strip():
                    yield text
        return

    paragraph = []
    with open(path, encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                if paragraph:
                    yield "\n".join(paragraph)
                    paragraph.clear()
            else:
                paragraph.append(line.rstrip())
    if paragraph:
        yield "\n".join(paragraph)


def _text_document_batches(paths, split, resume_state_dict,
                           tokenizer_batch_size):
    _ddp, ddp_rank, _ddp_local_rank, ddp_world_size = get_dist_info()
    if len(paths) < 2:
        raise ValueError(
            "Text training requires at least two files: training files and a final validation file")
    paths = paths[:-1] if split == "train" else paths[-1:]

    resume_file_idx = resume_state_dict["pq_idx"] if resume_state_dict else 0
    resume_batch_idx = resume_state_dict["rg_idx"] if resume_state_dict else None
    epoch = resume_state_dict.get("epoch", 1) if resume_state_dict else 1
    first_pass = True

    while True:
        yielded_batches = 0
        document_index = 0
        for file_idx, path in enumerate(paths):
            batch = []
            batch_idx = 0
            for text in _read_text_documents(path):
                assigned_rank = document_index % ddp_world_size
                document_index += 1
                if assigned_rank != ddp_rank:
                    continue
                batch.append(text)
                if len(batch) < tokenizer_batch_size:
                    continue
                skip_batch = (first_pass and (
                    file_idx < resume_file_idx or
                    (file_idx == resume_file_idx and
                     resume_batch_idx is not None and
                     batch_idx <= resume_batch_idx)))
                if not skip_batch:
                    yield batch, (file_idx, batch_idx, epoch)
                    yielded_batches += 1
                batch = []
                batch_idx += 1
            skip_batch = (first_pass and (
                file_idx < resume_file_idx or
                (file_idx == resume_file_idx and
                 resume_batch_idx is not None and
                 batch_idx <= resume_batch_idx)))
            if batch and not skip_batch:
                yield batch, (file_idx, batch_idx, epoch)
                yielded_batches += 1

        if yielded_batches == 0 and not first_pass:
            raise ValueError(
                f"Text split has no documents assigned to rank {ddp_rank} "
                f"of {ddp_world_size}")

        first_pass = False
        resume_file_idx = 0
        resume_batch_idx = None
        epoch += 1

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches read from parquet or text files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    data_dir = os.environ.get("NANOCHAT_DATA_DIR") or DATA_DIR
    if not os.path.exists(data_dir):
        data_dir = os.path.join(base_dir, "base_data")
    parquet_paths = (list_parquet_files(
        data_dir=data_dir, warn_on_legacy=warn_on_legacy)
        if os.path.isdir(data_dir) else [])
    if not parquet_paths:
        text_paths = _text_data_files(data_dir)
        if not text_paths:
            raise ValueError(
                f"No parquet or text data found in {data_dir}; expected parquet, .txt, .text, or .jsonl files")
        yield from _text_document_batches(
            text_paths, split, resume_state_dict, tokenizer_batch_size)
        return
    if len(parquet_paths) < 2:
        raise ValueError(
            f"Expected at least two parquet shards in {data_dir}; "
            "the last shard is reserved for validation")
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
