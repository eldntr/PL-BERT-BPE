import os
import json
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets import load_from_disk

from transformers import AutoTokenizer

from text_tokenizer import TextTokenizer
from phoneme_tokenizer import PhonemeTokenizer
from dataloader_ctc import FilePathDataset, collate_fn, LengthBucketSampler
from model import MultiTaskModel


# =========================
# 4. TRAIN LOOP
# =========================

def setup_ddp():
    """Initialize DDP environment"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()

def train():
    # ---------- DDP SETUP ----------
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main_process = (local_rank == 0)
    
    if is_main_process:
        print(f"🚀 Training with {world_size} GPUs (DDP)")
    
    # ---------- PATH & PARAM ----------
    dataset_path = "wiki_phoneme_final"
    phoneme_vocab_path = "./wiki_phoneme/phoneme_vocab.json"
    text_tokenizer_name = "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"

    # Batch size per GPU - total effective batch = batch_size * world_size
    # Dengan 4 GPU & batch_size=192: effective batch = 192 * 4 = 768
    batch_size = 192
    max_steps = 1_000_000  # 1 juta step
    save_every = 100_000   # simpan setiap 100rb step
    lr = 1e-4
    mlm_prob = 0.15
    lambda_ctc = 1.0
    log_every = 100

    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process:
        print(f"Per-GPU batch size: {batch_size}")
        print(f"Total effective batch size: {batch_size * world_size}")
        print(f"Device: {device}")

    # ---------- LOAD TOKENIZERS ----------
    text_tokenizer = TextTokenizer(text_tokenizer_name)
    bpe_vocab_size = len(text_tokenizer)

    phoneme_tokenizer = PhonemeTokenizer.load(phoneme_vocab_path)
    phoneme_vocab_size = phoneme_tokenizer.vocab_size

    if is_main_process:
        print("Phoneme vocab size:", phoneme_vocab_size)
        print("BPE vocab size:", bpe_vocab_size)

    # ---------- LOAD DATASET ----------
    if is_main_process:
        print("Loading dataset from", dataset_path)
    hf_dataset = load_from_disk(dataset_path)
    if is_main_process:
        print("Dataset size:", len(hf_dataset))

    dataset = FilePathDataset(hf_dataset, phoneme_tokenizer, mlm_prob=mlm_prob)

    # Gunakan DistributedSampler untuk DDP (bukan LengthBucketSampler)
    # DistributedSampler akan membagi data ke semua GPU secara otomatis
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=42
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, phoneme_tokenizer, mlm_prob=mlm_prob),
        pin_memory=True
    )

    # ---------- MODEL ----------
    model = MultiTaskModel(
        phoneme_vocab_size=phoneme_vocab_size,
        bpe_vocab_size=bpe_vocab_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
    ).to(device)
    
    # Wrap model dengan DDP
    # find_unused_parameters=True diperlukan karena model memiliki multiple output heads
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    global_step = 0
    model.train()

    # Training loop dengan step-based (bukan epoch-based)
    while global_step < max_steps:
        # Set epoch untuk sampler agar shuffle berbeda setiap epoch
        sampler.set_epoch(global_step // len(loader))
        
        for batch in loader:
            if global_step >= max_steps:
                break
                
            global_step += 1

            phoneme_input = batch["phoneme_input"].to(device)      # [B, T]
            mlm_labels     = batch["mlm_labels"].to(device)         # [B, T]
            attention_mask = batch["attention_mask"].to(device)     # [B, T]

            ctc_targets    = batch["ctc_targets"].to(device)        # [sum_L]
            input_lengths  = batch["input_lengths"].to(device)      # [B]
            target_lengths = batch["target_lengths"].to(device)     # [B]

            optimizer.zero_grad()

            mlm_logits, ctc_logits = model(phoneme_input, attention_mask=attention_mask)

            # ----- MLM LOSS -----
            B, T, Vp = mlm_logits.shape
            mlm_loss = F.cross_entropy(
                mlm_logits.view(B*T, Vp),
                mlm_labels.view(B*T),
                ignore_index=-100
            )

            # ----- CTC LOSS -----
            # CTC expects [T, B, C]
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, C]

            # Shift BPE ids by +1 (0 = blank)
            ctc_targets_shifted = ctc_targets + 1

            ctc_loss = ctc_loss_fn(
                ctc_log_probs,
                ctc_targets_shifted,
                input_lengths,
                target_lengths
            )

            loss = mlm_loss + lambda_ctc * ctc_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Hanya main process yang print log
            if is_main_process and global_step % log_every == 0:
                print(
                    f"Step {global_step}/{max_steps} | "
                    f"Loss: {loss.item():.4f} | MLM: {mlm_loss.item():.4f} | "
                    f"CTC: {ctc_loss.item():.4f}"
                )

            # Simpan checkpoint setiap 100rb step (hanya main process)
            if is_main_process and global_step % save_every == 0:
                ckpt_path = f"checkpoint_step_{global_step}.pt"
                torch.save({
                    "model_state": model.module.state_dict(),  # .module untuk unwrap DDP
                    "optimizer_state": optimizer.state_dict(),
                    "global_step": global_step,
                }, ckpt_path)
                print(f"✓ Saved checkpoint to {ckpt_path}")

    # Simpan checkpoint final (hanya main process)
    if is_main_process:
        final_ckpt_path = f"checkpoint_step_{global_step}_final.pt"
        torch.save({
            "model_state": model.module.state_dict(),  # .module untuk unwrap DDP
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
        }, final_ckpt_path)
        print(f"✓ Training complete! Final checkpoint saved to {final_ckpt_path}")
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    train()
