import os
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
import csv

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

def calculate_mlm_accuracy(mlm_logits, mlm_labels):
    """Calculate accuracy for masked tokens only"""
    mask = (mlm_labels != -100)
    if mask.sum() == 0:
        return 0.0
    
    predictions = mlm_logits.argmax(dim=-1)
    correct = (predictions == mlm_labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()

def calculate_wer(pred_ids, target_ids, target_lengths):
    """Calculate Word Error Rate (WER) for CTC predictions"""
    # Decode CTC predictions (remove blanks and repeated tokens)
    def decode_ctc(pred_seq):
        decoded = []
        prev = None
        for token in pred_seq:
            if token != 0 and token != prev:  # 0 is blank
                decoded.append(token)
            prev = token
        return decoded
    
    total_distance = 0
    total_length = 0
    
    for pred, target_len in zip(pred_ids, target_lengths):
        # Decode CTC output
        pred_decoded = decode_ctc(pred.tolist())
        
        # Get actual target (extract from flattened target_ids)
        target_decoded = target_ids[:target_len].tolist()
        target_ids = target_ids[target_len:]  # Remove processed part
        
        # Calculate Levenshtein distance
        distance = levenshtein_distance(pred_decoded, target_decoded)
        total_distance += distance
        total_length += len(target_decoded)
    
    wer = total_distance / total_length if total_length > 0 else 0.0
    return wer

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two sequences"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_f1_score(mlm_logits, mlm_labels):
    """Calculate F1 score for masked token predictions"""
    mask = (mlm_labels != -100)
    if mask.sum() == 0:
        return 0.0
    
    predictions = mlm_logits.argmax(dim=-1)[mask]
    labels = mlm_labels[mask]
    
    # Calculate per-class precision and recall
    # We'll use a simplified macro F1 over masked tokens
    correct = (predictions == labels).float()
    
    # For multi-class, we compute F1 as harmonic mean of precision and recall
    # Here we use accuracy as a proxy since we have many classes
    # For more detailed F1, we'd need to compute per-class metrics
    
    # Simple approach: treat as binary (correct vs incorrect)
    tp = correct.sum()
    fp = (1 - correct).sum()
    fn = fp  # In this context, FN ≈ FP for masked predictions
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1.item() if isinstance(f1, torch.Tensor) else f1

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
    batch_size = 1
    max_steps = 1_000_000  # 1 juta step
    save_every = 10_000    # simpan setiap 10rb step
    lr = 1e-4
    mlm_prob = 0.15
    lambda_ctc = 1.0
    log_every = 100

    device = torch.device(f"cuda:{local_rank}")
    
    # ---------- SETUP LOGGING ----------
    if is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "training_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # CSV file untuk menyimpan semua metrik
        log_file = os.path.join(log_dir, f"training_metrics_{timestamp}.csv")
        log_writer = open(log_file, 'w', newline='')
        csv_writer = csv.writer(log_writer)
        csv_writer.writerow([
            "step", "total_loss", "mlm_loss", "ctc_loss", 
            "mlm_accuracy", "wer", "f1_score", "learning_rate"
        ])
        
        print(f"📊 Logging metrics to: {log_file}")
    else:
        log_writer = None
        csv_writer = None
    
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

            # Calculate metrics setiap step (untuk logging)
            if global_step % log_every == 0:
                with torch.no_grad():
                    # MLM Accuracy
                    mlm_accuracy = calculate_mlm_accuracy(mlm_logits, mlm_labels)
                    
                    # F1 Score for MLM
                    f1_score = calculate_f1_score(mlm_logits, mlm_labels)
                    
                    # WER for CTC
                    ctc_pred = ctc_logits.argmax(dim=-1)  # [B, T]
                    wer = calculate_wer(ctc_pred, ctc_targets, target_lengths)
                
                # Gather metrics dari semua GPU (optional, untuk average across GPUs)
                if world_size > 1:
                    # Convert to tensors for all_reduce
                    metrics_tensor = torch.tensor([
                        loss.item(), mlm_loss.item(), ctc_loss.item(),
                        mlm_accuracy, wer, f1_score
                    ], device=device)
                    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                    metrics_tensor /= world_size
                    
                    total_loss_avg = metrics_tensor[0].item()
                    mlm_loss_avg = metrics_tensor[1].item()
                    ctc_loss_avg = metrics_tensor[2].item()
                    mlm_accuracy_avg = metrics_tensor[3].item()
                    wer_avg = metrics_tensor[4].item()
                    f1_score_avg = metrics_tensor[5].item()
                else:
                    total_loss_avg = loss.item()
                    mlm_loss_avg = mlm_loss.item()
                    ctc_loss_avg = ctc_loss.item()
                    mlm_accuracy_avg = mlm_accuracy
                    wer_avg = wer
                    f1_score_avg = f1_score
                
                # Hanya main process yang print dan save log
                if is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    print(
                        f"Step {global_step}/{max_steps} | "
                        f"Loss: {total_loss_avg:.4f} | MLM: {mlm_loss_avg:.4f} | "
                        f"CTC: {ctc_loss_avg:.4f} | "
                        f"MLM Acc: {mlm_accuracy_avg:.4f} | "
                        f"WER: {wer_avg:.4f} | "
                        f"F1: {f1_score_avg:.4f}"
                    )
                    
                    # Write to CSV
                    csv_writer.writerow([
                        global_step, total_loss_avg, mlm_loss_avg, ctc_loss_avg,
                        mlm_accuracy_avg, wer_avg, f1_score_avg, current_lr
                    ])
                    log_writer.flush()  # Ensure data is written immediately

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
        
        # Close log file
        if log_writer:
            log_writer.close()
            print(f"✓ Training logs saved to {log_file}")
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    train()
