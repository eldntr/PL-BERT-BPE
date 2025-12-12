import os
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
import csv

# Set memory optimization environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
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
                # PERBAIKAN: Kurangi 1 karena model memprediksi (id + 1)
                # Pastikan tidak menjadi negatif jika model memprediksi 0 (blank)
                actual_token_id = token - 1
                if actual_token_id >= 0:
                    decoded.append(actual_token_id)
            prev = token
        return decoded
    
    total_distance = 0
    total_length = 0
    
    # PERBAIKAN: target_ids adalah tensor yang di-flatten, 
    # kita perlu memproses dengan benar untuk setiap batch item
    target_offset = 0
    
    for pred, target_len in zip(pred_ids, target_lengths):
        # Decode CTC output
        pred_decoded = decode_ctc(pred.tolist())
        
        # PERBAIKAN: Ekstrak target yang benar menggunakan offset
        target_decoded = target_ids[target_offset:target_offset + target_len].tolist()
        target_offset += target_len  # Update offset untuk item berikutnya
        
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

def evaluate(model, val_loader, device, ctc_loss_fn, mlm_prob, world_size):
    """Evaluate model on validation set"""
    model.eval()
    
    total_loss = 0.0
    total_mlm_loss = 0.0
    total_ctc_loss = 0.0
    total_mlm_acc = 0.0
    total_wer = 0.0
    total_f1 = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            phoneme_input = batch["phoneme_input"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ctc_targets = batch["ctc_targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            
            mlm_logits, ctc_logits = model(phoneme_input, attention_mask=attention_mask)
            
            # MLM Loss
            B, T, Vp = mlm_logits.shape
            mlm_loss = F.cross_entropy(
                mlm_logits.view(B*T, Vp),
                mlm_labels.view(B*T),
                ignore_index=-100
            )
            
            # CTC Loss
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            ctc_targets_shifted = ctc_targets + 1
            ctc_loss = ctc_loss_fn(
                ctc_log_probs,
                ctc_targets_shifted,
                input_lengths,
                target_lengths
            )
            
            loss = mlm_loss + ctc_loss
            
            # Metrics
            mlm_acc = calculate_mlm_accuracy(mlm_logits, mlm_labels)
            f1 = calculate_f1_score(mlm_logits, mlm_labels)
            ctc_pred = ctc_logits.argmax(dim=-1)
            wer = calculate_wer(ctc_pred, ctc_targets, target_lengths)
            
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_ctc_loss += ctc_loss.item()
            total_mlm_acc += mlm_acc
            total_wer += wer
            total_f1 += f1
            num_batches += 1
            
            # Limit validation to 100 batches for speed
            if num_batches >= 100:
                break
    
    model.train()
    
    # Average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mlm_loss = total_mlm_loss / num_batches if num_batches > 0 else 0.0
    avg_ctc_loss = total_ctc_loss / num_batches if num_batches > 0 else 0.0
    avg_mlm_acc = total_mlm_acc / num_batches if num_batches > 0 else 0.0
    avg_wer = total_wer / num_batches if num_batches > 0 else 0.0
    avg_f1 = total_f1 / num_batches if num_batches > 0 else 0.0
    
    # Aggregate across GPUs
    if world_size > 1:
        metrics_tensor = torch.tensor([
            avg_loss, avg_mlm_loss, avg_ctc_loss,
            avg_mlm_acc, avg_wer, avg_f1
        ], device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metrics_tensor /= world_size
        
        result = {
            "loss": metrics_tensor[0].item(),
            "mlm_loss": metrics_tensor[1].item(),
            "ctc_loss": metrics_tensor[2].item(),
            "mlm_accuracy": metrics_tensor[3].item(),
            "wer": metrics_tensor[4].item(),
            "f1_score": metrics_tensor[5].item(),
        }
    else:
        result = {
            "loss": avg_loss,
            "mlm_loss": avg_mlm_loss,
            "ctc_loss": avg_ctc_loss,
            "mlm_accuracy": avg_mlm_acc,
            "wer": avg_wer,
            "f1_score": avg_f1,
        }
    
    # Clear GPU cache after evaluation
    torch.cuda.empty_cache()
    return result

def train():
    # ---------- DDP SETUP ----------
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main_process = (local_rank == 0)
    
    if is_main_process:
        print(f"🚀 Training with {world_size} GPUs (DDP)")
    
    # ---------- PATH & PARAM ----------
    train_dataset_path = "wiki_phoneme_train"
    val_dataset_path = "wiki_phoneme_val"
    phoneme_vocab_path = "phoneme_vocab.json"
    text_tokenizer_name = "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"

    # Batch size per GPU - total effective batch = batch_size * world_size * accumulation_steps
    # Dengan 2 GPU & batch_size=16 & accumulation_steps=4: 
    # effective batch = 16 * 2 * 4 = 128
    batch_size = 16
    accumulation_steps = 4  # Gradient accumulation untuk menghemat VRAM
    max_steps = 1_000_000  # 1 juta step
    save_every = 10_000    # simpan setiap 10rb step
    eval_every = 1_000    # evaluasi setiap 10rb step
    lr = 1e-4
    mlm_prob = 0.15
    lambda_ctc = 1.0
    log_every = 100
    
    # Resume training dari checkpoint (set None jika train from scratch)
    resume_from_checkpoint = "checkpoint_step_90000.pt" # Contoh: "checkpoint_step_10000.pt"

    device = torch.device(f"cuda:{local_rank}")
    
    # ---------- SETUP LOGGING ----------
    if is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "training_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # CSV file untuk menyimpan semua metrik training
        log_file = os.path.join(log_dir, f"training_metrics_{timestamp}.csv")
        log_writer = open(log_file, 'w', newline='')
        csv_writer = csv.writer(log_writer)
        csv_writer.writerow([
            "step", "total_loss", "mlm_loss", "ctc_loss", 
            "mlm_accuracy", "wer", "f1_score", "learning_rate"
        ])
        
        # CSV file untuk validation metrics
        val_log_file = os.path.join(log_dir, f"validation_metrics_{timestamp}.csv")
        val_log_writer = open(val_log_file, 'w', newline='')
        val_csv_writer = csv.writer(val_log_writer)
        val_csv_writer.writerow([
            "step", "val_loss", "val_mlm_loss", "val_ctc_loss",
            "val_mlm_accuracy", "val_wer", "val_f1_score"
        ])
        
        print(f"📊 Logging training metrics to: {log_file}")
        print(f"📊 Logging validation metrics to: {val_log_file}")
    else:
        log_writer = None
        csv_writer = None
        val_log_writer = None
        val_csv_writer = None
    
    if is_main_process:
        print(f"Per-GPU batch size: {batch_size}")
        print(f"Gradient accumulation steps: {accumulation_steps}")
        print(f"Total effective batch size: {batch_size * world_size * accumulation_steps}")
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
        print("Loading training dataset from", train_dataset_path)
    train_hf_dataset = load_from_disk(train_dataset_path)
    if is_main_process:
        print("Training dataset size:", len(train_hf_dataset))

    # Load validation dataset
    if is_main_process:
        print("Loading validation dataset from", val_dataset_path)
    val_hf_dataset = load_from_disk(val_dataset_path)
    if is_main_process:
        print("Validation dataset size:", len(val_hf_dataset))

    dataset = FilePathDataset(train_hf_dataset, phoneme_tokenizer, mlm_prob=mlm_prob)
    val_dataset = FilePathDataset(val_hf_dataset, phoneme_tokenizer, mlm_prob=mlm_prob)

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
    
    # Validation DataLoader (no DistributedSampler needed, just sample randomly)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Use same batch size as training for memory efficiency
        shuffle=False,
        num_workers=2,
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
    
    # Inisialisasi GradScaler untuk Mixed Precision Training (FP16)
    # Gunakan torch.amp (bukan deprecated torch.cuda.amp)
    scaler = GradScaler(device="cuda")

    global_step = 0
    
    # ---------- RESUME FROM CHECKPOINT ----------
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        if is_main_process:
            print(f"📥 Loading checkpoint from {resume_from_checkpoint}")
        
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        
        # Load model state
        model.module.load_state_dict(checkpoint["model_state"])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Load global step
        global_step = checkpoint.get("global_step", 0)
        
        # Load scaler state jika ada
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        
        if is_main_process:
            print(f"✓ Resumed from step {global_step}")
            print(f"✓ Will continue training to step {max_steps}")
    elif resume_from_checkpoint:
        if is_main_process:
            print(f"⚠️  Checkpoint {resume_from_checkpoint} not found, starting from scratch")
    
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

            # Forward pass dengan Mixed Precision (FP16)
            with autocast(device_type="cuda", dtype=torch.float16):
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
                
                # Gradient Accumulation: bagi loss dengan accumulation steps
                loss = loss / accumulation_steps
            
            # Backward pass dengan GradScaler
            scaler.scale(loss).backward()

            # Optimizer step hanya dilakukan setiap accumulation_steps
            if (global_step + 1) % accumulation_steps == 0:
                # Unscale gradient sebelum clip_grad_norm
                scaler.unscale_(optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step dengan scaler
                scaler.step(optimizer)
                scaler.update()
                
                # Reset gradients
                optimizer.zero_grad()

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
                    # Kalikan loss dengan accumulation_steps untuk mendapatkan loss asli
                    metrics_tensor = torch.tensor([
                        (loss * accumulation_steps).item(), 
                        mlm_loss.item(), 
                        ctc_loss.item(),
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
                    total_loss_avg = (loss * accumulation_steps).item()
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

            # Evaluasi pada validation set
            if global_step % eval_every == 0:
                if is_main_process:
                    print(f"\n{'='*80}")
                    print(f"🔍 Running validation at step {global_step}...")
                    print(f"{'='*80}")
                
                val_metrics = evaluate(model, val_loader, device, ctc_loss_fn, mlm_prob, world_size)
                
                if is_main_process:
                    print(f"Validation Results:")
                    print(f"  Loss: {val_metrics['loss']:.4f}")
                    print(f"  MLM Loss: {val_metrics['mlm_loss']:.4f}")
                    print(f"  CTC Loss: {val_metrics['ctc_loss']:.4f}")
                    print(f"  MLM Accuracy: {val_metrics['mlm_accuracy']:.4f}")
                    print(f"  WER: {val_metrics['wer']:.4f}")
                    print(f"  F1 Score: {val_metrics['f1_score']:.4f}")
                    print(f"{'='*80}\n")
                    
                    # Write validation metrics to CSV
                    val_csv_writer.writerow([
                        global_step,
                        val_metrics['loss'],
                        val_metrics['mlm_loss'],
                        val_metrics['ctc_loss'],
                        val_metrics['mlm_accuracy'],
                        val_metrics['wer'],
                        val_metrics['f1_score']
                    ])
                    val_log_writer.flush()

            # Simpan checkpoint setiap 10rb step (hanya main process)
            if is_main_process and global_step % save_every == 0:
                ckpt_path = f"checkpoint_step_{global_step}.pt"
                torch.save({
                    "model_state": model.module.state_dict(),  # .module untuk unwrap DDP
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),  # Save GradScaler state
                    "global_step": global_step,
                }, ckpt_path)
                print(f"✓ Saved checkpoint to {ckpt_path}")

    # Simpan checkpoint final (hanya main process)
    if is_main_process:
        final_ckpt_path = f"checkpoint_step_{global_step}_final.pt"
        torch.save({
            "model_state": model.module.state_dict(),  # .module untuk unwrap DDP
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),  # Save GradScaler state
            "global_step": global_step,
        }, final_ckpt_path)
        print(f"✓ Training complete! Final checkpoint saved to {final_ckpt_path}")
        
        # Close log files
        if log_writer:
            log_writer.close()
            print(f"✓ Training logs saved")
        if val_log_writer:
            val_log_writer.close()
            print(f"✓ Validation logs saved")
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    train()
