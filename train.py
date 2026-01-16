import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets import load_from_disk

from text_tokenizer import TextTokenizer
from phoneme_tokenizer import PhonemeTokenizer
from dataloader_ctc import FilePathDataset, collate_fn
from model import MultiTaskModel

from utils.config_utils import (
    load_config, 
    setup_logging, 
    log_training_step, 
    log_validation_step, 
    log_completion
)
from utils.metrics import (
    calculate_mlm_accuracy, 
    decode_ctc_greedy, 
    calculate_wer, 
    calculate_token_f1
)
from utils.evaluation import evaluate_model


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def train():
    config = load_config("config.yaml")
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main_process = (local_rank == 0)
    
    if is_main_process:
        print(f"🚀 Training with {world_size} GPUs (DDP)")
    
    dataset_path = config['data']['dataset_path']
    phoneme_vocab_path = config['data']['phoneme_vocab_path']
    text_tokenizer_name = config['data']['text_tokenizer_name']
    model_config = config['model']
    train_config = config['training']
    split_config = config['split']
    save_dir = config['logging']['save_dir']
    
    batch_size = train_config['batch_size']
    gradient_accumulation_steps = train_config['gradient_accumulation_steps']
    max_steps = train_config['max_steps']
    save_every = train_config['save_every']
    eval_every = train_config['eval_every']
    eval_steps = train_config['eval_steps']
    lr = train_config['learning_rate']
    mlm_prob = train_config['mlm_prob']
    lambda_ctc = train_config['lambda_ctc']
    log_every = train_config['log_every']

    device = torch.device(f"cuda:{local_rank}")

    train_log_file, val_log_file = setup_logging(config, local_rank, world_size)
    
    if is_main_process:
        print(f"Per-GPU batch size: {batch_size}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Effective batch: {batch_size * world_size * gradient_accumulation_steps}")
        if train_log_file:
            print(f"Train log: {train_log_file}")
            print(f"Val log: {val_log_file}")

    text_tokenizer = TextTokenizer(text_tokenizer_name)
    bpe_vocab_size = len(text_tokenizer)

    phoneme_tokenizer = PhonemeTokenizer.load(phoneme_vocab_path)
    phoneme_vocab_size = phoneme_tokenizer.vocab_size

    if is_main_process:
        print("Phoneme vocab size:", phoneme_vocab_size)
        print("BPE vocab size:", bpe_vocab_size)

    if is_main_process:
        print(f"Loading dataset from {dataset_path}...")
    
    hf_dataset = load_from_disk(dataset_path)

    test_size = split_config['val'] + split_config['test']
    train_test_split = hf_dataset.train_test_split(test_size=test_size, seed=split_config['seed'])
    train_dataset = train_test_split['train']
    temp_dataset = train_test_split['test']
    
    val_ratio = split_config['val'] / (split_config['val'] + split_config['test'])
    val_test_split = temp_dataset.train_test_split(test_size=1-val_ratio, seed=split_config['seed'])
    val_dataset = val_test_split['train']
    
    if is_main_process:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_ds = FilePathDataset(
        train_dataset,
        phoneme_tokenizer,
        mlm_prob=mlm_prob,
        max_length=model_config['max_position_embeddings']
    )
    val_ds = FilePathDataset(
        val_dataset,
        phoneme_tokenizer,
        mlm_prob=0.0,
        max_length=model_config['max_position_embeddings']
    )

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=local_rank, 
        shuffle=True, seed=split_config['seed']
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=local_rank, 
        shuffle=False, seed=split_config['seed']
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=train_config['num_workers'],
        collate_fn=lambda batch: collate_fn(
            batch,
            phoneme_tokenizer,
            mlm_prob=mlm_prob
        ),
        pin_memory=train_config['pin_memory']
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=train_config['num_workers'],
        collate_fn=lambda batch: collate_fn(
            batch,
            phoneme_tokenizer,
            mlm_prob=0.0
        ),
        pin_memory=train_config['pin_memory']
    )

    model = MultiTaskModel(
        phoneme_vocab_size=phoneme_vocab_size,
        bpe_vocab_size=bpe_vocab_size,
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        intermediate_size=model_config['intermediate_size'],
        max_position_embeddings=model_config['max_position_embeddings'],
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    global_step = 0
    accumulation_step = 0  
    accumulated_mlm_loss = 0.0
    accumulated_ctc_loss = 0.0
    accumulated_total_loss = 0.0
    
    accumulated_mlm_acc = 0.0
    accumulated_wer = 0.0
    accumulated_f1 = 0.0
    accumulated_precision = 0.0
    accumulated_recall = 0.0
    
    model.train()

    while global_step < max_steps:
        train_sampler.set_epoch(global_step // len(train_loader))
        
        for batch in train_loader:
            if global_step >= max_steps:
                break

            phoneme_input = batch["phoneme_input"].to(device)      # [B, T]
            mlm_labels     = batch["mlm_labels"].to(device)         # [B, T]
            attention_mask = batch["attention_mask"].to(device)     # [B, T]

            ctc_targets    = batch["ctc_targets"].to(device)        # [sum_L]
            input_lengths  = batch["input_lengths"].to(device)      # [B]
            target_lengths = batch["target_lengths"].to(device)     # [B]

            mlm_logits, ctc_logits = model(phoneme_input, attention_mask=attention_mask)

            B, T, Vp = mlm_logits.shape
            mlm_loss = F.cross_entropy(
                mlm_logits.view(B*T, Vp),
                mlm_labels.view(B*T),
                ignore_index=-100
            )

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
            
            with torch.no_grad():
                mlm_acc = calculate_mlm_accuracy(mlm_logits, mlm_labels)
                ctc_predictions = decode_ctc_greedy(ctc_log_probs, input_lengths, blank_id=0)
                wer = calculate_wer(ctc_predictions, ctc_targets, target_lengths)
                f1, precision, recall = calculate_token_f1(ctc_predictions, ctc_targets, target_lengths)
            
            loss = loss / gradient_accumulation_steps
            loss.backward()

            accumulated_mlm_loss += mlm_loss.item()
            accumulated_ctc_loss += ctc_loss.item()
            accumulated_total_loss += (mlm_loss.item() + lambda_ctc * ctc_loss.item())
            accumulated_mlm_acc += mlm_acc
            accumulated_wer += wer
            accumulated_f1 += f1
            accumulated_precision += precision
            accumulated_recall += recall
            accumulation_step += 1

            if accumulation_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1

                avg_mlm_loss = accumulated_mlm_loss / gradient_accumulation_steps
                avg_ctc_loss = accumulated_ctc_loss / gradient_accumulation_steps
                avg_total_loss = accumulated_total_loss / gradient_accumulation_steps
                avg_mlm_acc = accumulated_mlm_acc / gradient_accumulation_steps
                avg_wer = accumulated_wer / gradient_accumulation_steps
                avg_f1 = accumulated_f1 / gradient_accumulation_steps
                avg_precision = accumulated_precision / gradient_accumulation_steps
                avg_recall = accumulated_recall / gradient_accumulation_steps

                accumulated_mlm_loss = 0.0
                accumulated_ctc_loss = 0.0
                accumulated_total_loss = 0.0
                accumulated_mlm_acc = 0.0
                accumulated_wer = 0.0
                accumulated_f1 = 0.0
                accumulated_precision = 0.0
                accumulated_recall = 0.0

                if is_main_process and global_step % log_every == 0:
                    print(
                        f"Step {global_step}/{max_steps} | "
                        f"Loss: {avg_total_loss:.4f} | MLM: {avg_mlm_loss:.4f} | CTC: {avg_ctc_loss:.4f}\n"
                        f"  MLM Acc: {avg_mlm_acc:.4f} | WER: {avg_wer:.4f} | "
                        f"F1: {avg_f1:.4f} | P: {avg_precision:.4f} | R: {avg_recall:.4f}"
                    )
                    
                    metrics = {
                        'total_loss': avg_total_loss, 'mlm_loss': avg_mlm_loss, 
                        'ctc_loss': avg_ctc_loss, 'mlm_acc': avg_mlm_acc, 
                        'wer': avg_wer, 'f1': avg_f1, 
                        'precision': avg_precision, 'recall': avg_recall
                    }
                    log_training_step(train_log_file, global_step, metrics)

                if is_main_process and global_step % save_every == 0:
                    ckpt_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pt")
                    torch.save({
                        "model_state": model.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "global_step": global_step,
                    }, ckpt_path)
                    print(f"✓ Saved checkpoint to {ckpt_path}")

                if is_main_process and global_step % eval_every == 0:
                    print(f"\n{'='*60}")
                    print(f"Running validation at step {global_step}...")
                    val_metrics = evaluate_model(model, val_loader, device, ctc_loss_fn, eval_steps)
                    if val_metrics:
                        print(
                            f"[VAL] Loss: {val_metrics['mlm_loss'] + val_metrics['ctc_loss']:.4f} | "
                            f"MLM: {val_metrics['mlm_loss']:.4f} | CTC: {val_metrics['ctc_loss']:.4f}\n"
                            f"      MLM Acc: {val_metrics['mlm_acc']:.4f} | WER: {val_metrics['wer']:.4f} | "
                            f"F1: {val_metrics['f1']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}"
                        )
                        log_validation_step(val_log_file, global_step, val_metrics)
                    print(f"{'='*60}\n")

    if is_main_process:
        final_ckpt_path = os.path.join(save_dir, f"checkpoint_step_{global_step}_final.pt")
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
        }, final_ckpt_path)
        print(f"✓ Training complete! Final checkpoint saved to {final_ckpt_path}")
        log_completion(train_log_file, val_log_file, global_step)

    cleanup_ddp()


if __name__ == "__main__":
    train()
