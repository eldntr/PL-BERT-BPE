import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader

from text_utils import TextCleaner
from dataloader_ctc import FilePathDataset, collate_fn
from model import MultiTaskModel
import yaml

def compute_mlm_accuracy(logits, labels):
    """Compute MLM accuracy (ignoring -100 labels)"""
    B, T, V = logits.shape
    predictions = logits.argmax(dim=-1).view(-1)  # [B*T]
    labels_flat = labels.view(-1)  # [B*T]
    
    # Mask out -100 (ignored tokens)
    mask = labels_flat != -100
    if mask.sum() == 0:
        return 0.0
    
    correct = (predictions[mask] == labels_flat[mask]).sum().item()
    total = mask.sum().item()
    return correct / total


def compute_f1_macro(logits, labels, num_classes_to_consider=50):
    B, T, V = logits.shape
    predictions = logits.argmax(dim=-1).view(-1)  # [B*T]
    labels_flat = labels.view(-1)  # [B*T]
    
    # Mask out -100 (ignored tokens)
    mask = labels_flat != -100
    if mask.sum() == 0:
        return 0.0
    
    predictions_masked = predictions[mask]
    labels_masked = labels_flat[mask]

    unique_classes = torch.unique(labels_masked).cpu().numpy()
    num_classes = min(len(unique_classes), num_classes_to_consider)
    
    if num_classes == 0:
        return 0.0

    f1_scores = []
    for cls in unique_classes[:num_classes]:
        tp = ((predictions_masked == cls) & (labels_masked == cls)).sum().item()
        fp = ((predictions_masked == cls) & (labels_masked != cls)).sum().item()
        fn = ((predictions_masked != cls) & (labels_masked == cls)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def levenshtein_distance(s1, s2):
    """Standard Levenshtein distance for WER calculation."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
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

def compute_ctc_wer(ctc_logits, ctc_targets, input_lengths, target_lengths):
    """Compute CTC Word Error Rate (WER) using Levenshtein distance"""
    predictions = ctc_logits.argmax(dim=-1)  # [B, T]
    
    total_wer_distance = 0
    total_wer_targets = 0
    
    start_idx = 0
    for i in range(len(target_lengths)):
        end_idx = start_idx + target_lengths[i].item()
        target_seq = ctc_targets[start_idx:end_idx].cpu().tolist()
        
        # Greedy decode
        input_len = input_lengths[i].item()
        pred_seq = predictions[i, :input_len].cpu().tolist()
        
        pred_decoded = []
        prev_idx = -1
        for idx in pred_seq:
            if idx != prev_idx and idx != 0:
                pred_decoded.append(idx - 1)
            prev_idx = idx
        
        # Calculate standard Levenshtein distance
        distance = levenshtein_distance(pred_decoded, target_seq)
        total_wer_distance += distance
        total_wer_targets += len(target_seq)
        
        start_idx = end_idx
    
    wer = total_wer_distance / max(total_wer_targets, 1)
    return wer


@torch.no_grad()
def evaluate(
    model,
    test_loader,
    device,
    checkpoint_path=None
):
    """Evaluate model on test set with all metrics"""
    model.eval()
    
    total_mlm_loss = 0.0
    total_mlm_acc = 0.0
    total_f1_macro = 0.0
    total_ctc_loss = 0.0
    total_wer = 0.0
    num_batches = 0
    
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    
    print(f"Evaluating model from checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Number of test batches: {len(test_loader)}")
    print("-" * 80)
    
    for batch_idx, batch in enumerate(test_loader):
        if (batch_idx + 1) % 100 == 0:
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...")
        
        phoneme_input = batch["phoneme_input"].to(device)      # [B, T]
        mlm_labels     = batch["mlm_labels"].to(device)         # [B, T]
        attention_mask = batch["attention_mask"].to(device)     # [B, T]

        ctc_targets    = batch["ctc_targets"].to(device)        # [sum_L]
        input_lengths  = batch["input_lengths"].to(device)      # [B]
        target_lengths = batch["target_lengths"].to(device)     # [B]

        mlm_logits, ctc_logits = model(phoneme_input, attention_mask=attention_mask)

        # ----- MLM LOSS & ACCURACY & F1 MACRO -----
        B, T, Vp = mlm_logits.shape
        mlm_loss = F.cross_entropy(
            mlm_logits.view(B*T, Vp),
            mlm_labels.view(B*T),
            ignore_index=-100
        )
        mlm_acc = compute_mlm_accuracy(mlm_logits, mlm_labels)
        f1_macro = compute_f1_macro(mlm_logits, mlm_labels, num_classes_to_consider=50)

        # ----- CTC LOSS & WER -----
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, C]
        ctc_targets_shifted = ctc_targets + 1

        ctc_loss = ctc_loss_fn(
            ctc_log_probs,
            ctc_targets_shifted,
            input_lengths,
            target_lengths
        )
        
        wer = compute_ctc_wer(ctc_logits, ctc_targets, input_lengths, target_lengths)

        total_mlm_loss += mlm_loss.item()
        total_mlm_acc += mlm_acc
        total_f1_macro += f1_macro
        total_ctc_loss += ctc_loss.item()
        total_wer += wer
        num_batches += 1

    model.train()

    avg_mlm_loss = total_mlm_loss / max(num_batches, 1)
    avg_mlm_acc = total_mlm_acc / max(num_batches, 1)
    avg_f1_macro = total_f1_macro / max(num_batches, 1)
    avg_ctc_loss = total_ctc_loss / max(num_batches, 1)
    avg_wer = total_wer / max(num_batches, 1)
    
    return {
        "mlm_loss": avg_mlm_loss,
        "mlm_acc": avg_mlm_acc,
        "f1_macro": avg_f1_macro,
        "ctc_loss": avg_ctc_loss,
        "wer": avg_wer,
    }


def main():
    # Load configuration
    config_path = "Configs/config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    dataset_path = config["data_folder"]
    test_dataset_path = f"{dataset_path}/test"
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]

    # Try to find the latest .t7 checkpoint if hardcoded one is not found
    checkpoint_path = "checkpoint_step_1000000_final.t7" 
    
    batch_size = config.get("batch_size", 1)  # Or smaller for eval
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"🔍 Model Evaluation Script")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print()
   
    print(f"Loading test dataset from {test_dataset_path}...")
    hf_test_dataset = load_from_disk(test_dataset_path)
    print(f"Test dataset size: {len(hf_test_dataset)}")
    
    test_dataset = FilePathDataset(
        hf_test_dataset, 
        token_maps=dataset_params["token_maps"],
        tokenizer=dataset_params["tokenizer"],
        word_separator=dataset_params["word_separator"],
        token_separator=dataset_params["token_separator"],
        token_mask=dataset_params["token_mask"],
        token_pad=dataset_params["token_pad"],
        max_mel_length=dataset_params["max_mel_length"],
        word_mask_prob=dataset_params["word_mask_prob"],
        phoneme_mask_prob=dataset_params["phoneme_mask_prob"],
        replace_prob=dataset_params["replace_prob"]
    )
    
    bpe_vocab_size = len(test_dataset.token_maps)
    phoneme_vocab_size = len(test_dataset.text_cleaner.word_index_dictionary)
    
    print(f"Phoneme vocab size: {phoneme_vocab_size}")
    print(f"BPE vocab size: {bpe_vocab_size}")
    print()

    print(f"Test dataset after filtering: {len(test_dataset)}")
    print()
    
    # Setting masking to 0 for clean evaluation by default
    word_mask_prob = 0.0
    phoneme_mask_prob = 0.0
    replace_prob = 0.0
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=lambda batch: collate_fn(
            batch, 
            test_dataset.text_cleaner, 
            word_mask_prob=word_mask_prob,
            phoneme_mask_prob=phoneme_mask_prob,
            replace_prob=replace_prob
        ),
        pin_memory=True
    )
  
    print(f"Loading model...")
    model = MultiTaskModel(
        phoneme_vocab_size=phoneme_vocab_size,
        bpe_vocab_size=bpe_vocab_size,
        hidden_size=model_params["hidden_size"],
        num_layers=model_params["num_hidden_layers"],
        num_heads=model_params["num_attention_heads"],
        intermediate_size=model_params["intermediate_size"],
        max_position_embeddings=model_params["max_position_embeddings"],
    ).to(device)

    if not os.path.exists(checkpoint_path):
        import glob
        checkpoints = glob.glob("checkpoint_step_*.t7")
        if not checkpoints:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
        
        def get_step(ckpt):
            try:
                step_str = ckpt.split("checkpoint_step_")[-1].replace("_final.t7", "").replace(".t7", "")
                return int(step_str)
            except ValueError:
                return -1
                
        checkpoint_path = max(checkpoints, key=get_step)
        print(f"⚠️ Hardcoded checkpoint not found. Using latest: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
    print()

    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    metrics = evaluate(
        model,
        test_loader,
        device,
        checkpoint_path=checkpoint_path
    )
    
    print()
    print("-" * 80)
    print("FINAL METRICS:")
    print("-" * 80)
    print(f"MLM Loss:       {metrics['mlm_loss']:.4f}")
    print(f"MLM Accuracy:   {metrics['mlm_acc']:.4f} ({metrics['mlm_acc']*100:.2f}%)")
    print(f"F1 Macro Score: {metrics['f1_macro']:.4f}")
    print(f"CTC Loss:       {metrics['ctc_loss']:.4f}")
    print(f"WER:            {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print("-" * 80)

    results_file = f"eval_results_{os.path.basename(checkpoint_path)}.json"
    with open(results_file, "w") as f:
        json.dump({
            "checkpoint": checkpoint_path,
            "metrics": metrics
        }, f, indent=2)
    print(f"✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()
