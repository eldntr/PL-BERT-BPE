"""
Evaluation utilities
"""
import torch
import torch.nn.functional as F
from utils.metrics import (
    calculate_mlm_accuracy, 
    decode_ctc_greedy, 
    calculate_wer, 
    calculate_token_f1
)


def evaluate_model(model, val_loader, device, ctc_loss_fn, max_eval_steps=None):
    """Evaluate model on validation set"""
    model.eval()
    total_mlm_loss = 0.0
    total_ctc_loss = 0.0
    total_mlm_acc = 0.0
    total_wer = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_eval_steps and i >= max_eval_steps:
                break
                
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
            
            # Metrics
            mlm_acc = calculate_mlm_accuracy(mlm_logits, mlm_labels)
            ctc_predictions = decode_ctc_greedy(ctc_log_probs, input_lengths, blank_id=0)
            wer = calculate_wer(ctc_predictions, ctc_targets, target_lengths)
            f1, precision, recall = calculate_token_f1(ctc_predictions, ctc_targets, target_lengths)
            
            total_mlm_loss += mlm_loss.item()
            total_ctc_loss += ctc_loss.item()
            total_mlm_acc += mlm_acc
            total_wer += wer
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            num_batches += 1
    
    model.train()
    
    if num_batches == 0:
        return {}
    
    return {
        "mlm_loss": total_mlm_loss / num_batches,
        "ctc_loss": total_ctc_loss / num_batches,
        "mlm_acc": total_mlm_acc / num_batches,
        "wer": total_wer / num_batches,
        "f1": total_f1 / num_batches,
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
    }
