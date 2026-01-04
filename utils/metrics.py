"""
Metric calculation utilities for training evaluation
"""
import torch


def calculate_mlm_accuracy(logits, labels):
    """Hitung akurasi MLM (hanya untuk masked tokens)"""
    # logits: [B, T, V], labels: [B, T]
    predictions = torch.argmax(logits, dim=-1)  # [B, T]
    
    # Hanya hitung untuk token yang di-mask (labels != -100)
    mask = (labels != -100)
    if mask.sum() == 0:
        return 0.0
    
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def decode_ctc_greedy(log_probs, input_lengths, blank_id=0):
    """Greedy CTC decoding untuk mendapatkan prediksi token"""
    # log_probs: [T, B, V]
    # input_lengths: [B]
    predictions = torch.argmax(log_probs, dim=-1)  # [T, B]
    
    batch_decoded = []
    for b in range(predictions.shape[1]):
        pred_seq = predictions[:input_lengths[b], b].cpu().numpy()
        
        # Remove consecutive duplicates
        decoded = []
        prev = None
        for token in pred_seq:
            if token != prev:
                if token != blank_id:
                    decoded.append(token)
                prev = token
        
        batch_decoded.append(decoded)
    
    return batch_decoded


def calculate_wer(predictions, targets, target_lengths):
    """Hitung Word Error Rate (WER) untuk CTC"""
    # predictions: list of lists (decoded sequences)
    # targets: [sum_L] concatenated targets
    # target_lengths: [B]
    
    total_errors = 0
    total_words = 0
    
    # Split concatenated targets ke individual sequences
    targets_list = []
    offset = 0
    for length in target_lengths:
        targets_list.append(targets[offset:offset+length].cpu().numpy().tolist())
        offset += length
    
    for pred, target in zip(predictions, targets_list):
        # Simple WER: count substitutions, insertions, deletions
        # Gunakan edit distance sederhana
        d = edit_distance(pred, target)
        total_errors += d
        total_words += len(target)
    
    if total_words == 0:
        return 0.0
    
    return total_errors / total_words


def edit_distance(seq1, seq2):
    """Hitung edit distance (Levenshtein distance)"""
    len1, len2 = len(seq1), len(seq2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[len1][len2]


def calculate_token_f1(predictions, targets, target_lengths):
    """Hitung F1 score untuk token-level prediction"""
    # predictions: list of lists (decoded sequences)
    # targets: [sum_L] concatenated targets
    # target_lengths: [B]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Split concatenated targets
    targets_list = []
    offset = 0
    for length in target_lengths:
        targets_list.append(set(targets[offset:offset+length].cpu().numpy().tolist()))
        offset += length
    
    for pred, target_set in zip(predictions, targets_list):
        pred_set = set(pred)
        
        tp = len(pred_set & target_set)
        fp = len(pred_set - target_set)
        fn = len(target_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    if total_tp + total_fp == 0:
        precision = 0.0
    else:
        precision = total_tp / (total_tp + total_fp)
    
    if total_tp + total_fn == 0:
        recall = 0.0
    else:
        recall = total_tp / (total_tp + total_fn)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1, precision, recall
