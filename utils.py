import torch
import os

def length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    return mask

def scan_checkpoint(cp_dir, prefix):
    files = os.listdir(cp_dir)
    ckpts = []
    for f in files:
        if f.startswith(prefix) and f.endswith(".t7"):
            ckpts.append(f)
    if not ckpts:
        return None
    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts]
    latest_iter = sorted(iters)[-1]
    return latest_iter
