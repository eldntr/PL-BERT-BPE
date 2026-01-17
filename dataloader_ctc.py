import torch
from torch.utils.data import Dataset, Sampler
import random
import numpy as np

class LengthBucketSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        print("Computing phoneme lengths for bucketing...")
        self.lengths = []
        for idx in range(len(dataset)):
            ex = dataset.dataset[idx]
            total_len = sum(len(phon) for phon in ex["phonemes"])
            self.lengths.append((idx, total_len))
        self.lengths.sort(key=lambda x: x[1])
        print(f"Bucketing complete! Min length: {self.lengths[0][1]}, Max length: {self.lengths[-1][1]}")
    
    def __iter__(self):
        indices = [idx for idx, _ in self.lengths]
        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        return len(self.dataset)


class FilePathDataset(Dataset):
    def __init__(
        self,
        dataset,
        phoneme_tokenizer,
        mlm_prob=0.15,
        replace_prob=0.2,
        max_length=2048,
    ):
        self.dataset = dataset
        self.phoneme_tokenizer = phoneme_tokenizer
        self.mlm_prob = mlm_prob
        self.replace_prob = replace_prob
        self.max_length = max_length
        self.mask_id = phoneme_tokenizer.mask_id
        self.pad_id = phoneme_tokenizer.pad_id
        self.space_id = phoneme_tokenizer.space_id

        print(f"Filtering dataset: removing examples longer than {max_length} tokens...")
        original_len = len(dataset)
        self.valid_indices = []
        
        for idx in range(len(dataset)):
            ex = dataset[idx]
            seq_len = len(ex.get("phoneme_ids", []))
            
            if 0 < seq_len <= max_length:
                self.valid_indices.append(idx)
        
        removed = original_len - len(self.valid_indices)
        print(f"Dataset filtering complete: {original_len} → {len(self.valid_indices)} examples (removed {removed})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        ex = self.dataset[actual_idx]

        phoneme_ids = ex["phoneme_ids"]
        bpe_ids = ex["bpe_ids"]
        
        word_spans = []
        start = 0
        space_id = self.space_id
        
        for i, token_id in enumerate(phoneme_ids):
            if token_id == space_id:
                if i > start:
                    word_spans.append((start, i))
                start = i + 1
        
        if start < len(phoneme_ids):
            word_spans.append((start, len(phoneme_ids)))

        return {
            "phoneme_ids": phoneme_ids,
            "word_spans": word_spans,
            "bpe_ids": bpe_ids,
        }


def collate_fn(batch, phoneme_tokenizer, mlm_prob=0.15):

    pad_id = phoneme_tokenizer.pad_id
    mask_id = phoneme_tokenizer.mask_id

    # =========================
    # 1. Ambil sequence phoneme
    # =========================
    phon_seqs = [ex["phoneme_ids"] for ex in batch]   # list of lists
    spans     = [ex["word_spans"]   for ex in batch]
    bpe_seqs  = [ex["bpe_ids"]      for ex in batch]  # list of lists

    B = len(batch)
    max_T = max(len(x) for x in phon_seqs)

    # ============================
    # 2. Siapkan output container
    # ============================
    input_phon = torch.full((B, max_T), pad_id, dtype=torch.long)
    mlm_labels = torch.full((B, max_T), -100, dtype=torch.long)
    att_mask   = torch.zeros((B, max_T), dtype=torch.long)

    # ============================
    # 3. Whole word masking
    # ============================
    for i in range(B):
        seq = phon_seqs[i]
        L = len(seq)
        input_phon[i, :L] = torch.tensor(seq)
        att_mask[i, :L] = 1

        # pilih kata untuk masking
        word_spans = spans[i]
        num_words = len(word_spans)

        # ambil 15% kata
        num_mask = max(1, int(num_words * mlm_prob))
        chosen = random.sample(range(num_words), num_mask)

        # lakukan masking per kata
        for widx in chosen:
            start, end = word_spans[widx]
            
            # 80% mask
            if random.random() < 0.8:
                input_phon[i, start:end] = mask_id
            # 10% random phoneme
            elif random.random() < 0.5:
                random_ids = torch.randint(0, phoneme_tokenizer.vocab_size, (end-start,))
                input_phon[i, start:end] = random_ids
            # 10% keep original → nothing to do

            # set MLM label ke original
            mlm_labels[i, start:end] = torch.tensor(seq[start:end])

    # ============================
    # 4. Siapkan CTC targets
    # ============================
    # target = concat seluruh BPE per sample
    concat_bpe = []
    target_lengths = []

    for bpe_ids in bpe_seqs:
        concat_bpe.extend(bpe_ids)
        target_lengths.append(len(bpe_ids))

    concat_bpe = torch.tensor(concat_bpe, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    # input lengths untuk CTC (phoneme length sebelum padding)
    input_lengths = torch.tensor([len(seq) for seq in phon_seqs], dtype=torch.long)

    # final dictionary
    return {
        "phoneme_input": input_phon,        # [B, T]
        "mlm_labels": mlm_labels,           # [B, T]
        "attention_mask": att_mask,         # [B, T]
        "ctc_targets": concat_bpe,          # [sum_targets]
        "input_lengths": input_lengths,     # [B]
        "target_lengths": target_lengths,   # [B]
    }