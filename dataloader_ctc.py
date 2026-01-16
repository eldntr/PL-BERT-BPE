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
        mask_token_id=None,
        max_length=2048,
    ):
        self.phoneme_tokenizer = phoneme_tokenizer
        self.mlm_prob = mlm_prob
        self.mask_token_id = mask_token_id or phoneme_tokenizer.mask_id
        self.pad_id = phoneme_tokenizer.pad_id
        self.max_length = max_length
        print(f"Filtering dataset: removing examples longer than {max_length} tokens...")
        original_len = len(dataset)
        self.valid_indices = []
        for idx in range(len(dataset)):
            ex = dataset[idx]
            if "phoneme_ids" in ex and ex["phoneme_ids"] is not None:
                seq_len = len(ex["phoneme_ids"])
            else:
                seq_len = len(ex.get("phonemes", ""))
            if seq_len <= max_length and seq_len > 0:
                self.valid_indices.append(idx)
        self.dataset = dataset
        removed = original_len - len(self.valid_indices)
        print(f"Dataset filtering complete: {original_len} → {len(self.valid_indices)} examples (removed {removed})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        ex = self.dataset[actual_idx]
        if "phoneme_ids" in ex and ex["phoneme_ids"] is not None:
            flat_phon = ex["phoneme_ids"]
        else:
            phoneme_str = ex["phonemes"]
            flat_phon = self.phoneme_tokenizer.encode(phoneme_str)
        flat_bpe = ex["bpe_ids"]
        word_spans = []
        start = 0
        space_id = self.phoneme_tokenizer.space_id
        for i, token_id in enumerate(flat_phon):
            if token_id == space_id:
                if i > start:
                    word_spans.append((start, i))
                start = i + 1
        if start < len(flat_phon):
            word_spans.append((start, len(flat_phon)))

        return {
            "phoneme_ids": flat_phon,
            "word_spans": word_spans,
            "bpe_ids": flat_bpe,
        }


def collate_fn(batch, phoneme_tokenizer, mlm_prob=0.15):

    pad_id = phoneme_tokenizer.pad_id
    mask_id = phoneme_tokenizer.mask_id
    phon_seqs = [ex["phoneme_ids"] for ex in batch]
    spans     = [ex["word_spans"]   for ex in batch]
    bpe_seqs  = [ex["bpe_ids"]      for ex in batch]

    B = len(batch)
    max_T = max(len(x) for x in phon_seqs)
    input_phon = torch.full((B, max_T), pad_id, dtype=torch.long)
    mlm_labels = torch.full((B, max_T), -100, dtype=torch.long)
    att_mask   = torch.zeros((B, max_T), dtype=torch.long)
    for i in range(B):
        seq = phon_seqs[i]
        L = len(seq)
        input_phon[i, :L] = torch.tensor(seq)
        att_mask[i, :L] = 1
        word_spans = spans[i]
        num_words = len(word_spans)
        num_mask = max(1, int(num_words * mlm_prob))
        chosen = random.sample(range(num_words), num_mask)
        for widx in chosen:
            start, end = word_spans[widx]
            if random.random() < 0.8:
                input_phon[i, start:end] = mask_id
            elif random.random() < 0.5:
                random_ids = torch.randint(0, phoneme_tokenizer.vocab_size, (end-start,))
                input_phon[i, start:end] = random_ids
            mlm_labels[i, start:end] = torch.tensor(seq[start:end])
    concat_bpe = []
    target_lengths = []

    for bpe_ids in bpe_seqs:
        concat_bpe.extend(bpe_ids)
        target_lengths.append(len(bpe_ids))

    concat_bpe = torch.tensor(concat_bpe, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths = torch.tensor([len(seq) for seq in phon_seqs], dtype=torch.long)
    return {
        "phoneme_input": input_phon,
        "mlm_labels": mlm_labels,
        "attention_mask": att_mask,
        "ctc_targets": concat_bpe,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
    }