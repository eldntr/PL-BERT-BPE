import torch
from torch.utils.data import Dataset, Sampler
import random
import numpy as np

from text_utils import TextCleaner

class LengthBucketSampler(Sampler):
    """
    Bucket sampler berdasarkan PANJANG INPUT PHONEME SEBENARNYA
    (setelah TextCleaner, termasuk spasi).
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        print("Computing phoneme lengths for bucketing...")
        self.lengths = []

        for idx in range(len(dataset)):
            ex = dataset.dataset[idx]

            phoneme_str = " ".join(ex["phonemes"])
            phoneme_ids = dataset.text_cleaner(phoneme_str)

            total_len = len(phoneme_ids)
            self.lengths.append((idx, total_len))
        
        self.lengths.sort(key=lambda x: x[1])
        print(
            f"Bucketing complete! "
            f"Min length: {self.lengths[0][1]}, "
            f"Max length: {self.lengths[-1][1]}"
        )
    
    def __iter__(self):
        indices = [idx for idx, _ in self.lengths]
        batches = [
            indices[i:i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        return len(self.dataset)


class FilePathDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.text_cleaner = TextCleaner()
        self.pad_id = self.text_cleaner.word_index_dictionary["$"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]

        # phoneme: ['həlˈoʊ', 'wˈɜːld', '!'] → "həlˈoʊ wˈɜːld !"
        phoneme_str = " ".join(ex["phonemes"])
        phoneme_ids = self.text_cleaner(phoneme_str)

        # ===== SHIFT BPE (+1) =====
        bpe_ids = [i + 1 for i in ex["input_ids"]]

        return {
            "phoneme_ids": phoneme_ids,
            "bpe_ids": bpe_ids,
        }
    

def collate_fn(batch, text_cleaner, mlm_prob=0.15):

    pad_id = text_cleaner.word_index_dictionary["$"]
    mask_id = pad_id  # PL-BERT style: mask == pad
    vocab_size = len(text_cleaner.word_index_dictionary)

    phon_seqs = [ex["phoneme_ids"] for ex in batch]
    bpe_seqs  = [ex["bpe_ids"]     for ex in batch]

    B = len(batch)
    max_T = max(len(x) for x in phon_seqs)

    input_phon = torch.full((B, max_T), pad_id, dtype=torch.long)
    mlm_labels = torch.full((B, max_T), -100, dtype=torch.long)
    att_mask   = torch.zeros((B, max_T), dtype=torch.long)

    for i in range(B):
        seq = phon_seqs[i]
        L = len(seq)

        input_phon[i, :L] = torch.tensor(seq, dtype=torch.long)
        att_mask[i, :L] = 1

        # === char-level MLM masking ===
        if mlm_prob <= 0:
            num_mask = 0
            mask_idx = []
        else:
            num_mask = max(1, int(L * mlm_prob))
            mask_idx = random.sample(range(L), num_mask)

        for j in mask_idx:
            mlm_labels[i, j] = seq[j]

            r = random.random()
            if r < 0.8:
                input_phon[i, j] = mask_id
            elif r < 0.9:
                input_phon[i, j] = random.randint(0, vocab_size - 1)
            # else: keep original

    concat_bpe = []
    target_lengths = []

    for ids in bpe_seqs:
        concat_bpe.extend(ids)
        target_lengths.append(len(ids))

    return {
        "phoneme_input": input_phon,            # [B, T]
        "mlm_labels": mlm_labels,               # [B, T]
        "attention_mask": att_mask,             # [B, T]
        "ctc_targets": torch.tensor(concat_bpe, dtype=torch.long),
        "input_lengths": torch.tensor(
            [len(x) for x in phon_seqs], dtype=torch.long
        ),
        "target_lengths": torch.tensor(
            target_lengths, dtype=torch.long
        ),
    }


def build_dataloader(hf_dataset, batch_size, num_workers=0, mlm_prob=0.15, shuffle_batches=True):
    """
    Dataloader yang match dengan:
      - FilePathDataset: shift BPE +1 (blank=0 untuk CTC)
      - collate_fn: menghasilkan dict batch untuk model.compute_loss()
    """
    ds = FilePathDataset(hf_dataset)

    sampler = LengthBucketSampler(ds, batch_size=batch_size, shuffle=shuffle_batches)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, ds.text_cleaner, mlm_prob=mlm_prob),
    )
    return loader, ds