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
        text_tokenizer,
        mlm_prob=0.15,
        replace_prob=0.2,
        max_position_embedding=512,
    ):
        self.dataset = dataset
        self.phoneme_tokenizer = phoneme_tokenizer
        self.text_tokenizer = text_tokenizer
        self.mlm_prob = mlm_prob
        self.replace_prob = replace_prob
        self.max_position_embedding = max_position_embedding
        self.mask_id = phoneme_tokenizer.mask_id
        self.pad_id = phoneme_tokenizer.pad_id
        self.space_id = phoneme_tokenizer.space_id

        self.period_phoneme_id = phoneme_tokenizer.encode(".")[0] if phoneme_tokenizer.encode(".") else None
        period_bpe = text_tokenizer.encode(".")
        self.period_bpe_id = period_bpe[0] if period_bpe else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]

        phoneme_str = ex["phonemes"]
        bpe_ids = ex["bpe_ids"]
        
        phoneme_ids = self.phoneme_tokenizer.encode(phoneme_str)
 
        if len(phoneme_ids) > self.max_position_embedding:
            phoneme_ids, bpe_ids = self._truncate_at_period(phoneme_ids, bpe_ids)
        
        word_spans = []
        start = 0
        
        for i, token_id in enumerate(phoneme_ids):
            if token_id == self.space_id:
                if i > start:
                    word_spans.append((start, i))
                start = i + 1
        
        if start < len(phoneme_ids):
            word_spans.append((start, len(phoneme_ids)))
        
        num_words = len(word_spans)
        num_to_mask = max(1, int(num_words * self.mlm_prob))
        
        words_to_mask = set(random.sample(range(num_words), min(num_to_mask, num_words)))
        
        masked_phoneme_ids = phoneme_ids.copy()
        mlm_labels = [-100] * len(phoneme_ids)
        
        for word_idx in words_to_mask:
            start, end = word_spans[word_idx]
            
            for pos in range(start, end):
                mlm_labels[pos] = phoneme_ids[pos]
            
            rand_val = random.random()
            
            if rand_val < (1 - self.replace_prob):
                for pos in range(start, end):
                    masked_phoneme_ids[pos] = self.mask_id
            elif rand_val < (1 - self.replace_prob / 2):
                for pos in range(start, end):
                    masked_phoneme_ids[pos] = random.randint(0, self.phoneme_tokenizer.vocab_size - 1)

        return {
            "phoneme_ids": masked_phoneme_ids,
            "mlm_labels": mlm_labels,
            "bpe_ids": bpe_ids,
        }
    
    def _truncate_at_period(self, phoneme_ids, bpe_ids):
        """Memotong phoneme_ids dan bpe_ids pada titik (period) yang sama (kemunculan ke-N)."""
        if self.period_phoneme_id is None or self.period_bpe_id is None:
            return phoneme_ids[:self.max_position_embedding], bpe_ids[:self.max_position_embedding]

        # Cari semua posisi period di kedua sequence
        phoneme_period_positions = [i for i, pid in enumerate(phoneme_ids) if pid == self.period_phoneme_id]
        bpe_period_positions = [i for i, bid in enumerate(bpe_ids) if bid == self.period_bpe_id]
        
        if not phoneme_period_positions:
            return phoneme_ids[:self.max_position_embedding], bpe_ids[:self.max_position_embedding]

        # Cari period terakhir sebelum max_position_embedding di phoneme_ids
        valid_periods = [pos for pos in phoneme_period_positions if pos < self.max_position_embedding]
        
        if not valid_periods:
            return phoneme_ids[:self.max_position_embedding], bpe_ids[:self.max_position_embedding]
        
        # Hitung ini adalah period ke-berapa (0-indexed)
        # Contoh: jika valid_periods[-1] adalah posisi 13 dan itu period ketiga, maka period_index = 2
        last_valid_period_pos = valid_periods[-1]
        period_index = phoneme_period_positions.index(last_valid_period_pos)
 
        # Potong phoneme_ids setelah period terakhir yang valid (inclusive period)
        phoneme_cut_pos = last_valid_period_pos + 1

        # Potong bpe_ids pada period yang sama (period ke-period_index)
        if period_index < len(bpe_period_positions):
            bpe_cut_pos = bpe_period_positions[period_index] + 1
        else:
            # Jika bpe_ids memiliki lebih sedikit period, gunakan semua bpe_ids
            bpe_cut_pos = len(bpe_ids)
        
        return phoneme_ids[:phoneme_cut_pos], bpe_ids[:bpe_cut_pos]


def collate_fn(batch, phoneme_tokenizer, mlm_prob=0.15):
    pad_id = phoneme_tokenizer.pad_id

    phon_seqs = [ex["phoneme_ids"] for ex in batch]
    mlm_labels_list = [ex["mlm_labels"] for ex in batch]
    bpe_seqs = [ex["bpe_ids"] for ex in batch]

    # Filter out empty sequences
    valid_indices = [i for i, seq in enumerate(phon_seqs) if len(seq) > 0]
    if not valid_indices:
        # If all sequences are empty, return a batch with minimum length 1
        valid_indices = [0]
        phon_seqs[0] = [pad_id]
        mlm_labels_list[0] = [-100]
        bpe_seqs[0] = [0]
    
    phon_seqs = [phon_seqs[i] for i in valid_indices]
    mlm_labels_list = [mlm_labels_list[i] for i in valid_indices]
    bpe_seqs = [bpe_seqs[i] for i in valid_indices]

    B = len(phon_seqs)
    max_T = max(len(x) for x in phon_seqs) if phon_seqs else 1

    input_phon = torch.full((B, max_T), pad_id, dtype=torch.long)
    mlm_labels = torch.full((B, max_T), -100, dtype=torch.long)
    att_mask = torch.zeros((B, max_T), dtype=torch.long)

    for i in range(B):
        seq = phon_seqs[i]
        labels = mlm_labels_list[i]
        L = len(seq)
        
        input_phon[i, :L] = torch.tensor(seq)
        mlm_labels[i, :L] = torch.tensor(labels)
        att_mask[i, :L] = 1

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

