import torch
from torch.utils.data import Dataset
import random

class FilePathDataset(Dataset):
    def __init__(
        self,
        dataset,                   
        phoneme_tokenizer,      
        text_tokenizer,           
        mlm_prob=0.15,            
        mask_token_id=None,       
        max_position_embeddings=1536,  
    ):
        self.dataset = dataset
        self.phoneme_tokenizer = phoneme_tokenizer
        self.text_tokenizer = text_tokenizer
        self.mlm_prob = mlm_prob
        self.mask_token_id = mask_token_id or phoneme_tokenizer.mask_id
        self.pad_id = phoneme_tokenizer.pad_id
        self.max_position_embeddings = max_position_embeddings
        
        print(f"Filtering samples longer than {max_position_embeddings} tokens...")
        valid_indices = []
        skipped_count = 0
        
        for idx in range(len(dataset)):
            ex = dataset[idx]
            phoneme_words = ex["phonemes"]

            total_phon_len = sum(len(self.phoneme_tokenizer.encode(phon_str)) for phon_str in phoneme_words)
            
            if total_phon_len <= max_position_embeddings:
                valid_indices.append(idx)
            else:
                skipped_count += 1
        
        print(f"Filtered dataset: {len(valid_indices)}/{len(dataset)} samples valid ({skipped_count} skipped)")
        
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        ex = self.dataset[original_idx]

        phoneme_words = ex["phonemes"]       
        bpe_words     = ex["bpe_ids"]        

        flat_phon = []
        word_spans = []   # (start, end) indexes
        curr = 0

        for phoneme_str in phoneme_words:
            phon_ids = self.phoneme_tokenizer.encode(phoneme_str)
            start = curr
            flat_phon.extend(phon_ids)
            curr += len(phon_ids)
            end = curr
            word_spans.append((start, end))

        flat_bpe = []
        for ids in bpe_words:
            if self.text_tokenizer.use_pruning:
                ids = [self.text_tokenizer.original_to_compact[i] for i in ids]
            
            flat_bpe.extend(ids)

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

    # Whole word masking
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

            mlm_labels[i, start:end] = torch.tensor(seq[start:end])

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
