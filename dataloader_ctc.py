import torch
from torch.utils.data import Dataset
import random

# TO DO:membatasi random token agar tidak memilih special token,

class FilePathDataset(Dataset):
    def __init__(
        self,
        dataset,                   
        phoneme_tokenizer,      
        token_maps=None,           
        mlm_prob=0.15,            
        mask_token_id=None,       
        max_position_embeddings=1536,  
    ):
        self.dataset = dataset
        self.phoneme_tokenizer = phoneme_tokenizer
        self.token_maps = token_maps
        self.mlm_prob = mlm_prob
        self.mask_token_id = mask_token_id or phoneme_tokenizer.mask_id
        self.pad_id = phoneme_tokenizer.pad_id
        self.max_position_embeddings = max_position_embeddings

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]

        phoneme_words = ex["phonemes"]       
        bpe_words     = ex["bpe_ids"]        

        # Calculate phoneme IDs and lengths for each word
        phon_ids_per_word = [self.phoneme_tokenizer.encode(p) for p in phoneme_words]
        word_lens = [len(p_ids) for p_ids in phon_ids_per_word]
        total_len = sum(word_lens)

        if total_len > self.max_position_embeddings:
            # Random truncate: pick a contiguous window of words that fits
            num_words = len(phoneme_words)
            i = random.randint(0, num_words - 1)
            
            curr_len = 0
            start_idx = i
            end_idx = i
            
            # Forward expand
            for k in range(i, num_words):
                if curr_len + word_lens[k] <= self.max_position_embeddings:
                    curr_len += word_lens[k]
                    end_idx = k + 1
                else:
                    break
            
            # Backward expand if there's room left
            for k in range(start_idx - 1, -1, -1):
                if curr_len + word_lens[k] <= self.max_position_embeddings:
                    curr_len += word_lens[k]
                    start_idx = k
                else:
                    break
            
            # Final window: [start_idx, end_idx)
            # Note: in the extreme case where a single word is longer than max_position_embeddings,
            # end_idx might be start_idx or start_idx + 1 if we allow at least one word.
            # Here we ensure at least one word is taken if it's the start word.
            if start_idx == end_idx and num_words > 0:
                end_idx = start_idx + 1
                # We'll truncate this single word later if needed
            
            phon_ids_per_word = phon_ids_per_word[start_idx:end_idx]
            bpe_words = bpe_words[start_idx:end_idx]

        flat_phon = []
        word_spans = []   # (start, end) indexes
        curr = 0

        for phon_ids in phon_ids_per_word:
            # Truncate if a single word is still too long (rare)
            if curr + len(phon_ids) > self.max_position_embeddings:
                phon_ids = phon_ids[:self.max_position_embeddings - curr]
            
            if not phon_ids:
                break

            start = curr
            flat_phon.extend(phon_ids)
            curr += len(phon_ids)
            end = curr
            word_spans.append((start, end))

        flat_bpe = []
        for ids in bpe_words:
            if self.token_maps is not None:
                # User's requested format: token_maps[w]['token']
                # This assumes all IDs in bpe_words were included in the pruned vocab.
                ids = [self.token_maps[i]['token'] for i in ids]
            
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
