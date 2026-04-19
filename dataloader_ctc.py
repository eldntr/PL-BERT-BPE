import torch
from torch.utils.data import Dataset
import random
import pickle
from text_utils import TextCleaner, _special

# TO DO:membatasi random token agar tidak memilih special token,

class FilePathDataset(Dataset):
    def __init__(
        self,
        dataset,                        
        token_maps="token_maps.pkl", 
        tokenizer="GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct",
        word_separator=220,
        token_separator=" ",
        token_mask="<mask>",
        token_pad="<pad>",
        max_mel_length=512,
        word_mask_prob=0.15,            
        phoneme_mask_prob=0.8,
        replace_prob=0.5,
    ):
        self.data = dataset
        self.max_mel_length = max_mel_length
        self.word_mask_prob = word_mask_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.text_cleaner = TextCleaner()

        self.word_separator = word_separator
        self.token_separator = token_separator
        self.token_mask = self.text_cleaner.word_index_dictionary.get(token_mask, self.text_cleaner.word_index_dictionary.get('<mask>', 4))
        self.pad_id = self.text_cleaner.word_index_dictionary.get(token_pad, self.text_cleaner.word_index_dictionary.get('<pad>', 0))

        from transformers import AutoTokenizer
        if tokenizer is not None and isinstance(tokenizer, str):
            _tok = AutoTokenizer.from_pretrained(tokenizer)
            self.bos_bpe = _tok.bos_token_id if _tok.bos_token_id is not None else 128000
            self.eos_bpe = _tok.eos_token_id if _tok.eos_token_id is not None else 128001
        else:
            self.bos_bpe = 128000
            self.eos_bpe = 128001

        self.sos_phon = self.text_cleaner.word_index_dictionary.get('<sos>', 1)
        self.eos_phon = self.text_cleaner.word_index_dictionary.get('<eos>', 2)

        with open(token_maps, 'rb') as handle:
            self.token_maps = pickle.load(handle)  
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        phoneme_words = ex["phonemes"]       
        bpe_words     = ex["bpe_ids"]        

        phon_ids_per_word = [self.text_cleaner(p) for p in phoneme_words]
        word_lens = [len(p_ids) for p_ids in phon_ids_per_word]
        total_len = sum(word_lens)

        if total_len > self.max_mel_length:
            num_words = len(phoneme_words)
            i = random.randint(0, num_words - 1)
            
            curr_len = 0
            start_idx = i
            end_idx = i
            
            for k in range(i, num_words):
                if curr_len + word_lens[k] <= self.max_mel_length:
                    curr_len += word_lens[k]
                    end_idx = k + 1
                else:
                    break
            
            for k in range(start_idx - 1, -1, -1):
                if curr_len + word_lens[k] <= self.max_mel_length:
                    curr_len += word_lens[k]
                    start_idx = k
                else:
                    break
            
            if start_idx == end_idx and num_words > 0:
                end_idx = start_idx + 1
            
            phon_ids_per_word = phon_ids_per_word[start_idx:end_idx]
            bpe_words = bpe_words[start_idx:end_idx]

        flat_phon = [self.sos_phon]
        word_spans = []   # (start, end) indexes
        curr = 1

        for phon_ids in phon_ids_per_word:
            if curr + len(phon_ids) > self.max_mel_length - 1:
                phon_ids = phon_ids[:self.max_mel_length - 1 - curr]
            
            if not phon_ids:
                break

            start = curr
            flat_phon.extend(phon_ids)
            curr += len(phon_ids)
            end = curr
            word_spans.append((start, end))
            
        flat_phon.append(self.eos_phon)

        if self.token_maps is not None:
            compact_bos = self.token_maps.get(self.bos_bpe, {}).get('token', self.bos_bpe)
            compact_eos = self.token_maps.get(self.eos_bpe, {}).get('token', self.eos_bpe)
        else:
            compact_bos = self.bos_bpe
            compact_eos = self.eos_bpe

        flat_bpe = [compact_bos]
        for ids in bpe_words:
            if self.token_maps is not None:
                ids = [self.token_maps.get(i, {}).get('token', i) for i in ids]
            flat_bpe.extend(ids)
        flat_bpe.append(compact_eos)

        return {
            "phoneme_ids": flat_phon,
            "word_spans": word_spans,
            "bpe_ids": flat_bpe,
        }


def collate_fn(batch, text_cleaner, word_mask_prob=0.15, phoneme_mask_prob=0.8, replace_prob=0.5):

    pad_id = text_cleaner.word_index_dictionary.get('<pad>', 0)
    mask_id = text_cleaner.word_index_dictionary.get('<mask>', 4)
    vocab_size = len(text_cleaner.word_index_dictionary)

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

        num_mask = max(1, int(num_words * word_mask_prob))
        chosen = random.sample(range(num_words), num_mask)

        for widx in chosen:
            start, end = word_spans[widx]
            
            if random.random() < phoneme_mask_prob:
                input_phon[i, start:end] = mask_id
            
            elif random.random() < replace_prob:
                random_ids = torch.randint(len(_special), vocab_size, (end-start,))
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
        "phoneme_input": input_phon,        # [B, T]
        "mlm_labels": mlm_labels,           # [B, T]
        "attention_mask": att_mask,         # [B, T]
        "ctc_targets": concat_bpe,          # [sum_targets]
        "input_lengths": input_lengths,     # [B]
        "target_lengths": target_lengths,   # [B]
    }
