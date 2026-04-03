import torch
from torch.utils.data import Dataset
import random
import pickle
from text_utils import TextCleaner

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
        max_mel_length=1536,
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
        self.token_mask = self.text_cleaner(token_mask)[0]
        self.pad_id = self.text_cleaner(token_pad)[0]

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

        flat_phon = []
        flat_bpe = []
        word_spans = []   # (start, end) indexes
        curr = 0

        # We need to align BPE to phonemes for 1-to-1 classification in the requested template
        for phon_ids, bpe_id_list in zip(phon_ids_per_word, bpe_words):
            if curr + len(phon_ids) > self.max_mel_length:
                phon_ids = phon_ids[:self.max_mel_length - curr]
            
            if not phon_ids:
                break

            # Take the first BPE token ID for this word (standard for PL-BERT 1-to-1 alignment)
            # If your framework expects CTC, this logic should be reversed.
            if bpe_id_list and self.token_maps is not None:
                orig_id = bpe_id_list[0]
                target_bpe_id = self.token_maps[orig_id]['token']
            else:
                target_bpe_id = 0 # Padding/Unknown

            start = curr
            flat_phon.extend(phon_ids)
            # Repeat BPE ID for every phoneme in the word
            flat_bpe.extend([target_bpe_id] * len(phon_ids))
            
            curr += len(phon_ids)
            end = curr
            word_spans.append((start, end))

        return {
            "phoneme_ids": flat_phon,
            "word_spans": word_spans,
            "bpe_ids": flat_bpe,
        }


from torch.utils.data import DataLoader
from functools import partial

def build_dataloader(dataset, batch_size, num_workers, dataset_config):
    dataset = FilePathDataset(
        dataset,
        token_maps=dataset_config["token_maps"],
        tokenizer=dataset_config["tokenizer"],
        word_separator=dataset_config["word_separator"],
        token_separator=dataset_config["token_separator"],
        token_mask=dataset_config["token_mask"],
        token_pad=dataset_config["token_pad"],
        max_mel_length=dataset_config["max_mel_length"],
        word_mask_prob=dataset_config["word_mask_prob"],
        phoneme_mask_prob=dataset_config["phoneme_mask_prob"],
        replace_prob=dataset_config["replace_prob"]
    )
    
    # We use partial to pass text_cleaner and other params to collate_fn
    collate = partial(
        collate_fn, 
        text_cleaner=dataset.text_cleaner, 
        word_mask_prob=dataset.word_mask_prob,
        phoneme_mask_prob=dataset.phoneme_mask_prob,
        replace_prob=dataset.replace_prob
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True
    )

def collate_fn(batch, text_cleaner, word_mask_prob=0.15, phoneme_mask_prob=0.8, replace_prob=0.5):

    pad_id = text_cleaner.pad_id
    mask_id = text_cleaner.mask_id

    phon_seqs = [ex["phoneme_ids"] for ex in batch]   
    spans     = [ex["word_spans"]   for ex in batch]
    bpe_seqs  = [ex["bpe_ids"]      for ex in batch] 

    B = len(batch)
    max_T = max(len(x) for x in phon_seqs)

    input_phon = torch.full((B, max_T), pad_id, dtype=torch.long)
    mlm_labels = torch.full((B, max_T), -100, dtype=torch.long)
    
    # We will return original phone IDs for ground truth
    all_phoneme_labels = torch.full((B, max_T), pad_id, dtype=torch.long)
    
    masked_indices_list = []

    for i in range(B):
        seq = phon_seqs[i]
        L = len(seq)
        input_phon[i, :L] = torch.tensor(seq)
        all_phoneme_labels[i, :L] = torch.tensor(seq)

        word_spans = spans[i]
        num_words = len(word_spans)

        num_mask = max(1, int(num_words * word_mask_prob))
        chosen = random.sample(range(num_words), num_mask)
        
        curr_masked_indices = []
        for widx in chosen:
            start, end = word_spans[widx]
            
            if random.random() < phoneme_mask_prob:
                input_phon[i, start:end] = mask_id
            
            elif random.random() < replace_prob:
                random_ids = torch.randint(0, text_cleaner.vocab_size, (end-start,))
                input_phon[i, start:end] = random_ids

            mlm_labels[i, start:end] = torch.tensor(seq[start:end])
            curr_masked_indices.extend(list(range(start, end)))
        
        masked_indices_list.append(torch.tensor(curr_masked_indices))

    # BPE ground truth
    # Template expects 'words' which we'll use for CTC target
    # Since CTCLoss expects concatenated targets and target_lengths, but the template loops zip(words_pred, words, ...)
    # the template might expect individual target sequences.
    bpe_targets = [torch.tensor(b) for b in bpe_seqs]
    # We should pad them or return as list if loop handles it. Template does _s2s_pred[:_text_length], _text_input[:_text_length].
    # This implies 1-to-1 or at least consistent indexing.
    # In PL-BERT-BPE, CTC doesn't have 1-to-1.
    # But I'll follow the template structure.
    
    # input_lengths for CTC and mask
    input_lengths = torch.tensor([len(seq) for seq in phon_seqs], dtype=torch.long)

    # Return words, labels, phonemes, input_lengths, masked_indices
    # words = BPE targets
    # labels = Phoneme targets (MLM)
    # phonemes = Input Phoneme IDs
    # input_lengths = Input lengths
    # masked_indices = Masked indices
    
    # We convert BPE to padded tensor for simpler batching if needed, or leave as list
    # The template uses zip(...) so a list of tensors or a padded tensor is fine.
    # Let's pad BPE targets.
    from torch.nn.utils.rnn import pad_sequence
    words = pad_sequence([torch.tensor(b) for b in bpe_seqs], batch_first=True, padding_value=0)

    return words, all_phoneme_labels, input_phon, input_lengths, masked_indices_list
