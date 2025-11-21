from transformers import AutoTokenizer
import torch

class TextTokenizer:
    def __init__(self, model_name, use_fast=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

        # Fallback pad token (LLaMA tidak punya PAD)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.unk_id = self.tokenizer.unk_token_id
        self.vocab_size = len(self.tokenizer)

    def encode_word(self, word):
        """Encode satu kata tanpa special tokens (untuk P2BPE)."""
        return self.tokenizer.encode(word, add_special_tokens=False)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def decode(self, ids):
        return self.tokenizer.decode(ids).strip()

    def __len__(self):
        return self.vocab_size
