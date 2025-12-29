from transformers import AutoTokenizer
import torch

class TextTokenizer:
    def __init__(self, model_name, use_fast=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "<mask>"})

        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.unk_id = self.tokenizer.unk_token_id
        self.vocab_size = len(self.tokenizer)

    def encode(self, sentence):
        return self.tokenizer.encode(sentence, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids).strip()

    def __len__(self):
        return self.vocab_size
