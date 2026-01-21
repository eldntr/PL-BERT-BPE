from transformers import AutoTokenizer
import json
import torch

class TextTokenizer:
    def __init__(self, model_name, map_file=None, use_fast=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.unk_id = self.tokenizer.unk_token_id

        self.map_file = map_file
        self.original_to_compact = {}
        self.compact_to_original = {}
        self.use_pruning = False
        
        if map_file:
            print(f"Loading BPE Map from {map_file}")
            with open(map_file, 'r') as f:
                data = json.load(f)
                self.original_to_compact = {int(k): v for k, v in data["original_to_compact"].items()}
                self.compact_to_original = {int(k): v for k, v in data["compact_to_original"].items()}
            self.use_pruning = True
            self.vocab_size = len(self.original_to_compact)
        else:
            self.vocab_size = len(self.tokenizer)

    def encode_word(self, word):
        """Encode word. Jika pruning aktif, return Compact IDs."""
        raw_ids = self.tokenizer.encode(word, add_special_tokens=False)
        if self.use_pruning:
            return [self.original_to_compact.get(rid, self.original_to_compact.get(self.unk_id, 0)) for rid in raw_ids]
        return raw_ids

    def decode(self, ids):
        """Decode IDs. Jika pruning aktif, convert Compact -> Original dulu."""
        if self.use_pruning:
            original_ids = [self.compact_to_original.get(i, self.unk_id) for i in ids]
            return self.tokenizer.decode(original_ids).strip()
        else:
            return self.tokenizer.decode(ids).strip()

    def map_batch_ids(self, batch_ids_list):
        """Helper untuk mapping batch data yang sudah terlanjur raw ID di dataset"""
        if not self.use_pruning:
            return batch_ids_list
        
        mapped_batch = []
        for seq in batch_ids_list:
            mapped_seq = [self.original_to_compact.get(oid, self.original_to_compact.get(self.unk_id)) for oid in seq]
            mapped_batch.append(mapped_seq)
        return mapped_batch

    def __len__(self):
        return self.vocab_size