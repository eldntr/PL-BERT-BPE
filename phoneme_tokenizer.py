import json
import re

class PhonemeTokenizer:

    def __init__(self,
                 phoneme2id=None,
                 pad_token="<pad>",
                 mask_token="<mask>",
                 blank_token="<blank>"):

        self.phoneme2id = phoneme2id or {}

        for tok in [pad_token, mask_token, blank_token]:
            if tok not in self.phoneme2id:
                self.phoneme2id[tok] = len(self.phoneme2id)

        self.id2phoneme = {v: k for k, v in self.phoneme2id.items()}

        self.pad_id = self.phoneme2id[pad_token]
        self.mask_id = self.phoneme2id[mask_token]
        self.blank_id = self.phoneme2id[blank_token]

    # ----------------------------- SAVE/LOAD -----------------------------
    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            mapping = json.load(f)
        return cls(mapping)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.phoneme2id, f, indent=2)

    # ----------------------------- VOCAB BUILD -----------------------------
    def add_phoneme(self, p):
        if p not in self.phoneme2id:
            idx = len(self.phoneme2id)
            self.phoneme2id[p] = idx
            self.id2phoneme[idx] = p

    def build_from_sentence(self, phoneme_str):
        """phoneme input 's a j a' atau 'saja'."""
        units = self.tokenize_sequence(phoneme_str)
        for u in units:
            self.add_phoneme(u)

    # ----------------------------- TOKENIZE -----------------------------
    def tokenize_sequence(self, p_str):
        """
        Pisahkan IPA output espeak menjadi unit phoneme.
        Contoh:
            "s a j a" → ["s","a","j","a"]
            "sajamakan" → ["s","a","j","a","m","a","k","a","n"]
        """
        p_str = p_str.strip()

        if " " in p_str:
            return p_str.split()

        # fallback: per karakter IPA
        return list(p_str)

    # ----------------------------- ENCODE/DECODE -----------------------------
    def encode(self, phoneme_str):
        units = self.tokenize_sequence(phoneme_str)
        return [self.phoneme2id.get(u, self.blank_id) for u in units]

    def decode(self, ids):
        return [self.id2phoneme.get(i, "<unk>") for i in ids]

    @property
    def vocab_size(self):
        return len(self.phoneme2id)
