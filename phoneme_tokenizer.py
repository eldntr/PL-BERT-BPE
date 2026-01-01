import json
import re

class PhonemeTokenizer:

    def __init__(self,
                 phoneme2id=None,
                 pad_token="<pad>",
                 mask_token="<mask>",
                 blank_token="<blank>",
                 space_token="<space>",
                 bos_token="<bos>",
                 eos_token="<eos>"
                 ):

        self.phoneme2id = phoneme2id or {}

        for tok in [pad_token, mask_token, blank_token, space_token, bos_token, eos_token]:
            if tok not in self.phoneme2id:
                self.phoneme2id[tok] = len(self.phoneme2id)

        self.id2phoneme = {v: k for k, v in self.phoneme2id.items()}

        self.pad_id = self.phoneme2id[pad_token]
        self.mask_id = self.phoneme2id[mask_token]
        self.blank_id = self.phoneme2id[blank_token]  
        self.space_id = self.phoneme2id[space_token]
        self.space_token = space_token
        self.bos_id = self.phoneme2id[bos_token]
        self.eos_id = self.phoneme2id[eos_token]

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            mapping = json.load(f)
        return cls(mapping)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.phoneme2id, f, indent=2, ensure_ascii=False)

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

    def tokenize_sequence(self, p_str):
        """
        Pisahkan IPA output espeak menjadi unit phoneme (per-karakter IPA).
        Spasi akan dijadikan token khusus <space>.
        Contoh:
            "hˈeɪloʊ" → ["h","ˈ","e","ɪ","l","o","ʊ"]
            "h e l o" → ["h","<space>","e","<space>","l","<space>","o"]
        """
        p_str = p_str.strip()
        
        # Tokenize per karakter IPA, replace spasi dengan token khusus
        tokens = []
        for char in p_str:
            if char == ' ':
                tokens.append(self.space_token)
            else:
                tokens.append(char)
        
        return tokens

    def encode(self, phoneme_str):
        units = self.tokenize_sequence(phoneme_str)
        return [self.phoneme2id.get(u, self.blank_id) for u in units]

    def decode(self, ids):
        return [self.id2phoneme.get(i, self.blank_id) for i in ids]

    @property
    def vocab_size(self):
        return len(self.phoneme2id)
