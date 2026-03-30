# IPA Phonemizer: https://github.com/bootphon/phonemizer

import os
import string

_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʃʂʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘ᵻ̩̃"

_extra_punctuation = '-/()\"\'%&{}[]=+_*'
_punctuation += _extra_punctuation

# Multi-character symbols (diphthongs, affricates, R-colored vowels, etc.)
_multi_symbols = [
    'dʒ', 'tʃ', 'aɪ', 'eɪ', 'oʊ', 'aʊ', 'ɔɪ', 'iə',
    'ɪr', 'ɛr', 'ʊr', 'ɔːr', 'ɜːr', 'ɑːr',
    'n̩', 'ɑ̃', 'ɔ̃'
]

# Special tokens for training pipeline
_special = ['[PAD]', '<mask>', '[UNK]']

# Export all symbols:
symbols = _special + _multi_symbols + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        self.pad_id = dicts['[PAD]']
        self.mask_id = dicts['<mask>']
        self.unk_id = dicts['[UNK]']
        self.space_token = ' '
        self.vocab_size = len(dicts)


    def encode(self, text):
        if not text:
            return []
        
        # Direct match for special tokens or single characters
        if text in self.word_index_dictionary:
            return [self.word_index_dictionary[text]]
        
        # Handle space-separated phonemes (common in espeak output)
        if " " in text:
            units = text.split()
            ids = []
            for u in units:
                ids.extend(self.encode(u))
            return ids
        
        # Character fallback
        indexes = []
        for char in text:
            if char in self.word_index_dictionary:
                indexes.append(self.word_index_dictionary[char])
            else:
                indexes.append(self.unk_id)
        return indexes

    def __call__(self, text):
        return self.encode(text)