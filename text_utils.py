# IPA Phonemizer: https://github.com/bootphon/phonemizer

import os
import string

_pad = "$"
_mask = "█"  # Single character mask token
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad, _mask] + list(_punctuation) + list(_letters) + list(_letters_ipa)

letters = list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        self.pad_id = 0  # _pad is first symbol
        self.mask_id = 1  # _mask is second symbol
        self.vocab_size = len(dicts)
        print(f"TextCleaner vocab size: {self.vocab_size}")
    
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                indexes.append(self.word_index_dictionary['U']) # unknown token
#                 print(char)
        return indexes
    
    def encode(self, text):
        """Alias untuk __call__ agar kompatibel dengan interface phoneme_tokenizer"""
        return self.__call__(text)