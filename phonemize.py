import re
import warnings
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
import string
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder
from text_normalize import normalize_text

warnings.filterwarnings("ignore", message="Trying to detect language from a single word.")

class Phonemizer:
    def __init__(self):
        self.languages = [Language.ENGLISH, Language.INDONESIAN]
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        self.backend_en = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
        self.backend_id = EspeakBackend(language='id', preserve_punctuation=True, with_stress=True)

    @lru_cache(maxsize=100_000)
    def detect_lang(self, word: str) -> str:
        result = self.detector.detect_language_of(word)
        if result is None:
            return "id"
        return "en" if result == Language.ENGLISH else "id"

    @lru_cache(maxsize=100_000)
    def __call__(self, word: str):
        lang = self.detect_lang(word)
        backend = self.backend_en if lang == "en" else self.backend_id
        phon = backend.phonemize([word], strip=True)[0]
        return phon

phonemizer_instance = Phonemizer()

def phonemize(text, tokenizer):

    normalized = normalize_text(text)
    
    output = {
        "before": text,
        "after": normalized,
        "words": [],
        "bpe_ids": [],
        "phonemes": [],
    }

    i = 0
    prev_was_space = False  
    
    while i < len(normalized):
        if normalized[i].isspace():
            prev_was_space = True
            i += 1
            continue

        if normalized[i] in string.punctuation:
            punct = normalized[i]
            
            if prev_was_space and len(output["phonemes"]) > 0:  
                output["words"].append(" ")
                output["phonemes"].append(" ")
                space_bpe = tokenizer.encode(" ", add_special_tokens=False)
                output["bpe_ids"].append(space_bpe)

            bpe_ids = tokenizer.encode(punct, add_special_tokens=False)
            
            output["words"].append(punct)
            output["phonemes"].append(punct)
            output["bpe_ids"].append(bpe_ids)
            
            prev_was_space = False
            i += 1
            continue

        word_start = i
        while i < len(normalized) and not normalized[i].isspace() and normalized[i] not in string.punctuation:
            i += 1
        word = normalized[word_start:i]
        
        if not word: 
            continue
 
        if prev_was_space and len(output["phonemes"]) > 0:  
            output["words"].append(" ")
            output["phonemes"].append(" ")
            space_bpe = tokenizer.encode(" ", add_special_tokens=False)
            output["bpe_ids"].append(space_bpe)

        bpe_ids = tokenizer.encode(word, add_special_tokens=False)
        
        phon_str = phonemizer_instance(word)
        
        output["words"].append(word)
        output["phonemes"].append(phon_str)
        output["bpe_ids"].append(bpe_ids)
        
        prev_was_space = False

    return output


if __name__ == "__main__":
    from text_utils import TextCleaner
    from transformers import AutoTokenizer

    text_tok = AutoTokenizer.from_pretrained("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
    phon_tok = TextCleaner()

    text = "Halo, nama saya Budi. Saya sedang belajar pemrograman."
    res = phonemize(text, text_tok)

    print("--- Phomemize Result ---")
    print(f"Original: {res['before']}")
    print(f"Normalized: {res['after']}")
    print(f"Words: {res['words']}")
    print(f"Phonemes: {res['phonemes']}")
    print(f"BPE IDs: {res['bpe_ids']}")

    print(phon_tok.encode(" "))
    print(phon_tok.encode(res['phonemes']))