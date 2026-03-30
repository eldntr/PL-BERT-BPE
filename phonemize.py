import re
import warnings
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
import string
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder
from text_normalize import normalize_text

warnings.filterwarnings("ignore", message="Trying to detect language from a single word.")

languages = [Language.ENGLISH, Language.INDONESIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

@lru_cache(maxsize=100_000)
def detect_lang(word: str) -> str:
    result = detector.detect_language_of(word)
    if result is None:
        return "id"
    return "en" if result == Language.ENGLISH else "id"


# Initialize backends globally for better performance
backend_en = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
backend_id = EspeakBackend(language='id', preserve_punctuation=True, with_stress=True)
global_separator = Separator(phone=" ", word="")

@lru_cache(maxsize=100_000)
def phonemize_word(word: str, keep_stress=False):
    """Return phoneme string (IPA) for 1 word using phonemizer."""
    lang = detect_lang(word)
    backend = backend_en if lang == "en" else backend_id
    
    # Process word with separator
    phon = backend.phonemize([word], separator=global_separator, strip=True)[0]
    
    if not keep_stress:
        phon = re.sub(r"[ˈˌ]", "", phon)
        
    return phon


def phonemize(text, text_tokenizer, phoneme_tokenizer):

    normalized = normalize_text(text)
    
    # Truncate setelah normalize: batasi ke <150 kata, potong di titik terdekat jika ada
    # Optional, sesuaikan dengan resources anda :>
    words_norm = normalized.split()
    if len(words_norm) > 150:
        truncated = " ".join(words_norm[:150])
        last_period = truncated.rfind(".")
        if last_period > 0:
            normalized = truncated[:last_period + 1]
        else:
            normalized = truncated
    
    output = {
        "before": text,
        "after": normalized,
        "words": [],
        "bpe_ids": [],
        "phonemes": [],
    }

    # Parse character by character untuk track spacing
    i = 0
    prev_was_space = False  
    
    while i < len(normalized):
        # Skip whitespace dan tandai bahwa kita melewati spasi
        if normalized[i].isspace():
            prev_was_space = True
            i += 1
            continue

        if normalized[i] in string.punctuation:
            punct = normalized[i]
            
            # Jika ada spasi sebelum punctuation (rare case), tambahkan space token
            if prev_was_space and len(output["phonemes"]) > 0:  
                output["words"].append(" ")
                output["phonemes"].append(phoneme_tokenizer.space_token)
                space_bpe = text_tokenizer.encode_word(" ")
                output["bpe_ids"].append(space_bpe)

            bpe_ids = text_tokenizer.encode_word(punct)
            if len(bpe_ids) == 0:
                bpe_ids = [text_tokenizer.unk_id]
            
            output["words"].append(punct)
            output["phonemes"].append(punct)
            output["bpe_ids"].append(bpe_ids)
            
            prev_was_space = False
            i += 1
            continue
        
        # Handle word (alphanumeric + apostrophe)
        word_start = i
        while i < len(normalized) and not normalized[i].isspace() and normalized[i] not in string.punctuation:
            i += 1
        word = normalized[word_start:i]
        
        if not word: 
            continue
        
        # Tambahkan space token jika ada spasi sebelum word ini (kecuali di awal)
        if prev_was_space and len(output["phonemes"]) > 0:  
            output["words"].append(" ")
            output["phonemes"].append(phoneme_tokenizer.space_token)
            space_bpe = text_tokenizer.encode_word(" ")
            output["bpe_ids"].append(space_bpe)
        
        # Process word
        bpe_ids = text_tokenizer.encode_word(word)
        if len(bpe_ids) == 0:
            bpe_ids = [text_tokenizer.unk_id]
        
        phon_str = phonemize_word(word, keep_stress=False)
        
        output["words"].append(word)
        output["phonemes"].append(phon_str)
        output["bpe_ids"].append(bpe_ids)
        
        prev_was_space = False

    return output


if __name__ == "__main__":
    from text_tokenizer import TextTokenizer
    from text_utils import TextCleaner

    # Contoh penggunaan
    text_tok = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
    phon_tok = TextCleaner()

    text = "Halo, nama saya Budi. Saya sedang belajar pemrograman."
    res = phonemize(text, text_tok, phon_tok)

    print("--- Phomemize Result ---")
    print(f"Original: {res['before']}")
    print(f"Normalized: {res['after']}")
    print(f"Words: {res['words']}")
    print(f"Phonemes: {res['phonemes']}")
    print(f"BPE IDs: {res['bpe_ids']}")