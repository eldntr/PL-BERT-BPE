import re
import string
import subprocess
import warnings
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

from phoneme_tokenizer import PhonemeTokenizer
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


@lru_cache(maxsize=100_000)
def phonemize_word_espeak(word: str, ipa=True, keep_stress=False, sep=" "):
    """Return phoneme string (IPA) for 1 word."""
    lang = detect_lang(word)
    voice = "en-us" if lang == "en" else "id"

    cmd = ["espeak-ng", "-v", voice, "-q", f"--sep={sep}"]

    if ipa:
        cmd.insert(3, "--ipa")
    else:
        cmd.insert(3, "-x")

    cmd.append(word)

    try:
        out = subprocess.run(cmd, capture_output=True, timeout=5)
        phon = out.stdout.decode("utf-8", errors="ignore").strip()
        phon = phon.replace("\ufeff", "")
        if not keep_stress:
            phon = re.sub(r"[ˈˌ]", "", phon)
        return phon
    except:
        return word


def phonemize(text, llama_tokenizer):
    text = normalize_text(text)
    words = re.findall(r'\w+|[^\w\s]', text)

    tokenized = llama_tokenizer.tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    
    input_ids = tokenized["input_ids"]

    phonemes = []
    for word in words:
        if word in string.punctuation:
            phonemes.append(word)  
        else:
            phon = phonemize_word_espeak(word, ipa=True, keep_stress=True, sep="")
            phonemes.append(phon if phon else word)
    
    return {
        'input_ids': input_ids,
        'phonemes': phonemes
    }


if __name__ == "__main__":

    from text_tokenizer import TextTokenizer

    tokenizer = TextTokenizer(model_name="GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct", use_fast=True)
    sample_text = "Hello world!"
    result = phonemize(sample_text, tokenizer)
    print(result)