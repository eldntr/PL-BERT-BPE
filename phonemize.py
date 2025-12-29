import re
import string
import subprocess
import warnings
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


def phonemize(text, text_tokenizer):
    normalized = normalize_text(text)
    bpe_ids = [text_tokenizer.bos_id] + text_tokenizer.encode(normalized) + [text_tokenizer.eos_id]

    pattern = r'\w+|[^\w\s]'
    words = re.findall(pattern, normalized)

    phonemes = []
    for word in words:
        if word in string.punctuation:
            phonemes.append(word)
        else:
            phonemized = phonemize_word_espeak(word, ipa=True, keep_stress=True, sep="")
            phonemes.append(phonemized)

    phonemes = " ".join(phonemes)

    return {
        "bpe_ids": bpe_ids, 
        "phonemes": phonemes
    }

if __name__ == "__main__":
    from text_tokenizer import TextTokenizer
    from phoneme_tokenizer import PhonemeTokenizer

    text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")

    sample_text = "Hello world!"
    result = phonemize(sample_text, text_tokenizer)
    print(result)

    print(text_tokenizer.decode(result["bpe_ids"]))