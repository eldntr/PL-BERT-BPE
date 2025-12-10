import re
import subprocess
import warnings
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder
# from text_normalize import normalize_text
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


def phonemize(text, text_tokenizer, phoneme_tokenizer):
    """
    text_tokenizer: instance TextTokenizer
    phoneme_tokenizer: instance PhonemeTokenizer
    """

    normalized = normalize_text(text)
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", normalized)

    # Skip if more than 50 words
    if len(words) > 50:
        return {
            "before": text,
            "after": "",
            "words": [],
            "bpe_ids": [],
            "phonemes": [],
        }

    output = {
        "before": text,
        "after": " ".join(words),
        "words": [],
        "bpe_ids": [],
        "phonemes": [],
    }

    for i, w in enumerate(words):
        
        _w = " " + w if i > 0 else w

        # ----- BPE -----
        bpe_ids = text_tokenizer.encode_word(_w)
        if len(bpe_ids) == 0:
            continue

        # ----- PHONEME -----
        phon_str = phonemize_word_espeak(w, ipa=True, keep_stress=False, sep=" ")

        # ----- APPEND -----
        output["words"].append(w)
        output["bpe_ids"].append(bpe_ids)
        output["phonemes"].append(phon_str)

    return output


if __name__ == "__main__":
    from text_tokenizer import TextTokenizer

    text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
    phoneme_tokenizer = PhonemeTokenizer()

    sample_text = "Halo dunia"
    result = phonemize(sample_text, text_tokenizer, phoneme_tokenizer) # ['Ġdun', 'ia']
    print(result)