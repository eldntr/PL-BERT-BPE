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


def phonemize(text, text_tokenizer, max_length=2048):
    normalized = normalize_text(text)
    
    text_encoded = text_tokenizer.encode(normalized)
    bpe_ids = [text_tokenizer.bos_id] + text_encoded + [text_tokenizer.eos_id]

    result = ""
    i = 0
    
    while i < len(normalized):
        if normalized[i].isspace():
            result += normalized[i]
            i += 1
        elif normalized[i] in string.punctuation:
            result += normalized[i]
            i += 1
        else:
            word_start = i
            while i < len(normalized) and not normalized[i].isspace() and normalized[i] not in string.punctuation:
                i += 1
            word = normalized[word_start:i]
            phonemized = phonemize_word_espeak(word, ipa=True, keep_stress=True, sep="")
            result += phonemized
    
    return {
        "bpe_ids": bpe_ids,
        "phonemes": result
    }

if __name__ == "__main__":
    from text_tokenizer import TextTokenizer
    from phoneme_tokenizer import PhonemeTokenizer

    text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")

    sample_text = "hari ini! saya pergi, memancing"
    result = phonemize(sample_text, text_tokenizer)
    print(result)

    phoneme_tokenizer = PhonemeTokenizer()
    phoneme_tokenizer.build_from_sentence(result["phonemes"])  # Tokenize the phonemes
    encoded_phonemes = phoneme_tokenizer.encode(result["phonemes"])  # Encode the phonemes
    print(encoded_phonemes)  # Print encoded phonemes
    
    decoded_phonemes = phoneme_tokenizer.decode(encoded_phonemes)  # Decode the phonemes
    print(decoded_phonemes)  # Print decoded phonemes