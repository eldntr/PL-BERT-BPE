import re
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


def phonemize(text, text_tokenizer, phoneme_tokenizer):
    normalized = normalize_text(text)
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", normalized)

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

    # --- PRECOMPUTE SPACE TOKEN ---
    try:
        # encode spasi sebagai token
        space_token = text_tokenizer.encode_word(" ")
        space_token_id = space_token[0] if len(space_token) > 0 else None
    except:
        space_token_id = None

    for i, w in enumerate(words):

        # BPE encode normal (tanpa manipulasi spasi)
        bpe_ids = text_tokenizer.encode_word(w)
        if len(bpe_ids) == 0:
            continue

        # ----- Tambahkan spasi antar kata sebagai token -----
        if i > 0 and space_token_id is not None:
            bpe_ids = [space_token_id] + bpe_ids

        # ----- PHONEME -----
        phon_str = phonemize_word_espeak(w, ipa=True, keep_stress=False, sep=" ")

        # ----- APPEND -----
        output["words"].append(w)
        output["bpe_ids"].append(bpe_ids)
        output["phonemes"].append(phon_str)

    return output


if __name__ == "__main__":
    from text_tokenizer import TextTokenizer
    from phoneme_tokenizer import PhonemeTokenizer

    print("Loading tokenizers...")
    text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
    phoneme_tokenizer = PhonemeTokenizer()
    
    # Test cases
    test_texts = [
        "Hello world",
        "Hello world, this is a test.",
        "Saya suka makan nasi goreng.",
        "I love programming in Python.",
        "Ini adalah campuran bahasa Indonesia dan English words.",
    ]
    
    print("\n" + "="*60)
    print("Testing phonemize() function")
    print("="*60 + "\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: {text}")
        print("-" * 60)
        
        result = phonemize(text, text_tokenizer, phoneme_tokenizer)
        
        print(f"Before: {result['before']}")
        print(f"After:  {result['after']}")
        print(f"Words:  {result['words']}")
        print(f"Phonemes: {result['phonemes']}")
        print(f"BPE IDs: {result['bpe_ids']}")

        print("\nDecoded BPE tokens (per word):")
        for j, (word, bpe_ids) in enumerate(zip(result['words'], result['bpe_ids'])):
            decoded = text_tokenizer.decode(bpe_ids)
            print(f"  Word {j+1}: '{word}' -> IDs {bpe_ids} -> Decoded: '{decoded}'")

        print("\nDecoded full sentence from all BPE IDs:")
        all_bpe_ids = [bpe_id for bpe_ids in result['bpe_ids'] for bpe_id in bpe_ids]
        full_decoded = text_tokenizer.decode(all_bpe_ids)
        print(f"  All IDs: {all_bpe_ids}")
        print(f"  Decoded: '{full_decoded}'")
        
        print()
