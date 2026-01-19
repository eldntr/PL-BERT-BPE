import re
import subprocess
import warnings
import string
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
    
    Parse text character by character untuk mempertahankan spacing asli.
    Tanda baca menempel pada kata sebelumnya tanpa spasi.
    """

    normalized = normalize_text(text)
    
    # Truncate setelah normalize: batasi ke <150 kata, potong di titik terdekat jika ada
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

    # --- Tambahkan BOS di awal ---
    output["words"].append(phoneme_tokenizer.bos_token)
    output["phonemes"].append(phoneme_tokenizer.bos_token)
    output["bpe_ids"].append([text_tokenizer.bos_id])

    # Parse character by character untuk track spacing
    i = 0
    prev_was_space = False  # Track apakah karakter sebelumnya adalah spasi
    
    while i < len(normalized):
        # Skip whitespace dan tandai bahwa kita melewati spasi
        if normalized[i].isspace():
            prev_was_space = True
            i += 1
            continue
            
        # Handle punctuation
        if normalized[i] in string.punctuation:
            punct = normalized[i]
            
            # Jika ada spasi sebelum punctuation (rare case), tambahkan space token
            if prev_was_space and len(output["phonemes"]) > 1:  # > 1 karena sudah ada BOS
                output["words"].append(" ")
                output["phonemes"].append(phoneme_tokenizer.space_token)
                space_bpe = text_tokenizer.encode_word(" ")
                output["bpe_ids"].append(space_bpe)
            
            # Process punctuation
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
        
        if not word:  # Safety check
            continue
        
        # Tambahkan space token jika ada spasi sebelum word ini (kecuali di awal)
        if prev_was_space and len(output["phonemes"]) > 1:  # > 1 karena sudah ada BOS
            output["words"].append(" ")
            output["phonemes"].append(phoneme_tokenizer.space_token)
            space_bpe = text_tokenizer.encode_word(" ")
            output["bpe_ids"].append(space_bpe)
        
        # Process word
        bpe_ids = text_tokenizer.encode_word(word)
        if len(bpe_ids) == 0:
            bpe_ids = [text_tokenizer.unk_id]
        
        phon_str = phonemize_word_espeak(word, ipa=True, keep_stress=False, sep=" ")
        
        output["words"].append(word)
        output["phonemes"].append(phon_str)
        output["bpe_ids"].append(bpe_ids)
        
        prev_was_space = False

    # --- Tambahkan EOS di akhir ---
    output["words"].append(phoneme_tokenizer.eos_token)
    output["phonemes"].append(phoneme_tokenizer.eos_token)
    output["bpe_ids"].append([text_tokenizer.eos_id])

    return output