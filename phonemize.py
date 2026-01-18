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
    
    # --- MODIFIKASI 1: Regex baru untuk menangkap kata DAN tanda baca ---
    # Regex lama hanya alphanumeric: r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*"
    # Regex baru: Menangkap kata (\w+) ATAU non-whitespace/non-word ([^\w\s])
    words = re.findall(r"\w+(?:'\w+)*|[^\w\s]", normalized)

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

    # --- MODIFIKASI 2: Tambahkan BOS di awal ---
    output["words"].append(phoneme_tokenizer.bos_token)
    output["phonemes"].append(phoneme_tokenizer.bos_token)
    output["bpe_ids"].append([text_tokenizer.bos_id])  # Ambil BOS ID dari TextTokenizer

    for i, w in enumerate(words):
        # --- MODIFIKASI 3: Tambahkan SPASI sebelum kata (kecuali kata pertama) ---
        if i > 0:
            output["words"].append(" ")
            output["phonemes"].append(phoneme_tokenizer.space_token)  # Token <space>
            
            # Encode spasi untuk BPE (penting agar BPE tidak nempel)
            # LLaMA tokenizer biasanya sensitif spasi
            space_bpe = text_tokenizer.encode_word(" ")
            output["bpe_ids"].append(space_bpe)

        # ----- PROCESS WORD / PUNCTUATION -----
        # 1. BPE
        bpe_ids = text_tokenizer.encode_word(w)
        # Jika kosong (misal karakter aneh), skip atau beri UNK
        if len(bpe_ids) == 0:
            bpe_ids = [text_tokenizer.unk_id]

        # 2. PHONEME
        # Cek apakah ini tanda baca? Jika ya, jangan masuk espeak (bisa error/aneh)
        if re.match(r"^[^\w\s]+$", w):
            phon_str = w  # Gunakan tanda baca itu sendiri sebagai phoneme
        else:
            phon_str = phonemize_word_espeak(w, ipa=True, keep_stress=False, sep=" ")

        # ----- APPEND -----
        output["words"].append(w)
        output["bpe_ids"].append(bpe_ids)
        output["phonemes"].append(phon_str)

    # --- MODIFIKASI 4: Tambahkan EOS di akhir ---
    output["words"].append(phoneme_tokenizer.eos_token)
    output["phonemes"].append(phoneme_tokenizer.eos_token)
    output["bpe_ids"].append([text_tokenizer.eos_id])

    return output