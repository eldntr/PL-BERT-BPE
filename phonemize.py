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
    sentences = re.split(r'([.!?]+\s+)', normalized)
    
    all_bpe_ids = []
    all_phonemes = []
    current_chunk_text = ""
    current_chunk_tokens = 0

    max_chunk_size = max_length - 2
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        test_encoded = text_tokenizer.encode(sentence)

        if current_chunk_tokens + len(test_encoded) > max_chunk_size and current_chunk_text:

            chunk_encoded = text_tokenizer.encode(current_chunk_text)
            chunk_bpe_ids = [text_tokenizer.bos_id] + chunk_encoded + [text_tokenizer.eos_id]

            result = ""
            i = 0
            
            while i < len(current_chunk_text):
                if current_chunk_text[i].isspace():
                    result += current_chunk_text[i]
                    i += 1
                elif current_chunk_text[i] in string.punctuation:
                    result += current_chunk_text[i]
                    i += 1
                else:
                    word_start = i
                    while i < len(current_chunk_text) and not current_chunk_text[i].isspace() and current_chunk_text[i] not in string.punctuation:
                        i += 1
                    word = current_chunk_text[word_start:i]
                    phonemized = phonemize_word_espeak(word, ipa=True, keep_stress=True, sep="")
                    result += phonemized

            if all_bpe_ids:
                chunk_bpe_ids = chunk_bpe_ids[1:]  
            all_bpe_ids.extend(chunk_bpe_ids[:-1]) 
            all_phonemes.append(result)

            current_chunk_text = sentence
            current_chunk_tokens = len(test_encoded)
        else:
            current_chunk_text += sentence
            current_chunk_tokens += len(test_encoded)

    if current_chunk_text:
        chunk_encoded = text_tokenizer.encode(current_chunk_text)
        chunk_bpe_ids = [text_tokenizer.bos_id] + chunk_encoded + [text_tokenizer.eos_id]

        result = ""
        i = 0
        
        while i < len(current_chunk_text):
            if current_chunk_text[i].isspace():
                result += current_chunk_text[i]
                i += 1
            elif current_chunk_text[i] in string.punctuation:
                result += current_chunk_text[i]
                i += 1
            else:
                word_start = i
                while i < len(current_chunk_text) and not current_chunk_text[i].isspace() and current_chunk_text[i] not in string.punctuation:
                    i += 1
                word = current_chunk_text[word_start:i]
                phonemized = phonemize_word_espeak(word, ipa=True, keep_stress=True, sep="")
                result += phonemized
        
        if all_bpe_ids:
            chunk_bpe_ids = chunk_bpe_ids[1:] 
        all_bpe_ids.extend(chunk_bpe_ids)
        all_phonemes.append(result)

    if not all_bpe_ids:
        all_bpe_ids = [text_tokenizer.bos_id, text_tokenizer.eos_id]
    
    phonemes = "".join(all_phonemes)
    
    return {
        "bpe_ids": all_bpe_ids,
        "phonemes": phonemes
    }

if __name__ == "__main__":
    from text_tokenizer import TextTokenizer
    from phoneme_tokenizer import PhonemeTokenizer

    text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")

    sample_text = open("test.txt").read()
    result = phonemize(sample_text, text_tokenizer)
    print(result)

    phoneme_tokenizer = PhonemeTokenizer()
    phoneme_tokenizer.build_from_sentence(result["phonemes"])  # Tokenize the phonemes
    encoded_phonemes = phoneme_tokenizer.encode(result["phonemes"])  # Encode the phonemes
    print(encoded_phonemes)  # Print encoded phonemes
    
    decoded_phonemes = phoneme_tokenizer.decode(encoded_phonemes)  # Decode the phonemes
    print(decoded_phonemes)  # Print decoded phonemes