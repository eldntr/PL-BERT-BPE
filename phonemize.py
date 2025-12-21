import re
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
    tokens = re.findall(r'\w+|[^\w\s]', text)

    all_bpe_ids = []
    phonemes = []
    
    for i, token in enumerate(tokens):
        
        _token = " " + token if i > 0 else token

        token_bpe_ids = text_tokenizer.encode_word(_token)
        if len(token_bpe_ids) == 0:
            continue

        if re.match(r'^[^\w\s]+$', token):
            phon_str = token
        else:
            phon_str = phonemize_word_espeak(token, ipa=True, keep_stress=True, sep="")

        all_bpe_ids.append(token_bpe_ids)
        phonemes.append(phon_str)

    return {
        "bpe_ids": all_bpe_ids,
        "phonemes": phonemes,
    }


if __name__ == "__main__":
    from text_tokenizer import TextTokenizer
    from text_utils import TextCleaner
    from dataloader_ctc import FilePathDataset, collate_fn
    from torch.utils.data import DataLoader
    import torch

    print("Initializing tokenizers...")
    text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
    text_cleaner = TextCleaner()
    
    print("\n" + "=" * 80)
    print("TEST 1: PHONEMIZATION PROCESS")
    print("=" * 80)
    
    sample_text = "Hello world! This is a test of the phonemization process."
    print(f"\nOriginal text: '{sample_text}'")
    
    result = phonemize(sample_text, text_tokenizer)
    print(f"\nTokens: {len(result['phonemes'])} words/symbols")
    
    for i, (phoneme, bpe_ids) in enumerate(zip(result['phonemes'], result['bpe_ids']), 1):
        phon_ids = text_cleaner.encode(phoneme)
        bpe_tokens = [text_tokenizer.tokenizer.decode([id]) for id in bpe_ids]
        
        print(f"\n[{i}] Phoneme: '{phoneme}'")
        print(f"    Phoneme IDs: {phon_ids} (len={len(phon_ids)})")
        print(f"    BPE tokens: {bpe_tokens}")
        print(f"    BPE IDs: {bpe_ids} (len={len(bpe_ids)})")
    
    print("\n" + "=" * 80)
    print("TEST 2: DATALOADER WITH MLM MASKING")
    print("=" * 80)
    
    class MockDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    test_text = "Hello world! How are you today?"
    samples = [phonemize(test_text, text_tokenizer)]
    
    mock_dataset = MockDataset(samples)
    file_path_dataset = FilePathDataset(dataset=mock_dataset, text_cleaner=text_cleaner)
    dataloader = DataLoader(
        file_path_dataset,
        batch_size=1,
        collate_fn=lambda batch: collate_fn(batch, text_cleaner, mlm_prob=0.15),
    )
    
    batch = next(iter(dataloader))
    
    print(f"\nBatch info:")
    print(f"  Phoneme input shape: {batch['phoneme_input'].shape}")
    print(f"  MLM labels shape: {batch['mlm_labels'].shape}")
    print(f"  CTC targets shape: {batch['ctc_targets'].shape}")
    print(f"  Input lengths: {batch['input_lengths'].tolist()}")
    print(f"  Target lengths: {batch['target_lengths'].tolist()}")
    
    input_ids = batch["phoneme_input"][0].tolist()
    mlm_labels = batch["mlm_labels"][0].tolist()
    input_len = batch["input_lengths"][0].item()
    
    mask_count = sum(1 for i in range(input_len) if input_ids[i] == text_cleaner.mask_id and mlm_labels[i] != -100)
    random_count = sum(1 for i in range(input_len) if mlm_labels[i] != -100 and input_ids[i] != text_cleaner.mask_id and input_ids[i] != mlm_labels[i])
    kept_count = sum(1 for i in range(input_len) if mlm_labels[i] != -100 and input_ids[i] == mlm_labels[i])
    total_masked = sum(1 for i in range(input_len) if mlm_labels[i] != -100)
    
    print(f"\nMLM masking statistics:")
    print(f"  Total positions: {input_len}")
    print(f"  Masked positions: {total_masked} ({100*total_masked/input_len:.1f}%)")
    print(f"    → [MASK] token: {mask_count} ({100*mask_count/total_masked:.1f}%)")
    print(f"    → Random: {random_count} ({100*random_count/total_masked:.1f}%)")
    print(f"    → Kept original: {kept_count} ({100*kept_count/total_masked:.1f}%)")
    
    print(f"\nMasking examples (showing only masked positions):")
    print(f"{'Pos':<5} {'Original':<10} {'Input':<10} {'Type':<20}")
    print("-" * 50)
    
    shown = 0
    for i in range(input_len):
        if mlm_labels[i] != -100 and shown < 10:
            label_char = [c for c, idx in text_cleaner.word_index_dictionary.items() if idx == mlm_labels[i]][0]
            input_char = [c for c, idx in text_cleaner.word_index_dictionary.items() if idx == input_ids[i]][0]
            
            if input_ids[i] == text_cleaner.mask_id:
                mask_type = "[MASK]"
            elif input_ids[i] == mlm_labels[i]:
                mask_type = "Kept original"
            else:
                mask_type = f"Random"
            
            print(f"{i:<5} {label_char:<10} {input_char:<10} {mask_type:<20}")
            shown += 1
    
    print("\n" + "=" * 80)
    print("Tests completed successfully!")
    print("=" * 80)
