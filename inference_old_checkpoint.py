import torch
from phonemize import phonemize
from phoneme_tokenizer import PhonemeTokenizer
from text_tokenizer import TextTokenizer
from model import MultiTaskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load checkpoint first to get the old vocab size
ckpt = torch.load("checkpoint_step_100000.pt", map_location=device)

# Extract vocab size from checkpoint
old_phoneme_vocab_size = ckpt["model_state"]["encoder.embeddings.word_embeddings.weight"].shape[0]
print(f"Checkpoint phoneme vocab size: {old_phoneme_vocab_size}")

# 2) Load tokenizers
text_tokenizer = TextTokenizer(
    "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct",
    map_file="bpe_vocab_map.json",
)

# Load phoneme tokenizer - but we'll use old vocab size
phoneme_tokenizer = PhonemeTokenizer.load("./wiki_phoneme/phoneme_vocab.json")
print(f"Current phoneme vocab size: {phoneme_tokenizer.vocab_size}")

# 3) Create model with OLD vocab size from checkpoint
model = MultiTaskModel(
    phoneme_vocab_size=old_phoneme_vocab_size,  # Use old size!
    bpe_vocab_size=len(text_tokenizer),
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    intermediate_size=2048,
    max_position_embeddings=512,
).to(device)

# Now load should work
model.load_state_dict(ckpt["model_state"])
model.eval()
print("✓ Model loaded successfully!")

# 4) Prepare a sample (using OLD phonemization - without special tokens)
text = "Halo dunia"  # Note: no punctuation, old system doesn't handle it well

# Manual phonemization without special tokens (old way)
import re
from phonemize import phonemize_word_espeak

normalized = text.lower()
words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", normalized)

flat_phon = []
for w in words:
    # BPE
    bpe_ids = text_tokenizer.encode_word(w)
    if len(bpe_ids) == 0:
        continue
    
    # PHONEME
    phon_str = phonemize_word_espeak(w, ipa=True, keep_stress=False, sep=" ")
    phon_ids = phoneme_tokenizer.encode(phon_str)
    flat_phon.extend(phon_ids)

phoneme_input = torch.tensor(flat_phon, dtype=torch.long, device=device).unsqueeze(0)
attention_mask = (phoneme_input != phoneme_tokenizer.pad_id).long()

print(f"\nInput: '{text}'")
print(f"Phoneme IDs: {flat_phon}")

with torch.no_grad():
    mlm_logits, ctc_logits = model(phoneme_input, attention_mask=attention_mask)

    # Greedy CTC decode (blank=0, real BPE ids shifted by +1 in training)
    log_probs = torch.log_softmax(ctc_logits, dim=-1)      # [B, T, C]
    path = log_probs.argmax(-1)[0].tolist()                # best path
    bpe_path = []
    prev = None
    for p in path:
        if p == 0:           # blank
            prev = None
            continue
        if p != prev:        # collapse repeats
            bpe_path.append(p - 1)  # shift back to original BPE id
        prev = p

    decoded_text = text_tokenizer.decode(bpe_path)

print(f"CTC decode: '{decoded_text}'")
print(f"\n⚠️  Note: This checkpoint was trained WITHOUT special tokens (<s>, </s>, <space>)")
print("⚠️  For best results, retrain with the new phonemization pipeline!")
