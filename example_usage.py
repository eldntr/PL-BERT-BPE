import torch
import re
from model import MultiTaskModel
from text_tokenizer import TextTokenizer
from phoneme_tokenizer import PhonemeTokenizer
from text_normalize import normalize_text
from phonemize import phonemize_word_espeak

def load_model(checkpoint_path, phoneme_vocab_size, bpe_vocab_size, device="cpu"):
    model = MultiTaskModel(
        phoneme_vocab_size=phoneme_vocab_size,
        bpe_vocab_size=bpe_vocab_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"]
    
    # Handle DDP 'module.' prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def ctc_decode(logits, text_tokenizer):
    preds = torch.argmax(logits, dim=-1)[0].tolist()
    
    decoded_ids = []
    prev_idx = -1
    
    # Collapse repeats and remove blanks (index 0)
    for idx in preds:
        if idx != prev_idx and idx != 0:
            decoded_ids.append(idx - 1) # Shift to 0-index
        prev_idx = idx

    print(f"CTC raw indices: {preds}")
    print(f"CTC BPE ids:     {decoded_ids}")

    if hasattr(text_tokenizer, "tokenizer"):
        return text_tokenizer.tokenizer.decode(decoded_ids)
    return text_tokenizer.decode(decoded_ids)

def predict(model, text, phoneme_tokenizer, text_tokenizer, device):
    print(f"input:   {text}")

    # Normalize & Phonemize
    normalized = normalize_text(text)
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", normalized)
    
    phoneme_list = [
        phonemize_word_espeak(w, ipa=True, keep_stress=False, sep=" ") 
        for w in words
    ]
    phoneme_str = " ".join(phoneme_list)

    print(f"normalized: {normalized}")
    print(f"phonemes:   [{phoneme_str}]")

    # Tokenize
    input_ids = phoneme_tokenizer.encode(phoneme_str)
    print(f"input ids:  {input_ids}")

    if not input_ids: return ""
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Inference
    with torch.no_grad():
        _, ctc_logits = model(input_tensor)

    # Decode
    return ctc_decode(ctc_logits, text_tokenizer)

if __name__ == "__main__":
    # Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT = "checkpoint_step_100000.pt"
    PHONEME_VOCAB = "./wiki_phoneme/phoneme_vocab.json"
    TEXT_TOKENIZER = "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"

    text_tokenizer = TextTokenizer(TEXT_TOKENIZER)
    phoneme_tokenizer = PhonemeTokenizer.load(PHONEME_VOCAB)
    
    model = load_model(
        CHECKPOINT, 
        phoneme_tokenizer.vocab_size, 
        len(text_tokenizer), 
        device=DEVICE
    )

    text = "Saya makan nasi goreng"
    output = predict(model, text, phoneme_tokenizer, text_tokenizer, DEVICE)
    
    print(f"output: {output}")