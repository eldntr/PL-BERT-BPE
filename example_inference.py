import torch
from model import MultiTaskModel
from text_tokenizer import TextTokenizer
from phoneme_tokenizer import PhonemeTokenizer
from phonemize import phonemize

def load_model(checkpoint_path, phoneme_vocab_size, bpe_vocab_size, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"]
    
    # Handle DDP 'module.' prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Detect vocab size from checkpoint
    ckpt_phoneme_size = new_state_dict["encoder.embeddings.word_embeddings.weight"].shape[0]
    ckpt_bpe_size = new_state_dict["ctc_head.weight"].shape[0] - 1  # minus blank
    
    print(f"⚠️  Checkpoint vocab: phoneme={ckpt_phoneme_size}, bpe={ckpt_bpe_size}")
    print(f"⚠️  Current vocab:    phoneme={phoneme_vocab_size}, bpe={bpe_vocab_size}")
    
    if ckpt_phoneme_size != phoneme_vocab_size:
        print("❌ PHONEME VOCAB MISMATCH! Model outputs will be garbage.")
        print("   → Retrain model with new phoneme vocab (includes <s>, </s>, <space>)")
        
    if ckpt_bpe_size != bpe_vocab_size:
        print("❌ BPE VOCAB MISMATCH!")
        print("   → Ensure bpe_vocab_map.json matches training checkpoint")
    
    model = MultiTaskModel(
        phoneme_vocab_size=ckpt_phoneme_size,  # Use checkpoint size
        bpe_vocab_size=ckpt_bpe_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
        max_position_embeddings=1024,
    )
            
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
            decoded_ids.append(idx - 1)  # Shift back from CTC (blank=0, tokens=1+)
        prev_idx = idx

    print(f"CTC raw indices: {preds}")
    print(f"CTC compact IDs: {decoded_ids}")
    
    # Decode compact→original if pruning is used
    if text_tokenizer.use_pruning:
        original_ids = [text_tokenizer.compact_to_original.get(cid, text_tokenizer.unk_id) for cid in decoded_ids]
        print(f"Original IDs:    {original_ids[:20]}...")  # Debug first 20
        return text_tokenizer.tokenizer.decode(original_ids, skip_special_tokens=False)
    else:
        return text_tokenizer.tokenizer.decode(decoded_ids, skip_special_tokens=False)

def predict(model, text, phoneme_tokenizer, text_tokenizer, device):
    print(f"input:   {text}")

    # Phonemize dengan BOS/EOS, spaces, dan punctuation (SAMA SEPERTI TRAINING)
    ex = phonemize(text, text_tokenizer, phoneme_tokenizer)
    print(f"normalized: {ex['after']}")
    print(f"phonemes:   {ex['phonemes']}")

    # Tokenize: encode setiap phoneme item, lalu flatten (SAMA SEPERTI TRAINING)
    flat_phon = []
    for ph in ex["phonemes"]:
        ids = phoneme_tokenizer.encode(ph)
        flat_phon.extend(ids)

    print(f"input ids:  {flat_phon}")

    if not flat_phon:
        return ""
    
    # Filter IDs that exceed vocab size (if old checkpoint)
    model_vocab_size = 104  # Current vocab size with BOS/EOS/space
    filtered_ids = [i if i < model_vocab_size else phoneme_tokenizer.blank_id for i in flat_phon]
    if filtered_ids != flat_phon:
        print(f"⚠️  Filtered {len([i for i in flat_phon if i >= model_vocab_size])} OOV phonemes (using blank)")
    
    input_tensor = torch.tensor([filtered_ids], dtype=torch.long).to(device)
    attention_mask = (input_tensor != phoneme_tokenizer.pad_id).long()

    # Inference
    with torch.no_grad():
        _, ctc_logits = model(input_tensor, attention_mask=attention_mask)

    # Decode
    return ctc_decode(ctc_logits, text_tokenizer)

if __name__ == "__main__":
    # Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT = "checkpoint_step_100000.pt"
    PHONEME_VOCAB = "wikipedia-50/phoneme_vocab.json"
    TEXT_TOKENIZER = "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"

    text_tokenizer = TextTokenizer(TEXT_TOKENIZER, map_file="wikipedia-50/bpe_vocab_map.json")
    phoneme_tokenizer = PhonemeTokenizer.load(PHONEME_VOCAB)
    
    model = load_model(
        CHECKPOINT, 
        phoneme_tokenizer.vocab_size, 
        len(text_tokenizer), 
        device=DEVICE
    )

    text = "saya, makan nasi goreng!"
    output = predict(model, text, phoneme_tokenizer, text_tokenizer, DEVICE)
    
    print(f"output: {output}")