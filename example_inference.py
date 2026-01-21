import torch
import random
from model import MultiTaskModel
from text_tokenizer import TextTokenizer
from phoneme_tokenizer import PhonemeTokenizer
from phonemize import phonemize

def apply_mlm_masking(
    input_ids: torch.Tensor,
    pad_id: int,
    mask_id: int,
    vocab_size: int,
    mlm_prob: float = 0.15,
    rng: random.Random = None,
):
    """
    Standard BERT-style MLM masking:
    - pilih token (bukan PAD) dengan probabilitas mlm_prob
    - 80% -> ganti [MASK]
    - 10% -> ganti token random
    - 10% -> tetap token asli
    Return:
      masked_input_ids, labels
    labels = -100 untuk token yang tidak dimask (agar ignore loss)
    """
    if rng is None:
        rng = random.Random()

    device = input_ids.device
    labels = input_ids.clone()

    # kandidat: bukan PAD
    can_mask = (input_ids != pad_id)

    # sampling mask positions
    probs = torch.full(input_ids.shape, mlm_prob, device=device)
    probs = probs * can_mask.float()
    mask_positions = (torch.rand(input_ids.shape, device=device) < probs) & can_mask

    # labels: hanya posisi mask yang dilatih
    labels[~mask_positions] = -100

    # apply replacement rule
    masked_input_ids = input_ids.clone()

    # 80% [MASK]
    mask_choice = torch.rand(input_ids.shape, device=device)
    to_mask = mask_positions & (mask_choice < 0.8)
    masked_input_ids[to_mask] = mask_id

    # 10% random token
    to_rand = mask_positions & (mask_choice >= 0.8) & (mask_choice < 0.9)
    random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape, device=device)
    masked_input_ids[to_rand] = random_tokens[to_rand]

    # 10% keep original -> do nothing

    return masked_input_ids, labels


def apply_span_masking(
    input_ids: torch.Tensor,
    pad_id: int,
    mask_id: int,
    vocab_size: int,
    mlm_prob: float = 0.15,
    mean_span_len: int = 3,
    rng: random.Random = None,
):
    """
    Span masking sederhana:
    - target rasio token termask ~ mlm_prob
    - ambil beberapa span acak dengan panjang rata-rata mean_span_len
    """
    if rng is None:
        rng = random.Random()

    ids = input_ids.clone()
    labels = input_ids.clone()

    B, L = ids.shape
    device = ids.device

    # valid positions (non-pad)
    valid = (ids != pad_id)
    valid_lens = valid.sum(dim=1).tolist()

    # init labels ignore
    labels[:] = -100

    for b in range(B):
        n_valid = valid_lens[b]
        if n_valid <= 0:
            continue

        target_to_mask = max(1, int(n_valid * mlm_prob))
        masked = 0

        while masked < target_to_mask:
            # start idx di area valid
            start = rng.randint(0, n_valid - 1)

            # panjang span: geometric-ish sederhana
            span_len = max(1, int(rng.expovariate(1.0 / mean_span_len)))
            end = min(n_valid, start + span_len)

            # mask span
            for i in range(start, end):
                if ids[b, i].item() == pad_id:
                    continue
                if labels[b, i].item() != -100:
                    continue  # sudah termask

                labels[b, i] = ids[b, i]
                ids[b, i] = mask_id
                masked += 1

                if masked >= target_to_mask:
                    break

    return ids, labels


# ==================== END MLM MASKING ====================

def load_model(checkpoint_path, phoneme_vocab_size, bpe_vocab_size, device="cpu"):
    """Load model from checkpoint with vocab size detection."""
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
        print(f"Original IDs:    {original_ids}")  # Debug first 20
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


def predict_with_mlm_masking(
    model,
    text,
    phoneme_tokenizer,
    text_tokenizer,
    device,
    masking_mode="random",
    mlm_prob=0.15,
):
    """
    Inference dengan MLM masking untuk debugging/evaluasi.
    masking_mode: "random" atau "span"
    """
    print(f"\n=== MLM MASKING INFERENCE ===")
    print(f"input:   {text}")
    print(f"masking_mode: {masking_mode}, mlm_prob: {mlm_prob}")

    ex = phonemize(text, text_tokenizer, phoneme_tokenizer)
    print(f"normalized: {ex['after']}")
    print(f"phonemes:   {ex['phonemes']}")

    flat_phon = []
    for ph in ex["phonemes"]:
        ids = phoneme_tokenizer.encode(ph)
        flat_phon.extend(ids)

    print(f"input ids:  {flat_phon}")

    if not flat_phon:
        return ""

    input_tensor = torch.tensor([flat_phon], dtype=torch.long).to(device)
    attention_mask = (input_tensor != phoneme_tokenizer.pad_id).long()

    # Verify mask_id exists
    if getattr(phoneme_tokenizer, "mask_id", None) is None:
        print("❌ phoneme_tokenizer.mask_id not found!")
        print("   Ensure phoneme_vocab.json has <mask> token")
        return ""

    # Apply masking
    if masking_mode == "random":
        masked_input, mlm_labels = apply_mlm_masking(
            input_ids=input_tensor,
            pad_id=phoneme_tokenizer.pad_id,
            mask_id=phoneme_tokenizer.mask_id,
            vocab_size=phoneme_tokenizer.vocab_size,
            mlm_prob=mlm_prob,
        )
    elif masking_mode == "span":
        masked_input, mlm_labels = apply_span_masking(
            input_ids=input_tensor,
            pad_id=phoneme_tokenizer.pad_id,
            mask_id=phoneme_tokenizer.mask_id,
            vocab_size=phoneme_tokenizer.vocab_size,
            mlm_prob=mlm_prob,
            mean_span_len=3,
        )
    else:
        raise ValueError(f"Unknown masking_mode: {masking_mode}")

    print(f"masked ids:    {masked_input[0].tolist()}")
    print(f"mlm labels:    {mlm_labels[0].tolist()} (-100 = ignore)")

    # Inference
    with torch.no_grad():
        mlm_logits, ctc_logits = model(masked_input, attention_mask=attention_mask)

        # MLM evaluation di posisi termask
        masked_positions = (mlm_labels != -100)
        if masked_positions.any():
            pred_ids = torch.argmax(mlm_logits, dim=-1)  # [B, L]
            true_ids = mlm_labels[masked_positions]
            pred_masked = pred_ids[masked_positions]
            acc = (pred_masked == true_ids).float().mean().item()
            print(f"MLM masked-token accuracy: {acc:.4f}")

            # Show some examples
            num_show = min(5, masked_positions.sum().item())
            print(f"Sample predictions (first {num_show}):")
            for i in range(num_show):
                idx = torch.where(masked_positions)[1][i].item()
                true_val = mlm_labels[0, idx].item()
                pred_val = pred_ids[0, idx].item()
                print(f"  pos {idx}: true={true_val}, pred={pred_val}, match={true_val == pred_val}")

    # Decode CTC like normal
    ctc_text = ctc_decode(ctc_logits, text_tokenizer)
    print(f"CTC output: {ctc_text}")
    
    return ctc_text

if __name__ == "__main__":
    # Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT = "checkpoint_step_600000.pt"
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

    # ===== Mode 1: Normal CTC inference (no masking) =====
    print("\n" + "="*80)
    print("MODE 1: Normal CTC Inference")
    print("="*80)
    text = "aku remen mangan sega goreng ing omah"
    output = predict(model, text, phoneme_tokenizer, text_tokenizer, DEVICE)
    print(f"output: {output}")

    # ===== Mode 2: MLM with random masking =====
    print("\n" + "="*80)
    print("MODE 2: MLM with Random Masking")
    print("="*80)
    text = "Kulo mangan sega"
    output_mlm_random = predict_with_mlm_masking(
        model, text, phoneme_tokenizer, text_tokenizer, DEVICE,
        masking_mode="random",
        mlm_prob=0.15
    )

    # ===== Mode 3: MLM with span masking =====
    print("\n" + "="*80)
    print("MODE 3: MLM with Span Masking")
    print("="*80)
    output_mlm_span = predict_with_mlm_masking(
        model, text, phoneme_tokenizer, text_tokenizer, DEVICE,
        masking_mode="span",
        mlm_prob=0.15
    )