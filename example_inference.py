import torch
import random
from model import MultiTaskModel
from text_utils import TextCleaner
from phonemize import phonemize
import pickle
import yaml

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

def load_model(checkpoint_path, phoneme_vocab_size, bpe_vocab_size, model_params, device="cpu"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"]
    
    # Handle DDP 'module.' prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model = MultiTaskModel(
        phoneme_vocab_size=phoneme_vocab_size,
        bpe_vocab_size=bpe_vocab_size,
        hidden_size=model_params["hidden_size"],
        num_layers=model_params["num_hidden_layers"],
        num_heads=model_params["num_attention_heads"],
        intermediate_size=model_params["intermediate_size"],
        max_position_embeddings=model_params["max_position_embeddings"],
    )
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def ctc_decode(logits, llama_tokenizer, id_to_original):
    preds = torch.argmax(logits, dim=-1)[0].tolist()
    
    decoded_ids = []
    prev_idx = -1
    
    # Collapse repeats and remove blanks (index 0)
    for idx in preds:
        if idx != prev_idx and idx != 0:
            compact_id = idx - 1  # Shift back from CTC (blank=0, tokens=1+)
            # Map compact -> original BPE ID
            original_id = id_to_original.get(compact_id, compact_id)
            decoded_ids.append(original_id)
        prev_idx = idx

    print(f"CTC compact IDs: {decoded_ids[:20]}...")
    
    return llama_tokenizer.decode(decoded_ids, skip_special_tokens=True)

def predict(model, text, text_cleaner, llama_tokenizer, id_to_original, device):
    print(f"input:   {text}")

    # Normalize/Phonemize
    # Note: text_cleaner will be used on the phonemized output
    # For now, let's assume phonemize(text) returns a dict with 'phonemes' list of phonemes strings
    # But text_cleaner takes the whole string and returns list of indices.
    # Looking at dataloader_ctc.py: 
    # phon_ids_per_word = [self.text_cleaner(p) for p in phoneme_words]
    # flat_phon = [self.sos_phon] ... flat_phon.extend(phon_ids) ... flat_phon.append(self.eos_phon)
    
    ex = phonemize(text, llama_tokenizer)

    print(f"normalized: {ex['after']}")
    print(f"phonemes:   {ex['phonemes']}")

    sos_phon = text_cleaner.word_index_dictionary.get('<sos>', 1)
    eos_phon = text_cleaner.word_index_dictionary.get('<eos>', 2)
    pad_id = text_cleaner.word_index_dictionary.get('<pad>', 0)

    flat_phon = [sos_phon]
    for p in ex["phonemes"]:
        ids = text_cleaner(p)
        flat_phon.extend(ids)
    flat_phon.append(eos_phon)

    print(f"input ids:  {flat_phon}")

    if not flat_phon:
        return ""
    
    input_tensor = torch.tensor([flat_phon], dtype=torch.long).to(device)
    attention_mask = (input_tensor != pad_id).long()

    # Inference
    with torch.no_grad():
        _, ctc_logits = model(input_tensor, attention_mask=attention_mask)

    # Decode
    return ctc_decode(ctc_logits, llama_tokenizer, id_to_original)


def predict_with_mlm_masking(
    model,
    text,
    text_cleaner,
    llama_tokenizer,
    id_to_original,
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

    ex = phonemize(text, llama_tokenizer, None)
    print(f"normalized: {ex['after']}")
    print(f"phonemes:   {ex['phonemes']}")

    sos_phon = text_cleaner.word_index_dictionary.get('<sos>', 1)
    eos_phon = text_cleaner.word_index_dictionary.get('<eos>', 2)
    pad_id = text_cleaner.word_index_dictionary.get('<pad>', 0)
    mask_id = text_cleaner.word_index_dictionary.get('<mask>', 4)
    vocab_size = len(text_cleaner.word_index_dictionary)

    flat_phon = [sos_phon]
    for ph in ex["phonemes"]:
        ids = text_cleaner(ph)
        flat_phon.extend(ids)
    flat_phon.append(eos_phon)

    print(f"input ids:  {flat_phon}")

    if not flat_phon:
        return ""

    input_tensor = torch.tensor([flat_phon], dtype=torch.long).to(device)
    attention_mask = (input_tensor != pad_id).long()

    # Apply masking
    if masking_mode == "random":
        masked_input, mlm_labels = apply_mlm_masking(
            input_ids=input_tensor,
            pad_id=pad_id,
            mask_id=mask_id,
            vocab_size=vocab_size,
            mlm_prob=mlm_prob,
        )
    elif masking_mode == "span":
        masked_input, mlm_labels = apply_span_masking(
            input_ids=input_tensor,
            pad_id=pad_id,
            mask_id=mask_id,
            vocab_size=vocab_size,
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
    return ctc_decode(ctc_logits, llama_tokenizer, id_to_original)

if __name__ == "__main__":
    # Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT = "model/checkpoint_step_1000000_final.t7"
    CONFIG_PATH = "model/config.yml"
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    
    TOKEN_MAPS = dataset_params.get("token_maps", "token_maps.pkl")
    TEXT_TOKENIZER = dataset_params.get("tokenizer", "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")

    from transformers import AutoTokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(TEXT_TOKENIZER)

    # Load Token Maps
    with open(TOKEN_MAPS, 'rb') as f:
        token_maps = pickle.load(f)
    
    # Reverse map: compact_id -> original_id
    id_to_original = {v['token']: k for k, v in token_maps.items()}
    
    text_cleaner = TextCleaner()
    phoneme_vocab_size = len(text_cleaner.word_index_dictionary)
    bpe_vocab_size = len(token_maps)

    print(f"Phoneme vocab size: {phoneme_vocab_size}")
    print(f"BPE vocab size: {bpe_vocab_size}")

    model = load_model(
        CHECKPOINT, 
        phoneme_vocab_size, 
        bpe_vocab_size,
        model_params,
        device=DEVICE
    )

    # ===== Mode 1: Normal CTC inference (no masking) =====
    print("\n" + "="*80)
    print("MODE 1: Normal CTC Inference")
    print("="*80)
    text = "Burung-burung itu berkicau 'cuitt-cuitt' di dahan pohon yang rindang setiap pagi."
    output = predict(model, text, text_cleaner, llama_tokenizer, id_to_original, DEVICE)
    print(f"\nFinal output: {output}")

  