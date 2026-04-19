import json
import pickle
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

def build_pruned_vocab(dataset_path="wiki_phoneme_final_v2"):
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    used_token_ids = set()
    
    print("Scanning dataset for used BPE tokens...")
    for ex in tqdm(dataset):
        for word_bpe in ex["bpe_ids"]:
            used_token_ids.update(word_bpe)
   
    model_name = "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    special_ids = {
        tokenizer.pad_token_id, 
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id
    }

    special_ids = {sid for sid in special_ids if sid is not None}
    
    used_token_ids.update(special_ids)

    sorted_ids = sorted(list(used_token_ids))
    
    original_to_compact = {oid: i for i, oid in enumerate(sorted_ids)}
    compact_to_original = {i: oid for i, oid in enumerate(sorted_ids)}
    
    # Create token_maps as requested: {orig_id: {'token': compact_id, 'word': token_string}}
    token_maps = {}
    for oid in sorted_ids:
        token_maps[oid] = {
            'token': original_to_compact[oid],
            'word': tokenizer.convert_ids_to_tokens(oid)
        }

    vocab_size = len(tokenizer)
    print(f"Original Vocab Size: {vocab_size}")
    print(f"Pruned Vocab Size  : {len(sorted_ids)}")
    print(f"Reduction          : {100 - (len(sorted_ids)/vocab_size*100):.2f}% removed")

    output_map = {
        "original_to_compact": original_to_compact,
        "compact_to_original": compact_to_original
    }
    
    # Save the original mapping as JSON
    output_json = f"{dataset_path}/bpe_vocab_map.json"
    with open(output_json, "w") as f:
        json.dump(output_map, f, indent=2)
        
    # Save the token_maps as Pickle (as requested)
    output_pkl = "token_maps.pkl"
    with open(output_pkl, "wb") as f:
        pickle.dump(token_maps, f)
        
    print(f"Saved JSON mapping to {output_json}")
    print(f"Saved Pickle token maps to {output_pkl}")

if __name__ == "__main__":
    build_pruned_vocab("wikipedia-50")