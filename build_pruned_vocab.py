import json
from datasets import load_from_disk
from tqdm import tqdm
from text_tokenizer import TextTokenizer

def build_pruned_vocab(dataset_path="wiki_phoneme_final_v2"):
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    used_token_ids = set()
    
    print("Scanning dataset for used BPE tokens...")
    for ex in tqdm(dataset):
        for word_bpe in ex["bpe_ids"]:
            used_token_ids.update(word_bpe)
   
    tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")

    special_ids = {
        tokenizer.pad_id, 
        tokenizer.eos_id, 
        tokenizer.bos_id, 
        tokenizer.unk_id
    }

    special_ids = {sid for sid in special_ids if sid is not None}
    
    used_token_ids.update(special_ids)

    sorted_ids = sorted(list(used_token_ids))
    
    original_to_compact = {oid: i for i, oid in enumerate(sorted_ids)}
    compact_to_original = {i: oid for i, oid in enumerate(sorted_ids)}
    
    print(f"Original Vocab Size: {len(tokenizer)}")
    print(f"Pruned Vocab Size  : {len(sorted_ids)}")
    print(f"Reduction          : {100 - (len(sorted_ids)/len(tokenizer)*100):.2f}% removed")

    output_map = {
        "original_to_compact": original_to_compact,
        "compact_to_original": compact_to_original
    }
    
    output_file = f"{dataset_path}/bpe_vocab_map.json"
    with open(output_file, "w") as f:
        json.dump(output_map, f, indent=2)
        
    print(f"Saved mapping to {output_file}")