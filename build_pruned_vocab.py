import json
import torch
from datasets import load_from_disk
from tqdm import tqdm
from text_tokenizer import TextTokenizer

def build_pruned_vocab():
    # 1. Load Dataset yang sudah diproses
    dataset_path = "wiki_phoneme_final"
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # 2. Hitung frekuensi token
    # Kita menggunakan set untuk menyimpan unique token ID yang muncul
    used_token_ids = set()
    
    print("Scanning dataset for used BPE tokens...")
    # Iterasi dataset untuk mengumpulkan semua token ID yang muncul di 'bpe_ids'
    for ex in tqdm(dataset):
        # bpe_ids adalah list of list (karena per kata), jadi kita flatten
        for word_bpe in ex["bpe_ids"]:
            used_token_ids.update(word_bpe)
            
    # 3. Load Tokenizer Asli untuk referensi special tokens
    tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
    
    # Pastikan special tokens masuk ke dalam used_ids agar tidak hilang
    # (PAD, EOS, BOS, UNK)
    special_ids = {
        tokenizer.pad_id, 
        tokenizer.eos_id, 
        tokenizer.bos_id, 
        tokenizer.unk_id
    }
    # Filter None (jika ada special token yang tidak didefinisikan)
    special_ids = {sid for sid in special_ids if sid is not None}
    
    used_token_ids.update(special_ids)
    
    # 4. Buat Mapping: Original ID -> Compact ID
    # Sort agar urutannya deterministik
    sorted_ids = sorted(list(used_token_ids))
    
    original_to_compact = {oid: i for i, oid in enumerate(sorted_ids)}
    compact_to_original = {i: oid for i, oid in enumerate(sorted_ids)}
    
    print(f"Original Vocab Size: {len(tokenizer)}")
    print(f"Pruned Vocab Size  : {len(sorted_ids)}")
    print(f"Reduction          : {100 - (len(sorted_ids)/len(tokenizer)*100):.2f}% removed")
    
    # 5. Simpan Mapping
    output_map = {
        "original_to_compact": original_to_compact,
        "compact_to_original": compact_to_original
    }
    
    with open("bpe_vocab_map.json", "w") as f:
        json.dump(output_map, f, indent=2)
        
    print("Saved mapping to bpe_vocab_map.json")

if __name__ == "__main__":
    build_pruned_vocab()