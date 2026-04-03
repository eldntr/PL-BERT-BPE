import os
import json
import glob
import yaml
from datasets import load_dataset, load_from_disk, concatenate_datasets

from transformers import AutoTokenizer
from text_utils import TextCleaner
from phonemize import phonemize
from build_pruned_vocab import build_pruned_vocab

# Load configuration
config_path = "Configs/config.yml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(config["dataset_params"]["tokenizer"])

phoneme_tokenizer = TextCleaner()

parquet_folder = "wikipedia.id"
parquet_files = glob.glob(f"{parquet_folder}/*.parquet")

dataset = load_dataset("parquet", data_files=parquet_files)
dataset = dataset["train"] if "train" in dataset else list(dataset.values())[0]

root = "./wiki_phoneme"
num_shards = 5000
os.makedirs(root, exist_ok=True)


def process_shard(idx):
    out_dir = f"{root}/shard_{idx}"
    if os.path.exists(out_dir):
        print(f"Shard {idx} exists.")
        return

    print(f"Processing shard {idx}")
    shard = dataset.shard(num_shards, idx)

    processed = shard.map(
        lambda ex: phonemize(ex["text"], tokenizer),
        remove_columns=["text"],
        load_from_cache_file=False
    )
    
    processed = processed.filter(lambda ex: len(ex["words"]) > 0)

    os.makedirs(out_dir, exist_ok=True)
    processed.save_to_disk(out_dir)

    # Optional untuk inpeksi :D
    json_records = []
    for ex in processed:
        json_records.append({
            "before": ex["before"],
            "after": ex["after"],
            "words": ex["words"],
            "phonemes": ex["phonemes"],
            "bpe_ids": ex["bpe_ids"],
        })

    with open(f"{out_dir}/phonemize.json", "w") as f:
        json.dump(json_records, f, indent=2, ensure_ascii=False)

from pebble import ProcessPool

with ProcessPool(max_workers=28) as pool:
    pool.map(process_shard, range(num_shards), timeout=600)

shards = []
for s in os.listdir(root):
    d = f"{root}/{s}"
    try:
        shards.append(load_from_disk(d))
    except:
        pass

dataset = concatenate_datasets(shards)

dataset = dataset.filter(lambda ex: len(ex["words"]) > 0)

# Symbols are fixed in text_utils.TextCleaner, so no need to build vocabulary or save it.

dataset_output_dir = config["data_folder"]
dataset.save_to_disk(dataset_output_dir)
print(f"Saved dataset to {dataset_output_dir}")

train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

train_dir = f"{dataset_output_dir}/train"
test_dir = f"{dataset_output_dir}/test"

train_dataset.save_to_disk(train_dir)
test_dataset.save_to_disk(test_dir)
print(f"Saved training set ({len(train_dataset)} examples) to {train_dir}")
print(f"Saved test set ({len(test_dataset)} examples) to {test_dir}")

# phoneme_tokenizer.save(f"{dataset_output_dir}/phoneme_vocab.json")
# print(f"Saved phoneme vocab to {dataset_output_dir}/phoneme_vocab.json")

build_pruned_vocab(dataset_path=dataset_output_dir)
print(f"Built and saved pruned BPE vocab map to {dataset_output_dir}/bpe_vocab_map.json")