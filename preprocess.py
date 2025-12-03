import os
import json
import glob
from datasets import load_dataset, load_from_disk, concatenate_datasets

from text_tokenizer import TextTokenizer
from phoneme_tokenizer import PhonemeTokenizer
from phonemize import phonemize

# -----------------------------------------------------------------
# Load tokenizer
# -----------------------------------------------------------------
text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")

# phoneme tokenizer (empty, vocab will grow during processing)
phoneme_tokenizer = PhonemeTokenizer()

# -----------------------------------------------------------------
# Load parquet dataset
# -----------------------------------------------------------------
parquet_folder = "wikipedia.id"
parquet_files = glob.glob(f"{parquet_folder}/*.parquet")

dataset = load_dataset("parquet", data_files=parquet_files)
dataset = dataset["train"] if "train" in dataset else list(dataset.values())[0]

# -----------------------------------------------------------------
# Process shards
# -----------------------------------------------------------------
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
        lambda ex: phonemize(ex["text"], text_tokenizer, phoneme_tokenizer),
        remove_columns=["text"],
        load_from_cache_file=False
    )
    
    # Filter out examples with more than 50 words (empty words list)
    processed = processed.filter(lambda ex: len(ex["words"]) > 0)

    os.makedirs(out_dir, exist_ok=True)
    processed.save_to_disk(out_dir)

    # also save simple JSON for inspection
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


# -----------------------------------------------------------------
# Multiprocessing
# -----------------------------------------------------------------
from pebble import ProcessPool

with ProcessPool(max_workers=32) as pool:
    pool.map(process_shard, range(num_shards), timeout=60)

# -----------------------------------------------------------------
# Merge the shards
# -----------------------------------------------------------------
shards = []
for s in os.listdir(root):
    d = f"{root}/{s}"
    try:
        shards.append(load_from_disk(d))
    except:
        pass

dataset = concatenate_datasets(shards)

# Final filter: only keep examples with words (not more than 50)
dataset = dataset.filter(lambda ex: len(ex["words"]) > 0)

# -----------------------------------------------------------------
# Build phoneme vocab from processed data
# -----------------------------------------------------------------
print("Building phoneme vocabulary from processed data...")
for example in dataset:
    for phoneme_str in example["phonemes"]:
        phoneme_tokenizer.build_from_sentence(phoneme_str)

print(f"Total phonemes in vocab: {phoneme_tokenizer.vocab_size}")

# -----------------------------------------------------------------
# Save phoneme vocab
# -----------------------------------------------------------------
phoneme_tokenizer.save(f"phoneme_vocab.json")
print("Saved phoneme vocab to", f"phoneme_vocab.json")

# Note: phoneme_ids will be generated on-the-fly in the dataloader
# This prevents ID collision issues from multiprocessing

# -----------------------------------------------------------------
# Split dataset: Train / Validation / Test
# -----------------------------------------------------------------
print(f"Total dataset size: {len(dataset)}")

# Split: 98% train, 1% validation, 1% test
train_test_split = dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = train_test_split["train"]
val_test_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)
val_dataset = val_test_split["train"]
test_dataset = val_test_split["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# -----------------------------------------------------------------
# Save dataset splits
# -----------------------------------------------------------------
train_dataset.save_to_disk("wiki_phoneme_train")
val_dataset.save_to_disk("wiki_phoneme_val")
test_dataset.save_to_disk("wiki_phoneme_test")

print("Saved train dataset to wiki_phoneme_train")
print("Saved validation dataset to wiki_phoneme_val")
print("Saved test dataset to wiki_phoneme_test")

# Also save the full dataset for backward compatibility
dataset.save_to_disk("wiki_phoneme_final")
print("Saved full dataset to wiki_phoneme_final")