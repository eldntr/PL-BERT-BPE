import os
import glob
from datasets import load_dataset, load_from_disk, concatenate_datasets

from text_tokenizer import TextTokenizer
from phonemize import phonemize

text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")

parquet_folder = "wikipedia.id"
parquet_files = glob.glob(f"{parquet_folder}/*.parquet")

dataset = load_dataset("parquet", data_files=parquet_files)
dataset = dataset["train"] if "train" in dataset else list(dataset.values())[0]

root = "./wiki_phoneme"
num_shards = 50000
os.makedirs(root, exist_ok=True)


def process_shard(idx):
    out_dir = f"{root}/shard_{idx}"
    if os.path.exists(out_dir):
        print(f"Shard {idx} exists.")
        return

    print(f"Processing shard {idx}")
    shard = dataset.shard(num_shards, idx)

    processed = shard.map(
        lambda ex: phonemize(ex["text"], text_tokenizer),
        remove_columns=["text"],
        load_from_cache_file=False
    )

    os.makedirs(out_dir, exist_ok=True)
    processed.save_to_disk(out_dir)


from pebble import ProcessPool

with ProcessPool(max_workers=32) as pool:
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

print(f"Total dataset size: {len(dataset)}")

train_test_split = dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = train_test_split["train"]
val_test_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)
val_dataset = val_test_split["train"]
test_dataset = val_test_split["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

train_dataset.save_to_disk("wiki_phoneme_train")
val_dataset.save_to_disk("wiki_phoneme_val")
test_dataset.save_to_disk("wiki_phoneme_test")

print("Saved train dataset to wiki_phoneme_train")
print("Saved validation dataset to wiki_phoneme_val")
print("Saved test dataset to wiki_phoneme_test")

dataset.save_to_disk("wiki_phoneme_final")
print("Saved full dataset to wiki_phoneme_final")