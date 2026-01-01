import os
import json
import glob
from datasets import load_dataset, load_from_disk, concatenate_datasets

from text_tokenizer import TextTokenizer
from phoneme_tokenizer import PhonemeTokenizer
from phonemize import phonemize

text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
phoneme_tokenizer = PhonemeTokenizer()

parquet_folder = "wikipedia.id"
parquet_files = glob.glob(f"{parquet_folder}/*.parquet")

dataset = load_dataset("parquet", data_files=parquet_files)
dataset = dataset["train"] if "train" in dataset else list(dataset.values())[0]

root = "./wiki_phoneme"
num_shards = 500000
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

with ProcessPool(max_workers=20) as pool:
    pool.map(process_shard, range(num_shards), timeout=60)

shards = []
for s in os.listdir(root):
    d = f"{root}/{s}"
    try:
        shards.append(load_from_disk(d))
    except:
        pass

dataset = concatenate_datasets(shards)

print("Building phoneme vocabulary from processed data...")
for example in dataset:
    phoneme_tokenizer.build_from_sentence(example["phoneme"])

print(f"Total phonemes in vocab: {phoneme_tokenizer.vocab_size}")

phoneme_tokenizer.save(f"phoneme_vocab.json")
print("Saved phoneme vocab to", f"phoneme_vocab.json")

dataset.save_to_disk("wiki_phoneme_final")
print("Saved dataset to wiki_phoneme_final")