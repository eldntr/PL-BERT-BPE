import os
import json
import glob
from datasets import load_dataset, load_from_disk, concatenate_datasets

from text_tokenizer import TextTokenizer
from phoneme_tokenizer import PhonemeTokenizer
from phonemize import phonemize

text_tokenizer = TextTokenizer("GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")
phoneme_tokenizer = PhonemeTokenizer()


def split_at_nearest_period(text, max_words=50):
    """Split text at nearest period if word count exceeds max_words.
    Returns None if no period found and text exceeds max_words."""
    words = text.split()
    
    if len(words) <= max_words:
        return text

    best_split = None
    min_distance = float('inf')
    
    for i, word in enumerate(words):
        if '.' in word:
            distance = abs(i + 1 - max_words)
            if distance < min_distance:
                min_distance = distance
                best_split = i + 1

    if best_split is None:
        return None

    return ' '.join(words[:best_split])

def process_example(ex):
    truncated = split_at_nearest_period(ex["text"], max_words=50)
    if truncated is None:
        return None
    return phonemize(truncated, text_tokenizer)

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
        process_example,
        remove_columns=["text"],
        load_from_cache_file=False
    )

    processed = processed.filter(lambda ex: ex["bpe_ids"] is not None)   

    os.makedirs(out_dir, exist_ok=True)
    processed.save_to_disk(out_dir)


from pebble import ProcessPool

with ProcessPool(max_workers=32) as pool:
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
    phoneme_tokenizer.build_from_sentence(example["phonemes"])

print(f"Total phonemes in vocab: {phoneme_tokenizer.vocab_size}")

phoneme_tokenizer.save(f"phoneme_vocab.json")
print("Saved phoneme vocab to", f"phoneme_vocab.json")

print("Encoding phonemes to IDs...")
def add_phoneme_ids(ex):
    ex["phoneme_ids"] = phoneme_tokenizer.encode(ex["phonemes"])
    return ex

dataset = dataset.map(add_phoneme_ids, load_from_cache_file=False)

dataset.save_to_disk("wiki_phoneme_final")
print("Saved dataset to wiki_phoneme_final")