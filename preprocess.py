import os
import glob
import yaml
from datasets import load_dataset, load_from_disk, concatenate_datasets
from pebble import ProcessPool

from text_tokenizer import TextTokenizer
from phonemize import phonemize

# =========================
# CONFIG
# =========================
config_path = "Configs/config.yml"
config = yaml.safe_load(open(config_path))

PARQUET_FOLDER = "wikipedia.id"
OUT_ROOT = "./wiki_phoneme"
NUM_SHARDS = 5000

# aman untuk LLaMA (jauh < 2048 token)
MAX_TEXT_CHARS = 1500
MAX_PHONEME_LEN = 1800

os.makedirs(OUT_ROOT, exist_ok=True)

# =========================
# TOKENIZER (LLaMA / BPE)
# =========================
tokenizer = TextTokenizer(
    model_name=config["dataset_params"]["tokenizer"],
    use_fast=True
)

# =========================
# HELPERS
# =========================

def chunk_text_by_char(text, max_chars=1500):
    """Chunk text mentah supaya tokenizer tidak pernah >2048."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


def chunk_by_phoneme_length(input_ids, phonemes, max_len):
    """
    Chunk CTC-safe berbasis phoneme.
    input_ids: List[int]
    phonemes: List[str]
    """
    chunks = []
    cur_ids, cur_phon, cur_len = [], [], 0

    for tid, ph in zip(input_ids, phonemes):
        ph_len = len(ph) + 1  # +1 spasi

        if cur_len + ph_len > max_len and len(cur_ids) > 0:
            chunks.append({
                "input_ids": cur_ids,
                "phonemes": cur_phon
            })
            cur_ids, cur_phon, cur_len = [], [], 0

        cur_ids.append(tid)
        cur_phon.append(ph)
        cur_len += ph_len

    if len(cur_ids) > 0:
        chunks.append({
            "input_ids": cur_ids,
            "phonemes": cur_phon
        })

    return chunks


# =========================
# BATCH PREPROCESS (PENTING)
# =========================

def preprocess_batch(batch):
    """
    INPUT:
      batch["text"] : List[str]

    OUTPUT (FLAT):
      {
        "input_ids": List[List[int]],
        "phonemes":  List[List[str]]
      }
    """
    out_input_ids = []
    out_phonemes = []

    for text in batch["text"]:
        # 1. chunk TEXT dulu (sebelum tokenizer)
        text_chunks = chunk_text_by_char(text, MAX_TEXT_CHARS)

        for t in text_chunks:
            # 2. tokenize + phonemize
            result = phonemize(t, tokenizer)
            phonemes = result["phonemes"]
            input_ids = result["input_ids"]

            if len(phonemes) == 0:
                continue

            # 3. CTC chunking
            chunks = chunk_by_phoneme_length(
                input_ids=input_ids,
                phonemes=phonemes,
                max_len=MAX_PHONEME_LEN
            )

            for c in chunks:
                if len(c["phonemes"]) == 0:
                    continue
                out_input_ids.append(c["input_ids"])
                out_phonemes.append(c["phonemes"])

    return {
        "input_ids": out_input_ids,
        "phonemes": out_phonemes
    }


# =========================
# LOAD DATASET
# =========================

parquet_files = glob.glob(f"{PARQUET_FOLDER}/*.parquet")
dataset = load_dataset("parquet", data_files=parquet_files)
dataset = dataset["train"] if "train" in dataset else list(dataset.values())[0]

print("Original dataset size:", len(dataset))


# =========================
# SHARD PROCESSING
# =========================

def process_shard(idx):
    out_dir = f"{OUT_ROOT}/shard_{idx}"
    if os.path.exists(out_dir):
        print(f"Shard {idx} exists, skip.")
        return

    print(f"Processing shard {idx}")
    shard = dataset.shard(NUM_SHARDS, idx)

    processed = shard.map(
        preprocess_batch,
        batched=True,
        batch_size=8,
        remove_columns=shard.column_names,
        load_from_cache_file=False
    )

    # filter safety (opsional, tapi aman)
    processed = processed.filter(
        lambda x: len(" ".join(x["phonemes"])) <= MAX_PHONEME_LEN
    )

    if len(processed) == 0:
        print(f"Shard {idx} empty, skip save.")
        return

    os.makedirs(out_dir, exist_ok=True)
    processed.save_to_disk(out_dir)
    print(f"Shard {idx} saved with {len(processed)} samples.")


# =========================
# RUN MULTIPROCESS
# =========================

with ProcessPool(max_workers=32) as pool:
    pool.map(process_shard, range(NUM_SHARDS), timeout=600)


# =========================
# CONCATENATE SHARDS
# =========================

shards = []
for s in sorted(os.listdir(OUT_ROOT)):
    d = f"{OUT_ROOT}/{s}"
    try:
        shards.append(load_from_disk(d))
    except:
        pass

final_dataset = concatenate_datasets(shards)
print(f"Final dataset size: {len(final_dataset)}")

final_dataset.save_to_disk(config["data_folder"])
print(f"Dataset saved to {config['data_folder']}")
