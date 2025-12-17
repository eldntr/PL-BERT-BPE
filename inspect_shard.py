import os
import sys
from datasets import load_from_disk

# =========================
# CONFIG
# =========================
SHARD_ROOT = "./wiki_phoneme"   # folder shard
DEFAULT_SHARD = "shard_0"       # shard default untuk dicek
MAX_PRINT = 3                  # berapa sample ditampilkan


def inspect_shard(shard_name=None):
    shard_name = shard_name or DEFAULT_SHARD
    shard_path = os.path.join(SHARD_ROOT, shard_name)

    if not os.path.exists(shard_path):
        print(f"[ERROR] Shard path not found: {shard_path}")
        return

    print("=" * 80)
    print(f"Inspecting shard: {shard_path}")
    print("=" * 80)

    ds = load_from_disk(shard_path)

    print("\n[INFO] Dataset object")
    print(ds)

    print("\n[INFO] Number of samples:", len(ds))
    if len(ds) == 0:
        print("[WARNING] Shard is EMPTY!")
        return

    print("\n[INFO] Features / Columns:")
    for k, v in ds.features.items():
        print(f"  - {k}: {v}")

    print("\n[INFO] Showing samples:")
    for i in range(min(MAX_PRINT, len(ds))):
        ex = ds[i]
        print("-" * 40)
        print(f"Sample #{i}")

        input_ids = ex["input_ids"]
        phonemes = ex["phonemes"]

        print(f"  input_ids length : {len(input_ids)}")
        print(f"  phonemes length  : {len(phonemes)}")

        # Tampilkan sebagian isi
        print(f"  input_ids (head): {input_ids[:20]}")
        print(f"  phonemes  (head): {phonemes[:20]}")

        # Gabungan phoneme string (untuk cek panjang total)
        joined = " ".join(phonemes)
        print(f"  joined phoneme string length: {len(joined)}")

    print("\n[OK] Inspection done.")


if __name__ == "__main__":
    # Bisa jalankan:
    #   python inspect_shard.py
    # atau:
    #   python inspect_shard.py shard_10
    shard = sys.argv[1] if len(sys.argv) > 1 else None
    inspect_shard(shard)
