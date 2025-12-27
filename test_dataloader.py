from phoneme_tokenizer import PhonemeTokenizer
from dataloader_ctc import FilePathDataset, collate_fn
from torch.utils.data import DataLoader
import torch

# Setup
phoneme_tokenizer = PhonemeTokenizer()

# Buat dummy dataset
dummy_data = [
    {
        "phoneme": "hˈeɪloʊ wˈɜːld",
        "bpe_ids": [1234, 5678]
    },
    {
        "phoneme": "sˈaja mˈakan nˈasi",
        "bpe_ids": [111, 222, 333]
    },
    {
        "phoneme": "tˈɛst !",
        "bpe_ids": [999]
    },
    {
        "phoneme": "kˈɑːt kˈæt dˈɔːg",
        "bpe_ids": [1111, 2222, 3333]
    },
    {
        "phoneme": "pˈaɪθɑːn prˈɑːɡræmɪŋ",
        "bpe_ids": [4444, 5555]
    },
    {
        "phoneme": "məʃˈiːn lˈɜːnɪŋ mˈɑːdəl",
        "bpe_ids": [6666, 7777, 8888]
    },
    {
        "phoneme": "dˈiːp nˈɪrəl nˈɛtˌwɜːk",
        "bpe_ids": [9999, 1010, 1111]
    },
    {
        "phoneme": "tɹˈænsfɔɹmə ˈɑːrkɪtˌɛktʃɚ",
        "bpe_ids": [2020, 3030, 4040]
    },
    {
        "phoneme": "θˈɪŋk kɹɪtɪkəli ənd ɪmˈæʒɪnətɪvli",
        "bpe_ids": [5050, 6060]
    },
    {
        "phoneme": "tˈuː bɪ ɔɹ nˈɑːt tˈuː bɪ",
        "bpe_ids": [7070, 8080, 9090]
    }
]

# Build vocabulary dari dummy data
print("Building phoneme vocabulary...")
for item in dummy_data:
    phoneme_tokenizer.build_from_sentence(item["phoneme"])

print(f"Vocab size: {phoneme_tokenizer.vocab_size}")
print(f"Space ID: {phoneme_tokenizer.space_id}")
print(f"Mask ID: {phoneme_tokenizer.mask_id}")
print()

# Create dataset
dataset = FilePathDataset(
    dataset=dummy_data,
    phoneme_tokenizer=phoneme_tokenizer,
    mlm_prob=0.5,  # 50% untuk demo agar lebih terlihat
    replace_prob=0.2
)

print("=" * 80)
print("SINGLE SAMPLE FROM DATASET")
print("=" * 80)

# Get single sample
sample = dataset[0]
print(f"Original phoneme: {dummy_data[0]['phoneme']}")
print(f"Phoneme IDs (masked): {sample['phoneme_ids']}")
print(f"MLM Labels:           {sample['mlm_labels']}")
print(f"BPE IDs:              {sample['bpe_ids']}")
print()

# Decode untuk melihat hasil masking
decoded_input = [phoneme_tokenizer.id2phoneme.get(id, f"<unk:{id}>") for id in sample['phoneme_ids']]
print(f"Decoded input: {''.join(decoded_input)}")
print()

# Tampilkan mana yang di-mask
masked_positions = [i for i, label in enumerate(sample['mlm_labels']) if label != -100]
print(f"Masked positions: {masked_positions}")
print()

print("=" * 80)
print("BATCH FROM DATALOADER")
print("=" * 80)

# Create dataloader
def my_collate_fn(batch):
    return collate_fn(batch, phoneme_tokenizer)

dataloader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    collate_fn=my_collate_fn
)

# Get one batch
batch = next(iter(dataloader))

print("Batch keys:", batch.keys())
print()
print(f"phoneme_input shape:  {batch['phoneme_input'].shape}")
print(f"mlm_labels shape:     {batch['mlm_labels'].shape}")
print(f"attention_mask shape: {batch['attention_mask'].shape}")
print()
print(f"phoneme_input:\n{batch['phoneme_input']}")
print()
print(f"mlm_labels:\n{batch['mlm_labels']}")
print()
print(f"attention_mask:\n{batch['attention_mask']}")
print()
print(f"ctc_targets: {batch['ctc_targets']}")
print(f"input_lengths: {batch['input_lengths']}")
print(f"target_lengths: {batch['target_lengths']}")
print()

# Decode batch untuk visualisasi
print("=" * 80)
print("DECODED BATCH")
print("=" * 80)
for i in range(batch['phoneme_input'].shape[0]):
    phon_ids = batch['phoneme_input'][i].tolist()
    length = batch['input_lengths'][i].item()
    
    decoded = [phoneme_tokenizer.id2phoneme.get(id, f"<unk:{id}>") 
               for id in phon_ids[:length]]
    
    print(f"\nSample {i}:")
    print(f"  Input:  {''.join(decoded)}")
    print(f"  Length: {length}")
    
    # Tampilkan posisi yang di-mask
    labels = batch['mlm_labels'][i].tolist()[:length]
    masked_pos = [j for j, label in enumerate(labels) if label != -100]
    print(f"  Masked positions: {masked_pos}")
