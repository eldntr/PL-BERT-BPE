from phoneme_tokenizer import PhonemeTokenizer
from dataloader_ctc import FilePathDataset, collate_fn   # import dari file yang sudah dibuat

# Buat phoneme vocab manual
phoneme_tokenizer = PhonemeTokenizer()

# Tambahkan phoneme yang akan dipakai
for p in ["s","a","j","m","k","n","r","o","t","i"]:
    phoneme_tokenizer.add_phoneme(p)

print("Phoneme vocab:", phoneme_tokenizer.phoneme2id)

sample_data = [
    {
        "words": ["saya", "makan"],
        "phonemes": ["s a j a", "m a k a n"],
        "bpe_ids": [[10,11], [12,13]]
    },
    {
        "words": ["halo", "dunia"],
        "phonemes": ["h a l o", "d u n i a"],
        "bpe_ids": [[20], [21,22]]
    }
]

class ToyDataset:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

toy_dataset = ToyDataset(sample_data)

fp_dataset = FilePathDataset(
    toy_dataset, 
    phoneme_tokenizer,
    mlm_prob=0.5  # agar sering terlihat masking
)

item0 = fp_dataset[0]
print("Item 0 =", item0)

batch = [fp_dataset[0], fp_dataset[1]]

output = collate_fn(batch, phoneme_tokenizer, mlm_prob=0.5)

for k,v in output.items():
    print(k, "=", v)
