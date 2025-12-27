from phoneme_tokenizer import PhonemeTokenizer

# Inisialisasi tokenizer
phoneme_tokenizer = PhonemeTokenizer()

# Sample phoneme string
phoneme_str = 'hˈeɪloʊ dunˈia !'

print("Input phoneme string:")
print(f"  '{phoneme_str}'")
print()

# Tokenize sequence
units = phoneme_tokenizer.tokenize_sequence(phoneme_str)
print("Tokenized units:")
print(f"  {units}")
print(f"  Total units: {len(units)}")
print()

# Build vocabulary dari units
phoneme_tokenizer.build_from_sentence(phoneme_str)
print("Vocabulary built. Vocab size:", phoneme_tokenizer.vocab_size)
print()

# Encode to IDs
ids = phoneme_tokenizer.encode(phoneme_str)
print("Encoded IDs:")
print(f"  {ids}")
print(f"  Total IDs: {len(ids)}")
print()

# Show phoneme2id mapping
print("Phoneme to ID mapping:")
for phoneme, idx in sorted(phoneme_tokenizer.phoneme2id.items(), key=lambda x: x[1]):
    print(f"  {repr(phoneme):20s} -> {idx}")
