import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from phoneme_tokenizer import PhonemeTokenizer
from dataloader_ctc import FilePathDataset, collate_fn


def test_dataloader():
    """Test dataloader functionality"""
    
    print("=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset_path = "wiki_phoneme_final/wiki_phoneme_final"
    hf_dataset = load_from_disk(dataset_path)
    print(f"✓ Dataset loaded: {len(hf_dataset)} examples")
    print(f"  Dataset columns: {hf_dataset.column_names}")
    
    # Load phoneme tokenizer
    print("\n2. Loading phoneme tokenizer...")
    phoneme_tokenizer = PhonemeTokenizer.load("phoneme_vocab.json")
    print(f"✓ Phoneme vocab size: {phoneme_tokenizer.vocab_size}")
    
    # Create FilePathDataset
    print("\n3. Creating FilePathDataset...")
    dataset = FilePathDataset(
        hf_dataset, 
        phoneme_tokenizer, 
        mlm_prob=0.15, 
        max_length=512
    )
    print(f"✓ FilePathDataset created with {len(dataset)} valid examples")
    
    # Test single item
    print("\n4. Testing single item...")
    idx = 0
    item = dataset[idx]
    print(f"  Item keys: {item.keys()}")
    print(f"  phoneme_ids type: {type(item['phoneme_ids'])}")
    print(f"  phoneme_ids len: {len(item['phoneme_ids'])}")
    print(f"  word_spans type: {type(item['word_spans'])}")
    print(f"  word_spans len: {len(item['word_spans'])}")
    print(f"  bpe_ids type: {type(item['bpe_ids'])}")
    print(f"  bpe_ids len: {len(item['bpe_ids'])}")
    print(f"✓ Single item test passed")
    
    # Create DataLoader
    print("\n5. Creating DataLoader...")
    batch_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, phoneme_tokenizer, mlm_prob=0.15),
        num_workers=0
    )
    print(f"✓ DataLoader created")
    
    # Test batch
    print("\n6. Testing batch from DataLoader...")
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print(f"  Batch keys: {batch.keys()}")
            print(f"  phoneme_input shape: {batch['phoneme_input'].shape}")
            print(f"  mlm_labels shape: {batch['mlm_labels'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  ctc_targets shape: {batch['ctc_targets'].shape}")
            print(f"  input_lengths: {batch['input_lengths']}")
            print(f"  target_lengths: {batch['target_lengths']}")
            
            # Verify shapes
            B, T = batch['phoneme_input'].shape
            assert batch['mlm_labels'].shape == (B, T), "mlm_labels shape mismatch"
            assert batch['attention_mask'].shape == (B, T), "attention_mask shape mismatch"
            assert batch['input_lengths'].shape == (B,), "input_lengths shape mismatch"
            assert batch['target_lengths'].shape == (B,), "target_lengths shape mismatch"
            
            # Verify dtypes
            assert batch['phoneme_input'].dtype == torch.long, "phoneme_input dtype should be long"
            assert batch['mlm_labels'].dtype == torch.long, "mlm_labels dtype should be long"
            assert batch['attention_mask'].dtype == torch.long, "attention_mask dtype should be long"
            
            print(f"✓ Batch test passed (batch size: {B}, sequence length: {T})")
            break
    
    # Test multiple batches
    print("\n7. Testing multiple batches...")
    batch_count = 0
    for batch in loader:
        batch_count += 1
        if batch_count >= 5:
            break
    print(f"✓ Successfully iterated {batch_count} batches")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_dataloader()
