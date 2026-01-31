#!/usr/bin/env python3
"""
Preprocess Turkish Wikipedia dataset from styletts2-community/multilingual-pl-bert
for PL-BERT training.
"""

import os
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def analyze_phonemes(dataset, num_samples=None):
    """Analyze all phoneme characters in the dataset."""
    all_chars = set()
    
    print("Analyzing phoneme characters...")
    for i, sample in enumerate(tqdm(dataset, desc="Scanning phonemes")):
        for phoneme in sample['phonemes']:
            all_chars.update(phoneme)
        if num_samples and i >= num_samples:
            break
    
    return all_chars


def process_sample(sample):
    """Process a single sample: flatten input_ids."""
    flat_input_ids = []
    valid_phonemes = []
    
    for ids, phoneme in zip(sample['input_ids'], sample['phonemes']):
        if isinstance(ids, list):
            if len(ids) > 0:
                flat_input_ids.append(ids[0])
                valid_phonemes.append(phoneme)
        else:
            flat_input_ids.append(ids)
            valid_phonemes.append(phoneme)
    
    return {
        'phonemes': valid_phonemes,
        'input_ids': flat_input_ids
    }


def build_token_maps(dataset, tokenizer, word_separator=102):
    """Build token_maps dictionary from the dataset."""
    print("Building token maps...")
    unique_tokens = set()
    unique_tokens.add(word_separator)
    
    for sample in tqdm(dataset, desc="Collecting unique tokens"):
        for ids in sample['input_ids']:
            if isinstance(ids, list):
                if len(ids) > 0:
                    unique_tokens.add(ids[0])
            else:
                unique_tokens.add(ids)
    
    print(f"Found {len(unique_tokens)} unique tokens")
    
    token_maps = {}
    for idx, token_id in enumerate(sorted(unique_tokens)):
        try:
            word = tokenizer.decode([token_id]).strip().lower()
        except Exception:
            word = f"<unk_{token_id}>"
        
        token_maps[token_id] = {
            'word': word,
            'token': idx
        }
    
    return token_maps


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Turkish dataset for PL-BERT")
    parser.add_argument("--output_dir", type=str, default="turkish_recipe/plbert/data",
                        help="Output directory for processed data")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (None for all)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze phoneme characters without processing")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading tokenizer: bert-base-multilingual-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    print("Loading Turkish dataset from styletts2-community/multilingual-pl-bert...")
    dataset = load_dataset(
        'styletts2-community/multilingual-pl-bert',
        data_dir='tr',
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    phoneme_chars = analyze_phonemes(dataset, args.num_samples)
    phoneme_chars_path = os.path.join(args.output_dir, "phoneme_chars.txt")
    with open(phoneme_chars_path, 'w', encoding='utf-8') as f:
        sorted_chars = sorted(phoneme_chars)
        f.write(''.join(sorted_chars))
        f.write('\n\n# Character list:\n')
        for char in sorted_chars:
            f.write(f"'{char}' (U+{ord(char):04X})\n")
    print(f"Saved phoneme characters to {phoneme_chars_path}")
    print(f"Total unique phoneme characters: {len(phoneme_chars)}")
    
    if args.analyze_only:
        return
    
    token_maps = build_token_maps(dataset, tokenizer, word_separator=102)
    token_maps_path = os.path.join(args.output_dir, "token_maps_turkish.pkl")
    with open(token_maps_path, 'wb') as f:
        pickle.dump(token_maps, f)
    print(f"Saved token_maps to {token_maps_path}")
    print(f"Token maps size: {len(token_maps)} entries")
    print(f"Max simplified token index: {max(m['token'] for m in token_maps.values())}")
    
    print("Processing dataset...")
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    processed_dataset = dataset.map(
        process_sample,
        remove_columns=['id', 'url', 'title'],
        desc="Processing samples"
    )
    
    output_path = os.path.join(args.output_dir, "turkish_wikipedia_processed")
    processed_dataset.save_to_disk(output_path)
    print(f"Saved processed dataset to {output_path}")
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Dataset size: {len(processed_dataset)} samples")
    print(f"Token maps size: {len(token_maps)} unique tokens")
    print(f"Phoneme vocab size: {len(phoneme_chars)} characters")
    
    sample = processed_dataset[0]
    print(f"\nSample: phonemes={sample['phonemes'][:5]}..., input_ids={sample['input_ids'][:5]}...")


if __name__ == "__main__":
    main()
