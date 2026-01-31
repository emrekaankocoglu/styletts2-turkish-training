#!/usr/bin/env python3
"""
Prepare Turkish TTS datasets for AuxiliaryASR training.

Downloads and processes:
- omersaidd/tts_nisan_kumru_tur (speaker 0)
- omersaidd/tts_mazlum_kiper_tur (speaker 1)

Creates train_list.txt and val_list.txt in format:
    path/to/audio.wav|text|speaker_id
"""

import os
import sys
import random
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np

# Target sample rate for AuxiliaryASR
TARGET_SR = 24000

# Directories
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "Data"
AUDIO_DIR = DATA_DIR / "wavs"

# Dataset configs: (dataset_name, speaker_id)
DATASETS = [
    ("omersaidd/tts_nisan_kumru_tur", 0),
    ("omersaidd/tts_mazlum_kiper_tur", 1),
    ("Anilosan15/Turkish_TTS_Data", 2),  # 30.6k samples, speaker "sıla", 48kHz
]

# Train/val split ratio
VAL_RATIO = 0.05  # 5% for validation


def resample_audio(audio_array, orig_sr, target_sr):
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio_array
    
    # Use scipy for resampling
    from scipy import signal
    
    num_samples = int(len(audio_array) * target_sr / orig_sr)
    resampled = signal.resample(audio_array, num_samples)
    return resampled.astype(np.float32)


def process_dataset(dataset_name, speaker_id, audio_dir, entries):
    """Process a single dataset and add entries to the list."""
    from datasets import load_dataset
    
    print(f"\nLoading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split='train')
    
    print(f"Processing {len(ds)} samples...")
    speaker_dir = audio_dir / f"speaker_{speaker_id}"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, sample in enumerate(tqdm(ds, desc=f"Speaker {speaker_id}")):
        # Get audio data
        audio_data = sample['audio']
        audio_array = audio_data['array']
        orig_sr = audio_data['sampling_rate']
        
        # Resample if needed
        if orig_sr != TARGET_SR:
            audio_array = resample_audio(audio_array, orig_sr, TARGET_SR)
        
        # Normalize audio to [-1, 1] range
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val * 0.95
        
        # Save audio file
        audio_filename = f"s{speaker_id}_{idx:06d}.wav"
        audio_path = speaker_dir / audio_filename
        sf.write(str(audio_path), audio_array, TARGET_SR)
        
        # Get text
        text = sample['text'].strip()
        
        # Skip empty texts
        if not text:
            continue
        
        # Clean text: remove newlines, normalize whitespace
        text = ' '.join(text.split())
        
        # Add entry
        rel_path = audio_path.relative_to(DATA_DIR)
        entries.append((str(rel_path), text, speaker_id))
    
    print(f"Processed {len(ds)} samples from {dataset_name}")
    return len(ds)


def main():
    print("=" * 60)
    print("Turkish ASR Data Preparation")
    print("=" * 60)
    
    # Create directories
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all entries
    all_entries = []
    
    # Process each dataset
    for dataset_name, speaker_id in DATASETS:
        try:
            process_dataset(dataset_name, speaker_id, AUDIO_DIR, all_entries)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            raise
    
    print(f"\nTotal entries: {len(all_entries)}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_entries)
    
    val_size = int(len(all_entries) * VAL_RATIO)
    val_entries = all_entries[:val_size]
    train_entries = all_entries[val_size:]
    
    print(f"Train entries: {len(train_entries)}")
    print(f"Val entries: {len(val_entries)}")
    
    # Write train_list.txt
    train_path = DATA_DIR / "train_list.txt"
    with open(train_path, 'w', encoding='utf-8') as f:
        for rel_path, text, speaker_id in train_entries:
            f.write(f"{rel_path}|{text}|{speaker_id}\n")
    print(f"Wrote {train_path}")
    
    # Write val_list.txt
    val_path = DATA_DIR / "val_list.txt"
    with open(val_path, 'w', encoding='utf-8') as f:
        for rel_path, text, speaker_id in val_entries:
            f.write(f"{rel_path}|{text}|{speaker_id}\n")
    print(f"Wrote {val_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    # Calculate total duration
    total_duration = 0
    for entry in all_entries:
        audio_path = DATA_DIR / entry[0]
        if audio_path.exists():
            info = sf.info(str(audio_path))
            total_duration += info.duration
    
    print(f"Total audio duration: {total_duration / 3600:.2f} hours")
    print(f"Sample rate: {TARGET_SR} Hz")
    print(f"Train samples: {len(train_entries)}")
    print(f"Val samples: {len(val_entries)}")
    
    # Show sample entries
    print("\nSample train entries:")
    for entry in train_entries[:3]:
        print(f"  {entry[0]}|{entry[1][:50]}...|{entry[2]}")


if __name__ == "__main__":
    main()
