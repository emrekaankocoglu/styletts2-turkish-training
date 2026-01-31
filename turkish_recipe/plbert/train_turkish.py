#!/usr/bin/env python3
"""
Turkish PL-BERT Training Script
Multi-GPU training using Accelerate on 4x RTX 3090

Usage:
    accelerate launch --num_processes=4 --mixed_precision=fp16 train_turkish.py
"""

import os
import sys
import shutil
import os.path as osp
import time

import torch
from torch import nn
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from torch.optim import AdamW
from transformers import AlbertConfig, AlbertModel

from datasets import load_from_disk
from tqdm import tqdm

import yaml
import pickle
import json
import argparse
from datetime import datetime, timedelta

# Add parent paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'PL-BERT'))

from dataloader_turkish import build_dataloader
from model import MultiTaskModel
from utils import length_to_mask


def format_time(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def train(config_path):
    # Load config
    config = yaml.safe_load(open(config_path))
    
    # Load token maps
    with open(config['dataset_params']['token_maps'], 'rb') as handle:
        token_maps = pickle.load(handle)
    
    # Training params
    criterion = nn.CrossEntropyLoss()
    num_steps = config['num_steps']
    log_interval = config['log_interval']
    save_interval = config['save_interval']
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    # Setup DDP
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        split_batches=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Load dataset
    accelerator.print(f"Loading dataset from {config['data_folder']}...")
    dataset = load_from_disk(config["data_folder"])
    accelerator.print(f"Dataset size: {len(dataset)} samples")
    
    # Setup logging directory
    log_dir = config['log_dir']
    if accelerator.is_main_process:
        if not osp.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
        accelerator.print(f"Checkpoints will be saved to: {log_dir}")
    
    # Build dataloader
    batch_size = config["batch_size"]
    train_loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        dataset_config=config['dataset_params']
    )
    
    # Build model
    albert_base_configuration = AlbertConfig(**config['model_params'])
    bert = AlbertModel(albert_base_configuration)
    
    num_vocab = 1 + max([m['token'] for m in token_maps.values()])
    bert = MultiTaskModel(
        bert,
        num_vocab=num_vocab,
        num_tokens=config['model_params']['vocab_size'],
        hidden_size=config['model_params']['hidden_size']
    )
    
    # Print training config
    effective_batch_size = batch_size * gradient_accumulation_steps
    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"Training Configuration:")
    accelerator.print(f"  - Model vocab_size: {config['model_params']['vocab_size']}")
    accelerator.print(f"  - Token vocab (num_vocab): {num_vocab}")
    accelerator.print(f"  - Batch size per step: {batch_size}")
    accelerator.print(f"  - Gradient accumulation: {gradient_accumulation_steps}")
    accelerator.print(f"  - Effective batch size: {effective_batch_size}")
    accelerator.print(f"  - Learning rate: {config.get('lr', 1e-4)}")
    accelerator.print(f"  - Total steps: {num_steps}")
    accelerator.print(f"  - Mixed precision: {config['mixed_precision']}")
    accelerator.print(f"{'='*60}\n")
    
    # Checkpoint detection - find latest checkpoint (new-style dir or old-style .t7)
    iters = 0
    batches_to_skip = 0
    checkpoint_path = None
    checkpoint_type = None  # 'new' for accelerate dirs, 'old' for .t7 files
    
    if accelerator.is_main_process:
        try:
            if osp.exists(log_dir):
                entries = os.listdir(log_dir)
                
                # Check for new-style checkpoint directories (step_XXXXX/)
                new_style_ckpts = [e for e in entries if e.startswith("step_") and osp.isdir(osp.join(log_dir, e))]
                # Check for old-style .t7 files (step_XXXXX.t7)
                old_style_ckpts = [e for e in entries if e.startswith("step_") and e.endswith(".t7")]
                
                latest_new = 0
                latest_old = 0
                
                if new_style_ckpts:
                    new_iters = [int(f.split('_')[-1]) for f in new_style_ckpts]
                    latest_new = max(new_iters)
                
                if old_style_ckpts:
                    old_iters = [int(f.split('_')[-1].split('.')[0]) for f in old_style_ckpts]
                    latest_old = max(old_iters)
                
                # Use whichever is more recent
                if latest_new >= latest_old and latest_new > 0:
                    iters = latest_new
                    checkpoint_path = osp.join(log_dir, f"step_{iters}")
                    checkpoint_type = 'new'
                    accelerator.print(f"Found new-style checkpoint at step {iters}")
                elif latest_old > 0:
                    iters = latest_old
                    checkpoint_path = osp.join(log_dir, f"step_{iters}.t7")
                    checkpoint_type = 'old'
                    accelerator.print(f"Found old-style checkpoint at step {iters}")
        except Exception as e:
            accelerator.print(f"No checkpoint found: {e}")
    
    # Broadcast checkpoint info to all processes
    iters_tensor = torch.tensor([iters], device=accelerator.device)
    iters = int(accelerator.gather(iters_tensor)[0].item())
    
    # Broadcast checkpoint_type (0=none, 1=old, 2=new)
    type_code = 0 if checkpoint_type is None else (1 if checkpoint_type == 'old' else 2)
    type_tensor = torch.tensor([type_code], device=accelerator.device)
    type_code = int(accelerator.gather(type_tensor)[0].item())
    checkpoint_type = None if type_code == 0 else ('old' if type_code == 1 else 'new')
    
    if checkpoint_type and checkpoint_path is None:
        # Non-main processes need to reconstruct checkpoint_path
        if checkpoint_type == 'new':
            checkpoint_path = osp.join(log_dir, f"step_{iters}")
        else:
            checkpoint_path = osp.join(log_dir, f"step_{iters}.t7")
    
    # Optimizer with configurable lr
    lr = config.get('lr', 1e-4)
    optimizer = AdamW(bert.parameters(), lr=lr)
    
    # Load old-style checkpoint BEFORE prepare() (manual loading)
    if checkpoint_type == 'old' and accelerator.is_main_process:
        accelerator.print(f"Loading old-style checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        state_dict = checkpoint['net']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        bert.load_state_dict(new_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        accelerator.print(f"Loaded old-style checkpoint from step {iters}")
    
    # Prepare for distributed training
    bert, optimizer, train_loader = accelerator.prepare(bert, optimizer, train_loader)
    
    # Load new-style checkpoint AFTER prepare() (accelerate native)
    if checkpoint_type == 'new':
        accelerator.print(f"Loading new-style checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        
        # Load custom metadata
        metadata_path = osp.join(checkpoint_path, 'custom_checkpoint.json')
        if osp.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            iters = metadata.get('step', iters)
            batches_to_skip = metadata.get('batches_in_epoch', 0)
            accelerator.print(f"Loaded metadata: step={iters}, batches_to_skip={batches_to_skip}")
        accelerator.print(f"Loaded new-style checkpoint from step {iters}")
    
    # Skip batches to resume exact dataloader position
    if batches_to_skip > 0:
        accelerator.print(f"Skipping {batches_to_skip} batches to resume epoch position...")
        train_loader = accelerator.skip_first_batches(train_loader, batches_to_skip)
        accelerator.print(f"Skipped {batches_to_skip} batches")
    
    accelerator.print(f"Starting training from step {iters}...")
    
    # Training loop with tqdm
    running_loss = 0
    running_vocab_loss = 0
    running_token_loss = 0
    start_time = time.time()
    step_times = []
    
    # Track batches processed in current epoch for exact resume
    batches_in_epoch = batches_to_skip  # Start from where we left off
    
    # Create progress bar (only on main process)
    if accelerator.is_main_process:
        pbar = tqdm(total=num_steps, initial=iters, desc="Training", 
                    unit="step", dynamic_ncols=True)
    
    while iters < num_steps:
        # Reset batch counter at start of each epoch (except first if resuming)
        if batches_in_epoch > 0 and batches_to_skip > 0:
            # First iteration after resume - don't reset yet
            batches_to_skip = 0  # Clear the flag for next epoch
        else:
            batches_in_epoch = 0
        
        for batch in train_loader:
            step_start = time.time()
            batches_in_epoch += 1
            
            with accelerator.accumulate(bert):
                words, labels, phonemes, input_lengths, masked_indices = batch
                text_mask = length_to_mask(torch.Tensor(input_lengths))
                
                tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())
                
                # Vocabulary loss (word prediction)
                loss_vocab = 0
                for _s2s_pred, _text_input, _text_length, _masked_indices in zip(words_pred, words, input_lengths, masked_indices):
                    loss_vocab += criterion(_s2s_pred[:_text_length], _text_input[:_text_length])
                loss_vocab /= words.size(0)
                
                # Token loss (masked phoneme prediction)
                loss_token = 0
                sizes = 1
                for _s2s_pred, _text_input, _text_length, _masked_indices in zip(tokens_pred, labels, input_lengths, masked_indices):
                    if len(_masked_indices) > 0:
                        _text_input = _text_input[:_text_length][_masked_indices]
                        loss_tmp = criterion(_s2s_pred[:_text_length][_masked_indices], _text_input[:_text_length])
                        loss_token += loss_tmp
                        sizes += 1
                loss_token /= sizes
                
                loss = loss_vocab + loss_token
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            # Only count as a step after accumulation is complete
            if accelerator.sync_gradients:
                step_time = time.time() - step_start
                step_times.append(step_time)
                if len(step_times) > 100:
                    step_times.pop(0)
                
                running_loss += loss.item()
                running_vocab_loss += loss_vocab.item()
                running_token_loss += loss_token.item()
                
                iters += 1
                
                if accelerator.is_main_process:
                    # Calculate ETA
                    avg_step_time = sum(step_times) / len(step_times)
                    remaining_steps = num_steps - iters
                    eta_seconds = remaining_steps * avg_step_time
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f'{loss.item():.3f}',
                        'v_loss': f'{loss_vocab.item():.3f}',
                        't_loss': f'{loss_token.item():.3f}',
                        'ETA': format_time(eta_seconds)
                    })
                
                if iters % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    avg_vocab = running_vocab_loss / log_interval
                    avg_token = running_token_loss / log_interval
                    elapsed = time.time() - start_time
                    
                    accelerator.print(
                        f'\nStep [{iters}/{num_steps}] | '
                        f'Loss: {avg_loss:.4f} (V:{avg_vocab:.4f} T:{avg_token:.4f}) | '
                        f'Time: {format_time(elapsed)} | '
                        f'Speed: {iters/elapsed:.1f} steps/s'
                    )
                    running_loss = 0
                    running_vocab_loss = 0
                    running_token_loss = 0
                
                if iters % save_interval == 0:
                    accelerator.print(f'\nSaving checkpoint at step {iters}...')
                    
                    # Save using accelerate's native save_state
                    checkpoint_dir = osp.join(log_dir, f'step_{iters}')
                    accelerator.save_state(checkpoint_dir)
                    
                    # Save custom metadata (step count and dataloader position)
                    if accelerator.is_main_process:
                        metadata = {
                            'step': iters,
                            'batches_in_epoch': batches_in_epoch
                        }
                        metadata_path = osp.join(checkpoint_dir, 'custom_checkpoint.json')
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        accelerator.print(f'Checkpoint saved to {checkpoint_dir}')
                
                if iters >= num_steps:
                    break
        
        if iters >= num_steps:
            break
    
    if accelerator.is_main_process:
        pbar.close()
    
    total_time = time.time() - start_time
    accelerator.print(f"\nTraining complete! Total time: {format_time(total_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Turkish PL-BERT")
    parser.add_argument("--config", type=str, default="turkish_recipe/plbert/config_turkish.yml",
                        help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)
