#coding: utf-8
"""
Turkish PL-BERT Dataloader
Adapted from original PL-BERT dataloader for Turkish data format
"""

import os
import os.path as osp
import random
import numpy as np
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import Turkish-specific text cleaner
import sys
sys.path.insert(0, os.path.dirname(__file__))
from text_utils_turkish import TextCleaner

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,
                 token_maps="token_maps_turkish.pkl",
                 tokenizer="bert-base-multilingual-cased",
                 word_separator=102,  # [SEP] for bert-multilingual
                 token_separator=" ",
                 token_mask="M",
                 max_mel_length=512,
                 word_mask_prob=0.15,
                 phoneme_mask_prob=0.1,
                 replace_prob=0.2):
        
        self.data = dataset
        self.max_mel_length = max_mel_length
        self.word_mask_prob = word_mask_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.text_cleaner = TextCleaner()
        
        self.word_separator = word_separator
        self.token_separator = token_separator
        self.token_mask = token_mask
        
        with open(token_maps, 'rb') as handle:
            self.token_maps = pickle.load(handle)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        phonemes = self.data[idx]['phonemes']
        input_ids = self.data[idx]['input_ids']
        
        # Skip empty samples
        if len(phonemes) == 0 or len(input_ids) == 0:
            # Return a minimal valid sample
            return self.__getitem__((idx + 1) % len(self.data))

        words = []
        labels = ""
        phoneme = ""

        phoneme_list = ''.join(phonemes)
        if len(phoneme_list) == 0:
            return self.__getitem__((idx + 1) % len(self.data))
            
        masked_index = []
        for z in zip(phonemes, input_ids):
            z = list(z)
            
            # Handle token mapping - if token not in maps, use separator
            token_id = z[1] if z[1] in self.token_maps else self.word_separator
            
            words.extend([token_id] * len(z[0]))
            words.append(self.word_separator)
            labels += z[0] + " "

            if np.random.rand() < self.word_mask_prob:
                if np.random.rand() < self.replace_prob:
                    if np.random.rand() < (self.phoneme_mask_prob / self.replace_prob):
                        phoneme += ''.join([phoneme_list[np.random.randint(0, len(phoneme_list))] for _ in range(len(z[0]))])
                    else:
                        phoneme += z[0]
                else:
                    phoneme += self.token_mask * len(z[0])
                    
                masked_index.extend((np.arange(len(phoneme) - len(z[0]), len(phoneme))).tolist())
            else:
                phoneme += z[0]

            phoneme += self.token_separator

        mel_length = len(phoneme)
        masked_idx = np.array(masked_index)
        masked_index = []
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            phoneme = phoneme[random_start:random_start + self.max_mel_length]
            words = words[random_start:random_start + self.max_mel_length]
            labels = labels[random_start:random_start + self.max_mel_length]
            
            for m in masked_idx:
                if m >= random_start and m < random_start + self.max_mel_length:
                    masked_index.append(m - random_start)
        else:
            masked_index = masked_idx.tolist() if len(masked_idx) > 0 else []
            
        phoneme = self.text_cleaner(phoneme)
        labels = self.text_cleaner(labels)
        words = [self.token_maps[w]['token'] for w in words]
        
        assert len(phoneme) == len(words), f"Length mismatch: phoneme={len(phoneme)}, words={len(words)}"
        assert len(phoneme) == len(labels), f"Length mismatch: phoneme={len(phoneme)}, labels={len(labels)}"
        
        phonemes = torch.LongTensor(phoneme)
        labels = torch.LongTensor(labels)
        words = torch.LongTensor(words)
        
        return phonemes, words, labels, masked_index


class Collater(object):
    """Collate function for batching samples."""

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # Sort by length
        lengths = [b[1].shape[0] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        max_text_length = max([b[1].shape[0] for b in batch])

        words = torch.zeros((batch_size, max_text_length)).long()
        labels = torch.zeros((batch_size, max_text_length)).long()
        phonemes = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = []
        masked_indices = []
        
        for bid, (phoneme, word, label, masked_index) in enumerate(batch):
            text_size = phoneme.size(0)
            words[bid, :text_size] = word
            labels[bid, :text_size] = label
            phonemes[bid, :text_size] = phoneme
            input_lengths.append(text_size)
            masked_indices.append(masked_index)

        return words, labels, phonemes, input_lengths, masked_indices


def build_dataloader(df,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = FilePathDataset(df, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
