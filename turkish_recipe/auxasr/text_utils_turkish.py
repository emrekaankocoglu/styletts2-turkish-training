#coding:utf-8
"""
Text utilities for Turkish AuxiliaryASR
Converts phoneme sequences to token indices using word_index_dict
"""

import os
import os.path as osp
import pandas as pd

DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'Data/word_index_dict_turkish.txt')


class TextCleaner:
    def __init__(self, word_index_dict_path=DEFAULT_DICT_PATH):
        self.word_index_dictionary = self.load_dictionary(word_index_dict_path)
        self.unk_index = self.word_index_dictionary.get('<unk>', 3)
        print(f"TextCleaner loaded {len(self.word_index_dictionary)} tokens")

    def __call__(self, phoneme_list):
        """
        Convert list of phonemes to list of indices.
        
        Args:
            phoneme_list: List of phoneme strings (e.g., ['m', 'ɛ', 'r', 'h', 'a', 'b', 'a'])
        
        Returns:
            List of integer indices
        """
        indexes = []
        for phoneme in phoneme_list:
            # Each phoneme might be multi-character (e.g., 'tʃ')
            # Convert each character to index
            for char in phoneme:
                try:
                    indexes.append(self.word_index_dictionary[char])
                except KeyError:
                    # Unknown character - use <unk> token
                    indexes.append(self.unk_index)
        return indexes

    def load_dictionary(self, path):
        """Load word index dictionary from CSV file."""
        try:
            csv = pd.read_csv(path, header=None).values
            word_index_dict = {word: int(index) for word, index in csv}
            return word_index_dict
        except Exception as e:
            print(f"Error loading dictionary from {path}: {e}")
            # Return a minimal dictionary
            return {
                '<pad>': 0,
                '<sos>': 1,
                '<eos>': 2,
                '<unk>': 3,
                ' ': 4
            }
