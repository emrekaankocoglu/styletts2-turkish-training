import os
import os.path as osp
import sys
import time
from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
from torch import nn
import jiwer

import matplotlib.pylab as plt

def calc_wer(target, pred, ignore_indexes=[0]):
    target_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(target)))))
    pred_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(pred)))))
    target_str = ' '.join(target_chars)
    pred_str = ' '.join(pred_chars)
    error = jiwer.wer(target_str, pred_str)
    return error

def ctc_greedy_decode(tokens, blank, ignore_indexes=None):
    """
    Greedy CTC decode: collapse repeats then remove blank and ignored tokens.

    tokens: 1D iterable of token ids (e.g. argmax over time)
    blank: int blank token id
    ignore_indexes: iterable of token ids to drop (e.g. pad/sos/eos/unk/space)
    """
    if ignore_indexes is None:
        ignore_indexes = []
    ignore_set = set(ignore_indexes)
    ignore_set.add(int(blank))

    # Convert to a plain python list of ints
    if hasattr(tokens, "detach"):
        seq = tokens.detach().cpu().tolist()
    else:
        seq = list(tokens)

    collapsed = []
    prev = None
    for t in seq:
        t = int(t)
        if prev is None or t != prev:
            collapsed.append(t)
        prev = t

    decoded = [t for t in collapsed if t not in ignore_set]
    return decoded

def _edit_distance(a, b):
    """Levenshtein distance between 1D int sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # O(min(n,m)) memory
    if n < m:
        a, b = b, a
        n, m = m, n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[m]

def calc_ctc_error_rate(target, pred, blank, ignore_indexes=None):
    """
    CTC-style error rate:
    - decode pred with greedy CTC collapse (collapse repeats + remove blank)
    - filter target/pred by ignore_indexes
    - compute normalized edit distance: edits / max(1, len(target))

    This is closer to PER than true word-level WER.
    """
    if ignore_indexes is None:
        ignore_indexes = []
    ignore_set = set(ignore_indexes)
    ignore_set.add(int(blank))

    # Target -> filtered list
    if hasattr(target, "detach"):
        tgt = target.detach().cpu().tolist()
    else:
        tgt = list(target)
    tgt = [int(t) for t in tgt if int(t) not in ignore_set]

    # Pred -> greedy CTC decode list
    hyp = ctc_greedy_decode(pred, blank=int(blank), ignore_indexes=ignore_indexes)

    denom = max(1, len(tgt))
    return _edit_distance(tgt, hyp) / denom

def drop_duplicated(chars):
    ret_chars = [chars[0]]
    for prev, curr in zip(chars[:-1], chars[1:]):
        if prev != curr:
            ret_chars.append(curr)
    return ret_chars

def build_criterion(critic_params={}):
    criterion = {
        "ce": nn.CrossEntropyLoss(ignore_index=-1),
        "ctc": torch.nn.CTCLoss(**critic_params.get('ctc', {})),
    }
    return criterion

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    return train_list, val_list


def plot_image(image):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(image, aspect="auto", origin="lower",
                   interpolation='none')

    fig.canvas.draw()
    plt.close()

    return fig