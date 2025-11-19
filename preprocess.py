import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Iterator, List, Dict, Tuple
import pickle
from pathlib import Path

from cs336_basics.bpe_tokenizer import train_bpe, BPETokenizer


def load_precomputed_data(
    vocab_path: str = "tiny_stories_vocab.pkl",
    merges_path: str = "tiny_stories_merges.pkl",
    train_dataset_path: str = "train_dataset.npy",
    val_dataset_path: str = "val_dataset.npy"
):
    """
    Load precomputed vocab, merges, and datasets.
    
    Args:
        vocab_path: Path to the saved vocabulary
        merges_path: Path to the saved merges
        train_dataset_path: Path to the saved training dataset
        val_dataset_path: Path to the saved validation dataset
    
    Returns:
        A tuple of (tokenizer, train_data, val_data)
    """
    # Load vocab and merges
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(f"Vocabulary or merges file not found at {vocab_path} or {merges_path}")
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)
    
    tokenizer = BPETokenizer(vocab, merges)
    
    # Load datasets
    if not os.path.exists(train_dataset_path):
        raise FileNotFoundError(f"Training dataset not found at {train_dataset_path}")
    
    train_data = np.memmap(train_dataset_path, dtype=np.uint16, mode='r')
    
    val_data = None
    if os.path.exists(val_dataset_path):
        val_data = np.memmap(val_dataset_path, dtype=np.uint16, mode='r')
        
    
    return tokenizer, train_data, val_data
