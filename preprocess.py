import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Iterator, List, Dict, Tuple
import pickle
from pathlib import Path

from cs336_basics.bpe_tokenizer import train_bpe, BPETokenizer


def create_dataset_from_file(file_path: str, tokenizer: BPETokenizer):
    """Create a dataset from a text file by tokenizing the entire content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Encode the entire text
    encoded = tokenizer.encode(text)
    return np.array(encoded, dtype=np.int32)


def create_datasets_and_train_tokenizer(
    train_file_path: str,
    val_file_path: str | None,
    vocab_size: int = 10000,
    special_tokens: list[str] | None = None,
    output_vocab_path: str = "tiny_stories_vocab.pkl",
    output_merges_path: str = "tiny_stories_merges.pkl",
    output_train_dataset_path: str = "train_dataset.npy",
    output_val_dataset_path: str = "val_dataset.npy"
):
    """
    Train BPE tokenizer and create datasets from text files.

    Args:
        train_file_path: Path to the training text file
        val_file_path: Path to the validation text file (optional)
        vocab_size: Size of the vocabulary for BPE
        special_tokens: List of special tokens to include in vocabulary
        output_vocab_path: Path to save the vocabulary
        output_merges_path: Path to save the merges
        output_train_dataset_path: Path to save the training dataset
        output_val_dataset_path: Path to save the validation dataset
    """
    print("Starting BPE tokenizer training...")
    
    # Train BPE tokenizer
    vocab, merges = train_bpe(train_file_path, vocab_size, special_tokens or [])
    
    # Save the vocab and merges
    with open(output_vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(output_merges_path, 'wb') as f:
        pickle.dump(merges, f)
    
    print(f"Vocabulary and merges saved to {output_vocab_path} and {output_merges_path}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load the trained tokenizer
    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    
    # Create and save training dataset
    print("Creating training dataset...")
    train_data = create_dataset_from_file(train_file_path, tokenizer)
    np.save(output_train_dataset_path, train_data)
    print(f"Training dataset saved to {output_train_dataset_path} with shape {train_data.shape}")
    
    # Create and save validation dataset if provided
    if val_file_path and os.path.exists(val_file_path):
        print("Creating validation dataset...")
        val_data = create_dataset_from_file(val_file_path, tokenizer)
        np.save(output_val_dataset_path, val_data)
        print(f"Validation dataset saved to {output_val_dataset_path} with shape {val_data.shape}")
    else:
        print("No validation file provided or file does not exist")
        val_data = None
    
    return tokenizer, train_data, val_data


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
    
    train_data = np.load(train_dataset_path)
    
    val_data = None
    if os.path.exists(val_dataset_path):
        val_data = np.load(val_dataset_path)
    
    return tokenizer, train_data, val_data


def main():
    """Main preprocessing function."""
    # Configuration
    train_file = "data/TinyStoriesV2-GPT4-train.txt"
    val_file = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # Check if files exist before processing
    if not os.path.exists(train_file):
        print(f"Training file not found: {train_file}")
        return

    print("Training BPE tokenizer and creating datasets...")

    # Train tokenizer and create datasets
    tokenizer, train_data, val_data = create_datasets_and_train_tokenizer(
        train_file_path=train_file,
        val_file_path=val_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
