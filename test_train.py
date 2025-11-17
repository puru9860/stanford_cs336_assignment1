#!/usr/bin/env python3
"""
Test script to verify the training code works correctly.
"""

import os
import torch
import numpy as np
from cs336_basics.train import train_model, create_dataset_from_file
from cs336_basics.transformer import TransformerLM
from cs336_basics.bpe_tokenizer import BPETokenizer
import pickle

def test_training_without_data():
    """Test the training components by creating a minimal setup."""
    print("Testing training components...")
    
    # Configuration
    d_model = 128  # Smaller for testing
    num_heads = 4
    d_ff = int(8/3 * d_model)
    vocab_size = 1000  # Smaller for testing
    num_layers = 2
    theta = 10000
    max_seq_len = 128
    context_length = 64
    batch_size = 4
    num_epochs = 1
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple test tokenizer if files don't exist
    test_vocab = {i: bytes([i]) for i in range(256)}
    # Add some fake tokens to reach vocab_size
    for i in range(256, vocab_size):
        test_vocab[i] = f"token_{i}".encode("utf-8")
    
    # Create a simple tokenizer with test data
    tokenizer = BPETokenizer(test_vocab, [])
    
    # Create small dummy training and validation data
    print("Creating dummy training data...")
    train_data = np.random.randint(0, vocab_size, size=1000, dtype=np.int32)
    val_data = np.random.randint(0, vocab_size, size=200, dtype=np.int32)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    
    # Create model
    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads, 
        d_ff=d_ff,
        vocab_size=vocab_size,
        num_layers=num_layers,
        theta=theta,
        max_seq_len=max_seq_len
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model for just a few steps
    trained_model = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        context_length=context_length,
        batch_size=batch_size,
        learning_rate=3e-4,
        device=device,
        checkpoint_dir="./checkpoints",
        save_every=10  # Save more frequently for testing
    )
    
    print("Training test completed successfully!")
    return trained_model

if __name__ == "__main__":
    test_training_without_data()