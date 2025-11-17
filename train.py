import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Iterator, List, Dict, Tuple
import pickle
from pathlib import Path

from cs336_basics.transformer import TransformerLM
from cs336_basics.training_components import (
    cross_entropy_loss,
    AdamW,
    cosine_schedule,
    gradient_clipping,
    data_loader as data_loader_fn,
    save_checkpoint,
    load_checkpoint
)
from cs336_basics.preprocess import load_precomputed_data




def train_model(
    model: TransformerLM,
    train_data: np.ndarray,
    val_data: np.ndarray | None,
    num_epochs: int,
    context_length: int,
    batch_size: int,
    learning_rate: float = 1e-3,
    warmup_steps: int = 1000,
    total_steps: int | None = None,
    max_grad_norm: float = 1.0,
    device: torch.device = torch.device('cpu'),
    checkpoint_dir: str = "./checkpoints",
    save_every: int = 1000
):
    """
    Train the transformer model.

    Args:
        model: The transformer model to train
        train_data: Training data as numpy array of token IDs
        val_data: Validation data as numpy array of token IDs (optional)
        num_epochs: Number of epochs to train
        context_length: Length of context for each training example
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps (for learning rate scheduling)
        max_grad_norm: Maximum gradient norm for clipping
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N steps
    """
    # Move model to device
    model = model.to(device)

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Calculate total steps if not provided
    samples_per_epoch = len(train_data) - context_length
    steps_per_epoch = samples_per_epoch // batch_size
    if total_steps is None:
        total_steps = num_epochs * steps_per_epoch

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Use the data_loader function from training_components
        for batch_idx, (data, targets) in enumerate(data_loader_fn(train_data, batch_size, context_length, device)):
            # Forward pass
            logits = model(data)

            # Calculate loss
            # Reshape logits and targets for cross entropy
            logits = logits.view(-1, logits.size(-1))  # (batch_size * context_length, vocab_size)
            targets = targets.view(-1)  # (batch_size * context_length,)

            loss = cross_entropy_loss(logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            gradient_clipping(model.parameters(), max_grad_norm)

            # Update learning rate based on cosine schedule
            current_lr = cosine_schedule(
                step,
                warmup_steps,
                total_steps,
                min_lr=1e-6,
                max_lr=learning_rate
            )

            # Update the learning rate in optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Step
            optimizer.step()

            total_loss += loss.item()

            # Print progress
            if step % 100 == 0:
                print(f"Step {step}, Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

            # Validation
            if val_data is not None and step % 500 == 0:
                model.eval()
                val_loss = 0.0
                val_batches = 0

                # Create a small validation data loader
                val_loader = data_loader_fn(val_data[:min(len(val_data), 1000)], 4, context_length, device)

                with torch.no_grad():
                    for val_batch_idx, (val_data_batch, val_targets) in enumerate(val_loader):
                        val_logits = model(val_data_batch)
                        val_logits = val_logits.view(-1, val_logits.size(-1))
                        val_targets = val_targets.view(-1)

                        val_batch_loss = cross_entropy_loss(val_logits, val_targets)
                        val_loss += val_batch_loss.item()
                        val_batches += 1

                        if val_batches >= 10:  # Limit validation to save time
                            break

                if val_batches > 0:
                    val_loss /= val_batches
                    print(f"Validation Loss at step {step}: {val_loss:.4f}")

                model.train()

            # Save checkpoint
            if step % save_every == 0 and step > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
                save_checkpoint(model, optimizer, step, checkpoint_path)
                print(f"Saved checkpoint at step {step}")

            step += 1

            # Break if we've processed all data for this epoch
            if batch_idx >= steps_per_epoch - 1:
                break

        # Print epoch summary
        avg_loss = total_loss / min(steps_per_epoch, len(train_data) // batch_size)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_checkpoint(model, optimizer, step, final_checkpoint_path)
    print(f"Final model saved at {final_checkpoint_path}")

    return model


def main():
    """Main training function."""
    # Configuration
    d_model = 512
    num_heads = 8
    d_ff = int(8/3 * d_model)  # Following SwiGLU convention
    num_layers = 6
    theta = 10000
    max_seq_len = 512
    context_length = 256
    batch_size = 16
    num_epochs = 3
    learning_rate = 3e-4

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Load precomputed tokenizer and datasets
        print("Loading precomputed tokenizer and datasets...")
        tokenizer, train_data, val_data = load_precomputed_data(
            vocab_path="tiny_stories_vocab.pkl",
            merges_path="tiny_stories_merges.pkl",
            train_dataset_path="train_dataset.npy",
            val_dataset_path="val_dataset.npy"
        )

        vocab_size = len(tokenizer.vocab)
        print(f"Loaded tokenizer with vocab size: {vocab_size}")
        print(f"Training data shape: {train_data.shape}")
        if val_data is not None:
            print(f"Validation data shape: {val_data.shape}")

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

        # Train model
        trained_model = train_model(
            model=model,
            train_data=train_data,
            val_data=val_data,
            num_epochs=num_epochs,
            context_length=context_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            checkpoint_dir="./checkpoints"
        )

        print("Training completed!")

    except FileNotFoundError as e:
        print(f"Error loading precomputed data: {e}")
        print("Please run the preprocessing script first to generate the required files.")
        return


if __name__ == "__main__":
    main()
