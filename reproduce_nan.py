
import torch
import numpy as np
import os
from config import TrainConfig
from cs336_basics.transformer import TransformerLM
from cs336_basics.training_components import (
    cross_entropy_loss,
    AdamW,
    cosine_schedule,
    data_loader as data_loader_fn
)

def reproduce():
    # Create a dummy config with small batch size but same model params to reproduce instability
    cfg = TrainConfig()
    cfg.max_iters = 20
    cfg.eval_interval = 5
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {cfg.device}")

    # Create dummy data if real data not found
    data_dir = os.path.dirname(cfg.input_path)
    train_path = os.path.join(data_dir, 'train_dataset.npy')
    
    if not os.path.exists(train_path):
        print("Train data not found, creating dummy data...")
        os.makedirs(data_dir, exist_ok=True)
        # Create random tokens
        dummy_data = np.random.randint(0, cfg.vocab_size, size=(10000,), dtype=np.uint16)
        np.save(train_path, dummy_data)
        np.save(os.path.join(data_dir, 'val_dataset.npy'), dummy_data)
        
        # Mock vocab and merges
        import pickle
        with open(os.path.join(data_dir, 'tiny_stories_vocab.pkl'), 'wb') as f:
            pickle.dump({i: str(i) for i in range(cfg.vocab_size)}, f)
        with open(os.path.join(data_dir, 'tiny_stories_merges.pkl'), 'wb') as f:
            pickle.dump([], f)

    # Load data
    train_data = np.load(train_path, mmap_mode='r')
    
    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        theta=cfg.rope_theta,
        max_seq_len=cfg.context_length,
    ).to(cfg.device)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        betas=(0.9, 0.95), 
        eps=1e-8, 
        weight_decay=cfg.weight_decay
    )

    iter_num = 0
    data_indices = np.arange(len(train_data) - cfg.context_length)
    
    print("Starting training loop...")
    for i in range(0, len(data_indices), cfg.batch_size):
        if iter_num >= cfg.max_iters:
            break
            
        batch_indices = data_indices[i:i + cfg.batch_size]
        
        lr = cosine_schedule(iter_num, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        x, y = data_loader_fn(train_data, cfg.batch_size, cfg.context_length, batch_indices, cfg.device)

        logits = model(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = cross_entropy_loss(logits, y)
        
        print(f"Iter {iter_num}, Loss: {loss.item()}")
        
        if torch.isnan(loss):
            print("NaN loss detected!")
            return
            
        optimizer.zero_grad()
        loss.backward()
        # Missing gradient clipping here!
        optimizer.step()
        
        iter_num += 1

if __name__ == "__main__":
    reproduce()
