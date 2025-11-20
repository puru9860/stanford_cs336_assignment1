import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Iterator, List, Dict, Tuple
import pickle
from pathlib import Path
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.transformer import TransformerLM
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.training_components import (
    cross_entropy_loss,
    AdamW,
    cosine_schedule,
    gradient_clipping,
    data_loader as data_loader_fn,
    save_checkpoint,
    load_checkpoint
)
from config import TrainConfig


def train(
    model,
    cfg: TrainConfig
):
    device = cfg.device

    data_dir = os.path.dirname(cfg.input_path)
    train_path = os.path.join(data_dir, 'train_dataset.npy')
    val_path = os.path.join(data_dir, 'val_dataset.npy')
    vocab_path = os.path.join(data_dir, 'tiny_stories_vocab.pkl')
    merges_path = os.path.join(data_dir, 'tiny_stories_merges.pkl')

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        print(f"Vocab size: {len(vocab)}")
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)
    tokenizer = BPETokenizer(vocab, merges)
    
    train_data = np.load(train_path, mmap_mode='r')
    val_data = np.load(val_path, mmap_mode='r')

    writer = SummaryWriter(log_dir=cfg.log_dir)

    optimizer = AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        betas=(0.9, 0.95), 
        eps=1e-8, 
        weight_decay=cfg.weight_decay
    )
     
    iter_num = 0
    t0 = time.time()

    data_indices = np.arange(len(train_data) - cfg.context_length)
    # np.random.shuffle(data_indices)


    for epoch in range(cfg.epochs):
        total_loss = 0.0 
        pbar = tqdm(range(0, len(data_indices), cfg.batch_size), desc=f"Epoch {epoch}")
        for i in pbar:
            batch_indices = data_indices[i:i + cfg.batch_size]
            
            lr = cosine_schedule(iter_num, cfg.warmup_iters, cfg.max_iters, cfg.min_lr, cfg.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            x, y = data_loader_fn(train_data, cfg.batch_size, cfg.context_length, batch_indices, device)

            logits = model(x)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            loss = cross_entropy_loss(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

            if iter_num % cfg.log_interval == 0:
                writer.add_scalar("Loss/train", loss.item(), iter_num)
                writer.add_scalar("LearningRate", lr, iter_num)
            
            iter_num += 1
            
            if iter_num % cfg.eval_interval == 0:
                val_loss = evaluate(model, val_data, cfg.batch_size, cfg.context_length, device)
                model.train()
                tqdm.write(f"Epoch {epoch}, Iteration {iter_num}, Loss {total_loss / cfg.eval_interval}, Val Loss {val_loss}")
                writer.add_scalar("Loss/val", val_loss, iter_num)
            # else:
            #     print(f"Epoch {epoch}, Iteration {iter_num}, Loss {total_loss / cfg.eval_interval}")
            if iter_num % cfg.save_interval == 0:
                save_checkpoint(
                    cfg.checkpoint_dir,
                    iter_num,
                    model,
                    optimizer,
                    lr_scheduler,
                    iter_num,
                    total_loss / cfg.eval_interval
                )
    
    writer.close()
            

def evaluate(
    model,
    val_data,
    batch_size,
    context_length,
    device
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in tqdm(range(0, len(val_data) - context_length, batch_size), desc="Evaluating", leave=False):
            batch_indices = np.arange(i, i + batch_size)
            x, y = data_loader_fn(val_data, batch_size, context_length, batch_indices, device)
            logits = model(x)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            loss = cross_entropy_loss(logits, y)
            total_loss += loss.item()
    return total_loss / (len(val_data) - context_length) / batch_size


def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    device = cfg.device
    print(f"Using device: {device}")
    
    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        theta=cfg.rope_theta,
        max_seq_len=cfg.context_length,
    ).to(device)
     
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
     
    train(model, cfg)
            

if __name__ == "__main__":
    main()
