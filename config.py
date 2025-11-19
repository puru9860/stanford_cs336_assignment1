
# config.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainConfig:
    input_path: str = "data/tinystories/" 
    output_dir: str = "out/tinystories_run"
    
    vocab_size: int = 10000  
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4    
    num_heads: int = 16
    d_ff: int = 1344        
    rope_theta: float = 10000.0

    batch_size: int = 16
    epochs: int = 10
    max_iters: int = 5000    
    eval_interval: int = 200
    save_interval: int = 1000
    
    # --- Optimizer Params ---
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # --- System ---
    device: str = "cuda"  # 'cuda', 'mps', or 'cpu'
    # compile: bool = True  # torch.compile (only for Linux/CUDA usually)
    # use_wandb: bool = True
    # wandb_project: str = "cs336-assignment1"
