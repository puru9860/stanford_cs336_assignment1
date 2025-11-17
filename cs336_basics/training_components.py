from torch import nn
import torch
from collections.abc import Callable, Iterable
import os
import math
import numpy as np
from typing import IO, Any, BinaryIO, Optional

# from scratch components
def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # loss = log(sum(exp(shifted_logits))) - shifted_logit_of_the_correct_word
    shifted_logits = logits - logits.max(dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    
    target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_exp - target_logits
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                t = state.get('t', 0)
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                
                lr_t = lr * (math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1)))
                
                p.data = p.data - lr_t * m / (torch.sqrt(v) + eps)
                p.data = p.data - lr * weight_decay * p.data
                
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1

def cosine_schedule(
    current_step: int,
    warmup_steps: int,
    annealing_steps: int,
    min_lr: float,
    max_lr: float,
) -> float:
    if current_step < warmup_steps:
        return current_step / warmup_steps * max_lr
    elif current_step < annealing_steps:
        progress = (current_step - warmup_steps) / (annealing_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cosine_decay
    else:
        return min_lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

def data_loader(
    dataset: np.typing.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device,
):
    num_samples = len(dataset) - context_length
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        x_batch = []
        y_batch = []
        for idx in batch_indices:
            x = dataset[idx:idx + context_length]
            y = dataset[idx + 1:idx + context_length + 1]
            x_batch.append(x)
            y_batch.append(y)

        # Convert to numpy arrays first for efficiency, then to tensors
        x_tensor = torch.from_numpy(np.array(x_batch, dtype=np.int64)).to(device)
        y_tensor = torch.from_numpy(np.array(y_batch, dtype=np.int64)).to(device)

        yield x_tensor, y_tensor


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    filepath: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, filepath)
    
def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration
