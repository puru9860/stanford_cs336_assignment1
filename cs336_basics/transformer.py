
import torch
from torch import nn

from .components import MultiHeadAttention, SwiGLU
from .linear import Linear
from .RMSNorm import RMSNorm
from .embedding import Embedding

class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.rms_norm_1 = RMSNorm(d_model)
        self.rms_norm_2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, use_rope=True, theta=theta, max_seq_len=max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1] 
        token_positions = torch.arange(seq_len, device=x.device)

        attn = self.attn(self.rms_norm_1(x), token_positions=token_positions)
        x = x + attn
        
        return x + self.ffn(self.rms_norm_2(x))

class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, num_layers:int, theta: float, max_seq_len: int):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            Transformer(d_model, num_heads, d_ff, theta, max_seq_len)
            for _ in range(num_layers)
        ])
        self.rms_norm = RMSNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.rms_norm(x)
        return self.output_linear(x)
