import torch
from torch import nn
import math
from einops import rearrange, einsum

from .linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.device | None = None):
        super().__init__()
        self.dtype = dtype
        self.device = device if device is not None else torch.device('cpu')
        self.d_model = d_model
        
        # d_ff =  8 / 3 * d_model
        self.d_ff =  d_ff
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = nn.Parameter(self._initialize_weights(d_model, d_ff, factory_kwargs))
        self.w2 = nn.Parameter(self._initialize_weights(d_ff, d_model, factory_kwargs))
        self.w3 = nn.Parameter(self._initialize_weights(d_model, d_ff, factory_kwargs))

    def _initialize_weights(self, in_features: int, out_features:int, factory_kwargs: dict=None) -> torch.Tensor:
        weight =torch.empty(out_features, in_features, **factory_kwargs)
        std = math.sqrt(2.0 / (in_features + out_features))
        bound = 3 * std
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-bound, b=bound)
        
        return weight
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_input =  x @ self.w1.T
        silu_output = silu_input * torch.sigmoid(silu_input)
        
        return (silu_output * (x @ self.w3.T)) @ self.w2.T
    

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device('cpu')
        self.build_cache()
        
    def build_cache(self):
        theta = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device).float() / self.d_k))
        index = torch.arange(self.max_seq_len, device=self.device)
        index_theta = einsum(index, theta, 'i , j -> i j')

        cos = torch.cos(index_theta).to(self.device)
        sin = torch.sin(index_theta).to(self.device)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype)  # (seq_len, d_k/2)
        sin = self.sin[token_positions].to(dtype=x.dtype)  # (seq_len, d_k/2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotate_even = x_even * cos - x_odd * sin
        x_rotate_odd = x_even * sin + x_odd * cos

        x_rot = torch.stack((x_rotate_even, x_rotate_odd), dim=-1)
        x_rot = rearrange(x_rot, "... d_2 two -> ... (d_2 two)")
        
        return x_rot


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True) 

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    dk = Q.shape[-1]
    
    qk = einsum(Q, K.transpose(-2,-1), "... q_sq_len dk, ... dk k_sq_len -> ... q_sq_len k_sq_len")
    qk = qk/math.sqrt(dk)
    
    if mask is not None:
        qk = qk.masked_fill(~mask, -torch.inf)
    
    attn = softmax(qk)

    return einsum(attn, V, "... q_seq_len k_seq_len, ... k_seq_len d_v -> ... q_seq_len d_v")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, theta: float = 10000, max_seq_len: int = 2048):
        super().__init__()
        self.h_dim = d_model // num_heads
        self.n_head = num_heads
        self.qkv = Linear(d_model, d_model*3)
        self.out_proj = Linear(d_model, d_model)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(theta=theta, d_k=self.h_dim, max_seq_len=max_seq_len)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        qkv_out = self.qkv(x)
        q, k, v = rearrange(qkv_out, "b t (three h d) -> three b h t d", three=3, h=self.n_head).unbind(dim=0)
        mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], device=x.device)).bool()
        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        multiheaded_attn = scaled_dot_product_attention(q, k, v, mask)
        multiheaded_attn = rearrange(multiheaded_attn, "b h t d -> b t (h d)")

        return self.out_proj(multiheaded_attn)
