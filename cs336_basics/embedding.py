import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
import math
from einops import rearrange, einsum
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.device | None = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        embedding_init = self._initialize_weights(num_embeddings, embedding_dim, factory_kwargs)
        self.embeddings = nn.Parameter(embedding_init)
        self.dtype = dtype
        self.device = device

    def _initialize_weights(self, num_embeddings: int, embedding_dim:int, factory_kwargs: dict=None) -> torch.Tensor:
        weight =torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        std = math.sqrt(2.0 / (num_embeddings + embedding_dim))
        bound = 3 * std
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-bound, b=bound)
        
        return weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
