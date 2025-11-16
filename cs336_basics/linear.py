import torch
from torch import nn
import math
from einops import rearrange, einsum
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features:int, device: torch.device | None = None, dtype: torch.device | None = None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        factory_kwargs = {"device": device, "dtype": dtype}
        weight_init = self._initialize_weights(in_features, out_features, factory_kwargs)
        self.weight = nn.Parameter(weight_init)

    def _initialize_weights(self, in_features: int, out_features:int, factory_kwargs: dict=None) -> torch.Tensor:
        weight =torch.empty(out_features, in_features, **factory_kwargs)
        std = math.sqrt(2.0 / (in_features + out_features))
        bound = 3 * std
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-bound, b=bound)
        
        return weight
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... i, o i -> ... o")
