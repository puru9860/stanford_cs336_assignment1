import torch
from torch import nn
import math
from einops import rearrange, einsum


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.device | None = None
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        weight_init = self._initialize_weights(self.d_model, factory_kwargs)
        self.gamma = nn.Parameter(weight_init)

    def _initialize_weights(self, d_model: int, factory_kwargs: dict = None) -> torch.Tensor:
        weight = torch.empty(d_model, **factory_kwargs)
        std = math.sqrt(2.0 / d_model)
        bound = 3 * std
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-bound, b=bound)

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        total = torch.sum(x ** 2 , dim=2)
        rms = torch.sqrt(((1 / self.d_model) * total) + self.eps)
        rms = rearrange(rms, "bs seq -> bs seq 1")
        result = (x / rms) * self.gamma

        return result.to(in_dtype)
