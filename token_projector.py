"""TokenProjector: local + global residual heads (CVPR 2025-style).

Drop-in for MLPAligner: [B, T, 1024] -> [B, T, 2560].
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class _BranchHead(nn.Module):
    """output = Linear(x) + MLP(x)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
        self.linear = nn.Linear(input_dim, output_dim)
        mlp: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_cls(),
        ]
        if dropout:
            mlp.append(nn.Dropout(dropout))
        mlp.extend([nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim)])
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.mlp(x)


class TokenProjector(nn.Module):
    """Shared local head + separate global head; sequence length preserved.

    local = local_head(x)
    g = global_head(mean(x)) + mean(local)
    out = local + g  (broadcast over T)

    `branch`: "full" | "local" | "global" — eval ablations (train stays "full").
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 2560,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation.lower()
        self.branch = "full"
        self.local_head = _BranchHead(input_dim, output_dim, hidden_dim, self.activation, dropout)
        self.global_head = _BranchHead(input_dim, output_dim, hidden_dim, self.activation, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, X], got shape {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"last dim {x.size(-1)} != input_dim {self.input_dim}")

        local = self.local_head(x)
        if mask is None:
            pooled_in = x.mean(dim=1)
            pooled_local = local.mean(dim=1)
        else:
            m = mask.to(dtype=x.dtype).unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            pooled_in = (x * m).sum(dim=1) / denom
            pooled_local = (local * m).sum(dim=1) / denom

        g = self.global_head(pooled_in) + pooled_local
        if self.branch == "local":
            return local
        if self.branch == "global":
            return g.unsqueeze(1).expand_as(local)
        if self.branch != "full":
            raise ValueError(f"branch must be full|local|global, got {self.branch!r}")
        return local + g.unsqueeze(1)


# ponytail: shape + residual path sanity
if __name__ == "__main__":
    m = TokenProjector()
    t = torch.randn(2, 7, 1024)
    y = m(t)
    assert y.shape == (2, 7, 2560), y.shape
    m.branch = "local"
    assert m(t).shape == (2, 7, 2560)
    m.branch = "global"
    assert m(t).shape == (2, 7, 2560)
    print("ok", y.shape)
