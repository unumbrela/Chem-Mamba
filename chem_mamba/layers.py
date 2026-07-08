"""Selective state-space (Mamba S6) layers -- the O(N) global mixer.

Operates purely on INVARIANT scalar features, so the surrounding model stays
rotation/translation invariant and forces (via autograd) stay equivariant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class S6(nn.Module):
    """One directional selective scan: diagonal real A<0, input-dependent
    Delta/B/C.  Sequential scan (fine for the sizes here)."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):                                   # (B,T,d_model)
        B_, T, _ = x.shape
        A = -torch.exp(self.A_log)
        delta = F.softplus(self.dt_proj(x))
        Bmat = self.B_proj(x)
        Cmat = self.C_proj(x)
        Abar = torch.exp(delta[..., None] * A[None, None])            # (B,T,d,s)
        Bx = (delta * x)[..., None] * Bmat[:, :, None, :]            # (B,T,d,s)
        h = x.new_zeros(B_, self.d_model, self.d_state)
        ys = []
        for t in range(T):
            h = Abar[:, t] * h + Bx[:, t]
            ys.append((h * Cmat[:, t, None, :]).sum(-1))
        return torch.stack(ys, dim=1) + self.D * x


class BiMambaBlock(nn.Module):
    """Bidirectional selective SSM block, gated, pre-norm residual."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.fwd = S6(d_model, d_state)
        self.bwd = S6(d_model, d_state)
        self.out_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, u):
        x, gate = self.in_proj(self.norm(u)).chunk(2, dim=-1)
        x = F.silu(x)
        y = self.out_proj(torch.cat([self.fwd(x), self.bwd(x.flip(1)).flip(1)], -1))
        return u + y * F.silu(gate)
