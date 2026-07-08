"""
Three per-site charge-prediction models that share an identical featureiser and
readout, differing ONLY in how information is mixed across sites:

  LocalDeepSet   : each site sees a hard cutoff neighbourhood only  (local GNN /
                   LES-style *local* latent charges).  O(N).  Non-local blind.
  SelectiveSSM   : bidirectional diagonal selective state-space scan (Mamba S6
                   core).  O(N).  Global receptive field.        <-- our method
  TransformerNet : full self-attention.  O(N^2).  Global receptive field.
                   Included as an accuracy upper bound.

All predict per-site charges; a charge-conservation projection enforces
sum_i q_i = Q_tot for every model (standard practice, applied uniformly).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def site_features(positions, chi, J):
    """Per-site raw features: (B, N, 2) = [chi, J].  Position handled per-model."""
    return torch.stack([chi, J], dim=-1)


def enforce_conservation(q, Q):
    """Project predicted charges so sum_i q_i == Q (per sample)."""
    correction = (Q - q.sum(dim=1)) / q.shape[1]
    return q + correction[:, None]


# --------------------------------------------------------------------------- #
# Local baseline: DeepSet over a hard cutoff neighbourhood
# --------------------------------------------------------------------------- #
class LocalDeepSet(nn.Module):
    def __init__(self, d_model=64, cutoff=2.5):
        super().__init__()
        self.cutoff = cutoff
        self.msg = nn.Sequential(         # encodes a (dx, chi_j, J_j) neighbour
            nn.Linear(3, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model), nn.SiLU(),
        )
        self.self_enc = nn.Sequential(    # encodes own (chi_i, J_i)
            nn.Linear(2, d_model), nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, positions, chi, J, Q):
        B, N = chi.shape
        dx = positions[:, :, None] - positions[:, None, :]          # (B,N,N)
        mask = (dx.abs() <= self.cutoff) & (dx.abs() > 1e-6)        # exclude self
        neigh = torch.stack([dx, chi[:, None, :].expand(B, N, N),
                             J[:, None, :].expand(B, N, N)], dim=-1)  # (B,N,N,3)
        m = self.msg(neigh) * mask[..., None]                        # zero non-neighbours
        agg = m.sum(dim=2)                                           # (B,N,d) permutation-invariant
        own = self.self_enc(torch.stack([chi, J], dim=-1))          # (B,N,d)
        return self.readout(torch.cat([agg, own], dim=-1)).squeeze(-1)  # (B,N) RAW


# --------------------------------------------------------------------------- #
# Selective state-space (Mamba S6) core -- diagonal, real, input-dependent
# --------------------------------------------------------------------------- #
class S6(nn.Module):
    """One directional selective scan.  Faithful diagonal S6: input-dependent
    Delta, B, C; learned diagonal A<0.  Sequential scan (fine for PoC sizes)."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))          # A = -exp(A_log)  (d_model,d_state)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):                                # x: (B,T,d_model)
        B_, T, _ = x.shape
        A = -torch.exp(self.A_log)                       # (d_model,d_state)
        delta = F.softplus(self.dt_proj(x))              # (B,T,d_model)  > 0
        Bmat = self.B_proj(x)                            # (B,T,d_state)
        Cmat = self.C_proj(x)                            # (B,T,d_state)
        # discretise (zero-order hold): Abar=exp(delta*A), Bbar x = delta*x (x) B
        Abar = torch.exp(delta[..., None] * A[None, None])          # (B,T,d_model,d_state)
        Bx = (delta * x)[..., None] * Bmat[:, :, None, :]           # (B,T,d_model,d_state)
        h = x.new_zeros(B_, self.d_model, self.d_state)
        ys = []
        for t in range(T):
            h = Abar[:, t] * h + Bx[:, t]                           # (B,d_model,d_state)
            ys.append((h * Cmat[:, t, None, :]).sum(-1))           # (B,d_model)
        y = torch.stack(ys, dim=1)                                 # (B,T,d_model)
        return y + self.D * x


class BiMambaBlock(nn.Module):
    """Bidirectional selective SSM block with gated MLP, pre-norm residual."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)   # x path + gate
        self.fwd = S6(d_model, d_state)
        self.bwd = S6(d_model, d_state)
        self.out_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, u):
        x, gate = self.in_proj(self.norm(u)).chunk(2, dim=-1)
        x = F.silu(x)
        y_f = self.fwd(x)
        y_b = self.bwd(x.flip(1)).flip(1)
        y = self.out_proj(torch.cat([y_f, y_b], dim=-1)) * F.silu(gate)
        return u + y


class SelectiveSSMNet(nn.Module):
    def __init__(self, d_model=64, d_state=16, n_layers=3, cutoff=None):
        super().__init__()
        self.embed = nn.Linear(3, d_model)               # [chi, J, position]
        self.blocks = nn.ModuleList(
            [BiMambaBlock(d_model, d_state) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 1))

    def forward(self, positions, chi, J, Q):
        pos = (positions - positions.mean(1, keepdim=True)) / (positions.shape[1])
        h = self.embed(torch.stack([chi, J, pos], dim=-1))
        for blk in self.blocks:
            h = blk(h)
        return self.readout(h).squeeze(-1)   # (B,N) RAW; conservation applied by caller


# --------------------------------------------------------------------------- #
# Transformer: full attention, O(N^2), accuracy upper bound
# --------------------------------------------------------------------------- #
def sinusoidal_pe(n, d, device):
    pos = torch.arange(n, device=device).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-np.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerNet(nn.Module):
    def __init__(self, d_model=64, n_layers=3, n_heads=4, cutoff=None):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model,
            activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 1))

    def forward(self, positions, chi, J, Q):
        pos = (positions - positions.mean(1, keepdim=True)) / (positions.shape[1])
        h = self.embed(torch.stack([chi, J, pos], dim=-1))
        h = h + sinusoidal_pe(h.shape[1], self.d_model, h.device)[None]
        h = self.enc(h)
        return self.readout(h).squeeze(-1)   # (B,N) RAW; conservation applied by caller


def build(name, cutoff):
    if name == 'local':
        return LocalDeepSet(cutoff=cutoff)
    if name == 'ssm':
        return SelectiveSSMNet()
    if name == 'transformer':
        return TransformerNet()
    raise ValueError(name)
