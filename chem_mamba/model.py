"""
ChemMamba M1 model = short-range invariant backbone (SchNet-style) + optional
SSM long-range module, with energy and forces (autograd).

The A/B switch `use_ssm`:
  use_ssm=False : latent charges are a LOCAL function of backbone features
                  (LES-style local charge baseline).
  use_ssm=True  : latent charges see GLOBAL context via a bidirectional
                  selective SSM ordered by a rotation-invariant key.
Both feed the SAME physics Coulomb tail, so the ONLY difference is locality of
the charge assignment -- isolating the paper's central claim on a real backbone.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BiMambaBlock

KCOUL_MODEL = 1.5   # physics tail strength (matches the reference's Coulomb kernel)


def mlp(*dims, act=nn.SiLU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Interaction(nn.Module):
    """SchNet-style continuous-filter convolution (invariant)."""

    def __init__(self, d, n_rbf):
        super().__init__()
        self.filter = mlp(n_rbf, d, d)
        self.pre = nn.Linear(d, d)
        self.post = mlp(d, d, d)

    def forward(self, h, rbf, env, mask):
        W = self.filter(rbf) * env[..., None]              # (B,N,N,d) filters
        msg = W * self.pre(h)[:, None, :, :]               # from j to i
        agg = (msg * mask[..., None]).sum(2)               # (B,N,d)
        return h + self.post(agg)


class Backbone(nn.Module):
    def __init__(self, n_species=2, d=64, n_layers=2, cutoff=2.2, n_rbf=16):
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(n_species, d)
        self.register_buffer('centers', torch.linspace(0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2 * 0.5
        self.blocks = nn.ModuleList([Interaction(d, n_rbf) for _ in range(n_layers)])

    def forward(self, positions, species):
        dx = positions[:, :, None, :] - positions[:, None, :, :]
        r = dx.norm(dim=-1)
        N = r.shape[1]
        eye = torch.eye(N, device=r.device, dtype=torch.bool)[None]
        mask = (r < self.cutoff) & (~eye)
        rbf = torch.exp(-self.gamma * (r[..., None] - self.centers) ** 2)
        env = torch.where(r < self.cutoff,
                          0.5 * (torch.cos(np.pi * r / self.cutoff) + 1), torch.zeros_like(r))
        h = self.embed(species)
        for blk in self.blocks:
            h = blk(h, rbf, env, mask)
        return h


class LongRangeSSM(nn.Module):
    """Global context via bidirectional SSM over a rotation-invariant ordering."""

    def __init__(self, d, d_state=16, n_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([BiMambaBlock(d, d_state) for _ in range(n_layers)])

    def forward(self, h, positions, mix=True):
        B, N, d = h.shape
        if not mix:
            # CAPACITY-MATCHED CONTROL: identical parameters, but each atom is
            # processed in isolation (length-1 sequence) -> zero cross-atom reach.
            hs = h.reshape(B * N, 1, d)
            for blk in self.blocks:
                hs = blk(hs)
            return hs.reshape(B, N, d)
        key = (positions - positions.mean(1, keepdim=True)).norm(dim=-1)  # invariant
        order = key.argsort(dim=1)
        inv = order.argsort(dim=1)
        hs = torch.gather(h, 1, order[..., None].expand(-1, -1, d))
        for blk in self.blocks:
            hs = blk(hs)
        return torch.gather(hs, 1, inv[..., None].expand(-1, -1, d))       # back to atom order


class ChemMamba(nn.Module):
    def __init__(self, use_ssm=True, d=64, mix=True, **kw):
        super().__init__()
        self.use_ssm = use_ssm
        self.mix = mix                                     # False = capacity-matched local control
        self.backbone = Backbone(d=d, **kw)
        self.short_head = mlp(d, d, 1)
        self.ssm = LongRangeSSM(d) if use_ssm else None
        self.charge_head = mlp(d, d, 1)                    # feeds physics Coulomb tail
        self.res_head = mlp(d, d, 1)                       # non-electrostatic residual

    def energy(self, positions, species):
        h = self.backbone(positions, species)
        E_short = self.short_head(h).squeeze(-1).sum(1)

        feat = self.ssm(h, positions, self.mix) if self.use_ssm else h  # global vs local features
        q = self.charge_head(feat).squeeze(-1)
        q = q - q.mean(1, keepdim=True)                    # neutral clusters (sum q = 0)
        r = (positions[:, :, None, :] - positions[:, None, :, :]).norm(dim=-1)
        N = r.shape[1]
        gamma = KCOUL_MODEL / torch.sqrt(r * r + 1.0)
        gamma = gamma * (1 - torch.eye(N, device=r.device)[None])
        E_long = 0.5 * (q[:, :, None] * q[:, None, :] * gamma).sum((1, 2))

        E_res = self.res_head(feat).squeeze(-1).sum(1)
        return E_short + E_long + E_res

    def forward(self, positions, species):
        with torch.enable_grad():                          # robust to an outer no_grad
            positions = positions.clone().requires_grad_(True)
            E = self.energy(positions, species)
            F_pred = -torch.autograd.grad(E.sum(), positions,
                                          create_graph=self.training)[0]
        if not self.training:
            E, F_pred = E.detach(), F_pred.detach()
        return E, F_pred
