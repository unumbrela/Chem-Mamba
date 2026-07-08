"""ChemMamba for the real 4G-HDNNP charge-transfer benchmarks.

Extends the M1 model with: variable-size batches (padding + mask), total-charge
conditioning, Hirshfeld-charge supervision, and an erf-screened physical
Coulomb tail in real units (eV, Angstrom, e).

Model matrix (all share backbone + physics tail; only charge-assignment
locality and Q-awareness differ):
  local-Q : use_ssm=False, use_Q=False -> latent charge is a purely local
            function (structural stand-in for 3G-HDNNP / LES local charges).
  local+Q : use_ssm=False, use_Q=True  -> "just feed the total charge"
            cheap-global baseline (uniform conservation projection).
  ssm     : use_ssm=True,  use_Q=True, mix=True  -> ours: O(N) global context.
  ssm-iso : same params as ssm, mix=False -> capacity-matched control.
  ssm2    : ssm + per-layer summary token (gather globally -> redistribute
            locally; QEq-shaped inductive bias for charge routing).
  attn    : distance-biased full self-attention, O(N^2) -> nonlocal-reach
            upper reference (what unlimited pairwise reach buys).
  qeq     : local electronegativity + differentiable global charge
            equilibration solve, O(N^3) -> the 4G-HDNNP physics reference
            inside our exact framework (same backbone/splits/tail).
"""
import numpy as np
import torch
import torch.nn as nn

from .layers import BiMambaBlock
from .ewald import ewald_energy

KE = 14.399645  # Coulomb constant, eV * Angstrom / e^2


def mlp(*dims, act=nn.SiLU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def pair_dist(pos, cell=None):
    """Pairwise distances; minimum image for orthorhombic cells (diagonal)."""
    dx = pos[:, :, None, :] - pos[:, None, :, :]
    if cell is not None:
        L = torch.diagonal(cell, dim1=-2, dim2=-1)[:, None, None, :]
        dx = dx - L * torch.round(dx / L)
    return torch.sqrt((dx * dx).sum(-1) + 1e-12)           # eps: finite grads at r=0


class MaskedBackbone(nn.Module):
    """SchNet-style invariant message passing with padding masks."""

    def __init__(self, n_species, d=64, n_layers=2, cutoff=3.5, n_rbf=20):
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(n_species, d)
        self.register_buffer('centers', torch.linspace(0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2 * 0.5
        self.filters = nn.ModuleList([mlp(n_rbf, d, d) for _ in range(n_layers)])
        self.pres = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.posts = nn.ModuleList([mlp(d, d, d) for _ in range(n_layers)])

    def forward(self, pos, species, mask, cell=None):
        r = pair_dist(pos, cell)
        N = r.shape[1]
        eye = torch.eye(N, device=r.device, dtype=torch.bool)[None]
        pair = mask[:, :, None] & mask[:, None, :] & (~eye) & (r < self.cutoff)
        rbf = torch.exp(-self.gamma * (r[..., None] - self.centers) ** 2)
        env = 0.5 * (torch.cos(np.pi * torch.clamp(r / self.cutoff, max=1.0)) + 1)
        h = self.embed(species) * mask[..., None]
        for filt, pre, post in zip(self.filters, self.pres, self.posts):
            W = filt(rbf) * env[..., None] * pair[..., None]
            agg = (W * pre(h)[:, None, :, :]).sum(2)
            h = (h + post(agg)) * mask[..., None]
        return h


class MaskedSSM(nn.Module):
    """Bidirectional selective SSM over an invariant ordering; pads sort last
    and carry zero features, so they cannot inject state into real atoms.

    summary=True (ssm2): after each block, a masked-mean summary token is
    broadcast back to every atom (gather globally -> redistribute locally).
    This mirrors the two-phase structure of charge equilibration and makes
    global information available undecayed at every position."""

    def __init__(self, d, d_state=16, n_layers=2, summary=False):
        super().__init__()
        self.blocks = nn.ModuleList([BiMambaBlock(d, d_state) for _ in range(n_layers)])
        self.summary = summary
        if summary:
            self.sum_projs = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])

    def forward(self, h, pos, mask, mix=True, order='centroid'):
        B, N, d = h.shape
        h = h * mask[..., None]
        n = mask.sum(1, keepdim=True).clamp(min=1)
        if not mix:
            # capacity-matched control: identical params, length-1 sequences
            # (with summary=True the summary of a length-1 sequence is the atom
            # itself -> still zero cross-atom reach)
            hs = h.reshape(B * N, 1, d)
            for i, blk in enumerate(self.blocks):
                hs = blk(hs)
                if self.summary:
                    hs = hs + torch.nn.functional.silu(self.sum_projs[i](hs))
            return hs.reshape(B, N, d) * mask[..., None]
        if order == 'z':
            # depth ordering for slabs: PBC-safe (vacuum along z), orientation
            # fixed by the lattice; centroid-radial keys are ill-defined under PBC
            key = pos[..., 2]
        else:
            centroid = (pos * mask[..., None]).sum(1, keepdim=True) / n[..., None]
            key = (pos - centroid).norm(dim=-1)             # rotation-invariant
        key = torch.where(mask, key, torch.full_like(key, 1e9))
        srt = key.argsort(dim=1)
        inv = srt.argsort(dim=1)
        msk_s = torch.gather(mask, 1, srt)
        hs = torch.gather(h, 1, srt[..., None].expand(-1, -1, d))
        for i, blk in enumerate(self.blocks):
            hs = blk(hs) * msk_s[..., None]
            if self.summary:
                g = hs.sum(1, keepdim=True) / n[..., None]
                hs = hs + torch.nn.functional.silu(self.sum_projs[i](g)) * msk_s[..., None]
        return torch.gather(hs, 1, inv[..., None].expand(-1, -1, d)) * mask[..., None]


class MaskedAttention(nn.Module):
    """Distance-biased full self-attention: the O(N^2) nonlocal-reach upper
    reference.  Permutation equivariant (no ordering), rotation invariant
    (spatial information enters only through pairwise distances)."""

    def __init__(self, d, n_layers=2, n_heads=4, n_rbf=16, r_max=30.0):
        super().__init__()
        self.nh, self.dh = n_heads, d // n_heads
        self.register_buffer('centers', torch.linspace(0, r_max, n_rbf))
        self.gamma = (n_rbf / r_max) ** 2 * 0.5
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.qkvs = nn.ModuleList([nn.Linear(d, 3 * d) for _ in range(n_layers)])
        self.outs = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.biases = nn.ModuleList([mlp(n_rbf, 32, n_heads) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.mlps = nn.ModuleList([mlp(d, d, d) for _ in range(n_layers)])

    def forward(self, h, pos, mask, cell=None):
        B, N, d = h.shape
        r = pair_dist(pos, cell)
        rbf = torch.exp(-self.gamma * (r[..., None] - self.centers) ** 2)
        neg = torch.finfo(h.dtype).min
        key_mask = torch.where(mask, 0.0, neg)[:, None, None, :]   # (B,1,1,N)
        for ln, qkv, out, bias, ln2, ff in zip(
                self.norms, self.qkvs, self.outs, self.biases, self.norms2, self.mlps):
            x = ln(h)
            q, k, v = qkv(x).chunk(3, dim=-1)
            q = q.view(B, N, self.nh, self.dh).transpose(1, 2)      # (B,H,N,dh)
            k = k.view(B, N, self.nh, self.dh).transpose(1, 2)
            v = v.view(B, N, self.nh, self.dh).transpose(1, 2)
            logits = q @ k.transpose(-1, -2) / np.sqrt(self.dh)
            logits = logits + bias(rbf).permute(0, 3, 1, 2) + key_mask
            att = torch.softmax(logits, dim=-1)
            y = (att @ v).transpose(1, 2).reshape(B, N, d)
            h = h + out(y) * mask[..., None]
            h = h + ff(ln2(h)) * mask[..., None]
        return h * mask[..., None]


class ChemMamba4G(nn.Module):
    def __init__(self, n_species, d=64, n_layers=2, cutoff=3.5, n_rbf=20,
                 use_ssm=True, mix=True, use_Q=True, ssm_layers=2, d_state=16,
                 sigma=1.0, order='centroid', tail='coulomb', mixer=None,
                 summary=False, charge_mode='nn', detach_elec=False,
                 backbone='schnet', backbone_kw=None):
        super().__init__()
        self.use_ssm, self.mix, self.use_Q, self.sigma = use_ssm, mix, use_Q, sigma
        self.order = order
        self.tail = tail   # 'none': charges are a pure auxiliary head (diagnostic
                           # for physics-tail misspecification fighting the labels)
        self.detach_elec = detach_elec   # stop-gradient the charge->electrostatic
                           # path: charges still feed E_elec in the forward pass,
                           # but energy/force grads do NOT reach the charge head
                           # (trained on labels alone, exactly Test A's gradient).
                           # Isolates energy-charge gradient competition as the
                           # cause of SSM routing collapse under energy coupling.
        self.charge_mode = charge_mode   # 'nn' | 'qeq' (differentiable global solve)
        if backbone == 'mace':
            # equivariant MACE backbone (plug-and-play swap; lazy import keeps
            # mace-torch an optional dependency for the schnet path)
            from .mace_backbone import MACEBackbone
            self.backbone = MACEBackbone(n_species, d, n_layers, cutoff,
                                         **(backbone_kw or {}))
        else:
            self.backbone = MaskedBackbone(n_species, d, n_layers, cutoff, n_rbf)
        self.q_embed = nn.Linear(1, d) if use_Q else None
        self.ssm = MaskedSSM(d, d_state, ssm_layers, summary) if use_ssm else None
        self.attn = MaskedAttention(d) if mixer == 'attn' else None
        self.short_head = mlp(d, d, 1)
        self.charge_head = mlp(d, d, 1)  # 'nn': latent charge; 'qeq': electronegativity chi
        self.res_head = mlp(d, d, 1)
        if charge_mode == 'qeq':
            self.eta_head = mlp(d, d, 1)  # atomic hardness (softplus + floor)

    def qeq_solve(self, chi, eta, pos, mask, Q, cell=None):
        """Differentiable charge equilibration (the 4G physics reference):
        minimize sum_i(chi_i q_i + eta_i q_i^2 / 2) + off-diagonal Coulomb,
        subject to sum_i q_i = Q.  KKT system solved in float64; padded atoms
        get identity rows (q_pad = 0) and are excluded from the constraint."""
        B, N = chi.shape
        r = pair_dist(pos, cell)
        eye = torch.eye(N, device=r.device, dtype=torch.bool)[None]
        pair = mask[:, :, None] & mask[:, None, :] & (~eye)
        J = KE * torch.erf(r / (np.sqrt(2.0) * self.sigma)) / r * pair
        m = mask.double()
        diag = torch.where(mask, eta.double(), torch.ones_like(eta.double()))
        M = torch.zeros(B, N + 1, N + 1, dtype=torch.float64, device=chi.device)
        M[:, :N, :N] = J.double() + torch.diag_embed(diag)
        M[:, :N, N] = m
        M[:, N, :N] = m
        rhs = torch.zeros(B, N + 1, dtype=torch.float64, device=chi.device)
        rhs[:, :N] = -chi.double() * m
        rhs[:, N] = Q.double()
        sol = torch.linalg.solve(M, rhs[..., None]).squeeze(-1)
        return sol[:, :N].to(chi.dtype) * mask

    def energy_and_charges(self, pos, species, mask, Q, cell=None):
        h = self.backbone(pos, species, mask, cell)
        E_short = (self.short_head(h).squeeze(-1) * mask).sum(1)

        g = h + self.q_embed(Q[:, None, None]) if self.use_Q else h
        if self.use_ssm:
            feat = self.ssm(g, pos, mask, self.mix, self.order)
        elif self.attn is not None:
            feat = self.attn(g, pos, mask, cell)
        else:
            feat = g
        feat = feat * mask[..., None]

        E_onsite = 0.0
        if self.charge_mode == 'qeq':
            chi = self.charge_head(feat).squeeze(-1) * mask
            eta = (0.2 + torch.nn.functional.softplus(
                self.eta_head(feat).squeeze(-1))) * mask
            q = self.qeq_solve(chi, eta, pos, mask, Q, cell)
            # onsite QEq energy terms (chi q + eta q^2 / 2), part of the 4G form
            E_onsite = ((chi * q + 0.5 * eta * q * q) * mask).sum(1)
        else:
            q = self.charge_head(feat).squeeze(-1) * mask
            if self.use_Q:                                  # exact conservation
                n = mask.sum(1).clamp(min=1)
                q = q + mask * ((Q - q.sum(1)) / n)[:, None]

        # detach-elec (nn charge mode): the electrostatic energy uses the
        # charges in the forward pass but cannot backprop into the charge head,
        # so routing is trained by the labels alone (Test A's clean gradient)
        # while electrostatics stays physically live.  q (returned, supervised)
        # is unchanged; only the copy feeding E_elec is detached.
        q_elec = q.detach() if (self.detach_elec and self.charge_mode == 'nn') else q

        # Physics tail: erf-screened Coulomb.  'ewald' = full slab-safe sum
        # (M2: 3D Ewald + Yeh-Berkowitz dipole correction, same Gaussian
        # kernel); 'coulomb' = cluster sum / minimum image (correct for
        # clusters, misspecified for periodic slabs -- kept as the v1
        # reference); 'none' = charges become a pure auxiliary head.
        if self.tail == 'none':
            E_elec = torch.zeros_like(E_short)
        elif self.tail == 'ewald' and cell is not None:
            E_elec = KE * ewald_energy(q_elec, pos, cell, self.sigma)
        else:
            r = pair_dist(pos, cell)
            N = r.shape[1]
            eye = torch.eye(N, device=r.device, dtype=torch.bool)[None]
            pair = mask[:, :, None] & mask[:, None, :] & (~eye)
            kern = torch.erf(r / (np.sqrt(2.0) * self.sigma)) / r * pair
            E_elec = 0.5 * KE * (q_elec[:, :, None] * q_elec[:, None, :] * kern).sum((1, 2))

        E_res = (self.res_head(feat).squeeze(-1) * mask).sum(1)
        return E_short + E_elec + E_onsite + E_res, q

    def forward(self, pos, species, mask, Q, cell=None):
        with torch.enable_grad():
            pos = pos.clone().requires_grad_(True)
            E, q = self.energy_and_charges(pos, species, mask, Q, cell)
            F_pred = -torch.autograd.grad(E.sum(), pos, create_graph=self.training)[0]
        if not self.training:
            E, F_pred, q = E.detach(), F_pred.detach(), q.detach()
        return E, F_pred * mask[..., None], q


def make_model(name, n_species, **kw):
    """name in {local-Q, local+Q, ssm, ssm-iso, ssm2, attn, qeq}"""
    cfg = {
        'local-Q': dict(use_ssm=False, use_Q=False),
        'local+Q': dict(use_ssm=False, use_Q=True),
        'ssm':     dict(use_ssm=True, use_Q=True, mix=True),
        'ssm-iso': dict(use_ssm=True, use_Q=True, mix=False),
        'ssm2':    dict(use_ssm=True, use_Q=True, mix=True, summary=True),
        'attn':    dict(use_ssm=False, use_Q=True, mixer='attn'),
        # qeq: local chi/eta + global solve; Q enters only via the constraint,
        # exactly as in 4G-HDNNP (no Q embedding)
        'qeq':     dict(use_ssm=False, use_Q=False, charge_mode='qeq'),
    }[name]
    return ChemMamba4G(n_species, **cfg, **kw)
