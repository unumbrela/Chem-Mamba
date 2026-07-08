"""MACE equivariant backbone behind the MaskedBackbone interface.

Dense padded batches (pos, species, mask, cell) -> flat graph -> MACE
interactions -> invariant (l=0) node features -> dense (B, N, d).

Everything downstream (SSM / local / attn charge heads, conservation
projection, Coulomb/Ewald tails) is untouched: this is the plug-and-play
backbone swap.  Edges use the same minimum-image convention (orthorhombic
diagonal cells) as model4g.pair_dist, so the receptive-field protocol is
identical across backbones.

Requires TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (e3nn 0.4.4 loads its Wigner
constants with torch.load under torch>=2.6).
"""
import os
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import numpy as np
import torch
import torch.nn as nn
from e3nn import o3

from mace.modules import interaction_classes
from mace.modules.models import MACE
from mace.modules.utils import get_edge_vectors_and_lengths


class MACEBackbone(nn.Module):
    def __init__(self, n_species, d=64, n_layers=2, cutoff=3.5,
                 atomic_numbers=None, hidden=64, max_ell=3, correlation=3,
                 avg_num_neighbors=10.0):
        super().__init__()
        self.n_species = n_species
        self.cutoff = cutoff
        zs = list(atomic_numbers) if atomic_numbers is not None \
            else list(range(1, n_species + 1))
        self.mace = MACE(
            r_max=cutoff, num_bessel=8, num_polynomial_cutoff=5,
            max_ell=max_ell,
            interaction_cls=interaction_classes['RealAgnosticResidualInteractionBlock'],
            interaction_cls_first=interaction_classes['RealAgnosticResidualInteractionBlock'],
            num_interactions=n_layers, num_elements=n_species,
            hidden_irreps=o3.Irreps(f'{hidden}x0e + {hidden}x1o'),
            MLP_irreps=o3.Irreps('16x0e'),
            atomic_energies=np.zeros(n_species),
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=zs, correlation=correlation,
            gate=torch.nn.functional.silu)
        # invariant (l=0) slice of each product-layer output; charges are
        # scalars, so only the invariant part feeds the charge channel
        self.scalar_slices = []
        n_scalar = 0
        for prod in self.mace.products:
            irr = prod.linear.irreps_out
            n_sc = sum(mul * ir.dim for mul, ir in irr if ir.l == 0)
            self.scalar_slices.append(n_sc)
            n_scalar += n_sc
        self.proj = nn.Linear(n_scalar, d)

    def forward(self, pos, species, mask, cell=None):
        B, N = species.shape
        d3 = pos.dtype
        flat_ok = mask.reshape(-1)
        dense2flat = torch.cumsum(flat_ok.long(), 0) - 1
        pos_flat = pos.reshape(-1, 3)[flat_ok]
        node_attrs = torch.nn.functional.one_hot(
            species.reshape(-1)[flat_ok], self.n_species).to(d3)

        # edges via dense pairwise distances (systems are <=110 atoms);
        # minimum image for orthorhombic cells, same as model4g.pair_dist
        dx = pos[:, :, None, :] - pos[:, None, :, :]
        shift_d = torch.zeros_like(dx)
        if cell is not None:
            L = torch.diagonal(cell, dim1=-2, dim2=-1)[:, None, None, :]
            shift_d = -L * torch.round(dx / L)
            dx = dx + shift_d
        r = dx.norm(dim=-1)
        eye = torch.eye(N, device=pos.device, dtype=torch.bool)[None]
        pair = mask[:, :, None] & mask[:, None, :] & (~eye) & (r < self.cutoff)
        b_idx, i_idx, j_idx = pair.nonzero(as_tuple=True)
        # MACE convention: vectors = pos[receiver] - pos[sender] + shifts;
        # dx[b,i,j] = pos_i - pos_j + shift_d[b,i,j] => sender=j, receiver=i
        edge_index = torch.stack([dense2flat[b_idx * N + j_idx],
                                  dense2flat[b_idx * N + i_idx]], 0)
        shifts = shift_d[b_idx, i_idx, j_idx].detach()

        vectors, lengths = get_edge_vectors_and_lengths(pos_flat, edge_index, shifts)
        m = self.mace
        node_feats = m.node_embedding(node_attrs)
        edge_attrs = m.spherical_harmonics(vectors)
        edge_feats, cutoff_w = m.radial_embedding(
            lengths, node_attrs, edge_index, m.atomic_numbers)
        scalars = []
        for i, (interaction, product) in enumerate(zip(m.interactions, m.products)):
            node_feats, sc = interaction(
                node_attrs=node_attrs, node_feats=node_feats,
                edge_attrs=edge_attrs, edge_feats=edge_feats,
                edge_index=edge_index, cutoff=cutoff_w, first_layer=(i == 0))
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            scalars.append(node_feats[:, :self.scalar_slices[i]])
        h_flat = self.proj(torch.cat(scalars, dim=-1))

        h = torch.zeros(B * N, h_flat.shape[-1], device=pos.device, dtype=d3)
        h[flat_ok] = h_flat
        return h.view(B, N, -1)
