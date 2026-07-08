"""Slab-safe Ewald summation for the Gaussian-screened Coulomb tail (M2).

The cluster tail uses the kernel erf(r / (sqrt(2) sigma)) / r (interaction of
two Gaussian charges).  The periodic version sums that kernel over all images:

  E = 0.5 sum_{i != j or n != 0}  q_i q_j erf(|r_ij + nL| / (sqrt2 sigma)) / |.|

computed by standard point-charge Ewald (tinfoil) plus a short-ranged smearing
correction (erf/r = 1/r - erfc/r, and erfc(r / (sqrt2 sigma)) dies within a few
sigma, so it is a minimum-image sum), plus the Yeh-Berkowitz dipole correction
that converts tinfoil 3D Ewald into the correct slab limit (vacuum along z).

Orthorhombic (diagonal) cells only, matching the rest of the code.  Energies
are returned in e^2/Angstrom units; multiply by KE for eV.  Everything is
differentiable (forces via autograd through positions).
"""
import numpy as np
import torch


def _k_setup(L, alpha, device, dtype, kmax_sigma=3.5):
    """Reciprocal vectors and coefficients for a diagonal cell (3,) tensor.
    kmax chosen so exp(-k^2/(4 a^2)) <= exp(-kmax_sigma^2)."""
    kmax = 2.0 * alpha * kmax_sigma
    nmax = torch.ceil(kmax * L / (2 * np.pi)).long()
    ax = [torch.arange(-int(n), int(n) + 1, device=device) for n in nmax]
    n = torch.stack(torch.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    n = n[(n != 0).any(1)]
    k = 2 * np.pi * n.to(dtype) / L
    k2 = (k * k).sum(1)
    coeff = torch.exp(-k2 / (4 * alpha ** 2)) / k2
    keep = coeff > coeff.max() * 1e-12
    return k[keep], coeff[keep]


def ewald_energy(q, pos, cell, sigma=1.0, alpha=None, yb=True):
    """q (B,N) masked charges (0 on pads), pos (B,N,3), cell (B,3,3) diagonal.
    All cells in the batch must be identical (true for the AuMgO dataset)."""
    L = torch.diagonal(cell, dim1=-2, dim2=-1)
    if not torch.allclose(L, L[:1], atol=1e-6):
        raise ValueError('ewald_energy assumes a shared cell across the batch')
    L0 = L[0]
    V = L0.prod()
    if alpha is None:
        # erfc(alpha * min(Lx,Ly)/2) ~ erfc(3.5) ~ 7e-7: real-space sum is
        # minimum-image exact at this accuracy
        alpha = float(7.0 / L0.min())

    # real space: [erfc(alpha r) - erfc(r / (sqrt2 sigma))] / r in one pass
    # (the second term converts the point-charge kernel to the Gaussian one)
    dx = pos[:, :, None, :] - pos[:, None, :, :]
    dx = dx - L0 * torch.round(dx / L0)
    r = torch.sqrt((dx * dx).sum(-1) + 1e-12)
    N = r.shape[1]
    eye = torch.eye(N, device=r.device, dtype=torch.bool)[None]
    kern = (torch.erfc(alpha * r) - torch.erfc(r / (np.sqrt(2.0) * sigma))) / r
    kern = kern.masked_fill(eye, 0.0)
    E_real = 0.5 * (q[:, :, None] * q[:, None, :] * kern).sum((1, 2))

    # reciprocal space
    k, coeff = _k_setup(L0, alpha, pos.device, pos.dtype)
    phase = pos @ k.T                                        # (B,N,K)
    re = (q[..., None] * torch.cos(phase)).sum(1)            # (B,K)
    im = (q[..., None] * torch.sin(phase)).sum(1)
    E_recip = (2 * np.pi / V) * (coeff * (re * re + im * im)).sum(-1)

    # self, net-charge background, slab dipole (Yeh-Berkowitz)
    Qt = q.sum(1)
    E_self = -alpha / np.sqrt(np.pi) * (q * q).sum(1)
    E_bg = -np.pi / (2 * V * alpha ** 2) * Qt * Qt
    E = E_real + E_recip + E_self + E_bg
    if yb:
        Mz = (q * pos[..., 2]).sum(1)
        E = E + (2 * np.pi / V) * Mz * Mz
    return E
