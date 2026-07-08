"""
Self-contained 3D atomic dataset with a DIFFERENTIABLE reference potential.

Real 3D geometries (jittered clusters, full SO(3) x translation x permutation
symmetry) with a reference energy = short-range Lennard-Jones  +  (optional)
NON-LOCAL screened Coulomb, whose per-atom charges come from a differentiable
charge-equilibration (QEq) solve.  Forces are autograd of this reference, so
(E, F) pairs are exactly consistent.

`long_range=False`  -> pure short-range reference (for the NO-REGRESSION test).
`long_range=True`   -> adds the non-local Coulomb term (for LONG-RANGE-HELPS).

Swapping this for rMD17/SPICE + a MACE backbone is a config change; here we keep
it self-contained and fast so M1 verifies the *integration*, not SOTA numbers.
"""
import numpy as np
import torch

# two species: (chi, J, eps, sigma)
SPECIES = torch.tensor([[0.8, 1.8, 1.0, 1.0],
                        [-0.8, 2.0, 1.0, 1.0]])
LJ_CUTOFF = 3.0
KCOUL = 1.5        # Coulomb strength (Bjerrum-like); makes the non-local term sizeable


def sample_positions(n_cfg, N, rng, spacing=1.5, jitter=0.12):
    """Jittered ELONGATED (2 x 2 x L) lattice -> long-range along the rod genuinely
    exceeds a local backbone's receptive field, exposing non-locality; LJ stays
    well-behaved (no singularities)."""
    L = int(np.ceil(N / 4))
    grid = np.array([[x, y, z] for z in range(L) for x in range(2)
                     for y in range(2)], dtype=float)[:N] * spacing
    pos = grid[None] + rng.normal(0, jitter, size=(n_cfg, N, 3))
    pos = pos - pos.mean(1, keepdims=True)
    return pos.astype(np.float32)


def reference(positions, species, long_range=True):
    """positions (B,N,3) tensor; species (B,N) long. Returns E (B,), F (B,N,3)."""
    positions = positions.clone().requires_grad_(True)
    chi = SPECIES[species, 0]; J = SPECIES[species, 1]
    eps = SPECIES[species, 2]; sig = SPECIES[species, 3]
    B, N, _ = positions.shape
    dx = positions[:, :, None, :] - positions[:, None, :, :]
    r = dx.norm(dim=-1) + torch.eye(N, device=positions.device)[None] * 1e9  # self -> inf

    # short-range Lennard-Jones, truncated at cutoff, radius floored for stability
    sij = 0.5 * (sig[:, :, None] + sig[:, None, :])
    eij = torch.sqrt(eps[:, :, None] * eps[:, None, :])
    sr = (sij / r.clamp_min(0.85)) ** 6
    lj = 4 * eij * (sr * sr - sr)
    lj = torch.where(r <= LJ_CUTOFF, lj, torch.zeros_like(lj))
    E = 0.5 * lj.sum((1, 2))

    if long_range:
        gamma = KCOUL / torch.sqrt(r * r + 1.0)             # screened Coulomb, all pairs
        gamma = torch.where(r > 1e6, torch.zeros_like(gamma), gamma)
        A = gamma + torch.diag_embed(J)                     # (B,N,N)
        # QEq: solve [[A,-1],[1,0]][q;lam]=[-chi;0]  (neutral clusters, Q=0)
        M = torch.zeros(B, N + 1, N + 1, device=positions.device)
        M[:, :N, :N] = A; M[:, :N, N] = -1; M[:, N, :N] = 1
        rhs = torch.cat([-chi, torch.zeros(B, 1, device=positions.device)], 1)
        q = torch.linalg.solve(M, rhs)[:, :N]               # (B,N) non-local charges
        E_long = 0.5 * (q[:, :, None] * q[:, None, :] * gamma).sum((1, 2)) \
                 + (chi * q).sum(1) + 0.5 * (J * q * q).sum(1)
        E = E + E_long

    F = -torch.autograd.grad(E.sum(), positions)[0]
    return E.detach(), F.detach()


def make_dataset(n_cfg, N, seed, long_range=True, n_species=2):
    rng = np.random.default_rng(seed)
    pos = sample_positions(n_cfg, N, rng)
    species = rng.integers(0, n_species, size=(n_cfg, N))
    pt = torch.tensor(pos); st = torch.tensor(species, dtype=torch.long)
    Es, Fs = [], []
    for i in range(0, n_cfg, 256):                          # chunk to bound memory
        e, f = reference(pt[i:i + 256], st[i:i + 256], long_range)
        Es.append(e); Fs.append(f)
    return {'positions': pt, 'species': st,
            'E': torch.cat(Es), 'F': torch.cat(Fs)}


if __name__ == "__main__":
    d = make_dataset(512, 24, seed=0, long_range=True)
    d0 = make_dataset(512, 24, seed=0, long_range=False)   # same geometries, no LR
    E_long = d['E'] - d0['E']                               # the pure non-local term
    print("|F| mean (with LR):", d['F'].norm(dim=-1).mean().item())
    print("std(E_short) =", d0['E'].std().item())
    print("std(E_long ) =", E_long.std().item(),
          "  -> LR is", f"{100*E_long.std()/d['E'].std():.0f}% of total energy std")
