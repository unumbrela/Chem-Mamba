"""
Ground-truth data generator for the non-locality proof-of-concept.

We use charge equilibration (QEq / electronegativity equalization) as a
*controllable* physical model in which the per-site equilibrium charges q_i and
the total energy E are provably NON-LOCAL functions of the whole system:
changing the electronegativity chi at ONE site changes q at EVERY site, because
the equilibrium is the solution of a coupled linear system.

This is exactly the regime where:
  - a local model (fixed cutoff, e.g. a short-range GNN or LES's *local* latent
    charges) has an irreducible error, and
  - a global model (4G-HDNNP's O(N^3) charge equilibration, or -- our claim --
    an O(N) selective state-space scan) can succeed.

Energy functional (soft-core screened Coulomb, well-conditioned):
    E(q) = sum_i [ chi_i q_i + 1/2 J_i q_i^2 ] + 1/2 sum_{i!=j} q_i q_j gamma_ij
         = chi . q + 1/2 q^T A q,     A_ii = J_i,  A_ij = gamma_ij (i!=j)
    gamma_ij = 1 / sqrt((x_i - x_j)^2 + 1)          (soft Coulomb)
subject to charge conservation  sum_i q_i = Q_tot.

Minimising with a Lagrange multiplier lambda gives the linear system
    [ A   -1 ] [ q      ]   [ -chi  ]
    [ 1^T  0 ] [ lambda ] = [  Q    ]
whose solution is the ground truth (q*, E*).
"""

import numpy as np


def coupling_matrix(positions, J):
    """A_ii = J_i, A_ij = 1/sqrt(r_ij^2 + 1)."""
    dx = positions[:, None] - positions[None, :]
    gamma = 1.0 / np.sqrt(dx * dx + 1.0)
    A = gamma.copy()
    np.fill_diagonal(A, J)
    return A


def solve_qeq(positions, chi, J, Q_tot):
    """Return equilibrium charges q (N,) and energy E (scalar)."""
    N = len(chi)
    A = coupling_matrix(positions, J)
    M = np.zeros((N + 1, N + 1))
    M[:N, :N] = A
    M[:N, N] = -1.0
    M[N, :N] = 1.0
    rhs = np.concatenate([-chi, [Q_tot]])
    sol = np.linalg.solve(M, rhs)
    q = sol[:N]
    E = chi @ q + 0.5 * q @ A @ q
    return q, E


def make_config(N, rng, jitter=0.0, Q_choices=(-1.0, 0.0, 1.0),
                fixed=None):
    """
    Build one random 1D chain configuration and solve QEq.

    fixed: optional dict to pin a sub-region (used for the discriminating test):
        {'idx': array of site indices, 'chi': values, 'J': values,
         'Q': total charge}  -- those sites/Q are held fixed, the rest random.
    """
    positions = np.arange(N, dtype=float)
    if jitter:
        positions = positions + rng.uniform(-jitter, jitter, size=N)
    chi = rng.normal(0.0, 1.0, size=N)
    J = rng.uniform(1.0, 2.0, size=N)
    Q = float(rng.choice(Q_choices))

    if fixed is not None:
        idx = fixed['idx']
        chi[idx] = fixed['chi']
        J[idx] = fixed['J']
        Q = fixed['Q']

    q, E = solve_qeq(positions, chi, J, Q)
    return dict(positions=positions, chi=chi, J=J, Q=Q, q=q, E=E)


def make_dataset(n_samples, N, seed, jitter=0.0, Q_choices=(-1.0, 0.0, 1.0),
                 fixed=None):
    rng = np.random.default_rng(seed)
    cfgs = [make_config(N, rng, jitter, Q_choices, fixed) for _ in range(n_samples)]
    batch = {
        'positions': np.stack([c['positions'] for c in cfgs]).astype(np.float32),
        'chi': np.stack([c['chi'] for c in cfgs]).astype(np.float32),
        'J': np.stack([c['J'] for c in cfgs]).astype(np.float32),
        'Q': np.array([c['Q'] for c in cfgs], dtype=np.float32),
        'q': np.stack([c['q'] for c in cfgs]).astype(np.float32),
        'E': np.array([c['E'] for c in cfgs], dtype=np.float32),
    }
    return batch


def make_fixed_neighborhood_testset(n_samples, N, cutoff, seed, marked=None):
    """
    The DISCRIMINATING test.

    We freeze the local neighbourhood (within `cutoff`) of a marked site and the
    total charge (Q=0), and randomise everything *outside* that neighbourhood.
    The marked site's local input is therefore IDENTICAL across every sample,
    so any purely-local model must output a constant charge there -- yet the true
    q_marked varies because QEq is non-local. A global model can track it.
    """
    if marked is None:
        marked = N // 2
    rng = np.random.default_rng(seed)
    positions = np.arange(N, dtype=float)
    # indices whose local features feed the marked site (its fixed neighbourhood)
    near = np.where(np.abs(positions - marked) <= cutoff)[0]
    # freeze the near-region chi/J once
    chi_fixed = rng.normal(0.0, 1.0, size=len(near))
    J_fixed = rng.uniform(1.0, 2.0, size=len(near))
    fixed = dict(idx=near, chi=chi_fixed, J=J_fixed, Q=0.0)

    batch = make_dataset(n_samples, N, seed + 1, jitter=0.0,
                         Q_choices=(0.0,), fixed=fixed)
    batch['marked'] = marked
    batch['near'] = near
    return batch


if __name__ == "__main__":
    # sanity check
    b = make_dataset(4, 16, seed=0)
    print("energy sample:", b['E'])
    print("sum of charges (should equal Q):",
          b['q'].sum(1), "vs Q", b['Q'])
    t = make_fixed_neighborhood_testset(200, 32, cutoff=2.5, seed=0)
    print("marked site:", t['marked'], "near:", t['near'])
    print("std of true q at marked site (what a local model is stuck at):",
          t['q'][:, t['marked']].std())
