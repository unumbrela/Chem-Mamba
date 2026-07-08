"""
M1: end-to-end integration on a real 3D invariant backbone.

Gates (correctness, must pass before any result is trusted):
  [S] energy invariant to rotation / translation / permutation; force equivariant
  [F] autograd forces match finite differences

Comparisons (backbone-only 'local charges' vs backbone+SSM 'non-local charges'):
  no-regression : trained on SHORT-range-only reference -> SSM must not hurt
  long-range    : trained on reference WITH non-local Coulomb -> SSM should help

Run:  cd Chem-Mamba && PYTHONPATH=. python chem_mamba/train_m1.py
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn

from chem_mamba.data3d import make_dataset
from chem_mamba.model import ChemMamba

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def rand_rotation(seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R))            # proper rotation
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(DEV)


@torch.no_grad()
def _energy_only(model, pos, sp):
    p = pos.clone().requires_grad_(True)
    with torch.enable_grad():
        return model.energy(p, sp)


def symmetry_checks(model, pos, sp):
    R = rand_rotation()
    E0 = _energy_only(model, pos, sp)
    Erot = _energy_only(model, pos @ R.T, sp)
    Etr = _energy_only(model, pos + torch.randn(1, 1, 3, device=DEV) * 5, sp)
    perm = torch.randperm(pos.shape[1], device=DEV)
    Eperm = _energy_only(model, pos[:, perm], sp[:, perm])

    # force equivariance under rotation
    _, F0 = model(pos, sp)
    _, Frot = model(pos @ R.T, sp)
    ferr = (Frot - F0 @ R.T).abs().max().item()

    rel = lambda a, b: ((a - b).abs() / (E0.abs() + 1e-6)).max().item()
    return dict(rotation=rel(Erot, E0), translation=rel(Etr, E0),
                permutation=rel(Eperm, E0), force_equivariance=ferr)


def force_finite_diff(model, pos, sp, eps=1e-4):
    """Central-difference several coordinates in FLOAT64 (float32 central diff of
    near-equal energies suffers catastrophic cancellation).  Error normalised by
    force scale."""
    model = model.double()
    pos = pos[:1].double().clone(); sp = sp[:1]
    _, F = model(pos, sp)
    scale = F.pow(2).mean().sqrt().item()
    errs = []
    for (i, c) in [(3, 0), (7, 1), (12, 2), (20, 0)]:
        p = pos.clone(); p[0, i, c] += eps
        Ep = _energy_only(model, p, sp)[0]
        p = pos.clone(); p[0, i, c] -= eps
        Em = _energy_only(model, p, sp)[0]
        fd = -((Ep - Em) / (2 * eps)).item()
        errs.append(abs(fd - F[0, i, c].item()))
    model.float()
    return max(errs) / (scale + 1e-8)


def to_dev(d):
    return {k: v.to(DEV) for k, v in d.items()}


def predict_chunked(model, pos, sp, chunk=64):
    """Forces need autograd, so we cannot eval all configs at once (memory).
    Run in chunks and concatenate detached predictions."""
    Es, Fs = [], []
    for i in range(0, len(pos), chunk):
        E, Fp = model(pos[i:i + chunk], sp[i:i + chunk])
        Es.append(E.detach()); Fs.append(Fp.detach())
    return torch.cat(Es), torch.cat(Fs)


def train(use_ssm, tr, va, steps, bs=24, lr=2e-3, seed=0, bb_kw=None):
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed)
    model = ChemMamba(use_ssm=use_ssm, **(bb_kw or {})).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    N = tr['positions'].shape[1]
    Emu = tr['E'].mean(); Es = tr['E'].std(); Fs = tr['F'].std()
    n = len(tr['E']); g = torch.Generator().manual_seed(seed)
    best, best_state = 1e9, None
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        model.train()
        Ep, Fp = model(tr['positions'][idx], tr['species'][idx])
        loss = ((Ep - tr['E'][idx]) / Es).pow(2).mean() + \
               ((Fp - tr['F'][idx]) / Fs).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 200 == 0:
            model.eval()
            Ev, Fv = predict_chunked(model, va['positions'], va['species'])
            vl = (Ev - va['E']).abs().mean().item() / N + (Fv - va['F']).abs().mean().item()
            if vl < best:
                best = vl; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, te):
    N = te['positions'].shape[1]
    Ep, Fp = predict_chunked(model, te['positions'], te['species'])
    return (Ep - te['E']).abs().mean().item() / N, (Fp - te['F']).abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--N', type=int, default=64)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--bb_layers', type=int, default=2)     # backbone reach = layers x cutoff
    ap.add_argument('--bb_cutoff', type=float, default=2.2)
    args = ap.parse_args()
    bb_kw = dict(n_layers=args.bb_layers, cutoff=args.bb_cutoff)
    print(f"device={DEV}  N={args.N}  steps={args.steps}  "
          f"backbone reach ~= {args.bb_layers * args.bb_cutoff:.1f}\n")

    # correctness gates on a fresh model
    probe = to_dev(make_dataset(8, args.N, seed=99, long_range=True))
    m = ChemMamba(use_ssm=True, **bb_kw).to(DEV)
    s = symmetry_checks(m, probe['positions'], probe['species'])
    fd = force_finite_diff(m, probe['positions'], probe['species'])
    print("[S] max relative energy deviation (should be ~1e-5, float32):")
    for k, v in s.items():
        print(f"      {k:<20}: {v:.2e}")
    print(f"[F] force vs finite-difference relative error: {fd:.2e}\n")
    del m                                             # free the float64 gate model
    if DEV == 'cuda':
        torch.cuda.empty_cache()

    results = {}
    for tag, lr_flag in [('no-regression (short-only)', False),
                         ('long-range (non-local)', True)]:
        tr = to_dev(make_dataset(args.n, args.N, seed=1, long_range=lr_flag))
        va = to_dev(make_dataset(300, args.N, seed=2, long_range=lr_flag))
        te = to_dev(make_dataset(500, args.N, seed=3, long_range=lr_flag))
        row = {}
        for use_ssm in [False, True]:
            t0 = time.time()
            model = train(use_ssm, tr, va, args.steps, bb_kw=bb_kw)
            emae, fmae = evaluate(model, te)
            row['ssm' if use_ssm else 'local'] = (emae, fmae, time.time() - t0)
            del model
            if DEV == 'cuda':
                torch.cuda.empty_cache()
        results[tag] = row

    print(f"\n{'setting':<28}{'model':<8}{'E MAE/atom':>12}{'F MAE':>10}{'train_s':>9}")
    print('-' * 67)
    for tag, row in results.items():
        for name in ['local', 'ssm']:
            emae, fmae, dt = row[name]
            print(f"{tag:<28}{name:<8}{emae:>12.4f}{fmae:>10.4f}{dt:>9.1f}")
        el, _, _ = row['local']; es, _, _ = row['ssm']
        print(f"{'  -> E MAE improvement:':<36}{100*(1-es/el):>8.0f}%\n")


if __name__ == "__main__":
    main()
