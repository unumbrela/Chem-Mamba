"""
Proof-of-concept: can a linear-scaling selective SSM capture the NON-LOCAL
charge/energy assignment that a local model provably cannot?

Trains each model to predict per-site QEq charges, then reports on held-out data:
  - charge MAE        (overall accuracy)
  - energy MAE        (derived from predicted charges via the true kernel)
  - marked-site charge MAE on the DISCRIMINATING test, where the local model is
    information-theoretically stuck at std(q_marked).

Run:  python poc/run_poc.py
"""
import time, argparse
import numpy as np
import torch
import torch.nn as nn

from qeq_data import make_dataset, make_fixed_neighborhood_testset, coupling_matrix
from models import build, enforce_conservation

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CUTOFF = 2.5
N_TRAIN = 32


def to_dev(batch):
    return {k: torch.as_tensor(v).to(DEV) if isinstance(v, np.ndarray) else v
            for k, v in batch.items()}


def energy_from_charges(positions, chi, J, q):
    """Batched E = chi.q + 1/2 q^T A q with A_ii=J, A_ij=1/sqrt(dx^2+1)."""
    dx = positions[:, :, None] - positions[:, None, :]
    A = 1.0 / torch.sqrt(dx * dx + 1.0)
    B, N = chi.shape
    diag = torch.eye(N, device=positions.device, dtype=A.dtype)
    A = A * (1 - diag) + J[:, :, None] * diag
    quad = 0.5 * torch.einsum('bi,bij,bj->b', q, A, q)
    return (chi * q).sum(1) + quad


def predict(model, batch, idx=None):
    """Model forward + charge-conservation projection (applied uniformly)."""
    sel = (lambda x: x[idx]) if idx is not None else (lambda x: x)
    raw = model(sel(batch['positions']), sel(batch['chi']),
                sel(batch['J']), sel(batch['Q']))
    return enforce_conservation(raw, sel(batch['Q']))


def train_model(name, train, val, steps=4000, bs=256, lr=3e-3, seed=0, warmup=0.1):
    torch.manual_seed(seed)
    model = build(name, CUTOFF).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = int(warmup * steps)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, w),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps - w)],
        milestones=[w])
    n = len(train['E'])
    best_val, best_state = 1e9, None
    g = torch.Generator(device='cpu').manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        q_pred = predict(model, train, idx)
        loss = nn.functional.mse_loss(q_pred, train['q'][idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                vloss = nn.functional.l1_loss(predict(model, val), val['q']).item()
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = a.norm() * b.norm()
    return (a @ b / denom).item() if denom > 1e-8 else 0.0


@torch.no_grad()
def evaluate(model, test, disc):
    q = predict(model, test)
    charge_mae = nn.functional.l1_loss(q, test['q']).item()
    E_pred = energy_from_charges(test['positions'], test['chi'], test['J'], q)
    energy_mae = nn.functional.l1_loss(E_pred, test['E']).item()

    # Discriminating test: on the RAW readout (no conservation projection), how
    # much can each model TRACK the non-local variation of q at the marked site,
    # whose local neighbourhood is frozen?  A local model's prediction there is
    # structurally constant -> zero variance, zero correlation with the truth.
    m = disc['marked']
    p = model(disc['positions'], disc['chi'], disc['J'], disc['Q'])[:, m]
    t = disc['q'][:, m]
    return charge_mae, energy_mae, p.std().item(), pearson(p, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    args = ap.parse_args()

    print(f"device={DEV}  train chain length N={N_TRAIN}  cutoff={CUTOFF}\n")
    train = to_dev(make_dataset(4000, N_TRAIN, seed=1))
    val = to_dev(make_dataset(500, N_TRAIN, seed=2))
    test = to_dev(make_dataset(1000, N_TRAIN, seed=3))
    disc = to_dev(make_fixed_neighborhood_testset(1000, N_TRAIN, CUTOFF, seed=4))

    # trivial baselines
    q_std = disc['q'][:, disc['marked']].std().item()
    mean_q_mae = nn.functional.l1_loss(
        test['q'], test['q'].mean(0, keepdim=True).expand_as(test['q'])).item()
    print(f"[reference]  predict-the-mean charge MAE = {mean_q_mae:.4f}")
    print(f"[reference]  std(q_marked) on discriminating test = {q_std:.4f}")
    print(f"             (a purely local model CANNOT beat this on the marked site)\n")

    lr = {'local': 3e-3, 'ssm': 3e-3, 'transformer': 1e-3}
    rows = []
    for name in ['local', 'ssm', 'transformer']:
        t0 = time.time()
        model = train_model(name, train, val, steps=args.steps, lr=lr[name])
        params = sum(p.numel() for p in model.parameters())
        cmae, emae, pstd, corr = evaluate(model, test, disc)
        rows.append((name, params, cmae, emae, pstd, corr, time.time() - t0))

    print(f"\n{'model':<13}{'params':>9}{'chargeMAE':>11}{'energyMAE':>11}"
          f"{'markStd':>9}{'markCorr':>9}{'train_s':>9}")
    print('-' * 70)
    for name, params, cmae, emae, pstd, corr, dt in rows:
        print(f"{name:<13}{params:>9}{cmae:>11.4f}{emae:>11.4f}"
              f"{pstd:>9.4f}{corr:>9.3f}{dt:>9.1f}")
    print('-' * 70)
    print(f"true std(q_marked)={q_std:.4f}. Local prediction there is a constant")
    print(f"(markStd~0, markCorr~0): it is structurally blind to the far field.")


if __name__ == "__main__":
    main()
