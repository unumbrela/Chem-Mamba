"""Substitution-pair response probe for AuMgO (protocol P0-2, PAPER1_DESIGN v2).

Builds matched pairs with IDENTICAL coordinates that differ only in composition:
take a doped test structure (3 Al dopants >= 10.3 A from Au2) and substitute
Al -> Mg to get its undoped twin.  Any difference between a model's predictions
on the two twins can only come from a nonlocal pathway: the Au atoms' local
environments (receptive field = n_layers * cutoff = 7 A < 10.3 A) are bitwise
identical.  Unlike the group-level contrast in train_4g.aumgo_analysis, the
pair response has zero geometry leakage by construction.

Probes (per pair, doped minus twin):
  dq_Au2      total charge change of the Au2 cluster (me)        [charge models]
  contrast    mean per-atom dq(Au) - mean dq(bulk) (me); immune to the uniform
              conservation projection                            [charge models]
  dE          energy response (meV)                              [any model]
  dF_Au       mean |F_doped - F_twin| over Au atoms (meV/A)      [any model]
  dF_bulk     same over bulk atoms (scale reference; near-dopant atoms respond
              physically, so this is NOT expected to be zero)

Calibration logic (tail='none' variants match EFA's architecture = no explicit
electrostatics): ssm-iso/local models must give dq_Au2 = dF_Au = 0 exactly;
routing-capable models respond.  With a Coulomb/Ewald tail even local models
respond through the tail (wrong charges -> wrong response), so tail'ed
checkpoints are reported separately.

Usage:
  PYTHONPATH=. python chem_mamba/probe_pairs.py                    # calibrate defaults
  PYTHONPATH=. python chem_mamba/probe_pairs.py --ckpt results/AuMgO_ssm_s0_notail
  PYTHONPATH=. python chem_mamba/probe_pairs.py --export-efa       # write probe npz
"""
import argparse
import json
import os

import numpy as np
import torch

from chem_mamba.runner_data import load_dataset
from chem_mamba.model4g import make_model

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, 'results')
EFA_DIR = os.path.join(HERE, '..', 'data', 'efa_datasets')

# EFA official split (configs/trainer/4ghdnnp_AuMgO.py): num_train=4000,
# num_valid=500, split_seed=0 via np.random.seed(0) -> same MT19937 stream as
# our RandomState(0), verified in main().  EFA test = perm[4500:] which is a
# superset of our test perm[4750:], so probe pairs built from OUR test rows
# are unseen by both frameworks.
N_TRAIN_OURS, N_VAL_OURS = 4500, 250
N_SEEN_EFA = 4500


def build_pairs(data):
    """Return probe pair tensors from our AuMgO test split."""
    S = len(data['energy'])
    perm = np.random.RandomState(0).permutation(S)
    p2 = np.random.RandomState(0)  # EFA uses the legacy global-seed API;
    np.random.seed(0)              # confirm both give the same permutation
    assert (np.random.permutation(S) == perm).all(), 'split streams diverged'
    test = perm[N_TRAIN_OURS + N_VAL_OURS:]

    iAl = data['elems'].index('Al')
    iMg = data['elems'].index('Mg')
    iAu = data['elems'].index('Au')
    doped = [i for i in test if (data['species'][i] == iAl).any()]

    spc_d = data['species'][doped]
    spc_u = spc_d.clone()
    spc_u[spc_d == iAl] = iMg
    pos = data['pos'][doped]
    cell = data['lattice'][doped] if 'lattice' in data else None
    msk = data['mask'][doped]

    # per-atom minimum-image distance to the nearest substituted site.  Atoms
    # within the receptive field of a dopant respond LOCALLY and legitimately;
    # only atoms beyond it (Au and far bulk) are clean nonlocal probes.
    dmin, d_sub = [], []
    for k in range(len(doped)):
        al = pos[k][spc_d[k] == iAl]
        au = pos[k][spc_d[k] == iAu]
        d = al[:, None, :] - au[None, :, :]
        da = al[:, None, :] - pos[k][None, :, :]
        if cell is not None:
            L = torch.diagonal(cell[k])
            d = d - L * torch.round(d / L)
            da = da - L * torch.round(da / L)
        dmin.append(d.norm(dim=-1).min().item())
        d_sub.append(da.norm(dim=-1).min(0).values)

    return dict(idx=np.array(doped), pos=pos, spc_doped=spc_d, spc_twin=spc_u,
                mask=msk, cell=cell, iAu=iAu, iAl=iAl,
                min_Al_Au=np.array(dmin), d_sub=torch.stack(d_sub), test=test)


def run_model(ckpt_base, pairs, data, device='cuda', batch=8):
    """Evaluate one checkpoint on both twins; return per-pair responses."""
    cfg = json.load(open(ckpt_base + '.json'))['config']
    ck = torch.load(ckpt_base + '.pt', map_location=device, weights_only=False)
    model = make_model(cfg['model'], len(data['elems']), d=cfg['d'],
                       n_layers=cfg['layers'], cutoff=cfg['cutoff'],
                       order=cfg.get('order', 'centroid'),
                       tail=cfg.get('tail', 'coulomb')).to(device)
    model.load_state_dict(ck['state'])
    model.eval()

    P = len(pairs['idx'])
    out = {k: [] for k in ('E_d', 'E_u', 'F_d', 'F_u', 'q_d', 'q_u')}
    for s in range(0, P, batch):
        sl = slice(s, min(s + batch, P))
        pos = pairs['pos'][sl].to(device)
        msk = pairs['mask'][sl].to(device)
        cell = pairs['cell'][sl].to(device) if pairs['cell'] is not None else None
        Q = torch.zeros(pos.shape[0], device=device)
        for tag, spc in (('d', pairs['spc_doped']), ('u', pairs['spc_twin'])):
            E, F, q = model(pos, spc[sl].to(device), msk, Q, cell)
            out[f'E_{tag}'].append(E.detach().cpu())
            out[f'F_{tag}'].append(F.detach().cpu())
            out[f'q_{tag}'].append(q.detach().cpu())
    out = {k: torch.cat(v) for k, v in out.items()}

    spc_d, msk = pairs['spc_doped'], pairs['mask']
    au = (spc_d == pairs['iAu']) & msk
    subst = (spc_d == pairs['iAl']) & msk          # the 3 substituted sites
    bulk = msk & ~au & ~subst
    RF = 7.0                                       # receptive field, 2 x 3.5 A
    far = bulk & (pairs['d_sub'] > RF)             # clean nonlocal probes
    near = bulk & (pairs['d_sub'] <= RF)           # legitimate local response

    dq = out['q_d'] - out['q_u']
    dq_au2 = (dq * au).sum(1)                      # Au2 total response
    dq_au = (dq * au).sum(1) / au.sum(1)
    dq_far = (dq * far).sum(1) / far.sum(1)
    dF = (out['F_d'] - out['F_u']).norm(dim=-1)
    dF_au = (dF * au).sum(1) / au.sum(1)
    dF_far = (dF * far).sum(1) / far.sum(1)
    dF_near = (dF * near).sum(1) / near.sum(1)
    dE = out['E_d'] - out['E_u']

    return dict(
        model=cfg['model'], tail=cfg.get('tail', 'coulomb'),
        tag=os.path.basename(ckpt_base),
        dq_Au2_me=(1000 * dq_au2.mean()).item(),
        dq_Au2_me_std=(1000 * dq_au2.std()).item(),
        # contrast vs FAR bulk only: uniform projection cancels exactly and no
        # local pathway reaches either group => strictly zero for L0/L1 models
        contrast_me=(1000 * (dq_au - dq_far).mean()).item(),
        dE_meV_mean=(1000 * dE.mean()).item(),
        dE_meV_absmean=(1000 * dE.abs().mean()).item(),
        dF_Au_meV_A=(1000 * dF_au.mean()).item(),
        dF_far_meV_A=(1000 * dF_far.mean()).item(),
        dF_near_meV_A=(1000 * dF_near.mean()).item(),
        n_far_atoms=int(far.sum(1).float().mean().item()),
        n_pairs=len(pairs['idx']))


def truth_reference(data, pairs):
    """Group-level DFT reference for the expected response scale."""
    iAu, iAl = pairs['iAu'], pairs['iAl']
    spc, msk, q = data['species'], data['mask'], data['charges']
    au = (spc == iAu) & msk
    doped = ((spc == iAl) & msk).any(1)
    te = pairs['test']
    qAu2 = (q * au).sum(1)
    d_te = torch.tensor([i in set(np.array(te)[doped[te]].tolist()) for i in te])
    qd = qAu2[te][d_te].mean().item()
    qu = qAu2[te][~d_te].mean().item()
    return dict(qAu2_doped_e=qd, qAu2_undoped_e=qu,
                expected_dq_Au2_me=1000 * (qd - qu))


def export_efa(pairs, data):
    """Write probe pairs as an EFA-format npz (doped block then twin block).

    Energies/forces: real DFT labels for the doped originals (energy shifted
    with EFA's own per-element shifts so their eval metrics stay meaningful);
    zeros for the twins (no DFT truth exists -- predictions are what we want,
    via main_eval.py --collect_predictions)."""
    ref = np.load(os.path.join(EFA_DIR, 'AuMgO_preprocessed.npz'))
    shifts = ref['energy_shifts']
    z_of = {'O': 8, 'Mg': 12, 'Al': 13, 'Au': 79}
    z_map = np.array([z_of[e] for e in data['elems']])

    idx = pairs['idx']
    P = len(idx)
    pos = pairs['pos'].numpy()
    z_d = z_map[pairs['spc_doped'].numpy()]
    z_u = z_map[pairs['spc_twin'].numpy()]
    E_d = data['energy'][idx].numpy().astype(np.float64)
    E_d = E_d - shifts[z_d].sum(1)                  # match their shifted scale
    F_d = data['forces'][idx].numpy()
    cell = pairs['cell'].numpy()

    npz = dict(
        positions=np.concatenate([pos, pos]),
        atomic_numbers=np.concatenate([z_d, z_u]).astype(np.int64),
        energy=np.concatenate([E_d, np.zeros(P)]),
        forces=np.concatenate([F_d, np.zeros_like(F_d)]),
        lattice_vectors=np.concatenate([cell, cell]),
        total_charge=np.zeros(2 * P),
        energy_shifts=shifts)
    out = os.path.join(EFA_DIR, 'AuMgO_probe_pairs.npz')
    np.savez(out, **npz)
    meta = dict(n_pairs=P, layout='rows 0..P-1 doped, rows P..2P-1 twins',
                source_rows=idx.tolist(),
                min_Al_Au_A=pairs['min_Al_Au'].tolist(),
                note='twin energies/forces are zeros (no DFT labels); '
                     'use --collect_predictions and diff the two blocks')
    with open(out.replace('.npz', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f"[export] {out}  ({2*P} structures, {P} pairs)")


DEFAULT_CKPTS = [
    # tail='none' group: architecture-only routing (EFA-comparable, exact-zero
    # controls); then coupled group: response includes tail-mediated pathway
    'AuMgO_ssm-iso_s0_notail', 'AuMgO_ssm_s0_notail', 'AuMgO_ssm2_s0_notail',
    'AuMgO_attn_s0_notail',
    'AuMgO_local-Q_s0_ewald', 'AuMgO_ssm_s0_ewald30k',
    'AuMgO_ssm_s0_ewald_warm30k',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='*', default=None,
                    help='checkpoint basenames (no extension), default = '
                         'calibration matrix')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--export-efa', action='store_true')
    args = ap.parse_args()

    data = load_dataset('AuMgO')
    pairs = build_pairs(data)
    print(f"[pairs] {len(pairs['idx'])} doped test structures -> twins; "
          f"min Al-Au distance {pairs['min_Al_Au'].min():.2f} A "
          f"(receptive field 7.0 A)")
    ref = truth_reference(data, pairs)
    print(f"[truth] group-level Au2 charge: doped {ref['qAu2_doped_e']:+.3f} e, "
          f"undoped {ref['qAu2_undoped_e']:+.3f} e  -> expected pair response "
          f"~ {ref['expected_dq_Au2_me']:+.0f} me")

    if args.export_efa:
        export_efa(pairs, data)
        return

    names = args.ckpt if args.ckpt else DEFAULT_CKPTS
    rows = []
    for name in names:
        base = name if os.path.isabs(name) else os.path.join(RESULTS, name)
        if not (os.path.exists(base + '.pt') and os.path.exists(base + '.json')):
            print(f"[skip] {name}: missing .pt/.json")
            continue
        r = run_model(base, pairs, data, args.device, args.batch)
        rows.append(r)
        print(f"{r['tag']:38s} tail={r['tail']:7s} "
              f"dq_Au2 {r['dq_Au2_me']:+7.1f} me  contrast {r['contrast_me']:+7.1f} me  "
              f"dE |{r['dE_meV_absmean']:7.1f}| meV  dF_Au {r['dF_Au_meV_A']:7.1f}  "
              f"far {r['dF_far_meV_A']:7.1f}  near {r['dF_near_meV_A']:7.1f} meV/A")

    out = os.path.join(RESULTS, 'probe_pairs_AuMgO.json')
    with open(out, 'w') as f:
        json.dump(dict(truth=ref, n_pairs=int(len(pairs['idx'])),
                       min_Al_Au_A=float(pairs['min_Al_Au'].min()),
                       rows=rows), f, indent=1)
    print(f"[saved] {out}")


if __name__ == '__main__':
    main()
