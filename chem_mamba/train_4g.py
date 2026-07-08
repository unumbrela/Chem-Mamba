"""Train/eval ChemMamba4G on the 4G-HDNNP charge-transfer benchmarks.

Usage:
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset Carbon_chain --model ssm
Models: local-Q | local+Q | ssm | ssm-iso | ssm2 | attn | qeq  (see model4g.make_model)
Outputs JSON to chem_mamba/results/<dataset>_<model>_s<seed>.json
"""
import argparse
import json
import os

import numpy as np
import torch

from chem_mamba.runner_data import load_dataset
from chem_mamba.model4g import make_model


def composition_baseline(species, mask, E, train_idx, n_species):
    """Least-squares per-element reference energies from the train split."""
    counts = torch.zeros(len(E), n_species)
    for s in range(n_species):
        counts[:, s] = ((species == s) & mask).sum(1).float()
    A, b = counts[train_idx].double(), E[train_idx].double()
    # float64 + gelsd: the system is rank-deficient (few distinct compositions)
    # and |E| ~ 1e6 eV, so float32 QR is numerically unstable here
    E0 = torch.linalg.lstsq(A, b[:, None], driver='gelsd').solution.squeeze(1)
    return (E.double() - counts.double() @ E0).float(), E0.float()


def batches(idx, bs, shuffle=True):
    order = np.random.permutation(len(idx)) if shuffle else np.arange(len(idx))
    for i in range(0, len(order), bs):
        yield idx[order[i:i + bs]]


def evaluate(model, data, idx, device, bs=256):
    model.eval()
    se_E, se_F, se_q, n_atoms, n_fc, n_qs = 0., 0., 0., 0, 0, 0
    preds = {'E': [], 'q': []}
    for b in batches(idx, bs, shuffle=False):
        pos, spc = data['pos'][b].to(device), data['species'][b].to(device)
        msk, Q = data['mask'][b].to(device), data['Qtot'][b].to(device)
        cell = data['lattice'][b].to(device) if 'lattice' in data else None
        E, F, q = model(pos, spc, msk, Q, cell)
        n = msk.sum(1)
        se_E += (((E - data['Etgt'][b].to(device)) / n) ** 2 * n).sum().item()
        dF = (F - data['forces'][b].to(device)) * msk[..., None]
        se_F += (dF ** 2).sum().item()
        dq = (q - data['charges'][b].to(device)) * msk
        se_q += (dq ** 2).sum().item()
        n_atoms += n.sum().item()
        n_fc += 3 * n.sum().item()
        n_qs += n.sum().item()
        preds['E'].append(E.cpu())
        preds['q'].append(q.cpu())
    rmse = dict(E_meV_atom=1000 * np.sqrt(se_E / n_atoms),
                F_meV_A=1000 * np.sqrt(se_F / n_fc),
                q_me=1000 * np.sqrt(se_q / n_qs))
    return rmse, torch.cat(preds['E']), torch.cat(preds['q'])


def symmetry_check(model, data, idx, device):
    """Rotation invariance of energy on a real (padded, Q-conditioned) batch."""
    if 'lattice' in data:      # rotating positions but not the cell is ill-defined
        return float('nan')
    b = idx[:16]
    pos, spc = data['pos'][b].to(device), data['species'][b].to(device)
    msk, Q = data['mask'][b].to(device), data['Qtot'][b].to(device)
    th = torch.tensor(0.7)
    R = torch.tensor([[torch.cos(th), -torch.sin(th), 0.],
                      [torch.sin(th), torch.cos(th), 0.], [0., 0., 1.]]).to(device)
    with torch.no_grad():
        E1, _, _ = model(pos, spc, msk, Q)
        E2, _, _ = model(pos @ R.T, spc, msk, Q)
    return (E2 - E1).abs().max().item() / E1.abs().mean().item()


def carbon_chain_analysis(data, idx, q_pred):
    """Charge profile along the chain, oriented from the protonated end.

    Geometric identification: in C10H3+ the protonated terminal C carries 2 H;
    profile position = rank of distance from that C.  For C10H2 both ends are
    equivalent; use both orientations.  Returns per-rank mean true/pred charge
    and the far-end H statistics (the paper's discriminative observable).
    """
    out = {}
    pos, spc, msk = data['pos'][idx], data['species'][idx], data['mask'][idx]
    qt = data['charges'][idx]
    n = msk.sum(1)
    iC, iH = data['elems'].index('C'), data['elems'].index('H')

    for label, sel in [('cation', n == 13), ('neutral', n == 12)]:
        if sel.sum() == 0:
            continue
        P, S, QT, QP = pos[sel], spc[sel], qt[sel], q_pred[sel]
        N = int(n[sel][0])
        P, S, QT, QP = P[:, :N], S[:, :N], QT[:, :N], QP[:, :N]
        d = (P[:, :, None, :] - P[:, None, :, :]).norm(dim=-1)
        nH = ((d < 1.3) & (S[:, None, :] == iH)).sum(2)     # H neighbors per atom
        if label == 'cation':
            anchor = ((S == iC) & (nH == 2)).float().argmax(1)   # protonated C
            ok = ((S == iC) & (nH == 2)).any(1)
        else:
            anchor = ((S == iH)).float().argmax(1)               # either end
            ok = torch.ones(len(P), dtype=torch.bool)
        P, S, QT, QP, anchor = P[ok], S[ok], QT[ok], QP[ok], anchor[ok]
        da = (P - P[torch.arange(len(P)), anchor][:, None]).norm(dim=-1)
        rank = da.argsort(1).argsort(1)                          # 0 = anchor end
        prof_t = torch.zeros(N)
        prof_p = torch.zeros(N)
        for r in range(N):
            m = rank == r
            prof_t[r] = QT[m].mean()
            prof_p[r] = QP[m].mean()
        # far-end H = H atom with max rank
        hrank = torch.where(S == iH, rank, torch.full_like(rank, -1))
        far = hrank.argmax(1)
        ar = torch.arange(len(P))
        out[label] = dict(
            profile_true=prof_t.tolist(), profile_pred=prof_p.tolist(),
            farH_true_mean=QT[ar, far].mean().item(),
            farH_true_std=QT[ar, far].std().item(),
            farH_pred_mean=QP[ar, far].mean().item(),
            farH_pred_std=QP[ar, far].std().item(),
            farH_corr=float(np.corrcoef(QT[ar, far].numpy(),
                                        QP[ar, far].numpy())[0, 1]),
            n_structs=int(len(P)))
    if 'cation' in out and 'neutral' in out:
        out['farH_separation_true'] = out['cation']['farH_true_mean'] - out['neutral']['farH_true_mean']
        out['farH_separation_pred'] = out['cation']['farH_pred_mean'] - out['neutral']['farH_pred_mean']
    return out


def nacl_analysis(data, idx, q_pred, E_pred):
    """Na8Cl8+ vs Na9Cl8+ (both Q=+1, so Q carries zero discriminative info).
    Observable: charge of the displaced Na (file atom #1 = index 0), whose
    ionization state depends on the global electron balance of the cluster."""
    out = {}
    msk, qt = data['mask'][idx], data['charges'][idx]
    n = msk.sum(1)
    for label, sel in [('Na8Cl8', n == 16), ('Na9Cl8', n == 17)]:
        if sel.sum() == 0:
            continue
        t, p = qt[sel][:, 0], q_pred[sel][:, 0]
        et = data['Etgt'][idx][sel]
        ep = E_pred[sel]
        out[label] = dict(
            q0_true_mean=t.mean().item(), q0_true_std=t.std().item(),
            q0_pred_mean=p.mean().item(), q0_pred_std=p.std().item(),
            q0_rmse_me=1000 * ((p - t) ** 2).mean().sqrt().item(),
            q0_corr=float(np.corrcoef(t.numpy(), p.numpy())[0, 1]),
            E_rmse_meV_atom=1000 * (((ep - et) / n[sel]) ** 2).mean().sqrt().item(),
            n_structs=int(sel.sum()))
    return out


def ag_analysis(data, idx, q_pred, E_pred):
    """Ag3+ vs Ag3-: identical compositions/geometries, opposite total charge.
    Control system: Q IS the whole story; local+Q should suffice."""
    out = {}
    qt, n = data['charges'][idx], data['mask'][idx].sum(1)
    for label, sel in [('Ag3+', data['Qtot'][idx] > 0), ('Ag3-', data['Qtot'][idx] < 0)]:
        t, p = qt[sel][:, :3], q_pred[sel][:, :3]
        et, ep = data['Etgt'][idx][sel], E_pred[sel]
        out[label] = dict(
            q_rmse_me=1000 * ((p - t) ** 2).mean().sqrt().item(),
            E_rmse_meV_atom=1000 * (((ep - et) / n[sel]) ** 2).mean().sqrt().item(),
            n_structs=int(sel.sum()))
    return out


def aumgo_analysis(data, idx, q_pred):
    """Discriminative observable: net charge of the Au2 cluster, doped (3 Al,
    >=10 A from Au) vs undoped substrate.  Total charge Q=0 for both, so no
    global-scalar shortcut exists; only spatial nonlocal reach can resolve it."""
    spc, msk, qt = data['species'][idx], data['mask'][idx], data['charges'][idx]
    iAu, iAl = data['elems'].index('Au'), data['elems'].index('Al')
    au = (spc == iAu) & msk
    doped = ((spc == iAl) & msk).sum(1) > 0
    qAu_t = (qt * au).sum(1)
    qAu_p = (q_pred * au).sum(1)

    # Spatial contrast: doping response on Au vs bulk (Mg+O).  The uniform
    # conservation projection adds the same per-structure constant to every
    # atom, so it cancels EXACTLY here -- contrast isolates genuine spatially
    # resolved routing (true value ~ -132 me) from global-scalar shortcuts.
    bulk = msk & ~au & ~((spc == iAl) & msk)
    def gmean(sel_atoms, sel_str, q):
        x = torch.where(sel_atoms[sel_str], q[sel_str], torch.nan)
        return torch.nanmean(x)
    dq_au_p = gmean(au, doped, q_pred) - gmean(au, ~doped, q_pred)
    dq_bk_p = gmean(bulk, doped, q_pred) - gmean(bulk, ~doped, q_pred)
    dq_au_t = gmean(au, doped, qt) - gmean(au, ~doped, qt)
    dq_bk_t = gmean(bulk, doped, qt) - gmean(bulk, ~doped, qt)

    return dict(
        doped_true=qAu_t[doped].mean().item(), doped_pred=qAu_p[doped].mean().item(),
        undoped_true=qAu_t[~doped].mean().item(), undoped_pred=qAu_p[~doped].mean().item(),
        sep_true=(qAu_t[doped].mean() - qAu_t[~doped].mean()).item(),
        sep_pred=(qAu_p[doped].mean() - qAu_p[~doped].mean()).item(),
        corr=float(np.corrcoef(qAu_t.numpy(), qAu_p.numpy())[0, 1]),
        contrast_pred=(dq_au_p - dq_bk_p).item(),
        contrast_true=(dq_au_t - dq_bk_t).item(),
        n_doped=int(doped.sum()), n_undoped=int((~doped).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='Carbon_chain')
    ap.add_argument('--model', default='ssm',
                    choices=['local-Q', 'local+Q', 'ssm', 'ssm-iso',
                             'ssm2', 'attn', 'qeq'])
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--cutoff', type=float, default=3.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--order', default='centroid', choices=['centroid', 'z'],
                    help='SSM ordering key (z for slabs: PBC-safe depth profile)')
    ap.add_argument('--qw', default='none', choices=['none', 'element'],
                    help='element: weight charge loss by inverse per-element '
                         'variance so sparse species are not drowned out')
    ap.add_argument('--tail', default='coulomb', choices=['coulomb', 'none', 'ewald'],
                    help='ewald: slab-safe periodic tail (M2); '
                         'none: charges become a pure auxiliary head (diagnostic)')
    ap.add_argument('--detach-elec', action='store_true',
                    help='stop-gradient the charge->electrostatic-energy path: '
                         'charges feed E_elec but energy/force grads do not reach '
                         'the charge head (routing trained on labels alone). '
                         'NOTE: diverges -- E_elec becomes a param-gradient dead '
                         'end; kept only as a documented negative control')
    ap.add_argument('--init-from', default='',
                    help='warm-start weights from a checkpoint .pt (two-stage: '
                         'e.g. init from the tail=none Test A routing, then turn '
                         'the physical tail on to test whether routing survives)')
    ap.add_argument('--tag', default='',
                    help='suffix for the results filename (protocol variants)')
    ap.add_argument('--wf', type=float, default=1.0,
                    help='force-loss weight multiplier (E and q stay at 1); '
                         'also applied to the val-selection loss for consistency')
    ap.add_argument('--ema', type=float, default=0.0,
                    help='EMA decay for weight averaging (e.g. 0.999); 0 = off. '
                         'Val/test are evaluated with the EMA weights')
    ap.add_argument('--eval-batch', type=int, default=256,
                    help='eval batch size; lower it for large periodic systems '
                         'to cap the eval-time VRAM spike (forces need the '
                         'autograd graph even at eval). WSL2 lesson: pushing '
                         'total VRAM near the limit resets the whole device '
                         'and kills EVERY cuda process, not just the offender')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device

    data = load_dataset(args.dataset)
    n_species = len(data['elems'])
    S = len(data['energy'])
    perm = np.random.RandomState(0).permutation(S)          # split fixed across seeds/models
    n_tr, n_va = int(0.9 * S), int(0.05 * S)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    data['Etgt'], E0 = composition_baseline(
        data['species'], data['mask'], data['energy'], tr, n_species)
    natoms = data['mask'].sum(1)
    vE = ((data['Etgt'][tr] / natoms[tr]) ** 2).mean().item()
    vF = (data['forces'][tr][data['mask'][tr]] ** 2).mean().item()
    vq = (data['charges'][tr][data['mask'][tr]] ** 2).mean().item()
    print(f"[data] {args.dataset}: {S} structs, elems={data['elems']}, "
          f"E0={[f'{e:.2f}' for e in E0.tolist()]}, "
          f"std(E/atom)={np.sqrt(vE)*1000:.1f} meV, std(F)={np.sqrt(vF):.3f} eV/A, "
          f"std(q)={np.sqrt(vq)*1000:.0f} me")

    if args.qw == 'element':
        # count-balanced: every element contributes equally to the charge loss,
        # so sparse species (e.g. 2 Au among 110 atoms) are not drowned out
        wq_el = torch.zeros(n_species)
        n_tot = data['mask'][tr].sum().item()
        for s in range(n_species):
            n_s = ((data['species'][tr] == s) & data['mask'][tr]).sum().item()
            wq_el[s] = n_tot / (n_s * n_species)
        # renormalize the loss scale to the weighted second moment
        w_at = wq_el[data['species'][tr]][data['mask'][tr]]
        vq = (w_at * data['charges'][tr][data['mask'][tr]] ** 2).mean().item()
        wq_el = wq_el.to(dev)
        print(f"[loss] count-balanced charge weights: "
              f"{dict(zip(data['elems'], [f'{w:.1f}' for w in wq_el.tolist()]))}, "
              f"weighted std(q)={np.sqrt(vq)*1000:.0f} me")
    else:
        wq_el = None

    model = make_model(args.model, n_species, d=args.d, n_layers=args.layers,
                       cutoff=args.cutoff, order=args.order, tail=args.tail,
                       detach_elec=args.detach_elec).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model}: {n_par/1e3:.1f}k params")

    if args.init_from:
        ck = torch.load(args.init_from, map_location=dev, weights_only=False)
        model.load_state_dict(ck['state'])
        print(f"[init] warm-started weights from {args.init_from}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, 1e-5)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ''
    out = os.path.join(os.path.dirname(__file__), 'results',
                       f"{args.dataset}_{args.model}_s{args.seed}{suffix}.json")
    ckpt = out.replace('.json', '.ckpt.pt')   # rolling best (crash insurance)
    ema = None
    if args.ema > 0:
        # initialized after --init-from so warm starts average around the
        # loaded solution, not around the random init
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[ema] weight averaging on, decay={args.ema}")
    best = (1e18, None)
    step = 0
    while step < args.steps:
        for b in batches(tr, args.batch):
            if step >= args.steps:
                break
            model.train()
            pos, spc = data['pos'][b].to(dev), data['species'][b].to(dev)
            msk, Q = data['mask'][b].to(dev), data['Qtot'][b].to(dev)
            cell = data['lattice'][b].to(dev) if 'lattice' in data else None
            E, F, q = model(pos, spc, msk, Q, cell)
            n = msk.sum(1)
            lE = (((E - data['Etgt'][b].to(dev)) / n) ** 2).mean()
            dF = (F - data['forces'][b].to(dev)) * msk[..., None]
            lF = (dF ** 2).sum() / (3 * n.sum())
            dq = (q - data['charges'][b].to(dev)) * msk
            if wq_el is not None:
                dq = dq * wq_el[spc].sqrt()
            lq = (dq ** 2).sum() / n.sum()
            loss = lE / vE + args.wf * lF / vF + lq / vq
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(args.ema).add_(v, alpha=1 - args.ema)
                        else:
                            ema[k].copy_(v)
            step += 1
            if step % 500 == 0 or step == args.steps:
                if ema is not None:
                    raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(ema)
                rm, _, qv = evaluate(model, data, va, dev, bs=args.eval_batch)
                vloss = rm['E_meV_atom'] ** 2 / (vE * 1e6) + \
                        args.wf * rm['F_meV_A'] ** 2 / (vF * 1e6) + \
                        rm['q_me'] ** 2 / (vq * 1e6)
                flag = ''
                if vloss < best[0]:
                    best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()})
                    flag = ' *'
                    torch.save(dict(state=best[1], step=step, vloss=vloss), ckpt)
                # live routing trace for AuMgO: does the Al->Au contrast (true
                # -132 me) hold up, or decay toward 0, as coupling is applied?
                extra = ''
                if args.dataset == 'AuMgO':
                    ct = aumgo_analysis(data, va, qv)['contrast_pred'] * 1000
                    extra = f"  contrast {ct:+6.0f}"
                print(f"  step {step:5d}  val E {rm['E_meV_atom']:7.3f} meV/at  "
                      f"F {rm['F_meV_A']:7.1f} meV/A  q {rm['q_me']:6.2f} me{extra}{flag}",
                      flush=True)
                if ema is not None:      # resume training from the raw weights
                    model.load_state_dict(raw)

    if best[1] is not None:
        model.load_state_dict(best[1])
    rm, E_pred, q_pred = evaluate(model, data, te, dev, bs=args.eval_batch)
    sym = symmetry_check(model, data, te, dev)
    print(f"[test] {args.model}: E {rm['E_meV_atom']:.3f} meV/atom | "
          f"F {rm['F_meV_A']:.1f} meV/A | q {rm['q_me']:.2f} me | rot-inv {sym:.2e}")

    res = dict(dataset=args.dataset, model=args.model, seed=args.seed,
               params=n_par, test=rm, rot_invariance=sym,
               config=vars(args))
    if args.dataset == 'NaCl':
        res['nacl'] = nacl_analysis(data, te, q_pred, E_pred)
        for k, v in res['nacl'].items():
            print(f"[nacl] {k}: q0 true {v['q0_true_mean']*1000:+.0f}±{v['q0_true_std']*1000:.0f} me "
                  f"pred {v['q0_pred_mean']*1000:+.0f}±{v['q0_pred_std']*1000:.0f} me | "
                  f"q0 RMSE {v['q0_rmse_me']:.1f} me corr {v['q0_corr']:.3f} | "
                  f"E {v['E_rmse_meV_atom']:.2f} meV/at")
    if args.dataset == 'Ag_cluster':
        res['ag'] = ag_analysis(data, te, q_pred, E_pred)
        for k, v in res['ag'].items():
            print(f"[ag] {k}: q RMSE {v['q_rmse_me']:.1f} me | E {v['E_rmse_meV_atom']:.2f} meV/at")
    if args.dataset == 'AuMgO':
        res['aumgo'] = aumgo_analysis(data, te, q_pred)
        a = res['aumgo']
        print(f"[aumgo] Au2 net charge: doped true {a['doped_true']*1000:+.0f} me "
              f"pred {a['doped_pred']*1000:+.0f} me | undoped true "
              f"{a['undoped_true']*1000:+.0f} me pred {a['undoped_pred']*1000:+.0f} me | "
              f"separation true {a['sep_true']*1000:+.0f} pred {a['sep_pred']*1000:+.0f} me | "
              f"per-struct corr {a['corr']:.3f} | "
              f"CONTRAST pred {a['contrast_pred']*1000:+.0f} true {a['contrast_true']*1000:+.0f} me")
    if args.dataset == 'Carbon_chain':
        res['chain'] = carbon_chain_analysis(data, te, q_pred)
        c = res['chain']
        if 'farH_separation_true' in c:
            print(f"[chain] far-end H charge separation (cation-neutral): "
                  f"true {c['farH_separation_true']*1000:+.1f} me | "
                  f"pred {c['farH_separation_pred']*1000:+.1f} me | "
                  f"farH corr (cation) {c['cation']['farH_corr']:.3f}")

    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    torch.save(dict(state=model.state_dict(), test_idx=te,
                    E_pred=E_pred, q_pred=q_pred), out.replace('.json', '.pt'))
    print(f"[saved] {out}")


if __name__ == '__main__':
    main()
