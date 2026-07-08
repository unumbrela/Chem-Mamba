"""Parser for RuNNer `input.data` files (4G-HDNNP benchmarks, Ko et al. 2021).

Converts atomic units (Bohr / Ha / Ha:Bohr) to Angstrom / eV / eV:A and caches
to a .pt file.  Each structure: positions, species indices, per-atom Hirshfeld
charges, forces, total energy, total charge, (optional) lattice.
"""
import os
import torch

BOHR = 0.529177210903        # Angstrom
HA = 27.211386245988         # eV
HA_BOHR = HA / BOHR          # eV/Angstrom


def parse_runner(path):
    """Parse a RuNNer input.data file -> list of dict structures (SI-ish units)."""
    structures = []
    cur = None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == 'begin':
                cur = dict(pos=[], elem=[], q=[], F=[], lattice=[])
            elif key == 'lattice':
                cur['lattice'].append([float(x) * BOHR for x in parts[1:4]])
            elif key == 'atom':
                cur['pos'].append([float(x) * BOHR for x in parts[1:4]])
                cur['elem'].append(parts[4])
                cur['q'].append(float(parts[5]))
                cur['F'].append([float(x) * HA_BOHR for x in parts[7:10]])
            elif key == 'energy':
                cur['E'] = float(parts[1]) * HA
            elif key == 'charge':
                cur['Q'] = float(parts[1])
            elif key == 'end':
                structures.append(cur)
                cur = None
    return structures


def to_tensors(structures):
    """Pad to max N; returns dict of tensors + mask + element vocabulary."""
    elems = sorted({e for s in structures for e in s['elem']})
    e2i = {e: i for i, e in enumerate(elems)}
    S, Nmax = len(structures), max(len(s['elem']) for s in structures)
    pos = torch.zeros(S, Nmax, 3)
    frc = torch.zeros(S, Nmax, 3)
    chg = torch.zeros(S, Nmax)
    spc = torch.zeros(S, Nmax, dtype=torch.long)
    msk = torch.zeros(S, Nmax, dtype=torch.bool)
    E = torch.zeros(S)
    Q = torch.zeros(S)
    lat = None
    if structures[0]['lattice']:
        lat = torch.zeros(S, 3, 3)
    for i, s in enumerate(structures):
        n = len(s['elem'])
        pos[i, :n] = torch.tensor(s['pos'])
        frc[i, :n] = torch.tensor(s['F'])
        chg[i, :n] = torch.tensor(s['q'])
        spc[i, :n] = torch.tensor([e2i[e] for e in s['elem']])
        msk[i, :n] = True
        E[i] = s['E']
        Q[i] = round(s['Q'])
        if lat is not None:
            lat[i] = torch.tensor(s['lattice'])
    out = dict(pos=pos, forces=frc, charges=chg, species=spc, mask=msk,
               energy=E, Qtot=Q, elems=elems)
    if lat is not None:
        out['lattice'] = lat
    return out


def load_dataset(name, root=None):
    """Load one of: Carbon_chain, Ag_cluster, NaCl, AuMgO. Cached."""
    root = root or os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets')
    cache = os.path.join(root, name, 'cache.pt')
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    data = to_tensors(parse_runner(os.path.join(root, name, 'input.data')))
    torch.save(data, cache)
    return data


if __name__ == '__main__':
    for name in ['Carbon_chain', 'Ag_cluster', 'NaCl', 'AuMgO']:
        d = load_dataset(name)
        n = d['mask'].sum(1)
        print(f"\n=== {name} ===  {len(d['energy'])} structures, elems={d['elems']}, "
              f"periodic={'lattice' in d}")
        for nn_ in n.unique():
            sel = n == nn_
            comp = {d['elems'][int(z)]: int((d['species'][sel][0][:nn_] == z).sum())
                    for z in d['species'][sel][0][:nn_].unique()}
            print(f"  N={int(nn_):3d}: {int(sel.sum()):5d} structs, comp={comp}, "
                  f"Qtot={sorted(set(d['Qtot'][sel].tolist()))}, "
                  f"E range [{d['energy'][sel].min():.2f},{d['energy'][sel].max():.2f}] eV, "
                  f"sum(q) range [{d['charges'][sel].sum(1).min():+.3f},{d['charges'][sel].sum(1).max():+.3f}]")
        print(f"  |F| mean {d['forces'][d['mask']].norm(dim=-1).mean():.3f} eV/A, "
              f"pos extent {(d['pos'][d['mask']].max(0).values - d['pos'][d['mask']].min(0).values).tolist()}")
