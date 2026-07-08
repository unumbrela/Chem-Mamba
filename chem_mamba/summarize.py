"""Aggregate chem_mamba/results/*.json into per-dataset comparison tables."""
import glob
import json
import os
from collections import defaultdict

ORDER = ['local-Q', 'local+Q', 'ssm-iso', 'ssm']


def main():
    rows = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'results', '*.json'))):
        r = json.load(open(p))
        tag = r.get('config', {}).get('tag', '')
        key = r['model'] + (f"[{tag}]" if tag else '')
        rows[r['dataset']].setdefault(key, []).append(r)

    for ds, models in rows.items():
        print(f"\n## {ds}")
        print("| model | params | E (meV/atom) | F (meV/A) | q (me) | extra |")
        print("|---|---|---|---|---|---|")
        keys = [m for m in ORDER if m in models] + \
               sorted(k for k in models if k not in ORDER)
        for m in keys:
            rs = models[m]
            t = lambda k: sum(r['test'][k] for r in rs) / len(rs)
            extra = ''
            if 'chain' in rs[0]:
                sep = sum(r['chain'].get('farH_separation_pred', 0) for r in rs) / len(rs)
                true = rs[0]['chain'].get('farH_separation_true', 0)
                extra = f"farH sep {sep*1000:+.1f}/{true*1000:+.1f} me"
            if 'aumgo' in rs[0]:
                sep = sum(r['aumgo']['sep_pred'] for r in rs) / len(rs)
                true = rs[0]['aumgo']['sep_true']
                corr = sum(r['aumgo']['corr'] for r in rs) / len(rs)
                extra = f"Au2 dq sep {sep*1000:+.0f}/{true*1000:+.0f} me, corr {corr:.2f}"
            n_seed = len(rs)
            print(f"| {m} | {rs[0]['params']/1e3:.0f}k | {t('E_meV_atom'):.3f} | "
                  f"{t('F_meV_A'):.1f} | {t('q_me'):.2f} | {extra}"
                  f"{' (n=%d)' % n_seed if n_seed > 1 else ''} |")


if __name__ == '__main__':
    main()
