"""
Capacity-matched isolation of the NON-LOCAL benefit.

Compares the SAME SSM module (identical parameters) with cross-atom mixing OFF
vs ON, on two references:
  short-only  : no long-range term -> global reach should NOT help  (control)
  long-range  : non-local Coulomb  -> global reach SHOULD help       (test)

Clean signature of a genuine non-local benefit (not just capacity):
  global ~= isolated  on short-only,  AND  global < isolated  on long-range.

Run:  cd Chem-Mamba && PYTHONPATH=. python chem_mamba/ablate_nonlocal.py
"""
import argparse
import torch
from chem_mamba.train_m1 import train, evaluate, to_dev
from chem_mamba.data3d import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--N', type=int, default=64)
    ap.add_argument('--n', type=int, default=2000)
    args = ap.parse_args()
    print(f"N={args.N}  steps={args.steps}   (SSM params identical; only reach differs)\n")

    rows = []
    for tag, lr in [('short-only (control)', False), ('long-range (test)', True)]:
        tr = to_dev(make_dataset(args.n, args.N, 1, lr))
        va = to_dev(make_dataset(300, args.N, 2, lr))
        te = to_dev(make_dataset(500, args.N, 3, lr))
        res = {}
        for label, mix in [('isolated', False), ('global', True)]:
            model = train(True, tr, va, args.steps, bb_kw={'mix': mix})
            res[label] = evaluate(model, te)
            del model; torch.cuda.empty_cache()
        rows.append((tag, res))

    print(f"\n{'reference':<24}{'SSM reach':<11}{'E MAE/atom':>12}{'F MAE':>10}")
    print('-' * 57)
    for tag, res in rows:
        for label in ['isolated', 'global']:
            e, f = res[label]
            print(f"{tag:<24}{label:<11}{e:>12.4f}{f:>10.4f}")
        ei, _ = res['isolated']; eg, _ = res['global']
        fi, ff = res['isolated'][1], res['global'][1]
        print(f"{'  global vs isolated:':<35}{'ΔE ' + f'{100*(1-eg/ei):+.0f}%':>12}"
              f"{'ΔF ' + f'{100*(1-ff/fi):+.0f}%':>13}\n")
    print("Non-local benefit is REAL iff long-range shows a clear negative ΔE/ΔF")
    print("(global better) while short-only shows ~0 (global no better than isolated).")


if __name__ == "__main__":
    main()
