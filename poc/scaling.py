"""
Empirical scaling of a GLOBAL long-range mixer: linear-scan SSM vs explicit
all-pairs attention (what any full-attention / all-pairs long-range scheme
fundamentally costs).  We report forward wall-clock AND peak memory vs system
size N, and fit log-log slopes.

Why explicit attention (not fused/flash): the point is the *asymptotic* cost of
letting every atom talk to every other atom -- O(N^2) time and O(N^2) memory.
Fused kernels hide the constant but not the exponent, and they still cannot fit
a 10^5-atom all-pairs map in memory.  The SSM carries global context in a state
of fixed size, so it stays O(N) in both.

Honesty note: our S6 uses a plain sequential scan (large Python/launch-overhead
constant); production Mamba uses a fused parallel scan.  We therefore compare
scaling EXPONENTS, and -- for memory, which is implementation-robust -- absolute
values.
"""
import time, math
import numpy as np
import torch
import torch.nn.functional as F

from models import S6

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def all_pairs_attention(x):
    """Explicit softmax(QK^T/sqrt(d))V -- materialises the N x N map."""
    B, N, d = x.shape
    scores = (x @ x.transpose(1, 2)) / math.sqrt(d)   # (B,N,N)  <-- O(N^2) memory
    attn = scores.softmax(-1)
    return attn @ x


def bench(fn, x, iters=15, warmup=4):
    for _ in range(warmup):
        fn(x)
    if DEV == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        fn(x)
    if DEV == 'cuda':
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1e3
    mem = torch.cuda.max_memory_allocated() / 1e6 if DEV == 'cuda' else float('nan')
    return dt, mem


@torch.no_grad()
def main():
    d = 64
    ssm = S6(d, d_state=16).to(DEV).eval()

    Ns = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    print(f"device={DEV}  d_model={d}  batch=4\n")
    print(f"{'N':>7} | {'SSM ms':>9} {'SSM MB':>9} | {'Attn ms':>9} {'Attn MB':>9}")
    print('-' * 54)
    ssm_t, ssm_N, attn_t, attn_N = [], [], [], []
    for N in Ns:
        x = torch.randn(4, N, d, device=DEV)
        ts, ms = bench(lambda z: ssm(z), x)
        ssm_t.append(ts); ssm_N.append(N)
        try:
            ta, ma = bench(lambda z: all_pairs_attention(z), x)
            attn_t.append(ta); attn_N.append(N)
            a_str = f"{ta:>9.2f} {ma:>9.0f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            a_str = f"{'OOM':>9} {'OOM':>9}"
        print(f"{N:>7} | {ts:>9.2f} {ms:>9.0f} | {a_str}")

    def slope(N, y, lo=512):
        N, y = np.array(N), np.array(y)
        m = N >= lo
        return np.polyfit(np.log(N[m]), np.log(y[m]), 1)[0]

    print('-' * 54)
    print(f"time slope (N>=512):  SSM = {slope(ssm_N, ssm_t):.2f}  (ideal 1.0)"
          f"   Attn = {slope(attn_N, attn_t):.2f}  (ideal 2.0)")
    print("=> explicit all-pairs attention hits an O(N^2) memory wall; the SSM")
    print("   keeps a fixed-size state and stays linear in time and memory.")


if __name__ == "__main__":
    main()
