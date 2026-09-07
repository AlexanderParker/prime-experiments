"""The top-run share of the window, tau(y), at every prime rung y in [23, YMAX], and the descent
check (E-P2) at (q, Q) = (5, 10^4) for turns m <= 60.

tau(y) = (blocked run of the window of {5..y} ending at the window's top column W, in columns) / (window length)
window at rung y: columns (y/6, W], W = (y'^2 - 1)/6, y' the next prime; openings there = twin columns with
6k - 1 > y, plus column W when y'^2 - 2 is prime (position_frontier.md, reduction (R)).
Turn m of Q is the top slice of the window at y_m = sqrt((m + 1)Q): empty iff the top run at that scale
is >= Q/6 columns. Existence in the valve at Q <=> some m has tau-like top run < Q/6.

usage: uv run python research/valves/r2/top_run.py [YMAX]
"""
import sys, os, math, json
import numpy as np

YMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 19997
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s)


small = primes_upto(2 * YMAX + 100)
ynext = int(small[np.searchsorted(small, YMAX, side="right")])
N = ynext * ynext + 4                      # sieve odd numbers to N
half = N // 2 + 1                          # odd n = 2i + 1, i < half
odd = np.ones(half, dtype=np.uint8); odd[0] = 0   # n = 1 is not prime
for p in primes_upto(int(N ** 0.5) + 1):
    if p == 2:
        continue
    start = (p * p) // 2
    odd[start::p] = 0
twin = np.flatnonzero(odd[:-1] & odd[1:])  # index i: n = 2i + 1 prime and n + 2 prime
twin_n = 2 * twin + 1                       # lower members
print(f"sieved odd numbers to {N}: {int(odd.sum())} odd primes, {len(twin_n)} twins (lower member <= {N - 2})")

# --- tau at every prime rung y in [23, YMAX]
rungs = small[(small >= 23) & (small <= YMAX)]
rows = []
for y in rungs:
    yp = int(small[np.searchsorted(small, y, side="right")])
    W = (yp * yp - 1) // 6
    klo = y // 6                            # window = columns (y/6, W]
    # last open column <= W: twin columns (n + 1)/6 with n > y; or W itself if yp^2 - 2 prime
    j = np.searchsorted(twin_n, 6 * W - 1, side="right") - 1
    n_last = int(twin_n[j])
    k_last = (n_last + 1) // 6
    top_prime = bool(odd[(yp * yp - 2) // 2])
    if top_prime:
        k_last = W
    L_top = W - k_last
    Wlen = W - klo
    rows.append((int(y), yp, W, Wlen, k_last, L_top, L_top / Wlen))
rows_arr = np.array([(r[0], r[6], r[5], r[3]) for r in rows])
order = np.argsort(-rows_arr[:, 1])
print("top-run share tau(y) = L_top / window length; the ten largest:")
for i in order[:10]:
    y, tau, L, Wl = rows[int(i)][0], rows[int(i)][6], rows[int(i)][5], rows[int(i)][3]
    print(f"  y={y:6d} y'={rows[int(i)][1]:6d} W_len={Wl:9d} L_top={L:5d} tau={tau:.6f}  1/tau={1/tau if tau > 0 else float('inf'):.1f}")
for lo, hi in [(23, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10000), (10000, YMAX + 1)]:
    sel = [r for r in rows if lo <= r[0] < hi]
    taus = np.array([r[6] for r in sel]); Ls = np.array([r[5] for r in sel])
    print(f"  band [{lo},{hi}): rungs {len(sel)}, max tau {taus.max():.6f} at y={sel[int(taus.argmax())][0]}, "
          f"median tau {np.median(taus):.2e}, max L_top {Ls.max()} cols, rungs with L_top=0: {(Ls == 0).sum()}")
# certified turns by the truth: the largest m with tau < 1/(m+1), per band (min over rungs of floor(1/tau) - 1)
for lo, hi in [(23, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10000), (10000, YMAX + 1)]:
    sel = [r for r in rows if lo <= r[0] < hi]
    mmax = min((math.floor(r[3] / r[5]) - 1) if r[5] > 0 else 10 ** 9 for r in sel)
    print(f"  band [{lo},{hi}): every rung's top slice of fraction 1/(m+1) holds a twin for all m <= {mmax}")
with open(os.path.join(outdir, f"top_run_{YMAX}.json"), "w") as f:
    json.dump([dict(y=int(r[0]), ynext=int(r[1]), W=int(r[2]), Wlen=int(r[3]), k_last=int(r[4]), L_top=int(r[5]), tau=float(r[6])) for r in rows], f)

# --- E-P2: the descent as sets at (5, 10^4), m <= 60: twins in (mQ, (m+1)Q] == openings of {5..y_m} there
Q = 10 ** 4; q = 5
bad = 0; checked = 0
for m in range(1, 61):
    lo_n, hi_n = m * Q, (m + 1) * Q
    ym = math.isqrt(hi_n + 2)                        # {5..ym} = primes <= sqrt((m+1)Q + 2); nextprime(ym)^2 > (m+1)Q + 2
    gears = [int(p) for p in small if 5 <= p <= ym]
    ns = np.arange(lo_n + 1, hi_n + 1, dtype=np.int64)
    ns = ns[(ns % 6) == 5]                           # columns: 6k - 1
    op = np.ones(len(ns), dtype=bool)
    for g in gears:
        op &= (ns % g != 0) & ((ns + 2) % g != 0)
    openings = set(ns[op].tolist())
    tw = set(twin_n[(twin_n > lo_n) & (twin_n <= hi_n)].tolist())
    checked += 1
    if openings != tw:
        bad += 1
        print(f"  E-P2 MISMATCH at m={m}: openings {len(openings)} twins {len(tw)} ym={ym}")
print(f"E-P2 descent as sets at (q,Q)=(5,{Q}), turns 1..60: {checked} turns checked, {bad} mismatches "
      f"(machine {{5..sqrt((m+1)Q+2)}} openings in the slice == twins in the slice)")
