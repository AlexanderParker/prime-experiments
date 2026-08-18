"""Round 13 lateral: the EXCESS LAW - why excess climbs, and predictions.

From merge_decompose: F(M+q') = max over words w (incl. the empty word, k=1)
of [span(w) + FS_max(w;M)], and F2 = the k=1 term. Hence

    excess = F_new - F2 = max_{w != empty} [ span(w) - deficit(w) ],
    deficit(w) := F2 - FS_max(w;M)   (an extreme-value deficit: fewer
                                      occurrences -> worse best flank sum)

so the excess is a race: span grows LINEARLY in q' (span of the two k=2 words
is exactly {(q'-1)/3, (2q'+1)/3} or the mirror pair), while deficit grows only
LOGARITHMICALLY in the occurrence ratio. Fitted here, then used to predict
37->41 and 41->43 under both hypotheses.
"""
from math import prod, log

import numpy as np

from split_gap_law import primes

# (machine, F2, word-span, occurrences, FS_max) from merge_decompose.py output
DATA = [
    (13, 16, 6, 60, 12), (13, 16, 11, 12, 7),
    (17, 25, 13, 66, 12), (17, 25, 6, 1022, 17),
    (19, 31, 8, 10462, 25), (19, 31, 15, 1236, 17), (19, 31, 23, 31, 11),
    (23, 39, 10, 243370, 33), (23, 39, 19, 440, 18),
    (29, 55, 21, 205068, 30), (29, 55, 10, 7815770, 48),
    (29, 55, 31, 6500, 24), (29, 55, 41, 4, 14),
]

def machine(y):
    g = primes(5, y)
    return prod(g), prod(q - 2 for q in g)

print("DEFICIT LAW: deficit = F2 - FS_max  vs  x = ln(openings / occurrences)")
xs, ds = [], []
for y, F2, span, n, fs in DATA:
    P, O = machine(y)
    x = log(O / n)
    d = F2 - fs
    xs.append(x); ds.append(d)
    print(f"  machine {y:>2}: span {span:>2}, occ {n:>8}, FS_max {fs:>2}, "
          f"deficit {d:>2}, x = {x:>5.2f}, mean gap {P/O:.2f}")
A = np.polyfit(xs, ds, 1)
res = np.array(ds) - np.polyval(A, xs)
print(f"  fit: deficit = {A[0]:.2f} * x + {A[1]:.2f}   "
      f"(residual sd {res.std():.1f}, max |res| {abs(res).max():.0f})")

print()
print("GAP-VALUE ABUNDANCE (occurrences of a k=2 word = count of that gap value)")
def gapdist(y, chunk=100_000_000):
    gears = primes(5, y)
    P = prod(gears)
    cnt = np.zeros(200, np.int64)
    carry = None; a = 0
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, bool)
        for q in gears:
            u = pow(6, -1, q)
            for t in (u, q - u):
                killed[(t - a) % q::q] = True
        o = np.flatnonzero(~killed).astype(np.int64) + a
        if carry is not None:
            o = np.concatenate((carry, o))
        d = np.diff(o)
        cnt += np.bincount(d[d < 200], minlength=200)
        carry = o[-2:]; a += S
    return cnt, P

for y in (19, 23, 29):
    cnt, P = gapdist(y)
    O = cnt.sum()
    lam = P / O
    print(f"  machine {y}: mean gap {lam:.2f}; "
          f"share of gap g: " + ", ".join(
              f"g={g}:{cnt[g]/O:.2e}" for g in (6, 10, 14, 19, 25, 27, 29)
              if cnt[g]))
