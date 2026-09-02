# anchor 2,3,5. gear q >= 37: clean end zone = numbers n with n mod 30q in (m_max*q, 30q) U [0, m_min*q):
# open interval around every multiple of 30q, half-width h(q) = q (m-set has 1, 29) or 7q (q = +-7 mod 30).
from math import gcd
OPEN = {1, 11, 13, 17, 19, 29}
def mset(q): return sorted(m for m in range(1, 30) if (q * m) % 30 in OPEN)
def half(q): return mset(q)[0] * q          # first hit at m_min*q; last at m_max*q = 30q - m_min*q by symmetry
gears = [37, 41, 43, 47, 53, 59, 61, 67, 71]
print("gear  m_min  half-width  period  clean zone around every multiple of 30q")
for q in gears:
    print(f"{q:>4}  {mset(q)[0]:>5}  {half(q):>10}  {30*q:>6}  (30q*t - {half(q)}, 30q*t + {half(q)})")
print()
print("first few zones on the number line (centre +- half-width):")
for q in gears[:5]:
    zs = [(30*q*t - half(q), 30*q*t + half(q)) for t in range(1, 5)]
    print(f"  {q}: " + ", ".join(f"({a},{b})" for a, b in zs))
print()
print("pairwise skew: consecutive gears q < q'. per period of q' the q-zone centres drift by 30(q'-q) relative to q'-zone centres;")
print("zones overlap when |30(q t - q' t')| < h(q) + h(q'); realign fully at 30 q q'.")
for q, q2 in zip(gears, gears[1:]):
    L = 30 * q * q2
    hs = half(q) + half(q2)
    overlaps = []
    for t in range(1, q2 + 1):
        c = 30 * q * t
        t2 = round(c / (30 * q2))
        d = c - 30 * q2 * t2
        if abs(d) < hs and t2 >= 1:
            overlaps.append((t, t2, d))
    print(f"  {q},{q2}: drift {30*(q2-q)} per period, realign at {L}; overlaps per realignment {len(overlaps)} of {q2} periods of {q}: "
          + ", ".join(f"t={t},t'={t2},offset {d:+d}" for t, t2, d in overlaps[:6]) + (" ..." if len(overlaps) > 6 else ""))
print()
# meta sieve: numbers up to X inside the clean zone of every gear in a set
import numpy as np
X = 30 * 37 * 41 * 43 * 2
n = np.arange(X + 1)
def inzone(q):
    r = n % (30 * q); h = half(q)
    return (r < h) | (r > 30 * q - h)
allz = inzone(37)
for q in [41, 43]:
    allz &= inzone(q)
idx = np.flatnonzero(allz)
print(f"numbers <= {X} inside the clean zones of 37, 41 and 43 at once: {len(idx)}; density {len(idx)/X:.2e} (product of widths {(7/15)*(1/15)*(1/15):.2e})")
# runs
runs = []
start = idx[0]; prev = idx[0]
for i in idx[1:]:
    if i != prev + 1:
        runs.append((int(start), int(prev))); start = i
    prev = i
runs.append((int(start), int(prev)))
print("  as runs: " + ", ".join(f"[{a},{b}]" for a, b in runs[:12]) + (" ..." if len(runs) > 12 else ""))
allz &= inzone(47) & inzone(53)
idx2 = np.flatnonzero(allz)
print(f"with 47 and 53 as well: {len(idx2)} numbers; first ones {idx2[:10].tolist()}, and around {30*37*41*43*47*53} (30*37*41*43*47*53) all five align")
