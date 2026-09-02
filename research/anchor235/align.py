# anchor 2,3,5. condition: n lies in the clean END zone of every gear q with 37 <= q <= sqrt(n)
# (n within h_q of a multiple of 30q, h_q = 7q for q = +-7 mod 30 else q). Exact search.
import numpy as np
from math import isqrt, prod
X = 10_000_000
sieve = np.ones(isqrt(X) + 1, dtype=bool); sieve[:2] = False
for i in range(2, isqrt(isqrt(X)) + 1):
    if sieve[i]: sieve[i*i::i] = False
gears = [int(q) for q in np.flatnonzero(sieve) if q >= 37]
def h(q): return 7 * q if q % 30 in (7, 23) else q
n = np.arange(X + 1)
inzone = {q: (lambda r, q=q: (r < h(q)) | (r > 30 * q - h(q)))(n % (30 * q)) for q in gears}
ok = np.ones(X + 1, dtype=bool)
depth = np.zeros(X + 1, dtype=np.int16)
for q in gears:
    need = n >= q * q          # gear q is required from q^2 on
    ok &= ~need | inzone[q]
    depth += inzone[q] & need
good = np.flatnonzero(ok & (n >= 37 * 37))
print(f"n in [1369, {X}] inside the end zone of EVERY gear 37 <= q <= sqrt(n): {len(good)}" + (f"; they are {good[:20].tolist()}" if len(good) else ""))
# how far can the condition be pushed: max number of required gears all satisfied, and first failure
req = np.zeros(X + 1, dtype=np.int16)
for q in gears: req += (n >= q * q)
short = req - depth   # number of required gears whose end zone misses n
print("best n by 'required gears missed' (0 = fully aligned):")
for miss in range(0, 4):
    idx = np.flatnonzero((short == miss) & (n >= 1369))
    if len(idx):
        print(f"  missed {miss}: {len(idx)} numbers; largest {idx[-1]} (requires {int(req[idx[-1]])} gears), first {idx[:5].tolist()}")
# density prediction: fraction of n with all required zones = prod (2h_q/30q) over q <= sqrt(n)
for s in (41, 53, 71, 101):
    gs = [q for q in gears if q <= s]
    print(f"  expected fraction of n near {s*s} in all end zones of gears up to {s}: {prod(2*h(q)/(30*q) for q in gs):.2e} ({len(gs)} gears)")
