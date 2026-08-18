"""Harvester round 12: does the literal cap transfer to Polignac gap d != 2?

The twin cap mechanism (constructor's fuel_bound.py Theorem 3): a literal
chain for new gear q' is an interleaved walk r, r+2u', r+q', r+q'+2u', ...
(period 70) that must stay inside the 15-residue exposed set E mod 35;
the cap is a function of q' mod 210 only (48 invertible classes; max cap 6).

For separation-d pairs (centred frame: gear q blocks c = +-e*6^{-1} mod q,
e = d/2), the SAME architecture transfers with two d-specific inputs:
  (i)  the exposed set E_d mod 35: teeth of 5 and 7 at +-e*6^{-1};
       |E_d| = 15 generic, 20 if 5|e, 18 if 7|e, 24 if 35|e;
  (ii) the walk step: u'_d(q') = min positive representative of
       +-e*6^{-1} mod q' (twin case e=1: u' = round(q'/6)).
Frame caveat: for d = 0 mod 6 (3 | e) gear 3 keeps TWO free classes - the
single-slot-frame collapse fails and the walk lives mod 105 (two subframes);
here we verify the transfer for d != 0 mod 6.

CHECKS:
  T1  class invariance: cap(q') computed directly from the walk equals the
      cap of q' mod 210's class representative, for every prime q' in
      (d, 2000] - the finite-check structure (48 classes) survives per d.
  T2  cap tables per d: the analog of {2:24, 3:4, 4:14, 6:6} and the
      absolute cap.
  T3  the mod-3 endpoint core (round-12 Lean target), numeric: for offsets a
      and x, y both avoiding the adjacent pair {a, a+1} mod 3: x = y mod 3;
      gap multiple of 3; run length = 2 mod 3.
"""
import numpy as np
from sympy import primerange
from collections import Counter

def exposed_d(e, m):
    """Exposed residues mod m (m = 35) for separation-2e pattern, gears 5,7."""
    a = np.ones(m, bool)
    for q in (5, 7):
        c = (e * pow(6, -1, q)) % q
        a[c::q] = False
        a[(q - c) % q::q] = False
    return a

def u_d(e, q1):
    c = (e * pow(6, -1, q1)) % q1
    return min(c, q1 - c)

def literal_cap_d(e, q1, E):
    s1 = (2 * u_d(e, q1)) % 35
    best = 0
    for r in range(35):
        for phase in (0, 1):
            run = mx = 0
            for i in range(140):
                j, par = divmod(i + phase, 2)
                pos = (r + j * q1 + (s1 if par else 0)) % 35
                if E[pos]:
                    run += 1
                    mx = max(mx, run)
                else:
                    run = 0
            best = max(best, mx)
    return best

print("d (sep) | e | |E_d| | class-invariance mod 210 | cap spectrum | max cap")
for d in (2, 4, 8, 10, 14, 16, 20, 28):   # d != 0 mod 6; includes 5|e (d=10,20), 7|e (d=14,28)
    e = d // 2
    E = exposed_d(e, 35)
    caps_by_class = {}
    fails = 0
    tested = 0
    for q1 in primerange(max(11, d + 1), 2000):
        cap = literal_cap_d(e, q1, E)
        cls = q1 % 210
        if cls in caps_by_class and caps_by_class[cls] != cap:
            fails += 1
        caps_by_class.setdefault(cls, cap)
        tested += 1
    spec = Counter(caps_by_class.values())
    print(f"  d={d:>2}  | {e} |  {int(E.sum()):>2}  | "
          f"{'OK' if fails == 0 else f'{fails} MISMATCHES'} ({tested} primes, "
          f"{len(caps_by_class)} classes) | {dict(sorted(spec.items()))} | "
          f"{max(caps_by_class.values())}")

# T3: mod-3 endpoint core
bad = 0
for a in range(3):
    for x in range(60):
        for y in range(x + 1, 120):
            fx = x % 3 != a and x % 3 != (a + 1) % 3
            fy = y % 3 != a and y % 3 != (a + 1) % 3
            if fx and fy:
                if x % 3 != y % 3 or (y - x) % 3 != 0 or (y - x - 1) % 3 != 2:
                    bad += 1
print(f"T3 mod-3 endpoint core (all offsets, x<y<120): {'OK' if bad == 0 else f'{bad} FAILS'}")
