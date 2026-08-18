"""Harvester round 13, chunk 1: the EXCLUDED case d = 0 mod 6 (3 | e).

Universal frame = HALVED COORDINATES: position n, pair (2n+1, 2n+1+2e);
gear q blocks n = 0 and n = -e (mod q). Gear 2 is automatic. Gear 3 is
EXPLICIT here (the twin slot-frame quotients it away only because 3 blocks
two of three classes when 3 does not divide e; when 3 | e it blocks ONE, so
two classes survive and there is no single-lattice frame - the mod-105 case).

A literal chain = a maximal run of CONSECUTIVE frame-admissible q'-kills
(n = 0 or -e mod q', n not = 0, -e mod 3) that are all exposed to gears 5, 7.
Computed exactly over one full period 105*q' (gcd(q',105) = 1), doubled for
wraparound. This definition is frame-free and reproduces the twin caps.

CHECKS
  C1  reproduce Constructor's twin table (validation of the frame change)
  C2  3 | e: is the cap finite, and what is it?
  C3  invariance class: cap as a function of q' mod 105 (phi = 48) vs mod 210
  C4  |E_d| = prod over {3,5,7} of (q - r_q), r_q = 1 if q | e else 2 (HL factor)
"""
import numpy as np
from sympy import primerange
from collections import Counter

def exposed_mask(e, q, n):
    return (n % q != 0) & (n % q != (-e) % q)

def cap(e, q1):
    """Max literal-chain length: consecutive frame-admissible q1-kills all 5,7-exposed."""
    P = 105 * q1
    n = np.arange(2 * P)
    kill = (n % q1 == 0) | (n % q1 == (-e) % q1)
    adm3 = exposed_mask(e, 3, n)
    cand = np.flatnonzero(kill & adm3)
    if cand.size == 0:
        return 0
    ok = exposed_mask(e, 5, cand) & exposed_mask(e, 7, cand)
    best = run = 0
    for v in ok:
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)

def esize(e):
    tot = 1
    for q in (3, 5, 7):
        tot *= q - (1 if e % q == 0 else 2)
    return tot

print("d     e   3|e   |E_d| mod105  HLpred  cap spectrum by class          max  invariance")
for d in (2, 4, 6, 12, 18, 24, 30, 42, 8, 10):
    e = d // 2
    E = int(sum(1 for n in range(105)
                if all(exposed_mask(e, q, np.array([n]))[0] for q in (3, 5, 7))))
    caps105, caps210 = {}, {}
    f105 = f210 = 0
    for q1 in primerange(max(11, e + 1), 1200):
        if q1 % 3 == 0 or q1 % 5 == 0 or q1 % 7 == 0:
            continue
        c = cap(e, q1)
        for tab, mod, _ in ((caps105, 105, 0), (caps210, 210, 0)):
            k = q1 % mod
            if k in tab and tab[k] != c:
                if mod == 105:
                    f105 += 1
                else:
                    f210 += 1
            tab.setdefault(k, c)
    spec = dict(sorted(Counter(caps105.values()).items()))
    inv = f"mod105 {'OK' if f105 == 0 else str(f105)+' bad'} ({len(caps105)} cls)"
    print(f"{d:>3} {e:>4}   {'Y' if e%3==0 else 'n'}     {E:>4}       {esize(e):>4}   "
          f"{str(spec):<28} {max(caps105.values()):>3}  {inv}")
