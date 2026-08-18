"""Harvester round 15: settle the padded-link FRAME CONFLICT with an explicit example.

CLAIM A (mine, round 14, HALVED coordinates n where the pair is (2n+1, 2n+1+2e)):
  for 3 not dividing e every survivor sits in one class mod 3, so every gap is
  divisible by 3, and a padded link (gap = 0 mod q') costs at least 3q'.
CLAIM B (lateral/constructor/mechanic, SLOT frame, slot k = pair (6k-1, 6k+1)):
  a twin padded link needs a gap of EXACTLY q' in M.

This script exhibits one real twin padded link at machine 31, probe q' = 37 and
writes it out in ALL THREE frames (slot, halved, member) so the units are pinned.
Then it measures the supply rate to cross-check mechanic's full-period census,
and re-states the twins vs d = 0 mod 6 contrast in ONE consistent unit.
"""
import numpy as np
from math import prod

GEARS31 = [5, 7, 11, 13, 17, 19, 23, 29, 31]
QP = 37

def slot_survivors(gears, lo, hi):
    n = hi - lo
    a = np.ones(n, bool)
    for q in gears:
        u = pow(6, -1, q)
        for r in (u % q, (-u) % q):
            s = (r - lo) % q
            a[s::q] = False
    return np.flatnonzero(a) + lo

u37 = pow(6, -1, QP)
teeth = sorted({u37 % QP, (-u37) % QP})
print(f"machine 31 gears {GEARS31}; probe q' = {QP}; 6^-1 = {u37} mod {QP}; "
      f"teeth at k = {teeth} mod {QP}\n")

LIMIT, CH = 60_000_000, 5_000_000
found, n_gap37, n_padded, n_surv = None, 0, 0, 0
prev = None
for lo in range(0, LIMIT, CH):
    S = slot_survivors(GEARS31, lo, min(lo + CH, LIMIT))
    if prev is not None:
        S = np.concatenate([[prev], S])
    g = np.diff(S)
    n_surv += len(S) - (1 if prev is not None else 0)
    idx = np.flatnonzero(g == QP)
    n_gap37 += len(idx)
    for i in idx:
        k1 = int(S[i])
        if k1 % QP in teeth:
            n_padded += 1
            if found is None:
                found = (k1, int(S[i + 1]), int(S[i - 1]) if i > 0 else None,
                         int(S[i + 2]) if i + 2 < len(S) else None)
    prev = int(S[-1])

k1, k2, kprev, knext = found
print("A CONCRETE TWIN PADDED LINK (first one found):")
print(f"  SLOT frame   : k1 = {k1}, k2 = {k2}, gap = {k2 - k1} slots = q' = {QP}")
print(f"                 k1 mod {QP} = {k1 % QP}, k2 mod {QP} = {k2 % QP}  "
      f"(same tooth -> ZERO letter)")
print(f"  HALVED frame : n1 = {3*k1 - 1}, n2 = {3*k2 - 1}, "
      f"gap = {3*(k2 - k1)} = 3q' = {3*QP}")
print(f"  MEMBER frame : pair1 = ({6*k1 - 1}, {6*k1 + 1}), "
      f"pair2 = ({6*k2 - 1}, {6*k2 + 1})")
print(f"                 member gap = {6*(k2 - k1)} = 6q' = {6*QP}")
kill1 = [m for m in (6*k1 - 1, 6*k1 + 1) if m % QP == 0]
kill2 = [m for m in (6*k2 - 1, 6*k2 + 1) if m % QP == 0]
print(f"  the kills    : {QP} divides {kill1[0]} (= {kill1[0]//QP}x{QP}) and "
      f"{kill2[0]} (= {kill2[0]//QP}x{QP})")
print(f"  consecutive? : neighbours are k = {kprev} and {knext}; "
      f"no machine-31 survivor strictly between k1 and k2 (gap == {QP} exactly)")
print(f"  all gaps divisible by 3 in HALVED units: "
      f"3*{k2-k1} = {3*(k2-k1)}, and {3*(k2-k1)} % 3 = {3*(k2-k1) % 3}\n")

P = prod(GEARS31)
rate37, ratepad = n_gap37 / LIMIT, n_padded / LIMIT
print(f"SUPPLY RATE over {LIMIT:,} slots ({n_surv:,} survivors): "
      f"gaps == {QP}: {n_gap37:,}; padded links (endpoint on a tooth): {n_padded:,}")
print(f"  extrapolated to the full period P = {P:,}: "
      f"gaps=={QP} ~ {rate37 * P:,.0f}; padded ~ {ratepad * P:,.0f}   "
      f"(mechanic's census: 26,366)\n")

print("THE CONTRAST, RESTATED IN ONE UNIT (members):")
for lbl, e, r3 in (("twins / 3 does not divide e", 1, 2), ("d = 0 mod 6 (3 | e)", 3, 1)):
    dens = 0.5 * (3 - r3) / 3 * prod((q - 2) / q for q in [5, 7, 11, 13, 17, 19, 23, 29, 31])
    lam = 1 / dens
    cost = (6 if r3 == 2 else 2) * QP
    print(f"  {lbl:<28} padded link costs {cost:>4} members = "
          f"{cost/QP:.0f}q'; mean gap {lam:6.2f} members; cost/lambda = {cost/lam:5.2f}")
