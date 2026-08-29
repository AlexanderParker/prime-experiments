"""Round 26 (mechanic): the gear-5 transform of the EXACT, CYCLICALLY CLOSED
full-period gap histograms - the object Lateral's U6/U9 wait on.

H_5(1) = sum_g n_g e^{2 pi i g / 5} over the whole period.  Round 21 measured
arg H_5(1) = +126 deg +- 2 at seven machines and called it a machine-independent
phase; rounds 21-25 could only do it from LINEAR-CLOSE histograms (short by the
wrap gap) and, above m31, only from PREFIXES.  With research/ghist_transfer.py's
exact cyclic histograms it is exact at every machine through 41.

usage: <venv>/python research/gear5_transform_r26.py research/data/r26/ghist_*.csv
"""
import csv
import cmath
import math
import sys

print("  machine   arg H_5(1)     |H_5(1)|/H0   mean gap   1.015/mean   "
      "mod-5 class counts")
rows = []
for path in sys.argv[1:]:
    h, y = {}, None
    for r in csv.DictReader(open(path)):
        y = int(r['y'])
        h[int(r['gap'])] = int(r['count'])
    H0 = sum(h.values())
    H1 = sum(c * cmath.exp(2j * math.pi * g / 5) for g, c in h.items())
    mean = sum(g * c for g, c in h.items()) / H0
    N = [0] * 5
    for g, c in h.items():
        N[g % 5] += c
    arg = math.degrees(cmath.phase(H1))
    rows.append((y, arg, abs(H1) / H0, mean, N))
    print(f"   m{y:<3d}    {arg:+9.4f}      {abs(H1)/H0:9.6f}   {mean:8.4f}   "
          f"{1.015/mean:9.6f}   {N}")
rows.sort()
print("\n  arg ladder:", ", ".join(f"{r[1]:.3f}" for r in rows))
print("  amplitude :", ", ".join(f"{r[2]:.5f}" for r in rows))
print("\nNOTE for Lateral: these are CYCLIC totals.  A linear-close histogram")
print("is short by exactly one gap, of size (the period's FIRST gap), so its")
print("mod-5 class counts differ from these in exactly one class - the class")
print("of that first gap.  At m31 the first gap is 7, so class 2 here reads")
print("2,069,637,132 against the round-21 linear value 2,069,637,131.")
