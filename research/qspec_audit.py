"""Round 21 (mechanic): AUDIT of the r17 C13 qualifying-spectrum table.

For each (machine, q') the tool qualifying_spectrum.py prints Q_j with an
ADDRESS.  This script goes to each address, re-derives the window from the raw
opening predicate, and asserts that the window really is j consecutive gaps
whose j-2 middle gaps all meet the floor a = 2u', with the claimed sum.  That
decides, independently of any scan tool, whether the tool's value or the C13
table's value is the correct one.
"""
from math import prod

import numpy as np


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def openings_near(y, k0, span=400):
    gears = [p for p in primes_upto(y) if p >= 5]
    ks = np.arange(k0 - span, k0 + span)
    ex = np.zeros(len(ks), bool)
    for g in gears:
        u = pow(6, -1, g)
        ex |= ((ks % g) == (u % g)) | ((ks % g) == ((-u) % g))
    return ks[~ex]


def check(y, qp, j, k0, claimed):
    a = 2 * round(qp / 6)
    op = openings_near(y, k0)
    i = int(np.where(op == k0)[0][0])
    w = op[i:i + j + 1]
    gaps = np.diff(w)
    assert len(gaps) == j, f"{y}->{qp} j={j}: not enough openings"
    mids = gaps[1:-1]
    ok_mid = bool((mids >= a).all())
    total = int(gaps.sum())
    status = "OK" if (total == claimed and ok_mid) else "MISMATCH"
    print(f"  machine {y} -> q'={qp}  j={j}  k={k0:,}  gaps {list(int(g) for g in gaps)}"
          f"  sum {total} (claimed {claimed})  middles>={a}: {ok_mid}   {status}")
    assert status == "OK"


# (y, q', j, address, tool's Q_j)  - the entries where C13 and the tool disagree
CASES = [
    (11, 13, 4, 115, 18),
    (11, 13, 5, 115, 20),
    (13, 17, 4, 810, 23),
    (17, 19, 4, 32832, 31),
    (17, 19, 5, 1293, 32),
    (17, 19, 6, 9173, 34),
    (23, 29, 3, 14995460, 43),
    (23, 29, 5, 8057950, 55),
    (23, 29, 6, 8057950, 60),
]

print("AUDIT of disputed C13 qualifying-spectrum entries "
      "(tool value verified at its own address):")
for y, qp, j, k0, claimed in CASES:
    check(y, qp, j, k0, claimed)
print("ALL DISPUTED ENTRIES VERIFIED AT THEIR ADDRESSES - the tool's values are"
      " correct and the r17 C13 table entries are wrong.")
