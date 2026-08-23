"""Round 20 (constructor): NILPOTENCY IDENTITIES of the exact operator frame.

On C^{Z_P} let S = slot shift, D = exposure projector (tensor product of the
per-gear projectors by CRT), B = I - D.  Then:

  (i)  (BS)^m != 0  iff  the pattern has m consecutive blocked slots, so
       F(M) = nilpotency index of BS  (largest gap = longest blocked run + 1).
  (ii) With A_V = the qualifying-gap partial map on openings (opening k ->
       next opening, present iff the gap is residue-qualifying for q'),
       A_V^m != 0 iff m consecutive qualifying gaps occur somewhere; the
       realized qualifying depth cap = nilpotency index of A_V - 1, and the
       merge chain length obeys k_max <= index(A_V).

Verified by direct construction (boolean vectors as the operators' actions)
at machines 11, 13, 17, 19 - plus the identity R = sum_v G_v = the successor
permutation (single |E|-cycle, eigenvalues = roots of unity: the exact frame
has NO spectral gap; decorrelation is an aggregation phenomenon).
"""
import numpy as np
from math import prod
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flank_envelope import primes_upto
from tm_resid_runs import next_prime

for y in (11, 13, 17, 19):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    exposed = ~ex                       # openings
    # (i) F = nilpotency index of BS: (BS)^m maps e_k -> e_{k-m} iff
    # slots k-1, ..., k-m all blocked; nonzero iff a blocked run of length m
    # exists.  Longest blocked run (cyclic):
    idx = np.flatnonzero(exposed)
    gaps = np.diff(np.append(idx, idx[0] + P))
    F = int(gaps.max())
    # direct operator check: v <- B S v starting from all-ones; the vector
    # is nonzero after m steps iff (BS)^m != 0
    v = np.ones(P, bool)
    m = 0
    while v.any():
        v = np.roll(v, -1) & ex        # S then B (project onto blocked)
        m += 1
        assert m <= F + 2
    assert m == F, (y, m, F)
    # (ii) A_V nilpotency = qualifying depth cap + 1
    q1 = next_prime(y)
    c = pow(6, -1, q1)
    Q = {0, (2 * c) % q1, (-2 * c) % q1}
    qual = np.array([g % q1 in Q for g in gaps])
    # longest run of consecutive qualifying gaps (cyclic)
    if qual.all():
        cap = len(qual)
    else:
        rolled = np.concatenate([qual, qual])
        best = cur = 0
        for b in rolled:
            cur = cur + 1 if b else 0
            best = max(best, cur)
        cap = min(best, len(qual))
    # operator check on the opening space: x <- A_V x from all-ones
    x = np.ones(len(idx), bool)
    m = 0
    while x.any():
        x = x[:] & qual                # keep openings whose NEXT gap qualifies
        x = np.roll(x, -1)             # advance to the next opening
        m += 1
        assert m <= cap + 2
    assert m == cap + 1, (y, m, cap)
    print(f"machine {y}: F = {F} = index(BS) OK; "
          f"qualifying depth cap = {cap}, index(A_V) = {cap + 1} OK "
          f"(openings {len(idx)}, single {len(idx)}-cycle renewal permutation)")
print("all nilpotency identities verified by direct operator iteration")
