"""Sanity checks for round 10: triple 8-way bridge + the 26-term master identity.

  M1  card_triple_inter_eq: |M_q ^ M_r ^ M_s| = sum of 8 disjoint side classes
      (LLL..RRR), pairwise exclusive by slot cap.
  M2  three_gear_master (END-TO-END, subtraction-free, 26 filter-card terms):
      distinct + 12 pair side classes = 6 single side classes + 8 triple side
      classes. Every term is one CRT class; floor forms already verified in
      assembly_check.py.
"""

from itertools import product
from sympy import primerange

def side(p, ch, k):
    return (6 * k - 1) % p == 0 if ch == 'L' else (6 * k + 1) % p == 0

def cnt(spec, t):
    return sum(1 for k in range(1, t + 1)
               if all(side(p, ch, k) for p, ch in spec))

trials = [(5, 7, 11), (7, 11, 13), (5, 11, 13), (5, 7, 13), (11, 13, 17)]
ts = [0, 50, 385, 1001, 5005]
m1 = m2 = 0
for (q, r, s) in trials:
    for t in ts:
        Mq = {k for k in range(1, t + 1) if side(q, 'L', k) or side(q, 'R', k)}
        Mr = {k for k in range(1, t + 1) if side(r, 'L', k) or side(r, 'R', k)}
        Ms = {k for k in range(1, t + 1) if side(s, 'L', k) or side(s, 'R', k)}
        # M1: triple = 8 disjoint classes
        parts8 = [cnt([(q, a), (r, b), (s, c)], t) for a, b, c in product('LR', repeat=3)]
        assert sum(parts8) == len(Mq & Mr & Ms), (q, r, s, t)
        m1 += 1
        # M2: 26-term identity
        distinct = len(Mq | Mr | Ms)
        pair12 = sum(cnt([(g1, a), (g2, b)], t)
                     for (g1, g2) in [(q, r), (q, s), (r, s)]
                     for a, b in product('LR', repeat=2))
        single6 = sum(cnt([(g, a)], t) for g in (q, r, s) for a in 'LR')
        triple8 = sum(parts8)
        assert distinct + pair12 == single6 + triple8, (q, r, s, t)
        m2 += 1
print(f"M1 triple 8-way bridge: {m1} cases OK")
print(f"M2 26-term master identity: {m2} cases OK")
