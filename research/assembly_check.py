"""Sanity checks for the 3-gear assembly formalisation (Harvester r9).

Statements to formalise:
  A1  three_sets_ie (n=3 inclusion-exclusion, subtraction-free, any sets):
      |A u B u C| + |A^B| + |A^C| + |B^C| = |A| + |B| + |C| + |A^B^C|.
  A2  three_gear_assembly: A1 instantiated at mark sets M_q = {k : q|6k-1 or
      q|6k+1} - "assembled = sieve overcount" since overcount := marks - distinct.
  A3  card_marks_eq: |M_q| = |left class| + |right class| (sides disjoint by
      slot cap).
  A4  card_pair_inter_eq: |M_q ^ M_r| = |LL| + |LR| + |RL| + |RR| (4 disjoint
      side classes; each one CRT class - the bridge to floor counts).
  Paper-side (numeric check only): triple intersection = 8 side classes; and
  the fully-assembled closed form: overcount = sum(4-way pairs) - 8-way triple,
  every term equal to its floor-count formula via the class reps.
"""

from itertools import product
from sympy import primerange
from sympy.ntheory.modular import crt

def marks(q, t):
    return {k for k in range(1, t + 1) if (6*k-1) % q == 0 or (6*k+1) % q == 0}

def side_class(spec, t):
    """spec: list of (prime, side) with side 'L' (divides 6k-1) or 'R'."""
    out = set()
    for k in range(1, t + 1):
        if all((6*k-1) % p == 0 if s == 'L' else (6*k+1) % p == 0 for p, s in spec):
            out.add(k)
    return out

def class_count_floor(spec, t):
    """Floor formula: one CRT class mod prod(p); count (t + M - a)//M."""
    mods = [p for p, _ in spec]
    residues = [1 if s == 'L' else p - 1 for p, s in spec]  # 6k = 1 or -1 mod p
    M = 1
    for p in mods:
        M *= p
    c = int(crt(mods, residues)[0])
    a = (pow(6, -1, M) * c) % M
    assert 1 <= a < M
    return (t + M - a) // M

trials = [(5, 7, 11), (7, 11, 13), (5, 11, 13), (5, 7, 13)]
ts = [0, 50, 385, 500, 1001, 2002]
a1 = a2 = a3 = a4 = a5 = 0
for (q, r, s) in trials:
    for t in ts:
        A, B, C = marks(q, t), marks(r, t), marks(s, t)
        # A1/A2 assembly identity
        lhs = len(A | B | C) + len(A & B) + len(A & C) + len(B & C)
        rhs = len(A) + len(B) + len(C) + len(A & B & C)
        assert lhs == rhs, (q, r, s, t)
        a2 += 1
        # A3 per-gear side split + floor counts
        for g in (q, r, s):
            L = side_class([(g, 'L')], t); R = side_class([(g, 'R')], t)
            assert len(marks(g, t)) == len(L) + len(R)
            assert len(L) == class_count_floor([(g, 'L')], t)
            assert len(R) == class_count_floor([(g, 'R')], t)
        a3 += 1
        # A4 pair bridge: 4 disjoint side classes, each = floor count
        for (g1, g2) in [(q, r), (q, s), (r, s)]:
            inter = marks(g1, t) & marks(g2, t)
            parts = [side_class([(g1, s1), (g2, s2)], t)
                     for s1, s2 in product('LR', repeat=2)]
            assert sum(len(p) for p in parts) == len(inter)
            assert set().union(*parts) == inter
            for (s1, s2), p in zip(product('LR', repeat=2), parts):
                assert len(p) == class_count_floor([(g1, s1), (g2, s2)], t)
        a4 += 1
        # paper-side: triple = 8 classes; fully assembled closed-form overcount
        triple = A & B & C
        parts8 = [side_class([(q, s1), (r, s2), (s, s3)], t)
                  for s1, s2, s3 in product('LR', repeat=3)]
        assert sum(len(p) for p in parts8) == len(triple)
        overcount = len(A) + len(B) + len(C) - len(A | B | C)
        pair_sum = sum(class_count_floor([(g1, s1), (g2, s2)], t)
                       for (g1, g2) in [(q, r), (q, s), (r, s)]
                       for s1, s2 in product('LR', repeat=2))
        triple_sum = sum(class_count_floor([(q, s1), (r, s2), (s, s3)], t)
                         for s1, s2, s3 in product('LR', repeat=3))
        assert overcount == pair_sum - triple_sum, (q, r, s, t)
        a5 += 1
print(f"A1/A2 assembly identity: {a2} cases OK")
print(f"A3 per-gear side split + floor counts: {a3} cases OK")
print(f"A4 pair bridge (4 classes, floor counts): {a4} cases OK")
print(f"paper-side: triple = 8 classes; assembled closed-form overcount"
      f" = pairs - triples: {a5} cases OK")
