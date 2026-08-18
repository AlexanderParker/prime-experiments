"""Sanity checks for the CORR triple formalisation (Harvester r8).

Statements to formalise (the >= 3-gear both-sided layer of the master formula):
  C1  general two-sided class: for coprime moduli mL, mR (both coprime to 6,
      both > 1), {k >= 1 : mL | 6k-1 and mR | 6k+1} is exactly one CRT class
      k = a (mod mL*mR), 1 <= a < mL*mR, count (t + M - a) // M.
  C2  triple instantiation: mL = q*r, mR = s (distinct primes >= 5) - the
      first genuinely new CORR case (3 gears, both-sided).
  C3  the signed identity (inclusion-exclusion for one triple, subtraction-free):
      |A or B| + |triple| = |A| + |B| over slots [1,t], where
      A = {q|L and s|R}, B = {r|L and s|R}, triple = {qr|L and s|R} = A and B.
"""

from itertools import combinations
from sympy import primerange
from sympy.ntheory.modular import crt

ps = list(primerange(5, 20))  # 5,7,11,13,17,19
triples = list(combinations(ps, 3))
print(f"{len(triples)} unordered triples from {ps}; testing all ordered (q,r|s) roles")

c1 = c2 = c3 = 0
for q, r, s in triples:
    for (a1, a2, b) in [(q, r, s), (q, s, r), (r, s, q)]:  # which two go left
        mL, mR = a1 * a2, b
        M = mL * mR
        # C1/C2: class rep via CRT: 6k = 1 mod mL, 6k = -1 mod mR
        c = int(crt([mL, mR], [1, mR - 1])[0])
        a = (pow(6, -1, M) * c) % M
        assert 1 <= a < M, (mL, mR, a)
        # membership over one period + a bit
        for k in range(1, M + 50):
            hit = (6 * k - 1) % mL == 0 and (6 * k + 1) % mR == 0
            assert hit == (k % M == a), (mL, mR, k)
        # count formula
        for t in [0, a - 1, a, a + M - 1, a + M, 2 * M + 13]:
            if t < 0:
                continue
            cnt = sum(1 for k in range(1, t + 1) if k % M == a % M)
            assert cnt == (t + M - a) // M, (mL, mR, t)
        c1 += 1
    c2 += 1
    # C3 signed identity, roles: q|L&s|R, r|L&s|R, overlap qr|L&s|R
    for t in [0, 37, q * r * s // 2, q * r * s, 2 * q * r * s + 11]:
        A = [k for k in range(1, t + 1) if (6*k-1) % q == 0 and (6*k+1) % s == 0]
        B = [k for k in range(1, t + 1) if (6*k-1) % r == 0 and (6*k+1) % s == 0]
        T = [k for k in range(1, t + 1) if (6*k-1) % (q*r) == 0 and (6*k+1) % s == 0]
        U = sorted(set(A) | set(B))
        assert set(A) & set(B) == set(T), (q, r, s, t)
        assert len(U) + len(T) == len(A) + len(B), (q, r, s, t)
    c3 += 1

print(f"C1 two-sided class membership + count (all 3 role splits): {c1} cases OK")
print(f"C2 triples covered: {c2} OK")
print(f"C3 signed identity |A or B| + |triple| = |A| + |B| (overlap == triple): {c3} triples OK")
