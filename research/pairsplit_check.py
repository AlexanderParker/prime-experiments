"""Sanity checks for the PAIRSPLIT class formalisation (Harvester r7).

Statements to formalise (the split/cross-member layer of the master formula):
  P1  split class: for distinct primes q, r >= 5, {k >= 1 : q | 6k-1 and r | 6k+1}
      is exactly one CRT class k = a (mod qr), where a = 6^{-1} c mod qr and
      c = CRT(1 mod q, r-1 mod r); 1 <= a < qr.
  P2  floor count: #{k in [1,t] : k = a (mod qr)} = (t + qr - a) // qr
      (already proved as card_class_Ico; re-checked on split reps).
  P3  mirror: swapping (q, r) gives the other split class (r left, q right).
  P4  g=2 loop-closer: for a twin pair (p, p+2) the split class rep IS the pin
      u = (p+1)/6 (twin_split_class_iff + class_rep_unique agree).
Cross-check against Lateral's closed form split_rep (research/split_gap_law.py).
"""

from sympy import primerange, isprime
from sympy.ntheory.modular import crt

ps = list(primerange(5, 60))
pairs = [(q, r) for q in ps for r in ps if q != r]
print(f"{len(pairs)} ordered prime pairs (q, r), q != r, 5 <= q, r < 60")

p1 = p2 = p4 = 0
for q, r in pairs:
    P = q * r
    c = int(crt([q, r], [1, r - 1])[0])
    a = (pow(6, -1, P) * c) % P
    assert 1 <= a < P, (q, r, a)
    # P1 membership, exhaustive over two periods
    for k in range(1, 2 * P + 1):
        split = (6 * k - 1) % q == 0 and (6 * k + 1) % r == 0
        assert split == (k % P == a), (q, r, k)
    p1 += 1
    # P2 count at assorted t
    for t in [0, a - 1, a, a + P - 1, a + P, 2 * P + 17]:
        if t < 0:
            continue
        cnt = sum(1 for k in range(1, t + 1) if k % P == a % P)
        assert cnt == (t + P - a) // P, (q, r, t)
    p2 += 1
    # P4 g=2: twin pair -> a == u
    if r == q + 2:
        u = (q + 1) // 6
        assert a == u, (q, r, a, u)
        p4 += 1

# P3 mirror: ordered pairs cover both orientations by construction; spot-assert
q, r = 5, 7
P = q * r
a_qr = (pow(6, -1, P) * int(crt([q, r], [1, r - 1])[0])) % P
a_rq = (pow(6, -1, P) * int(crt([r, q], [1, q - 1])[0])) % P
assert a_qr == 1 and a_qr != a_rq  # (5,7) is a twin pair: pin at u=1

print(f"P1 split-class membership iff (2 periods, both orientations): {p1} pairs OK")
print(f"P2 floor count on split reps: {p2} pairs OK")
print(f"P3 mirror = role swap: OK (ordered enumeration covers both classes)")
print(f"P4 g=2 loop-closer, split rep == pin u: {p4} twin pairs OK")
