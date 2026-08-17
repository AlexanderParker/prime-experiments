"""Sanity checks for the SAME-side census formalisation (Harvester r6).

Statements to formalise (first layer of Lateral's master supply formula):
  S1  slot-map inversion: for m coprime to 6, {k >= 1 : m | 6k-1} is exactly one
      CRT class k = a (mod m), 6a = 1 (mod m); right member analogously with
      6a' = -1 (mod m).
  S2  floor count: #{k in [1,t] : k = a (mod m)} = (t + m - a) // m  (1 <= a <= m).
  S3  window corollary ("exactly once if it fits"): a <= t < a + m  =>  count = 1.
  S4  own value: if q*r = 5 (mod 6), the left class rep is (q*r+1)/6 - the slot
      holding q*r itself; if q*r = 1 (mod 6), the right rep is (q*r-1)/6.
  S5  self-block: pin slot u of twin (p,p+2) has both members prime yet p | its
      own left member - never a survivor of a machine with p <= y. (Trivial.)
"""

from sympy import primerange

ps = list(primerange(5, 60))
pairs = [(q, r) for i, q in enumerate(ps) for r in ps[i + 1:]]
print(f"{len(pairs)} prime pairs (q, r), 5 <= q < r < 60")

s1 = s2 = s3 = s4 = 0
for q, r in pairs:
    P = q * r
    a = pow(6, -1, P)            # left rep: 6a = 1 (mod P)
    ar = (P - a) % P             # right rep: 6ar = -1 (mod P)
    assert 1 <= a < P and 1 <= ar < P
    # S1 membership, exhaustive over two periods
    for k in range(1, 2 * P + 1):
        left = (6 * k - 1) % q == 0 and (6 * k - 1) % r == 0
        assert left == (k % P == a), (q, r, k)
        right = (6 * k + 1) % q == 0 and (6 * k + 1) % r == 0
        assert right == (k % P == ar), (q, r, k)
    s1 += 1
    # S2 count formula at assorted t
    for t in [0, 1, a - 1, a, a + 1, P - 1, P, P + 1, 2 * P, 2 * P + 37, 3 * P + 11]:
        if t < 0:
            continue
        cnt = sum(1 for k in range(1, t + 1) if k % P == a % P)
        assert cnt == (t + P - a) // P, (q, r, t)
        cntr = sum(1 for k in range(1, t + 1) if k % P == ar % P)
        assert cntr == (t + P - ar) // P, (q, r, t)
    s2 += 1
    # S3 exactly once in fitting windows
    for t in [a, a + 1, a + P - 1]:
        cnt = sum(1 for k in range(1, t + 1) if k % P == a % P)
        assert cnt == 1, (q, r, t)
    s3 += 1
    # S4 own value
    if P % 6 == 5:
        assert a == (P + 1) // 6 and 6 * a - 1 == P, (q, r)
        s4 += 1
    if P % 6 == 1:
        assert ar == (P - 1) // 6 and 6 * ar + 1 == P, (q, r)
        s4 += 1

print(f"S1 class membership iff (left+right, 2 periods): {s1} pairs OK")
print(f"S2 floor count formula: {s2} pairs OK")
print(f"S3 window 'exactly once': {s3} pairs OK")
print(f"S4 own-value reps: {s4} instances OK (every pair is 1 or 5 mod 6)")
