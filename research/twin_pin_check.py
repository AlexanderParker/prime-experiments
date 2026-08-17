"""Sanity checks for the g=2 pinning theorems before formalising (Harvester r5).

Statements (the g=2 slice of Lateral's split-gap law, research/split_gap_law.py):
  T1  a twin pair (p, p+2), p > 3, has p = 5 mod 6; u = (p+1)/6 gives
      6u-1 = p, 6u+1 = p+2: the pair IS slot u (the pin), p kills left,
      p+2 kills right, and u <= (y+1)/6 for every y >= p.
  T2  the split class is exactly the CRT class of u:
      (p | 6k-1 and p+2 | 6k+1)  <=>  k = u  (mod P), P = p(p+2).
  T3  mirror class at P - u: p | 6(P-u)+1 and p+2 | 6(P-u)-1.
  T4  product slot kp = u(p+1): 6kp - 1 = p(p+2), struck by BOTH gears.
  T5  uniqueness of the own-slot pin: for a prime pair (q, q+g), g > 0, if the
      slot holding q itself (6k-1 = q) is split-killed ((q+g) | 6k+1), then
      g = 2. (Both odd primes force g even; divisibility forces g <= 2.)
"""

from sympy import isprime, primerange

twins = [(p, p + 2) for p in primerange(5, 3000) if isprime(p + 2)]
print(f"{len(twins)} twin pairs with 5 <= p < 3000")

t1 = t2 = t3 = t4 = t5 = 0
for p, p2 in twins:
    P = p * p2
    # T1
    assert p % 6 == 5, (p,)
    u = (p + 1) // 6
    assert 6 * u - 1 == p and 6 * u + 1 == p2
    assert p % p == 0 and (6 * u + 1) % p2 == 0
    t1 += 1
    # T2 - full loop over one period for small p, sampled classes beyond
    if p <= 150:
        for k in range(1, 2 * P + 1):
            lhs = (6 * k - 1) % p == 0 and (6 * k + 1) % p2 == 0
            rhs = k % P == u % P
            assert lhs == rhs, (p, k)
        t2 += 1
    else:
        for t in range(3):
            k = u + t * P
            assert (6 * k - 1) % p == 0 and (6 * k + 1) % p2 == 0
        t2 += 1
    # T3 mirror
    km = P - u
    assert (6 * km + 1) % p == 0 and (6 * km - 1) % p2 == 0
    t3 += 1
    # T4 product slot
    kp = u * (p + 1)
    assert 6 * kp - 1 == P and P % p == 0 and P % p2 == 0
    t4 += 1

# T5 uniqueness over all prime pairs to 400
for q in primerange(5, 400):
    for qp in primerange(q + 1, 404):
        g = qp - q
        # own slot of q exists iff q = 5 mod 6; then k = (q+1)/6
        if q % 6 == 5:
            k = (q + 1) // 6
            assert 6 * k - 1 == q
            if (6 * k + 1) % qp == 0:
                assert g == 2, (q, qp)
                t5 += 1
print(f"T1 pin+location: {t1} pairs OK")
print(f"T2 CRT class iff: {t2} pairs OK (exhaustive to p=150, sampled beyond)")
print(f"T3 mirror class: {t3} pairs OK")
print(f"T4 product slot: {t4} pairs OK")
print(f"T5 own-slot pin implies g=2: all prime pairs q<q'<=400 OK ({t5} pins, all g=2)")
