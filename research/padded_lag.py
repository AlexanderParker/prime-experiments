"""Round 20 lateral, second named construct: THE AUTOCORRELATION AT THE PADDED
LAG q' - and an elementary divisibility law behind it.

From the round-18 closed form, gear q sees lag g as "opposite teeth"
(c_q = q-3 rather than q-4) iff g = +-2u_q mod q. Since 2u_q = 2*6^{-1} =
3^{-1} (mod q), that condition is 3g = +-1 (mod q), i.e.

    gear q is ENHANCED at lag g   <=>   q | 3g - 1  or  q | 3g + 1.

At the padded lag g = q' this says: the gears that see a padded link as a
literal-link lag are exactly the prime divisors of 3q'-1 and 3q'+1. So the
arithmetic of padding is governed by the FACTORISATION OF 3q' +- 1 - an
elementary, explicit, finite set.

The selection factor of the padded lag is sigma(q') = prod_q c_q(q')/(q-2),
tested here against the measured padding supply (gaps of size exactly q').
"""
from math import prod
from split_gap_law import primes

def isprime(n):
    if n < 2: return False
    d = 2
    while d * d <= n:
        if n % d == 0: return False
        d += 1
    return True

def factors(n):
    out, d = [], 2
    while d * d <= n:
        while n % d == 0:
            if d not in out: out.append(d)
            n //= d
        d += 1
    if n > 1 and n not in out: out.append(n)
    return out

def c_q(q, g):
    u = pow(6, -1, q)
    if g % q == 0: return q - 2
    if g % q in ((2 * u) % q, (-2 * u) % q): return q - 3
    return q - 4

# measured padding supply (gaps of size exactly q') and periods, rounds 14/15
STEPS = [(13, 17, 0), (17, 19, 0), (19, 23, 86), (23, 29, 6),
         (29, 31, 2090), (31, 37, 26367), (37, 41, None), (41, 43, None)]

print("=" * 78)
print("PART C: the divisibility law for the padded lag")
bad = 0
for _, qp, _ in STEPS:
    for q in primes(5, 60):
        law = (3 * qp - 1) % q == 0 or (3 * qp + 1) % q == 0
        obs = c_q(q, qp) == q - 3
        if law != obs: bad += 1
print(f"  verification over all (gear, q') pairs above: {bad} mismatches")
print(f"  {'q':>4} {'3q-1':>6} {'3q+1':>6} {'enhanced gears (divisors)':>34}")
for _, qp, _ in STEPS:
    f1, f2 = factors(3 * qp - 1), factors(3 * qp + 1)
    en = sorted({p for p in f1 + f2 if p >= 5})
    print(f"  {qp:>4} {3*qp-1:>6} {3*qp+1:>6} {str(en):>34}")

print("=" * 78)
print("PART D: does sigma(q') track the erratic padding supply?")
print(f"  {'step':>9} {'supply':>8} {'period':>12} {'share':>10} "
      f"{'c_5':>4} {'c_7':>4} {'sigma':>8} {'share/sigma':>12}")
for y, qp, sup in STEPS:
    gears = primes(5, y)
    P = prod(gears)
    s = 1.0
    for q in gears:
        s *= c_q(q, qp) / (q - 2)
    if sup is None:
        print(f"  {y:>4}->{qp:<3} {'?':>8} {P:>12} {'-':>10} "
              f"{c_q(5,qp):>4} {c_q(7,qp):>4} {s:>8.4f} {'-':>12}")
    else:
        sh = sup / P
        print(f"  {y:>4}->{qp:<3} {sup:>8} {P:>12} {sh:>10.2e} "
              f"{c_q(5,qp):>4} {c_q(7,qp):>4} {s:>8.4f} "
              f"{(sh/s if s else 0):>12.2e}")
print("  (share = supply/period; share/sigma is the residual after removing")
print("   the endpoint arithmetic. Steps with c_5 = 2 are the enhanced ones.)")
