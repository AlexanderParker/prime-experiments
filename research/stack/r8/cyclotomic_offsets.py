"""The cyclotomic offsets j = +-c (tree node R5.f.xiv, random lane round 2 (c)).

For a twin centre s = 6c the offsets j = +c, -c give s' = s(s+1), s(s-1) with members
  j = +c: s^2 + s - 1 (lower), Phi_3(s) = s^2 + s + 1 (upper);
  j = -c: s^2 - s - 1 (lower), Phi_6(s) = s^2 - s + 1 (upper).
Identities: every prime factor of Phi_3(s) is 1 mod 3; of Phi_6(s) is 1 mod 6; of s^2 +- s - 1 is
+-1 mod 5 (4(s^2 +- s - 1) = (2s +- 1)^2 - 5).  So a quarter of all gears are inert at these offsets.
Free certificates: N = Phi_3(s) has N - 1 = s(s+1) with s + 1 prime; M = s^2 + s - 1 has M + 1 = s(s+1).

Tests: hit rates H+, H- (both members prime) over the twin centres to PMAX against the generic
per-offset rung rate; the identities on the composites (s <= PID); the predicted singular-series
enhancement K = prod_g (1 - nu(g)/(g-2)) / (1 - 2/g), nu(g) = #{x != +-1 : g | (x^2+x-1)(x^2+x+1)};
the ladder depth by s -> s(s +- 1) greedily from every twin centre <= PLAD.

usage: uv run python cyclotomic_offsets.py [PMAX] [PID] [PLAD]
"""
import sys, math
import numpy as np
from sympy import isprime, factorint

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6
PID = int(sys.argv[2]) if len(sys.argv) > 2 else 10**5
PLAD = int(sys.argv[3]) if len(sys.argv) > 3 else 10**4

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 2)
ps = set(int(x) for x in P)
centres = [int(x) + 1 for x in P if x >= 5 and (int(x) + 2) in ps]

# hits and identities
Hp = Hm = 0; viol = 0; checked = 0; expect = 0.0
for s in centres:
    lo_p, up_p = s * s + s - 1, s * s + s + 1
    lo_m, up_m = s * s - s - 1, s * s - s + 1
    a = isprime(lo_p); b = isprime(up_p); c_ = isprime(lo_m); d = isprime(up_m)
    Hp += a and b; Hm += c_ and d
    expect += 2 * 12 * 0.6601618 / (math.log(s * s) ** 2)   # two offsets, twin columns per column near s^2
    if s <= PID:
        for m, cls in ((up_p, 3), (up_m, 6), (lo_p, 5), (lo_m, 5)):
            if m > 1:
                for g in factorint(m):
                    checked += 1
                    if cls == 3 and g % 3 != 1: viol += 1
                    if cls == 6 and g % 6 != 1: viol += 1
                    if cls == 5 and g % 5 not in (1, 4): viol += 1
print(f"twin centres {len(centres)} (to {PMAX}): hits at +c (both prime) {Hp}, at -c {Hm}; total {Hp+Hm} against a base-rate expectation {expect:.1f} -> measured enhancement {(Hp+Hm)/expect:.2f}")
print(f"identity checks on prime factors (s <= {PID}): {checked} factors, violations {viol}")

# predicted K from the singular series over g <= 10^6
K = 1.0
for g in P:
    g = int(g)
    if g < 5: continue
    nu = sum(1 for x in range(g) if x % g not in (1, g - 1) and ((x * x + x - 1) * (x * x + x + 1)) % g == 0) if g <= 2000 else None
    if nu is None:
        # for large g use the character formula: nu = 2 + (5|g) + 2*[g = 1 mod 3] approximately; skip beyond 2000 (converges)
        break
    K *= (1 - nu / (g - 2)) / (1 - 2 / g)
print(f"predicted singular-series enhancement K (product to g <= 2000) = {K:.3f}")

# ladder depth by s -> s(s +- 1)
depths = []
for s in centres:
    if s > PLAD: break
    cur = s; depth = 0
    while True:
        nxt = None
        for sp in (cur * (cur + 1), cur * (cur - 1)):
            if isprime(sp - 1) and isprime(sp + 1): nxt = sp; break
        if nxt is None or len(str(nxt)) > 400: break
        cur = nxt; depth += 1
    depths.append(depth)
from collections import Counter
print(f"cyclotomic ladder depth from every twin centre <= {PLAD}: " + ", ".join(f"depth {k}: {v}" for k, v in sorted(Counter(depths).items())) + f"; longest {max(depths)}")
