"""Ideas lane tests.

Test 1 (gear-role composition law): for every column c >= 1,
    G(3c(48c^2-1)) = G(c) u G(2c), disjoint,   where G(n) = {primes p >= 5 : p | 36n^2-1}
because T_3(s)^2 - 1 = (s^2-1)((2s)^2-1)^2 with s = 6c, T_3(s) = 4s^3-3s = 6*3c(48c^2-1).
Checked by full factorisation for c <= C_MAX.

Test 2 (the exact structure finiteness would force, measured): for the machine of q, the
open columns in (0, q'^2/6) must lie in the 2 strike classes of q' if there is no twin in
the window of q'.  Count the classes mod q' they actually occupy.
"""
import sys
from sympy import factorint, primerange, isprime

def G(n):
    return {p for p in factorint(36 * n * n - 1) if p >= 5}

C_MAX = 3000
bad = 0
four = []
for c in range(1, C_MAX + 1):
    N = 3 * c * (48 * c * c - 1)
    g, g1, g2 = G(N), G(c), G(2 * c)
    if g != g1 | g2 or (g1 & g2):
        bad += 1
    if len(g) == 4:
        four.append(c)
print(f"test1: c <= {C_MAX}: violations of G(c*) = G(c) u G(2c) (disjoint): {bad}")
print(f"test1: columns c* with exactly four gears: {len(four)}; first c: {four[:12]}")
twin_c = [c for c in four if isprime(6*c-1) and isprime(6*c+1) and isprime(12*c-1) and isprime(12*c+1)]
print(f"test1: of those, c and 2c both twin centres: {len(twin_c)}; first c: {twin_c[:12]}")

def open_columns(q, X):
    gears = list(primerange(5, q + 1))
    struck = bytearray(X + 1)
    for p in gears:
        u = pow(6, -1, p)
        for r in (u % p, (-u) % p):
            for n in range(r, X + 1, p):
                struck[n] = 1
    return [n for n in range(1, X + 1) if not struck[n]]

for q, qn in ((101, 103), (211, 223), (307, 311)):
    X = (qn * qn - 1) // 6
    op = open_columns(q, X)
    classes = {n % qn for n in op}
    twins = [n for n in op if isprime(6*n-1) and isprime(6*n+1)]
    print(f"test2: machine {q}, columns (0,{X}]: open {len(op)}, twins {len(twins)}, "
          f"classes mod {qn} occupied {len(classes)} of {qn} (finiteness would force <= 2)")
