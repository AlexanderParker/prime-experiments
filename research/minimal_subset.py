"""Minimal gear subset whose openings equal the window's twins, exactly.

Law: gear q is NECESSARY iff one of its root kills pairs with a PRIME partner
inside the window (a pseudo-twin like (209,211) = (11*19, prime) that only q
can unmask). Two members of a pair cannot share the factor q, so the test is
exact. If all of q's root-kill partners are composite, other gears cover its
work and q is droppable.

Findings (verified exact against trial division for y = 13..59):
  - the minimal set is all gears minus the top one or two, and droppability is
    transient: 11 is droppable at y=13, re-recruited from y=17 on when the
    window reaches (209,211); 41 stays droppable through y=43,47.
  - unification: "q necessary" = "q owns a lone-killer (fragile) slot in the
    window" - the one-away census of docs/band-attribution.md and the minimal
    subset question are the same object.
"""
import math

def isprime(n):
    if n < 2: return False
    d = 2
    while d*d <= n:
        if n % d == 0: return False
        d += 1
    return True

def primes(a, b):
    return [p for p in range(a, b+1) if isprime(p)]

def minimal_subset(y):
    gears = primes(5, y)
    needed = []
    for q in gears:
        m = q
        nec = False
        while q*m <= y*y:
            n = q*m
            if n % 6 in (1, 5):
                partner = n - 2 if n % 6 == 1 else n + 2
                rough = all(m % p for p in primes(2, q-1))
                if rough and isprime(partner) and partner > y and n > y:
                    nec = True
                    break
            m += 1
        if nec:
            needed.append(q)
    return needed

def verify(y, sub):
    K = (y*y - 1) // 6
    us = {q: pow(6, -1, q) for q in sub}
    got = set(k for k in range(1, K+1)
              if 6*k - 1 > y and all(k % q not in (us[q], (-us[q]) % q) for q in sub))
    truth = set(k for k in range(1, K+1)
                if 6*k - 1 > y and isprime(6*k - 1) and isprime(6*k + 1))
    return got == truth

if __name__ == "__main__":
    for y in primes(13, 60):
        sub = minimal_subset(y)
        dropped = [q for q in primes(5, y) if q not in sub]
        print(f"y={y:3d}: minimal {sub}  dropped {dropped}  exact: {verify(y, sub)}")
