"""Coprime-killer census: the second category of killers, after squares.

For a window (members < y^2), each gear q's coprime kills are q*r for primes
r in (q, y^2/q). Census laws (verified in the 23-window, table in
docs/class-tree.md):

1. Coprimes first appear exactly ONE step below the top gear - the top gear
   has none (q*next > y^2, the horizon theorem in coprime language), and the
   second gear's single coprime is the top-pair product (19*23 = 437 in the
   23-set).
2. The count fans out downward like pi(y^2/q) - pi(q): 0, 1, 4, 6, 10, 17, 24
   for gears 23 down to 5.
3. In-set coprimes (both factors working gears) are exactly the crossed teeth
   of the pair machines (143 = 11x13, 221 = 13x17, 437 = 19x23); out-set ones
   are fresh semiprime re-entry of primes beyond the set.
4. The pseudo-twin fraction (partner prime - the deciding, fragile slots) is
   high and rises as q falls: gear 5 has 18 of 24, and its first six coprimes
   35, 55, 65, 85, 95, 115 ALL sit beside primes.
"""
def isprime(n):
    if n < 2: return False
    d = 2
    while d*d <= n:
        if n % d == 0: return False
        d += 1
    return True

def primes(a, b):
    return [p for p in range(a, b+1) if isprime(p)]

def coprime_census(y):
    W = y*y
    out = {}
    for q in primes(5, y):
        rows = []
        for r in primes(q+1, W//q):
            n = q*r
            if n >= W: continue
            partner = n-2 if n % 6 == 1 else n+2
            rows.append((n, r, partner, isprime(partner), r <= y))
        out[q] = rows
    return out

if __name__ == "__main__":
    for q, rows in sorted(coprime_census(23).items(), reverse=True):
        pt = sum(1 for r in rows if r[3])
        print(f"gear {q}: {len(rows)} coprimes, {pt} pseudo-twins")
        for n, r, partner, pp, inset in rows:
            print(f"   {q}*{r} = {n}  partner {partner} {'PRIME' if pp else 'comp'}  {'in-set' if inset else 'out-set'}")
