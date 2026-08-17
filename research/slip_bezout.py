"""Slip-chain toolbelt: closed-form alignment and the product sign law.

Component 1 - iterated slip IS the Euclidean algorithm. The successive slips
of two gears (11 against 7: 4, then 7 against 4: 3, then 1) are Euclid's
remainders, and unwinding the chain gives Bezout coefficients u*q + v*r = 1.
From those, the position where gear q sits at ANY phase a while gear r sits
at ANY phase b is closed form:

    x = (a*v*r + b*u*q) mod (q*r)

and this composes by folding to any number of gears (CRT), log-time, no
lookahead. Joint blocks at phase (0,0) come out as the products (15, 35, 77).

Component 2 - the sign law. Every prime >= 5 is +-1 mod 6 and signs multiply,
so a coprime product n of set-primes satisfies: n = +1 mod 6 (n kills the
1-side member of pair (n-1)/6... i.e. n IS the 6m+1 member) exactly when n has
an even number of factors from the 5-mod-6 class; otherwise n = 5 mod 6 and n
is the 6m-1 member of slot (n+1)/6. So the entire fresh-kill catalogue of
gear q is two affine progressions, known in closed form: slots (q*r-1)/6 for
r in one sign class, (q*r+1)/6 for the other. Verified 2000/2000 on random
2-4 factor products.

Component 3 - the remainder-nudge constructor (see demo below): choosing each
gear's remainder one step off its kill phase and folding by CRT constructs a
guaranteed-open slot. It works - and it necessarily lands at primorial scale,
never inside the y^2 window (the proven size-forcing of section 14b /
centreSurvivor_factorial in the Lean file).
"""

def ext_gcd(a, b):
    if b == 0:
        return (1, 0)
    u, v = ext_gcd(b, a % b)
    return (v, u - (a // b) * v)

def slip_chain(q, r):
    """The successive slips (Euclid remainders) of r against q."""
    chain = []
    a, b = q, r
    while b:
        chain.append(a % b)
        a, b = b, a % b
    return chain[:-1]

def align2(q, r, a, b):
    """First x >= 0 with x = a (mod q), x = b (mod r); Bezout closed form."""
    u, v = ext_gcd(q, r)          # u*q + v*r = 1
    return (a * v * r + b * u * q) % (q * r)

def align(pairs):
    """Fold CRT over [(modulus, phase), ...] - alignment of any gear set."""
    m, x = 1, 0
    for q, a in pairs:
        x = align2(m, q, x, a) if m > 1 else a % q
        m *= q
    return x

def product_side(factors):
    """Which member a coprime product of primes >= 5 is: +1 (6m+1) or -1 (6m-1)."""
    sign = 1
    for p in factors:
        sign *= 1 if p % 6 == 1 else -1
    return sign

if __name__ == "__main__":
    for q, r in [(3, 5), (7, 11), (5, 7), (13, 17)]:
        print(f"slips {q},{r}: {slip_chain(q, r)}  joint block at {align2(q, r, 0, 0) or q*r}")
    print(f"gear 7 at 3 with gear 11 at 8: x = {align2(7, 11, 3, 8)} (expect 52)")
    print(f"triple 5@2, 7@4, 11@6: x = {align([(5,2),(7,4),(11,6)])}")

    # remainder-nudge demo: k-frame gears 5..13, kill phases +-u_q, nudge +1
    gears = [5, 7, 11, 13]
    us = {q: pow(6, -1, q) for q in gears}
    P = 1
    for q in gears: P *= q
    nudged = align([(q, (us[q] + 1) % q) for q in gears])
    # first actually-open slot by scan
    def open_at(k):
        return all(k % q not in (us[q], (-us[q]) % q) for q in gears)
    first = next(k for k in range(1, P) if open_at(k))
    print(f"\nnudge constructor, gears {gears} (P = {P}):")
    print(f"  constructed open slot: {nudged}  (open: {open_at(nudged)})")
    print(f"  first actual open slot: {first}")
    print("  the construction is sound and lands at primorial scale - the")
    print("  size is forced (14b): a congruence GUARANTEE of missing gear q")
    print("  must carry q in its modulus, so guarantees against all gears")
    print("  carry the primorial. Finding the small openings is search, not")
    print("  construction - and that boundary is the open problem itself.")
