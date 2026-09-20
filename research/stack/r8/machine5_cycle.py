"""Range check with machine 5 as the cycle (owner, 2026-09-21). Machine 5 = base 2, 3 and gear 5,
period 30. Its known opening (-1, 1) recurs at every multiple of 30: the copies are the pairs
(30k-1, 30k+1), k = 1, 2, 3, ... The overlay is every gear g >= 7. Gear g strikes copy k iff
g | 30k -+ 1, i.e. k = +-30^{-1} mod g: two classes of k per gear, at distance 15^{-1} mod g.
So the overlay acting on the copies is a machine of the same shape as the 6n+-1 machine, with 5
folded into the base. Copy k is a twin iff no gear g <= sqrt(30k+1) strikes k.

Range check for machine q: is there a twin copy with q < 30k-1 and 30k+1 <= q#? Report the first
twin copy above q for q up to 79, and the overlay classes for the first gears.
"""
from sympy import isprime, primerange

K = 200
tw = [k for k in range(1, K + 1) if isprime(30 * k - 1) and isprime(30 * k + 1)]
print("twin copies k <= 200:", tw)
print("gaps between twin copies:", [b - a for a, b in zip(tw, tw[1:])])
for g in [7, 11, 13, 17, 19, 23]:
    inv = pow(30, -1, g)
    print(f"gear {g}: strikes copies k = {min(inv, g - inv)} or {max(inv, g - inv)} mod {g}; "
          f"class distance {pow(15, -1, g)} = 15^-1 mod {g}")
# first failures and their striking gears
for k in range(1, 13):
    if k not in tw:
        m = [(x, [p for p in primerange(7, 200) if x % p == 0]) for x in (30 * k - 1, 30 * k + 1)]
        print(f"copy k={k} struck: {m}")
P = 1
for q in primerange(2, 80):
    P *= q
    k = next((k for k in tw if 30 * k - 1 > q), None)
    assert k is not None and 30 * k + 1 <= P or q < 7
    print(f"q={q:2d}: first twin copy above q is k={k} (pair {30*k-1}, {30*k+1}); q# = {P}")
