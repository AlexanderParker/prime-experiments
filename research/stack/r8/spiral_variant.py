"""The owner's spiral variant (tree node R5.f.xxxv): start at the open origin column 0 (pair -1, 1),
add a larger gear G to the machine q, mirror across a multiple of G's product (all gears 5..G),
then mirror back across multiples of smaller products that still contain every gear of q, and see
whether the walk can land inside the window of q, (q, q^2], carrying q's full set.

A mirror about axis a maps column n -> 2a - n (members 6n+-1 -> the negated pair); for every gear h
dividing a, the landing is open for h iff n was (h's two classes are symmetric under n -> -n).
So a walk whose axes are all multiples of P = product(5..q) keeps n mod P fixed up to sign: the
landing is congruent to +-0 mod P, i.e. it is a multiple of P, and the window (q, q^2] holds a
positive multiple of P only when P < q^2/6.

Check: for q = 7..23, all walks of up to 4 mirrors with axes chosen among multiples k*Pi of the
products Pi of the gear sets {5..G} (G = next prime), {5..q}, and the intermediate sets, with
1 <= k <= 6: record every landing's residue mod P and whether any landing lies in the window.
Then the tweak: axes that drop some gear of q (product of a subset) - report the least product of a
carried set that fits in the window, i.e. the most gears a landing can be certified for.
"""
from math import prod
from itertools import product as iprod

def primes_upto(n):
    s = [True] * (n + 1); s[:2] = [False, False]
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, n + 1, i): s[j] = False
    return [p for p in range(n + 1) if s[p]]

primes = primes_upto(200)
for q in [7, 11, 13, 17, 19, 23]:
    gears = [g for g in primes if 5 <= g <= q]
    P = prod(gears); G = next(p for p in primes if p > q)
    PG = P * G
    win_lo = (q + 7) // 6; win_hi = (q * q - 1) // 6
    axes = [k * PG for k in range(1, 7)] + [k * P for k in range(1, 7)]
    landings = set(); in_window = []
    for depth in range(1, 5):
        for combo in iprod(axes, repeat=depth):
            n = 0
            for a in combo: n = 2 * a - n
            landings.add(n % P)
            if win_lo <= n <= win_hi: in_window.append((combo, n))
    # the tweak: carried subsets whose product fits the window
    fits = [(len(S), S) for r in range(1, len(gears) + 1) for S in __import__('itertools').combinations(gears, r) if prod(S) <= win_hi]
    best = max(fits, default=(0, ()))
    print(f"q = {q}: P = {P}, window columns [{win_lo}, {win_hi}]; landings mod P = {sorted(landings)}; "
          f"landings inside the window: {len(in_window)}; most gears a certified landing can carry into the window: "
          f"{best[0]} of {len(gears)} (e.g. {best[1]}, product {prod(best[1]) if best[1] else 1})")
