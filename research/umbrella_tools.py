"""Closed-form umbrella tools, built one gear first, then any number.

One gear q, slot k: centre m = k mod q into (-q/2, q/2]. With u' = min(6^{-1}
mod q, q - 6^{-1} mod q), the teeth sit at +-u', so

    |m| <  u' : SHORT umbrella, shield at distance |m|, edges k-(u'-1-  ...)
    |m| == u' : TOOTH
    |m| >  u' : LONG umbrella

and the current umbrella's room on each side is closed form:
    left room  = distance to previous tooth - 1
    right room = distance to next tooth - 1

Any gear set: the joint umbrella containing k is
    [k - min(left rooms), k + min(right rooms)]
one modular op per gear - valid because open runs intersect as intervals.
Verified against brute force for singles, pairs, triples and 8-gear sets.
The alignment law (max joint run = smallest gear's long umbrella) is visible
as the bound min over gears of that gear's current room span.
"""

def uprime(q):
    u = pow(6, -1, q)
    return min(u, q - u)

def gear_umbrella(q, k):
    """(kind, left_room, right_room) for gear q at slot k; kind in
    'short'/'long'/'tooth'. Rooms = open slots available either side within
    this gear's current umbrella."""
    up = uprime(q)
    m = k % q
    mc = m if m <= q // 2 else m - q          # centred representative
    if abs(mc) == up:
        return ('tooth', -1, -1)
    if abs(mc) < up:                          # short umbrella around shield 0
        return ('short', mc + up - 1, up - 1 - mc)
    # long umbrella between +u' and q-u'
    pos = m - up if m > up and m < q - up else (m if m > q//2 else m + q) - up
    # simpler: distance from lower tooth u' (walking upward inside [u'+1, q-u'-1])
    d_lower = (m - up) % q
    d_upper = (q - up - m) % q
    return ('long', d_lower - 1, d_upper - 1)

def joint_umbrella(gears, k):
    """Maximal joint open interval containing k, or None if k is blocked.
    Closed form: min room per side over the gears."""
    lo = hi = None
    for q in gears:
        kind, lr, rr = gear_umbrella(q, k)
        if kind == 'tooth':
            return None
        lo = lr if lo is None else min(lo, lr)
        hi = rr if hi is None else min(hi, rr)
    return (k - lo, k + hi)

if __name__ == "__main__":
    import random
    from math import prod
    random.seed(5)
    ps = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def brute(gears, k):
        us = {q: pow(6, -1, q) for q in gears}
        def open_(x):
            return all(x % q not in (us[q], (-us[q]) % q) for q in gears)
        if not open_(k): return None
        a = k
        while open_(a - 1): a -= 1
        b = k
        while open_(b + 1): b += 1
        return (a, b)

    fails = 0
    for trial in range(4000):
        n = random.choice([1, 2, 3, 5, 8])
        gears = random.sample(ps, n)
        k = random.randrange(2, 10**6)
        if joint_umbrella(gears, k) != brute(gears, k):
            fails += 1
            if fails < 4:
                print('MISMATCH', gears, k, joint_umbrella(gears, k), brute(gears, k))
    print(f"4000 random trials (1,2,3,5,8-gear sets): mismatches {fails}")

    # alignment-law sighting: max joint run of {5,7,11,13} = long umbrella of 5 = 2
    gears = [5, 7, 11, 13]
    best = 0
    k = 1
    while k < prod(gears):
        j = joint_umbrella(gears, k)
        if j:
            best = max(best, j[1] - j[0] + 1)
            k = j[1] + 1
        else:
            k += 1
    print(f"max joint umbrella of {gears}: {best} slots (alignment law: long umbrella of 5 = 2)")
