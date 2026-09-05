"""Branch 5d.ii.i / thin place 1 (prover, round 48).  Core tool: exact maximum
blocked span of an arbitrary SET of gears, by phase search, no period scan.

Column frame.  Gear g strikes column k iff k = +-u_g (mod g), u_g = 6^{-1} mod g.
The two teeth are separated by d_g = 2 u_g = 3^{-1} (mod g); the short arc is
a_g = min(d_g, g - d_g) and the long arc is g - a_g.

Over the full period of a gear set every combination of phases occurs exactly once
(CRT), so a run of L consecutive struck columns exists somewhere in the period iff
the L positions 0..L-1 can be covered by choosing, for each gear, one translate of
its tooth pair.  Hence

    F(A) = the smallest L that cannot be covered by the gears of A

(a maximal gap of span S is a blocked run of S - 1 columns whose two endpoints are
open, and the endpoints of a maximal blocked run are open).

The search assigns a gear to the LEFTMOST uncovered position (two ways: that
position carries one tooth or the other), prunes on capacity, and memoises failed
states.  Exhaustive, so "not coverable" is a proof.
"""


def arcs(g):
    """(separation d_g, short arc a_g, long arc g - a_g)."""
    d = pow(3, -1, g)
    a = min(d, g - d)
    return d, a, g - a


def coverable(L, gears, memo_cap=4_000_000):
    """Can columns 0..L-1 all be struck, one phase per gear (each gear used once)?"""
    if L <= 0:
        return True
    gears = sorted(gears)
    n = len(gears)
    full = (1 << L) - 1
    masks = []
    cap = []
    dsep = []
    for g in gears:
        d = pow(3, -1, g)
        dsep.append(d)
        ms = []
        for o in range(g):
            m = 0
            for i in range(o, L, g):
                m |= 1 << i
            for i in range((o + d) % g, L, g):
                m |= 1 << i
            ms.append(m)
        masks.append(ms)
        cap.append(max(bin(m).count("1") for m in ms))

    fail = set()

    def search(covered, avail):
        if covered == full:
            return True
        key = (covered, avail)
        if key in fail:
            return False
        u = ~covered & full
        todo = bin(u).count("1")
        tot = 0
        a = avail
        while a:
            b = a & -a
            tot += cap[b.bit_length() - 1]
            a ^= b
        if tot < todo:
            fail.add(key)
            return False
        pos = (u & -u).bit_length() - 1
        a = avail
        while a:
            b = a & -a
            i = b.bit_length() - 1
            a ^= b
            g, d = gears[i], dsep[i]
            o1 = pos % g
            o2 = (pos - d) % g
            for o in ((o1,) if o1 == o2 else (o1, o2)):
                if search(covered | masks[i][o], avail ^ b):
                    return True
        if len(fail) < memo_cap:
            fail.add(key)
        return False

    return search(0, (1 << n) - 1)


def F_of(gears, lo=1):
    """Smallest L not coverable = the record span F of this gear set."""
    L = max(1, lo)
    while coverable(L, gears):
        L += 1
    return L


if __name__ == "__main__":
    import time
    LADDER = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prev = 1
    for i in range(2, len(primes) + 1):
        gs = primes[:i]
        q = gs[-1]
        t = time.time()
        f = F_of(gs, lo=prev)
        prev = f
        ok = "OK" if LADDER.get(q) == f else f"MISMATCH (record {LADDER.get(q)})"
        print(f"F({{5..{q}}}) = {f:4d}  {ok}   {time.time()-t:.1f}s", flush=True)
