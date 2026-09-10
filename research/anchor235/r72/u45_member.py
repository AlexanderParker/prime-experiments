"""u45_member.py -- membership in D_J(M) by the same covering COUNT the census uses.

ol_pattern.py decides the covering problem by search (or by meet-in-the-middle).  The census
instrument of u45_census.py counts the solutions of the same kind of problem by a dynamic
programme over the gears whose state is the set of closed offsets not yet struck, with equal masks
collapsed.  Counting is not harder than deciding here, and the DP has no search tree to explode,
so it is used as the fast decider for the J-run scans at m37 -- and it returns the exact NUMBER of
columns of the period at which the window occurs, which the search cannot.

realised_count(gaps, gears) = the number of columns x of one period at which the machine's next J
gaps are exactly `gaps`.  Zero iff the window is not realised.
"""


def realised_count(gaps, gears, cap=4_000_000):
    off = [0]
    for g in gaps:
        off.append(off[-1] + int(g))
    S = off[-1]
    openset = set(off)
    closed = [o for o in range(S + 1) if o not in openset]
    full = 0
    for o in closed:
        full |= 1 << o
    dp = {full: 1}
    for g in gears:
        u = pow(6, -1, g)
        t1, t2 = u % g, (-u) % g
        ms = {}
        for r in range(g):
            bad = False
            for o in off:
                q = (r + o) % g
                if q == t1 or q == t2:
                    bad = True
                    break
            if bad:
                continue
            m = 0
            o1 = (t1 - r) % g
            while o1 <= S:
                if o1 in openset:
                    m = -1
                    break
                m |= 1 << o1
                o1 += g
            if m == -1:
                continue
            o2 = (t2 - r) % g
            while o2 <= S:
                m |= 1 << o2
                o2 += g
            ms[m] = ms.get(m, 0) + 1
        if not ms:
            return 0
        nd = {}
        for st, c in dp.items():
            for m, mu in ms.items():
                k = st & ~m
                nd[k] = nd.get(k, 0) + c * mu
        dp = nd
        if len(dp) > cap:
            raise MemoryError(f"membership DP: {len(dp):,} states for span {S}")
    return dp.get(0, 0)


def realised(gaps, gears, cap=4_000_000):
    return realised_count(gaps, gears, cap) > 0
