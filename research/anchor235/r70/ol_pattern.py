"""ol_pattern.py -- EXACT membership in D_k(M) for a local pattern, with no scan of the period.

The machine {5..y} strikes column k iff k = +-u_g (mod g) with u_g = 6^{-1} mod g.  By CRT the map
x -> (x mod g)_g is a bijection onto the product of the Z_g, so as x runs over one period the
tuple of residues runs over every combination independently.  In the OFFSET coordinate of a
window starting at x, gear g strikes the offsets o with o = (+-u_g - x) mod g, i.e. TWO residue
classes mod g whose difference is d_g = 2u_g mod g, and whose position t_g = (u_g - x) mod g is
free and independent across gears.

So a local pattern -- a set OPEN of offsets required to be open and the complementary set CLOSED
of offsets in [0, S] required to be struck -- is REALISED in the machine iff there is a choice of
t_g in Z_g, one per gear, with

    (i)  no gear strikes any offset of OPEN,   (ii) every offset of CLOSED is struck by some gear.

(i) is independent per gear (it deletes at most 2|OPEN| phases of each), (ii) is a covering
condition.  The solver below is exact: it filters each gear's phases by (i), then decides (ii) by
meet-in-the-middle over a split of the gears into two groups, testing every surviving combination.
No sampling, no relaxation; a NO is a proof that the window is not realised anywhere in the period.

D_3(m37) is exactly what this gives at y = 37, where the period 1.24e12 has never been scanned.
"""
import numpy as np

GEARS_ALL = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]


def u_of(g):
    return pow(6, -1, g)


def d_of(g):
    return (2 * u_of(g)) % g


def _valid_phases(g, opens):
    """t in Z_g such that neither struck class {t, t+d} contains any offset of `opens`."""
    d = d_of(g)
    bad = np.zeros(g, dtype=bool)
    for o in opens:
        bad[o % g] = True
        bad[(o - d) % g] = True
    return np.flatnonzero(~bad)


def _masks(g, phases, S, nw):
    """(len(phases), nw) bitmasks of the offsets in [0, S] struck by gear g at each phase."""
    d = d_of(g)
    r = np.arange(S + 1) % g
    out = np.zeros((phases.size, nw), dtype=np.uint64)
    for i, t in enumerate(phases):
        idx = np.flatnonzero((r == t) | (r == (t + d) % g))
        np.bitwise_or.at(out[i], idx >> 6, np.uint64(1) << (idx & 63).astype(np.uint64))
    return out


def _product(arrs, nw, cap):
    cur = np.zeros((1, nw), dtype=np.uint64)
    for a in arrs:
        if cur.shape[0] * a.shape[0] > cap:
            return None
        cur = (cur[:, None, :] | a[None, :, :]).reshape(-1, nw)
    return cur


def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) &
                                               np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def realised(offsets, gears, bcap=12_000_000, stats=None):
    """Is the window with openings exactly at `offsets` (0 = offsets[0] < ... < offsets[-1] = S,
    every other column of [0, S] struck) realised in the machine with gear set `gears`?"""
    S = int(offsets[-1])
    nw = (S + 1 + 63) // 64
    opens = list(int(o) for o in offsets)
    openset = set(opens)
    closed = [o for o in range(S + 1) if o not in openset]
    cmask = np.zeros(nw, dtype=np.uint64)
    for o in closed:
        cmask[o >> 6] |= np.uint64(1) << np.uint64(o & 63)
    # (i) per-gear phase filter
    ph, mk = {}, {}
    for g in gears:
        p = _valid_phases(g, opens)
        if p.size == 0:
            if stats is not None:
                stats["killed_by_gear"] = g
            return False
        ph[g] = p
        mk[g] = _masks(g, p, S, nw)
    # every closed offset must be coverable by SOME gear at SOME surviving phase
    cover_union = np.zeros(nw, dtype=np.uint64)
    for g in gears:
        cover_union |= np.bitwise_or.reduce(mk[g], axis=0)
    if np.any(cmask & ~cover_union):
        if stats is not None:
            stats["killed_by_coverage"] = True
        return False
    # capacity: the sum over gears of the largest number of closed offsets one phase can cover
    cap_sum = 0
    for g in gears:
        cap_sum += int(_popcount(mk[g] & cmask).sum(axis=1).max())
    if cap_sum < len(closed):
        if stats is not None:
            stats["killed_by_capacity"] = (cap_sum, len(closed))
        return False
    # (ii) meet in the middle: split the gears so that the enumerated side A is as small as
    # possible while the vectorised side B stays inside `bcap` rows
    sizes = {g: ph[g].size for g in gears}
    best = None
    n = len(gears)
    for m in range(1 << n):
        na = nb = 1
        for i, g in enumerate(gears):
            if m >> i & 1:
                na *= sizes[g]
            else:
                nb *= sizes[g]
            if na > 4_000_000 or nb > bcap:
                break
        else:
            if nb <= bcap and (best is None or na < best[0]):
                best = (na, m)
    if best is None:
        raise MemoryError(f"no split inside the budget for span {S}: {sizes}")
    _, m = best
    A = [g for i, g in enumerate(gears) if m >> i & 1]
    B = [g for i, g in enumerate(gears) if not (m >> i & 1)]
    Am = _product([mk[g] for g in A], nw, 4_000_000)
    Bm = _product([mk[g] for g in B], nw, bcap)
    if stats is not None:
        stats["split"] = (A, [int(Am.shape[0])], B, [int(Bm.shape[0])])
    unionB = np.bitwise_or.reduce(Bm, axis=0)
    Bt = np.ascontiguousarray(Bm.T)          # (nw, nB)
    res = cmask[None, :] & ~Am               # residual to be covered by B, per A-row
    res = np.unique(res, axis=0)
    keep = ~np.any(res & ~unionB[None, :], axis=1)
    res = res[keep]
    if res.shape[0] == 0:
        return False
    order = np.argsort(_popcount(res).sum(axis=1))
    acc = np.empty(Bt.shape[1], dtype=np.uint64)
    for i in order:
        R = res[i]
        acc[:] = 0
        for w in range(nw):
            if R[w]:
                acc |= R[w] & ~Bt[w]
        if not acc.all():
            return True
    return False


def realised_word(gaps, gears, **kw):
    """Is (g_1, ..., g_k) a realised window of k consecutive gaps of the machine?"""
    off = np.concatenate([[0], np.cumsum(np.asarray(gaps, dtype=np.int64))])
    return realised(off, gears, **kw)


# ---------------------------------------------------------------- the same decision, by search
#
# Same problem, solved as an exact-cover search instead of by enumeration: pick the closed offset
# with the fewest ways left of being struck, branch over them, recurse.  Identical verdicts (gated
# against `realised` on every triple of m23 and m29); orders of magnitude faster at m37, where the
# enumeration side runs to 10^9 combinations.

def _setup(offsets, gears):
    S = int(offsets[-1])
    opens = [int(o) for o in offsets]
    openset = set(opens)
    closed = [o for o in range(S + 1) if o not in openset]
    cmask = 0
    for o in closed:
        cmask |= 1 << o
    per = {}
    for g in gears:
        d = d_of(g)
        bad = set()
        for o in opens:
            bad.add(o % g)
            bad.add((o - d) % g)
        ms = []
        for t in range(g):
            if t in bad:
                continue
            m = 0
            for o in closed:
                if o % g == t or o % g == (t + d) % g:
                    m |= 1 << o
            ms.append(m)
        if not ms:
            return None, None, None
        per[g] = ms
    return cmask, per, closed


def realised_search(offsets, gears, budget=4_000_000, counter=None):
    """Exact: is the window with openings exactly at `offsets` realised in the machine?"""
    cmask, per, closed = _setup(offsets, gears)
    if cmask is None:
        return False
    cov = {}          # gear -> {position: [masks striking it]}
    cap = {}          # gear -> the most closed offsets one surviving phase can strike
    for g in gears:
        cg = {}
        for m in per[g]:
            mm = m
            while mm:
                b = mm & -mm
                cg.setdefault(b.bit_length() - 1, []).append(m)
                mm ^= b
        cov[g] = cg
        cap[g] = max(bin(m).count("1") for m in per[g])
    nodes = [0]

    def rec(unc, rem):
        if unc == 0:
            return True
        nodes[0] += 1
        if nodes[0] > budget:
            raise RuntimeError("search budget exceeded")
        # capacity: no assignment of the gears still free can strike more offsets of `unc` than
        # the sum over them of the best single phase, so if that falls short, stop here
        need = bin(unc).count("1")
        tot = 0
        for g in rem:
            b = 0
            for m in per[g]:
                v = bin(m & unc).count("1")
                if v > b:
                    b = v
            tot += b
            if tot >= need:
                break
        if tot < need:
            return False
        best, bestopts = None, None
        u = unc
        while u:
            b = u & -u
            p = b.bit_length() - 1
            u ^= b
            opts = [(g, m) for g in rem for m in cov[g].get(p, ())]
            if not opts:
                return False
            if bestopts is None or len(opts) < len(bestopts):
                best, bestopts = p, opts
                if len(opts) <= 1:
                    break
        for g, m in bestopts:
            if rec(unc & ~m, tuple(x for x in rem if x != g)):
                return True
        return False

    try:
        out = rec(cmask, tuple(gears))
    finally:
        if counter is not None:
            counter["nodes"] = counter.get("nodes", 0) + nodes[0]
    return out


def realised_word_search(gaps, gears, **kw):
    off = np.concatenate([[0], np.cumsum(np.asarray(gaps, dtype=np.int64))])
    return realised_search(off, gears, **kw)
