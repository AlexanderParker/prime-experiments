"""Branch 2g.i.a: the glue as a covering statement.  Prover, 2026-09-05.

The glue restated with no sieve in it.  For a 3-run x0 < x1 < x2 < x3 of M with
L = x1-x0, v = x2-x1, R = x3-x2, put T = L+R-1 offsets and the hole h = L-1.  A colouring
sigma : G -> {left, right} gives gear g the base b_g = x0+1 (left) or x0+1+v (right); offset
j is COVERED iff (b_g + j) mod g in {u_g, -u_g} for some g.  The colouring GLUES iff every
j != h is covered.  By CRT there is z with z = b_g (mod g) for all g, and then column z+j is
blocked in M exactly when j is covered; z+h is an opening (it is x1 mod every left gear and
x2 mod every right gear).  So a glueing colouring certifies F_2(M) >= L+R.

Generalised: the second opening need not be x2.  With y any opening of M and R' its right gap,
take b_left = x0+1 and b_right = y-L+1; the hole is x1 on the left and y on the right, and a
glueing colouring certifies F_2(M) >= L + R'.  y = x2 is the standard glue.  A shift that lands
y on a BLOCKED column removes the hole and certifies F(M) >= (blocked run length) + 1.

Certificate classes: C2 (two colours), C2+f (f gears given a free residue), Cs (shifted /
cross glue).  Everything here needs only x0 mod g -- no period is held in memory.
"""
import numpy as np, sys, random
from math import prod

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def gears_of(top):
    return [p for p in PRIMES if p <= top]


def us_of(gears):
    return [pow(6, -1, g) for g in gears]


def sieve(gears, us):
    P = prod(gears)
    blocked = np.zeros(P, dtype=bool)
    for g, u in zip(gears, us):
        blocked[u % g::g] = True
        blocked[(-u) % g::g] = True
    return P, blocked


def gap_stats(P, blocked):
    opens = np.flatnonzero(~blocked)
    gaps = np.diff(np.concatenate([opens, [opens[0] + P]]))
    F = int(gaps.max())
    F2 = int((gaps + np.roll(gaps, -1)).max())
    left, right = np.roll(gaps, 1), np.roll(gaps, -1)
    ns = left + right
    N = {}
    for v, s in zip(gaps.tolist(), ns.tolist()):
        if s > N.get(v, -1):
            N[v] = s
    return opens, gaps, F, F2, N


# --------------------------------------------------------------- covering core
def covbits(g, u, base, T):
    """bit j set iff (base + j) mod g in {u, -u} mod g, for 0 <= j < T."""
    m = 0
    for r in {(u - base) % g, ((-u) - base) % g}:
        j = r
        while j < T:
            m |= 1 << j
            j += g
    return m


def cov_pair(gears, us, x0, L, RR, y):
    """coverage masks (left, right) for the generalised glue with second opening y."""
    T = L + RR - 1
    bl = x0 + 1
    br = y - L + 1
    covL = [covbits(g, u, bl, T) for g, u in zip(gears, us)]
    covR = [covbits(g, u, br, T) for g, u in zip(gears, us)]
    return T, L - 1, covL, covR


def solve_cover(covL, covR, T, h):
    """is there sigma with union of chosen masks = all bits except h?  returns mask or None."""
    k = len(covL)
    target = ((1 << T) - 1) ^ (1 << h)
    suf = [0] * (k + 1)
    for i in range(k - 1, -1, -1):
        suf[i] = suf[i + 1] | covL[i] | covR[i]
    order = sorted(range(k), key=lambda i: -bin(covL[i] | covR[i]).count('1'))
    covL = [covL[i] for i in order]
    covR = [covR[i] for i in order]
    suf = [0] * (k + 1)
    for i in range(k - 1, -1, -1):
        suf[i] = suf[i + 1] | covL[i] | covR[i]

    sol = [None]

    def rec(i, acc, mask):
        need = target & ~acc
        if not need:
            sol[0] = mask
            return True
        if i == k or (need & ~suf[i]):
            return False
        for bit, c in ((0, covL[i]), (1, covR[i])):
            if rec(i + 1, acc | c, mask | (bit << i)):
                return True
        return False

    if rec(0, 0, 0):
        out = [0] * k
        for pos, i in enumerate(order):
            out[i] = (sol[0] >> pos) & 1
        return out
    return None


def glue(gears, us, x0, L, v, R):
    """standard glue: y = x2 = x0+L+v.  Returns sigma list (0=left,1=right) or None."""
    T, h, covL, covR = cov_pair(gears, us, x0, L, R, x0 + L + v)
    return solve_cover(covL, covR, T, h)


def propagate(covL, covR, T, h):
    """unit propagation on the covering instance.  Returns (status, detail).

    status 'ok' (propagation completed with no contradiction, assignment may be partial),
    'conflict' (a gear forced to both sides), 'empty' (a column with no candidate)."""
    k = len(covL)
    assign = [None] * k
    forcing = {}
    while True:
        acc = 0
        for i in range(k):
            if assign[i] is not None:
                acc |= (covR[i] if assign[i] else covL[i])
        need = (((1 << T) - 1) ^ (1 << h)) & ~acc
        if not need:
            return 'ok', (assign, forcing)
        changed = False
        j = 0
        nb = need
        while nb:
            if nb & 1:
                cands = []
                for i in range(k):
                    if assign[i] is not None:
                        continue
                    if (covL[i] >> j) & 1:
                        cands.append((i, 0))
                    if (covR[i] >> j) & 1:
                        cands.append((i, 1))
                if not cands:
                    return 'empty', (assign, forcing, j)
                if len(cands) == 1:
                    i, side = cands[0]
                    assign[i] = side
                    forcing.setdefault(i, []).append((j, side))
                    changed = True
                    break
                # a gear that appears on both sides for this column is not forced
            nb >>= 1
            j += 1
        if not changed:
            return 'ok', (assign, forcing)


def forced_sides(covL, covR, T, h):
    """columns with a unique (gear,side) candidate among ALL gears (no propagation):
    returns dict gear index -> set of sides forced, plus the witness columns."""
    k = len(covL)
    need = ((1 << T) - 1) ^ (1 << h)
    forced = {}
    for j in range(T):
        if not ((need >> j) & 1):
            continue
        cands = []
        for i in range(k):
            if (covL[i] >> j) & 1:
                cands.append((i, 0))
            if (covR[i] >> j) & 1:
                cands.append((i, 1))
        if len(cands) == 1:
            i, side = cands[0]
            forced.setdefault(i, {}).setdefault(side, []).append(j)
    return forced


# --------------------------------------------------------------- free gears (C2+f)
def glue_free(gears, us, x0, L, v, R, f):
    """two colours plus up to f gears given an arbitrary residue (not covering the hole)."""
    T, h, covL, covR = cov_pair(gears, us, x0, L, R, x0 + L + v)
    k = len(gears)
    target = ((1 << T) - 1) ^ (1 << h)
    # options per gear: (tag, mask).  tag 'L','R' or ('F', r)
    opts = []
    for i, (g, u) in enumerate(zip(gears, us)):
        o = [('L', covL[i]), ('R', covR[i])]
        fr = []
        for r in range(g):
            m = covbits(g, u, r, T)    # free gear: any base residue r mod g
            if (m >> h) & 1:
                continue
            fr.append((('F', r), m))
        opts.append((o, fr))
    best = [None]

    def rec(i, acc, used, chosen):
        if acc & target == target:
            best[0] = list(chosen)
            return True
        if i == k:
            return False
        # prune: remaining gears' best possible union
        rem = 0
        for t in range(i, k):
            for _, m in opts[t][0]:
                rem |= m
            if used < f:
                for _, m in opts[t][1]:
                    rem |= m
        if (target & ~acc) & ~rem:
            return False
        for tag, m in opts[i][0]:
            chosen.append(tag)
            if rec(i + 1, acc | m, used, chosen):
                return True
            chosen.pop()
        if used < f:
            for tag, m in opts[i][1]:
                chosen.append(tag)
                if rec(i + 1, acc | m, used + 1, chosen):
                    return True
                chosen.pop()
        return False

    if rec(0, 0, 0, []):
        return best[0]
    return None


# --------------------------------------------------------------- shifted glue (Cs)
def pattern(gears, us, x0, L, s, W, sigma):
    """blocked pattern over offsets 0..W-1 for left base x0+1, right base x0+1+s."""
    m = 0
    for i, (g, u) in enumerate(zip(gears, us)):
        base = (x0 + 1 + s) if sigma[i] else (x0 + 1)
        m |= covbits(g, u, base, W)
    return m


def shifted_best(gears, us, x0, L, v, R, tmax=8):
    """for t = 1..tmax the overlapped glue: right base x0+1+v+t.  A success certifies
    F(M) >= (length of the blocked run covering offsets 0..L+R-2-t) + 1.
    Returns the least t with a full blocked run of length L+R-1-t, or None."""
    k = len(gears)
    for t in range(1, tmax + 1):
        T = L + R - 1 - t
        if T <= 0:
            break
        covL = [covbits(g, u, x0 + 1, T) for g, u in zip(gears, us)]
        covR = [covbits(g, u, x0 + 1 + v + t, T) for g, u in zip(gears, us)]
        target = (1 << T) - 1
        suf = [0] * (k + 1)
        for i in range(k - 1, -1, -1):
            suf[i] = suf[i + 1] | covL[i] | covR[i]
        sol = [None]

        def rec(i, acc, mask):
            need = target & ~acc
            if not need:
                sol[0] = mask
                return True
            if i == k or (need & ~suf[i]):
                return False
            for bit, c in ((0, covL[i]), (1, covR[i])):
                if rec(i + 1, acc | c, mask | (bit << i)):
                    return True
            return False
        if rec(0, 0, 0):
            return t, sol[0]
    return None


# --------------------------------------------------------------- run enumeration
def runs_with(opens, gaps, vmin=6, sum_gt=None, only_attaining=None, cap=None):
    """yield (x0, L, v, R) for 3-runs; middle index i, left gaps[i-1], right gaps[i+1]."""
    n = gaps.size
    left = np.roll(gaps, 1)
    right = np.roll(gaps, -1)
    sel = gaps >= vmin
    if sum_gt is not None:
        sel &= (left + right) > sum_gt
    if only_attaining is not None:
        att = np.zeros(n, dtype=bool)
        for v, s in only_attaining.items():
            att |= (gaps == v) & (left + right == s)
        sel &= att
    idx = np.flatnonzero(sel)
    idx = idx[(idx >= 1) & (idx + 1 < n)]
    if cap is not None:
        idx = idx[:cap]
    for i in idx.tolist():
        yield int(opens[i - 1]), int(gaps[i - 1]), int(gaps[i]), int(gaps[i + 1])


# --------------------------------------------------------------- reporting helpers
def strikers(gears, us, c):
    return [g for g, u in zip(gears, us) if c % g in (u % g, (-u) % g)]


def flank_table(gears, us, lo, hi):
    """columns lo..hi with their strikers; returns dict col -> [gears]"""
    return {c: strikers(gears, us, c) for c in range(lo, hi + 1)}
