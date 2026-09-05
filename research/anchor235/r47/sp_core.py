"""Branch 2g.i.a.i: separability of flanks by gear.  Prover, 2026-09-05.

For a 3-run (x0, L, v, R) of a machine M the LEFT FLANK is the blocked column set
[x0+1, x1-1] and the RIGHT FLANK is [x2+1, x3-1] (x1 = x0+L, x2 = x1+v, x3 = x2+R).
A BLOCKING ASSIGNMENT picks, for every flank column, a gear that strikes it; A is the set
used on the left, B the set used on the right.

    s(run)  = min |A n B|                     the SHARED NUMBER
    u(run)  = min |A u B| among the minimisers the USED NUMBER
    sigma   = s / u                           the SEPARATION INDEX
    ov(run) = # gears striking both flanks    the RAW OVERLAP (>= s)

s = 0 ("separable") gives a search-free certificate F_2(M) >= L + R: colour A left, B right,
unused left, take the CRT point z with z = x0 mod (left gears), z = x2 - L mod (right gears);
then z+1..z+L-1 are blocked, z+L is an opening (x1 mod every left gear, x2 mod every right
gear) and z+L+1..z+L+R-1 are blocked.

Graded version: for a full two-colouring sigma (no shared, no unused) let a = the longest
blocked run left of x1 using left gears only and b = the longest right of x2 using right gears
only; the same CRT point certifies F_2 >= a+b+2, so the LOSS is c = (L-1-a) + (R-1-b).
c = 0 iff s = 0.

Both A and B may be taken MINIMAL covers (shrinking a cover can only shrink the intersection),
which is what makes the exact minimum cheap: at most a few dozen minimal covers a side.
"""
import numpy as np
from math import prod

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
FLAD = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}


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


# ------------------------------------------------------------------ flank masks
def flank_masks(gears, us, start, n):
    """bit i set iff gear strikes column start+i, for 0 <= i < n.  Returns list per gear."""
    out = []
    for g, u in zip(gears, us):
        m = 0
        for t in {u % g, (-u) % g}:
            j = (t - start) % g
            while j < n:
                m |= 1 << j
                j += g
        out.append(m)
    return out


def run_masks(gears, us, x0, L, v, R):
    """left-flank masks (bit i = column x1-1-i, i.e. bit 0 is nearest x1, running LEFTWARD)
    and right-flank masks (bit j = column x2+1+j, running RIGHTWARD)."""
    x1, x2 = x0 + L, x0 + L + v
    nl, nr = L - 1, R - 1
    ml = []
    for g, u in zip(gears, us):
        m = 0
        for t in {u % g, (-u) % g}:
            # column x1-1-i = t (mod g)  =>  i = (x1-1-t) mod g
            i = (x1 - 1 - t) % g
            while i < nl:
                m |= 1 << i
                i += g
        ml.append(m)
    mr = flank_masks(gears, us, x2 + 1, nr)
    return ml, mr, nl, nr


# ------------------------------------------------------------------ the exact minimum
def _or_table(masks, n):
    """orv[mask] = OR of masks[i] over i in mask, for all 2^n subsets."""
    orv = [0] * (1 << n)
    for m in range(1, 1 << n):
        lb = m & -m
        orv[m] = orv[m ^ lb] | masks[lb.bit_length() - 1]
    return orv


def _minimal_covers(orv, n, full):
    """all subset masks that cover `full` and whose every one-element removal does not."""
    out = []
    for m in range(1 << n):
        if orv[m] != full:
            continue
        mm = m
        minimal = True
        while mm:
            lb = mm & -mm
            mm ^= lb
            if orv[m ^ lb] == full:
                minimal = False
                break
        if minimal:
            out.append(m)
    return out


def separability(ml, mr, nl, nr):
    """returns dict with s, u, sigma, ov, shared sets, loss, and the covers."""
    n = len(ml)
    fullL = (1 << nl) - 1 if nl > 0 else 0
    fullR = (1 << nr) - 1 if nr > 0 else 0
    orL = _or_table(ml, n)
    orR = _or_table(mr, n)
    assert orL[(1 << n) - 1] == fullL and orR[(1 << n) - 1] == fullR, "flank not blocked"
    CL = _minimal_covers(orL, n, fullL)
    CR = _minimal_covers(orR, n, fullR)
    best = None
    for A in CL:
        for B in CR:
            sh = A & B
            k = bin(sh).count('1')
            un = bin(A | B).count('1')
            key = (k, un)
            if best is None or key < best[0]:
                best = (key, A, B, sh)
    (s, u), A, B, sh = best
    ov = sum(1 for i in range(n) if ml[i] and mr[i])
    # the loss of the graded certificate: max over full two-colourings of a + b
    besta = -1
    for m in range(1 << n):
        a = _trail_ones(orL[m], nl)
        b = _trail_ones(orR[((1 << n) - 1) ^ m], nr)
        if a + b > besta:
            besta = a + b
            bm = m
    loss = (nl + nr) - besta
    return dict(s=s, u=u, sigma=(s / u if u else 0.0), ov=ov, sharedmask=sh,
                Amask=A, Bmask=B, loss=loss, bestcol=bm, ncovL=len(CL), ncovR=len(CR))


def _trail_ones(x, n):
    a = 0
    while a < n and (x >> a) & 1:
        a += 1
    return a


def all_min_shared(ml, mr, nl, nr, s):
    """every shared set attaining the minimum s (as gear-index masks)."""
    n = len(ml)
    fullL = (1 << nl) - 1 if nl > 0 else 0
    fullR = (1 << nr) - 1 if nr > 0 else 0
    orL = _or_table(ml, n)
    orR = _or_table(mr, n)
    CL = _minimal_covers(orL, n, fullL)
    CR = _minimal_covers(orR, n, fullR)
    out = set()
    for A in CL:
        for B in CR:
            if bin(A & B).count('1') == s:
                out.add(A & B)
    return sorted(out)


def sep_run(gears, us, x0, L, v, R):
    ml, mr, nl, nr = run_masks(gears, us, x0, L, v, R)
    return separability(ml, mr, nl, nr)


# ------------------------------------------------------------------ run enumeration
def attaining_runs(opens, gaps, N, vmin=6):
    """all 3-runs whose (v, L+R) attains N(v), for v >= vmin."""
    n = gaps.size
    left, right = np.roll(gaps, 1), np.roll(gaps, -1)
    ns = left + right
    out = []
    for v in sorted(N):
        if v < vmin:
            continue
        idx = np.flatnonzero((gaps == v) & (ns == N[v]))
        idx = idx[(idx >= 1) & (idx + 1 < n)]
        for i in idx.tolist():
            out.append((int(opens[i - 1]), int(gaps[i - 1]), int(v), int(gaps[i + 1])))
    return out


def letters(g, v):
    """move-lemma class of gear g at middle v: 'pad' if g | v, 'letter' if v = +-d_g mod g."""
    u = pow(6, -1, g)
    d = (2 * u) % g
    if v % g == 0:
        return 'pad'
    if v % g in (d % g, (-d) % g):
        return 'letter'
    return 'stuck'
