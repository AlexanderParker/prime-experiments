"""lc_core.py -- the closure step as a MACHINE: dictionary in, dictionary out.

The parent branch (research/proof/branching_identity.md, Theorem 5) proves that the depth-m
window dictionary of M + q' with multiplicity is determined by the depth-K window dictionary of
M with multiplicity, K = K_m(M -> q').  It ran the step once per rung, always from a period it
had built.  Here the step is implemented so that its OUTPUT is a legal INPUT, and the ladder can
be iterated without ever building a period again.

A dictionary is a pair (win, mult):
    win  : uint8 array (n, K)  -- n distinct windows of K consecutive gap sizes.  A trailing run
           of zeros means "the window stops here" (the span-bounded form); a row with no zero has
           depth exactly K.
    mult : int64 array (n,)    -- the number of openings of M carrying that window; sum = N.

The step, for one phase z in Z_{q'}:
    offsets  o_0 = 0, o_i = g_1 + ... + g_i          (the openings of M inside the window)
    struck   o_i is struck iff (o_i + z) mod q' in {0, d},  d = 2 * 6^{-1} mod q'
    survive  o_0 unstruck  =>  the pair (window, z) IS one opening of M + q'
    new gaps successive differences of the unstruck offsets

Every opening of M + q' arises exactly once as such a pair (file 05 (A): copy -> phase is a
bijection of Z_{q'}), so the multiset of new windows weighted by mult is D_m^#(M + q') exactly --
PROVIDED every surviving pair reaches m new gaps inside the K old ones.  Pairs that do not are
the LOSS; loss = 0 certifies the step.

Two forms:
  * fixed depth   -- emit exactly m new gaps; loss = the pairs that ran out of window.
  * span-bounded  -- emit new gaps while the cumulative span stays <= S.  Then loss = 0 BY
    CONSTRUCTION: running out of the input window means the next surviving opening lies beyond
    offset S, so the next new gap would leave the span cap anyway.  A span-S dictionary records
    exactly the opening pattern of the machine inside [x, x + S] for every opening x, and THAT
    object is exactly closed under the rung step.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]


def u_of(g):
    return pow(6, -1, g)


# ------------------------------------------------------------------ base machines (periods)

def sieve_machine(gears):
    P = 1
    for g in gears:
        P *= g
    blocked = np.zeros(P, dtype=bool)
    for g in gears:
        v = u_of(g)
        blocked[v % g::g] = True
        blocked[(-v) % g::g] = True
    return P, np.flatnonzero(~blocked).astype(np.int64)


def gaps_of(P, O):
    g = np.empty(O.size, dtype=np.int64)
    g[:-1] = O[1:] - O[:-1]
    g[-1] = O[0] + P - O[-1]
    return g


def base_gaps(top):
    """Cyclic gap array (uint8) of machine {5..top} by direct sieve (top <= 23)."""
    gears = [p for p in PRIMES if p <= top]
    P, O = sieve_machine(gears)
    return P, gaps_of(P, O).astype(np.uint8)


# ------------------------------------------------------------------ grouping

def pack64(win):
    """Rows of a uint8 window array packed into ceil(K/8) uint64 columns, most significant
    first, so that lexicographic order on the packed columns is lexicographic on the rows."""
    n, K = win.shape
    nb = (K + 7) // 8
    out = np.zeros((nb, n), dtype=np.uint64)
    for i in range(K):
        out[i // 8] = (out[i // 8] << np.uint64(8)) | win[:, i].astype(np.uint64)
    if K % 8:
        out[nb - 1] <<= np.uint64(8 * (8 - K % 8))
    return out


def group_windows(win, mult):
    """Deduplicate rows of `win`, summing `mult` exactly (int64 reduceat, no float accumulator)."""
    n, K = win.shape
    if n <= 1:
        return win, mult
    p = pack64(win)
    order = np.argsort(p[0], kind="stable") if p.shape[0] == 1 else np.lexsort(p[::-1])
    ps = p[:, order]
    new = np.zeros(n, dtype=bool)
    new[0] = True
    for r in range(ps.shape[0]):
        new[1:] |= ps[r, 1:] != ps[r, :-1]
    starts = np.flatnonzero(new)
    return win[order[starts]], np.add.reduceat(mult[order], starts)


class Accum:
    """Row accumulator that regroups whenever it gets big, so peak memory stays bounded."""

    HARDCAP = 55_000_000

    def __init__(self, width, limit=8_000_000):
        self.w = []
        self.m = []
        self.rows = 0
        self.width = width
        self.limit = limit

    def add(self, win, mult):
        if win.shape[0] == 0:
            return
        self.w.append(win)
        self.m.append(mult)
        self.rows += win.shape[0]
        if self.rows > self.limit:
            self.compact()

    def compact(self):
        if len(self.w) <= 1:
            if self.w:
                self.w[0], self.m[0] = group_windows(self.w[0], self.m[0])
                self.rows = self.w[0].shape[0]
            return
        if self.rows > 2 * self.HARDCAP:
            raise MemoryError(f"pending rows {self.rows:,} would not group inside the budget")
        ww, mm = group_windows(np.concatenate(self.w), np.concatenate(self.m))
        self.w, self.m, self.rows = [ww], [mm], ww.shape[0]
        if self.rows > self.HARDCAP:
            raise MemoryError(f"dictionary exceeded {self.HARDCAP:,} distinct rows "
                              f"({self.rows:,} and still growing)")

    def result(self):
        self.compact()
        if not self.w:
            return np.zeros((0, self.width), dtype=np.uint8), np.zeros(0, dtype=np.int64)
        return self.w[0], self.m[0]


def dict_from_gaps(gaps, K, span_cap=None):
    """D_K^#(M) from a full period; with span_cap, the span-bounded dictionary V_S(M)."""
    N = gaps.size
    gp = np.concatenate([gaps, gaps[:K]])
    acc = Accum(K)
    step = 20_000_000
    for lo in range(0, N, step):
        hi = min(lo + step, N)
        nn = hi - lo
        win = np.empty((nn, K), dtype=np.uint8)
        for i in range(K):
            win[:, i] = gp[lo + i:hi + i]
        if span_cap is not None:
            off = np.zeros(nn, dtype=np.int32)
            for i in range(K):
                off += win[:, i]
                win[:, i] = np.where(off <= span_cap, win[:, i], np.uint8(0))
                off = np.minimum(off, span_cap + 1)
        acc.add(win, np.ones(nn, dtype=np.int64))
    return acc.result()


# ------------------------------------------------------------------ the closure step

def closure_step(win, mult, q, m=None, mode="span", chunk=1_500_000, tag_order=False,
                 collect=True, theta=0):
    """One rung of the ladder.  Returns (win2, mult2, stats).

    mode = "span"  : the input is a span-bounded dictionary V_S(M) (every row terminated by a
                     zero).  Every unstruck offset inside the window is at span <= S, so the
                     output is simply the differences of the unstruck offsets, zero-padded --
                     loss = 0 by construction, and the output is V_S(M + q').
    mode = "fixed" : the input is a full-depth dictionary D_K^#(M).  A pair (window, phase) is
                     kept only if it yields m complete new gaps inside the K old ones; the rest
                     are the LOSS.

    tag_order      : also emit, as the LAST column, the ORDER of the first new gap (how many old
                     gaps it fuses), and measure kmax = the largest number of old gaps spanned.
    """
    n, K = win.shape
    d = (2 * u_of(q)) % q
    if m is None:
        m = K
    width = m + (1 if tag_order else 0)
    acc = Accum(width)
    loss = 0
    kmax = 0
    mass = 0
    over0 = 0
    ordh = np.zeros(K + 2, dtype=np.int64)
    specJ = np.zeros((K + 2) * 256, dtype=np.float64)   # (order, value) mass, exact < 2^53
    mmin = 1 << 30
    depth_full = int((win[:, K - 1] != 0).sum())
    lut = np.zeros(q + 1, dtype=np.uint8)          # 1 = struck; index q = "not an opening"
    lut[q] = 1
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        w = win[lo:hi]
        mu = mult[lo:hi]
        nn = hi - lo
        off = np.zeros((K + 1, nn), dtype=np.int16)
        for i in range(K):
            off[i + 1] = off[i] + w[:, i]
        alive = np.ones(nn, dtype=bool)
        offq = np.empty((K + 1, nn), dtype=np.uint8)
        offq[0] = (off[0] % q).astype(np.uint8)
        for i in range(1, K + 1):
            alive &= (w[:, i - 1] != 0)
            offq[i] = np.where(alive, (off[i] % q).astype(np.uint8), np.uint8(q))
        ar = np.arange(nn)
        oo = np.zeros((m + 2, nn), dtype=np.int16)
        ii = np.zeros((m + 2, nn), dtype=np.int16) if tag_order else None
        for z in range(q):
            lut[:q] = 0
            lut[(-z) % q] = 1
            lut[(d - z) % q] = 1
            free = lut[offq] == 0                   # (K+1, nn) bool: offset i survives
            ok = free[0]
            if not ok.any():
                continue
            cnt = free.sum(axis=0, dtype=np.int16)  # surviving offsets inside the window
            c = np.zeros(nn, dtype=np.int16)
            for i in range(K + 1):
                fi = free[i]
                if not fi.any():
                    continue
                slot = np.minimum(c[fi], m + 1)
                rows = ar[fi]
                oo[slot, rows] = off[i][fi]
                if tag_order:
                    ii[slot, rows] = i
                c += fi
            ncap = np.minimum(cnt - 1, m)           # number of new gaps we can record
            outw = np.zeros((nn, width), dtype=np.uint8)
            for j in range(m):
                outw[:, j] = np.where(j < ncap, oo[j + 1] - oo[j], 0).astype(np.uint8)
            if tag_order:
                o1 = np.where(ncap >= 1, ii[1], 0)
                outw[:, width - 1] = o1.astype(np.uint8)
                kk = np.where(ncap >= 1, ii[np.minimum(np.maximum(ncap, 0), m + 1), ar], 0)
                kmax = max(kmax, int(kk[ok].max()))
                key = (o1[ok].astype(np.int64) << 8) | outw[:, 0][ok].astype(np.int64)
                specJ += np.bincount(key, weights=mu[ok].astype(np.float64),
                                     minlength=(K + 2) * 256)[:(K + 2) * 256]
                for jj in range(1, int(o1.max()) + 1):
                    mj = ok & (o1 == jj)
                    if mj.any():
                        ordh[jj] += int(mu[mj].sum())
            if mode == "span":
                good = ok
            else:
                good = ok & (cnt - 1 >= m)
            if theta:
                # keep only windows whose total span can still carry a record of >= theta.
                # A sub-run of a window never spans more than the window, so a row below the
                # threshold can never be the ancestor of a gap of size >= theta.
                good = good & (outw[:, :m].astype(np.int32).sum(axis=1) >= theta)
            bad = ok & ~good
            if bad.any():
                loss += int(mu[bad].sum())
            if ok.any():
                mmin = min(mmin, int((cnt[ok] - 1).min()))
            e0 = ok & (cnt <= 1)
            if e0.any():
                over0 += int(mu[e0].sum())
            if good.all():
                mass += int(mu.sum())
                if collect:
                    acc.add(*group_windows(np.ascontiguousarray(outw), mu))
            else:
                sel = np.flatnonzero(good)
                if sel.size:
                    mass += int(mu[sel].sum())
                    if collect:
                        acc.add(*group_windows(np.ascontiguousarray(outw[sel]), mu[sel]))
    win2, mult2 = acc.result()
    ng = width - (1 if tag_order else 0)
    if win2.shape[0]:
        used = ng
        while used > 1 and not win2[:, used - 1].any():
            used -= 1
        if used < ng:
            keep = list(range(used)) + ([width - 1] if tag_order else [])
            win2, mult2 = group_windows(np.ascontiguousarray(win2[:, keep]), mult2)
    return win2, mult2, {"loss": loss, "kmax": kmax, "mass": mass, "depth_full": depth_full,
                         "over0": over0, "ordh": ordh, "mmin": mmin,
                         "specJ": np.rint(specJ).astype(np.int64).reshape(K + 2, 256), "n_in": n, "K_in": K}


# ------------------------------------------------------------------ readouts

def _grouped_sum(keys, mult, size):
    h = np.zeros(size, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    new = np.empty(keys.size, dtype=bool)
    new[0] = True
    np.not_equal(ks[1:], ks[:-1], out=new[1:])
    st = np.flatnonzero(new)
    h[ks[st]] = np.add.reduceat(mult[order], st)
    return h


def spectrum(win, mult):
    v = win[:, 0].astype(np.int64)
    return _grouped_sum(v, mult, int(v.max()) + 1)


def order_hist(win, mult, ordcol):
    v = win[:, ordcol].astype(np.int64)
    return _grouped_sum(v, mult, int(v.max()) + 1)


def fj(win, mult, j):
    """F_j = the largest span of j consecutive gaps realised (None if the depth is short)."""
    if win.shape[1] < j:
        return None
    good = win[:, j - 1] != 0
    if not good.any():
        return None
    return int(win[good, :j].astype(np.int64).sum(axis=1).max())


def letters_of(vals, q):
    d = (2 * u_of(q)) % q
    r = vals % q
    lt = np.full(vals.shape, 3, dtype=np.uint8)     # BAD
    lt[r == 0] = 0                                  # PAD
    lt[r == d] = 1                                  # UP
    lt[r == (-d) % q] = 2                           # DOWN
    return lt


def word_stats(win, mult, q):
    """(L, L_pad, W_m/N, Z_m/N) with respect to the incoming gear q, read on the dictionary.
    L = max(L_bare, L_pad) with L_bare the longest realised legal word using only NON-PAD
    letters and L_pad the longest realised legal word using at least one PAD letter (the corpus
    decomposition, alignment-rules 4.x).  L_allpad is the longest realised all-PAD run.  W[m] = the mult-weighted number of positions whose first m gaps form a legal
    word; Z[m] the all-PAD count."""
    n, K = win.shape
    lt = letters_of(win.astype(np.int64), q)
    alive = np.ones(n, dtype=bool)
    ok = np.ones(n, dtype=bool)
    bare = np.ones(n, dtype=bool)
    haspad = np.zeros(n, dtype=bool)
    allpad = np.ones(n, dtype=bool)
    prev = np.zeros(n, dtype=np.uint8)
    Wc, Zc = [int(mult.sum())], [int(mult.sum())]
    L = Lb = Lp = Lz = 0
    for i in range(K):
        alive &= (win[:, i] != 0)
        c = lt[:, i]
        ok &= alive & (c != 3) & ~((c != 0) & (c == prev))
        bare &= ok & (c != 0)
        haspad = ok & (haspad | (c == 0))
        allpad &= alive & (c == 0)
        prev = np.where(c == 0, prev, c)
        Wc.append(int(mult[ok].sum()))
        Zc.append(int(mult[allpad].sum()))
        if Wc[-1] > 0:
            L = i + 1
        if bare.any():
            Lb = i + 1
        if haspad.any():
            Lp = i + 1
        if Zc[-1] > 0:
            Lz = i + 1
    return L, Lp, Wc, Zc, Lb, Lz


def witnesses(win, mult, q, targets, maxout=3):
    """For each (order J -> value Q*_J), return up to `maxout` realised J-windows of M whose
    first J gaps sum to that value and which fuse into ONE gap of M + q' at some phase: the
    record's composition, flank + middles + flank, straight off the dictionary."""
    n, K = win.shape
    d = (2 * u_of(q)) % q
    out = {}
    for J, tv in targets.items():
        if J > K:
            continue
        cand = np.flatnonzero(win[:, :J].astype(np.int32).sum(axis=1) == tv)
        if cand.size == 0:
            continue
        w = np.ascontiguousarray(win[cand])
        nn = w.shape[0]
        off = np.zeros((J + 1, nn), dtype=np.int32)
        for i in range(J):
            off[i + 1] = off[i] + w[:, i]
        offq = (off % q).astype(np.uint8)
        got = []
        for z in range(q):
            a, b = (-z) % q, (d - z) % q
            strk = (offq == a) | (offq == b)
            good = ~strk[0] & ~strk[J]
            for i in range(1, J):
                good &= strk[i]
            for h in np.flatnonzero(good)[:maxout]:
                if len(got) < maxout:
                    got.append({"z": int(z), "gaps": [int(x) for x in w[h, :J]],
                                "letters": [int(x) for x in
                                            letters_of(w[h, :J].astype(np.int64), q)],
                                "mult": int(mult[cand[h]])})
        if got:
            out[J] = got
    return out
