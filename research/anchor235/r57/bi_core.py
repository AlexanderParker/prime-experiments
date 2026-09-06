"""bi_core.py -- shared machinery for the branching-identity branch (R4.i.a).

Everything here is exact integer arithmetic on full periods.

Objects
-------
A machine M is a list of gears (primes >= 5).  Its period is P = prod q, its openings are the
columns struck by no gear, and its gap sequence is the cyclic sequence of differences of
consecutive openings (N = number of openings = number of gaps).

Adding a gear q' makes q' copies of M's period; copy j realises deletion phase
r_j = -c - jP (mod q'), and j -> r_j is a bijection of Z_{q'} (docs/proofs/05 (A)).  So every
statement about copies is a statement about PHASES, and we work with phases throughout.

Letters.  With c = 6^{-1} mod q' and d = 2c, a gap value v has letter
    PAD  (0) if v = 0      (mod q')
    UP   (1) if v = +d     (mod q')
    DOWN (2) if v = -d     (mod q')
    BAD  (3) otherwise.
An opening x is struck in phase r iff x - r in {0, d}; writing x = r + c + t c with t in {-1,+1}
(t = -1 <-> x = r, t = +1 <-> x = r + d) a gap between two struck openings equals
(t_next - t_prev) c, i.e. PAD keeps t, UP takes t from -1 to +1, DOWN from +1 to -1.  That is
file 05 (F) in coordinates.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31]

PAD, UP, DOWN, BAD = 0, 1, 2, 3


def u_of(g):
    return pow(6, -1, g)


def sieve_machine(gears, vs=None):
    """Full-period openings of a machine, as int64 in [0, P).  vs = tooth values (family)."""
    if vs is None:
        vs = [u_of(g) for g in gears]
    P = 1
    for g in gears:
        P *= g
    blocked = np.zeros(P, dtype=bool)
    for g, v in zip(gears, vs):
        blocked[v % g::g] = True
        blocked[(-v) % g::g] = True
    O = np.flatnonzero(~blocked).astype(np.int64)
    return P, O


def gaps_of(P, O):
    """Cyclic gap sequence: gap(n) = O[n+1] - O[n], last wraps."""
    g = np.empty(O.size, dtype=np.int64)
    g[:-1] = O[1:] - O[:-1]
    g[-1] = O[0] + P - O[-1]
    return g


def letter_table(vmax, q, u=None):
    """Letter of every gap VALUE 0..vmax, as a small lookup table."""
    d = (2 * (u_of(q) if u is None else u)) % q
    v = np.arange(vmax + 1, dtype=np.int64)
    r = v % q
    t = np.full(vmax + 1, BAD, dtype=np.uint8)
    t[r == 0] = PAD
    t[r == d] = UP
    t[r == (-d) % q] = DOWN
    t[0] = BAD          # a gap is never 0; keep the slot harmless
    return t


def letters(gaps, q, u=None):
    """Letter array (PAD/UP/DOWN/BAD) of a gap array with respect to gear q.  Table lookup, so a
    200-million-entry gap array never becomes int64."""
    g = np.asarray(gaps)
    tab = letter_table(int(g.max()), q, u)
    return tab[g]


def add_gear_stream(P_old, O_old, q, want_gaps=True, u=None):
    """Build M + q from M's full-period openings.

    Returns (P_new, N_new, gaps_new_int16_or_None, order_hist).
    order_hist[J] = number of new gaps of order exactly J (J = number of old gaps fused).
    The new gaps are produced copy by copy so nothing of size q * N_old is materialised as int64.
    """
    N_old = O_old.size
    res = (O_old % q).astype(np.int64)
    u = u_of(q) if u is None else u
    t1, t2 = u % q, (-u) % q
    P_new = P_old * q
    gap_parts = []
    order_hist = np.zeros(64, dtype=np.int64)
    first_open = None   # first surviving opening overall (for the final wrap)
    first_idx = None
    prev_open = None    # last surviving opening seen so far (absolute)
    prev_idx = None     # its tiled old-gap index
    N_new = 0
    for j in range(q):
        sh = (j * P_old) % q
        r = res + sh
        r -= q * (r >= q)
        keep = np.flatnonzero((r != t1) & (r != t2))
        if keep.size == 0:
            continue
        opens = O_old[keep] + j * P_old
        tidx = keep + j * N_old
        N_new += keep.size
        if prev_open is not None:
            # seam gap from the previous copy's last survivor to this copy's first
            if want_gaps:
                gap_parts.append(np.array([opens[0] - prev_open], dtype=np.uint8))
            order_hist[tidx[0] - prev_idx] += 1
        else:
            first_open, first_idx = int(opens[0]), int(tidx[0])
        if keep.size > 1:
            if want_gaps:
                gap_parts.append((opens[1:] - opens[:-1]).astype(np.uint8))
            order_hist += np.bincount(tidx[1:] - tidx[:-1], minlength=64)[:64]
        prev_open, prev_idx = int(opens[-1]), int(tidx[-1])
    # final wrap
    if want_gaps:
        gap_parts.append(np.array([first_open + P_new - prev_open], dtype=np.uint8))
    order_hist[first_idx + q * N_old - prev_idx] += 1
    gaps_new = np.concatenate(gap_parts) if want_gaps else None
    return P_new, N_new, gaps_new, order_hist, first_open


def machine_gaps(top, cache=True):
    """(P, first_opening, cyclic gap array uint8) of machine {5..top}: from the sieve for
    top <= 23 and by the merge construction above that.  Cached in results/ for top >= 29.
    The first opening is kept because the ABSOLUTE residues of the openings decide which copy
    realises which deletion phase; letters and words do not need it, the direct build does."""
    gears = [p for p in PRIMES if p <= top]
    if top <= 23:
        P, O = sieve_machine(gears)
        return P, int(O[0]), gaps_of(P, O).astype(np.uint8)
    path = os.path.join(OUT, f"gaps_m{top}.npy")
    meta = os.path.join(OUT, f"gaps_m{top}.meta.txt")
    if cache and os.path.exists(path) and os.path.exists(meta):
        P, f0 = [int(x) for x in open(meta).read().split()]
        return P, f0, np.load(path)
    below = max(p for p in PRIMES if p < top)
    P_old, f_old, g_old = machine_gaps(below, cache=cache)
    O_old = np.empty(g_old.size, dtype=np.int64)
    O_old[0] = f_old
    np.cumsum(g_old[:-1].astype(np.int64), out=O_old[1:])
    O_old[1:] += f_old
    P_new, N_new, g, _, f_new = add_gear_stream(P_old, O_old, gears[-1])
    del O_old
    if cache:
        np.save(path, g)
        open(meta, "w").write(f"{P_new} {f_new}")
    return P_new, f_new, g


# ---------------------------------------------------------------- words and chains

def wrap_pad(a, k):
    """a with its first k entries appended, so cyclic windows of length <= k+1 are contiguous."""
    if k <= 0:
        return a
    if k <= a.size:
        return np.concatenate([a, a[:k]])
    reps = -(-k // a.size)
    return np.concatenate([a, np.tile(a, reps)[:k]])


def word_counts(lt, m, chunk=20_000_000):
    """(W_m, Z_m): the number of cyclic positions n at which the m gaps lt[n..n+m-1] form a
    LEGAL word (file 05 (F): no two consecutive nonzero letters equal, pads transparent), and
    the number at which they are ALL PAD.  W_0 = Z_0 = N.  Chunked, so a 200-million-gap
    machine costs one padded copy and nothing else."""
    N = lt.size
    if m == 0:
        return N, N
    lp = wrap_pad(lt, m)
    W = Z = 0
    pos = 0
    while pos < N:
        end = min(pos + chunk, N)
        n = end - pos
        ok = np.ones(n, dtype=bool)
        allpad = np.ones(n, dtype=bool)
        prev_nz = np.zeros(n, dtype=np.uint8)
        for i in range(m):
            c = lp[pos + i:end + i]
            ok &= (c != BAD)
            ok &= ~((c != PAD) & (c == prev_nz))
            allpad &= (c == PAD)
            prev_nz = np.where(c == PAD, prev_nz, c)
        W += int(ok.sum())
        Z += int((allpad & ok).sum())
        pos = end
    return W, Z


def chain_counts(lt, q, N, rmax=12):
    """C_r for r = 0 .. rmax.  C_0 = q N; for r >= 1, C_r = W_{r-1} + Z_{r-1}."""
    C = [q * N]
    for r in range(1, rmax + 1):
        W, Z = word_counts(lt, r - 1)
        C.append(W + Z)
        if W == 0:
            C.extend([0] * (rmax - r))
            break
    return C[:rmax + 1]


def orders_from_chains(C):
    """n_J = C_{J-1} - 2 C_J + C_{J+1} for J >= 1."""
    C = list(C) + [0, 0]
    return [C[J - 1] - 2 * C[J] + C[J + 1] for J in range(1, len(C) - 1)]


# ---------------------------------------------------------------- the local weight eps_J

def eps_of_window(lts, q):
    """eps_J for a J-window given as a list of J letter arrays (offsets 0 .. J-1).

    eps_J(n) = the number of phases in which the J consecutive gaps at n fuse into exactly one
    gap of M + q': the J-1 interior openings all struck, the two endpoints not struck.

    J = 1 is special (no interior opening): eps_1 = q - 4 + |{phases striking both endpoints}|
    = q - 2 (PAD), q - 3 (UP or DOWN), q - 4 (BAD).
    """
    J = len(lts)
    if J == 1:
        l = lts[0]
        e = np.full(l.size, q - 4, dtype=np.int64)
        e[l == PAD] = q - 2
        e[(l == UP) | (l == DOWN)] = q - 3
        return e
    total = np.zeros(lts[0].size, dtype=np.int64)
    for t0 in (-1, +1):
        # read the middle word lts[1] .. lts[J-2] from tooth t0 at x_1
        ok = np.ones(lts[0].size, dtype=bool)
        t = np.full(lts[0].size, t0, dtype=np.int8)
        for i in range(1, J - 1):
            c = lts[i]
            ok &= (c != BAD)
            # UP requires t = -1 and sets +1; DOWN requires t = +1 and sets -1; PAD keeps t
            ok &= ~((c == UP) & (t == 1))
            ok &= ~((c == DOWN) & (t == -1))
            t = np.where(c == UP, np.int8(1), np.where(c == DOWN, np.int8(-1), t))
        # left endpoint x_0 unstruck: first gap not PAD and not the letter equal to t0 * d
        l1 = lts[0]
        bad_left = UP if t0 == 1 else DOWN
        ok &= (l1 != PAD) & (l1 != bad_left)
        # right endpoint x_J unstruck: last gap not PAD and not the letter equal to -t_{J-1} * d
        lJ = lts[-1]
        ok &= (lJ != PAD)
        ok &= ~((t == 1) & (lJ == DOWN))
        ok &= ~((t == -1) & (lJ == UP))
        total += ok
    return total


def residues_mod(gaps, q, r0=0):
    """Residues mod q of the openings, from the cyclic gap array (opening 0 has residue r0)."""
    N = gaps.size
    res = np.empty(N, dtype=np.uint8)
    res[0] = r0 % q
    acc = int(r0) % q
    step = 4_000_000
    cur = np.zeros(1, dtype=np.int64)
    pos = 0
    while pos < N - 1:
        end = min(pos + step, N - 1)
        blk = np.cumsum(gaps[pos:end].astype(np.int64)) + acc
        res[pos + 1:end + 1] = (blk % q).astype(np.uint8)
        acc = int(blk[-1] % q)
        pos = end
    return res


def orders_direct(res, q, P_old, chunk=20_000_000):
    """Order histogram of M + q, built directly on the TILED period, copy by copy in order.

    Copy j strikes the old opening i iff res[i] is in {p_j, p_j + d} with p_j = -u - jP (mod q);
    the openings of M + q are the unstruck tiled indices and the order of a new gap is the
    distance between consecutive unstruck tiled indices, cyclically over the whole tiled period.
    The seams between copies are handled by carrying the last unstruck index across copies -- a
    run of struck openings that crosses a seam is a genuine merge and must not be cut.

    This is the independent vehicle: it never mentions words, letters or chain counts.
    """
    N = res.size
    u = u_of(q)
    d = (2 * u) % q
    hist = np.zeros(64, dtype=np.int64)
    last = None
    first = None
    for j in range(q):
        p = (-u - j * P_old) % q
        p2 = (p + d) % q
        pos = 0
        while pos < N:
            end = min(pos + chunk, N)
            r = res[pos:end]
            keep = np.flatnonzero((r != p) & (r != p2))
            if keep.size:
                idx = keep.astype(np.int64) + pos + j * N
                if last is None:
                    first = int(idx[0])
                else:
                    hist[int(idx[0]) - last] += 1
                if idx.size > 1:
                    hist += np.bincount(np.diff(idx), minlength=64)[:64]
                last = int(idx[-1])
            pos = end
    if last is not None:
        hist[first + q * N - last] += 1
    return hist
