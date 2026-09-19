"""Exact rung census: for every twin centre s <= SMAX and every rung s' of s, sieve the stretch
of s' exactly and record T(s'), the nearest-rung offset of s', a seeded random rung offset of s',
and the per-gear local factor L(s') (full period) and L_win(s') (actual window).

Output: rungs.csv next to this script.  Run: uv run --directory C:/dev/primes python sieve_rungs.py [SMAX] [S_LO] [S_HI]
"""
import sys, os, time, math, csv
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
SMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
S_LO = int(sys.argv[2]) if len(sys.argv) > 2 else 0
S_HI = int(sys.argv[3]) if len(sys.argv) > 3 else SMAX
GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
SMALL = 20000  # primes below this: slice assignment; above: banded fancy indexing


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.nonzero(s)[0]


def twin_centres_upto(n):
    """all s = 6c <= n with s-1, s+1 prime"""
    s = np.ones(n + 2, dtype=bool)
    s[:2] = False
    for p in range(2, int((n + 1) ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    c = np.arange(6, n + 1, 6)
    return c[s[c - 1] & s[c + 1]]


_P = None
_INV6 = None


def _init():
    global _P, _INV6
    P = primes_upto((SMAX + 1) ** 2 + 2)
    P = P[P >= 5]
    inv6 = np.where(P % 6 == 1, (5 * P + 1) // 6, (P + 1) // 6)
    assert np.all((6 * inv6) % P == 1)
    _P = P.astype(np.int64)
    _INV6 = inv6.astype(np.int64)


def sieve_stretch(sp):
    """boolean arrays over offsets j = -(2c'-1) .. 2c'-1: minus[i] = s'^2+6j-1 prime, plus[i] likewise."""
    c = sp // 6
    jlo = -(2 * c - 1)
    W = 4 * c - 1
    P = _P[_P <= sp + 1]
    inv6 = _INV6[: len(P)]
    sq = sp * sp
    # struck classes: j = -(sq -+ 1) * inv6 mod p
    r_minus = ((-(sq - 1)) % P) * inv6 % P
    r_plus = ((-(sq + 1)) % P) * inv6 % P
    f_minus = (r_minus - jlo) % P
    f_plus = (r_plus - jlo) % P
    arrs = []
    for first in (f_minus, f_plus):
        a = np.ones(W, dtype=bool)
        nsmall = int(np.searchsorted(P, SMALL))
        for i in range(nsmall):
            a[int(first[i])::int(P[i])] = False
        lo = nsmall
        n = len(P)
        while lo < n:
            pmin = int(P[lo])
            hi = int(np.searchsorted(P, 2 * pmin))
            hi = min(hi, n)
            h = W // pmin + 1
            pp = P[lo:hi]
            ff = first[lo:hi]
            off = ff[:, None] + pp[:, None] * np.arange(h, dtype=np.int64)[None, :]
            idx = off[off < W]
            a[idx] = False
            lo = hi
        arrs.append(a)
    return jlo, arrs[0], arrs[1]


def local_factor(sp, jlo, W):
    """L (full period) and L_win (actual window) over GEARS."""
    L = 1.0
    Lw = 1.0
    sq = sp * sp
    for g in GEARS:
        inv = pow(6, -1, g)
        r1 = (-(sq - 1)) % g * inv % g
        r2 = (-(sq + 1)) % g * inv % g
        struck = {r1, r2}
        L *= (g - len(struck)) / (g - 2)
        # actual window count of unstruck offsets (arithmetic count of j in [jlo, jlo+W) per class)
        jhi = jlo + W - 1
        cnt = sum((jhi - r) // g - (jlo - 1 - r) // g for r in struck)
        unstruck = W - cnt
        Lw *= unstruck / ((g - 2) / g * W)
    return L, Lw


def work(args):
    s, j, sp, seed = args
    t0 = time.time()
    jlo, mn, pl = sieve_stretch(sp)
    W = len(mn)
    tw = mn & pl
    idx = np.nonzero(tw)[0]
    T = len(idx)
    js = idx + jlo
    # nearest to centre, negative on ties
    order = np.lexsort((js, np.abs(js)))  # primary |j|, secondary j ascending (negative first)
    jn = int(js[order[0]]) if T else 0
    rng = np.random.default_rng(seed)
    jr = int(js[rng.integers(T)]) if T else 0
    L, Lw = local_factor(sp, jlo, W)
    return (s, j, sp, T, jn, jr, L, Lw, time.time() - t0)


def selfcheck():
    import gmpy2
    for sp in (30, 882, 1050, 10008):
        jlo, mn, pl = sieve_stretch(sp)
        sq = sp * sp
        for i in range(len(mn)):
            jj = jlo + i
            assert bool(mn[i]) == bool(gmpy2.is_prime(sq + 6 * jj - 1)), (sp, jj)
            assert bool(pl[i]) == bool(gmpy2.is_prime(sq + 6 * jj + 1)), (sp, jj)
    print("selfcheck ok")


if __name__ == "__main__":
    t0 = time.time()
    _init()
    selfcheck()
    S = twin_centres_upto(SMAX)
    TC = twin_centres_upto((SMAX + 1) ** 2 + 1)
    tasks = []
    for s in S:
        if not (S_LO < s <= S_HI):
            continue
        s = int(s)
        lo = (s - 1) ** 2
        hi = (s + 1) ** 2
        a = np.searchsorted(TC, lo, side="right")
        b = np.searchsorted(TC, hi, side="left")
        for sp in TC[a:b]:
            sp = int(sp)
            j = (sp - s * s) // 6
            assert s * s + 6 * j == sp
            tasks.append((s, j, sp, sp))
    print(f"parents {len(S)} in range, rungs {len(tasks)}, prep {time.time()-t0:.1f}s", flush=True)
    out = os.path.join(HERE, f"rungs_{S_LO}_{S_HI}.csv")
    with Pool(2, initializer=_init) as pool, open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "j", "sp", "T", "j_nearest", "j_random", "L", "L_win"])
        done = 0
        for r in pool.imap_unordered(work, tasks, chunksize=4):
            w.writerow(r[:8])
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(tasks)} elapsed {time.time()-t0:.0f}s last {r[8]:.2f}s sp={r[2]}", flush=True)
    print(f"done {done} rungs in {time.time()-t0:.0f}s -> {out}")
