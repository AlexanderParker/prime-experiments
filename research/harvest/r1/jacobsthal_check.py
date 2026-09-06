"""Three covering objects, ordered.  Harvester r1, correction of law-register verdict 1.

(a) the REAL twin-candidate max gap at p_n# for n = 3..9, by direct sieve, against 6 F(M);
(b) F(M) for {5..23} recomputed by a column sieve (cross-check of the documents' ladder);
(c) the FREE two-residue covering record for the initial segments {5}..{5..17}, against
    OEIS A072753 = 2, 4, 10, 24, 31;
(d) the PROJECT adversary A(K) of docs/proofs/20 (two classes at separation 3^{-1} mod g,
    free phase, primes free) for K = 1..5.

Run: uv run python research/harvest/r1/jacobsthal_check.py
"""

import sys
from sympy import primerange

import numpy as np

OUT = "research/harvest/r1/results"

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def log(*a):
    print(*a)
    sys.stdout.flush()


# ---------------------------------------------------------------- (a) real gaps at primorials


def real_primorial_gap(n):
    """max cyclic gap of {k : gcd(k(k+2), p_n#) = 1} mod p_n#."""
    ps = PRIMES[:n]
    N = 1
    for p in ps:
        N *= p
    struck = np.zeros(N, dtype=bool)
    for p in ps:
        struck[0::p] = True                      # k = 0 mod p
        start = (p - 2) % p                      # k = -2 mod p
        struck[start::p] = True
    idx = np.flatnonzero(~struck)
    assert idx.size > 0
    d = np.diff(idx)
    wrap = idx[0] + N - idx[-1]
    return N, int(idx.size), int(max(d.max() if d.size else 0, wrap)), idx[:8].tolist()


# ---------------------------------------------------------------- (b) F(M) by a column sieve


def u_g(g):
    return pow(6, -1, g)


def F_columns(gears):
    """widest gap between consecutive open columns over the full period prod(gears)."""
    N = 1
    for g in gears:
        N *= g
    struck = np.zeros(N, dtype=bool)
    for g in gears:
        u = u_g(g)
        struck[u % g:: g] = True
        struck[(-u) % g:: g] = True
    idx = np.flatnonzero(~struck)
    d = np.diff(idx)
    wrap = idx[0] + N - idx[-1]
    return int(max(d.max() if d.size else 0, wrap))


# ---------------------------------------------------------------- covering searches


def masks_free(g, L):
    """all strike masks in [0,L) of a gear g with TWO FREE residue classes mod g."""
    base = [0] * g
    for r in range(g):
        m = 0
        for k in range(r, L, g):
            m |= 1 << k
        base[r] = m
    out = []
    for a in range(g):
        for b in range(a, g):
            out.append((base[a] | base[b], (a, b)))
    return out


def masks_fixed(g, L):
    """all strike masks in [0,L) of a gear g with teeth at +-u_g mod g (free phase)."""
    d = (2 * u_g(g)) % g                        # 3^{-1} mod g
    base = [0] * g
    for r in range(g):
        m = 0
        for k in range(r, L, g):
            m |= 1 << k
        base[r] = m
    out = []
    for c in range(g):
        out.append((base[c] | base[(c + d) % g], (c, (c + d) % g)))
    return out


def cap_free(g, L):
    q, r = divmod(L, g)
    return 2 * q + (2 if r >= 2 else r)


def cap_fixed(g, L):
    d = (2 * u_g(g)) % g
    a = min(d, g - d)
    q, r = divmod(L, g)
    return 2 * q + (2 if r > a else (1 if r >= 1 else 0))


def cover_exists(L, pool, K, wildcard=False):
    """can K gears drawn from `pool` (list of (g, masks, cap)) cover [0, L)?
    pool entries are distinct primes, each usable once.  wildcard = an extra,
    unlimited supply of gears covering exactly one arbitrary column (Lemma 4)."""
    full = (1 << L) - 1
    order = sorted(range(len(pool)), key=lambda i: -pool[i][2])

    def rec(cov, used, budget):
        if cov == full:
            return True
        if budget == 0:
            return False
        holes = L - bin(cov).count("1")
        # capacity prune over the best `budget` unused gears
        caps = []
        for i in order:
            if not (used >> i) & 1:
                caps.append(pool[i][2])
                if len(caps) == budget:
                    break
        if wildcard:
            caps = (caps + [1] * budget)[:budget]
        if sum(caps) < holes:
            return False
        c = (~cov & full)
        c = (c & -c).bit_length() - 1            # leftmost uncovered column
        for i in range(len(pool)):
            if (used >> i) & 1:
                continue
            g, ms, _ = pool[i]
            seen = set()
            for m, _tag in ms:
                if not (m >> c) & 1:
                    continue
                nm = cov | m
                if nm in seen:
                    continue
                seen.add(nm)
                if rec(nm, used | (1 << i), budget - 1):
                    return True
        if wildcard and rec(cov | (1 << c), used, budget - 1):
            return True
        return False

    return rec(0, 0, K)


def free_record(gears):
    """largest L that the given gear set covers with two free classes each."""
    L = 1
    while True:
        pool = [(g, masks_free(g, L), cap_free(g, L)) for g in gears]
        if not cover_exists(L, pool, len(gears)):
            return L - 1
        L += 1


def project_A(K):
    """A(K) of docs/proofs/20: least L that no K primes (free choice, fixed separation,
    free phase) can cover.  Returns (A, longest coverable = A - 1)."""
    L = 1
    while True:
        pool = []
        for g in primerange(5, 3 * L + 3):
            pool.append((g, masks_fixed(g, L), cap_fixed(g, L)))
        if not cover_exists(L, pool, K, wildcard=True):
            return L, L - 1
        L += 1


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    doc_F = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}

    log("== (b) F(M) by a column sieve, against the documents' ladder ==")
    Fs = {}
    for i in range(2, 9):                        # gear sets {5..M}, M = 5..23
        gears = PRIMES[2:i + 1]
        M = gears[-1]
        Fs[M] = F_columns(gears)
        log(f"  {{5..{M:2d}}}  F = {Fs[M]:3d}   documents: {doc_F[M]:3d}   "
            f"{'OK' if Fs[M] == doc_F[M] else 'MISMATCH'}")

    log("")
    log("== (a) real twin-candidate max gap at p_n#, against 6 F(M) ==")
    for n in range(3, 10):
        N, nopen, gap, first = real_primorial_gap(n)
        M = PRIMES[n - 1]
        six = 6 * Fs[M]
        log(f"  n={n}  p_n={M:2d}  p_n# = {N:>11d}  open = {nopen:>8d}  "
            f"max gap = {gap:>4d}   6 F({M}) = {six:>4d}   "
            f"{'OK' if gap == six else 'MISMATCH'}")

    log("")
    log("== (c) FREE two-residue covering record, initial segments, vs A072753 ==")
    a072753 = [2, 4, 10, 24, 31]
    free = []
    for K in range(1, 6):
        gears = PRIMES[2:2 + K]
        r = free_record(gears)
        free.append(r)
        log(f"  K={K}  gears {gears}  free record = {r:3d}   "
            f"A072753 = {a072753[K - 1]:3d}   "
            f"{'OK' if r == a072753[K - 1] else 'MISMATCH'}")

    log("")
    log("== (d) the project's adversary A(K) (docs/proofs/20) ==")
    doc_A = [2, 5, 7, 16, 22]
    adv = []
    for K in range(1, 6):
        A, cov = project_A(K)
        adv.append(cov)
        log(f"  K={K}  A(K) = {A:3d}  longest covered = {cov:3d}   "
            f"doc 20: A = {doc_A[K - 1]:3d}   "
            f"{'OK' if A == doc_A[K - 1] else 'MISMATCH'}")

    log("")
    log("== the three objects, as longest coverable runs of columns ==")
    log("   K   gears {5..p}   real F(M)-1   project A(K)-1   free two-class")
    for K in range(1, 6):
        M = PRIMES[2 + K - 1]
        log(f"  {K}   {{5..{M:2d}}}        {Fs[M] - 1:5d}          {adv[K - 1]:5d}"
            f"           {free[K - 1]:5d}")
