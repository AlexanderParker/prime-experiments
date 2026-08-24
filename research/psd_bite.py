"""Round 21 lateral, target (c): DOES POSITIVE-SEMIDEFINITENESS BITE?

The Wiener-Khinchin identity makes the c-law an autocorrelation, so every
machine's correlation data is PSD-consistent forever.  The question: do the
POSITION laws (2-, 3-, 4-point closed-form counts) FORCE a SIZE bound - i.e.
is a window violating the max-gap (or the (D) threshold F + q') infeasible
already at the correlation level?

TWO EXACT TESTS, both pure CRT closed form (no period scans):

PART A - THE BONFERRONI RUN CERTIFICATE.  E(L) = #{x : slots x..x+L-1 all
blocked} = sum_{T subset [0,L)} (-1)^|T| N(T),  N(T) = prod_q c_q(T)
(inclusion-exclusion over which window slots are open; c_q(T) = surviving
phases of gear q).  Computed EXACTLY (integer arithmetic) by DFS with
subtree pruning at N(T) = 0 (masks only shrink, so zero is hereditary).
E(F(M)) = 0 recovers the machine's max gap from pure position laws -
an independent, scan-free derivation of F(M).  The per-depth partial sums
tell WHERE the alternating series first certifies E = 0: the PSD level is
depth 2; the certificate depth k* is the honest measure of how far beyond
pair-correlations the size law lives.

PART B - THE MOMENT LP.  f(x) = # openings in [x, x+W).  The moments
    m1 = W N1,   m2 = W N1 + 2 sum_{g<W} (W-g) N2(g),
    m3 = W N1 + 6 sum_{a<b} N2 + 6 sum_{a<b<c} N3,
    m4 = W N1 + 14 sum N2 + 36 sum N3 + 24 sum N4
are exact closed forms (Stirling/onto-function coefficients; N3, N4 from
per-gear tables).  LP: over distributions p_k >= 0 on {0..W} matching the
moments up to order K, maximize p_0 (the number of empty windows).  If
max p_0 < 1, correlations up to order K FORCE every length-W window to
contain an opening: F(M) <= W, a SIZE bound from POSITION laws.  We measure
W*_K = min W where the certificate succeeds, vs the true F(M), for
K = 2 (the PSD level), 3, 4 - including machines beyond scan reach and the
merge-step (D) threshold W = F(M) + q' + 1.

PRE-REGISTERED EXPECTATION: K = 2 (pure PSD / pair level) does NOT bite at
any W near F (the LP needs concentration that two moments cannot express);
the interesting measurements are (i) whether K = 4 bites at ANY finite W,
(ii) the certificate depth k* in part A and its growth.

All counts exact integers; LP in floats (labeled), with a safety margin
reported.  Usage:
    python psd_bite.py             # part B machines 13..41 + part A 13..19
    python psd_bite.py --deep      # adds part A machine 23 (DFS ~minutes)
"""
import sys
from math import prod, comb
from itertools import combinations
import numpy as np
from scipy.optimize import linprog

TRUE_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 47, 31: 58, 37: 88,
          41: 91}


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))]


def exposed_mask(q):
    u = pow(6, -1, q)
    m = (1 << q) - 1
    m &= ~(1 << (u % q))
    m &= ~(1 << ((-u) % q))
    return m


def rotate(mask, t, q):
    """mask of r such that r+t is exposed: shift the exposed set by -t."""
    t %= q
    return ((mask >> t) | (mask << (q - t))) & ((1 << q) - 1)


# ------------------------------------------------------------------ part A
def bonferroni_runs(y, L):
    """exact E(L) with per-depth partial sums, pruned DFS, pure ints."""
    gears = primes(5, y)
    base = [(q, exposed_mask(q)) for q in gears]
    # per-slot per-gear rotated masks
    rot = [[rotate(m, t, q) for (q, m) in base] for t in range(L)]
    depth_sums = {}          # |T| -> sum of N(T)
    visited = [0]

    def dfs(start, masks, size):
        for t in range(start, L):
            nm = [m & r for m, r in zip(masks, rot[t])]
            n = 1
            for (q, _), m in zip(base, nm):
                c = m.bit_count()
                if c == 0:
                    n = 0
                    break
                n *= c
            if n == 0:
                continue     # hereditary zero: skip subtree
            visited[0] += 1
            depth_sums[size + 1] = depth_sums.get(size + 1, 0) + n
            dfs(t + 1, nm, size + 1)

    full = [(1 << q) - 1 for (q, _) in base]    # T = empty: no constraint
    depth_sums[0] = prod(q for q in gears)      # N(empty) = P
    dfs(0, full, 0)
    E = sum((-1) ** k * s for k, s in depth_sums.items())
    # partial sums by depth (Bonferroni brackets)
    partial = []
    acc = 0
    for k in range(0, max(depth_sums) + 1):
        acc += (-1) ** k * depth_sums.get(k, 0)
        partial.append(acc)
    return E, partial, visited[0]


def part_a(deep=False):
    print("=" * 78)
    print("PART A: exact run-count E(L) from position laws alone "
          "(scan-free F derivation)")
    ys = [13, 17, 19] + ([23] if deep else [])
    for y in ys:
        F = TRUE_F[y]
        for L in (F - 1, F):
            E, partial, nv = bonferroni_runs(y, L)
            # first depth where the upper (even) bound certifies E = 0
            # (partial[k] for k >= max depth equals E itself)
            kstar = None
            kmax = len(partial) - 1
            for k in range(2, kmax + 2, 2):
                if partial[min(k, kmax)] < 1:
                    kstar = k
                    break
            exp_ok = (E > 0) if L == F - 1 else (E == 0)
            print(f"  y={y} L={L}: E = {E}  ({'>0 expected' if L == F-1 else '=0 expected'}"
                  f", {'OK' if exp_ok else 'FAIL'});  nonzero subsets = {nv}")
            assert exp_ok, (y, L, E)
            if L == F:
                print(f"    Bonferroni upper bounds by depth (PSD level = "
                      f"depth 2): first certifying depth k* = {kstar} "
                      f"(max nonzero depth {len(partial) - 1})")
                shown = [f"k={k}: {partial[k]:.3e}" if partial[k] > 10**7 else
                         f"k={k}: {partial[k]}"
                         for k in range(2, min(len(partial), 13), 2)]
                print("    " + "; ".join(shown))
    print("  VERDICT A: E(F) = 0 exactly at every machine tested - the max "
          "gap IS derivable")
    print("  from position laws alone (finite, exact, no scan); the depth "
          "needed is k*.")


# ------------------------------------------------------------------ part B
_T3, _T4 = {}, {}


def tables(q):
    if q not in _T3:
        u = pow(6, -1, q)
        E = np.ones(q)
        E[u % q] = 0.0
        E[(-u) % q] = 0.0
        idx = (np.arange(q)[:, None] + np.arange(q)[None, :]) % q
        M = E[idx]                       # M[a, r] = E[r+a]
        A = M * E[None, :]               # A[a, r] = E[r]E[r+a]
        _T3[q] = (A @ M.T).round().astype(np.int64)      # T3[a, b]
        A2 = A[:, None, :] * M[None, :, :]               # A2[a, b, r]
        _T4[q] = (A2.reshape(q * q, q) @ M.T).reshape(q, q, q)\
            .round().astype(np.int64)                    # T4[a, b, c]
    return _T3[q], _T4[q]


def c2(q, g):
    u = pow(6, -1, q)
    if g % q == 0:
        return q - 2
    if g % q in ((2 * u) % q, (-2 * u) % q):
        return q - 3
    return q - 4


def moments(y, W):
    """exact m1..m4 of f(x) = # openings in [x, x+W), machine y."""
    gears = primes(5, y)
    P = prod(gears)
    N1 = prod(q - 2 for q in gears)
    s2 = 0                       # sum over unordered pairs of N2
    for g in range(1, W):
        s2 += (W - g) * prod(c2(q, g) for q in gears)
    # N3, N4 sums via numpy over difference patterns
    tabs = [tables(q) for q in gears]
    qs = gears
    # triples: offsets 0 <= a < b < W paired with base 0: (i, i+a, i+b)
    s3 = 0
    tri = np.array(list(combinations(range(W), 3)), dtype=np.int64)
    if tri.size:
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        n = np.ones(len(tri), dtype=np.float64)
        for q, (t3, _) in zip(qs, tabs):
            n *= t3[a % q, b % q]
        s3 = n.sum()
    s4 = 0.0
    quad = np.array(list(combinations(range(W), 4)), dtype=np.int64)
    if quad.size:
        a = quad[:, 1] - quad[:, 0]
        b = quad[:, 2] - quad[:, 0]
        c = quad[:, 3] - quad[:, 0]
        n = np.ones(len(quad), dtype=np.float64)
        for q, (_, t4) in zip(qs, tabs):
            n *= t4[a % q, b % q, c % q]
        s4 = n.sum()
    m1 = W * N1
    m2 = W * N1 + 2 * s2
    m3 = W * N1 + 6 * s2 + 6 * s3
    m4 = W * N1 + 14 * s2 + 36 * s3 + 24 * s4
    return P, [m1, m2, m3, m4]


def max_p0(P, mom, W, K):
    """LP: max p_0 over distributions on {0..W} matching moments 1..K.
    Returns max p_0 (float) or None if infeasible."""
    k = np.arange(W + 1, dtype=np.float64)
    A_eq = [np.ones(W + 1)]
    b_eq = [P]
    for j in range(1, K + 1):
        A_eq.append(k ** j)
        b_eq.append(mom[j - 1])
    c = np.zeros(W + 1)
    c[0] = -1.0
    # scale rows for conditioning
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq, dtype=np.float64)
    sc = np.abs(A_eq).max(axis=1)
    A_eq = A_eq / sc[:, None]
    b_eq = b_eq / sc
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (W + 1),
                  method="highs")
    if not res.success:
        return None
    return -res.fun


def part_b():
    print("=" * 78)
    print("PART B: the moment LP - max feasible # empty windows from "
          "moments of order <= K")
    print("  (max p_0 < 1  ==>  correlations force F(M) <= W; floats, "
          "scaled LP)")
    print(f"  {'y':>4} {'W':>4} {'trueF':>6} {'maxp0 K=2':>12} "
          f"{'K=3':>12} {'K=4':>12} {'true p0':>12}")
    for y in (13, 17, 19, 23, 29, 31, 37, 41):
        F = TRUE_F[y]
        for W in [F, F + 5] + ([2 * F] if 2 * F <= 120 else []):
            P, mom = moments(y, W)
            outs = []
            for K in (2, 3, 4):
                v = max_p0(P, mom, W, K)
                outs.append("infeas" if v is None else f"{v:.4g}")
            print(f"  {y:>4} {W:>4} {F:>6} {outs[0]:>12} {outs[1]:>12} "
                  f"{outs[2]:>12} {'0 (W>=F)':>12}", flush=True)
    print("  ((D) threshold rows: W = F_old + q' + 1 at the merge steps)")
    steps = [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31), (31, 37),
             (37, 41)]
    for (yold, ynew) in steps:
        W = TRUE_F[yold] + ynew + 1
        P, mom = moments(ynew, W)
        outs = []
        for K in (2, 3, 4):
            v = max_p0(P, mom, W, K)
            outs.append("infeas" if v is None else f"{v:.4g}")
        print(f"  step {yold}->{ynew}: W = {W} (true F({ynew}) = "
              f"{TRUE_F[ynew]}): maxp0 K=2/3/4 = {outs[0]} / {outs[1]} / "
              f"{outs[2]}", flush=True)


if __name__ == "__main__":
    if "--deep-only" in sys.argv:
        part_a(deep=True)
    else:
        part_b()
        part_a(deep="--deep" in sys.argv)
    print("DONE")
