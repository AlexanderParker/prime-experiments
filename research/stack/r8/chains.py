"""Claim 3: chain memory. Canonical chain from 6 (nearest rung, negative j on ties) and 200 random
chains from twin centres s0 in [1000, 3000]."""
import os, math, time
import numpy as np
import gmpy2
from gmpy2 import mpz
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SP = [int(p) for p in np.nonzero(np.array([gmpy2.is_prime(i) for i in range(100001)]))[0] if p >= 5]
SP_ARR = np.array(SP, dtype=np.int64)
INV6 = np.where(SP_ARR % 6 == 1, (5 * SP_ARR + 1) // 6, (SP_ARR + 1) // 6)


def is_tc(m):
    return gmpy2.is_prime(m - 1) and gmpy2.is_prime(m + 1)


def nearest_rung(s):
    """offset j of the twin centre s^2 + 6j nearest to s^2; negative on ties."""
    s = mpz(s)
    sq = s * s
    if sq < 10 ** 12:
        j = 1
        while True:
            if is_tc(sq - 6 * j): return -j
            if is_tc(sq + 6 * j): return j
            j += 1
    lnx = float(gmpy2.log(sq))
    J = int(10 * lnx * lnx / 2.64) + 100
    while True:
        W = 2 * J + 1
        jlo = -J
        sqm = np.array([int(sq % int(p)) for p in SP], dtype=np.int64)
        a = np.ones(W, dtype=bool)
        for r0 in (sqm - 1, sqm + 1):
            first = (((-r0) % SP_ARR) * INV6 % SP_ARR - jlo) % SP_ARR
            for i in range(len(SP)):
                a[int(first[i])::int(SP[i])] = False
        idx = np.nonzero(a)[0] + jlo
        idx = idx[idx != 0]
        order = np.lexsort((idx, np.abs(idx)))
        for j in idx[order]:
            j = int(j)
            if is_tc(sq + 6 * j):
                return j
        J *= 2


def chi_tables(pairs, label):
    a = np.array(pairs, dtype=object)
    x = np.array([int(p[0]) for p in pairs]); y = np.array([int(p[1]) for p in pairs])
    sgn = np.zeros((2, 2), dtype=int)
    for u, v in zip(x, y):
        sgn[int(u > 0), int(v > 0)] += 1
    res = np.zeros((5, 5), dtype=int)
    for u, v in zip(x, y):
        res[u % 5, v % 5] += 1
    c1, p1, _, _ = stats.chi2_contingency(sgn)
    c2, p2, _, _ = stats.chi2_contingency(res)
    print(f"{label}: n {len(pairs)}")
    print(f"  sign table [[--, -+], [+-, ++]] = {sgn.tolist()}  chi2 {c1:.3f} p {p1:.3f}")
    print(f"  residue mod 5 table (rows first offset, cols next):")
    for r in range(5):
        print("    ", res[r].tolist())
    print(f"  chi2 {c2:.3f} df 16 p {p2:.3f}")
    # marginals
    print(f"  marginal sign of first: - {np.sum(x<0)} + {np.sum(x>0)};  second: - {np.sum(y<0)} + {np.sum(y>0)}")
    print(f"  marginal mod 5 of first: {np.bincount(x%5, minlength=5).tolist()}  second: {np.bincount(y%5, minlength=5).tolist()}")
    n = len(pairs)
    print(f"  predicted marginal [1/9,2/9,1/9,3/9,2/9]*n = {[round(n*k/9,1) for k in (1,2,1,3,2)]}")
    # conditional test: block A rows {0,1} x cols {0,2,3}; block B rows {2,3,4} x cols {1,3,4}
    A = res[np.ix_([0, 1], [0, 2, 3])]
    B = res[np.ix_([2, 3, 4], [1, 3, 4])]
    off = res.sum() - A.sum() - B.sum()
    ca, pa, _, _ = stats.chi2_contingency(A)
    cb, pb, _, _ = stats.chi2_contingency(B)
    print(f"  conditional on child type: block rows{{0,1}}xcols{{0,2,3}} chi2 {ca:.3f} df 2 p {pa:.3f}; "
          f"block rows{{2,3,4}}xcols{{1,3,4}} chi2 {cb:.3f} df 4 p {pb:.3f}; entries outside both blocks: {off}")
    return p1, p2


if __name__ == "__main__":
    t0 = time.time()
    print("=== canonical chain 6 -> ... (nearest rung, negative on ties) ===")
    s = mpz(6)
    chain = [s]; offs = []
    for step in range(9):
        j = nearest_rung(s)
        offs.append(j)
        s = s * s + 6 * j
        chain.append(s)
        print(f"  node {step}: s digits {len(str(chain[step]))}, j = {j:+d}, j mod 5 = {j%5}, |j| vs expected ln^2(s^2)/2.64 = {abs(j)} vs {float(gmpy2.log(chain[step]*chain[step]))**2/2.64:.0f}  ({time.time()-t0:.0f}s)", flush=True)
    print("  chain:", [str(c) if len(str(c)) < 40 else f"{str(c)[:12]}...({len(str(c))} digits)" for c in chain])
    print("  offsets:", offs)
    print("  consecutive sign pairs:", [(int(a > 0), int(b > 0)) for a, b in zip(offs[:-1], offs[1:])])
    print("  consecutive mod-5 pairs:", [(a % 5, b % 5) for a, b in zip(offs[:-1], offs[1:])])

    print("\n=== 200 random chains from s0 in [1000, 3000] ===")
    D = np.genfromtxt(os.path.join(HERE, "rungs_0_3000.csv"), delimiter=",", names=True)
    rows = {}
    for r in D:
        rows.setdefault(int(r["s"]), []).append(r)
    parents = sorted(p for p in rows if 1000 <= p <= 3000)
    rng = np.random.default_rng(12345)
    pairs_rand = []   # (chosen random rung offset at node, nearest offset of the child)
    pairs_near = []   # (nearest offset at node, nearest offset of the child)
    for ci in range(200):
        s0 = parents[rng.integers(len(parents))]
        r = rows[s0][rng.integers(len(rows[s0]))]
        j0 = int(r["j"]); s1 = int(r["sp"]); jn1 = int(r["j_nearest"]); j1 = int(r["j_random"])
        s2 = mpz(s1) * s1 + 6 * j1
        assert is_tc(s2)
        jn2 = nearest_rung(s2)
        s3 = s2 * s2 + 6 * jn2
        jn3 = nearest_rung(s3)
        s4 = s3 * s3 + 6 * jn3
        jn4 = nearest_rung(s4)
        pairs_rand += [(j0, jn1), (j1, jn2)]
        pairs_near += [(jn2, jn3), (jn3, jn4)]
        if ci % 50 == 49:
            print(f"  chain {ci+1}: s0 {s0} j0 {j0:+d} jn1 {jn1:+d} j1 {j1:+d} jn2 {jn2:+d} jn3 {jn3:+d} jn4 {jn4:+d}  ({time.time()-t0:.0f}s)", flush=True)
    chi_tables(pairs_rand + pairs_near, "ALL pairs (node offset, child's nearest offset)")
    chi_tables(pairs_rand, "pairs with a RANDOM chosen rung at the node (s0->s1, s1->s2)")
    chi_tables(pairs_near, "pairs with the NEAREST rung at the node (s2->s3, s3->s4)")
    print(f"total {time.time()-t0:.0f}s")
