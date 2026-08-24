"""Round 22 lateral, target (a) - THE NON-TENSOR SECTOR AS LINEAR ALGEBRA.

WHERE THIS COMES FROM.  Round 21 (eigenvalue-statistics.md) showed the
machine's tensor operators are Poisson BY CONSTRUCTION, so anything the
project still wants - a GUE-bearing operator, and (D) - has to live in the
part of the algebra that does NOT factor over gears:

    E_M = (x)_q E_q   IS a Kronecker product (exposure factorises);
    B   = I - (x)_q E_q  is NOT (blocking is the COMPLEMENT of a product).

That B is the same object whose nilpotency index is F(M) (matrix-formulation
piece 2/4), and it is Wall V in operator form.  This file measures HOW BIG
that non-factoring part is, exactly, and asks the round's spine question:
DOES IT GROW WITH THE MACHINE?

THE RIGHT DIMENSION.  For a vector/operator on (x)_q V_q and a bipartition
of the gears into G1 | G2, the exact measure of "how far from a product" is
the SCHMIDT RANK (operator Schmidt rank / Kronecker rank): reshape the
object as a d1 x d2 matrix (d1 = prod G1, d2 = prod G2 - CRT makes this a
genuine reshape) and take the matrix rank.  Rank 1 = a product; rank r = a
sum of r products and no fewer.  Two facts make this the honest measure:
  * max over bipartitions of the Schmidt rank is a CERTIFIED LOWER BOUND on
    the tensor rank (fewest rank-1 terms over the full m-fold partition);
  * rank over GF(p) <= rank over Q, so a mod-p rank is itself a certified
    lower bound - growth measured this way cannot be an artifact.

PART 1 - DEPTH 1 (a theorem).  Reshaping b = 1 - (x)_q e_q across ANY cut
gives J - x y^T with x = (x)_{G1} e_q, y = (x)_{G2} e_q and J = 1 1^T.  Both
x and y are non-constant 0/1 vectors (every gear has teeth and exposure), so
{1,x} and {1,y} are independent pairs and

    SCHMIDT RANK OF B IS EXACTLY 2 AT EVERY CUT, EVERY MACHINE.

Likewise BS = (x)S_q - (x)(E_q S_q) has operator Schmidt rank exactly 2.
So the ENTIRE non-tensor sector at depth 1 is ONE rank-one correction.  It
does NOT grow.  The difficulty of F is therefore NOT dimensional.

PART 2 - DEPTH n (the measurement).  What F actually asks about is not b but
the WINDOW indicator

    v_n(k) = prod_{i=0}^{n-1} b(k+i)      (slots k..k+n-1 all blocked),
    F(M) = min { n : v_n == 0 }           (= nilpotency index of BS, since
                                           (BS)^n = diag(v_n) S^n).

Expanding each factor by inclusion-exclusion,
    v_n = sum_{I subset of [n]} (-1)^{|I|} (x)_q ( prod_{i in I} e_q(. + i) ),
a signed sum of at most 2^n rank-1 tensors, so at every cut

    rank_n <= min( 2^n, d1, d2 )   and   rank_1 = 2   and   rank_F = 0.

The profile rank_n between those endpoints is the size of the non-tensor
sector at window depth n.  MEASURED HERE, exactly (integer Gram matrix, rank
over two large primes = certified lower bound on the rational rank).

PART 3 - THE MERGE CUT (a second theorem, and the arity contrast).  Cut off
the TOP gear q': V[r, k] = prod over i with (k+i) OPEN in the old machine of
[ r+i in T_{q'} ].  So the column depends ONLY on the old machine's opening
pattern O in the window - which is the merge law's own statement - and since
|T_{q'}| = 2, for n <= q' the column VANISHES unless |O| <= 2.  Hence

    rank_n(merge cut) = dim span{ 1 (if n < F_old) } + #{singleton classes}
                        + #{literal pairs}   <=  2n + 1,

LINEAR in the window and with a machine-independent FORM.  The merge
direction is fixed-arity; that is exactly why the merge law is a theorem.

Usage: python nontensor.py             # machines 11..19
       python nontensor.py --big       # adds machine 23 (37.2M slots)
"""
import sys
import time
from math import prod
from itertools import combinations
import numpy as np

P1, P2 = 2147483647, 2147483629          # two large primes for mod-p rank


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** .5) + 1))]


def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q


def blocked(gears):
    P = prod(gears)
    b = np.zeros(P, bool)
    for q in gears:
        t1, t2 = teeth(q)
        b[t1::q] = True
        b[t2::q] = True
    return b


def rank_modp(A, p):
    """rank over GF(p) of an integer matrix (certified <= rank over Q)."""
    M = (np.asarray(A, dtype=np.int64) % p).copy()
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = np.flatnonzero(M[r:, c])
        if piv.size == 0:
            continue
        i = r + int(piv[0])
        if i != r:
            M[[r, i]] = M[[i, r]]
        inv = pow(int(M[r, c]), p - 2, p)
        M[r] = (M[r] * inv) % p
        nz = np.flatnonzero(M[:, c])
        nz = nz[nz != r]
        if nz.size:
            M[nz] = (M[nz] - np.outer(M[nz, c], M[r])) % p
        r += 1
    return r


def exact_rank(A):
    """rank of an integer matrix: mod-p at two primes (agree => that is the
    rational rank with overwhelming certainty; each is a certified lower
    bound in any case)."""
    r1 = rank_modp(A, P1)
    r2 = rank_modp(A, P2)
    assert r1 == r2, ("mod-p rank disagreement - rerun with more primes",
                      r1, r2)
    return r1


def gram_rank(v, P, d1, d2):
    """Schmidt rank of the 0/1 vector v (length P = d1*d2, gcd(d1,d2)=1)
    across the CRT cut, via the exact integer Gram matrix V V^T."""
    supp = np.flatnonzero(v)
    if supp.size == 0:
        return 0
    V = np.zeros((d1, d2), np.float32)
    V[supp % d1, supp % d2] = 1.0
    G = V @ V.T                                # exact: counts <= d2 < 2^24
    assert G.max() <= 2 ** 24, "float32 Gram would lose exactness"
    return exact_rank(np.rint(G).astype(np.int64))


def all_cuts(gears, dcap, dmin=1):
    """bipartitions with prod(G1) <= dcap, smaller side first."""
    out = []
    m = len(gears)
    for k in range(1, m):
        for G1 in combinations(gears, k):
            d1 = prod(G1)
            if d1 > dcap or d1 < dmin:
                continue
            G2 = tuple(g for g in gears if g not in G1)
            if d1 < prod(G2):
                out.append((G1, G2))
    return sorted(out, key=lambda c: prod(c[0]))


# ---------------------------------------------------------------- part 1
def part1(ys):
    print("\n=== PART 1: depth-1 Schmidt rank of B = I - (x)E_q ==========")
    print("  theorem: exactly 2 at every cut, every machine (J - x y^T with")
    print("  x, y non-constant).  Asserted below by exact integer rank.")
    for y in ys:
        gears = primes(5, y)
        P = prod(gears)
        b = (~blocked(gears)).astype(np.int64)      # exposure vector e
        bb = 1 - b                                   # blocking vector
        ranks = set()
        for G1, G2 in all_cuts(gears, 10 ** 9):
            d1, d2 = prod(G1), prod(G2)
            r = gram_rank(bb, P, d1, d2)
            ranks.add(r)
            re = gram_rank(b, P, d1, d2)
            assert re == 1, ("exposure must be rank 1 (it IS a product)",
                             y, G1, re)
        assert ranks == {2}, (y, ranks)
        print(f"  machine {y:2d}: {len(all_cuts(gears, 10**9))} cuts - "
              f"rank(exposure) = 1 at all, rank(blocking) = 2 at all  OK")


# ---------------------------------------------------------------- part 2
def part2(ys, dcap, supp_cap, dmin=1):
    print("\n=== PART 2: depth-n Schmidt rank profile of the window "
          "indicator ==")
    print("  rank_n <= min(2^n, d1, d2);  rank_1 = 2;  rank_F = 0.")
    print("  'sat' = the depth from which rank_n equals its cap min(2^n,d1)")
    rows = []
    for y in ys:
        gears = primes(5, y)
        P = prod(gears)
        b = blocked(gears)
        cuts = all_cuts(gears, dcap, dmin if y >= 23 else 1)
        # always include the corridor cut and the merge cut
        names = {}
        for G1, G2 in cuts:
            names[G1] = f"{{{','.join(map(str, G1))}}}"
        v = np.ones(P, bool)
        t0 = time.time()
        prof = {G1: [] for G1, _ in cuts}
        n = 0
        while True:
            n += 1
            v = v & np.roll(b, -(n - 1))
            ns = int(v.sum())
            if ns == 0:
                break
            if ns <= supp_cap:
                for G1, G2 in cuts:
                    prof[G1].append((n, gram_rank(v, P, prod(G1), prod(G2))))
            else:
                for G1, _ in cuts:
                    prof[G1].append((n, None))       # too dense to build
        F = n
        assert F == {11: 7, 13: 11, 17: 18, 19: 25, 23: 34}[y], (y, F)
        print(f"\n  machine {y} (P = {P:,}, F = {F}, {time.time()-t0:.0f}s)")
        print(f"   {'cut':>14} {'d1':>5}  rank_n for n = 1..F-1 "
              f"(. = not built)")
        for G1, G2 in cuts:
            d1 = prod(G1)
            s = "".join(f"{r:4d}" if r is not None else "   ."
                        for _, r in prof[G1])
            print(f"   {names[G1]:>14} {d1:5d}  {s}")
            got = [(n, r) for n, r in prof[G1] if r is not None]
            for n, r in got:
                assert r <= min(2 ** n, d1, P // d1), (y, G1, n, r)
            rows.append(dict(y=y, G1=G1, d1=d1, F=F, prof=got))
    return rows


# ---------------------------------------------------------------- part 3
def merge_cut_theory(gears_old, qp, n):
    """predicted rank at the merge cut, from the old machine's window
    patterns only (part-3 theorem).  Returns (rank, #singletons, #pairs,
    empty_present)."""
    Pold = prod(gears_old)
    bo = blocked(gears_old)
    op = ~bo
    T = set(teeth(qp))
    u = pow(6, -1, qp) % qp
    # realized opening patterns O (as tuples of offsets) in windows of len n
    W = np.zeros(Pold, np.int64)
    pats = set()
    for k in range(Pold):
        pass
    # vectorised: pattern index = bitmask of opens in the window
    if n <= 62:
        code = np.zeros(Pold, np.int64)
        for i in range(n):
            code |= (np.roll(op, -i).astype(np.int64) << i)
        codes = np.unique(code)
    else:
        raise ValueError("n too large for bitmask")
    vecs = []
    nsing = npair = 0
    empty = False
    for c in codes:
        O = [i for i in range(n) if (int(c) >> i) & 1]
        if len(O) == 0:
            empty = True
            vecs.append(np.ones(qp, np.int64))
        elif len(O) == 1:
            i = O[0]
            v = np.zeros(qp, np.int64)
            for t in T:
                v[(t - O[0]) % qp] = 1
            vecs.append(v)
            nsing += 1
        elif len(O) == 2:
            i, j = O
            if (j - i) % qp in ((2 * u) % qp, (-2 * u) % qp):
                # unique r with {r+i, r+j} = T
                for t in T:
                    r = (t - i) % qp
                    if (r + j) % qp in T:
                        v = np.zeros(qp, np.int64)
                        v[r] = 1
                        vecs.append(v)
                        npair += 1
                        break
        else:
            if n <= qp:
                continue                 # theorem: vanishes
            # n > q': offsets can collide mod q'; handle generally
            res = set(i % qp for i in O)
            if len(res) <= 2:
                v = np.zeros(qp, np.int64)
                for r in range(qp):
                    if all(((r + i) % qp) in T for i in O):
                        v[r] = 1
                if v.any():
                    vecs.append(v)
    if not vecs:
        return 0, nsing, npair, empty
    A = np.array(vecs, np.int64)
    return exact_rank(A), nsing, npair, empty


def part3(ys, nmax):
    print("\n=== PART 3: the merge cut - exact structural rank ============")
    print("  columns depend only on the OLD machine's opening pattern O in")
    print("  the window; for n <= q' they vanish unless |O| <= 2.")
    print("   y  q'   n   rank(measured)  rank(theory)  #singleton  #pair"
          "  empty")
    for y in ys:
        gears = primes(5, y)
        qp = gears[-1]
        gold = gears[:-1]
        P = prod(gears)
        b = blocked(gears)
        v = np.ones(P, bool)
        for n in range(1, min(nmax, qp) + 1):
            v = v & np.roll(b, -(n - 1))
            if not v.any():
                break
            meas = gram_rank(v, P, qp, P // qp)
            th, ns, npr, emp = merge_cut_theory(gold, qp, n)
            assert meas == th, (y, n, meas, th)
            assert th <= 2 * n + 1, (y, n, th)
            print(f"  {y:3d} {qp:3d} {n:3d}   {meas:8d}      {th:8d}"
                  f"      {ns:6d}   {npr:5d}   {str(emp):>5}")
    print("  ASSERTED: measured == theory at every row, and rank <= 2n+1.")


# ---------------------------------------------------------------- verdict
def verdict(rows):
    print("\n=== VERDICT: does the non-tensor sector grow? ================")
    print("  certified LOWER BOUND on the tensor rank of the deepest window")
    print("  indicator = max over measured cuts of max_n rank_n:")
    print("   machine   TR_low   argmax cut     d1   peak/d1   F")
    bym = {}
    for r in rows:
        if not r["prof"]:
            continue
        best = max(rr for _, rr in r["prof"])
        cur = bym.get(r["y"])
        if cur is None or best > cur[0]:
            bym[r["y"]] = (best, r["G1"], r["d1"], r["F"])
    for y in sorted(bym):
        best, G1, d1, F = bym[y]
        print(f"   {y:5d}   {best:6d}   {str(G1):>14} {d1:5d}   "
              f"{best / d1:6.3f}  {F:3d}")
    print("\n   corridor cut {5,7} (d1 = 35 FIXED across machines):")
    for r in sorted(rows, key=lambda r: r["y"]):
        if r["G1"] == (5, 7) and r["prof"]:
            best = max(rr for _, rr in r["prof"])
            nb = [n for n, rr in r["prof"] if rr == best][0]
            print(f"     machine {r['y']:2d}: peak rank {best:3d} / 35 at "
                  f"depth {nb:2d}  (F = {r['F']})")
    print("  (i) depth 1: rank exactly 2, every cut, every machine - the")
    print("      sector is ONE rank-one correction and does NOT grow.")
    print("  (ii) merge cut: rank <= 2n+1 - LINEAR in the window, form")
    print("      machine-independent.  Fixed-arity, and that IS the merge law.")
    print("  (iii) general cuts, deep windows:")
    print("   machine   cut        d1   max_n rank_n   peak/d1   "
          "FULL (rank == d1)?")
    for r in rows:
        if not r["prof"]:
            continue
        best = max(rr for _, rr in r["prof"])
        full = best == r["d1"]
        print(f"   {r['y']:5d}   {str(r['G1']):>16} {r['d1']:5d}"
              f"   {best:6d}        {best / r['d1']:6.3f}        "
              f"{str(full):>5}")


def main():
    big = "--big" in sys.argv
    ys = [11, 13, 17, 19] + ([23] if big else [])
    print(__doc__.split("Usage:")[0])
    part1([11, 13, 17])
    rows = part2(ys, dcap=400, supp_cap=6_000_000 if big else 2_000_000,
                 dmin=35)
    part3(ys, nmax=14)
    verdict(rows)


if __name__ == "__main__":
    main()
