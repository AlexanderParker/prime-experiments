"""Round 23 lateral - WHAT REPLACES SPECTRUM IN A NILPOTENT SECTOR (part 1).

THE SETUP (round 22's finding, which is why this file exists).  The machine's
growth lives in the NILPOTENT direction: with S the slot shift and B =
diag(blocked), N := BS satisfies N^n = diag(v_n) S^n, v_n(k) = prod_{i=1..n}
b(k+i), and N^F = 0 with F = F(M) the maximal gap.  Spectrum {0}: no
eigenvalues, no spectral radius signal, no bounded-order signature.  So:
WHEN THE SPECTRUM IS EMPTY, WHAT CARRIES THE INFORMATION?

THE ANSWER, IN ONE THEOREM AND ONE NEGATIVE.

THEOREM (JORDAN = GAP HISTOGRAM).  N is PERMUTATION-similar (hence unitarily
equivalent) to the direct sum over the machine's gaps of nilpotent Jordan
blocks:
        N  ~=  (+)_g  J_g^{(+) W_1(g)},
one block of size g for each gap of g slots.  Proof: N e_k = b(k+1) e_{k+1},
so the directed graph of N is the disjoint union of the chains of consecutive
blocked slots; between openings at m and m+g the chain is m -> m+1 -> ... ->
m+g-1 -> (dead), a single Jordan block of size g.  Equivalently, by counting,
    rank(N^n) = sum_g W_1(g) (g-n)_+           [the gap-histogram TAIL SUM]
    #Jordan blocks of size exactly L = W_1(L),  largest block = F.
CONSEQUENCE (the negative): EVERY UNITARY INVARIANT OF N IS A FUNCTION OF THE
GAP HISTOGRAM ALONE - singular values, all Schatten norms, the numerical
range, the pseudospectrum, the resolvent norms, the Jordan type, the kernel
filtration.  Nothing in the operator's unitary-invariant world can carry
information the histogram does not already carry, so no such invariant can
bound F non-circularly.  This is the sharpest form yet of why the spectral
frame stalls here, and it is a theorem rather than a failed attempt.
It also UPGRADES round 22's path-decomposition theorem: the Hermitian
A = N + N^T is the union of PATH graphs P_g, which is exactly the symmetrised
shadow of this Jordan decomposition (blocks <-> paths, same index set).

WHAT SURVIVES, AND WHAT IT BUYS.  Three of the invariants are still worth
having because they turn F into an ANALYTIC or VARIATIONAL quantity:

  (1) THE NORM CLIFF (part 2).  N^n is a PARTIAL ISOMETRY: its singular
      values are 0/1 with exactly rank(N^n) ones.  So ||N^n||_op = 1 for
      every n < F and 0 for n >= F - a step function.  There is NO decay rate
      to estimate; Gelfand's formula only sees F at the discontinuity.
      Corollary with teeth: any envelope ||N^n|| <= C lambda^n with lambda<1
      forces C >= lambda^{1-F}, i.e. THE WHOLE OF F SITS IN THE CONSTANT.
      That is precisely why every analytic/decay-rate frame has stalled.

  (2) THE NUMERICAL RADIUS (part 3).  w(N) = cos(pi/(F+1)) EXACTLY, and the
      numerical RANGE is the disk of that radius.  So
          F = pi / arccos( w(N) ) - 1,
      and w(N) = max over unit x of |<Nx,x>| is a VARIATIONAL quantity: the
      maximal gap is the optimum of a concave maximisation, hence
      SDP-representable, hence has a dual certificate for every upper bound.

  (3) THE PSEUDOSPECTRUM (part 4).  The spectrum is {0} but the
      eps-pseudospectral radius is r_eps = eps^{1/F} (1+o(1)); precisely
      F = lim_{eps->0} log(1/eps) / log(1/r_eps).  With eps = e^{-1/t} this
      is a MASLOV DEQUANTISATION statement: t * log ||(zI-N)^{-1}|| -> F, so
      THE (+,x) RESOLVENT COMPUTES THE (max,+) LONGEST PATH.  The project's
      three vehicles for F - Constructor's Kleene star (max,+), the Boolean
      filtration, and the analytic resolvent - are one computation in three
      semirings.

  (4) THE CERTIFICATE (part 5, developed in potential_arity.py).  The one
      frame that is NOT a unitary invariant, and the only one with a
      direction of proof: a POTENTIAL h with h(k) >= h(k-1) + 1 on blocked
      slots gives F <= 1 + osc(h), tightly.  Its multiplicative form
      w = exp(h/t) is exactly a SCHUR TEST on A, and its tropical limit is
      exactly Constructor's max-plus potential inequality.

  (5) THE NON-INVARIANT CONTENT (part 6).  ker N^n is a COORDINATE subspace
      (spanned by the e_k with v_n(k) = 0), so the kernel flag's DIMENSIONS
      are the histogram tail sums - unitary data again - while the flag's
      POSITION relative to the CRT gear basis is not a unitary invariant at
      all.  That position is exactly what round 22 measured as the Schmidt
      rank profile of v_n, and it is the part that GROWS.  So: the invariants
      = the histogram (circular); the growth = the alignment of the kernel
      flag with the gear tensor basis.

  (6) THE MOMENT FRAME REDUCES (part 7).  tr(A^{2t}) = sum_L m_t(L) r_L with
      m_t(L) the number of closed 2t-walks of range L and r_L = rank(N^L) -
      so every trace/moment (equivalently every exponential-sum) attack on
      lambda_max(A) is a positive combination of the r_L ladder that round 21
      already computes exactly and scan-free.  No new information: recorded
      as a NON-GAIN so it is not rebuilt later.

Everything below is asserted.  Exact integers where the claim is exact;
floats are labeled.

Usage: python nilpotent_invariants.py           # machines 11,13,17,19
       python nilpotent_invariants.py --big     # adds machine 23 for part 3
"""
import sys
import time
from math import prod, cos, pi, acos, log

import numpy as np


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


def gap_hist(b):
    idx = np.flatnonzero(~b)
    P = b.size
    g = np.diff(np.append(idx, idx[0] + P))
    assert g.sum() == P
    return np.bincount(g), idx.size


def dense_N(b):
    P = b.size
    N = np.zeros((P, P))
    k = np.arange(P)
    N[(k + 1) % P, k] = b[(k + 1) % P]
    return N


# ------------------------------------------------------------------ part 1
def part1(ys):
    print("=" * 74)
    print("PART 1 - JORDAN STRUCTURE = GAP HISTOGRAM (exact integers)")
    print("=" * 74)
    print("   y      P     F   #blocks  sum sizes  rank(N^n) == tail sum?"
          "  Jordan == W_1?")
    for y in ys:
        gears = primes(5, y)
        b = blocked(gears)
        P = b.size
        W, nopen = gap_hist(b)
        F = int(np.flatnonzero(W)[-1])
        gs = np.arange(W.size)
        # rank(N^n) measured directly: count k with b(k+1..k+n) all blocked
        v = np.ones(P, bool)
        ranks = []
        for n in range(0, F + 2):
            if n > 0:
                v &= b[(np.arange(P) + n) % P]
            ranks.append(int(v.sum()))
        tail = [int(np.dot(W, np.maximum(gs - n, 0))) for n in range(F + 2)]
        ok_rank = ranks == tail
        # Jordan block sizes from the rank sequence
        jb = np.array([ranks[n] - ranks[n + 1] for n in range(F + 1)])
        blocks = np.zeros(F + 2, dtype=np.int64)
        for L in range(1, F + 1):
            blocks[L] = jb[L - 1] - (jb[L] if L < len(jb) else 0)
        ok_jordan = all(int(blocks[L]) == int(W[L] if L < W.size else 0)
                        for L in range(1, F + 1))
        assert ok_rank and ok_jordan, y
        assert int(blocks.sum()) == nopen and int(
            (blocks * np.arange(F + 2)).sum()) == P
        print(f"  {y:3d} {P:9d} {F:5d} {int(blocks.sum()):9d} "
              f"{int((blocks*np.arange(F+2)).sum()):10d}"
              f"      {str(ok_rank):>5s}"
              f"              {str(ok_jordan):>5s}")
    # explicit permutation-similarity check at machine 11 and 13
    for y in (11, 13):
        gears = primes(5, y)
        b = blocked(gears)
        P = b.size
        W, _ = gap_hist(b)
        F = int(np.flatnonzero(W)[-1])
        openidx = np.flatnonzero(~b)
        perm, sizes = [], []
        for i, m in enumerate(openidx):
            g = int((openidx[(i + 1) % len(openidx)] - m) % P) or P
            perm.extend(((m + j) % P) for j in range(g))
            sizes.append(g)
        assert sorted(perm) == list(range(P))
        Np = dense_N(b)[np.ix_(perm, perm)]
        J = np.zeros((P, P))
        o = 0
        for g in sizes:
            for j in range(g - 1):
                J[o + j + 1, o + j] = 1.0
            o += g
        assert np.array_equal(Np, J), y
        cnt = np.bincount(np.array(sizes), minlength=W.size)
        assert np.array_equal(cnt[:W.size], W)
        print(f"  y={y}: N is PERMUTATION-equal to (+)_g J_g^(W_1(g)) - "
              f"exact matrix identity, {len(sizes)} blocks, max {max(sizes)}")
    print()


# ------------------------------------------------------------------ part 2
def part2():
    print("=" * 74)
    print("PART 2 - THE NORM CLIFF: N^n is a partial isometry")
    print("=" * 74)
    y = 11
    gears = primes(5, y)
    b = blocked(gears)
    P = b.size
    W, _ = gap_hist(b)
    F = int(np.flatnonzero(W)[-1])
    N = dense_N(b)
    M = np.eye(P)
    print("   n  rank(N^n)  singular values      ||N^n||_op   ||N^n||_F^2")
    for n in range(1, F + 2):
        M = N @ M
        sv = np.linalg.svd(M, compute_uv=False)
        r = int(round(sv.sum()))
        nz = sv[sv > 1e-9]
        assert np.allclose(nz, 1.0, atol=1e-9), n
        assert abs(np.linalg.norm(M, 'fro') ** 2 - r) < 1e-8
        opn = 0.0 if nz.size == 0 else 1.0
        print(f"  {n:3d} {r:9d}  {'all 0 or 1':18s} {opn:10.1f} "
              f"{np.linalg.norm(M,'fro')**2:13.1f}")
    print(f"  machine 11: F = {F}; ||N^n||_op = 1 for n < F, 0 for n >= F.")
    print("  => Schatten_p norm = rank^(1/p) for every p; no decay rate at")
    print("     all.  Any envelope ||N^n|| <= C lam^n (lam<1) forces")
    print(f"     C >= lam^(1-F) = lam^{1-F}: F sits entirely in the CONSTANT.")
    print()


# ------------------------------------------------------------------ part 3
def perron_weight(b):
    """component-wise Perron vector of A = N + N^T (union of paths P_g)."""
    P = b.size
    openidx = np.flatnonzero(~b)
    w = np.empty(P)
    for i, m in enumerate(openidx):
        g = int((openidx[(i + 1) % len(openidx)] - m) % P) or P
        j = np.arange(1, g + 1)
        w[(m + j - 1) % P] = np.sin(pi * j / (g + 1))
    return w


def part3(ys, big):
    print("=" * 74)
    print("PART 3 - NUMERICAL RADIUS w(N) = cos(pi/(F+1)) EXACTLY")
    print("=" * 74)
    # (a) machine 11: dense check of the numerical RANGE (a disk)
    b = blocked(primes(5, 11))
    N = dense_N(b)
    W, _ = gap_hist(b)
    F = int(np.flatnonzero(W)[-1])
    vals = []
    for th in np.linspace(0, 2 * pi, 24, endpoint=False):
        H = (np.exp(1j * th) * N + np.exp(-1j * th) * N.T) / 2
        vals.append(np.linalg.eigvalsh(H)[-1])
    vals = np.array(vals)
    tgt = cos(pi / (F + 1))
    assert abs(vals.max() - tgt) < 1e-10 and abs(vals.min() - tgt) < 1e-10
    print(f"  m11 (dense, floats): max_x |<Nx,x>| is direction-INDEPENDENT to "
          f"{abs(vals.max()-vals.min()):.2e}")
    print(f"       => numerical range = disk of radius {vals.mean():.12f}, "
          f"cos(pi/(F+1)) = {tgt:.12f}, F = {F}")
    # (b) every machine: two-sided certificate, O(P), no eigensolver
    print()
    print("  Two-sided check at scale (Rayleigh lower bound + SCHUR-TEST upper")
    print("  bound with the exact path Perron weight; floats):")
    print("   y     F   lam_max lower   Schur theta (upper)   2cos(pi/(F+1))"
          "    F from w")
    for y in ys + ([23] if big else []):
        gears = primes(5, y)
        b = blocked(gears)
        P = b.size
        W, _ = gap_hist(b)
        F = int(np.flatnonzero(W)[-1])
        w = perron_weight(b)
        k = np.arange(P)
        # (A w)_k = b(k) w_{k-1} + b(k+1) w_{k+1}
        Aw = b[k] * w[(k - 1) % P] + b[(k + 1) % P] * w[(k + 1) % P]
        theta = float((Aw / w).max())
        lo = float(Aw @ w / (w @ w))
        tgt = 2 * cos(pi / (F + 1))
        assert lo <= tgt + 1e-9 <= theta + 1e-9
        assert abs(theta - tgt) < 1e-9, (y, theta, tgt)
        Fw = pi / acos(min(theta / 2, 1.0)) - 1
        print(f"  {y:3d} {F:5d}  {lo:15.12f}   {theta:19.12f}  "
              f"{tgt:15.12f}  {Fw:9.6f}")
    print("  The Schur test w > 0 with A w <= theta w PROVES lam_max <= theta,")
    print("  hence F <= pi/arccos(theta/2) - 1.  With the exact path Perron")
    print("  weight it is TIGHT - the certificate frame loses nothing.")
    print()


# ------------------------------------------------------------------ part 4
def log_jordan_resolvent_norm(L, z):
    """log ||(zI - J_L)^{-1}||_2 for the nilpotent Jordan block of size L.
    Scaled by z^L to avoid overflow: entries of z^L R are z^(L-n-1) <= 1."""
    R = np.zeros((L, L))
    for n in range(L):
        idx = np.arange(L - n)
        R[idx + n, idx] = z ** (L - n - 1)
    return -L * log(z) + log(float(np.linalg.svd(R, compute_uv=False)[0]))


def jordan_resolvent_norm(L, z):
    from math import exp
    return exp(log_jordan_resolvent_norm(L, z))


def part4(ys):
    print("=" * 74)
    print("PART 4 - PSEUDOSPECTRUM: the empty spectrum still encodes F")
    print("=" * 74)
    # exact resolvent of the real machine at m11, compared with the block form
    b = blocked(primes(5, 11))
    N = dense_N(b)
    P = b.size
    W, _ = gap_hist(b)
    F = int(np.flatnonzero(W)[-1])
    print("  m11: ||(zI-N)^{-1}|| from the full 385x385 resolvent vs the")
    print("  largest-Jordan-block formula (they must agree exactly):")
    print("      z        full resolvent      block F formula     "
          "log||R||/log(1/z)")
    for z in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]:
        R = np.linalg.inv(z * np.eye(P) - N)
        n1 = float(np.linalg.svd(R, compute_uv=False)[0])
        n2 = jordan_resolvent_norm(F, z)
        assert abs(n1 - n2) / n1 < 1e-8, (z, n1, n2)
        print(f"  {z:9.1e}  {n1:18.6e}  {n2:18.6e}  "
              f"{log(n1)/log(1/z):17.6f}")
    print(f"  -> the exponent converges to F = {F} (spectral radius is 0).")
    print()
    print("  eps-PSEUDOSPECTRAL RADIUS r_eps (solve ||R(z)|| = 1/eps), and the")
    print("  recovered F = log(1/eps)/log(1/r_eps)  [floats]:")
    print("   y     F     eps=1e-6  F_hat      eps=1e-12  F_hat      "
          "eps=1e-24  F_hat")
    for y in ys:
        gears = primes(5, y)
        b = blocked(gears)
        W, _ = gap_hist(b)
        F = int(np.flatnonzero(W)[-1])
        out = []
        for eps in [1e-6, 1e-12, 1e-24]:
            lo, hi = eps ** (1.0 / F) * 1e-3, 1.0
            for _ in range(200):
                mid = (lo * hi) ** .5
                if log_jordan_resolvent_norm(F, mid) > log(1 / eps):
                    lo = mid
                else:
                    hi = mid
            r = (lo * hi) ** .5
            out.append((r, log(1 / eps) / log(1 / r)))
        # F_hat decreases monotonically to F from above as eps -> 0
        assert all(f >= F - 1e-9 for _, f in out), y
        assert out[0][1] >= out[1][1] >= out[2][1], y
        assert abs(out[2][1] - F) < 0.05, (y, out[2][1], F)
        print(f"  {y:3d} {F:5d}  " + "  ".join(
            f"{r:10.3e} {f:7.3f}" for r, f in out))
    print("  MASLOV DEQUANTISATION: with z = e^{-1/t}, t*log||(zI-N)^{-1}||")
    print("  -> F.  The (+,x) resolvent computes the (max,+) longest path:")
    print("  the analytic resolvent, Constructor's Kleene star and the Boolean")
    print("  filtration are ONE computation in three semirings.")
    print()


# ------------------------------------------------------------------ part 5
def part5(ys):
    print("=" * 74)
    print("PART 5 - THE POTENTIAL: the one frame with a direction of proof")
    print("=" * 74)
    print("  h(k) >= h(k-1) + 1 on blocked slots  =>  F <= 1 + osc(h).")
    print("  Multiplicative form w = exp(h/t): beta := max over blocked k of")
    print("  w_{k-1}/w_k and kappa := max w / min w give F <= 1 + log kappa /")
    print("  log(1/beta) - a similarity D_w^{-1} N D_w bound, i.e. the same")
    print("  certificate seen as a weighted-shift norm.  Checked exact:")
    print("   y     F   1+osc(h_dist)   t     1+log k/log(1/b)")
    for y in ys:
        gears = primes(5, y)
        b = blocked(gears)
        P = b.size
        W, _ = gap_hist(b)
        F = int(np.flatnonzero(W)[-1])
        d = np.zeros(P, dtype=np.int64)
        prev = -1
        oidx = np.flatnonzero(~b)
        prev = int(oidx[-1]) - P
        for k in range(P):
            if not b[k]:
                prev = k
            d[k] = k - prev
        assert (d[b] == d[(np.flatnonzero(b) - 1) % P] + 1).all()
        assert int(d.max()) == F - 1
        for t in (1.0,):
            w = np.exp(d / t)
            beta = float((w[(np.flatnonzero(b) - 1) % P]
                          / w[b]).max())
            kap = float(w.max() / w.min())
            bnd = 1 + log(kap) / log(1 / beta)
            assert abs(bnd - F) < 1e-9, (y, bnd, F)
            print(f"  {y:3d} {F:5d} {1+int(d.max()-d.min()):13d}  {t:4.1f} "
                  f"{bnd:19.6f}")
    print("  Tight in both forms.  The ARITY of h is the only thing that can")
    print("  fail - measured in research/potential_arity.py.")
    print()


# ------------------------------------------------------------------ part 6
def part6(ys):
    print("=" * 74)
    print("PART 6 - THE KERNEL FLAG IS A COORDINATE FLAG (so its dimensions")
    print("         are histogram data; only its POSITION is new)")
    print("=" * 74)
    print("   y     F   dim ker N^n = P - tail(n)?   flag strictly increasing?")
    for y in ys:
        gears = primes(5, y)
        b = blocked(gears)
        P = b.size
        W, _ = gap_hist(b)
        F = int(np.flatnonzero(W)[-1])
        gs = np.arange(W.size)
        v = np.ones(P, bool)
        dims = []
        for n in range(1, F + 1):
            v &= b[(np.arange(P) + n) % P]
            dims.append(P - int(v.sum()))
        tail = [P - int(np.dot(W, np.maximum(gs - n, 0)))
                for n in range(1, F + 1)]
        assert dims == tail
        assert all(dims[i] < dims[i + 1] for i in range(len(dims) - 1))
        print(f"  {y:3d} {F:5d}                        True"
              f"                        True")
    b = blocked(primes(5, 11))
    N = dense_N(b)
    P = b.size
    M = np.eye(P)
    for n in range(1, 4):
        M = N @ M
        zc = np.flatnonzero(~M.any(axis=0))
        assert np.linalg.matrix_rank(M) == P - zc.size
    print("  m11: ker N^n is spanned by basis vectors e_k (a COORDINATE")
    print("  subspace) - verified n = 1,2,3.  So the flag ker N < ker N^2 <")
    print("  ... is a nested family of SUBSETS of Z_P; its sizes are the")
    print("  histogram tail sums (unitary data, circular), while its position")
    print("  against the CRT gear basis is NOT a unitary invariant - that is")
    print("  exactly the Schmidt-rank profile round 22 measured GROWING.")
    print()


# ------------------------------------------------------------------ part 7
def part7():
    print("=" * 74)
    print("PART 7 - THE MOMENT FRAME REDUCES TO THE r_L LADDER (a NON-GAIN)")
    print("=" * 74)
    b = blocked(primes(5, 11))
    P = b.size
    N = dense_N(b)
    A = N + N.T
    W, _ = gap_hist(b)
    F = int(np.flatnonzero(W)[-1])
    v = np.ones(P, bool)
    r = [P]
    for n in range(1, F + 2):
        v &= b[(np.arange(P) + n) % P]
        r.append(int(v.sum()))
    print("    t   tr(A^2t) direct      sum_L m_t(L) r_L      equal?")
    Ap = np.eye(P)
    for t in range(1, 7):
        Ap = Ap @ A @ A
        direct = int(round(np.trace(Ap)))
        # closed 2t-walks on Z from 0, classified by (max, min)
        m = {}
        def walks(pos, lo, hi, steps):
            if steps == 0:
                if pos == 0:
                    m[hi - lo] = m.get(hi - lo, 0) + 1
                return
            walks(pos + 1, lo, max(hi, pos + 1), steps - 1)
            walks(pos - 1, min(lo, pos - 1), hi, steps - 1)
        walks(0, 0, 0, 2 * t)
        form = sum(c * (r[L] if L < len(r) else 0) for L, c in m.items())
        assert direct == form, (t, direct, form)
        print(f"  {t:3d} {direct:18d} {form:21d}      True")
    print("  So every trace/moment (equivalently every exponential-sum) bound")
    print("  on lam_max(A) is a POSITIVE combination of r_L = rank(N^L), the")
    print("  ladder round 21 already computes exactly and scan-free.  The")
    print("  moment frame contains no information beyond it.  NON-GAIN,")
    print("  recorded so it is not built again.")
    print()



# ------------------------------------------------------------------ part 8
def part8():
    print("=" * 74)
    print("PART 8 - WEYL ON THE MERGE STEP IS VACUOUS (a checked NON-GAIN)")
    print("=" * 74)
    print("  A_new = A_old + Delta with Delta the edges whose RIGHT endpoint is")
    print("  newly blocked by q'.  Delta's own graph is a union of paths, so")
    print("  Weyl gives lam_max(A_new) <= lam_max(A_old) + lam_max(Delta).")
    print("   step      F_old  longest new-blocked run  lam(Delta)  Weyl bound")
    for y, qp in [(11, 13), (13, 17), (17, 19), (19, 23)]:
        old = primes(5, y)
        new = old + [qp]
        bo = blocked(old)
        bn = blocked(new)
        P = bn.size
        rep = np.tile(bo, qp)
        newly = bn & ~rep
        # longest run of consecutive newly-blocked slots (cyclic)
        x = np.concatenate([newly, newly])
        run = best = 0
        for v in x:
            run = run + 1 if v else 0
            best = max(best, run)
        best = min(best, P)
        Wo, _ = gap_hist(bo)
        Fo = int(np.flatnonzero(Wo)[-1])
        # Delta: edge {k-1,k} for each newly blocked k -> paths of best+1 nodes
        lam_d = 2 * cos(pi / (best + 2))
        weyl = 2 * cos(pi / (Fo + 1)) + lam_d
        print(f"   {y:3d}->{qp:3d}   {Fo:5d}  {best:22d}  {lam_d:10.4f}"
              f"  {weyl:10.4f}   {'VACUOUS' if weyl >= 2 else 'bites'}")
    print("  Every step exceeds 2, so the Weyl/perturbation route to the merge")
    print("  law proves nothing: the merge step's whole content is WHICH edges")
    print("  are added, not how many.  Recorded so it is not retried.")
    print()


def main():
    big = "--big" in sys.argv
    ys = [11, 13, 17, 19]
    t0 = time.time()
    part1(ys)
    part2()
    part3(ys, big)
    part4(ys)
    part5(ys)
    part6(ys)
    part7()
    part8()
    print(f"total {time.time()-t0:.1f}s - all assertions passed")


if __name__ == "__main__":
    main()
