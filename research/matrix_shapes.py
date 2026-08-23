"""Round 21 exploration: DIFFERENT matrix shapes over the machine's raw
materials (residues u_q, cycle lengths q, slip amounts) - NOT the CRT tensor
frame (that is research/matrix_machine.py / docs/novel/matrix-formulation.md).

Directive (human, verbatim): "crt as tensor product is one way to use a
matrix, we have residues and cycle lengths, and slip amounts. try different
matrix shapes and operations to see if there are any new mechanisms."

Shapes tested (each section prints a VERDICT line; every claim asserted):

  SHAPE 2  gear x window incidence -> the covering IP, its LP relaxation,
           LP DUAL certificates, integrality gap machines 11..19, and the
           Bonferroni/Kounias moment hierarchy (level-2 closed-form
           certificate, level ceiling law, level needed per machine).
  SHAPE 3  gap-word circulant in the RANK frame (index = opening rank, not
           position): spectrum, autocorrelation, cyclotomic rank.
  SHAPE 1  gear x gear matrices (first-collision times, u_q mod r):
           determinant/spectral structure vs known quantities.
  SHAPE 5  tropical/graph: tooth-PAIR consecutive-slot digraph (coverage
           side analogue of Constructor R37's window pair-support graph).
  SHAPE 4  lap/slip permutation matrices of the merge law: orbit and
           commutator structure.
  SHAPE 6  polynomial/resultant forms of tooth data.

House rules: exact arithmetic (int/Fraction) for every claim; floats only
in lines labeled FLOAT/SOLVER (discovery, then exact verification). Small
machines only (P <= 2e6 for any period scan). Run:

    uv run python research/matrix_shapes.py            # all sections
    uv run python research/matrix_shapes.py 2 3        # chosen sections
"""
import sys
from fractions import Fraction
from itertools import combinations, product
from math import prod, gcd

import numpy as np

# ---------------------------------------------------------------- utilities
def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for p in range(2, int(n ** .5) + 1):
        if s[p]:
            s[p * p::p] = False
    return [int(p) for p in np.flatnonzero(s)]

def gears_of(y):
    return [p for p in primes_upto(y) if p >= 5]

def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q

def sieve_openings(gears):
    P = prod(gears)
    a = np.ones(P, bool)
    for q in gears:
        t1, t2 = teeth(q)
        a[t1::q] = False
        a[t2::q] = False
    return a

def F_exact(gears):
    """max gap between consecutive openings over the period (slot frame)."""
    idx = np.flatnonzero(sieve_openings(gears))
    P = prod(gears)
    gaps = np.diff(np.append(idx, idx[0] + P))
    return int(gaps.max()), gaps, idx

F_KNOWN = {11: 7, 13: 11, 17: 18, 19: 25}   # slot-frame ladder, gears 5..y


# ===========================================================================
# SHAPE 2 - GEAR x WINDOW INCIDENCE: the covering IP, LP duality, and the
# Bonferroni/Kounias moment hierarchy.
# ===========================================================================
#
# EXACT IP.  A window of W consecutive slots starting at slot k is fully
# covered iff every i in [0,W) has (k+i) mod q in T_q for some gear q.  By
# CRT every phase tuple (r_q)_q occurs as some k, so
#     max coverable W  =  F(M) - 1        (max blocked run)
# and the IP over phase choices is EXACT, not a model.
#
# LP RELAXATION (level 1).  z_{q,r} >= 0, sum_r z_{q,r} = 1, and for each
# position i:  sum_q z_q[(u-i) mod q] + z_q[(-u-i) mod q] >= 1.
#
# LEVEL 2.  Add pair variables z2_{q,q'} (a joint distribution per gear
# pair) and the KOUNIAS CUT, valid POINTWISE for 0/1 indicators
# (brute-verified below):  for every k,
#     1{some gear covers i} <= sum_j 1{A_j} - sum_{j != k} 1{A_j & A_k}.
# Averaged over the window with exact hit counts this yields the CLOSED-FORM
# counting certificate (no LP needed):
#     if  sum_q 2*ceil(W/q) - 4*sum_{j != k} floor(W/(q_j q_k)) < W
#     for some k, then no fully covered window of width W exists, i.e.
#     F(M) <= W.

def covers(q, i):
    u, v = teeth(q)
    return (u - i) % q, (v - i) % q

def check_pointwise_cuts(nmax=6):
    """brute-force validity of the Kounias cut and the chain cut over ALL
    0/1 event patterns on up to nmax events."""
    for n in range(1, nmax + 1):
        for bits in product((0, 1), repeat=n):
            m = sum(bits)
            un = 1 if m else 0
            # Kounias, every k
            for k in range(n):
                s2 = sum(bits[j] * bits[k] for j in range(n) if j != k)
                assert un <= m - s2, (bits, k)
            # depth-3 chain cut, every ordered (k, l)
            for k in range(n):
                for l in range(n):
                    if l == k:
                        continue
                    s2 = sum(bits[j] * bits[k] for j in range(n) if j != k)
                    s3 = sum(bits[j] * bits[l] * (1 - bits[k])
                             for j in range(n) if j not in (k, l))
                    assert un <= m - s2 - s3, (bits, k, l)
    return True

def hit_count(q, r, W):
    """exact #positions i in [0,W) with r in covers(q,i) - i.e. positions
    hit by gear q at phase r."""
    u, v = teeth(q)
    n = 0
    for t in (u, v):
        # i == (t - r) mod q hits, every q steps
        first = (t - r) % q
        if first < W:
            n += (W - 1 - first) // q + 1
    return n

def pair_hit_count(qa, ra, qb, rb, W):
    """exact #positions hit by BOTH gear a at phase ra and gear b at rb."""
    n = 0
    for ta in teeth(qa):
        for tb in teeth(qb):
            # i == (ta-ra) mod qa and i == (tb-rb) mod qb -> CRT
            m = qa * qb
            # solve
            r1, r2 = (ta - ra) % qa, (tb - rb) % qb
            # CRT combine
            inv = pow(qa, -1, qb)
            x = (r1 + qa * ((r2 - r1) * inv % qb)) % m
            if x < W:
                n += (W - 1 - x) // m + 1
    return n

def closed_form_cert(gears, kind, Wmax=200000):
    """min W such that the counting certificate signs (returns None if it
    never does up to Wmax).  kind = 'density' (level 1) or 'kounias'
    (level 2, best k)."""
    for W in range(1, Wmax + 1):
        s1 = sum(2 * ((W + q - 1) // q) for q in gears)
        if kind == 'density':
            if s1 < W:
                return W
        else:
            for k in gears:
                sub = 4 * sum(W // (j * k) for j in gears if j != k)
                if s1 - sub < W:
                    return W
    return None

def uniform_lp1_feasible(gears):
    """exact: uniform z_{q,r}=1/q gives coverage sum 2/q at EVERY position
    of EVERY width (the two covering phases are always distinct)."""
    for q in gears:
        u, v = teeth(q)
        assert (u - v) % q != 0
    return sum(Fraction(2, q) for q in gears)

# ---- float LP machinery (SOLVER - discovery only; claims verified exact) --
def lp1_value(gears, W):
    """max t s.t. coverage(i) >= t, per-gear sums = 1, z >= 0. SOLVER."""
    from scipy.optimize import linprog
    off, idx = {}, 0
    for q in gears:
        off[q] = idx; idx += q
    nz = idx
    # vars: z (nz), t (1). minimize -t
    c = np.zeros(nz + 1); c[-1] = -1.0
    A_ub, b_ub = [], []
    for i in range(W):
        row = np.zeros(nz + 1)
        for q in gears:
            a, b = covers(q, i)
            row[off[q] + a] += 1.0
            row[off[q] + b] += 1.0
        row[-1] = -1.0
        A_ub.append(-row); b_ub.append(0.0)        # coverage - t >= 0
    A_eq, b_eq = [], []
    for q in gears:
        row = np.zeros(nz + 1)
        row[off[q]:off[q] + q] = 1.0
        A_eq.append(row); b_eq.append(1.0)
    bounds = [(0, None)] * nz + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    assert res.status == 0, res.message
    return -res.fun, res

def lp1_farkas_exact(gears, W, res):
    """exact infeasibility certificate for {coverage >= 1} at width W from
    the solver duals: weights y_i >= 0 with
        sum_q max_r (y-mass gear q covers at phase r)  <  sum_i y_i."""
    duals = res.ineqlin.marginals            # <= 0 for our -row form
    y = [Fraction(max(0.0, -d)).limit_denominator(10 ** 7)
         for d in duals]
    total = sum(y)
    if total == 0:
        return False
    lhs = Fraction(0)
    for q in gears:
        best = Fraction(0)
        for r in range(q):
            m = sum(y[i] for i in range(W) if r in covers(q, i))
            best = max(best, m)
        lhs += best
    return lhs < total, lhs, total

def lp2_value(gears, W):
    """decoupled level-2 LP (Kounias cuts; per-pair joints, NO marginal
    consistency - a valid weakening, so infeasibility still certifies).
    max t s.t. for all i, k:  S1(i) - sum_{j!=k} p2_{jk}(i) >= t.  SOLVER."""
    from scipy.optimize import linprog
    off, idx = {}, 0
    for q in gears:
        off[q] = idx; idx += q
    poff = {}
    for a, b in combinations(gears, 2):
        poff[(a, b)] = idx; idx += a * b
    nz = idx
    c = np.zeros(nz + 1); c[-1] = -1.0
    rows, b_ub = [], []
    # precompute per-pair covering index lists per position
    for i in range(W):
        cov1 = {q: covers(q, i) for q in gears}
        for k in gears:
            row = np.zeros(nz + 1)
            for q in gears:
                a, b = cov1[q]
                row[off[q] + a] += 1.0
                row[off[q] + b] += 1.0
            for j in gears:
                if j == k:
                    continue
                a, b = (j, k) if j < k else (k, j)
                base = poff[(a, b)]
                ra_list = cov1[a]; rb_list = cov1[b]
                for ra in ra_list:
                    for rb in rb_list:
                        row[base + ra * b + rb] -= 1.0
            row[-1] = -1.0
            rows.append(-row); b_ub.append(0.0)
    A_eq, b_eq = [], []
    for q in gears:
        r = np.zeros(nz + 1); r[off[q]:off[q] + q] = 1.0
        A_eq.append(r); b_eq.append(1.0)
    for (a, b), base in poff.items():
        r = np.zeros(nz + 1); r[base:base + a * b] = 1.0
        A_eq.append(r); b_eq.append(1.0)
    bounds = [(0, None)] * nz + [(None, None)]
    res = linprog(c, A_ub=rows, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    assert res.status == 0, res.message
    return -res.fun, res

def lp2_farkas_exact(gears, W, res):
    """exact infeasibility certificate at width W from the level-2 duals.
    y_{i,k} >= 0 on the cuts; feasibility would force
        sum_{ik} y_{ik} <= sum_q max_r C_q(r) + sum_pairs max_{ra,rb} D(ra,rb)
    where C, D are the y-weighted coefficient blocks; certificate signs if
    RHS < LHS.  All Fractions."""
    duals = res.ineqlin.marginals
    ks = list(range(len(duals)))
    y = {}
    n = 0
    for i in range(W):
        for k_ in gears:
            d = duals[n]; n += 1
            v = Fraction(max(0.0, -float(d))).limit_denominator(10 ** 7)
            if v:
                y[(i, k_)] = v
    total = sum(y.values())
    if total == 0:
        return False, None, None
    rhs = Fraction(0)
    for q in gears:
        best = None
        for r in range(q):
            m = sum(v for (i, k_), v in y.items() if r in covers(q, i))
            best = m if best is None else max(best, m)
        rhs += best
    for a, b in combinations(gears, 2):
        best = None
        for ra in range(a):
            # positions where gear a at phase ra hits
            hits_a = set(i for i in range(W) if ra in covers(a, i))
            for rb in range(b):
                m = Fraction(0)
                for (i, k_), v in y.items():
                    if k_ in (a, b) and i in hits_a and rb in covers(b, i):
                        m -= v
                best = m if best is None else max(best, m)
        rhs += best
    return rhs < total, rhs, total

def level_slope(gears, chain):
    """exact uniform-product slope of the chain cut: coverage-per-position
    lower bound sum minus subtracted terms; the LEVEL-(len(chain)+1)
    criterion.  chain = ordered list of distinguished gears."""
    s = sum(Fraction(2, q) for q in gears)
    prior = []
    for k in chain:
        f = Fraction(1)
        for kp in prior:
            f *= (1 - Fraction(2, kp))
        s -= Fraction(2, k) * f * sum(Fraction(2, j)
                                      for j in gears if j != k and j not in prior)
        prior.append(k)
    return s

def level_needed(gears, lmax=8):
    """min chain length t (level = t+1... reported as moment degree t+1)
    such that some chain gives uniform-product slope < 1.  Greedy: smallest
    gears first is the best chain (they subtract most)."""
    gs = sorted(gears)
    for t in range(1, lmax + 1):
        if level_slope(gears, gs[:t]) < 1:
            return t + 1
    return None

def shape2():
    print("=" * 78)
    print("SHAPE 2 - GEAR x WINDOW: covering IP, LP duality, moment hierarchy")
    print("=" * 78)

    # pointwise cut validity, brute-forced
    assert check_pointwise_cuts(6)
    print("pointwise cut validity: Kounias (every k) and depth-3 chain cut "
          "hold for ALL 0/1 patterns on <= 6 events (exhaustive)")

    # exact hit-count bounds used by the closed-form certificate
    for q in (5, 7, 11):
        for W in (3, 8, 20, 37):
            for r in range(q):
                assert hit_count(q, r, W) <= 2 * ((W + q - 1) // q)
    for (qa, qb) in ((5, 7), (5, 11), (7, 13)):
        for W in (10, 40, 90):
            for ra in range(qa):
                for rb in range(qb):
                    assert pair_hit_count(qa, ra, qb, rb, W) >= \
                        4 * (W // (qa * qb))
    print("exact hit-count bounds verified: per-gear <= 2*ceil(W/q), "
          "per-pair >= 4*floor(W/qq') for all phases (exhaustive small)")

    # IP == F - 1 exactly (period sieve, house-limit machines)
    print()
    for y in (11, 13, 17, 19):
        gears = gears_of(y)
        F, gaps, idx = F_exact(gears)
        assert F == F_KNOWN[y]
        print(f"machine {y}: IP max covered width = F - 1 = {F - 1} "
              f"(exact period sieve; the phase IP is EXACT by CRT)")

    # level-1 LP: uniform certificate / integrality gap
    print()
    for y in (11, 13, 17, 19):
        gears = gears_of(y)
        s = uniform_lp1_feasible(gears)
        if s >= 1:
            print(f"machine {y}: LP-1 FEASIBLE AT EVERY WIDTH - exact "
                  f"uniform certificate, coverage = sum 2/q = {s} >= 1 at "
                  f"every position -> INTEGRALITY GAP INFINITE "
                  f"(true F-1 = {F_KNOWN[y]-1})")
        else:
            print(f"machine {y}: uniform coverage sum 2/q = {s} < 1 - "
                  f"LP-1 threshold is finite; computing it...")
            Wlo, Whi = F_KNOWN[y] - 1, 400
            # scan up from Wlo for first infeasible width (monotone)
            Wstar = None
            W = Wlo
            while W <= Whi:
                t, res = lp1_value(gears, W)
                if t < 1 - 1e-9:
                    Wstar = W
                    break
                W += 1
            assert Wstar is not None
            ok, lhs, tot = lp1_farkas_exact(gears, Wstar, res)
            assert ok, "exact Farkas verification failed"
            print(f"   LP-1 min infeasible width = {Wstar} (SOLVER), "
                  f"EXACT dual certificate verified: "
                  f"sum_q max_r y-mass = {lhs} < {tot} = sum y "
                  f"-> F({y}) <= {Wstar} by LP duality alone "
                  f"(true F = {F_KNOWN[y]}; ratio "
                  f"{Wstar / F_KNOWN[y]:.1f}x)")

    # closed-form certificates (exact, machine-independent arithmetic)
    print()
    print("closed-form counting certificates (EXACT, no LP, no scan):")
    print(f"{'machine':>8} {'F':>4} {'density W*':>11} {'kounias W*':>11}")
    for y in (11, 13, 17, 19, 23, 29, 31):
        gears = gears_of(y)
        wd = closed_form_cert(gears, 'density', 2000)
        wk = closed_form_cert(gears, 'kounias', 20000)
        Ftrue = F_KNOWN.get(y)
        if Ftrue is not None and wk is not None:
            assert Ftrue - 1 < wk
        print(f"{y:>8} {str(Ftrue or '?'):>4} "
              f"{str(wd) if wd else 'never':>11} "
              f"{str(wk) if wk else 'never':>11}")

    # level ceiling law: uniform-product slope per level
    print()
    print("moment-hierarchy ceiling (exact Fractions; slope < 1 = level "
          "still certifies):")
    for y in (13, 17, 19, 23, 29, 31, 37, 41, 43):
        gears = gears_of(y)
        s2 = level_slope(gears, [5])
        lvl = level_needed(gears)
        print(f"  machine {y}: level-2 slope (k=5) = {float(s2):.4f} "
              f"{'< 1 CERTIFIES' if s2 < 1 else '>= 1 dead'};  "
              f"min chain depth with slope < 1: {lvl - 1 if lvl else '>8'} "
              f"(moment degree {lvl if lvl else '>9'})")

    # level-2 LP (best degree-2 certificate) on machines 11..19
    print()
    print("level-2 LP (Kounias cuts + pair joints; SOLVER discovery, exact "
          "Farkas verification):")
    F_EXTERN = {**F_KNOWN, 23: 34}         # F(23) external (census ladder);
                                           # period 3.7e7 > scan cap - the
                                           # LP bound below is SCAN-FREE
    for y in (11, 13, 17, 19, 23):
        gears = gears_of(y)
        # bisection for min infeasible width
        lo, hi = F_EXTERN[y] - 1, 400
        tlo, _ = lp2_value(gears, lo)
        if tlo < 1 - 1e-9:
            print(f"  machine {y}: level-2 LP already infeasible at "
                  f"W = F-1 = {lo}?! investigate")
            continue
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            t, _ = lp2_value(gears, mid)
            if t >= 1 - 1e-9:
                lo = mid
            else:
                hi = mid
        Wstar = hi
        t, res = lp2_value(gears, Wstar)
        assert t < 1 - 1e-9
        ok, rhs, tot = lp2_farkas_exact(gears, Wstar, res)
        tag = "EXACT dual certificate verified" if ok else \
              "float only (exact rounding failed)"
        ncert = sum(1 for v in res.ineqlin.marginals if -float(v) > 1e-12)
        print(f"  machine {y}: min infeasible W = {Wstar} -> F <= {Wstar} "
              f"({tag}; certificate support {ncert} rational cut weights); "
              f"true F = {F_EXTERN[y]}, ratio "
              f"{Wstar / F_EXTERN[y]:.2f}x"
              + ("   [SCAN-FREE: period exceeds house scan cap]"
                 if y == 23 else ""))


# ===========================================================================
# SHAPE 3 - GAP-WORD CIRCULANT IN THE RANK FRAME.  Index = opening RANK n
# (not position k).  The position frame CRT-factorises (tensor doc); the
# rank frame does not - it is where (D)'s objects (j consecutive gaps) live.
# Objects: the circulant with first row = the gap word g_0..g_{A-1}
# (A = prod(q-2)).  Its eigenvalues are the DFT of g; its rank is A minus
# the number of cyclotomic factors Phi_d dividing G(x) = sum g_n x^n.
# ===========================================================================

def eval_mod_p(folded, d, bound=1 << 61):
    """exact nonvanishing test: evaluate the folded word (coeffs mod x^d-1)
    at an element of order d in GF(p), p prime, p = 1 mod d.  Nonzero image
    proves Phi_d does not divide G.  Returns True if PROVEN nonzero."""
    import sympy
    p = d + 1
    while True:
        p = sympy.nextprime(p)
        if (p - 1) % d == 0:
            break
    ggen = sympy.primitive_root(p)
    w = pow(ggen, (p - 1) // d, p)
    # try all primitive d-th roots: Phi_d | G iff G vanishes at ALL of them;
    # to prove Phi_d does NOT divide, one nonzero value at any primitive
    # root suffices... but the hom picks one root; different primitive roots
    # are conjugate over Q so vanishing at one <=> at all.  One test enough.
    v = 0
    wk = 1
    for tcoef in folded:
        v = (v + tcoef * wk) % p
        wk = wk * w % p
    return v % p != 0

def shape3():
    print("=" * 78)
    print("SHAPE 3 - RANK-FRAME GAP-WORD CIRCULANT: spectrum, "
          "autocorrelation, cyclotomic rank")
    print("=" * 78)
    import sympy
    for y in (11, 13, 17, 19):
        gears = gears_of(y)
        P = prod(gears)
        F, gaps, idx = F_exact(gears)
        A = len(gaps)
        assert A == prod(q - 2 for q in gears)
        g = gaps.astype(np.int64)

        # palindrome-up-to-rotation (k -> -k symmetry of the opening set)
        rev = g[::-1]
        is_pal = False
        gg = np.concatenate([g, g])
        for s in range(A):
            if np.array_equal(rev, gg[s:s + A]):
                is_pal = True
                break
        assert is_pal, "gap word not a rotated palindrome?"

        # exact rank-frame autocorrelation / covariance, lags 0..12
        lags = range(0, 13)
        R = {l: int(np.dot(g, np.roll(g, -l))) for l in lags}
        # covariance A*R(l) - P^2 (exact integers); corr = cov(l)/cov(0)
        cov = {l: A * R[l] - P * P for l in lags}
        corr = [Fraction(cov[l], cov[0]) for l in lags]
        pretty = " ".join(f"{float(c):+.3f}" for c in corr[1:9])
        print(f"machine {y}: A = {A}; rank-frame gap autocorrelation "
              f"(exact, lags 1..8): {pretty}")

        # FLOAT: spectrum of the circulant = FFT of g; top non-DC lines
        spec = np.abs(np.fft.fft(g.astype(float))) ** 2
        order = np.argsort(spec[1:])[::-1] + 1
        top = [(int(j), float(spec[j]) / float(spec[0]))
               for j in order[:6] if j <= A // 2]
        tops = ", ".join(f"j={j} (A/gcd={A // gcd(j, A)}, "
                         f"pow={p:.2e})" for j, p in top[:4])
        print(f"   FLOAT top spectral lines (|ghat|^2 / DC^2): {tops}")
        # IDENTIFICATION (FLOAT check of an exact-frame explanation): a
        # position-periodic structure of period L slots appears in the RANK
        # frame at frequency j = P/L (time change: L slots = L*A/P ranks).
        # The corridor wheels L = 35 and L = 385 should be the top lines.
        cand = {35: P // 35 if P % 35 == 0 else None,
                385: P // 385 if P % 385 == 0 else None}
        topj = top[0][0] if top else None
        wheel_hits = [L for L, j in cand.items()
                      if j and any(abs(tj - j) <= 5 for tj, _ in top[:3])]
        print(f"   corridor-wheel transplant j = P/35 = "
              f"{cand[35]}, j = P/385 = {cand[385]}: top-3 lines hit "
              f"wheels {wheel_hits if wheel_hits else 'NONE'}")

        # exact cyclotomic rank: for each divisor d | A prove Phi_d does
        # not divide G (GF(p) evaluation; escalate to exact rem if 0)
        if A <= 400000:
            divs = sorted(sympy.divisors(A))
            deficient = []
            for d in divs:
                if d == 1:
                    # Phi_1 = x - 1: G(1) = sum g = P != 0
                    assert int(g.sum()) == P != 0
                    continue
                folded = np.zeros(d, object)
                for t in range(0, A, d):
                    folded[:min(d, A - t)] += g[t:t + d]
                # numpy object fold is wrong when d doesn't divide into A
                # cleanly; redo exact:
                folded = [0] * d
                for n_ in range(A):
                    folded[n_ % d] += int(g[n_])
                if not eval_mod_p(folded, d):
                    # possible vanishing - exact check
                    x = sympy.symbols('x')
                    Gp = sympy.Poly(list(reversed(folded)), x)
                    rem = Gp.rem(sympy.Poly(sympy.cyclotomic_poly(d, x), x))
                    if rem.is_zero:
                        deficient.append(d)
            if deficient:
                print(f"   cyclotomic RANK DEFICIENCY at d = {deficient} "
                      f"(exact) - hidden periodic structure!")
            else:
                print(f"   circulant FULL RANK {A} (exact: Phi_d does not "
                      f"divide G for any of the {len(divs)} divisors d | A "
                      f"- GF(p) witness each)")


# ===========================================================================
# SHAPE 1 - GEAR x GEAR matrices from residues + first collisions.
# T[i][j] = first doubly-blocked slot of gears (q_i, q_j);
# U[i][j] = u_{q_i} mod q_j.  Determinant / spectral probes vs known
# quantities (F ladder, fuel caps).
# ===========================================================================

def first_collision(qa, qb):
    best = None
    for ta in teeth(qa):
        for tb in teeth(qb):
            m = qa * qb
            inv = pow(qa, -1, qb)
            x = (ta + qa * ((tb - ta) * inv % qb)) % m
            if x == 0:
                x = m
            best = x if best is None else min(best, x)
    return best

def shape1():
    print("=" * 78)
    print("SHAPE 1 - GEAR x GEAR slip/collision matrices: determinant and "
          "spectral probes")
    print("=" * 78)
    import sympy
    FUEL = {11: 2, 13: 2, 17: 2, 19: 3, 23: 3, 29: 4, 31: 4}  # kmax ladder
    allg = gears_of(31)
    m = len(allg)
    T = sympy.zeros(m, m)
    U = sympy.zeros(m, m)
    for i, qa in enumerate(allg):
        for j, qb in enumerate(allg):
            if i == j:
                T[i, j] = qa       # self: first self-blocked slot = tooth u
                U[i, j] = 0
            else:
                T[i, j] = first_collision(qa, qb)
                U[i, j] = pow(6, -1, qa) % qb
    print("first-collision matrix T (leading 4x4):")
    for i in range(4):
        print("   ", [int(T[i, j]) for j in range(4)])
    print("probes on leading principal minors (machines 11..31):")
    rows = []
    for mm in range(3, m + 1):
        y = allg[mm - 1]
        Tm = T[:mm, :mm]
        Um = U[:mm, :mm]
        detT = int(Tm.det())
        detU = int(Um.det())
        trT = int(Tm.trace())
        # FLOAT spectral radius
        ev = np.linalg.eigvals(np.array(Tm.tolist(), float))
        rho = float(np.abs(ev).max())
        Fv = F_KNOWN.get(y)
        fuel = FUEL.get(y)
        rows.append((y, detT, detU, trT, rho, Fv, fuel))
        print(f"   machine {y}: det T = {detT}, det U = {detU}, "
              f"tr T = {trT}, rho(T) FLOAT = {rho:.2f}   "
              f"[F = {Fv}, fuel = {fuel}]")
    # candidate relations tested exactly (all expected to fail -> null):
    hits = []
    for (y, detT, detU, trT, rho, Fv, fuel) in rows:
        if Fv is None:
            continue
        for name, val in (("detT", detT), ("detU", detU), ("trT", trT)):
            for fname, f in (("F", Fv), ("F-1", Fv - 1), ("fuel", fuel)):
                if f and val % f == 0 and abs(val) // f < 10 ** 6:
                    # divisibility alone is weak; only record equality or
                    # constant small ratio across >= 3 machines (checked
                    # after loop)
                    hits.append((name, fname, y, val // f))
    # constant-ratio scan
    found = False
    for name in ("detT", "detU", "trT"):
        for fname in ("F", "F-1", "fuel"):
            rs = [r for (n_, f_, y_, r) in hits if n_ == name and f_ == fname]
            if len(rs) >= 3 and len(set(rs)) == 1:
                print(f"   CONSTANT RATIO FOUND: {name}/{fname} = {rs[0]}")
                found = True
    if not found:
        print("   no equality / constant-ratio relation between "
              "{det T, det U, tr T, rho(T)} and {F, F-1, fuel} across "
              "machines (exact scan) -> NULL")


# ===========================================================================
# SHAPE 5 - TOOTH-PAIR / TOOTH-TRIPLE DIGRAPHS (coverage-side analogue of
# Constructor R37's window pair-support graph; different object - nodes are
# TOOTH assignments to consecutive slots, not window gaps).
# ===========================================================================

def tooth_nodes(gears):
    """all (gear, tooth) labels."""
    out = []
    for q in gears:
        for t in teeth(q):
            out.append((q, t))
    return out

def compat(a, b, shift):
    """can tooth a hit slot k and tooth b hit slot k+shift for some k?
    a = (qa, ta).  CRT: always if qa != qb; if same gear need
    tb - ta == shift mod q."""
    (qa, ta), (qb, tb) = a, b
    if qa != qb:
        return True
    return (tb - ta) % qa == shift % qa

def shape5():
    print("=" * 78)
    print("SHAPE 5 - TOOTH-PAIR AND TOOTH-TRIPLE DIGRAPHS (tropical/"
          "longest-path probes, coverage side)")
    print("=" * 78)
    # same-gear consecutive slots impossible (exact, all gears to 100):
    for q in gears_of(100):
        u, v = teeth(q)
        for ta in (u, v):
            for tb in (u, v):
                assert (tb - ta) % q != 1 % q or q == 0
    print("same gear can never block two CONSECUTIVE slots (exact, all "
          "gears q <= 100): tooth differences {0, +-2u} never = 1 mod q")

    for y in (11, 13, 17, 19):
        gears = gears_of(y)
        nodes = tooth_nodes(gears)
        # pair graph: nodes = (a, b) with a at slot 0, b at slot 1
        pnodes = [(a, b) for a in nodes for b in nodes if compat(a, b, 1)]
        # edge (a,b) -> (b,c): need compat(a,c,2) as well
        edges = {}
        for (a, b) in pnodes:
            edges[(a, b)] = [(b, c) for c in nodes
                             if compat(b, c, 1) and compat(a, c, 2)]
        # cycle detection (iterative DFS, 3-color)
        color = {v_: 0 for v_ in pnodes}
        cyc = False
        for s in pnodes:
            if color[s]:
                continue
            stack = [(s, iter(edges[s]))]
            color[s] = 1
            while stack:
                v_, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    color[v_] = 2
                    stack.pop()
                    continue
                if color[nxt] == 1:
                    cyc = True
                    break
                if color[nxt] == 0:
                    color[nxt] = 1
                    stack.append((nxt, iter(edges[nxt])))
            if cyc:
                break
        print(f"machine {y}: tooth-PAIR digraph ({len(pnodes)} nodes) is "
              f"{'CYCLIC - certifies NOTHING (2-point insufficiency at '
                 'every machine; the 5-7 corridor alternation is a cycle)'
              if cyc else 'ACYCLIC'}")
        assert cyc, "pair graph unexpectedly acyclic - investigate"

    # tooth-TRIPLE graph: nodes = consistent triples on slots 0,1,2
    print()
    for y in (11, 13, 17):
        gears = gears_of(y)
        nodes = tooth_nodes(gears)
        tnodes = [(a, b, c) for a in nodes for b in nodes for c in nodes
                  if compat(a, b, 1) and compat(b, c, 1) and compat(a, c, 2)]
        tedges = {}
        for (a, b, c) in tnodes:
            tedges[(a, b, c)] = [
                (b, c, d) for d in nodes
                if compat(c, d, 1) and compat(b, d, 2) and compat(a, d, 3)]
        color = {v_: 0 for v_ in tnodes}
        cyc = False
        for s in tnodes:
            if color[s]:
                continue
            stack = [(s, iter(tedges[s]))]
            color[s] = 1
            while stack and not cyc:
                v_, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    color[v_] = 2
                    stack.pop()
                    continue
                if color[nxt] == 1:
                    cyc = True
                elif color[nxt] == 0:
                    color[nxt] = 1
                    stack.append((nxt, iter(tedges[nxt])))
            if cyc:
                break
        msg = ""
        if not cyc:
            # DAG: longest path (edges) by memoised DFS; a path with s
            # edges witnesses s+3 coverable consecutive slots, so
            # max run <= longest + 3 hence F <= longest + 4 IF the graph
            # relaxation were tight; it is an upper bound in any case
            memo = {}
            def lp(v_):
                if v_ in memo:
                    return memo[v_]
                memo[v_] = 1 + max((lp(w) for w in tedges[v_]), default=0)
                return memo[v_]
            sys.setrecursionlimit(100000)
            L = max(lp(v_) for v_ in tnodes) - 1
            Ft = F_KNOWN.get(y)
            msg = (f" -> longest path {L} edges: max run <= {L + 3}, "
                   f"F <= {L + 4} (true F = {Ft})")
        print(f"machine {y}: tooth-TRIPLE digraph ({len(tnodes)} nodes) is "
              f"{'CYCLIC' if cyc else 'ACYCLIC'}{msg}")


# ===========================================================================
# SHAPE 4 - LAP/SLIP PERMUTATION MATRICES of the merge law.
# pi: r -> r + P mod q' (per-lap slip of a fixed old opening's phase);
# D = teeth projector.  New exact identity tested: the slip-delete operator
# (I - D) pi is NILPOTENT with index = max lap-run between the two kills
# = max(a, q'-a) where a = lap distance = 2u' / P mod q'.
# ===========================================================================

def shape4():
    print("=" * 78)
    print("SHAPE 4 - LAP/SLIP PERMUTATION ALGEBRA of the merge law")
    print("=" * 78)
    steps = [(prod([5, 7, 11]), 13), (prod([5, 7, 11, 13]), 17),
             (prod([5, 7, 11, 13, 17]), 19), (prod(gears_of(19)), 23)]
    for P, qp in steps:
        u = pow(6, -1, qp)
        Tset = {u % qp, (-u) % qp}
        # exact matrices over int
        pi = np.zeros((qp, qp), np.int64)
        for r in range(qp):
            pi[(r + P) % qp, r] = 1
        D = np.zeros((qp, qp), np.int64)
        for t in Tset:
            D[t, t] = 1
        I = np.eye(qp, dtype=np.int64)
        # (1) conjugate-orbit identity: sum_l pi^l D pi^-l = 2 I
        acc = np.zeros((qp, qp), np.int64)
        Pl = I.copy()
        for l in range(qp):
            acc += Pl @ D @ Pl.T
            Pl = pi @ Pl
        assert np.array_equal(acc, 2 * I)
        # (2) nilpotency of (I-D) pi and its index vs lap-gap formula
        N = (I - D) @ pi
        Nm = N.copy()
        mdx = 1
        while Nm.any():
            Nm = np.sign(Nm @ N)
            mdx += 1
        a = (2 * u) * pow(P, -1, qp) % qp
        pred = max(a, qp - a)
        assert mdx == pred, (qp, mdx, a, pred)
        print(f"P = {P}, q' = {qp}: sum_l pi^l D pi^-l = 2I exact; "
              f"(I-D)pi nilpotent, index {mdx} == max(a, q'-a), "
              f"a = 2u'/P mod q' = {a}  (max laps a doomed class survives)")
    print("   -> the slip operator's only invariant is the LAP GAP "
          "max(a, q'-a); the full-cycle permutation structure is "
          "Constructor R35's no-spectral-gap boundary in miniature")


# ===========================================================================
# SHAPE 6 - POLYNOMIAL / RESULTANT forms of the tooth data.
# ===========================================================================

def shape6():
    print("=" * 78)
    print("SHAPE 6 - RESULTANTS OF TOOTH POLYNOMIALS")
    print("=" * 78)
    import sympy
    x = sympy.symbols('x')
    gs = gears_of(13)
    vals = {}
    for qa, qb in combinations(gs, 2):
        # exposure polynomials A_q(x) = sum_{r exposed} x^r
        Aa = sum(x ** r for r in range(qa) if r not in teeth(qa))
        Ab = sum(x ** r for r in range(qb) if r not in teeth(qb))
        res = sympy.resultant(sympy.Poly(Aa, x), sympy.Poly(Ab, x))
        # tooth-difference polynomials x^{2u}-1 (slip separations)
        ta = sympy.Poly(x ** (2 * (pow(6, -1, qa) % qa)) - 1, x)
        tb = sympy.Poly(x ** (2 * (pow(6, -1, qb) % qb)) - 1, x)
        res2 = sympy.resultant(ta, tb)
        fc = first_collision(qa, qb)
        vals[(qa, qb)] = (int(res), int(res2), fc)
        print(f"   ({qa},{qb}): Res(A_qa, A_qb) = {int(res)}, "
              f"Res(x^2u-1, x^2u'-1) = {int(res2)}, first collision = {fc}")
    # res2 is determined by gcd(2u, 2u') alone (classical): assert & null it
    for (qa, qb), (r1, r2, fc) in vals.items():
        da, db = 2 * (pow(6, -1, qa) % qa), 2 * (pow(6, -1, qb) % qb)
        g_ = gcd(da, db)
        assert r2 == 0, (qa, qb, r2)   # x^a-1, x^b-1 share root 1 always
    print("   Res(x^2u-1, x^2u'-1) = 0 identically (shared root x = 1): "
          "carries NO machine information - NULL")


# ===========================================================================
def main():
    args = sys.argv[1:]
    todo = set(args) if args else {'2', '3', '1', '5', '4', '6'}
    if '2' in todo:
        shape2()
    if '3' in todo:
        shape3()
    if '1' in todo:
        shape1()
    if '5' in todo:
        shape5()
    if '4' in todo:
        shape4()
    if '6' in todo:
        shape6()

if __name__ == '__main__':
    main()
