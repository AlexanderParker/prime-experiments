"""Round 22 lateral, target (b) - SPECTRAL STATISTICS OF THE NON-TENSOR
SECTOR ALONE: the only place in the machine where a GUE-bearing operator
could live.

Round 21 (docs/novel/eigenvalue-statistics.md) refuted the Riemann bridge on
the machine's TENSOR operators: any CRT-product spectrum is Berry-Tabor,
hence Poisson BY CONSTRUCTION, so that test could never have found GUE.  The
honest continuation is to test the sector that does NOT factor, i.e.
everything built from B = I - (x)_q E_q.  This file does that, and the answer
is a THEOREM rather than a statistic.

THE OPERATORS OF THE NON-TENSOR SECTOR, AND THEIR EXACT SPECTRA.

(1) THE HERMITIAN PART OF THE BLOCKED WALK.  A = BS + (BS)^T, where
    (BS)[j,k] = b_j [j = k+1] - the nilpotent operator whose index is F(M).
    A is the adjacency matrix of the undirected graph on Z_P with an edge
    {k, k+1} exactly when k+1 is BLOCKED.  Between two consecutive openings
    at slots m and m+g the vertices m .. m+g-1 are chained and the edge
    {m+g-1, m+g} is absent, so:

      PATH-DECOMPOSITION THEOREM.  A is the disjoint union, over the gaps of
      the machine, of PATH graphs: a gap of g slots contributes P_g (a path
      on g vertices).  Hence, exactly,

        spec(A) = multiset union over gaps g, with multiplicity W_1(g), of
                  { 2 cos(pi j / (g+1)) : j = 1..g }.

      The spectrum of the non-tensor sector IS the gap histogram, read
      through Chebyshev.  Consequences, all exact:
        * the number of DISTINCT eigenvalues is |Farey(F+1)| - 2 =
          sum_{b=2}^{F+1} phi(b) = O(F^2) - a few hundred levels out of P;
        * every level therefore carries multiplicity ~ P / F^2;
        * the distinct levels are 2 cos(pi a/b) with b <= F+1: an explicit
          algebraic set of bounded degree, a FAREY / CHEBYSHEV spectrum.
      GUE needs P distinct levels with repulsion.  This spectrum has O(F^2)
      levels with astronomically large ties.  GUE is excluded structurally,
      not statistically.

(2) THE DEEP OPERATORS - where the sector's dimension actually grows.
    (BS)^n = diag(v_n) S^n, and research/nontensor.py measures its Schmidt
    rank across gear cuts GROWING with depth and with the machine.  But every
    one of those operators is NILPOTENT: spec((BS)^n) = {0} with multiplicity
    P, for every n >= 1.  There is no spectrum to have statistics.

    More generally, for ANY 0/1 vector w and any step t, diag(w) S^t + h.c.
    is a union of paths and cycles on the t-step lattice, so its spectrum
    lies in { 2 cos(pi a/b) } u { 2 cos(2 pi a/b) } - always degenerate,
    never repulsive.

(3) THE WORD-LEVEL TRANSFER MATRIX H (matrix-formulation piece 6) is
    block-triangular by word length, so its eigenvalues ARE its diagonal:
    coef_diag(w) = q' - #distinct{(t - p) mod q'} - INTEGERS in a window of
    width <= 2j+2 at word length j.  An integer spectrum with huge
    multiplicities; again not GUE.

THE DICHOTOMY (the round-22 answer to the bridge question).  In this machine
   * where the spectrum is RICH, the operator FACTORISES     -> Poisson;
   * where the operator does NOT factorise, the spectrum is either
     DEGENERATE (Farey/Chebyshev, integer) or EMPTY (nilpotent).
The growth of the non-tensor sector happens exactly in the NILPOTENT
direction, which has no spectrum at all.  So there is no operator anywhere in
the machine whose spectrum could be GUE - the two round-21 failures and this
one are a single structural fact.

WHAT IS STILL MEASURED (floats, labeled): the level statistics of the
DISTINCT spectrum of A - the Farey/Chebyshev set - against Poisson / GOE /
GUE, because that is the one honest statistical question left: the distinct
levels are neither a product spectrum nor random.

Usage: python nontensor_spec.py            # machines 11..23
       python nontensor_spec.py --big      # adds machine 29 gap histogram
"""
import sys
from math import prod, pi, cos, log, gcd
import numpy as np

RT_POISSON = 2 * log(2) - 1          # 0.386294
RT_GOE = 0.53590
RT_GUE = 0.60266

F_KNOWN = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}


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


def gap_hist(gears):
    b = blocked(gears)
    idx = np.flatnonzero(~b)
    P = b.size
    g = np.diff(np.append(idx, idx[0] + P))
    assert g.sum() == P
    return np.bincount(g), idx.size, P


def totient(n):
    r, m = n, n
    p = 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            r -= r // p
        p += 1
    if m > 1:
        r -= r // m
    return r


# ------------------------------------------------------------ part A
def partA():
    print("\n=== PART A: path-decomposition theorem, verified exactly =====")
    gears = primes(5, 11)
    P = prod(gears)
    b = blocked(gears)
    # A = BS + (BS)^T : edge {k, k+1} iff k+1 blocked
    A = np.zeros((P, P))
    k = np.arange(P)
    nxt = (k + 1) % P
    e = b[nxt]
    A[k[e], nxt[e]] = 1.0
    A[nxt[e], k[e]] = 1.0
    spec = np.linalg.eigvalsh(A)
    h, nopen, _ = gap_hist(gears)
    pred = []
    for g in range(1, h.size):
        if h[g]:
            pred += [2 * cos(pi * j / (g + 1)) for j in range(1, g + 1)] * h[g]
    pred = np.sort(np.array(pred))
    assert pred.size == P, (pred.size, P)
    assert np.abs(pred - spec).max() < 1e-9, np.abs(pred - spec).max()
    print(f"  machine 11 (P = {P}): dense eigvalsh vs path prediction - "
          f"max |diff| = {np.abs(pred - spec).max():.2e}   THEOREM VERIFIED")
    print(f"  (gap histogram: {ded(h)})")
    # combinatorial verification of the decomposition at bigger machines
    for y in (13, 17, 19, 23):
        gs = primes(5, y)
        h, nopen, Pp = gap_hist(gs)
        tot = sum(int(h[g]) * g for g in range(h.size))
        assert tot == Pp, (y, tot, Pp)
        assert int(h.sum()) == nopen
        F = h.size - 1
        assert F == F_KNOWN[y], (y, F, F_KNOWN[y])
        print(f"  machine {y:2d}: {int(h.sum()):>12,} paths, "
              f"sum of lengths = P = {Pp:,}, longest path F = {F}  OK")


def ded(h):
    return ", ".join(f"{g}:{int(h[g])}" for g in range(h.size) if h[g])


# ------------------------------------------------------------ part B
def partB(ys):
    print("\n=== PART B: how many DISTINCT levels does the sector have? ===")
    print("   y     F      P            distinct levels   ties/level    "
          "distinct/P")
    for y in ys:
        F = F_KNOWN[y]
        P = prod(primes(5, y))
        # distinct = |Farey(F+1)| - 2 = sum_{b=2}^{F+1} phi(b)
        nd = sum(totient(bb) for bb in range(2, F + 2))
        # cross-check by direct construction
        S = set()
        for g in range(1, F + 1):
            for j in range(1, g + 1):
                a, bq = j, g + 1
                d = gcd(a, bq)
                S.add((a // d, bq // d))
        assert len(S) == nd, (y, len(S), nd)
        print(f"  {y:3d}  {F:4d}  {P:>14,}   {nd:12,}   {P / nd:12,.0f}   "
              f"{nd / P:.3e}")
    print("  ASSERTED: distinct-level count = sum_{b<=F+1} phi(b) exactly.")
    print("  A GUE spectrum of size P would have P distinct levels and NO")
    print("  ties; this sector has O(F^2) levels and P/F^2 ties on each.")


# ------------------------------------------------------------ part C
def rtilde(x):
    s = np.diff(np.sort(x))
    s = s[s > 0]
    r = s[1:] / s[:-1]
    return float(np.minimum(r, 1 / r).mean())


def unfold_farey(F):
    """the distinct spectrum of A at max gap F, unfolded by its own smooth
    density (2 cos(pi x) of the Farey set: unfolding = pull back to x)."""
    xs = sorted({(j / (g + 1)) for g in range(1, F + 1)
                 for j in range(1, g + 1)})
    return np.array(xs)


def partC(ys):
    print("\n=== PART C: level statistics of the DISTINCT (Farey/Chebyshev)"
          " spectrum ==")
    print("  floats, labeled.  <r~>: Poisson 0.38629, GOE 0.53590, "
          "GUE 0.60266")
    print("   y     F   #levels   <r~> (in x)   <r~> (in 2cos)   "
          "min s/mean s   P(s<0.1)   verdict")
    for y in ys:
        F = F_KNOWN[y]
        xs = unfold_farey(F)
        lev = 2 * np.cos(pi * xs)
        rx, rl = rtilde(xs), rtilde(lev)
        s = np.diff(np.sort(xs))
        hard = float(s.min() / s.mean())
        frac = float((s < 0.1 * s.mean()).mean())
        assert hard > 3 / pi ** 2 - 1e-9, (y, hard)   # Hall's hard gap
        v = ("Poisson-like" if abs(rx - RT_POISSON) < 0.05 else
             "GOE-like" if abs(rx - RT_GOE) < 0.05 else
             "GUE-like" if abs(rx - RT_GUE) < 0.05 else
             "MORE RIGID THAN GUE" if rx > RT_GUE else "NEITHER")
        print(f"  {y:3d}  {F:4d}  {xs.size:7,}    {rx:.5f}       "
              f"{rl:.5f}       {hard:.5f}      {frac:.4f}    {v}")
    print(f"  ASSERTED: min spacing / mean spacing > 3/pi^2 = "
          f"{3 / pi**2:.5f} at every machine -")
    print("  a HARD GAP.  Farey spacings follow Hall's distribution, whose")
    print("  support starts at 3/pi^2 of the mean: the distinct spectrum of")
    print("  the non-tensor sector is MORE RIGID THAN GUE, not less.  <r~> ~")
    print("  0.70 sits ABOVE the GUE value 0.6027, on the way to the clock")
    print("  value 1.  So GUE is bracketed a THIRD time and again not hit:")
    print("     clock 1.000  >  Farey/Chebyshev 0.70  >  GUE 0.603  >")
    print("     GOE 0.536  >  Poisson 0.386 = the machine's tensor sector.")


# ------------------------------------------------------------ part D
def partD():
    print("\n=== PART D: the other two non-tensor operators ===============")
    gears = primes(5, 13)
    P = prod(gears)
    b = blocked(gears)
    # nilpotent BS: spectrum {0}, index F
    v = np.ones(P, bool)
    n = 0
    while v.any():
        n += 1
        v = v & np.roll(b, -(n - 1))
    print(f"  (BS)^n = diag(v_n) S^n: nilpotent, index {n} = F(13) - "
          f"spectrum {{0}} with multiplicity P = {P:,} at every depth.")
    assert n == F_KNOWN[13]
    # generic 0/1 diag times shift, symmetrised -> paths + cycles
    rng = np.random.default_rng(0)
    for t in (1, 2, 3):
        w = b
        # graph on Z_P with edges {k, k+t} when w[k+t]; degree <= 2 always
        deg = np.zeros(P, np.int64)
        np.add.at(deg, np.arange(P)[w[(np.arange(P) + t) % P]], 1)
        np.add.at(deg, ((np.arange(P) + t) % P)[w[(np.arange(P) + t) % P]], 1)
        assert deg.max() <= 2, (t, deg.max())
    print("  diag(w) S^t + h.c. has max degree 2 for every t (asserted "
          "t = 1,2,3): always a union of paths and cycles, so its spectrum")
    print("  is always inside {2cos(pi a/b)} u {2cos(2 pi a/b)}.")
    print("  word-level H: block-triangular by word length (matrix-"
          "formulation piece 6), eigenvalues = diagonal = INTEGERS")
    print("  q' - #distinct residues, at most 2j+2 values per length block.")


def main():
    big = "--big" in sys.argv
    print(__doc__.split("Usage:")[0])
    partA()
    ys = [11, 13, 17, 19, 23] + ([29, 31, 37] if big else [])
    partB(ys)
    partC(ys)
    partD()
    print("\n=== VERDICT ==================================================")
    print("  The non-tensor sector cannot carry GUE.  Its Hermitian")
    print("  operators are unions of paths (Farey/Chebyshev spectra, O(F^2)")
    print("  distinct levels, P/F^2-fold ties); its high-Schmidt-rank")
    print("  operators are nilpotent (spectrum {0}); its cross-machine")
    print("  transfer matrix is triangular with an integer diagonal.")
    print("  Combined with round 21 (tensor operators -> Poisson by")
    print("  construction), NO operator of this machine is GUE, and the")
    print("  reason is structural: richness of spectrum and failure to")
    print("  factorise are mutually exclusive here.")


if __name__ == "__main__":
    main()
