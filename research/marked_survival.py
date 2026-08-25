"""Round 23 (constructor): AUDIT OF THE MARKED QUALIFYING SPECTRUM Q^[J] at
23 -> 29, and the exact Q_J(new) recomputed FROM THE OLD MACHINE by phases.

Why.  Round 22 reported two independent failures at one step: my bounded-state
Kleene certificates (99/99/91 against budget 74 at 29 -> 31) and Mechanic's
marked qualifying spectrum (Q^[5](23) = 85 against the same budget, true 71).
research/kleene_history.py settles my side (a three-gap-history state is EXACT
at that step).  This script audits the other side, because a sound relaxation
CANNOT report 85 there.  The argument, then the computation:

  SANDWICH LEMMA.  Fix a phase phi.  Let a relaxed window x_0 < ... < x_m of
  OLD openings have marked set M (|M| = J-1) with every unmarked interior
  killed, and consecutive marked distances >= a.  Let S = surviving interiors;
  by definition S is a SUBSET of M, so consecutive members of S are also at
  distance >= a.  Let s^- be the largest survivor <= x_0 and s^+ the smallest
  survivor >= x_m.  The survivors in [s^-, s^+] are exactly {s^-} u S u {s^+},
  so that is a NEW-machine window of |S| + 1 gaps whose middle distances all
  clear the floor, and its span is >= x_m - x_0.  Hence

      Q_J(new) <= Q^[J](old) <= max_{1 <= j <= J} Q_j(new).

  In particular, wherever Q_j(new) is non-decreasing up to J, Q^[J](old) is
  FORCED to equal Q_J(new): the relaxation cannot lose anything at all.

Three independent computations here, all exact and full period:

  (1) Q_j(29; 10) recomputed FROM MACHINE 23 by phase decomposition (machine
      29's openings in copy j of machine 23's period are exactly the old
      openings not killed by gear 29 at the corresponding phase), asserted
      equal to the direct machine-29 scan values from kleene_history.py.
  (2) THE SURVIVOR-COUNT BOUND: max span, over every phase and every start,
      of a window carrying at most J-1 SURVIVING interiors.  It ignores the
      floor and the marked choice entirely, so it dominates every version of
      Q^[J](old), correct or not.  If that number is below 85 the reported
      value cannot be right whatever the marked bookkeeping does.
  (3) The exact reason: an explicit count of how many openings gear q' can
      kill inside a window of a given span (at most 2*ceil(span/q')).

Usage: uv run python research/marked_survival.py [old q' q'']
       (default 23 29 31; also runs 19 23 29 as the control)
"""
import sys
import time
from math import prod

import numpy as np

# exact Q_J(M; 2u') from the direct full-period scans (kleene_history.py logs)
DIRECT_QJ = {
    13: [16, 18, 23, 0, 0, 0, 0],
    17: [25, 28, 31, 32, 34, 0, 0],
    19: [31, 35, 37, 38, 0, 0, 0],
    23: [39, 43, 50, 55, 60, 0, 0],
    29: [55, 65, 68, 71, 71, 71, 0],
}


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def openings(y):
    gears = primes(5, y)
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    return np.flatnonzero(~ex).astype(np.int64), P


def qspec_from_gaps(g, a, Jmax):
    """Q_j = max sum of j consecutive gaps whose j-2 middle gaps are >= a."""
    out = []
    n = len(g)
    for J in range(2, Jmax + 1):
        if n < J + 2:
            out.append(0)
            continue
        ok = np.ones(n - J, bool)
        tot = g[:n - J].astype(np.int64)
        for t in range(1, J):
            if t <= J - 2:
                ok &= g[t:n - J + t] >= a
            tot = tot + g[t:n - J + t]
        tot = np.where(ok, tot, -1)
        out.append(int(max(0, tot.max())))
    return out


def audit(old, qp, qpp, Jmax=7):
    t0 = time.time()
    op, P = openings(old)
    n = len(op)
    new = qp
    a = 2 * round(qpp / 6)
    c = pow(6, -1, qp)
    res = (op % qp).astype(np.int32)
    print("\n=== step %d -> %d  (floor a = 2u''(%d) = %d), machine %d has "
          "%d openings in period %d" % (old, new, qpp, a, old, n, P))

    # ---------- (1) exact Q_j(new; a) from the old machine, by phase --------
    WRAP = 4000
    ext = np.concatenate([op, op[:WRAP] + P])
    rext = (ext % qp).astype(np.int32)
    best = np.zeros(Jmax - 1, np.int64)
    for phi in range(qp):
        k1, k2 = (c + phi) % qp, (-c + phi) % qp
        surv = (rext != k1) & (rext != k2)
        arr = ext[surv]
        g = np.diff(arr)
        vals = qspec_from_gaps(g, a, Jmax)
        best = np.maximum(best, np.array(vals, np.int64))
    got = [int(v) for v in best]
    print("  (1) Q_J(%d; %d) from machine %d by phases, J = 2..%d : %s"
          % (new, a, old, Jmax, got))
    if new in DIRECT_QJ:
        exp = DIRECT_QJ[new][:Jmax - 1]
        print("      direct full-period machine-%d scan               : %s"
              % (new, exp))
        assert got == exp, (got, exp)
        print("      PHASE DECOMPOSITION AGREES WITH THE DIRECT SCAN")

    # ---------- (2) survivor-count bound: dominates every Q^[J] ------------
    # a window may carry at most J-1 SURVIVING interiors; maximise its span
    print("  (2) survivor-count bound (floor and marked choice ignored, so it "
          "dominates Q^[J](%d) however the marking is done):" % old)
    surv_bound = {}
    for phi in range(qp):
        k1, k2 = (c + phi) % qp, (-c + phi) % qp
        surv = np.flatnonzero((rext != k1) & (rext != k2))   # indices in ext
        # first survivor index strictly greater than i, for every i
        t = np.searchsorted(surv, np.arange(n), side="right")
        for J in range(2, Jmax + 1):
            j = t + (J - 1)
            good = j < len(surv)
            idx = np.where(good, surv[np.minimum(j, len(surv) - 1)], 0)
            span = np.where(good, ext[idx] - op, 0)
            v = int(span.max())
            if v > surv_bound.get(J, 0):
                surv_bound[J] = v
    print("      J          : %s" % "  ".join("%4d" % J
                                              for J in range(2, Jmax + 1)))
    print("      bound      : %s" % "  ".join("%4d" % surv_bound[J]
                                              for J in range(2, Jmax + 1)))
    print("      Q_J(new)   : %s" % "  ".join("%4d" % got[J - 2]
                                              for J in range(2, Jmax + 1)))
    for J in range(2, Jmax + 1):
        assert surv_bound[J] >= got[J - 2], (J, surv_bound[J], got[J - 2])

    # ---------- (3) the arithmetic reason -----------------------------------
    print("  (3) gear %d can kill at most 2*ceil(span/%d) openings in a window "
          "of that span:" % (qp, qp))
    for span in (55, 71, 74, 85):
        print("      span %3d -> at most %2d killed openings; machine-%d "
              "windows of that span hold about %.1f openings"
              % (span, 2 * -(-span // qp), old, span * n / P + 1))
    print("  (%.0f s)" % (time.time() - t0))
    return got, surv_bound


def main():
    args = [int(x) for x in sys.argv[1:]]
    steps = [(19, 23, 29), (23, 29, 31)] if not args else [tuple(args)]
    for old, qp, qpp in steps:
        got, sb = audit(old, qp, qpp)
        budget = DIRECT_QJ[old][0] // 1 if False else None
    print("\nall assertions passed")


if __name__ == "__main__" and "--corrected" not in sys.argv and "--2329" not in sys.argv:
    main()


# ---------------------------------------------------------------------------
# THE CORRECTED MARKED SPECTRUM
#
# research/marked_qspec.py's feasibility search returns True as soon as the
# marked quota J-1 is filled, without checking that the SURVIVING interiors
# still to come are marked too.  Since an unmarked interior is required to be
# killed, that admits windows the definition forbids, and the reported value
# is then not an upper bound on anything.  Fixed below: the recursion may only
# succeed after every interior has been decided.
# ---------------------------------------------------------------------------
from functools import lru_cache                                   # noqa: E402


def feasible_correct(iv, forced, need, a):
    """Is there M subset of the interiors with |M| = need, M containing every
    FORCED (surviving) interior, and consecutive members of M at distance
    >= a?"""
    n = len(iv)

    @lru_cache(maxsize=None)
    def rec(idx, cnt, last):
        if idx == n:
            return cnt == need
        if not forced[idx] and rec(idx + 1, cnt, last):
            return True
        if cnt < need and (last is None or iv[idx] - last >= a):
            if rec(idx + 1, cnt + 1, iv[idx]):
                return True
        return False

    ok = rec(0, 0, None)
    rec.cache_clear()
    return ok


def marked_spectrum_correct(old, qp, qpp, Jmax=5, span_cap=200, seed=0,
                            verbose=True):
    """Same object as marked_qspec.marked_spectrum, with the feasibility bug
    fixed and the 29-bin maximum tracked incrementally instead of by
    numpy.max on every step (semantics identical, ~10x faster)."""
    op, P = openings(old)
    n = len(op)
    up = pow(6, -1, qp)
    a = 2 * round(qpp / 6)
    ext = np.concatenate([op, op[:600] + P])
    rext = [int(x) % qp for x in ext]
    extl = [int(x) for x in ext]
    best = {J: seed for J in range(2, Jmax + 1)}
    bw = {J: None for J in range(2, Jmax + 1)}
    t0 = time.time()
    for i in range(n):
        cov = [0] * qp
        best_cov = 0
        n_int = 0
        for m in range(1, 600):
            span = extl[i + m] - extl[i]
            if span > span_cap:
                break
            if m >= 2:
                r = rext[i + m - 1]
                for cc in ((r - up) % qp, (r + up) % qp):
                    cov[cc] += 1
                    if cov[cc] > best_cov:
                        best_cov = cov[cc]
                n_int = m - 1
            if n_int - best_cov > Jmax - 1:
                break
            for J in range(2, Jmax + 1):
                if n_int < J - 1 or span <= best[J]:
                    continue
                if n_int - best_cov > J - 1:
                    continue
                iv = tuple(extl[i + 1:i + m])
                for c in range(qp):
                    if n_int - cov[c] > J - 1:
                        continue
                    k1, k2 = (c - up) % qp, (c + up) % qp
                    forced = tuple(rext[i + 1 + t] not in (k1, k2)
                                   for t in range(n_int))
                    if sum(forced) > J - 1:
                        continue
                    if feasible_correct(iv, forced, J - 1, a):
                        best[J] = span
                        bw[J] = (extl[i], extl[i + m], span, c)
                        break
    if verbose:
        print("      corrected scan over %d openings of machine %d in %.0f s"
              % (n, old, time.time() - t0))
    return best, bw, a


def corrected_check():
    """Corrected Q^[J](old) against the exact Q_J(new) at every step where
    both are computable, plus the sandwich the lemma predicts."""
    print("\n=== CORRECTED MARKED SPECTRUM  Q^[J](old)  vs  exact Q_J(new)")
    for old, qp, qpp, Jmax in ((11, 13, 17, 5), (13, 17, 19, 5),
                               (17, 19, 23, 5), (19, 23, 29, 7)):
        best, bw, a = marked_spectrum_correct(old, qp, qpp, Jmax=Jmax)
        got = [best[J] for J in range(2, Jmax + 1)]
        exact = DIRECT_QJ[qp][:Jmax - 1]
        ub = [max(exact[:J - 1]) for J in range(2, Jmax + 1)]
        print("  %2d -> %-2d  (floor %d)  J = 2..%d" % (old, qp, a, Jmax))
        print("      corrected Q^[J](%2d)      = %s" % (old, got))
        print("      exact     Q_J(%2d)        = %s" % (qp, exact))
        print("      sandwich  max_{j<=J} Q_j = %s" % ub)
        for t, J in enumerate(range(2, Jmax + 1)):
            assert got[t] >= exact[t], ("relaxation broken", old, J)
            assert got[t] <= ub[t], ("sandwich violated", old, J, got[t],
                                     ub[t])
        assert max(got) == max(exact), ("max over J differs", old, got, exact)
        print("      max over J: corrected %d == exact %d   (the criterion "
              "value is EXACT)" % (max(got), max(exact)))
    print("\ncorrected-marked-spectrum assertions passed")


if __name__ == "__main__" and "--corrected" in sys.argv:
    corrected_check()


def corrected_2329():
    """The step where marked_qspec.py reported the failure.  Seeded at 70, so
    the phase search only fires on windows that would beat the exact
    max_J Q_J(29; 10) = 71.  By the sandwich lemma nothing can."""
    print("\n=== CORRECTED Q^[J](23) for the step 23 -> 29 (q'' = 31), "
          "seeded at 70")
    best, bw, a = marked_spectrum_correct(23, 29, 31, Jmax=7, seed=70)
    got = [best[J] for J in range(2, 8)]
    exact = DIRECT_QJ[29][:6]
    print("      floor a = %d ; corrected Q^[J](23), J = 2..7 (seeded 70) = %s"
          % (a, got))
    print("      exact Q_J(29; 10)                                       = %s"
          % exact)
    print("      research/marked_qspec.py reported                       = "
          "[55, 65, 68, 85, 73, 73]")
    print("      witnesses above the seed: %s"
          % {J: bw[J] for J in bw if bw[J]})
    assert max(got) <= 71, ("SANDWICH LEMMA VIOLATED at 23 -> 29", got)
    print("      NOTHING above 71 exists: max_J Q^[J](23) = 71 <= budget 74")
    print("\ncorrected 23 -> 29 assertions passed")


if __name__ == "__main__" and "--2329" in sys.argv:
    corrected_2329()
