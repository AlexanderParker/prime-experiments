"""Round 20 (constructor): THE RENEWAL LADDER - a rigorous, closed-form (CRT)
upper bound on deep qualifying-run counts, converging to exact.

R32's exposure bound dropped the "no opening strictly between" condition and
fell short x2-x29 at the constrained cases (X20: that gap IS the renewal
factor).  This tool restores the condition rigorously, one interior point at
a time:

  A run of m consecutive qualifying gaps (v_1..v_m), v_i in V(q'), occupies
  openings at offsets X = {0, v_1, v_1+v_2, ..., sum v_i} and has NO opening
  at any interior offset.  For ANY chosen set Y of interior offsets, the run
  event implies
      W'(X, Y) = (all X exposed) AND (all Y blocked),
  and the number of slots k mod P satisfying W' is EXACT closed-form CRT
  arithmetic (inclusion-exclusion over exposed subsets of Y):
      #W' = sum over T subseteq Y of (-1)^|T| prod_q c_q(X u T),
  where c_q(O) = #{r mod q : r+o avoids both teeth, all o in O}.  Hence

      run_m(M)  <=  sum over qualifying tuples of  #W'(X(tuple), Y),

  monotone improving in |Y| and exact when Y = all interior points.  s = 0 is
  R32's exposure bound; s >= 1 is the renewal factor made rigorous.  A total
  below 1 is a ZERO CERTIFICATE: no qualifying run of depth m exists in the
  whole period - reachable at machines the period scan cannot touch (37+).

Assertions: every bound >= the exact census count (tm_resid_runs.csv) at
machines 19..31; each #W' >= 0; s-monotonicity.
"""
import os
import sys
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
from flank_envelope import primes_upto
from tm_resid_runs import next_prime
import csv

# machine y -> (F exactly known, lambda from R31 fit or None)
FEXACT = {19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
LAM = {19: 1.20, 23: 1.59, 29: 2.73}
FJ = {19: [25, 31, 35, 38, 47, 50], 23: [34, 39, 50, 58, 65, 77],
      29: [43, 55, 65, 70, 85, 90]}  # F_1..F_6 exact


def qual_values(q1, F):
    c = pow(6, -1, q1)
    Q = {0, (2 * c) % q1, (-2 * c) % q1}
    return [v for v in range(1, F + 1) if v % q1 in Q]


def count_Wprime(gears, X, Y):
    """Exact #{k mod P : all X exposed, all Y blocked} by CRT + IE over Y."""
    nY = len(Y)
    G = np.ones(1 << nY, dtype=np.int64)
    for q in gears:
        c = pow(6, -1, q)
        t1, t2 = c % q, (q - c) % q
        r = np.arange(q)
        expo = lambda o: ((r + o) % q != t1) & ((r + o) % q != t2)
        okX = np.ones(q, bool)
        for o in X:
            okX &= expo(o)
        mask = np.zeros(q, np.int64)
        for i, o in enumerate(Y):
            mask |= expo(o).astype(np.int64) << i
        cnt = np.bincount(mask[okX], minlength=1 << nY)
        # superset-sum: cnt[T] = # r with X exposed and all of T exposed
        # (in-place along axes of the 2^nY hypercube - no index temporaries)
        c = cnt.reshape((2,) * nY) if nY else cnt
        for ax in range(nY):
            sl0 = [slice(None)] * nY
            sl1 = [slice(None)] * nY
            sl0[ax], sl1[ax] = 0, 1
            c[tuple(sl0)] += c[tuple(sl1)]
        G *= cnt
    signs = np.array([(-1) ** bin(T).count("1") for T in range(1 << nY)],
                     dtype=np.int64)
    total = int((signs * G).sum())
    assert total >= 0, "IE count negative - bug"
    return total


def bisection_order(lo, hi):
    """Interior points of [lo, hi] in balanced-bisection (BFS) order, so that
    the first s points are NESTED as s grows."""
    out = []
    queue = [(lo, hi)]
    while queue:
        a, b = queue.pop(0)
        if a > b:
            continue
        mid = (a + b) // 2
        out.append(mid)
        queue.append((a, mid - 1))
        queue.append((mid + 1, b))
    return out


def pick_Y(tup, s):
    """First s interior points per gap in bisection order (nested in s);
    all interior points if the gap is short enough (that gap then exact)."""
    Y = []
    off = 0
    for v in tup:
        Y.extend(off + o for o in bisection_order(1, v - 1)[:s])
        off += v
    return sorted(set(Y))


def bound_runs(y, m, smax=3, show_tuples=False):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    q1 = next_prime(y)
    F = FEXACT[y]
    V = qual_values(q1, F)
    rho = prod(1 - 2 / q for q in gears)
    ngaps = round(P * rho)
    print(f"\n=== machine {y}  q'={q1}  F={F}  V={V}  m={m} "
          f"(j={m + 2})  tuples {len(V) ** m}")
    prev = None
    results = {}
    for s in range(0, smax + 1):
        tot = 0
        best = []
        for idx in range(len(V) ** m):
            tup, k = [], idx
            for _ in range(m):
                tup.append(V[k % len(V)])
                k //= len(V)
            X = [0]
            for v in tup:
                X.append(X[-1] + v)
            Y = pick_Y(tup, s)
            cW = count_Wprime(gears, X, Y)
            tot += cW
            if cW > 0:
                best.append((cW, tuple(tup)))
        if prev is not None:
            assert tot <= prev + 1e-9, "ladder not monotone - bug"
        prev = tot
        results[s] = tot
        pj = tot / ngaps
        tag = "  ZERO CERTIFICATE" if tot < 1 else ""
        print(f"  s={s}: run_{m} <= {tot:>14,}   p_{m + 2} <= {pj:.3e}"
              f"   surviving tuples {sum(1 for c, _ in best if c > 0)}{tag}")
        if show_tuples and s == smax:
            for cW, tup in sorted(best, reverse=True)[:8]:
                print(f"        {tup}: <= {cW:,}")
        if tot < 1:
            break
    # deep pass: push surviving tuples to s=5 where the IE stays <= 2^20
    if prev is not None and prev >= 1 and best:
        tot5 = 0
        capped = False
        for cW, tup in best:
            Y = pick_Y(tup, 5)
            if len(Y) > 20:
                tot5 += cW
                capped = True
                continue
            X = [0]
            for v in tup:
                X.append(X[-1] + v)
            tot5 += min(cW, count_Wprime(gears, X, Y))
        results["5*"] = tot5
        pj = tot5 / ngaps
        tag = "  ZERO CERTIFICATE" if tot5 < 1 else ""
        print(f"  s=5*: run_{m} <= {tot5:>13,}   p_{m + 2} <= {pj:.3e}"
              f"{'   (some tuples capped at s=3)' if capped else ''}{tag}")
    return results, ngaps


def main():
    exact = {}
    p = os.path.join(DDIR, "tm_resid_runs.csv")
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                exact[int(r["y"])] = r
    cases = [(19, 3), (19, 4), (23, 3), (23, 4), (29, 2), (29, 3), (29, 4)]
    if "--37" in sys.argv:
        cases += [(37, 3), (37, 4)]
    if "--31" in sys.argv:
        cases += [(31, 3), (31, 4)]
    for y, m in cases:
        res, ngaps = bound_runs(y, m, smax=3, show_tuples=(m >= 3))
        r = exact.get(y)
        final = min(res.values())
        if r and m <= 4:
            ex = int(r[f"run{m}"])
            for s, tot in res.items():
                assert tot >= ex, f"bound below exact! y={y} m={m} s={s}"
            print(f"  exact census: run_{m} = {ex:,}"
                  f"   (tightness at deepest: x{final / ex:.1f})"
                  if ex else f"  exact census: run_{m} = 0")
        # (D) requirement, where lambda + F_j known (R32 form)
        if y in LAM and m + 2 <= 6:
            need = (FJ[y][m + 1] - FJ[y][0] - next_prime(y)) / LAM[y]
            if need > 0:
                req = np.exp(-need)
                got = final / ngaps
                print(f"  (D) requirement p_{m + 2} <= {req:.3e}: "
                      f"ladder gives {got:.3e}  -> "
                      f"{'CLEARS' if got <= req else 'still short x%.1f' % (got / req)}")


if __name__ == "__main__":
    main()
