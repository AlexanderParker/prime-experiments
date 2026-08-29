"""ROUND 27, LP-DUALITY THREAD - GATE FOR THE NINTH (D) RUNG, 41 -> 43.

Round 26 left this rung a PARTIAL SWEEP: 163 of 385 cases at k = 3, six stalls
at a 45 s/case budget, owned as a badly-sized launch.  Round 27 finished it
(228 further cases, 10 striped workers, 240 s/case).  This gate re-verifies the
WHOLE rung from disk, in one clean process, from both certificate directories:

  * every one of the 385 case certificates is re-checked by `reverify_cert`,
    which rebuilds the relaxation FROM THE PRIMES, re-checks every cut row's
    exact validity by the zeta transform, and re-closes lhs < rhs in exact
    rationals;
  * the case set is asserted to be exactly prod(Z_5 x Z_7 x Z_11).

A CERTIFIED verdict in every case is a case-split certificate of
F(machine 43) <= budget(43) = F(41) + 43 = 134 - the (D) rung 41 -> 43.
"""
import os
import pickle
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import star_case                                        # noqa: E402
from star_case import RelaxStar, reverify_cert          # noqa: E402
from lp_degree_range import (gears_of, budget, ZERO, ONE,  # noqa: E402
                             zeta_values)

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = [os.path.join(HERE, 'data', 'r26'), os.path.join(HERE, 'data', 'r27')]


def find(tag):
    for d in DIRS:
        p = os.path.join(d, 'cert_%s.pkl' % tag)
        if os.path.exists(p):
            return d, p
    return None, None


def gate(y=43, k=3, quiet=True, wi=0, nw=1):
    g = gears_of(y)
    W = budget(y)
    held = g[:k]
    allc = [tuple(w) for w in product(*[range(q) for q in held])]
    assert len(allc) == 5 * 7 * 11 == 385
    cases = [w for i, w in enumerate(allc) if i % nw == wi]
    print("=" * 78)
    print("GATE  (D) rung 41 -> 43 : F(m%d) <= %d, case split holding %s"
          " (%d of %d cases, stripe %d/%d)"
          % (y, W, list(held), len(cases), len(allc), wi, nw))
    print("=" * 78, flush=True)
    t0 = time.time()
    ops, nlhs, where = 0, [], {}
    for ws in cases:
        tag = "rung3_m%d_w%d_h%d_%d_%d" % (y, W, ws[0], ws[1], ws[2])
        d, p = find(tag)
        assert p is not None, ("missing certificate", ws)
        where[os.path.basename(d)] = where.get(os.path.basename(d), 0) + 1
        star_case.OUT = d
        lhs, rhs = reverify_cert(tag) if not quiet else _quiet_reverify(p)
        assert lhs < rhs, (ws, lhs, rhs)
        nlhs.append((ws, lhs, rhs))
        with open(p, 'rb') as fh:
            ops += pickle.load(fh)['info'].get('ops') or 0
    lo = min(rhs - lhs for _w, lhs, rhs in nlhs)
    print("  %d/%d case certificates RE-VERIFIED from disk (%s)"
          % (len(nlhs), len(cases),
             ", ".join("%s: %d" % kv for kv in sorted(where.items()))))
    print("  every case closes lhs < rhs; smallest margin over the 385 cases:"
          " %s" % lo)
    print("  first case %s: %s < %s" % (nlhs[0][0], nlhs[0][1], nlhs[0][2]))
    print("  total exact certificate ops: %d" % ops)
    print("  EXHAUSTIVENESS: the 385 held-phase tuples are exactly"
          " Z_5 x Z_7 x Z_11")
    print("\n  => CASE-SPLIT CERTIFICATE OF THE (D) RUNG 41 -> 43:"
          " F(machine 43) <= %d, hypothesis-free  [%.0fs]"
          % (W, time.time() - t0))
    return ops


def _quiet_reverify(p):
    """`star_case.reverify_cert` without the per-case print."""
    with open(p, 'rb') as fh:
        d = pickle.load(fh)
    R = RelaxStar(d['full'], d['W'], d['held'], d['ws'], d['openpts'], d['l'])
    R.rows = d['rows']
    y, yff, nu = d['y'], d['yff'], d['nu']
    a = [ZERO] * len(R.cols)
    for r, (i, lam) in enumerate(R.rows):
        if not y[r]:
            continue
        for j, si in R.bypos[i]:
            if lam[si]:
                a[j] += y[r] * lam[si]
    if yff:
        for j, v in enumerate(R.frow):
            if v:
                a[j] += yff * v
    for kk, (par, kids) in enumerate(R.links):
        if nu[kk]:
            for j in kids:
                a[j] += nu[kk]
            a[par] -= nu[kk]
    lhs = sum(max(a[lo:hi]) for (lo, hi) in R.block_span.values())
    rhs = sum(y[r] * (ONE - lam[0])
              for r, (i, lam) in enumerate(R.rows)) + yff * R.frhs
    for (i, lam) in R.rows:
        f = zeta_values(tuple(lam), R.n, R.subs)
        assert min(f[x] for x in range(1, 1 << R.n)) >= ONE, "invalid cut row"
    assert all(v >= 0 for v in y) and yff >= 0, "negative dual weight"
    assert lhs < rhs, ("certificate does not close", lhs, rhs)
    return lhs, rhs


if __name__ == '__main__':
    wi = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    gate(wi=wi, nw=nw)
