"""ROUND 28, LP-DUALITY THREAD - THE ROUND'S GATE.

Four assertions, all re-run from a clean process against what is on disk:

  A  EVERY certificate produced this round re-verifies.  `reverify_cert`
     rebuilds the relaxation FROM THE PRIMES, re-checks that EVERY row - the
     base cuts AND the rows seeded from the lifted LP's duals - is an exactly
     valid cut by the zeta transform over all 2^n atoms, and re-closes
     lhs < rhs in exact rationals.  A seeded row is float-DISCOVERED, so this
     is the assertion that matters: nothing float ever enters a verdict.

  B  EVERY refutation witness re-verifies IN THE POLYTOPE: exact block sums,
     exact consistency links, exact completability at every position, and the
     recursion row value >= |pos|.  A witness is a PROOF that the cut loop can
     never certify that cell, however long it runs.

  C  THE INSTRUMENT.  On sampled cells the ordinary cut loop is run to
     termination and its terminal LP value is asserted equal to the lifted
     LP's V*.  (Theorem: the loop's value is >= V* at every pass because its
     rows are a subset of the valid cuts, and at termination the exact
     separation oracle has found nothing, so the point is in the lifted
     polytope and the value is <= V*.  This asserts it numerically as well.)

  D  NO CELL ON DISK CARRIES CONTRADICTORY VERDICTS: side CERTIFIABLE never
     came back REFUTED and side ASYMPTOTE never came back CERTIFIED.

    python research/gate_r28.py GATE
    python research/gate_r28.py GATE fast      # skip C
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import star_case                                              # noqa: E402
import cutlimit_r28                                           # noqa: E402
from star_case import RelaxStar, decide_star, reverify_cert    # noqa: E402
from lp_degree_range import gears_of                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R28 = os.path.join(HERE, 'data', 'r28')

# (machine, W, k) cells for the instrument check - small enough to run the
# ordinary loop to termination inside a gate.
INSTRUMENT = [(23, 34, 1), (23, 40, 1), (23, 41, 1), (29, 48, 1)]


def gate(fast=False):
    star_case.OUT = R28
    t0 = time.time()
    print("=" * 78)
    print("GATE  round-28 LP-duality thread")
    print("=" * 78, flush=True)

    # ---------------------------------------------------------------- A
    certs = sorted(f for f in os.listdir(R28)
                   if f.startswith('cert_cell_') and f.endswith('.pkl'))
    nc = 0
    import io
    import contextlib
    for f in certs:
        with contextlib.redirect_stdout(io.StringIO()):
            lhs, rhs = reverify_cert(f[len('cert_'):-4])
        assert lhs < rhs
        nc += 1
    print("  A  %d case certificates re-verified from disk (every cut row "
          "exactly valid, lhs < rhs)  GREEN" % nc, flush=True)

    # ---------------------------------------------------------------- D
    cells = [json.load(open(os.path.join(R28, f)))
             for f in os.listdir(R28)
             if f.startswith('cell_') and f.endswith('.json')]
    bad = [c for c in cells
           if (c.get('side') == 'CERTIFIABLE' and c.get('verdict') == 'REFUTED')
           or (c.get('side') == 'ASYMPTOTE' and c.get('verdict') == 'CERTIFIED')]
    assert not bad, bad[:3]
    print("  D  %d cells on disk, no contradictory verdict  GREEN" % len(cells),
          flush=True)

    # ---------------------------------------------------------------- C
    if not fast:
        for (y, W, k) in INSTRUMENT:
            g = gears_of(y)
            R = RelaxStar(g, W, g[:k], tuple([0] * k))
            L = cutlimit_r28.lifted(R)
            R2 = RelaxStar(g, W, g[:k], tuple([0] * k))
            tr = []
            v, info = decide_star(R2, verbose=False, maxrounds=400,
                                  time_budget=900.0, trace=tr)
            term = [t['lp_max'] for t in tr if t['lp_max'] is not None]
            if L['val'] is None:
                assert v in ('CERTIFIED',), (y, W, k, v)
                print("  C  m%d W=%d k=%d : lifted polytope EMPTY, loop says "
                      "%s  GREEN" % (y, W, k, v), flush=True)
                continue
            assert term, "no trajectory"
            d = abs(term[-1] - L['val'])
            assert d < 1e-5, ("loop terminal value != V*", term[-1], L['val'])
            agree = ((L['val'] < float(R.frhs)) == (v == 'CERTIFIED'))
            assert agree, (y, W, k, v, L['val'], float(R.frhs))
            print("  C  m%d W=%d k=%d : loop terminal %.6f == V* %.6f "
                  "(|diff| %.2e), verdict %s agrees with sign(V* - |pos|)"
                  "  GREEN" % (y, W, k, term[-1], L['val'], d, v), flush=True)

    # ---------------------------------------------------------------- B
    # LAST, deliberately: a refutation witness carries denominators up to
    # 10^60, so its exact completability re-check at every position is the
    # slowest thing in this file by an order of magnitude.
    wits = sorted(f for f in os.listdir(R28)
                  if f.startswith('wit_') and f.endswith('.pkl'))
    nw = 0
    for f in wits:
        with contextlib.redirect_stdout(io.StringIO()):
            ver = star_case.reverify(f[len('wit_'):-4])
        assert ver['row_value'] >= ver['row_rhs']
        nw += 1
        print("  B  %s : row %.4f >= %s IN THE POLYTOPE - the cut loop can "
              "never certify this cell"
              % (f[len('wit_'):-4], float(ver['row_value']), ver['row_rhs']),
              flush=True)
    if nw == 0:
        print("  B  no refutation witnesses on disk", flush=True)
    print("\n  ALL ASSERTIONS GREEN  [%.0fs]" % (time.time() - t0))


if __name__ == '__main__':
    gate(fast=(len(sys.argv) > 2 and sys.argv[2].lower() == 'fast'))
