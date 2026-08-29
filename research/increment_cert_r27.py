"""ROUND 27, LP-DUALITY THREAD - CAN THE VEHICLE CERTIFY THE INCREMENT LAW?

THE MANAGER'S HYPOTHESIS (round-26 derivation block):

    F(M + q') - F_2(M)  <=  s_min(q') = min(2u' mod q', -2u' mod q')

at every LITERAL step, where u' is gear q''s tooth (u' = 6^{-1} mod q').

THE HALF A DUAL CERTIFICATE CAN CARRY is the upper half - "no legal window of
machine M+q' exceeds F_2(M) + s_min(q')" - and that is EXACTLY the case-split
vehicle run at the INCREMENT WIDTH

    W_inc(M -> q') = F_2(M) + s_min(q')

instead of at the ladder's budget width F(M) + q'.  W_inc is strictly smaller at
every step (22 vs 28, 31 vs 37, 39 vs 48, 49 vs 63, 65 vs 74), so this is a
STRICTLY HARDER obligation than the (D) rung the same vehicle already certifies,
and it is not implied by it.

WHAT THIS DOES NOT DO, stated up front: the other half of the increment law -
F_2(M) >= W_inc - s_min, i.e. that the two-gap record really is that large - is a
REALISABILITY statement.  A dual certificate cannot carry it; it needs an
exhibited configuration.  Those are emitted here too (`witness_f2`), as explicit
phase vectors checked by CRT arithmetic with no period scan, so the pair
"certificate + witness" closes the literal-step statement without a scan
anywhere.

    python research/increment_cert_r27.py TABLE
    python research/increment_cert_r27.py RUN [step-index ...]
    python research/increment_cert_r27.py GATE
"""
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import star_case                                              # noqa: E402
from star_case import RelaxStar, decide_star, reverify_cert   # noqa: E402
from lp_degree_range import gears_of, teeth, budget, F_EXACT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R27 = os.path.join(HERE, 'data', 'r27')

# F_2(M): the largest ADJACENT GAP PAIR SUM of machine M.
#   11,13,17,19,23 - full-period scan (`window_dict.true_pairs`), exact;
#   29             - project record (round 26 summary), exact;
#   19, 23         - ALSO LP-certified scan-free by the windowed vehicle
#                    (round 26): F_2(19) <= 31 and F_2(23) <= 39, both tight.
F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 41: 103, 53: 159}


def smin(q):
    u = teeth(q)[0]
    return min((2 * u) % q, (-2 * u) % q)


STEPS = [(11, 13), (13, 17), (17, 19), (19, 23), (23, 29), (29, 31)]


def w_inc(M, q):
    return F2[M] + smin(q)


def table():
    print("  step    q'   u'  s_min   F_2(M)   W_inc   F(M+q')  budget"
          "   W_inc<budget")
    for M, q in STEPS:
        W = w_inc(M, q)
        print("  %2d->%-2d  %3d  %3d  %5d  %7d  %6d  %8d  %6d   %s"
              % (M, q, q, teeth(q)[0], smin(q), F2[M], W, F_EXACT[q],
                 budget(q), W < budget(q)))


# ------------------------------------------------------------------ witnesses
def witness_f2(M):
    """An EXPLICIT configuration of machine M realising an adjacent gap pair of
    sum F_2(M) - the lower half of the increment law, with no period scan: a
    phase vector r, and the assertion (checked by CRT arithmetic on [0, s]) that
    positions 0, a, s are open and every other position of (0, s) is blocked.

    Found by an exact-cover BACKTRACK over the gears' phases (no period scan):
    fix the open split (a, s-a), then cover [1, s-1] minus {a} by choosing, for
    the smallest still-uncovered position, a gear-phase that covers it and
    covers none of 0, a, s.  The OBJECT is what matters, and the object is
    re-checked from its own numbers by `check_witness`."""
    from lp_degree_range import hits
    g = gears_of(M)
    s = F2[M]
    span = s + 1
    for a in range(1, s):
        need = frozenset(set(range(1, s)) - {a})
        keep = (0, a, s)
        opts = {}
        for q in g:
            opts[q] = [(r, frozenset(hits(q, r, span)))
                       for r in range(q)
                       if not any(p in hits(q, r, span) for p in keep)]

        def rec(covered, avail, chosen):
            if covered >= need:
                return chosen
            p = min(need - covered)
            for i, q in enumerate(avail):
                for (r, h) in opts[q]:
                    if p in h:
                        out = rec(covered | h, avail[:i] + avail[i + 1:],
                                  chosen + [(q, r)])
                        if out is not None:
                            return out
            return None

        got = rec(frozenset(), tuple(g), [])
        if got is None:
            continue
        ph = dict(got)
        for q in g:                      # unused gears: any admissible phase
            if q not in ph:
                ph[q] = opts[q][0][0]
        r = tuple(ph[q] for q in g)
        blocked = set()
        for q, rq in zip(g, r):
            blocked |= set(hits(q, rq, span))
        openp = sorted(set(range(span)) - blocked)
        if openp == [0, a, s]:
            return dict(machine=M, phases=list(r), gears=list(g),
                        span=s, split=(a, s - a), openings=openp)
    return None


def check_witness(w):
    """Re-check a witness from its own numbers - CRT arithmetic only."""
    from lp_degree_range import hits
    g, r, s = tuple(w['gears']), tuple(w['phases']), w['span']
    assert g == gears_of(w['machine'])
    blocked = set()
    for q, rq in zip(g, r):
        blocked |= set(hits(q, rq, s + 1))
    openp = sorted(set(range(s + 1)) - blocked)
    assert openp == list(w['openings']), (openp, w['openings'])
    assert len(openp) == 3 and openp[0] == 0 and openp[-1] == s
    return True


# --------------------------------------------------------------- the run
def run_step(M, q, ks=(1, 2, 3), tb=600.0, verbose=True, stop_on_fail=True):
    """Certify F(machine q) <= W_inc(M -> q) by the case split, escalating k."""
    g = gears_of(q)
    W = w_inc(M, q)
    star_case.OUT = R27
    os.makedirs(R27, exist_ok=True)
    for k in ks:
        held = g[:k]
        cases = list(product(*[range(p) for p in held]))
        print("  %d->%d  W_inc = %d  hold %s (%d cases)"
              % (M, q, W, list(held), len(cases)), flush=True)
        ops, bad, t0 = 0, [], time.time()
        for ws in cases:
            tag = "inc_m%d_w%d_h%s" % (q, W, "_".join(map(str, ws)))
            R = RelaxStar(g, W, held, ws)
            v, info = decide_star(R, verbose=False, maxrounds=400, tag=tag,
                                  time_budget=tb)
            ops += info.get('ops') or 0
            if v != 'CERTIFIED':
                bad.append((ws, v, info.get('lp_max')))
                if verbose:
                    print("     case %s -> %s  lp=%s"
                          % (str(ws), v, info.get('lp_max')), flush=True)
            del R
            if bad and stop_on_fail:
                # one failing case already kills the case split at this k;
                # finishing the sweep buys nothing, so escalate immediately.
                break
        print("     k=%d: %d/%d certified, %d exact ops  [%.0fs]"
              % (k, len(cases) - len(bad), len(cases), ops, time.time() - t0),
              flush=True)
        if not bad:
            return dict(M=M, q=q, W=W, k=k, cases=len(cases), ops=ops,
                        secs=time.time() - t0, verdict='CERTIFIED')
    return dict(M=M, q=q, W=W, k=None, verdict='NOT CERTIFIED', bad=bad)


def gate():
    """Re-verify every increment certificate on disk from a clean rebuild."""
    star_case.OUT = R27
    t0 = time.time()
    print("=" * 78)
    print("GATE  increment-width certificates, re-verified from disk")
    print("=" * 78, flush=True)
    seen = {}
    for f in sorted(os.listdir(R27)):
        if not (f.startswith('cert_inc_m') and f.endswith('.pkl')):
            continue
        tag = f[len('cert_'):-4]
        reverify_cert(tag)
        key = tag.split('_h')[0]
        seen[key] = seen.get(key, 0) + 1
    for M, q in STEPS:
        W = w_inc(M, q)
        key = 'inc_m%d_w%d' % (q, W)
        if key in seen:
            print("  %d->%d at W_inc = %d : %d case certificates re-verified"
                  % (M, q, W, seen[key]), flush=True)
    for M in (11, 13, 17, 19, 23, 29):
        w = witness_f2(M)
        assert w is not None and check_witness(w)
        assert w['span'] == F2[M]
        print("  F_2(%d) >= %d WITNESSED: phases %s, openings %s (split %s)"
              % (M, F2[M], w['phases'], w['openings'], w['split']), flush=True)
    print("\n  ALL ASSERTIONS GREEN  [%.0fs]" % (time.time() - t0))


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'TABLE')
    if cmd == 'TABLE':
        table()
    elif cmd == 'RUN':
        idx = [int(x) for x in a[1:]] or list(range(len(STEPS)))
        out = []
        for i in idx:
            M, q = STEPS[i]
            out.append(run_step(M, q))
        print()
        for r in out:
            print("  %d->%d  W_inc %d : %s (k=%s, %s cases)"
                  % (r['M'], r['q'], r['W'], r['verdict'], r['k'],
                     r.get('cases')))
    elif cmd == 'GATE':
        gate()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
