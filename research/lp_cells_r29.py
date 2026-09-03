"""ROUND 29, LP-DUALITY THREAD - THE CELL DRIVER.

A resumable parallel driver over CELLS, in the round-28 shape but writing into
research/data/r29/.  A cell is (machine y, width W, held count k, held phases
ws); its verdict comes from `cutlimit_r28.limit_decide` - the LIFTED LP that
gives the cut loop's LIMIT V* directly, and then the EXACT verdict on whichever
side of |pos| that limit falls:

    V* <  |pos| (or the lifted polytope EMPTY)   CERTIFIABLE - and the lifted
        duals seed `decide_star`, which returns an exact rational DUAL
        CERTIFICATE (re-checked from its own integers);
    V* >= |pos|                                  ASYMPTOTE - no amount of cut
        generation can ever certify the cell, and the primal optimum is
        rationalised and verified IN THE POLYTOPE, i.e. an exact REFUTATION.

There is no cut loop in the method: a stall is never a verdict, and the float
LP only ever FINDS objects that exact rational arithmetic then decides.

PLANS (round-29 brief):
  K12    the SMALLEST-k question for 31->37 at the (D) budget width W = 95:
         every k = 2 case (35) and every k = 1 case (5).  Round 26 recorded
         k = 2 as a STALL (LP max 40.994 against 40), which is an undecided
         cell; this decides it.
  INC41  the increment width of 37 -> 41, W_inc = F_2(37) + s_min(41)
         = 90 + 14 = 104, at k = 3: all 385 cases.  This is round-28 E12's
         test at the next step AND, if it certifies, the upper half of the
         increment law one step past the six literal steps.
  RUNG10 the rung-ten increment width 43 -> 47, W_inc = F_2(43) + s_min(47)
         = 116 + 16 = 132, at the affordable k (5, then 4) - Constructor's
         spectrum-plus-depth bound at that step is the SAME number 132.
  WC     W_c(y, 3) = min{W : G < 0} at case 0, by bisection with the sign
         pattern asserted around the crossing (round-28 E9).

    uv run python research/lp_cells_r29.py K12    [workers]
    uv run python research/lp_cells_r29.py INC41  [workers]
    uv run python research/lp_cells_r29.py RUNG10 [workers]
    uv run python research/lp_cells_r29.py WC <y> <k> <lo> <hi>
    uv run python research/lp_cells_r29.py TABLE
"""
import json
import os
import sys
import time
from itertools import product
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
R29 = os.path.join(HERE, 'data', 'r29')

# ---------------------------------------------------------------- the numbers
# F_2(M), the largest ADJACENT GAP PAIR SUM of machine M.  11..29 are
# `increment_cert_r27.F2`; 37 is the project record (Constructor R68 witness
# table, Mechanic's m37 census); 43 is Mechanic's round-28 exact value.
F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 37: 90, 41: 103,
      43: 116, 53: 159}


def smin(q):
    from lp_degree_range import teeth
    u = teeth(q)[0]
    return min((2 * u) % q, (-2 * u) % q)


def w_inc(M, q):
    return F2[M] + smin(q)


def cellname(y, W, k, ws):
    return "cell_m%d_w%d_k%d_h%s" % (y, W, k, "_".join(str(x) for x in ws))


def run_one(job):
    """Decide one cell.

    FAST PATH (`fast=True`), and it is SOUND, not a shortcut: run the ordinary
    `decide_star` first.  If it returns CERTIFIED the cell carries an exact
    rational dual certificate and there is nothing left to decide - the lifted
    LP would only tell us the same thing more slowly.  MEASURED at machine 41,
    W = 104, k = 3: 18 s per cell against 180 s for lifted-LP-first, because
    every cell of that sweep certifies at ITERATION ZERO off the base cuts
    alone.  Anything that is NOT a certificate (a stall, NODUAL, a refutation)
    falls back to `cutlimit_r28.limit_decide` on a FRESH relaxation, which
    DECIDES the cell - so no cell is ever left on a stall."""
    y, W, k, ws, tb = job[:5]
    wit = job[5] if len(job) > 5 else True
    fast = job[6] if len(job) > 6 else False
    import star_case
    import cutlimit_r28
    from lp_degree_range import gears_of
    from star_case import RelaxStar, decide_star
    star_case.OUT = R29
    name = cellname(y, W, k, ws)
    p = os.path.join(R29, name + '.json')
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    t0 = time.time()
    g = gears_of(y)
    R = RelaxStar(g, W, g[:k], tuple(ws))
    out = dict(machine=y, W=W, k=k, ws=list(ws), cell=name,
               npos=len(R.pos), ncols=len(R.cols), frhs=int(R.frhs))
    try:
        if fast:
            v, info = decide_star(R, verbose=False, maxrounds=2, tag=name,
                                  time_budget=min(tb, 300.0))
            if v == 'CERTIFIED':
                out.update(method='plain-loop', side='CERTIFIABLE',
                           verdict='CERTIFIED', its=info['its'],
                           ops=info['ops'], secs=info['secs'],
                           lhs=str(info['lhs']), rhs=str(info['rhs']),
                           rows=info['rows'])
                out['total_secs'] = time.time() - t0
                with open(p, 'w') as fh:
                    json.dump(out, fh, indent=1)
                print("  %-30s %-12s %-17s frhs=%-4s %s  [%.0fs]"
                      % (name, 'CERTIFIABLE', 'plain-loop it=%d' % info['its'],
                         out['frhs'], 'CERTIFIED', out['total_secs']),
                      flush=True)
                return out
            out['plain_loop'] = v
            R = RelaxStar(g, W, g[:k], tuple(ws))     # fresh rows
        out['method'] = 'lifted'
        out.update(cutlimit_r28.limit_decide(R, tag=name, tb=tb, witness=wit))
    except Exception as e:                                   # noqa: BLE001
        # NEVER a verdict: an exception is an ERROR record, and the round-28
        # lesson stands - a non-optimal solver status is not a failure to
        # certify.
        out.update(verdict='ERROR', error=repr(e)[:400])
    out['total_secs'] = time.time() - t0
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print("  %-30s %-12s V*=%-11s frhs=%-4s %s  [%.0fs]"
          % (name, out.get('side', '-'),
             ('%.4f' % out['vstar']) if out.get('vstar') is not None
             else ('EMPTY' if out.get('empty') else '-'),
             out.get('frhs'), out.get('verdict'), out['total_secs']),
          flush=True)
    return out


def drive(jobs, workers=3):
    os.makedirs(R29, exist_ok=True)
    t0 = time.time()
    print("  %d cells on %d workers" % (len(jobs), workers), flush=True)
    with Pool(workers) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    print("  DONE %d cells  [%.0fs]" % (len(res), time.time() - t0), flush=True)
    return res


def cases(y, k):
    from lp_degree_range import gears_of
    held = gears_of(y)[:k]
    return [tuple(w) for w in product(*[range(q) for q in held])]


# ------------------------------------------------------------------ the plans
def jobs_k12():
    """31->37 at the budget width 95: is k = 3 really the smallest?"""
    return jobs_k2() + jobs_k1()


def jobs_k2():
    return [(37, 95, 2, ws, 1200.0, True) for ws in cases(37, 2)]


def jobs_k1():
    """n = 9 free gears - the memory-heavy end of the lifted LP (round 28
    measured n = 9 at 817 s / cell).  Run these FEW AT A TIME."""
    return [(37, 95, 1, ws, 5400.0, True) for ws in cases(37, 1)]


def jobs_inc41(k=3):
    W = w_inc(37, 41)
    return [(41, W, k, ws, 900.0, True, True) for ws in cases(41, k)]


def jobs_f41(W=91):
    """round-28 E10: is the FULL k = 3 case split tight on F at machine 41 -
    does it certify F(41) <= 91 and fail at 90?"""
    return [(41, W, 3, ws, 600.0, True, True) for ws in cases(41, 3)]


def gap_witness(M, s):
    """An explicit phase vector of machine M with 0 and s OPEN and every
    position of (0, s) BLOCKED - i.e. a realised gap of size s, F(M) >= s.

    WHY IT IS HERE.  A certificate at width W says "no fully blocked window of
    width W".  If F(M) >= W + 1 that statement is FALSE, so the case-split
    vehicle must fail in the case whose held phases are this witness's - and
    this locates that case EXACTLY, instead of sweeping for it.  Exact-cover
    backtrack over the gears, CRT arithmetic only, no period scan."""
    from lp_degree_range import hits, gears_of
    g = gears_of(M)
    span = s + 1
    need = frozenset(range(1, s))
    keep = (0, s)
    opts = {q: [(r, frozenset(hits(q, r, span))) for r in range(q)
                if not any(p in hits(q, r, span) for p in keep)] for q in g}

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
        return None
    ph = dict(got)
    for q in g:
        if q not in ph:
            ph[q] = opts[q][0][0]
    r = tuple(ph[q] for q in g)
    blocked = set()
    for q, rq in zip(g, r):
        blocked |= set(hits(q, rq, span))
    openp = sorted(set(range(span)) - blocked)
    assert openp == [0, s], openp
    return dict(machine=M, gears=list(g), phases=list(r), gap=s,
                openings=openp)


def refine_jobs(y, W, k, tb=900.0):
    """Every k-case on disk that did NOT certify, refined into its q_{k+1}
    children at k+1.  EXHAUSTIVE BY CONSTRUCTION: the children of a case are
    that case split on the next gear's phases, so the certified k-cases plus
    the children of the uncertified ones still partition every configuration.
    This is round 28's own move at machine 43 (case (0,0,0) at k = 3 split
    into its 13 k = 4 sub-cases), applied to a whole sweep."""
    from lp_degree_range import gears_of
    g = gears_of(y)
    bad = []
    for ws in cases(y, k):
        p = os.path.join(R29, cellname(y, W, k, ws) + '.json')
        if not os.path.exists(p):
            raise SystemExit("sweep incomplete: %s missing" % p)
        with open(p) as fh:
            if json.load(fh).get('verdict') != 'CERTIFIED':
                bad.append(tuple(ws))
    out = []
    for ws in bad:
        for v in range(g[k]):
            out.append((y, W, k + 1, ws + (v,), tb, True, True))
    print("  %d of %d k=%d cases did not certify; refining into %d k=%d cells"
          % (len(bad), len(cases(y, k)), k, len(out), k + 1), flush=True)
    return out


def jobs_e11(y=31, W=74, k=3):
    """round-28 E11: every cell whose LIFTED POLYTOPE IS EMPTY certifies at
    iteration zero once seeded.  Testing it needs cells decided BY THE LIFTED
    ROUTE (fast=False), which this round's big sweeps deliberately avoid
    because the plain loop is 10x cheaper - so the sample is generated here, at
    the cheapest machine that still has 385 cases (m31, k = 3, n = 6 free
    gears, the 29 -> 31 rung's own budget width)."""
    return [(y, W, k, ws, 600.0, True, False) for ws in cases(y, k)]


def jobs_rung10():
    W = w_inc(43, 47)
    out = [(47, W, 5, ws, 1800.0, True)
           for ws in [(0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (2, 3, 5, 7, 9)]]
    out += [(47, W, 4, ws, 3600.0, True)
            for ws in [(0, 0, 0, 0), (1, 2, 3, 4)]]
    return out


def table():
    rows = []
    for f in sorted(os.listdir(R29)):
        if not (f.startswith('cell_') and f.endswith('.json')):
            continue
        with open(os.path.join(R29, f)) as fh:
            rows.append(json.load(fh))
    print("  %-30s %-6s %-5s %-5s %-11s %-6s %s"
          % ('cell', 'side', 'npos', 'frhs', 'V*', 'G', 'verdict'))
    for r in rows:
        v = r.get('vstar')
        print("  %-30s %-6s %-5s %-5s %-11s %-6s %s"
              % (r['cell'], (r.get('side') or '-')[:6], r.get('npos'),
                 r.get('frhs'),
                 ('%.4f' % v) if v is not None else
                 ('EMPTY' if r.get('empty') else '-'),
                 ('%+.4f' % r['gap']) if r.get('gap') is not None else '-inf',
                 r.get('verdict')))
    print("  %d cells on disk" % len(rows))
    return rows


# ------------------------------------------------------------------------ W_c
def _G(y, k, W, ws=None):
    import cutlimit_r28
    from star_case import RelaxStar
    from lp_degree_range import gears_of
    ws = tuple(ws if ws is not None else [0] * k)
    g = gears_of(y)
    R = RelaxStar(g, W, g[:k], ws)
    L = cutlimit_r28.lifted(R)
    v = (None if L['val'] is None else L['val'] - float(R.frhs))
    return v, len(R.pos), L.get('secs')


def wc(y, k, lo, hi, ws=None):
    """min{W : G < 0} by bisection; the sign pattern around the crossing is
    then ASSERTED width by width, so single-crossing is checked, not assumed."""
    t0 = time.time()
    seen = {}

    def g(W):
        if W not in seen:
            seen[W] = _G(y, k, W, ws)
            print("      W=%-4d |pos|=%-4d G=%-10s [%.0fs]"
                  % (W, seen[W][1],
                     ('%+.4f' % seen[W][0]) if seen[W][0] is not None
                     else 'EMPTY(-inf)', seen[W][2] or 0.0), flush=True)
        return seen[W][0]

    glo, ghi = g(lo), g(hi)
    assert glo is not None and glo >= 0, "lo=%d is already certifiable" % lo
    assert ghi is None or ghi < 0, "hi=%d is not certifiable" % hi
    a, b = lo, hi
    while b - a > 1:
        m = (a + b) // 2
        if g(m) is None or g(m) < 0:
            b = m
        else:
            a = m
    W_c = b
    for W in range(max(lo, W_c - 3), min(hi, W_c + 3) + 1):
        v = g(W)
        neg = (v is None or v < 0)
        assert neg == (W >= W_c), \
            "sign pattern is not single-crossing at W=%d" % W
    out = dict(machine=y, k=k, W_c=W_c, lo=lo, hi=hi,
               ws=list(ws or [0] * k),
               widths={str(W): (None if seen[W][0] is None else seen[W][0])
                       for W in sorted(seen)},
               npos={str(W): seen[W][1] for W in sorted(seen)},
               secs=time.time() - t0)
    p = os.path.join(R29, 'wc_m%d_k%d.json' % (y, k))
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print("    W_c(%d, %d) = %d   (single crossing asserted)  [%.0fs]"
          % (y, k, W_c, out['secs']), flush=True)
    return out


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'TABLE')
    os.makedirs(R29, exist_ok=True)
    if cmd == 'K12':
        drive(jobs_k12(), int(a[1]) if len(a) > 1 else 3)
    elif cmd == 'K2':
        drive(jobs_k2(), int(a[1]) if len(a) > 1 else 2)
    elif cmd == 'K1':
        drive(jobs_k1(), int(a[1]) if len(a) > 1 else 1)
    elif cmd == 'INC41':
        drive(jobs_inc41(int(a[2]) if len(a) > 2 else 3),
              int(a[1]) if len(a) > 1 else 3)
    elif cmd == 'RUNG10':
        drive(jobs_rung10(), int(a[1]) if len(a) > 1 else 2)
    elif cmd == 'GAPWIT':
        w = gap_witness(int(a[1]), int(a[2]))
        print(json.dumps(w, indent=1))
        if w:
            p = os.path.join(R29, 'gapwit_m%s_s%s.json' % (a[1], a[2]))
            with open(p, 'w') as fh:
                json.dump(w, fh, indent=1)
            print("  held phases (5,7,11) = %s" % (tuple(w['phases'][:3]),))
    elif cmd == 'REFINE':
        drive(refine_jobs(int(a[1]), int(a[2]), int(a[3])),
              int(a[4]) if len(a) > 4 else 3)
    elif cmd == 'E11':
        drive(jobs_e11(), int(a[1]) if len(a) > 1 else 3)
    elif cmd == 'F41':
        drive(jobs_f41(int(a[2]) if len(a) > 2 else 91),
              int(a[1]) if len(a) > 1 else 3)
    elif cmd == 'CELL':
        y, W, k = int(a[1]), int(a[2]), int(a[3])
        ws = tuple(int(x) for x in a[4].split(',')) if len(a) > 4 else \
            tuple([0] * k)
        tb = float(a[5]) if len(a) > 5 else 1800.0
        print(json.dumps({kk: v for kk, v in run_one((y, W, k, ws, tb, True))
                          .items() if kk != 'traj'}, indent=1))
    elif cmd == 'WC':
        wc(int(a[1]), int(a[2]), int(a[3]), int(a[4]))
    elif cmd == 'WCALL':
        # round-28 E9: W_c(y, 3) at the all-zero case, cheapest machine first.
        for (y, lo, hi) in ((23, 10, 48), (29, 12, 63), (31, 16, 74),
                            (37, 40, 95), (41, 52, 104)):
            print("  W_c(%d, 3) in [%d, %d]" % (y, lo, hi), flush=True)
            try:
                wc(y, 3, lo, hi)
            except AssertionError as e:                       # noqa: BLE001
                print("    NOT BRACKETED / NOT SINGLE-CROSSING: %s" % e,
                      flush=True)
            except Exception as e:                            # noqa: BLE001
                print("    ERROR: %r" % (e,), flush=True)
    elif cmd == 'TABLE':
        table()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
