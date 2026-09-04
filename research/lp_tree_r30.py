"""ROUND 30, LP-DUALITY THREAD - THE MIXED-k TREE DRIVER, MIRROR-HALVED.

One driver per output directory (research/data/r30/), results written FROM
THE CHILD one JSON per cell, workers at High priority, resumable per cell.

THE OBJECT.  A cell is (machine y, width W, held count k, held phases ws).
Its verdict comes from `star_case.decide_star` on `RelaxStar` - the composed
level-2 covering relaxation over the FREE gears on the position set the held
gears leave uncovered - and, where the free-gear count n <= LIFTED_MAX_N, from
`cutlimit_r28.limit_decide` (the lifted LP: certificate, or an exact
in-polytope refutation).  A CERTIFIED cell carries an exact rational dual
certificate, saved as a pickle by `star_case.save_cert` and re-verified from
its own integers; nothing is trusted from the float solver.

THE TREE.  Decide every MIRROR-ORBIT REPRESENTATIVE at level k.  A cell that
does not certify is SPLIT on the next gear's phases into q_{k+1} children at
level k+1 (exhaustive by construction).  The certified cells of all levels,
each expanded by the mirror (research/lp_mirror_r30.py), partition the
held-phase product - asserted by the emitter.  This is the 37 -> 41 move of
round 29 with the mirror lemma applied on top:

    MIRROR(ws) = ((1 - W - w_q) mod q : q held)

sends the case at ws to a case with the reflected position set and an
isomorphic relaxation (round 29, lemma), so one representative per orbit is
decided and the other member's certificate is TRANSCRIBED, not solved.  The
self-mirror case (one per level) is its own representative.

PLANS:
    uv run python research/lp_tree_r30.py LEVEL <y> <W> <k> [workers] [tb]
        decide every orbit representative at level k (the root level);
    uv run python research/lp_tree_r30.py REFINE <y> <W> <k> [workers] [tb]
        split every level-k representative on disk that did NOT certify into
        its children at level k+1 (children canonicalised under the mirror)
        and decide them;
    uv run python research/lp_tree_r30.py CELL <y> <W> <k> <ws> [tb]
        one cell in the foreground (pricing);
    uv run python research/lp_tree_r30.py TABLE <y> <W>
        every cell on disk for that (y, W).

WHAT IS RECORDED PER CELL (integers and strings; every rational a string of an
exact Fraction): npos, ncols, nlinks, build_secs, the plain loop's verdict,
iteration count, LP-maximum trajectory, exact op count, lhs / rhs, whether
every row is the base cut, and - when the lifted route ran - V*, empty,
side, and the refutation's row value.  A non-optimal solver status is never
a verdict (ERROR record).
"""
import json
import os
import sys
import time
from fractions import Fraction
from itertools import product
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
R30 = os.path.join(HERE, 'data', 'r30')

LIFTED_MAX_N = 9          # round 28: n = 9 costs ~800 s a solve; n = 10 not run
# Cut passes the plain loop may take before the cell is handed on.  MEASURED
# this round at m53 W=171 k=4: a certifiable cell closes at ITERATION ZERO in
# 17-24 s (the base-cut polytope is already EMPTY); a cell that does not
# close there crawls (47.96 -> 47.40 against 47 over 12 passes and 611 s at
# case (0,0,2,12)) and is cheaper to SPLIT than to chase - its 17 children at
# k = 5 cost ~20 s each and the lifted LP decides them exactly.
PLAIN_ROUNDS = 4


def set_high_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:                                        # noqa: BLE001
        pass


def mirror(ws, held, W):
    return tuple((1 - W - w) % q for w, q in zip(ws, held))


def canon(ws, held, W):
    return min(tuple(ws), mirror(ws, held, W))


def cellname(y, W, k, ws):
    return "cell_m%d_w%d_k%d_h%s" % (y, W, k, "_".join(str(x) for x in ws))


def cellpath(y, W, k, ws):
    return os.path.join(R30, cellname(y, W, k, ws) + '.json')


def reps(y, W, k):
    """one representative per mirror orbit of the level-k held-phase product"""
    from lp_degree_range import gears_of
    held = gears_of(y)[:k]
    out = []
    seen = set()
    for ws in product(*[range(q) for q in held]):
        c = canon(ws, held, W)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def run_one(job):
    y, W, k, ws, tb = job[:5]
    p = cellpath(y, W, k, ws)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    set_high_priority()
    import star_case
    import cutlimit_r28
    from lp_degree_range import gears_of, base_cut
    from star_case import RelaxStar, decide_star
    star_case.OUT = R30
    name = cellname(y, W, k, ws)
    t0 = time.time()
    g = gears_of(y)
    held = g[:k]
    R = RelaxStar(g, W, held, tuple(ws))
    out = dict(machine=y, W=W, k=k, ws=list(ws), cell=name,
               held=list(held), free=list(R.gears), n=R.n,
               npos=len(R.pos), ncols=len(R.cols), nlinks=len(R.links),
               frhs=int(R.frhs), inexact_cells=R.inexact,
               build_secs=time.time() - t0,
               mirror=list(mirror(ws, held, W)),
               self_mirror=(mirror(ws, held, W) == tuple(ws)))
    try:
        tr = []
        v, info = decide_star(R, verbose=False, maxrounds=PLAIN_ROUNDS,
                              tag=name, time_budget=tb, trace=tr)
        out['plain_verdict'] = v
        out['plain_its'] = info.get('its')
        out['plain_traj'] = [t.get('lp_max') for t in tr]
        out['plain_secs'] = info.get('secs')
        if v == 'CERTIFIED':
            base = base_cut(R.n, 2)
            out.update(method='plain-loop', verdict='CERTIFIED',
                       its=info['its'], ops=info['ops'],
                       lhs=str(info['lhs']), rhs=str(info['rhs']),
                       margin=str(Fraction(info['rhs']) - Fraction(info['lhs'])),
                       nrows=len(R.rows),
                       rows_all_base_cut=all(tuple(lam) == base
                                             for (_i, lam) in R.rows))
        elif v == 'REFUTED':
            out.update(method='plain-loop', verdict='REFUTED',
                       how=info.get('how'),
                       row_value=str(info.get('row_value')),
                       row_rhs=str(info.get('row_rhs')))
        else:
            out['plain_lp_max'] = info.get('lp_max')
            if R.n <= LIFTED_MAX_N:
                R2 = RelaxStar(g, W, held, tuple(ws))           # fresh rows
                out['method'] = 'lifted'
                L = cutlimit_r28.limit_decide(R2, tag=name, tb=tb,
                                              witness=True)
                L.pop('traj', None)
                out.update(L)
                if L.get('verdict') == 'CERTIFIED':
                    base = base_cut(R2.n, 2)
                    out['margin'] = str(Fraction(L['rhs']) - Fraction(L['lhs']))
                    out['nrows'] = len(R2.rows)
                    out['rows_all_base_cut'] = all(
                        tuple(lam) == base for (_i, lam) in R2.rows)
            else:
                # the lifted LP is out of reach at this n: the cell is
                # UNDECIDED here and is refined one gear deeper.  Never a
                # verdict about the cell, only about this level.
                out.update(method='plain-loop', verdict='NOCERT-SPLIT')
    except Exception as e:                                   # noqa: BLE001
        out.update(verdict='ERROR', error=repr(e)[:400])
    out['total_secs'] = time.time() - t0
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print("  %-34s |pos|=%-3d %-13s its=%-3s ops=%-8s margin=%-14s [%.0fs]"
          % (name, out['npos'], out.get('verdict'), out.get('its'),
             out.get('ops'), (out.get('margin') or '-')[:14],
             out['total_secs']), flush=True)
    return out


def drive(jobs, workers=4):
    # THE DRIVER TOO, not only the workers.  MEASURED this round: with the
    # box at 100% CPU (34 python processes of five lanes on 20 cores) a
    # Normal-priority parent dispatching to High-priority workers throttled
    # the sweep to ~2 cells/min against the ~10/min the workers' own 20 s
    # cells allow - the workers sat idle waiting for the starved parent.
    set_high_priority()
    os.makedirs(R30, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(cellpath(*j[:4]))]
    print("  %d cells (%d already on disk) on %d workers"
          % (len(jobs), len(jobs) - len(todo), workers), flush=True)
    t0 = time.time()
    if not todo:
        return
    with Pool(workers) as pool:
        for _ in pool.imap_unordered(run_one, todo, chunksize=1):
            pass
    print("  DONE %d cells  [%.0fs]" % (len(todo), time.time() - t0),
          flush=True)


def level_jobs(y, W, k, tb):
    return [(y, W, k, ws, tb) for ws in reps(y, W, k)]


def refine_jobs(y, W, k, tb, limit=None):
    """children (canonicalised under the mirror) of every level-k
    representative on disk that did not certify; `limit` takes only the
    first `limit` refusals (a PRICED SAMPLE of the refinement, never a
    partition claim - the step manifest asserts what is actually covered)"""
    from lp_degree_range import gears_of
    g = gears_of(y)
    held1 = g[:k + 1]
    bad = []
    for ws in reps(y, W, k):
        p = cellpath(y, W, k, ws)
        if not os.path.exists(p):
            raise SystemExit("level %d incomplete: %s missing" % (k, p))
        with open(p) as fh:
            if json.load(fh).get('verdict') != 'CERTIFIED':
                bad.append(tuple(ws))
    if limit is not None:
        bad = bad[:limit]
    kids = set()
    for ws in bad:
        for v in range(g[k]):
            kids.add(canon(ws + (v,), held1, W))
    kids = sorted(kids)
    print("  %d of %d level-%d representatives did not certify; %d children"
          " at level %d after mirror canonicalisation"
          % (len(bad), len(reps(y, W, k)), k, len(kids), k + 1), flush=True)
    return [(y, W, k + 1, ws, tb) for ws in kids]


def table(y, W):
    rows = []
    for f in sorted(os.listdir(R30)):
        if f.startswith('cell_m%d_w%d_' % (y, W)) and f.endswith('.json'):
            with open(os.path.join(R30, f)) as fh:
                rows.append(json.load(fh))
    byk = {}
    for r in rows:
        byk.setdefault(r['k'], []).append(r)
    for k in sorted(byk):
        sel = byk[k]
        vs = {}
        for r in sel:
            vs[r.get('verdict')] = vs.get(r.get('verdict'), 0) + 1
        cert = [r for r in sel if r.get('verdict') == 'CERTIFIED']
        its0 = sum(1 for r in cert if r.get('its') == 0)
        base = sum(1 for r in cert if r.get('rows_all_base_cut'))
        ops = sum(r.get('ops') or 0 for r in cert)
        secs = sum(r.get('total_secs') or 0 for r in sel)
        margins = [Fraction(r['margin']) for r in cert if r.get('margin')]
        print("  k=%d: %d cells on disk, verdicts %s; certified at it=0: %d,"
              " all-base-cut: %d; ops %d; wall %.0fs; margin min %s max %s"
              % (k, len(sel), vs, its0, base, ops, secs,
                 min(margins) if margins else '-',
                 max(margins) if margins else '-'))
        for r in sel:
            if r.get('verdict') != 'CERTIFIED':
                print("     %s  %s  lp_max=%s  V*=%s"
                      % (r['cell'], r.get('verdict'), r.get('plain_lp_max'),
                         r.get('vstar')))
    return rows


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'HELP')
    os.makedirs(R30, exist_ok=True)
    if cmd == 'LEVEL':
        y, W, k = int(a[1]), int(a[2]), int(a[3])
        workers = int(a[4]) if len(a) > 4 else 4
        tb = float(a[5]) if len(a) > 5 else 900.0
        drive(level_jobs(y, W, k, tb), workers)
    elif cmd == 'REFINE':
        y, W, k = int(a[1]), int(a[2]), int(a[3])
        workers = int(a[4]) if len(a) > 4 else 4
        tb = float(a[5]) if len(a) > 5 else 900.0
        limit = int(a[6]) if len(a) > 6 else None
        drive(refine_jobs(y, W, k, tb, limit), workers)
    elif cmd == 'CELL':
        y, W, k = int(a[1]), int(a[2]), int(a[3])
        ws = tuple(int(x) for x in a[4].split(','))
        tb = float(a[5]) if len(a) > 5 else 900.0
        o = run_one((y, W, k, ws, tb))
        print(json.dumps({kk: v for kk, v in o.items()
                          if kk not in ('plain_traj',)}, indent=1))
    elif cmd == 'TABLE':
        table(int(a[1]), int(a[2]))
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
