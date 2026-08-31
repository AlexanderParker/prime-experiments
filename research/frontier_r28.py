"""ROUND 28, LP-DUALITY THREAD - THE CUT-LOOP FRONTIER MAP, and the padded step.

A resumable parallel driver over CELLS.  A cell is (machine y, width W, held
count k, held phases ws); its verdict is computed by `cutlimit_r28.limit_decide`
- the LIFTED LP that gives the cut loop's LIMIT V* directly, then the EXACT
verdict on whichever side of |pos| that limit falls.

One JSON per cell in research/data/r28/, so the driver resumes from its own
output and a killed run loses at most the cells in flight.

    python research/frontier_r28.py MAP    [workers]   # (a) the frontier map
    python research/frontier_r28.py PAD    [workers]   # (b) 31->37 at 80 / 88
    python research/frontier_r28.py TABLE             # print what is on disk
"""
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
R28 = os.path.join(HERE, 'data', 'r28')


def cellname(y, W, k, ws):
    return "cell_m%d_w%d_k%d_h%s" % (y, W, k, "_".join(str(x) for x in ws))


def run_one(job):
    y, W, k, ws, tb = job[:5]
    wit = job[5] if len(job) > 5 else False
    import star_case
    import cutlimit_r28
    from lp_degree_range import gears_of
    from star_case import RelaxStar
    star_case.OUT = R28
    name = cellname(y, W, k, ws)
    p = os.path.join(R28, name + '.json')
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    t0 = time.time()
    g = gears_of(y)
    R = RelaxStar(g, W, g[:k], tuple(ws))
    out = dict(machine=y, W=W, k=k, ws=list(ws), cell=name)
    try:
        out.update(cutlimit_r28.limit_decide(R, tag=name, tb=tb,
                                             witness=wit))
    except Exception as e:                                   # noqa: BLE001
        out.update(verdict='ERROR', error=repr(e)[:400])
    out['total_secs'] = time.time() - t0
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print("  %-28s %-10s V*=%-10s frhs=%-4s %s  [%.0fs]"
          % (name, out.get('side', '-'),
             ('%.4f' % out['vstar']) if out.get('vstar') is not None
             else ('EMPTY' if out.get('empty') else '-'),
             out.get('frhs'), out.get('verdict'), out['total_secs']),
          flush=True)
    return out


def drive(jobs, workers=6):
    os.makedirs(R28, exist_ok=True)
    t0 = time.time()
    print("  %d cells on %d workers" % (len(jobs), workers), flush=True)
    with Pool(workers) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    print("  DONE %d cells  [%.0fs]" % (len(res), time.time() - t0),
          flush=True)
    return res


# ------------------------------------------------------------------ the plans
def map_jobs():
    """(a) THE FRONTIER MAP.  One representative case (all-zero held phases)
    per (machine, width), swept across the width axis at the k the ladder uses,
    so the question 'at which (machine, width) pairs does the loop converge'
    gets an answer at every point rather than at the two round-27 points."""
    J = []
    plan = [
        # machine, k, widths  (k chosen so the free-gear count stays <= 8,
        # which is what keeps the lifted LP's 2^n columns per position cheap)
        (23, 1, range(30, 50, 2)),
        (29, 1, range(36, 66, 2)),
        (31, 1, range(44, 78, 2)),
        (37, 2, range(70, 98, 2)),
        (41, 3, range(92, 132, 2)),
        (43, 4, range(100, 138, 2)),
        (47, 4, range(112, 154, 3)),
        (53, 5, range(120, 160, 4)),
    ]
    for (y, k, ws_) in plan:
        for W in ws_:
            J.append((y, W, k, tuple([0] * k), 300.0, False))
    return J


def deep_jobs():
    """The round-27 cell itself, at the k round 27 ran (n = 9 free gears, so
    each of these is ~10 minutes), plus its width neighbours - the points that
    decide whether the decelerating loop had a limit above 43 or below it."""
    return [(43, W, 3, (0, 0, 0), 300.0, True)
            for W in (117, 110, 120, 125, 128, 134)]


def pad_jobs():
    """(b) THE PADDED STEP 31->37.  W = 80 is the increment width
    F_2(31) + s_min(37) = 68 + 12, which the true F(37) = 88 exceeds by 8;
    W = 88 is the truth.  Full case splits at both."""
    from itertools import product
    from lp_degree_range import gears_of
    g = gears_of(37)
    J = []
    for W in (80, 88):
        for k in (2,):
            for ws in product(*[range(q) for q in g[:k]]):
                J.append((37, W, k, tuple(ws), 300.0, False))
    return J


def pad3_jobs():
    from itertools import product
    from lp_degree_range import gears_of
    g = gears_of(37)
    J = []
    for W in (88, 80):
        for ws in product(*[range(q) for q in g[:3]]):
            J.append((37, W, 3, tuple(ws), 300.0, False))
    return J


def table():
    rows = []
    for f in sorted(os.listdir(R28)):
        if f.startswith('cell_') and f.endswith('.json'):
            with open(os.path.join(R28, f)) as fh:
                rows.append(json.load(fh))
    rows.sort(key=lambda r: (r['machine'], r['k'], r['W'], r['ws']))
    print("  machine  k   W   ws        |pos|    V*         gap      verdict")
    for r in rows:
        print("  %5d  %2d %4d  %-8s %5s   %-10s %-9s %s"
              % (r['machine'], r['k'], r['W'],
                 ",".join(map(str, r['ws'])), r.get('frhs'),
                 ('%.4f' % r['vstar']) if r.get('vstar') is not None
                 else ('EMPTY' if r.get('empty') else '-'),
                 ('%+.4f' % r['gap']) if r.get('gap') is not None else '-',
                 r.get('verdict')))
    return rows


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'TABLE')
    w = int(a[1]) if len(a) > 1 else 6
    if cmd == 'MAP':
        drive(map_jobs(), w)
    elif cmd == 'PAD':
        drive(pad_jobs(), w)
    elif cmd == 'PAD3':
        drive(pad3_jobs(), w)
    elif cmd == 'DEEP':
        drive(deep_jobs(), w)
    elif cmd == 'TABLE':
        table()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
