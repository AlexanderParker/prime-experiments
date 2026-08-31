"""ROUND 28, LP-DUALITY THREAD - IS THE DECELERATION LAWFUL?

Round 27 watched the cut loop's LP maximum fall towards the target and slow
down, and could not say whether that was slow convergence or an asymptote.
With `cutlimit_r28.lifted` the loop's LIMIT V* is known independently, so the
question splits exactly:

    lp_max_t - |pos|   =   (lp_max_t - V*)   +   (V* - |pos|)
    "gap to the target"     THE CONVERGENCE       THE OFFSET, a constant

The second term does not move.  If it is positive the loop can never reach the
target however long it runs - the deceleration is the sequence bending towards
a limit that is in the wrong place, not a rate that is dying.  The first term
is the actual convergence, and THAT is what a rate can be fitted to.

This file measures both on the same cell: the ordinary cut loop with its
trajectory recorded, and the lifted LP.  It reports the excess e_t = lp_max_t -
V*, the per-pass ratios e_{t+1}/e_t, and the geometric fit.

    python research/decel_r28.py <y> <W> <k> [ws] [time budget]
    python research/decel_r28.py PLAN                 # the round-28 cell list
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import star_case                                          # noqa: E402
import cutlimit_r28                                       # noqa: E402
from star_case import RelaxStar, decide_star              # noqa: E402
from lp_degree_range import gears_of                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R28 = os.path.join(HERE, 'data', 'r28')

PLAN = [(31, 60, 1), (31, 66, 1), (31, 74, 1),
        (37, 82, 2), (37, 88, 2), (37, 95, 2),
        (41, 110, 3), (41, 129, 3)]


def one(y, W, k, ws=None, tb=600.0):
    ws = tuple(ws if ws is not None else [0] * k)
    g = gears_of(y)
    star_case.OUT = R28
    os.makedirs(R28, exist_ok=True)
    name = "decel_m%d_w%d_k%d_h%s" % (y, W, k, "_".join(map(str, ws)))
    p = os.path.join(R28, name + '.json')
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    R = RelaxStar(g, W, g[:k], ws)
    L = cutlimit_r28.lifted(R)
    vstar = L['val']
    R2 = RelaxStar(g, W, g[:k], ws)
    tr = []
    t0 = time.time()
    v, info = decide_star(R2, verbose=False, maxrounds=400, tag=None,
                          time_budget=tb, trace=tr)
    traj = [t['lp_max'] for t in tr if t['lp_max'] is not None]
    out = dict(machine=y, W=W, k=k, ws=list(ws), frhs=int(R.frhs),
               vstar=vstar, empty=(vstar is None),
               offset=(None if vstar is None else vstar - float(R.frhs)),
               loop_verdict=v, loop_secs=time.time() - t0,
               loop_its=info.get('its'), traj=traj,
               lifted_secs=L.get('secs'))
    if vstar is not None and traj:
        e = [t - vstar for t in traj]
        out['excess'] = e
        out['ratios'] = [e[i + 1] / e[i] for i in range(len(e) - 1)
                         if e[i] > 1e-9]
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    return out


def show(o):
    print("  m%-3d W=%-4d k=%d  |pos|=%-4d  V*=%-10s  offset=%-9s  loop=%s"
          " (%d passes, %.0fs)"
          % (o['machine'], o['W'], o['k'], o['frhs'],
             'EMPTY' if o['vstar'] is None else '%.5f' % o['vstar'],
             '-' if o['offset'] is None else '%+.4f' % o['offset'],
             o['loop_verdict'], o.get('loop_its') or 0, o['loop_secs']))
    if o.get('excess'):
        print("     excess: " + " ".join("%.4f" % x for x in o['excess']))
    if o.get('ratios'):
        r = o['ratios']
        print("     ratios: " + " ".join("%.3f" % x for x in r))
        mid = r[len(r) // 3:]
        if mid:
            print("     mean ratio over the last two thirds: %.3f"
                  % (sum(mid) / len(mid)))


def main():
    a = sys.argv[1:]
    if a and a[0].upper() == 'PLAN':
        for (y, W, k) in PLAN:
            show(one(y, W, k))
        return
    y, W, k = int(a[0]), int(a[1]), int(a[2])
    ws = tuple(int(x) for x in a[3].split(',')) if len(a) > 3 else None
    tb = float(a[4]) if len(a) > 4 else 600.0
    show(one(y, W, k, ws, tb))


if __name__ == '__main__':
    main()
