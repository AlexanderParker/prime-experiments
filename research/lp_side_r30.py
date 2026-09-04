"""ROUND 30, LP-DUALITY THREAD - THE TWO SINGLE-WORKER SIDE JOBS THAT SCORE
ROUND-29 PREDICTIONS E14 AND E15.  Outputs in research/data/r30/.

  E14  W_c(43, 3) = min{W : G < 0} at the all-zero case of machine 43 with
       three held gears, by the round-29 bisection (`lp_cells_r29.wc`) with
       the sign pattern asserted width by width around the crossing.  Every
       width is one lifted LP at n = 9.  Prediction: W_c(43,3) >= 92.
           uv run python research/lp_side_r30.py E14 [lo] [hi]
  E15  machine 41, W = 104, k = 2: V* and |pos| for one representative of
       each of the 18 mirror orbits of the 35 cases, by the LIFTED route
       (`cutlimit_r28.limit_decide` with witness=False - a V* reading, not a
       verdict; the cells' verdicts are round 29's).  Prediction: the number
       of distinct (V*, |pos|) classes over the 35 cases is STRICTLY FEWER
       than the 18 mirror orbits.
           uv run python research/lp_side_r30.py E15
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
R30 = os.path.join(HERE, 'data', 'r30')


def high():
    try:
        import psutil
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:                                        # noqa: BLE001
        pass


def e14(lo=84, hi=117):
    import lp_cells_r29 as C
    C.R29 = R30                       # write wc_m43_k3.json into r30
    high()
    return C.wc(43, 3, lo, hi)


def e15():
    import cutlimit_r28
    from lp_tree_r30 import reps
    from lp_degree_range import gears_of
    from star_case import RelaxStar
    high()
    y, W, k = 41, 104, 2
    g = gears_of(y)
    for ws in reps(y, W, k):
        p = os.path.join(R30, 'e15_m41_w104_k2_h%s.json'
                         % "_".join(str(x) for x in ws))
        if os.path.exists(p):
            continue
        t0 = time.time()
        R = RelaxStar(g, W, g[:k], tuple(ws))
        out = dict(machine=y, W=W, k=k, ws=list(ws), npos=len(R.pos))
        try:
            L = cutlimit_r28.lifted(R)
            out.update(vstar=L.get('val'), status=L.get('status'),
                       empty=(L.get('val') is None and L.get('status') == 2),
                       lifted_secs=L.get('secs'))
        except Exception as e:                               # noqa: BLE001
            out['error'] = repr(e)[:300]
        out['total_secs'] = time.time() - t0
        with open(p, 'w') as fh:
            json.dump(out, fh, indent=1)
        print("  E15 case %s |pos|=%d V*=%s [%.0fs]"
              % (ws, out['npos'], out.get('vstar'), out['total_secs']),
              flush=True)


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'HELP')
    os.makedirs(R30, exist_ok=True)
    if cmd == 'E14':
        e14(int(a[1]) if len(a) > 1 else 84, int(a[2]) if len(a) > 2 else 117)
    elif cmd == 'E15':
        e15()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
