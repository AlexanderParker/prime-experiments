"""
LATERAL round 30, BLOCK B follow-up - THE DEPTH-2 FAILURE ON THE FAMILY IS THE
SELF-MIRROR WINDOW.  The pair of gaps around slot 0 is (d_0, d_0), d_0 = the
first opening after 0 (mirror symmetry), so F_2 >= 2 d_0 at every member, and
the depth-2 half F_2 <= F + q' can fail by that window alone whenever
2 d_0 > F + q'.  This script measures, per step: d_0 on the family (a tiny
loop, no sieve), how often F_2 is attained by the wrap pair (F_2 == 2 d_0), the
real machine's d_0 and its percentile, and whether every slack <= 0 member is a
wrap-pair member.
Usage: uv run python research/d0_family_r30.py
"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tooth_L_r30 import COLS, CI, space_of, TWIN, OUT, gate

def d0_of(gears, vs):
    k = 1
    while any(k % q in (v % q, (-v) % q) for q, v in zip(gears, vs)):
        k += 1
    return k

def load(step):
    if step == '19_23':
        fs = sorted(f for f in os.listdir(OUT) if f.startswith('L_19_23_c') and f.endswith('.npy'))
    elif step == '23_29':
        fs = sorted(f for f in os.listdir(OUT) if f.startswith('L_23_29_s') and f.endswith('.npy'))
    else:
        return np.load(os.path.join(OUT, 'L_%s.npy' % step))
    return np.concatenate([np.load(os.path.join(OUT, f)) for f in fs])

STEPS = [('7_11',[5,7],11),('11_13',[5,7,11],13),('13_17',[5,7,11,13],17),
         ('17_19',[5,7,11,13,17],19),('19_23',[5,7,11,13,17,19],23),('23_29',[5,7,11,13,17,19,23],29)]
for step, og, qp in STEPS:
    try:
        arr = load(step)
    except ValueError:
        continue
    ov = list(itertools.product(*space_of(og)))
    ti = ov.index(tuple(TWIN[q] for q in og))
    _, first = np.unique(arr[:, 0], return_index=True)
    rows = arr[first]
    oi = rows[:, CI['oi']]; F = rows[:, CI['F']]; F2 = rows[:, CI['F2']]
    d0 = np.array([d0_of(og, ov[int(o)]) for o in oi])
    gate(bool(np.all(F2 >= 2 * d0)), "%s: F_2 >= 2 d_0 at all %d members (the wrap pair (d_0, d_0) is a 2-window)" % (step, len(oi)))
    slack = F + qp - F2
    wrap = F2 == 2 * d0
    bad = slack <= 0
    real = oi == ti
    tag = step.replace('_', '->') + ('  [SAMPLE]' if step == '23_29' else '')
    print("%s: %d members | d_0: min %d median %.0f max %d; REAL d_0 = %d (%.1f%% of members have larger d_0) | F_2 attained by the wrap pair at %d (%.2f%%) | slack<=0 at %d, of which wrap-pair %d | max 2d_0 - (F+q') = %d"
          % (tag, len(oi), d0.min(), np.median(d0), d0.max(), d0[real][0] if real.any() else -1,
             100.0 * np.mean(d0 > (d0[real][0] if real.any() else 0)), wrap.sum(), 100.0 * wrap.mean(),
             bad.sum(), (bad & wrap).sum(), int((2 * d0 - F - qp).max())))
    gate(bool(np.all(wrap[bad])), "%s: every slack <= 0 member has F_2 = 2 d_0 (the self-mirror 2-window)" % step)
    if step == '19_23':
        gate(int(bad.sum()) == 1 and int(d0[bad][0]) == 25 and int(F[bad][0]) == 26,
             "19->23: the unique depth-2 failure is teeth %s with d_0 = 25, F = 26, F_2 = 50" % (ov[int(oi[bad][0])],))
    # the second family-wide statement: excluding the wrap pair, slack stays positive?
    F2nw = np.where(wrap, -1, F2)
    print("      excluding wrap-pair members, min slack = %d" % int((F + qp - F2)[~wrap].min()))
print("\nALL GATES PASSED")
