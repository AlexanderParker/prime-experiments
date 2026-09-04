"""Round 30 (constructor): C6 - is the padded letter's flank envelope at the
F_3 wall?  Reads the counted census triple tables (research/data/r30/occ_<y>.npz)
and reports every F_3-attaining 3-window with the legality of its middle, plus
Phi(q') + q' against F_3 and F_2 + s_min.  Exact integers only."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from occ_census_r30 import KNOWN_F, KNOWN_F2, letters, cls_of, R30

for y in (11, 13, 17, 19, 23, 29, 31, 37):
    p = os.path.join(R30, "occ_%d.npz" % y)
    if not os.path.exists(p):
        print("m%d: no counted census on disk" % y); continue
    z = np.load(p)
    F = KNOWN_F[y]; K = F + 1
    q1, a, b, lam = letters(y)
    s_min = min(a, b)
    T = z["trip"].reshape(K, K, K)
    i, j, k = np.nonzero(T)
    s = i + j + k
    F3 = int(s.max())
    att = [(int(i[t]), int(j[t]), int(k[t]), int(T[i[t], j[t], k[t]])) for t in np.flatnonzero(s == F3)]
    mids = sorted({m for _, m, _, _ in att})
    leg = {m: cls_of(m, q1, a, b) for m in mids}
    P1 = z["phi1"]; B = len(lam) + 1
    padded = [v for v in lam if v % q1 == 0 and v <= F]
    line = "m%-3d F_3 = %3d  maximisers %s  middles %s legal? %s | F_2+s_min = %d (F_3 - that = %+d)" % (
        y, F3, [(u, v, w) for u, v, w, _ in att], mids,
        {m: ("class %s" % {0: "PADDED", 1: "a", -1: "b"}[c] if c is not None else "no") for m, c in leg.items()},
        KNOWN_F2[y] + s_min, F3 - KNOWN_F2[y] - s_min)
    print(line)
    # GATE (C6): only m31's F_3 maximiser has a legal middle, and it is padded
    if y == 31:
        assert mids == [37] and leg[37] == 0, ("C6 at m31", mids, leg)
    else:
        assert all(c is None for c in leg.values()), ("C6: legal middle at m%d" % y, leg)
    for v in padded:
        key = lam.index(v) + 1
        enc = int(P1[key])
        if enc < 0:
            print("      padded letter %d: NOT realised" % v); continue
        fs, gL = divmod(enc, K)
        print("      padded letter %d: Phi = %d  (%d,%d)   Phi + q' = %d  vs F_3 %d (slack %d)  vs F_2+s_min %d (slack %+d)"
              % (v, fs, gL, fs - gL, fs + v, F3, F3 - fs - v, KNOWN_F2[y] + s_min, KNOWN_F2[y] + s_min - fs - v))
        if y == 31:
            assert fs + v == F3, "C6: Phi(37) + 37 == F_3(31)"
        else:
            assert fs + v < F3, ("C6: padded envelope strictly below F_3", y)
print("all assertions passed")
