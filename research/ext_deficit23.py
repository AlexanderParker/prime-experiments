"""Harvester round 22: the 23 -> 29 extension rung of the deficit ladder.

Round 21 measured the extension deficit (true family max at the new gear minus the best
lift of any old-gear maximiser) as 9 (13->17), 18 (17->19), 36 (19->23) - a doubling on
three points, with the 36 computed from a single lineage.  Round 22 (ext_deficit19.py)
recomputed all three over COMPLETE winner sets and got 9, 18, 36 exactly.

The doubling is already REFUTED by arithmetic: with h_2 from OEIS A288815 (Ziller-Morack)
    F = h_2/2 :  75(13)  96(17)  129(19)  183(23)  225(29)
the 23->29 increment is only 225-183 = 42, and the deficit can never exceed the
increment, so it cannot be 72.  The cap law (paired-jacobsthal-values.md 4b) plus the
observed record 2-gap sums 12, 15, 18 (+3 per rung) PREDICTS deficit = 42 - 21 = 21.

This script measures it: for every delta in the complete 23-winner set and every lift
residue mod 29, the exact maximal gap of the 29-machine (period 1,078,282,205), computed
block-by-block so nothing of that size is ever materialised.

Usage:  python ext_deficit23.py <shard> <nshards>
        python ext_deficit23.py merge <nshards>
"""
import os
import sys
import numpy as np
from math import prod
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from family_scan import survivors

QS23 = [5, 7, 11, 13, 17, 19, 23]
QNEW = 29
Q23 = prod(QS23)                 # 37,182,145
Q29 = Q23 * QNEW                 # 1,078,282,205
TRUE_G29 = 75                    # h_2(29) = 450 -> F = 225 -> G = 75


def _lift(S, s_mod, r, Qold, qnew):
    """max cyclic gap in Z_{Qold*qnew} of the survivors S of the Qold-machine, with the
    new gear's teeth {0, -r} mod qnew removed, computed one block of the qnew copies at
    a time so nothing of size Qold*qnew is ever materialised."""
    Qn = Qold * qnew
    best, first, last = 0, None, None
    for t in range(qnew):
        ct = (t * Qold) % qnew
        a = (-ct) % qnew
        b = (-r - ct) % qnew
        keep = (s_mod != a) & (s_mod != b)
        v = S[keep]
        if v.size == 0:
            continue
        if v.size > 1:
            best = max(best, int(np.diff(v).max()))
        lo = int(v[0]) + t * Qold
        hi = int(v[-1]) + t * Qold
        if first is None:
            first = lo
        else:
            best = max(best, lo - last)
        last = hi
    if first is None:
        return Qn
    return max(best, first + Qn - last)


def selftest():
    """the block-wise lift must reproduce the materialised 19->23 table computed by
    ext_deficit19.py (independent code path, same answers)."""
    q19 = [5, 7, 11, 13, 17, 19]
    Q19 = prod(q19)
    ref = np.load("research/data/ext19_to23.npy")   # (delta_new, delta_old, r, G)
    tab = {(int(a[1]), int(a[2])): int(a[3]) for a in ref}
    deltas = sorted({int(a[1]) for a in ref})[:6]
    for d in deltas:
        S = np.flatnonzero(survivors(q19, d, Q19)).astype(np.int64)
        s_mod = (S % 23).astype(np.int8)
        for r in range(23):
            g = _lift(S, s_mod, r, Q19, 23)
            assert g == tab[(d, r)], (d, r, g, tab[(d, r)])
    print(f"selftest: block-wise lift matches the materialised 19->23 table on "
          f"{len(deltas)*23} cases - ALL ASSERTIONS GREEN", flush=True)


def run(winners):
    out = []
    for d in winners:
        S23 = np.flatnonzero(survivors(QS23, int(d), Q23)).astype(np.int32)
        s_mod = (S23 % QNEW).astype(np.int8)
        for r in range(QNEW):
            g = _lift(S23, s_mod, r, Q23, QNEW)
            out.append((int(d), r, int(g)))
        print(f"  delta_23 = {d}: best lift G = "
              f"{max(g for dd, rr, g in out if dd == d)}", flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "merge":
        ns = int(sys.argv[2])
        rows = []
        for i in range(ns):
            rows += np.load(f"research/data/ext23_{i}.npy").tolist()
        best = max(r[2] for r in rows)
        arg = [r for r in rows if r[2] == best]
        msg = (f"23 -> 29 over {len({r[0] for r in rows})} complete 23-winners x 29 "
               f"lifts: best extension G = {best} (F = {3*best}); true max G = "
               f"{TRUE_G29} (F = {3*TRUE_G29}); DEFICIT = {3*(TRUE_G29-best)}")
        print(msg, flush=True)
        print(f"  argmax lifts: {arg[:6]}", flush=True)
        with open("research/data/ext_deficit23.out", "w") as fh:
            fh.write(msg + "\n" + repr(arg[:20]) + "\n")
        np.save("research/data/ext23_all.npy", np.array(rows, np.int64))
    else:
        s, ns = int(sys.argv[1]), int(sys.argv[2])
        w = np.load("research/data/family_w23_delta.npy")
        mine = w[s::ns]
        rows = run(mine)
        np.save(f"research/data/ext23_{s}.npy", np.array(rows, np.int64))
        print(f"shard {s}: {len(mine)} winners done", flush=True)
