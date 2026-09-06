"""What transfers to the bottom machine: the gear set {5..q} in the COLUMN coordinate.

Column k has members 6k - 1, 6k + 1; gear g strikes k iff k = +- u_g (mod g), u_g = 6^{-1} mod g.
By the conjugacy (document 1, L19) this is the same machine as the top machine {5..q} in the
pair coordinate, re-coordinatised - so counting laws must agree and metric laws must not.
"""

import json
import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from common import col_teeth, open_mask, teeth  # noqa: E402

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31]
N = 1_000_000


def col_mask_range(gears, lo, hi):
    """Boolean array over columns [lo, hi): True where every gear misses."""
    k = np.arange(lo, hi)
    m = np.ones(hi - lo, dtype=bool)
    for g in gears:
        r = k % g
        for t in col_teeth(g):
            m &= r != t
    return m


def next_open(mask):
    """L(x) = distance to the next open position, for every x where it is defined."""
    n = len(mask)
    L = np.full(n, -1, dtype=np.int64)
    nxt = -1
    for i in range(n - 1, -1, -1):
        if mask[i]:
            nxt = i
        L[i] = nxt - i if nxt >= 0 else -1
    return L


def mex_prog(gears, x, B, ut):
    vals = set()
    for g in gears:
        for t in ut[g]:
            v = (t - x) % g
            while v <= B:
                vals.add(v)
                v += g
    j = 0
    while j in vals:
        j += 1
    return j


def mex_plain(gears, x, ut):
    s = {(t - x) % g for g in gears for t in ut[g]}
    j = 0
    while j in s:
        j += 1
    return j


def main():
    out = {"conjugacy": [], "column": []}

    # ---- 1. the conjugacy, checked directly on small wheels
    for gears in [[5, 7], [5, 7, 11], [5, 7, 11, 13], [5, 7, 11, 13, 17], [7, 11, 13]]:
        W = prod(gears)
        pm = open_mask(gears)
        cm = np.ones(W, dtype=bool)
        kk = np.arange(W)
        for g in gears:
            r = kk % g
            for t in col_teeth(g):
                cm &= r != t
        inv6 = pow(6, -1, W)
        img = (inv6 * (kk + 1)) % W
        mism = int((pm != cm[img]).sum())
        out["conjugacy"].append({"gears": gears, "W": W, "mismatch": mism,
                                 "pair_open": int(pm.sum()), "col_open": int(cm.sum()),
                                 "prod_g2": prod(g - 2 for g in gears)})
        print("conj", out["conjugacy"][-1], flush=True)

    # ---- 2. the column machine {5..q}: metric laws and the corrected mex
    for t in range(2, len(PRIMES) + 1):
        gears = PRIMES[:t]
        m = len(gears)
        ut = {g: col_teeth(g) for g in gears}
        W = prod(gears)
        span = min(W, N + 4000)
        mask = col_mask_range(gears, 0, span)
        L = next_open(mask)
        valid = L >= 0
        Lv = L[valid][: min(N, int(valid.sum()))]
        F = int(Lv.max())

        # arcs / letters per gear in the column coordinate
        arcs = {}
        letters = {}
        for g in gears:
            tt = ut[g]
            d = (tt[1] - tt[0]) % g
            letters[g] = sorted({d, (g - d) % g})
            op = [r for r in range(g) if r not in tt]
            # arc lengths
            best = []
            run = 0
            start = tt[0]
            for i in range(g):
                r = (start + i) % g
                if r in tt:
                    if run:
                        best.append(run)
                    run = 0
                else:
                    run += 1
            if run:
                best.append(run)
            arcs[g] = sorted(best, reverse=True)

        # longest run of consecutive open columns, longest step-2 chain
        run = 0
        best_run = 0
        for v in mask[:span]:
            run = run + 1 if v else 0
            best_run = max(best_run, run)
        ch = mask[: span - 2] & mask[2:span]
        chain2 = int(ch.sum())

        # the gap census holes below the record
        pos = np.flatnonzero(mask)
        gaps = np.diff(pos)
        holes = sorted(d for d in range(1, F + 1) if not (gaps == d).any())

        rec = {"gears": gears, "m": m, "positions": int(len(Lv)), "F_range": F,
               "longest_open_run": best_run, "chain2_starts": chain2,
               "arcs": arcs, "letters": letters, "gap_holes": holes}

        # the plain mex (no recurrences) and the corrected one
        xs = np.arange(len(Lv))
        step = max(1, len(Lv) // 200000)
        sample = list(range(0, len(Lv), step))
        badplain = 0
        for x in sample:
            if mex_plain(gears, x, ut) != L[x]:
                badplain += 1
        rec["plain_mex_bad"] = badplain
        rec["plain_mex_tested"] = len(sample)

        for B in (2 * m, F):
            bad = cert = 0
            for x in sample:
                M = mex_prog(gears, x, B, ut)
                if M <= B:
                    cert += 1
                    if M != L[x]:
                        bad += 1
            rec[f"corr_mex_B{B}_certified"] = cert
            rec[f"corr_mex_B{B}_bad"] = bad
            rec[f"corr_mex_B{B}_terms"] = sum(2 * (1 + B // g) for g in gears)
        out["column"].append(rec)
        print("col", json.dumps(rec), flush=True)

    with open(__file__.rsplit("s5_")[0] + "results/s5_bottom.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
