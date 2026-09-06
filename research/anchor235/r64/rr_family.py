"""rr_family.py -- is the top-band attainment law (F_2 is attained by a pair whose larger member
is a near-record gap) a real-teeth law or a structural one?

The tooth-counterfactual family of alignment-rules section 5: same gears, teeth at +-v_g with v_g
uniform in [1, (g-1)/2], seed 20260906 (the same members as r57/r58/r62/r63).  Full periods at
{5..13}, {5..17}, {5..19}, plus the real machine as member 0.

For each member: F, F_2, the F_2 pair, rho = (larger member of the pair)/F, n1(F), and D_top for
the 0.8 F band.

Usage: uv run python rr_family.py
"""
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import build_levels, u_of                    # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
SEED = 20260906


def profile(size):
    mx = int(size.max())
    m = np.bincount(size, minlength=mx + 1)
    left = np.roll(size, 1)
    right = np.roll(size, -1)
    n1 = np.zeros(mx + 1, dtype=np.int64)
    np.maximum.at(n1, size, np.maximum(left, right))
    return m, n1, mx


def stats(size):
    m, n1, F = profile(size)
    sizes = [v for v in range(1, F + 1) if m[v]]
    F2 = max(v + int(n1[v]) for v in sizes)
    arg = [v for v in sizes if v + int(n1[v]) == F2]
    rho = max(max(v, int(n1[v])) for v in arg) / F
    band = [v for v in sizes if v >= 0.8 * F]
    Dtop = F2 - max(v + int(n1[v]) for v in band)
    return dict(F=F, F2=F2, pair=[(v, int(n1[v])) for v in arg], rho=round(rho, 3),
                n1F=int(n1[F]), Dtop=int(Dtop), nsizes=len(sizes))


def main():
    rng = random.Random(SEED)
    res = {}
    for gears in ([5, 7, 11, 13], [5, 7, 11, 13, 17], [5, 7, 11, 13, 17, 19]):
        key = f"{{5..{gears[-1]}}}"
        rows = []
        lv = build_levels(gears)[-1]
        rows.append({"member": "real", **stats(lv.size)})
        for k in range(20):
            vs = [rng.randrange(1, (g + 1) // 2) for g in gears]
            lv = build_levels(gears, vs=vs)[-1]
            rows.append({"member": k, "teeth": vs, **stats(lv.size)})
        res[key] = rows
        rr = [r["rho"] for r in rows]
        dd = [r["Dtop"] for r in rows]
        print(f"\n{key}: rho (larger member of the F_2 pair / F) over the real machine "
              f"and 20 members")
        print(f"  real: rho={rows[0]['rho']} F={rows[0]['F']} F_2={rows[0]['F2']} "
              f"pair={rows[0]['pair']} n1(F)={rows[0]['n1F']} D_top={rows[0]['Dtop']}")
        print(f"  family rho: min={min(rr[1:])} median={sorted(rr[1:])[10]} max={max(rr[1:])}; "
              f"members with rho >= 0.8: {sum(1 for x in rr[1:] if x >= 0.8)}/20")
        print(f"  family D_top: {dd[1:]}  ; members with D_top = 0: "
              f"{sum(1 for x in dd[1:] if x == 0)}/20")
        print(f"  family n1(F): {[r['n1F'] for r in rows[1:]]}  (real {rows[0]['n1F']})")
    with open(os.path.join(OUT, "family.json"), "w") as f:
        json.dump(res, f, default=int)


if __name__ == "__main__":
    main()
