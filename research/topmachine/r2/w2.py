"""Section 3 (W2): how the record on a prefix approaches the wheel record.

Reads results/inuse.json (the exact chunked censuses) and tests

    F_range(N) = max { d : W / c(d) <= N } - 1 ,    c(d) = number of gaps of length >= d,

i.e. the range record is a first-hit on the wheel's own gap census.  Also re-scans the
gears 7..31 period to collect EVERY position of the longest gaps, to see where the record
blocks sit relative to the origin and to the small gears' wheel.

usage: uv run python research/topmachine/r2/w2.py results/w2.json
"""

import json
import os
import sys
from math import prod

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gapcensus import census_formula  # noqa: E402

CHUNK = 1 << 25


def positions_of_long_gaps(gears, M, dmin):
    """All (gap length, start of the pair-free block) with gap >= dmin over [0, M)."""
    out = []
    last_open = -1
    base = 0
    while base < M:
        n = min(CHUNK, M - base)
        a = np.ones(n, dtype=bool)
        for g in gears:
            a[(-base) % g :: g] = False
            a[(-2 - base) % g :: g] = False
        idx = np.flatnonzero(a)
        if len(idx):
            pos = idx.astype(np.int64) + base
            prev = np.empty(len(pos), dtype=np.int64)
            prev[0] = last_open
            prev[1:] = pos[:-1]
            gaps = pos - prev
            sel = np.flatnonzero(gaps >= dmin)
            for j in sel:
                out.append((int(gaps[j]), int(prev[j]) + 1))
            last_open = int(pos[-1])
        base += n
    return out


def main():
    src = json.load(open(os.path.join(os.path.dirname(sys.argv[1]), "inuse.json")))
    out = {"machines": []}

    for r in src:
        gears = r["gears"]
        W = r["W"]
        gc = {int(k): v for k, v in r["gap_counts"].items()}
        fo = {int(k): v for k, v in r["first_occurrence"].items()}
        scale = r["scanned"] / W  # counts are over the prefix; rescale to a period
        dmax = max(gc)
        tail = {}
        s = 0
        for d in sorted(gc, reverse=True):
            s += gc[d]
            tail[d] = s / scale  # number of gaps >= d per period
        rows = []
        for d in sorted(gc):
            if d < 6:
                continue
            c = tail[d]
            rows.append(
                {
                    "d": d,
                    "count_per_period": round(gc[d] / scale, 1),
                    "count_ge_d_per_period": round(c, 1),
                    "W_over_c": W / c,
                    "first_occurrence": fo.get(d),
                    "ratio_first_over_W_c": (fo[d] / (W / c)) if d in fo else None,
                }
            )
        # predicted ladder against the measured one
        ladder = []
        for e in r["record_ladder"]:
            N = e["N"]
            pred = max([d for d in tail if W / tail[d] <= N], default=1) - 1
            ladder.append({"N": N, "F_measured": e["F_range"], "F_predicted": pred,
                           "diff": e["F_range"] - pred})
        out["machines"].append(
            {
                "gears": gears,
                "W": W,
                "scanned": r["scanned"],
                "F_wheel_or_seen": r["F_seen"],
                "table": rows,
                "ladder": ladder,
                "N_to_reach_top": fo.get(dmax),
                "N_to_reach_top_over_W": fo.get(dmax) / W if dmax in fo else None,
                "top_multiplicity_per_period": round(gc[dmax] / scale, 2),
            }
        )
        print("===", gears, "W=%.4g" % W, "F=%d" % r["F_seen"])
        for x in rows[-10:]:
            print(
                "   d=%2d  per period %12.1f  >=d %12.1f  W/c=%12.4g  first=%12s  ratio=%s"
                % (x["d"], x["count_per_period"], x["count_ge_d_per_period"], x["W_over_c"],
                   x["first_occurrence"],
                   ("%.2f" % x["ratio_first_over_W_c"]) if x["ratio_first_over_W_c"] else "-")
            )
        print("   ladder", [(e["N"], e["F_measured"], e["F_predicted"]) for e in ladder])

    # census formula against the exact 8-gear full-period census
    g8 = src[0]["gears"]
    gc8 = {int(k): v for k, v in src[0]["gap_counts"].items()}
    chk = []
    for d in range(1, 17):
        f = census_formula(g8, d)
        chk.append({"d": d, "formula": f, "measured": gc8.get(d, 0), "ok": f == gc8.get(d, 0)})
        print("census formula d=%2d  formula %-14d measured %-14d ok=%s"
              % (d, f, gc8.get(d, 0), f == gc8.get(d, 0)))
    out["census_formula_check_8gear"] = chk
    out["census_formula_mismatches"] = sum(1 for x in chk if not x["ok"])

    # where the record blocks sit
    gears = g8
    W = prod(gears)
    longs = positions_of_long_gaps(gears, W, 30)
    Wsmall = gears[0] * gears[1] * gears[2]
    by = {}
    for d, p in longs:
        by.setdefault(d, []).append(p)
    rec = {}
    for d in sorted(by):
        ps = sorted(by[d])
        rec[d] = {
            "count": len(ps),
            "positions": ps[:40],
            "min_over_W": ps[0] / W,
            "distance_to_nearest_small_wheel_multiple": [
                int(min(p % Wsmall, Wsmall - (p % Wsmall))) for p in ps[:40]
            ],
        }
        print("gap %d: %d blocks, first at %d (%.4f of the period)" % (d, len(ps), ps[0], ps[0] / W))
        print("   positions", ps[:20])
    out["long_gap_positions"] = rec
    out["small_wheel"] = Wsmall

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
