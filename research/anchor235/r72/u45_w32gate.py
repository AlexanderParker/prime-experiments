"""u45_w32gate.py -- gate the census instrument and the first-hit formula against W32's own
published numbers (research/proof/top_machine_2.md 3.7).

Two checks, both against numbers already on the record:
  (a) the exact full-period gap census of the eight-gear wheel {7, 11, 13, 17, 19, 23, 29, 31},
      W = 6,685,349,671, 2,075,517,675 open pairs, the counts listed at d = 1..13 and 26..33;
  (b) the predicted first-hit ladder F_range(N) = max{d : W/c(d) <= N} - 1 at
      N = 10^4 .. 10^10 for the three wheels, against the published predicted rows.

Usage: uv run python research/anchor235/r72/u45_w32gate.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

from u45_census import c_of_gears  # noqa: E402

W731 = [7, 11, 13, 17, 19, 23, 29, 31]
W1341 = [13, 17, 19, 23, 29, 31, 37, 41]
W1947 = [19, 23, 29, 31, 37, 41, 43, 47]

PUBLISHED_CENSUS = {1: 472665375, 2: 862511104, 3: 159289858, 5: 184619162, 6: 144217264,
                    7: 103162336, 8: 33487932, 9: 40269000, 10: 26762220, 11: 18694656,
                    12: 12247392, 13: 9432336, 26: 3296, 27: 1036, 28: 688, 29: 116,
                    30: 92, 31: 100, 32: 12, 33: 8}
PUBLISHED_PRED = {
    "7..31": [17, 20, 24, 27, 30, 32, 32],
    "13..41": [12, 14, 16, 17, 18, 18, 18],
    "19..47": [10, 12, 13, 15, 16, 16, 16],
}
PUBLISHED_MEAS = {
    "7..31": [19, 21, 24, 27, 30, 32, 32],
    "13..41": [12, 15, 16, 17, 18, 18, 18],
    "19..47": [11, 12, 14, 14, 16, 16, 16],
}
NS = [10 ** k for k in range(4, 11)]


def census(gears, dmax):
    out = {}
    for d in range(1, dmax + 1):
        v, _ = c_of_gears(gears, d, teeth=(0, -2))
        out[d] = v
        if v == 0:
            break
    return out


def first_hit(c, W, N):
    last = 0
    for d in sorted(c):
        if c[d] and c[d] * N >= W:
            last = d
    return last - 1


def main():
    rep = {}
    t0 = time.time()
    for name, gears, dmax in (("7..31", W731, 36), ("13..41", W1341, 22),
                              ("19..47", W1947, 20)):
        W = 1
        for g in gears:
            W *= g
        c = census(gears, dmax)
        m = {d: c[d] - c.get(d + 1, 0) for d in c}
        F = max(d for d in c if c[d])
        pred = [first_hit(c, W, N) for N in NS]
        row = {"gears": gears, "W": W, "openings": c[1], "F_top": F,
               "pred": pred, "published_pred": PUBLISHED_PRED[name],
               "published_measured": PUBLISHED_MEAS[name],
               "m": {str(k): v for k, v in m.items() if v}}
        if name == "7..31":
            bad = {d: (PUBLISHED_CENSUS[d], m.get(d, 0)) for d in PUBLISHED_CENSUS
                   if PUBLISHED_CENSUS[d] != m.get(d, 0)}
            row["census_gate"] = "ALL 20 PUBLISHED COUNTS AGREE" if not bad else bad
        rep[name] = row
        print(f"{name}: W = {W:,}, openings = {c[1]:,}, F_top = {F}", flush=True)
        if "census_gate" in row:
            print(f"  census gate: {row['census_gate']}", flush=True)
        print(f"  first hit  {pred}", flush=True)
        print(f"  published  {PUBLISHED_PRED[name]}   (measured {PUBLISHED_MEAS[name]})",
              flush=True)
    rep["secs"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "w32gate.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
