"""lf_gate.py -- validation of this lane's sieve against numbers on the record.

Gates (research/proof/first_realisation.md 3.1, 3.3; the corpus):
  F(11) = 7, F(13) = 11, F(17) = 18, F(19) = 25, F(23) = 34, F(29) = 43 (m29 on its half period is
  2 x 10^8 columns; scanned here in full), and the FIRST record run of m23 (length 33) begins at
  column 12,694,429.  Then the recorded record positions of m29 (200,906,186; run 42),
  m31 (1,468,940,243; run 57), m37 (90,816,580,903; run 87) are re-verified by a direct sieve of
  the neighbourhood: the column before is open, the L columns are struck, the column after is open.

Usage: uv run python lf_gate.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lf_common import period, struck_segment, runs_from_struck  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

RECORDS = {  # p: (first record run start, run length)  -- first_realisation.md 3.3
    23: (12_694_429, 33),
    29: (200_906_186, 42),
    31: (1_468_940_243, 57),
    37: (90_816_580_903, 87),
}
EXPECT_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43}


def scan_half_period(p, chunk=1 << 24):
    P = period(p)
    limit = P // 2 + 2048
    gmax, xfirst = 0, None
    x = 1
    over = 2048
    while x < limit:
        n = min(chunk, limit - x)
        arr = struck_segment(p, x - 1, n + over)
        arr[0] = False  # column x-1: we only want runs starting at or after x; treat as open
        # careful: arr[0] corresponds to column x-1; but for x = 1 column 0 is open anyway.
        if x > 1:
            arr[0] = struck_segment(p, x - 1, 1)[0]
        for (s, L) in runs_from_struck(arr, x - 1):
            if s < x or s >= x + n:
                continue
            if L > gmax:
                gmax, xfirst = L, s
        x += n
    return gmax, xfirst


def verify_neighbourhood(p, x, L):
    arr = struck_segment(p, x - 1, L + 2)
    return (not arr[0]) and bool(arr[1:L + 1].all()) and (not arr[L + 1])


def main():
    out = {}
    t0 = time.time()
    for p in [11, 13, 17, 19, 23, 29]:
        gmax, xfirst = scan_half_period(p)
        F = gmax + 1
        ok = (F == EXPECT_F[p])
        line = f"m{p}: F = {F} (expected {EXPECT_F[p]}) first record run at column {xfirst:,}  [{'OK' if ok else 'MISMATCH'}]  t = {time.time() - t0:.1f}s"
        print(line, flush=True)
        out[f"m{p}"] = {"F": F, "x_first": xfirst, "ok": ok}
    ok23 = out["m23"]["x_first"] == RECORDS[23][0]
    print(f"m23 first record start {out['m23']['x_first']:,} against 12,694,429: {'OK' if ok23 else 'MISMATCH'}")
    ok29 = out["m29"]["x_first"] == RECORDS[29][0]
    print(f"m29 first record start {out['m29']['x_first']:,} against 200,906,186: {'OK' if ok29 else 'MISMATCH'}")
    for p, (x, L) in RECORDS.items():
        v = verify_neighbourhood(p, x, L)
        print(f"m{p}: run of {L} at {x:,} with open columns on both sides: {v}")
        out[f"m{p}_record_verified"] = bool(v)
    with open(os.path.join(RES, "gate.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
