"""Validation: the covering formulation of the record (cover.py) against an exact
full-period scan.  15 wheels, chunked scan so memory stays under 40 MB."""

import json
import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from cover import F_cover  # noqa: E402


def F_scan(gears, limit=4 * 10**8):
    """Longest run of consecutive struck pairs over the whole period, cyclic."""
    W = prod(gears)
    if W > limit:
        return None
    CH = 1 << 22
    best, run, first_run, pos, beststart = 0, 0, None, 0, -1
    while pos < W:
        n = min(CH, W - pos)
        a = np.ones(n, dtype=bool)
        for g in gears:
            for t in (0, (g - 2) % g):
                a[(t - pos) % g :: g] = False
        idx = np.flatnonzero(a)
        if len(idx) == 0:
            run += n
            if run > best:
                best, beststart = run, pos + n - run
        else:
            if run + int(idx[0]) > best:
                best, beststart = run + int(idx[0]), pos - run
            if len(idx) > 1:
                d = np.diff(idx) - 1
                j = int(np.argmax(d))
                if d[j] > best:
                    best, beststart = int(d[j]), pos + int(idx[j]) + 1
            run = n - 1 - int(idx[-1])
            if first_run is None:
                first_run = int(idx[0])
        pos += n
    if first_run is not None and run + first_run > best:
        best, beststart = run + first_run, W - run
    return best, beststart


def main():
    wheels = [
        [7, 11, 13], [11, 13, 17], [13, 17, 19], [17, 19, 23], [19, 23, 29], [23, 29, 31],
        [7, 11, 13, 17], [11, 13, 17, 19], [13, 17, 19, 23], [17, 19, 23, 29],
        [7, 11, 13, 17, 19], [11, 13, 17, 19, 23], [13, 17, 19, 23, 29],
        [17, 19, 23, 29, 31], [19, 23, 29, 31, 37],
    ]
    out = []
    agree = 0
    for gs in wheels:
        fc, st = F_cover(gs)
        fs, at = F_scan(gs)
        ok = fc == fs
        agree += ok
        out.append({"gears": gs, "F_cover": fc, "F_scan": fs, "scan_start": at, "agree": ok})
        print(gs, "F_cover", fc, st, "F_scan", fs, "at", at, "agree", ok)
    print("agreements: %d of %d" % (agree, len(wheels)))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
