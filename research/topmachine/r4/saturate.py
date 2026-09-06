"""Growing the largest gear at fixed N, and the walk above the gear zone.

P9   the above-zone record A(q, N): its size, its position, its growth
P10  no saturation: F_range grows with Q because the gear zone is [1, Q]

usage: uv run python research/topmachine/r4/saturate.py
"""

import json
import os
from math import isqrt, log

import numpy as np

from zone import primes_upto, range_record, smooth_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def zone_prediction(q, Q, first_above):
    """max gap of the q-smooth pair list on [1, Q], last gap run to first_above."""
    sp = [n for n in smooth_pairs(q, Q) if n <= Q - 2]
    if not sp:
        return None, None, sp
    best, at = 0, None
    for a, b in zip(sp, sp[1:]):
        if b - a - 1 > best:
            best, at = b - a - 1, a + 1
    last = first_above - sp[-1] - 1
    if last > best:
        best, at = last, sp[-1] + 1
    return best, at, sp


def main():
    # ---------- P9: the above-zone record ----------
    say("# Above the gear zone, and growing the largest gear")
    say()
    say("## P9  the walk above the gear zone, x in (Q, N], Q = floor(sqrt N)")
    say()
    say("| q | N | Q | F_range | zone record | above-zone record A | at | A / (log N)^2 | "
        "which wins |")
    say("|---|---|---|---|---|---|---|---|---|")
    rows9 = []
    for q in (5, 7, 11, 13, 17, 19, 23, 29, 37):
        for N in (10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8):
            Q = isqrt(N)
            gears = [int(p) for p in primes_upto(Q) if p > q]
            st = range_record(gears, N, Q)
            A, Aat = st["best_above"]
            Z, Zat = st["best_zone"]
            rows9.append({"q": q, "N": N, "Q": Q, "F": st["best"][0], "zone": Z,
                          "above": A, "above_at": Aat, "zone_at": Zat})
            say(f"| {q} | 1e{len(str(N))-1} | {Q:,} | {st['best'][0]:,} | {Z:,} | {A:,} | "
                f"{Aat:,} | {A / log(N) ** 2:.3f} | {'zone' if Z >= A else 'above'} |")
    say()

    # ---------- P10: fixed N, growing Q ----------
    say("## P10  fixed N = 10^7, the largest gear Q swept; is there saturation?")
    say()
    N = 10 ** 7
    for q in (5, 11, 19):
        say(f"**q = {q}**, N = 10^7 (sqrt N = 3,162)")
        say()
        say("| Q | m | F_range | at | zone record (blocks starting <= Q) | at | "
            "record above Q | at | predicted zone record from the smooth list |")
        say("|---|---|---|---|---|---|---|---|---|")
        for Q in (100, 316, 1000, 3162, 10 ** 4, 31623, 10 ** 5, 10 ** 6):
            gears = [int(p) for p in primes_upto(min(Q, N)) if p > q]
            if not gears:
                continue
            st = range_record(gears, N, min(Q, N))
            pred, pat, _ = zone_prediction(q, min(Q, N), st["first_above_Q"] or (Q + 1))
            say(f"| {Q:,} | {len(gears)} | {st['best'][0]:,} | {st['best'][1]:,} | "
                f"{st['best_zone'][0]:,} | {st['best_zone'][1]:,} | "
                f"{st['best_above'][0]:,} | {st['best_above'][1]:,} | "
                f"{pred if pred is not None else '-'} |")
        say()

    # ---------- P10b: does the walk above a FIXED region saturate as Q grows? ----------
    say("## P10b  a fixed test region (10^5, 10^7]: adding gears above 10^5")
    say()
    say("| q | Q | m | max walk on (10^5, 10^7] | at | density on the region |")
    say("|---|---|---|---|---|---|")
    for q in (5, 11, 19):
        for Q in (3162, 10 ** 4, 31623, 10 ** 5, 3 * 10 ** 5, 10 ** 6):
            gears = [int(p) for p in primes_upto(Q) if p > q]
            st = range_record(gears, N, 10 ** 5)
            say(f"| {q} | {Q:,} | {len(gears)} | {st['best_above'][0]:,} | "
                f"{st['best_above'][1]:,} | {st['count'] / N:.6f} |")
        say()

    json.dump(rows9, open(os.path.join(RES, "saturate.json"), "w"), indent=1)
    with open(os.path.join(RES, "saturate.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
