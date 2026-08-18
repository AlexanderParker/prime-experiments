"""Round 10 lateral: UNIFORMITY of top-gap addresses - the drift law and the
word-pinning law.

Two candidate laws, both tested exactly:

LAW A (word-pinning): the neighbourhood WORD of a top gap (the pattern of
openings in a window around it) determines its address class mod 5*7*11(*13)
up to a small compatible set: each opening forbids 2 offsets per small gear
(it must avoid both teeth), so enough openings pin the phase. If the observed
addresses equal the compatible set, then

    #top-stratum classes <= sum over near-top words of #compatible phases

and uniformity in y reduces to (i) non-growing word counts (measured, round
9) and (ii) a uniform per-word phase bound (measured here). No recursion
needed - the address is locally word-determined.

LAW B (drift recursion): new maximal-gap left address = (old top-stratum
left address) - (left flank of the merge word), flanks from the finite
alphabet {1..5}. Spot-checked from round-9 data (47-2=45, 122-5=117,
115-5=110, 252-2=250, 322-2=320); verified systematically here.

Machines y = 13..29 (y=29 streamed, period 1.078e9), thresholds at 0.9 F with
F = 11, 18, 25, 34, 43 (round 9, exact).

Run: uv run python research/address_drift.py    (repo root; numpy)
"""
from collections import defaultdict
from math import prod

import numpy as np

from split_gap_law import primes
from topgap_corridor import chunk_openings
from topgap_nesting import local_openings

FKNOWN = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43}

def thresh_gaps(y, T, chunk=20_000_000):
    """All (leftpos, gap) with gap >= T over the full period."""
    gears = primes(5, y)
    P = prod(gears)
    out = []
    carry = None
    a = 0
    while a < P:
        S = min(chunk, P - a)
        ops = chunk_openings(gears, a, S)
        ext = ops if carry is None else np.concatenate((carry, ops))
        d = np.diff(ext)
        for i in np.flatnonzero(d >= T):
            out.append((int(ext[i]), int(d[i])))
        carry = ext[-2:]
        a += S
    return sorted(set(out)), P

def pattern_of(y, t, G, W):
    """Offsets (relative to t) of openings in [t-W, t+G+W]."""
    ops = local_openings(y, t - W, t + G + W + 1)
    return tuple(int(o - t) for o in ops)

def compatible_phases(pattern, gears_small):
    """Per-gear sets of a = t mod q keeping every opening off both teeth."""
    sets = {}
    for q in gears_small:
        u = pow(6, -1, q)
        bad = {(u - s) % q for s in pattern} | {(-u - s) % q for s in pattern}
        sets[q] = sorted(set(range(q)) - bad)
    return sets

def main():
    print("=" * 72)
    print("LAW A: word-pinning - observed addresses vs word-compatible phases")
    summary = []
    strata = {}
    for y in (13, 17, 19, 23, 29):
        F = FKNOWN[y]
        T = int(np.ceil(0.9 * F))
        gaps, P = thresh_gaps(y, T)
        strata[y] = gaps
        groups = defaultdict(list)          # pattern -> [t...]
        for t, G in gaps:
            groups[(G, pattern_of(y, t, G, 20))].append(t)
        maxwords = 0
        pred_total = 0
        obs_total = set()
        fails = 0
        rows = []
        for (G, pat), ts in groups.items():
            cp = compatible_phases(pat, (5, 7, 11))
            npred = len(cp[5]) * len(cp[7]) * len(cp[11])
            obs = {t % 385 for t in ts}
            # containment: every observed address must be word-compatible
            ok = all(t % 5 in cp[5] and t % 7 in cp[7] and t % 11 in cp[11]
                     for t in ts)
            if not ok:
                fails += 1
            pred_total += npred
            obs_total |= obs
            maxwords = max(maxwords, npred)
            rows.append((G, len(ts), len(pat), npred, len(obs)))
        tight = sum(r[4] for r in rows) / pred_total if pred_total else 0
        print(f"  y={y}: near-top gaps {len(gaps)}, distinct (G, word) {len(groups)}, "
              f"containment fails {fails}")
        print(f"        per-word compatible phases mod 385: max {maxwords}; "
              f"sum predicted {pred_total} vs observed classes {len(obs_total)} "
              f"(tightness {tight:.2f})")
        summary.append((y, len(gaps), len(groups), maxwords, pred_total,
                        len(obs_total)))
    print("  UNIFORMITY TABLE: y | near-top | words | max phases/word | "
          "sum pred | observed classes")
    for row in summary:
        print(f"    {row[0]:>3} | {row[1]:>7} | {row[2]:>5} | {row[3]:>15} | "
              f"{row[4]:>8} | {row[5]:>6}")

    print("=" * 72)
    print("LAW B: drift recursion - new max address = old stratum address - flank")
    for yo, yn in ((13, 17), (17, 19), (19, 23), (23, 29)):
        Fo, Fn = FKNOWN[yo], FKNOWN[yn]
        old_addr = {t % 385 for t, G in strata[yo]}          # near-top stratum
        # widen: all old gaps >= 0.5 F_old at the relevant addresses? use stratum
        new_max = [t for t, G in strata[yn] if G == Fn]
        hits = 0
        detail = []
        for t in new_max:
            # left flank of merge word: distance from t to next old opening
            ops = local_openings(yo, t, t + Fn + 1)
            lf = int(ops[1] - ops[0]) if len(ops) > 1 else -1   # first old gap after t
            src = (t + lf) % 385
            ok_src = src in old_addr
            ok_self = t % 385 in old_addr
            hits += ok_src or ok_self
            detail.append((t % 385, lf, src, ok_src, ok_self))
        print(f"  step {yo}->{yn}: {len(new_max)} new maxima; "
              f"address reachable from old near-top stratum (self or +flank): "
              f"{hits}/{len(new_max)}")
        for a, lf, src, ok_src, ok_self in detail[:6]:
            print(f"    new addr {a:>3} (+first old gap {lf:>2} -> {src:>3}): "
                  f"src-in-old {ok_src}, self-in-old {ok_self}")

if __name__ == "__main__":
    main()
