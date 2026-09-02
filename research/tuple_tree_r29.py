"""Tuple side of the section trees - manager, round 29 (exploratory follow-up).

The fusion tree of a blocked run, read bottom-up, is a sequence of merge events: a kill at
offset o by gear g joins the piece of length a on its left and the piece of length b on its
right into one piece of length a + 1 + b (a or b is 0 when the kill lands next to an already
blocked slot). This script forgets the gears and looks only at the pieces: the (a, b) pairs,
their balance, how the tuple evolves, and whether the tuple trees of different sections are
the same object.

Usage: python tuple_tree_r29.py [--qmax 5003]
"""
import argparse
from collections import Counter

import numpy as np

from word_tree_r29 import spf_sieve, death_rungs, runs_of


def merges_of_run(sub):
    """Bottom-up merge events of the run word sub (death rungs, all > 0).

    Returns list of (gear, a, b, kind) in order of increasing gear, where kind is
    'ext' (a == 0 or b == 0: kill extends a piece or starts one) or 'join' (both > 0).
    Within one gear the kills are applied left to right on the tuple that exists before the gear.
    """
    n = len(sub)
    blocked = np.zeros(n, dtype=bool)
    events = []
    for g in sorted(set(sub.tolist())):
        for o in np.flatnonzero(sub == g):
            a = 0
            i = o - 1
            while i >= 0 and blocked[i]:
                a += 1; i -= 1
            b = 0
            i = o + 1
            while i < n and blocked[i]:
                b += 1; i += 1
            kind = "join" if a > 0 and b > 0 else "ext"
            events.append((int(g), a, b, kind))
            blocked[o] = True
    return events


def tuple_after(sub, gmax):
    blk = sub <= gmax
    return tuple(e - s + 1 for s, e in runs_of(blk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmax", type=int, default=5003)
    a = ap.parse_args()
    primes = [int(p) for p in np.flatnonzero(spf_sieve(a.qmax + 100) == np.arange(a.qmax + 101)) if p >= 5]
    ps = [p for p in primes if p <= a.qmax]
    Wmax = (ps[-1] ** 2 - 2) // 6
    spf = spf_sieve(6 * Wmax + 1)
    r_all = death_rungs(Wmax, spf)

    bands = [(5, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 5003)]
    per_band = {b: dict(n=0, ext=0, join=0, ratios=[], ratios_top=[], one=0, one_top=0, bal_top=[]) for b in bands}
    top_tuples = Counter()   # the 3-tuple at the top, normalised to fractions of the run, rounded
    pair_hist = Counter()    # (min, max) of join pairs, pooled q >= 1000
    join_by_stage = {}       # stage = fraction of the tree height (by merge index) -> ratios
    print("=== tuple trees of the maximal run per section ===")
    for i in range(len(ps) - 1):
        p, q = ps[i], ps[i + 1]
        k_lo = p * p // 6 + 1
        k_hi = (q * q - 2) // 6
        r = r_all[k_lo - 1:k_hi]
        rs = runs_of(r > 0)
        if not rs:
            continue
        s, e = max(rs, key=lambda t: t[1] - t[0])
        sub = r[s:e + 1]
        ev = merges_of_run(sub)
        band = next(b for b in bands if b[0] <= q <= b[1])
        d = per_band[band]
        m = len(ev)
        joins = [(j, x) for j, x in enumerate(ev) if x[3] == "join"]
        d["n"] += m
        d["ext"] += m - len(joins)
        d["join"] += len(joins)
        for j, (g, a_, b_, _) in joins:
            lo, hi = min(a_, b_), max(a_, b_)
            d["ratios"].append(lo / hi)
            if lo == 1:
                d["one"] += 1
            if j >= 0.75 * m:
                d["ratios_top"].append(lo / hi)
                if lo == 1:
                    d["one_top"] += 1
            if q >= 1000:
                pair_hist[(min(lo, 8), min(hi, 8))] += 1
                stage = min(int(10 * j / m), 9)
                join_by_stage.setdefault(stage, []).append(lo / hi)
        # top 3-tuple as fractions of the run
        present = sorted(set(sub.tolist()), reverse=True)
        if len(present) >= 3:
            t3 = tuple_after(sub, present[2])
            if len(t3) == 3:
                L = len(sub)
                top_tuples[tuple(round(x / L, 1) for x in t3)] += 1
        # the last three merges' balance
        for g, a_, b_, kind in ev[-3:]:
            if kind == "join":
                d["bal_top"].append(min(a_, b_) / max(a_, b_))
        if q in (31, 997):
            print(f"  section {p} -> {q}: run {len(sub)}, {m} merges, {len(joins)} joins")
            for g, a_, b_, kind in ev:
                if kind == "join":
                    print(f"    gear {g:>4}: [{a_}] + 1 + [{b_}] -> {a_ + 1 + b_}")
            print(f"    tuple after gear 7:  {tuple_after(sub, 7)}")
            print(f"    tuple after gear 13: {tuple_after(sub, 13)}")

    print("\n=== merge statistics by q range (maximal run per section) ===")
    print(f"{'q range':>12} {'merges':>7} {'ext':>6} {'join':>6} {'join frac':>9} {'med ratio':>9} "
          f"{'mean ratio':>10} {'min=1':>6} {'top quarter: med ratio':>22} {'min=1':>6} {'last3 med':>9}")
    for b in bands:
        d = per_band[b]
        if not d["join"]:
            continue
        print(f"{b[0]:>5}-{b[1]:<6} {d['n']:>7} {d['ext']:>6} {d['join']:>6} {d['join'] / d['n']:>9.3f} "
              f"{np.median(d['ratios']):>9.3f} {np.mean(d['ratios']):>10.3f} {d['one'] / d['join']:>6.3f} "
              f"{np.median(d['ratios_top']):>22.3f} {d['one_top'] / max(1, len(d['ratios_top'])):>6.3f} "
              f"{np.median(d['bal_top']) if d['bal_top'] else 0:>9.3f}")
    print("\n  join ratio min/max by stage of the tree (tenths of the merge sequence, q >= 1000):")
    print("   stage:      " + " ".join(f"{k:>6}" for k in sorted(join_by_stage)))
    print("   median:     " + " ".join(f"{np.median(join_by_stage[k]):>6.3f}" for k in sorted(join_by_stage)))
    print("   mean:       " + " ".join(f"{np.mean(join_by_stage[k]):>6.3f}" for k in sorted(join_by_stage)))
    print("   count:      " + " ".join(f"{len(join_by_stage[k]):>6}" for k in sorted(join_by_stage)))
    print("\n  join pairs (min, max) capped at 8, pooled q >= 1000, top 15:")
    tot = sum(pair_hist.values())
    for (lo, hi), c in pair_hist.most_common(15):
        print(f"    ({lo}, {hi}): {c:>6}  {c / tot:.3f}")
    print("\n  top 3-tuple as fractions of the run, most common (all sections with >= 3 levels):")
    tot = sum(top_tuples.values())
    for t, c in top_tuples.most_common(10):
        print(f"    {t}: {c:>4}  {c / tot:.3f}")
    print(f"  distinct top 3-tuples: {len(top_tuples)} of {tot}")


if __name__ == "__main__":
    main()
