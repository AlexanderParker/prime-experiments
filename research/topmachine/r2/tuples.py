"""Section 1: three and four gears at once.

(a) the joint struck set of a real triple / quadruple against the CRT product;
(b) the total-collision classes and the origin;
(c) the three-gear collision law (dominoes of width 2 in a window);
(d) the exact record of every triple and quadruple of odd primes 7..97;
(e) the sub-threshold reduction: what the record depends on.

usage: uv run python research/topmachine/r2/tuples.py results/tuples.json
"""

import json
import os
import sys
from itertools import combinations
from math import prod

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r1"))
from cover import F_cover, witness  # noqa: E402

P = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def strike_masks(gears, W):
    out = []
    for g in gears:
        a = np.zeros(W, dtype=bool)
        a[0::g] = True
        a[(g - 2) % g :: g] = True
        out.append(a)
    return out


def joint_census(gears):
    W = prod(gears)
    ms = strike_masks(gears, W)
    k = np.zeros(W, dtype=np.int8)
    for a in ms:
        k += a
    m = len(gears)
    meas = [int((k == j).sum()) for j in range(m + 1)]
    # CRT prediction: choose which gears strike (2 ways each) and which miss (g-2 ways each)
    pred = [0] * (m + 1)
    for j in range(m + 1):
        s = 0
        for S in combinations(range(m), j):
            s += prod(2 for _ in S) * prod(gears[i] - 2 for i in range(m) if i not in S)
        pred[j] = s
    # total collisions: struck by every gear
    allstruck = np.flatnonzero(k == m)
    # of those, the ones where every gear uses the SAME tooth (whole dominoes coincide)
    same = []
    for n in allstruck:
        teeth = set()
        for g in gears:
            t = 0 if n % g == 0 else 1
            teeth.add(t)
        if len(teeth) == 1:
            same.append(int(n))
    return {
        "gears": gears,
        "W": W,
        "measured_by_strikers": meas,
        "crt_predicted": pred,
        "deviation": [a - b for a, b in zip(meas, pred)],
        "total_collision_classes": len(allstruck),
        "two_to_the_m": 2 ** len(gears),
        "identical_domino_classes": sorted(same),
    }


def collision_check(Lmax=12, gmin=13):
    """Exhaustive: in a window of L < gmin - 1, can three gears' traces pairwise intersect
    with three DISTINCT traces?  Traces are {} , {x} (edge) or {x, x+2}."""
    bad = 0
    tested = 0
    for L in range(1, Lmax + 1):
        traces = []
        for x in range(-2, L):
            t = tuple(sorted(p for p in (x, x + 2) if 0 <= p < L))
            if t:
                traces.append(t)
        traces = sorted(set(traces))
        for a, b, c in combinations(traces, 3):
            tested += 1
            sa, sb, sc = set(a), set(b), set(c)
            if sa & sb and sb & sc and sa & sc:
                bad += 1
    return {"windows_up_to": Lmax, "triples_tested": tested, "pairwise_overlapping_triples": bad}


def cover_waste(gears):
    L, st = F_cover(gears)
    w = witness(gears, L)
    if w is None:
        return {"gears": gears, "L": L, "status": st, "waste": None}
    tot = 0
    for k, v in w.items():
        if k == "pool":
            for mm in v:
                tot += bin(mm).count("1")
        else:
            tot += bin(v).count("1")
    return {
        "gears": gears,
        "m": len(gears),
        "L": L,
        "status": st,
        "positions_covered_with_multiplicity": tot,
        "waste": tot - L,
    }


def main():
    out = {}

    # (a) joint census, real triples and quadruples
    rows = []
    for gs in [
        [7, 11, 13], [11, 13, 17], [13, 17, 19], [17, 19, 23], [19, 23, 29],
        [23, 29, 31], [29, 31, 37], [7, 13, 31], [11, 19, 41],
        [7, 11, 13, 17], [11, 13, 17, 19], [13, 17, 19, 23], [17, 19, 23, 29],
        [7, 13, 19, 31],
    ]:
        r = joint_census(gs)
        rows.append(r)
        print(
            "joint", gs, "by strikers", r["measured_by_strikers"],
            "deviation", r["deviation"],
            "all-struck", r["total_collision_classes"], "=2^m", r["two_to_the_m"],
            "identical dominoes at", r["identical_domino_classes"],
        )
    out["joint_census"] = rows
    out["joint_deviations"] = sum(sum(abs(x) for x in r["deviation"]) for r in rows)
    print("total deviation from the CRT product:", out["joint_deviations"])

    # (b)(c) the collision law
    out["collision_law"] = collision_check()
    print("collision law:", out["collision_law"])

    # waste in the optimal record cover
    ws = []
    for gs in [
        [11, 13], [11, 13, 17], [11, 13, 17, 19], [11, 13, 17, 19, 23],
        [13, 17, 19, 23, 29, 31], [17, 19, 23, 29, 31, 37, 41],
        [19, 23, 29, 31, 37, 41, 43, 47],
        [29, 31, 37, 41, 43], [29, 31, 37, 41, 43, 47],
    ]:
        r = cover_waste(gs)
        ws.append(r)
        print("cover waste", r)
    out["cover_waste"] = ws

    # (d) exhaustive records for triples and quadruples of odd primes 7..97
    for m, key in ((3, "triples"), (4, "quadruples")):
        tab = {}
        vals = {}
        n = 0
        for gs in combinations(P, m):
            f, st = F_cover(list(gs))
            assert st == "exact", (gs, st)
            key7 = 7 in gs
            vals.setdefault((key7, f), 0)
            vals[(key7, f)] += 1
            tab.setdefault(f, []).append(list(gs))
            n += 1
        out[key] = {
            "count": n,
            "records": {str(k): len(v) for k, v in sorted(tab.items())},
            "by_has7": {str(k): v for k, v in sorted(vals.items())},
            "examples": {str(k): v[:3] for k, v in sorted(tab.items())},
        }
        print(key, n, "sets; records", out[key]["records"], "by has-7", out[key]["by_has7"])

    # (e) the sub-threshold reduction: F depends only on m and the gears <= F + 1
    red = []
    tests = []
    for m in range(3, 9):
        # every set of m primes from 7..97 whose small gears (<= 2m+3) are a given pattern
        smalls = [p for p in P if p <= 2 * m + 3]
        bigs = [p for p in P if p > 2 * m + 3]
        for r in range(0, min(len(smalls), m) + 1):
            for S in combinations(smalls, r):
                if m - r > len(bigs):
                    continue
                variants = [
                    list(S) + list(bigs[:m - r]),
                    list(S) + list(bigs[-(m - r):]) if m - r else list(S),
                    list(S) + list(bigs[1:][: m - r]),
                ]
                variants = [v for v in variants if len(v) == m]
                fs = []
                for v in variants:
                    f, st = F_cover(sorted(v))
                    if st != "exact":
                        f = None
                    fs.append(f)
                ok = len(set(fs)) == 1 and None not in fs
                tests.append(ok)
                red.append({"m": m, "small": list(S), "variants": variants, "F": fs, "ok": ok})
                if not ok:
                    print("  REDUCTION EXCEPTION", m, S, variants, fs)
    out["subthreshold_reduction"] = red
    out["reduction_cases"] = len(tests)
    out["reduction_exceptions"] = sum(1 for t in tests if not t)
    print("sub-threshold reduction: %d cases, %d exceptions" % (len(tests), out["reduction_exceptions"]))
    # the rule itself: F as a function of (m, small gears)
    rule = {}
    for r in red:
        if r["ok"]:
            rule["m=%d small=%s" % (r["m"], r["small"])] = r["F"][0]
    out["rule"] = rule
    for k in sorted(rule):
        print("  %-28s F_top = %s" % (k, rule[k]))

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
