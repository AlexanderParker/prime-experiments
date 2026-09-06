"""Sections 4 and 5: the smallest gears as an anchor, and the removal law.

4. The corridor of the smallest gears, uniform descent, the palindrome, the metric anchor.
5. Removal (raising the split): what divides, what grows, what is untouched.

usage: uv run python research/topmachine/r2/anchor_removal.py results/anchor_removal.json
"""

import json
import os
import sys
from itertools import combinations
from math import prod

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r1"))
from cover import F_cover  # noqa: E402


def open_mask(gears, W):
    a = np.ones(W, dtype=bool)
    for g in gears:
        a[0::g] = False
        a[(g - 2) % g :: g] = False
    return a


def cyclic_runs(mask):
    W = len(mask)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0
    d = np.diff(idx)
    lens = []
    cur = 1
    for x in d:
        if x == 1:
            cur += 1
        else:
            lens.append(cur)
            cur = 1
    lens.append(cur)
    if mask[0] and mask[-1] and len(lens) > 1:
        lens[0] += lens[-1]
        lens.pop()
    return max(lens)


def longest_chain2(mask):
    W = len(mask)
    best = 0
    cur = np.ones(W, dtype=bool) & mask
    L = 0
    acc = mask.copy()
    while acc.any() and L < 200:
        L += 1
        best = L
        acc = acc & np.roll(mask, -2 * L)
    return best


def main():
    out = {}

    # ---------- 4. the corridor of the smallest gears ----------
    corr = []
    for small in [[7, 11], [11, 13], [13, 17], [17, 19], [7, 11, 13], [11, 13, 17], [13, 17, 19]]:
        Ws = prod(small)
        ms = open_mask(small, Ws)
        slots = int(ms.sum())
        pred = prod(g - 2 for g in small)
        corr.append(
            {
                "small_wheel_gears": small,
                "W_small": Ws,
                "corridor_slots": slots,
                "prod_g_minus_2": pred,
                "match": slots == pred,
                "corridor_density": slots / Ws,
            }
        )
        print("corridor", small, "slots", slots, "= prod(g-2)", pred, "density %.4f" % (slots / Ws))
    out["corridor"] = corr

    # the same count for the bottom machine's anchor, in the SAME pair coordinate:
    # gear 2 has its two teeth COLLAPSED (0 = -2 mod 2), so it leaves g - 1 = 1 slot.
    anch = []
    for small in [[2], [3], [2, 3], [2, 3, 5], [5]]:
        Ws = prod(small)
        ms = open_mask(small, Ws)
        anch.append(
            {
                "gears": small,
                "W": Ws,
                "slots": int(ms.sum()),
                "prod_g_minus_2": prod(g - 2 for g in small),
                "density": float(ms.mean()),
                "teeth_collapse": [g for g in small if (0 % g) == ((-2) % g)],
            }
        )
        print("anchor-formula check", small, "slots", int(ms.sum()),
              "prod(g-2)", prod(g - 2 for g in small), "density %.4f" % ms.mean(),
              "collapsed teeth at", anch[-1]["teeth_collapse"])
    out["small_gear_slot_counts"] = anch

    # uniform descent: every open pair of a bigger wheel lies in a corridor slot,
    # and each slot carries exactly prod_{larger}(g - 2) of them
    desc = []
    for small, extra in [
        ([7, 11], [13, 17]),
        ([11, 13], [17, 19]),
        ([13, 17], [19, 23]),
        ([7, 11, 13], [17, 19]),
        ([11, 13, 17], [19, 23]),
    ]:
        gears = small + extra
        W = prod(gears)
        Ws = prod(small)
        m = open_mask(gears, W)
        idx = np.flatnonzero(m)
        res = idx % Ws
        cnt = np.bincount(res, minlength=Ws)
        occupied = np.flatnonzero(cnt)
        expect = prod(g - 2 for g in extra)
        ms = open_mask(small, Ws)
        ok_slots = np.array_equal(np.flatnonzero(ms), occupied)
        ok_uniform = bool((cnt[occupied] == expect).all())
        desc.append(
            {
                "small": small,
                "gears": gears,
                "slots_occupied": len(occupied),
                "corridor_slots": int(ms.sum()),
                "same_slots": ok_slots,
                "per_slot": int(cnt[occupied][0]),
                "expected_per_slot": expect,
                "uniform": ok_uniform,
            }
        )
        print("descent", gears, "slots used", len(occupied), "of", int(ms.sum()),
              "same:", ok_slots, "per slot", int(cnt[occupied][0]), "expected", expect,
              "uniform:", ok_uniform)
    out["uniform_descent"] = desc

    # palindrome: the gap word of a wheel read cyclically starting at the shield
    pal = []
    for gears in [[7, 11], [11, 13], [7, 11, 13], [11, 13, 17], [13, 17, 19], [7, 11, 13, 17]]:
        W = prod(gears)
        m = open_mask(gears, W)
        idx = np.flatnonzero(m)
        shield = (W - 1) % W
        j = int(np.searchsorted(idx, shield))
        assert idx[j] == shield
        rot = np.concatenate((idx[j:], idx[:j] + W))
        gaps = np.diff(np.concatenate((rot, [rot[0] + W]))).tolist()
        is_pal = gaps == gaps[::-1]
        pal.append({"gears": gears, "W": W, "gap_word_head": gaps[:24],
                    "palindrome_from_shield": is_pal, "length": len(gaps)})
        print("palindrome", gears, "gap word from the shield is a palindrome:", is_pal,
              "head", gaps[:16])
    out["palindrome"] = pal

    # the metric anchor: ceilings depend on q' alone
    met = []
    for gears in [
        [7, 11, 13], [7, 13, 19], [7, 17, 23], [7, 11, 13, 17],
        [11, 13, 17], [11, 19, 29], [11, 13, 17, 19],
        [13, 17, 19], [13, 23, 31], [13, 17, 19, 23],
        [17, 19, 23], [17, 29, 37],
        [19, 23, 29], [23, 29, 31],
    ]:
        W = prod(gears)
        m = open_mask(gears, W)
        run = cyclic_runs(m)
        ch = longest_chain2(m)
        q0 = gears[0]
        clump = 2 * (q0 - 3) + 1
        # measure the clump directly
        k = 0
        for n in range(-(q0 - 1), q0 - 2):
            if n not in (0, -2) and m[n % W]:
                k += 1
        met.append(
            {
                "gears": gears,
                "longest_run": run,
                "q_prime_minus_3": q0 - 3,
                "longest_chain2": ch,
                "q_prime_minus_2": q0 - 2,
                "clump_slots": k,
                "predicted_clump": clump,
                "ok": run == q0 - 3 and ch == q0 - 2 and k == clump,
            }
        )
        print("metric", gears, "run", run, "= q'-3", q0 - 3, "| chain", ch, "= q'-2", q0 - 2,
              "| clump", k, "=", clump, "ok", met[-1]["ok"])
    out["metric_anchor"] = met
    out["metric_exceptions"] = sum(1 for x in met if not x["ok"])

    # ---------- 5. the removal law ----------
    rem = []
    for gears in [
        [11, 13, 17, 19, 23],
        [13, 17, 19, 23, 29],
        [17, 19, 23, 29, 31],
        [7, 11, 13, 17, 19],
    ]:
        chain = []
        for k in range(len(gears)):
            G = gears[k:]
            W = prod(G)
            row = {
                "gears": G,
                "m": len(G),
                "W": W,
                "open": prod(g - 2 for g in G),
                "dominoes": prod(g - 4 for g in G),
                "run_ceiling": G[0] - 3,
                "chain_ceiling": G[0] - 2,
                "clump_slots": 2 * (G[0] - 3) + 1,
            }
            if len(G) >= 2:
                f, st = F_cover(G)
                row["F_top"] = f
                row["status"] = st
                row["parity_prediction"] = 2 * len(G) - (len(G) % 2)
                row["large_gear_regime"] = G[0] > 2 * len(G) + 1
            chain.append(row)
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            a["W_divides"] = a["W"] % b["W"] == 0 and a["W"] // b["W"] == a["gears"][0]
            a["open_divides"] = a["open"] // b["open"] == a["gears"][0] - 2
            a["dom_divides"] = a["dom_ok"] = a["dominoes"] // b["dominoes"] == a["gears"][0] - 4
            a["F_step"] = (a.get("F_top"), b.get("F_top"))
        rem.append(chain)
        print("removal chain from", gears)
        for row in chain:
            print("   ", row["gears"], "W=%d" % row["W"], "open=%d" % row["open"],
                  "dom=%d" % row["dominoes"], "run<=%d chain<=%d clump=%d"
                  % (row["run_ceiling"], row["chain_ceiling"], row["clump_slots"]),
                  "F=%s (parity %s, large-gear %s)"
                  % (row.get("F_top"), row.get("parity_prediction"), row.get("large_gear_regime")),
                  "W divides:", row.get("W_divides"), "open divides:", row.get("open_divides"),
                  "dominoes divide:", row.get("dom_divides"))
    out["removal_chains"] = rem

    # sharp form: F(G \ {g}) is the same for every g, in the large-gear regime
    sharp = []
    for gears in [
        [11, 13, 17, 19], [13, 17, 19, 23], [17, 19, 23, 29], [19, 23, 29, 31],
        [13, 17, 19, 23, 29], [17, 19, 23, 29, 31], [23, 29, 31, 37, 41],
        [17, 19, 23, 29, 31, 37], [29, 31, 37, 41, 43, 47],
        [7, 11, 13, 17], [7, 11, 13, 17, 19],
    ]:
        m = len(gears)
        fs = {}
        for g in gears:
            G = [x for x in gears if x != g]
            f, st = F_cover(G)
            fs[g] = f
        base, st = F_cover(gears)
        allsame = len(set(fs.values())) == 1
        sharp.append(
            {
                "gears": gears,
                "F": base,
                "F_after_removal": fs,
                "all_equal": allsame,
                "large_gear_regime": gears[0] > 2 * m + 1,
                "parity_drop": base - list(fs.values())[0] if allsame else None,
            }
        )
        print("removal-independence", gears, "F=%d" % base, "->", fs,
              "all equal:", allsame, "large-gear:", gears[0] > 2 * m + 1)
    out["removal_independence"] = sharp

    # nesting ratio
    nest = []
    for gears in [[11, 13, 17], [13, 17, 19, 23], [7, 11, 13, 17]]:
        W = prod(gears)
        big = open_mask(gears, W)
        small = open_mask(gears[1:], W)
        sub = bool((big & ~small).sum() == 0)
        nest.append(
            {
                "gears": gears,
                "removed": gears[0],
                "subset": sub,
                "ratio": int(big.sum()) / int(small.sum()),
                "predicted_ratio": 1 - 2 / gears[0],
            }
        )
        print("nesting", gears, "subset:", sub, "ratio %.6f" % nest[-1]["ratio"],
              "predicted %.6f" % nest[-1]["predicted_ratio"])
    out["nesting"] = nest

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
