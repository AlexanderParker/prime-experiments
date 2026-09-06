"""Top machine: chains at step 2 (shared-member dominoes), the gap spectrum table,
wheel records for the fixed range machines, and the density near small-wheel multiples."""

import json
import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from cover import F_cover  # noqa: E402


def open_mask(gears, W):
    a = np.ones(W, dtype=bool)
    for g in gears:
        a[0::g] = False
        a[(g - 2) % g :: g] = False
    return a


def main():
    out = {}
    wheels = [
        [7, 11, 13], [11, 13, 17], [13, 17, 19], [17, 19, 23], [19, 23, 29], [23, 29, 31],
        [7, 11, 13, 17], [11, 13, 17, 19], [13, 17, 19, 23], [17, 19, 23, 29],
        [7, 11, 13, 17, 19], [11, 13, 17, 19, 23],
    ]

    # A. chains at step 2 vs runs at step 1
    out["chains"] = []
    for gs in wheels:
        W = prod(gs)
        m = open_mask(gs, W)
        rec = {"gears": gs}
        # step-1 runs of length L: count of starts
        runs = {}
        chains = {}
        for L in range(1, gs[0]):
            acc1 = m.copy()
            acc2 = m.copy()
            for j in range(1, L):
                acc1 &= np.roll(m, -j)
                acc2 &= np.roll(m, -2 * j)
            runs[L] = int(acc1.sum())
            chains[L] = int(acc2.sum())
        rec["run_starts"] = runs
        rec["chain_starts"] = chains
        rec["run_formula"] = {L: prod(max(g - 2 - L, 0) for g in gs) if L >= 2 else
                              prod(g - 2 for g in gs) for L in runs}
        rec["chain_formula"] = {L: (prod(g - 2 for g in gs) if L == 1 else
                                    prod(max(g - 1 - L, 0) for g in gs)) for L in chains}
        rec["run_mismatch"] = {L: (runs[L], rec["run_formula"][L])
                               for L in runs if runs[L] != rec["run_formula"][L]}
        rec["chain_mismatch"] = {L: (chains[L], rec["chain_formula"][L])
                                 for L in chains if chains[L] != rec["chain_formula"][L]}
        rec["max_run"] = max(L for L in runs if runs[L] > 0)
        rec["max_chain"] = max(L for L in chains if chains[L] > 0)
        rec["pred_max_run"] = gs[0] - 3
        rec["pred_max_chain"] = gs[0] - 2
        rec["shared_domino_count"] = chains[2]
        rec["pred_shared_domino"] = prod(g - 3 for g in gs)
        out["chains"].append(rec)
        print(gs, "maxrun", rec["max_run"], "(pred", rec["pred_max_run"], ")",
              "maxchain", rec["max_chain"], "(pred", rec["pred_max_chain"], ")",
              "shared dominoes", rec["shared_domino_count"], "=prod(g-3)",
              rec["shared_domino_count"] == rec["pred_shared_domino"],
              "run/chain formula mismatches", len(rec["run_mismatch"]),
              len(rec["chain_mismatch"]))

    # B. gap spectrum table
    out["gap_spectrum"] = []
    for gs in wheels:
        W = prod(gs)
        m = open_mask(gs, W)
        idx = np.flatnonzero(m)
        gaps = np.concatenate((np.diff(idx), [idx[0] + W - idx[-1]]))
        h = {}
        for L, c in zip(*np.unique(gaps, return_counts=True)):
            h[int(L)] = int(c)
        out["gap_spectrum"].append({
            "gears": gs, "hist": h,
            "missing": [L for L in range(1, max(h) + 1) if L not in h],
            "odd_counts": [L for L, c in h.items() if c % 2],
            "eq_3_5": h.get(3) == h.get(5),
        })
        print(gs, "gaps", h, "missing", out["gap_spectrum"][-1]["missing"],
              "odd", out["gap_spectrum"][-1]["odd_counts"], "n3==n5",
              out["gap_spectrum"][-1]["eq_3_5"])

    # C. wheel records for the fixed range machines of range.py
    PR = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    out["fixed_wheel_records"] = []
    for q in (5, 11, 17):
        gs = [p for p in PR if p > q][:8]
        F, st = F_cover(gs)
        out["fixed_wheel_records"].append({"q": q, "gears": gs, "wheel_F": F,
                                           "W": prod(gs), "status": st})
        print("fixed machine q'>%d" % q, gs, "wheel record", F, "W=", prod(gs), st)

    # D. density near multiples of the small wheel, controlled
    allp = PR
    out["near_wheel"] = []
    N = 10**7
    for q in (5, 11, 17):
        gs = [p for p in allp if p > q][:8]
        a = np.ones(N, dtype=bool)
        for g in gs:
            a[0::g] = False
            a[(g - 2) % g :: g] = False
        base = float(a.mean())
        Wsm = prod(gs[:3])
        row = {"q": q, "gears": gs, "small_wheel": Wsm, "base": base, "windows": {}}
        for L in (5, 20, 100, 500):
            vals = []
            for t in range(1, min(3000, N // Wsm)):
                c = t * Wsm
                if c - L >= 0 and c + L < N:
                    vals.append(float(a[c - L : c + L].mean()))
            row["windows"][L] = {"mean": float(np.mean(vals)), "ratio": float(np.mean(vals)) / base,
                                 "n": len(vals)}
        out["near_wheel"].append(row)
        print("near small wheel q'>%d" % q, {k: round(v["ratio"], 4) for k, v in row["windows"].items()},
              "base", round(base, 5))

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
