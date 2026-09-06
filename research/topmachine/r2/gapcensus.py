"""Section 2: the exact gap census of a top-machine wheel, and the mechanism of W1.

A gap of length d at n means: n open, n + d open, and n + 1 .. n + d - 1 all struck.

Per gear g, write r = n mod g.  Then
  * n open        forbids r in {0, -2}
  * n + d open    forbids r in {-d, -d-2}
  * n + j struck (1 <= j <= d-1) needs SOME gear with r in {-j, -(j+2)}

So with F = {0, -2, -d, -d-2} the count is an inclusion-exclusion over the subsets S of the
interior positions that are left uncovered:

    N_d(G) = sum_{S subset of [1, d-1]} (-1)^{|S|} prod_g ( g - |E_g(S)| )
    E_g(S) = ( {0, -2, -d, -d-2} union {-j, -(j+2) : j in S} )  mod g

This is exact for every gear set of pairwise coprime gears (CRT).  When every gear exceeds
d + 2 no class collapses, |E_g(S)| = e(S) is the same integer for every gear, and N_d is a
UNIVERSAL polynomial in the gears.

usage: uv run python research/topmachine/r2/gapcensus.py results/gapcensus.json
"""

import json
import sys
from itertools import combinations
from math import prod

import numpy as np


def base_forbidden(d):
    return [0, -2, -d, -d - 2]


def esize(g, d, S):
    E = set(x % g for x in base_forbidden(d))
    for j in S:
        E.add((-j) % g)
        E.add((-j - 2) % g)
    return len(E)


def census_formula(gears, d):
    """Exact N_d by inclusion-exclusion.  2^(d-1) terms."""
    tot = 0
    pos = list(range(1, d))
    for k in range(len(pos) + 1):
        sgn = -1 if k % 2 else 1
        for S in combinations(pos, k):
            tot += sgn * prod(g - esize(g, d, S) for g in gears)
    return tot


def signature(d):
    """The universal signature: {e -> signed count} for gears > d + 2 (no collapse)."""
    sig = {}
    pos = list(range(1, d))
    BIG = 10**9  # a modulus larger than every class, so nothing collapses
    for k in range(len(pos) + 1):
        sgn = -1 if k % 2 else 1
        for S in combinations(pos, k):
            e = esize(BIG, d, S)
            sig[e] = sig.get(e, 0) + sgn
    return {k: v for k, v in sorted(sig.items()) if v != 0}


def sig_str(sig):
    if not sig:
        return "0"
    return " ".join(
        ("%+d" % v) + "*prod(g-%d)" % k for k, v in sorted(sig.items())
    )


def measure(gears):
    W = prod(gears)
    a = np.ones(W, dtype=bool)
    for g in gears:
        a[0::g] = False
        a[(g - 2) % g :: g] = False
    idx = np.flatnonzero(a)
    gaps = np.diff(idx)
    gaps = np.concatenate((gaps, [idx[0] + W - idx[-1]]))
    h = {}
    for L, c in zip(*np.unique(gaps, return_counts=True)):
        h[int(L)] = int(c)
    return h


WHEELS = [
    [7, 11, 13],
    [11, 13, 17],
    [13, 17, 19],
    [17, 19, 23],
    [19, 23, 29],
    [23, 29, 31],
    [7, 11, 13, 17],
    [11, 13, 17, 19],
    [13, 17, 19, 23],
    [17, 19, 23, 29],
    [7, 11, 13, 17, 19],
    [11, 13, 17, 19, 23],
    [7, 13, 19, 31],
    [11, 17, 23, 29],
    [7, 11, 17, 23, 29],
]


def main():
    out = {}

    # --- the universal signatures, gap length by gap length
    sigs = {}
    for d in range(1, 17):
        sigs[d] = signature(d)
    out["signatures"] = {str(d): {str(k): v for k, v in s.items()} for d, s in sigs.items()}
    print("universal signatures (gears > d + 2):")
    for d in range(1, 17):
        print("  d=%2d  N_d = %s" % (d, sig_str(sigs[d])))

    # --- which gap lengths share a signature (the W1 question, generalised)
    groups = {}
    for d, s in sigs.items():
        key = tuple(sorted(s.items()))
        groups.setdefault(key, []).append(d)
    coincidences = [v for v in groups.values() if len(v) > 1 and any(sigs[v[0]])]
    out["coincident_gap_lengths"] = coincidences
    out["zero_gap_lengths"] = [d for d in sigs if not sigs[d]]
    print("coincident gap lengths (identical universal polynomial):", coincidences)
    print("identically-zero gap lengths:", out["zero_gap_lengths"])

    # --- collapse thresholds: the largest gear that can break the universal polynomial
    thr = {}
    for d in range(1, 17):
        worst = 0
        pos = list(range(1, d))
        for k in range(len(pos) + 1):
            for S in combinations(pos, k):
                cls = set(base_forbidden(d))
                for j in S:
                    cls.add(-j)
                    cls.add(-j - 2)
                cls = sorted(cls)
                for i in range(len(cls)):
                    for j in range(i + 1, len(cls)):
                        worst = max(worst, abs(cls[i] - cls[j]))
        thr[d] = worst
    out["max_class_difference"] = thr
    print("largest class difference by d (a gear g can collapse only if g divides one):", thr)

    # --- formula against exact measurement, every wheel, every gap length
    rows = []
    bad = 0
    for gears in WHEELS:
        h = measure(gears)
        dmax = max(h)
        row = {"gears": gears, "W": prod(gears), "measured": h, "formula": {}, "ok": True}
        for d in range(1, min(dmax, 15) + 1):
            f = census_formula(gears, d)
            row["formula"][d] = f
            if f != h.get(d, 0):
                row["ok"] = False
                bad += 1
                print("  MISMATCH", gears, "d=", d, "formula", f, "measured", h.get(d, 0))
        rows.append(row)
        print(
            "wheel", gears, "gaps", dict(sorted(h.items())), "formula ok:", row["ok"]
        )
    out["wheels"] = rows
    out["formula_mismatches"] = bad

    # --- W1 explicitly: N_3 and N_5, with and without gear 7
    w1 = []
    for gears in WHEELS:
        n3 = census_formula(gears, 3)
        n5 = census_formula(gears, 5)
        univ3 = prod(g - 4 for g in gears) - 2 * prod(g - 5 for g in gears) + prod(
            g - 6 for g in gears
        )
        w1.append(
            {
                "gears": gears,
                "N3": n3,
                "N5": n5,
                "equal": n3 == n5,
                "has7": 7 in gears,
                "universal_poly": univ3,
                "N3_is_universal": n3 == univ3,
                "N5_is_universal": n5 == univ3,
            }
        )
        print(
            "W1 %-24s N3=%-8d N5=%-8d equal=%-5s 7=%-5s universal=%d (N3 univ %s, N5 univ %s)"
            % (
                gears,
                n3,
                n5,
                n3 == n5,
                7 in gears,
                univ3,
                n3 == univ3,
                n5 == univ3,
            )
        )
    out["W1"] = w1

    # --- the per-gear class tables that carry the mechanism
    tab = {}
    for g in (7, 11, 13, 17, 19, 23):
        tab[g] = {
            "d3_forbidden": sorted(set(x % g for x in [0, -2, -3, -5])),
            "d3_allowed_count": g - len(set(x % g for x in [0, -2, -3, -5])),
            "d5_forbidden": sorted(set(x % g for x in [0, -2, -5, -7])),
            "d5_allowed_count": g - len(set(x % g for x in [0, -2, -5, -7])),
            "marked_d3": [(-1) % g, (-4) % g],
            "marked_d5": [(-3) % g, (-4) % g],
        }
        print(
            "gear %2d: d=3 forbidden %s (%d allowed) | d=5 forbidden %s (%d allowed)"
            % (
                g,
                tab[g]["d3_forbidden"],
                tab[g]["d3_allowed_count"],
                tab[g]["d5_forbidden"],
                tab[g]["d5_allowed_count"],
            )
        )
    out["class_tables"] = tab

    # --- the moments of the signature, and the degree of N_d in the gears
    # prod_g (g - e) = sum_k (-e)^k sigma_{m-k}(gears), so
    #     N_d = sum_k (-1)^k sigma_{m-k}(gears) M_k ,   M_k = sum_e c_e e^k .
    # N_d has degree m - r in the gears, where r is the number of vanishing moments;
    # it is a pure CONSTANT (independent of the gears) exactly when r = m.
    mom = {}
    for d in range(1, 17):
        s = sigs[d]
        Ms = []
        for k in range(0, 20):
            Ms.append(sum(c * (e**k) for e, c in s.items()))
        r = 0
        while r < len(Ms) and Ms[r] == 0:
            r += 1
        mom[d] = {"moments": Ms[: r + 2], "r": r if s else None}
        print("d=%2d  vanishing moments r=%s  first nonzero M_r=%s" % (d, mom[d]["r"], Ms[r] if s else "-"))
    out["moments"] = mom

    # r(d) against the parity-law covering number, and F_top(m) read off the census
    par = {}
    for d in range(1, 17):
        L = d - 1
        pred = -(-((L + 1) // 2) // 2) + -(-(L // 2) // 2)  # ceil(ceil(L/2)/2)+ceil(floor(L/2)/2)
        par[d] = {"r": mom[d]["r"], "parity_bound": pred, "agree": mom[d]["r"] == pred}
        print("d=%2d L=%2d  r=%s  parity covering number=%d  agree=%s"
              % (d, L, mom[d]["r"], pred, par[d]["agree"]))
    out["r_vs_parity"] = par
    Ftop_from_census = {}
    for m in range(2, 9):
        ds = [d for d in range(1, 17) if mom[d]["r"] is not None and mom[d]["r"] <= m]
        Ftop_from_census[m] = max(ds) - 1
    out["F_from_census"] = Ftop_from_census
    print("F_top(m) = max{d : r(d) <= m} - 1 :", Ftop_from_census,
          " parity law 2m-(m mod 2):", {m: 2 * m - (m % 2) for m in range(2, 9)})

    # degree test: same m, different gear sets -> N_d equal iff r >= m
    # (hypothesis: every gear > d + 2, else the gear's classes collapse)
    deg = []
    for m in (3, 4, 5):
        for d in range(1, 15):
            sets = [
                [p for p in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43] if p > d + 2][:m],
                [p for p in [53, 59, 61, 67, 71, 73, 79, 83] if p > d + 2][:m],
                [p for p in [101, 103, 107, 109, 113, 127, 131, 137] if p > d + 2][:m],
            ]
            vals = [census_formula(s, d) for s in sets]
            r = mom[d]["r"]
            const = len(set(vals)) == 1
            pred = (r is not None and r >= m) or d == 4
            deg.append({"m": m, "d": d, "vals": vals, "constant": const, "r": r,
                        "predicted_constant": pred, "sets": sets})
            if const != pred:
                print("  DEGREE MISMATCH m=%d d=%d  %s  r=%s" % (m, d, vals, r))
        print(
            "m=%d constants at d =" % m,
            [(x["d"], x["vals"][0]) for x in deg if x["m"] == m and x["constant"]],
        )
    out["degree_test"] = deg
    out["degree_mismatches"] = sum(
        1 for x in deg if x["constant"] != x["predicted_constant"]
    )
    print("degree-test mismatches:", out["degree_mismatches"])

    # L9 from the census law: prod(g - e) is odd iff e is even (gears odd), so
    # N_d = sum_e c_e prod(g - e) is odd iff sum over EVEN e of c_e is odd.
    par9 = []
    for d in range(1, 17):
        s = sigs[d]
        even_sum = sum(v for k, v in s.items() if k % 2 == 0)
        par9.append({"d": d, "sum_c_e_over_even_e": even_sum, "predicted_odd": even_sum % 2 == 1})
        print("L9 d=%2d  sum of c_e over even e = %4d  -> N_d odd: %s"
              % (d, even_sum, even_sum % 2 == 1))
    out["L9_parity"] = par9
    odd_ds = [x["d"] for x in par9 if x["predicted_odd"]]
    print("gap lengths with an odd count, predicted from the census law:", odd_ds)
    out["odd_count_gap_lengths"] = odd_ds

    # the census's own record (closed boundaries) against the cover's (free boundaries)
    sys.path.insert(0, __file__.rsplit("r2", 1)[0] + "r1")
    from cover import F_cover  # noqa: E402

    cmp = []
    for gears in WHEELS:
        h = measure(gears)
        dmax = max(h)
        f, st = F_cover(gears)
        cmp.append({"gears": gears, "census_F": dmax - 1, "cover_F": f, "agree": dmax - 1 == f})
        print("record: census max gap - 1 = %2d   cover F = %2d   agree %s  %s"
              % (dmax - 1, f, dmax - 1 == f, gears))
    out["record_agreement"] = cmp
    out["record_disagreements"] = sum(1 for x in cmp if not x["agree"])

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
