"""The five definitions of 'retains the simplicity' and their exact thresholds in q' vs m.

(a) the mex closed form is exact          (b) the parity law holds
(c) the record is a free-domino tiling    (d) the symmetry group is exactly (Z/2)^m
(e) the record depends on m and the gears <= F + 1 only
"""

import json
import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from common import (  # noqa: E402
    feasible,
    gear_patterns,
    minpieces,
    open_mask,
    teeth,
    walk_lengths,
)

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
          83, 89, 97, 101, 103, 107, 109, 113]


def record(gears, cap=80):
    L = 0
    while L < cap:
        if not feasible(gears, L + 1):
            return L
        L += 1
    return L


def parity_pieces(L):
    return -(-(-(-L // 2)) // 2) if False else ((L + 1) // 2 + 1) // 2 + ((L // 2) + 1) // 2


def cover_witness(gears, L):
    """Return a cover of [0, L) as a list of (gear-or-None, bitmask); None = pool gear."""
    full = (1 << L) - 1
    small = [g for g in gears if g <= L + 1]
    c0 = len(gears) - len(small)

    bypos = {}
    for g in small:
        d = {}
        for msk in gear_patterns(g, L):
            for x in range(L):
                if (msk >> x) & 1:
                    d.setdefault(x, []).append(msk)
        bypos[g] = d

    def dfs(covered, unused, c, acc):
        if covered == full:
            return acc
        R = full & ~covered
        if not unused:
            if minpieces(R, L) <= c:
                return acc + [("pool", R)]
            return None
        x = (R & -R).bit_length() - 1
        if c > 0:
            for piece in ([(1 << x) | (1 << (x + 2))] if x + 2 < L else []) + [1 << x]:
                r = dfs(covered | piece, unused, c - 1, acc + [("pool", piece)])
                if r is not None:
                    return r
        for i, g in enumerate(unused):
            for msk in bypos[g].get(x, ()):
                r = dfs(covered | msk, unused[:i] + unused[i + 1:], c, acc + [(g, msk)])
                if r is not None:
                    return r
        return None

    return dfs(0, tuple(small), c0, [])


def mex_exact_test(gears):
    W = prod(gears)
    if W > 3_000_000:
        return None
    mask = open_mask(gears)
    L = walk_lengths(mask)
    tt = {g: teeth(g) for g in gears}
    bad = 0
    for x in range(W):
        s = {(t - x) % g for g in gears for t in tt[g]}
        j = 0
        while j in s:
            j += 1
        if j != L[x]:
            bad += 1
    return bad, int(L.max()), W


def main():
    out = {"a_mex": [], "b_parity": [], "c_tiling": [], "e_subthreshold": []}

    # ---------------- (a) the mex closed form, and the sharp criterion F < q'
    sets_a = [
        [5, 7], [5, 7, 11], [5, 7, 11, 13], [5, 11, 13, 17],
        [7, 11, 13], [7, 11, 13, 17], [7, 11, 13, 17, 19],
        [9, 11, 13], [9, 11, 13, 17], [9, 11, 13, 17, 19],
        [11, 13, 17], [11, 13, 17, 19], [11, 13, 17, 19, 23],
        [13, 17, 19], [13, 17, 19, 23], [13, 17, 19, 23, 29],
        [15, 17, 19], [15, 17, 19, 23], [15, 17, 19, 23, 29],
        [17, 19, 23], [17, 19, 23, 29], [17, 19, 23, 29, 31],
        [21, 23, 29, 31], [25, 29, 31, 37], [3, 5, 7], [2, 3, 5], [2, 5, 7],
    ]
    for gears in sets_a:
        r = mex_exact_test(gears)
        if r is None:
            continue
        bad, F, W = r
        m = len(gears)
        qp = min(gears)
        rec = {"gears": gears, "m": m, "q'": qp, "W": W, "F": F, "mex_bad": bad,
               "q'>2m": qp > 2 * m, "F<q'": F < qp}
        out["a_mex"].append(rec)
        print("a", rec, flush=True)

    # ---------------- (b) the parity law boundary, using odd composite coprime gears
    sets_b = []
    for m in range(2, 9):
        thr = 2 * m + 1
        # q' just below, at, and above the threshold
        for qp in (thr - 4, thr - 2, thr, thr + 2, thr + 4):
            if qp < 3 or qp % 2 == 0:
                continue
            rest = [p for p in PRIMES if p > qp + 1][: m - 1]
            gears = [qp] + rest
            # pairwise coprimality
            ok = all(
                np.gcd(gears[i], gears[j]) == 1
                for i in range(len(gears)) for j in range(i + 1, len(gears))
            )
            if ok and len(gears) == m:
                sets_b.append(gears)
    for gears in sets_b:
        m = len(gears)
        F = record(gears)
        rec = {"gears": gears, "m": m, "q'": min(gears), "F": F,
               "parity_value": 2 * m - (m % 2), "holds": F == 2 * m - (m % 2),
               "q'>2m+1": min(gears) > 2 * m + 1}
        out["b_parity"].append(rec)
        print("b", rec, flush=True)

    # ---------------- (c) the record cover: are all pieces free dominoes?
    for gears in sets_b + [[5, 7], [5, 7, 11], [7, 11, 13], [7, 11, 13, 17], [3, 5, 7]]:
        m = len(gears)
        F = record(gears)
        w = cover_witness(gears, F)
        pieces = 0
        alldom = True
        if w:
            for who, msk in w:
                cells = [x for x in range(F) if (msk >> x) & 1]
                if who == "pool":
                    pieces += minpieces(msk, F)
                else:
                    pieces += 1
                    if not (len(cells) <= 1 or (len(cells) == 2 and cells[1] - cells[0] == 2)):
                        alldom = False
        rec = {"gears": gears, "m": m, "q'": min(gears), "F": F,
               "pieces": pieces, "parity_pieces": parity_pieces(F),
               "all_free_dominoes": alldom, "q'>F+1": min(gears) > F + 1}
        out["c_tiling"].append(rec)
        print("c", rec, flush=True)

    # ---------------- (e) the sub-threshold reduction with small gears present
    pools = [[29, 31, 37], [41, 43, 47], [53, 59, 61], [67, 71, 73]]
    fams = {}
    for small in ([], [2], [3], [5], [2, 3], [3, 5], [5, 7], [2, 3, 5], [7], [7, 11]):
        for pool in pools:
            for t in range(1, 4):
                gears = sorted(small + pool[:t])
                if len(set(gears)) != len(gears):
                    continue
                ok = all(
                    np.gcd(gears[i], gears[j]) == 1
                    for i in range(len(gears)) for j in range(i + 1, len(gears))
                )
                if not ok:
                    continue
                F = record(gears)
                S = tuple(sorted(g for g in gears if g <= F + 1))
                fams.setdefault((len(gears), S), []).append((tuple(gears), F))
    bad = 0
    comparable = 0
    for key, vals in sorted(fams.items()):
        if len(vals) < 2:
            continue
        comparable += len(vals)
        Fs = {f for _, f in vals}
        if len(Fs) > 1:
            bad += 1
        out["e_subthreshold"].append({"m": key[0], "S": list(key[1]),
                                      "sets": [list(g) for g, _ in vals],
                                      "F": sorted(Fs)})
        print("e", key, sorted(Fs), len(vals), flush=True)
    out["e_summary"] = {"families": len(out["e_subthreshold"]), "comparable": comparable,
                        "families_with_disagreement": bad}
    print("e summary", out["e_summary"])

    with open(__file__.rsplit("s4_")[0] + "results/s4_thresholds.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
