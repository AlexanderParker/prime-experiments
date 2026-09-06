"""The law table, part 1: the structural laws of documents 1 and 2, on gear sets whose
smallest gear is 2, 3, 5, 7, 11, 13.  Exact over full wheel periods."""

import json
import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from common import (  # noqa: E402
    all_struck_counts,
    chain_starts,
    gap_census,
    longest_chain,
    longest_run,
    open_mask,
    record_scan,
    run_starts,
    slots,
    teeth,
)

SETS = [
    [2, 3, 5], [2, 5, 7], [2, 3, 5, 7], [2, 5, 7, 11], [2, 7, 11, 13],
    [2, 3, 5, 7, 11], [2, 3, 5, 7, 11, 13],
    [3, 5, 7], [3, 5, 7, 11], [3, 7, 11, 13], [3, 5, 7, 11, 13], [3, 11, 13, 17],
    [5, 7, 11], [5, 7, 11, 13], [5, 11, 13, 17], [5, 7, 11, 13, 17],
    [7, 11, 13], [7, 11, 13, 17], [7, 13, 19, 23],
    [11, 13, 17], [11, 13, 17, 19],
    [13, 17, 19], [13, 17, 19, 23],
]


def arcs(g):
    """Lengths of the maximal arcs of open residues of gear g, in cyclic order."""
    s = set(slots(g))
    if not s:
        return []
    out = []
    r = 0
    # rotate to a struck residue so arcs are not split by the wrap
    start = next(x for x in range(g) if x not in s)
    seq = [(start + i) % g for i in range(g)]
    run = 0
    for x in seq:
        if x in s:
            run += 1
        else:
            if run:
                out.append(run)
            run = 0
    if run:
        out.append(run)
    return sorted(out, reverse=True)


def partner_law(g, W=None):
    """Every strike of gear g has a same-gear partner at distance exactly 2."""
    bad = 0
    tt = set(teeth(g))
    for r in range(g):
        if r in tt:
            if not ((r - 2) % g in tt or (r + 2) % g in tt):
                bad += 1
    return bad


def mirror_test(mask):
    W = len(mask)
    idx = np.arange(W)
    img = (-idx - 2) % W
    mism = int((mask != mask[img]).sum())
    fixed = [int(n) for n in range(W) if (-n - 2) % W == n]
    return mism, fixed


def affine_group(mask):
    """Brute force all affine maps n -> c n + b of Z_W preserving the open set."""
    W = len(mask)
    idx = np.arange(W)
    good = []
    for c in range(W):
        img_c = (c * idx) % W
        for b in range(W):
            img = (img_c + b) % W
            if np.array_equal(mask[img], mask):
                good.append((c, b))
    return good


def clump_width(gears):
    """#{n in [-(q'-1), q'-3] : n open}, with n taken mod W."""
    qp = min(gears)
    W = prod(gears)
    mask = open_mask(gears)
    lo, hi = -(qp - 1), qp - 3
    if hi < lo:
        return 0, 2 * (qp - 3) + 1
    cnt = sum(1 for n in range(lo, hi + 1) if mask[n % W])
    return cnt, 2 * (qp - 3) + 1


def census_law(gears, d):
    """L22 (document 2): N_d by inclusion-exclusion over the uncovered interior."""
    from itertools import combinations

    tot = 0
    interior = list(range(1, d))
    for k in range(len(interior) + 1):
        for S in combinations(interior, k):
            p = 1
            for g in gears:
                E = {0 % g, (-2) % g, (-d) % g, (-d - 2) % g}
                for j in S:
                    E.add((-j) % g)
                    E.add((-(j + 2)) % g)
                p *= g - len(E)
            tot += (-1) ** k * p
    return tot


def palindrome_from_shield(mask):
    """The cyclic gap word read from the shield n = -1 is a palindrome."""
    W = len(mask)
    openpos = list(np.flatnonzero(mask))
    shield = (W - 1) % W
    if shield not in openpos:
        return None
    i = openpos.index(shield)
    seq = openpos[i:] + [p + W for p in openpos[:i]]
    word = [seq[j + 1] - seq[j] for j in range(len(seq) - 1)]
    word.append(openpos[i] + W - seq[-1])
    return word == word[::-1]


def main():
    out = []
    for gears in SETS:
        W = prod(gears)
        qp = min(gears)
        m = len(gears)
        mask = open_mask(gears)
        nopen = int(mask.sum())
        rec = {"gears": gears, "W": W, "q'": qp, "m": m}

        # L1 slot count
        rec["L1_slots"] = {g: len(slots(g)) for g in gears}
        rec["L1_holds"] = all(len(slots(g)) == g - 2 for g in gears)
        # L2 arcs
        rec["L2_arcs"] = {g: arcs(g) for g in gears}
        rec["L2_holds"] = all(arcs(g) == sorted([g - 3, 1], reverse=True) for g in gears)
        # L3 partner
        rec["L3_bad"] = sum(partner_law(g) for g in gears)
        # L4 forbidden gap 4
        cen = gap_census(mask)
        rec["L4_gap4"] = cen.get(4, 0)
        # L5 wheel count
        rec["L5_open"] = nopen
        rec["L5_prod"] = prod(g - 2 for g in gears)
        rec["L5_holds"] = nopen == prod(g - 2 for g in gears)
        # L6 shield / antipode / clump
        rec["L6_shield"] = bool(mask[(-1) % W])
        rec["L6_two"] = bool(mask[2 % W])
        rec["L6_negfour"] = bool(mask[(-4) % W])
        cw, cwpred = clump_width(gears)
        rec["L6_clump"] = [cw, cwpred]
        # L7 mirror
        mism, fixed = mirror_test(mask)
        rec["L7_mismatch"] = mism
        rec["L7_fixed"] = fixed
        # L8 group (brute force where cheap)
        if W <= 400:
            grp = affine_group(mask)
            rec["L8_order"] = len(grp)
            rec["L8_pred"] = 2 ** m
            rec["L8_form"] = all((b - (c - 1)) % W == 0 for c, b in grp)
            rec["L8_adj"] = sum(1 for c, b in grp if c % W in (1, W - 1))
        # L9 gap parity
        rec["L9_odd_lengths"] = sorted(d for d, n in cen.items() if n % 2 == 1)
        # L10 ceilings
        lr, lc = longest_run(mask), longest_chain(mask)
        rec["L10_run"] = [lr, qp - 3]
        rec["L10_chain"] = [lc, qp - 2]
        # L11 run spectrum counts
        A = {L: run_starts(mask, L) for L in range(1, max(2, lr + 2))}
        rec["L11_A"] = {L: [A[L], prod(g - 2 - L for g in gears) if L >= 2 else prod(g - 2 for g in gears)] for L in A}
        rec["L11_holds"] = all(
            A[L] == (prod(g - 2 - L for g in gears) if L >= 2 else prod(g - 2 for g in gears))
            for L in A
        )
        # L15 dominoes and member-sharing
        rec["L15_N1"] = [A.get(2, 0), prod(g - 4 for g in gears)]
        rec["L15_chain2"] = [chain_starts(mask, 2), prod(g - 3 for g in gears)]
        # L16/L17/L18 record
        F = record_scan(mask)
        C = all_struck_counts(mask, F + 1)
        rec["F_scan"] = F
        rec["L17_parity"] = 2 * m - (m % 2)
        rec["L18_mult"] = C[F] if F < len(C) else None
        # L20 fold
        for mod in (2, 3, 6):
            classes = np.bincount(np.flatnonzero(mask) % mod, minlength=mod)
            rec[f"L20_mod{mod}"] = classes.tolist()
        # L22 census law vs measured
        dmax = min(max(cen), 12)
        pred = {d: census_law(gears, d) for d in range(1, dmax + 1)}
        rec["L22_mismatch"] = sum(1 for d in range(1, dmax + 1) if pred[d] != cen.get(d, 0))
        rec["L22_dmax"] = dmax
        rec["census"] = {d: cen.get(d, 0) for d in range(1, dmax + 1)}
        # L24 N3 vs N5
        rec["L24"] = [cen.get(3, 0), cen.get(5, 0)]
        # L27 parity of N_d
        rec["L27_odd_d"] = sorted(d for d, n in cen.items() if n % 2 == 1)
        # L28 all-struck classes and total collisions
        allstruck = np.ones(W, dtype=bool)
        for g in gears:
            r = np.arange(W) % g
            hit = np.zeros(W, dtype=bool)
            for t in teeth(g):
                hit |= r == t
            allstruck &= hit
        rec["L28_allstruck"] = int(allstruck.sum())
        rec["L28_pred"] = prod(len(teeth(g)) for g in gears)
        # total collisions: same tooth for every gear
        tot = 0
        for n in np.flatnonzero(allstruck):
            same0 = all(n % g == 0 for g in gears)
            same2 = all(n % g == (-2) % g for g in gears)
            if same0 or same2:
                tot += 1
        rec["L28_total_collisions"] = tot
        # L35 palindrome
        rec["L35_palindrome"] = palindrome_from_shield(mask)
        out.append(rec)
        print(json.dumps(rec))
    with open(__file__.rsplit("s1_")[0] + "results/s1_structure.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
