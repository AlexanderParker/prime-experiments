"""Top machine on the raw line: the wheel and its slots.

Pair n = (n, n+2).  Gear g strikes n iff n = 0 or n = -2 (mod g).
Wheel W = prod of the gear set; open pairs are W-periodic.

Sections: counts, arcs, runs, record, gap spectrum, mirror, symmetry group,
origin clump, dominoes, conjugacy to the bottom's column coordinate.
"""

import json
import sys
from math import prod

import numpy as np

PRIMES = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]


def open_mask(gears, W):
    a = np.ones(W, dtype=bool)
    for g in gears:
        a[0::g] = False
        a[(g - 2) % g :: g] = False
    return a


def col_mask(gears, W):
    """Same gears written in the bottom's column coordinate: teeth +-6^{-1} mod g."""
    a = np.ones(W, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        a[u % g :: g] = False
        a[(-u) % g :: g] = False
    return a


def runs_of(mask):
    """Cyclic run lengths of True in mask.  Returns (dict length->count, longest, start of a longest)."""
    W = len(mask)
    if mask.all():
        return {W: 1}, W, 0
    if not mask.any():
        return {}, 0, -1
    idx = np.flatnonzero(mask)
    # break points where consecutive True indices are not adjacent (cyclically)
    d = np.diff(idx)
    breaks = np.flatnonzero(d != 1)
    # run start positions in idx-space
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    lens = ends - starts + 1
    startpos = idx[starts]
    # wrap: if first and last element are both True, merge first and last run
    if mask[0] and mask[-1] and len(lens) > 1:
        lens[0] += lens[-1]
        startpos[0] = idx[starts[-1]] - lens[-1] + lens[0] - lens[0] + startpos[-1] - W
        lens = lens[:-1]
        startpos = startpos[:-1]
    hist = {}
    for L, c in zip(*np.unique(lens, return_counts=True)):
        hist[int(L)] = int(c)
    j = int(np.argmax(lens))
    return hist, int(lens[j]), int(startpos[j])


def affine_symmetries(mask, W, limit=6000):
    """Brute force affine maps n -> c n + b preserving the open set (small W only)."""
    if W > limit:
        return None
    O = mask
    n = np.arange(W)
    found = []
    for c in range(1, W):
        from math import gcd

        if gcd(c, W) != 1:
            continue
        img_base = (c * n) % W
        for b in range(W):
            # cheap necessary test on a few points first
            if not O[(img_base[np.flatnonzero(O)[:8]] + b) % W].all():
                continue
            if np.array_equal(O[(img_base + b) % W], O):
                found.append((c, b))
    return found


def analyse(gears):
    W = prod(gears)
    m = open_mask(gears, W)
    out = {"gears": gears, "W": W}
    cnt = int(m.sum())
    out["open_pairs"] = cnt
    out["prod_g_minus_2"] = prod(g - 2 for g in gears)
    out["count_matches"] = cnt == out["prod_g_minus_2"]

    # arcs of each gear alone
    arcs = {}
    for g in gears:
        gm = np.ones(g, dtype=bool)
        gm[0] = False
        gm[(g - 2) % g] = False
        h, longest, _ = runs_of(gm)
        arcs[g] = {"arc_lengths": sorted(h.keys(), reverse=True), "hist": h}
    out["single_gear_arcs"] = arcs

    # runs of open pairs
    hist_open, longest_open, start_open = runs_of(m)
    out["open_run_hist"] = hist_open
    out["longest_open_run"] = longest_open
    out["longest_open_run_start"] = start_open
    out["q_prime_minus_3"] = gears[0] - 3

    # record: longest run of consecutive struck n
    hist_closed, F, startF = runs_of(~m)
    out["record_F_top"] = F
    out["record_start"] = startF
    out["record_multiplicity"] = hist_closed.get(F, 0)
    out["closed_run_hist_tail"] = {k: v for k, v in sorted(hist_closed.items())[-12:]}

    # gap spectrum (gap between consecutive open pairs = closed run + 1)
    idx = np.flatnonzero(m)
    gaps = np.diff(idx)
    wrap = idx[0] + W - idx[-1]
    gaps = np.concatenate((gaps, [wrap]))
    gh = {}
    for L, c in zip(*np.unique(gaps, return_counts=True)):
        gh[int(L)] = int(c)
    out["gap_hist"] = gh
    out["max_gap"] = int(gaps.max())
    out["gap_counts_even"] = {k: (v % 2 == 0) for k, v in gh.items()}
    out["odd_gap_lengths"] = [k for k, v in gh.items() if v % 2 == 1]

    # mirror n -> -n-2
    mir = m[(-np.arange(W) - 2) % W]
    out["mirror_mismatches"] = int((mir != m).sum())
    fixed = np.flatnonzero(((-np.arange(W) - 2) % W) == np.arange(W))
    out["mirror_fixed_points"] = [int(x) for x in fixed]
    out["mirror_fixed_open"] = [bool(m[x]) for x in fixed]

    # forced-open residues near the origin
    q0 = gears[0]
    lo, hi = -(2 * q0), 2 * q0
    clump = []
    for n in range(lo, hi + 1):
        clump.append((n, bool(m[n % W])))
    out["origin_window"] = clump
    out["origin_predicted_run"] = q0 - 3

    # antipode analogues
    out["state_at_minus1"] = bool(m[(-1) % W])
    out["state_at_2"] = bool(m[2 % W])
    out["state_at_minus4"] = bool(m[(-4) % W])

    # dominoes: n and n+1 both open
    dom = int((m & np.roll(m, -1)).sum())
    out["dominoes"] = dom
    out["prod_g_minus_4"] = prod(g - 4 for g in gears)
    out["domino_matches"] = dom == out["prod_g_minus_4"]

    # conjugacy to the bottom's column coordinate
    inv6 = pow(6, -1, W)
    n = np.arange(W, dtype=np.int64)
    k = (inv6 * (n + 1)) % W
    cb = col_mask(gears, W)
    out["conjugacy_mismatches"] = int((cb[k] != m).sum())

    # distribution of open pairs mod 6, 2, 3
    for mod in (2, 3, 6):
        d = {}
        r = n % mod
        for j in range(mod):
            d[j] = int(m[r == j].sum())
        out[f"open_mod_{mod}"] = d

    # symmetry group (brute force where feasible)
    sym = affine_symmetries(m, W)
    if sym is not None:
        out["affine_symmetry_count"] = len(sym)
        out["affine_symmetries"] = sym
        adj = [(c, b) for (c, b) in sym if c in (1, W - 1)]
        out["adjacency_preserving"] = adj
    out["predicted_affine_group_size"] = 2 ** len(gears)

    return out


def main():
    sets = [
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
    ]
    res = [analyse(g) for g in sets]
    with open(sys.argv[1], "w") as f:
        json.dump(res, f, indent=1)
    for r in res:
        print(
            r["gears"],
            "W=%d" % r["W"],
            "open=%d(%s)" % (r["open_pairs"], r["count_matches"]),
            "maxopenrun=%d(pred %d)" % (r["longest_open_run"], r["q_prime_minus_3"]),
            "F=%d@%d x%d" % (r["record_F_top"], r["record_start"], r["record_multiplicity"]),
            "mirror_mis=%d" % r["mirror_mismatches"],
            "conj_mis=%d" % r["conjugacy_mismatches"],
            "dom=%d(%s)" % (r["dominoes"], r["domino_matches"]),
            "sym=%s" % r.get("affine_symmetry_count", "-"),
        )


if __name__ == "__main__":
    main()
