"""The law table, part 3: the pairwise laws (chain, merge, letters), the hop collapse,
the removal law, the corridor, and the odd gap length - all with small gears present."""

import json
import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from common import (  # noqa: E402
    gap_census,
    gaps_of,
    longest_chain,
    longest_run,
    open_mask,
    record_scan,
    teeth,
    walk_lengths,
)

LADDERS = [
    [2, 3, 5, 7],
    [2, 5, 7, 11],
    [3, 5, 7, 11],
    [5, 7, 11, 13],
    [7, 11, 13, 17],
    [11, 13, 17, 19],
]


def chain_law(M, g):
    """Two openings x < y of M are both struck by g in some copy iff y - x = 0, +-2 mod g."""
    W = prod(M)
    mask = open_mask(M)
    op = np.flatnonzero(mask)
    tt = teeth(g)
    bad = tested = 0
    # both struck by g in SOME copy of M's period: exists s with (x+s) and (y+s) both on teeth
    for i in range(len(op)):
        for j in range(i + 1, len(op)):
            x, y = int(op[i]), int(op[j])
            d = (y - x) % g
            both = any((d + t1 - t2) % g == 0 for t1 in tt for t2 in tt)
            pred = d in {t1 - t2 for t1 in tt for t2 in tt} or any(
                (y - x) % g == (t1 - t2) % g for t1 in tt for t2 in tt
            )
            tested += 1
            if both != pred:
                bad += 1
    return tested, bad


def chain_law_fast(M, g):
    """Same as chain_law but stated as: both struck in some copy iff (y-x) mod g is a
    difference of two teeth.  Checked by direct search over the phase."""
    W = prod(M)
    mask = open_mask(M)
    op = np.flatnonzero(mask)
    tt = teeth(g)
    diffs = {(t1 - t2) % g for t1 in tt for t2 in tt}
    bad = 0
    n = len(op)
    step = max(1, n // 400)
    tested = 0
    for i in range(0, n, step):
        for j in range(i + 1, n, step):
            x, y = int(op[i]), int(op[j])
            d = (y - x) % g
            realis = any(((x + s) % g in tt) and ((y + s) % g in tt) for s in range(g))
            tested += 1
            if realis != (d in diffs):
                bad += 1
    return tested, bad


def merge_law(M, g):
    """Every gap of M + g is a gap of M or a merge of consecutive gaps of M whose interior
    openings g strikes."""
    Mm = open_mask(M)
    W = prod(M)
    big = list(M) + [g]
    Wb = prod(big)
    reps = Wb // W
    maskM = np.tile(Mm, reps)
    maskB = open_mask(big)
    opM = np.flatnonzero(maskM)
    opB = np.flatnonzero(maskB)
    tt = set(teeth(g))
    posM = set(int(v) for v in opM)
    bad = 0
    for i in range(len(opB)):
        a = int(opB[i])
        b = int(opB[(i + 1) % len(opB)])
        if b <= a:
            b += Wb
        # interior openings of M inside (a, b) must all be struck by g
        k = np.searchsorted(opM, a, side="right")
        while k < len(opM) and opM[k] < b:
            if (int(opM[k]) % g) not in tt:
                bad += 1
                break
            k += 1
        if a not in posM or (b % Wb) not in posM:
            bad += 1
    return len(opB), bad


def alternation(M, g):
    """In a run of consecutive M-openings all struck by g, the nonzero letter classes
    strictly alternate."""
    Mm = open_mask(M)
    big = list(M) + [g]
    Wb = prod(big)
    reps = Wb // prod(M)
    maskM = np.tile(Mm, reps)
    opM = np.flatnonzero(maskM)
    tt = teeth(g)
    runs = 0
    bad = 0
    cur = []
    for v in list(opM) + [None]:
        if v is not None and (int(v) % g) in tt:
            cur.append(int(v))
        else:
            if len(cur) >= 2:
                runs += 1
                letters = [(cur[i + 1] - cur[i]) % g for i in range(len(cur) - 1)]
                nz = [x for x in letters if x != 0]
                if any(nz[i] == nz[i + 1] for i in range(len(nz) - 1)):
                    bad += 1
                if any(x not in {(t2 - t1) % g for t1 in tt for t2 in tt if (t2 - t1) % g != 0}
                       for x in nz):
                    bad += 1
            cur = []
    return runs, bad


def hop_chain(M, g):
    """Longest hop chain when gear g is added to M, and the double-hop rule."""
    Mm = open_mask(M)
    big = list(M) + [g]
    Wb = prod(big)
    reps = Wb // prod(M)
    maskM = np.tile(Mm, reps)
    maskB = open_mask(big)
    LM = walk_lengths(maskM)
    LB = walk_lengths(maskB)
    tt = set(teeth(g))
    longest = 0
    doubles = 0
    rule_bad = 0
    for x in range(Wb):
        y = (x + int(LM[x])) % Wb
        hops = 0
        yy = y
        while (yy % g) in tt:
            hops += 1
            nxt = (yy + 1) % Wb
            yy = (nxt + int(LM[nxt])) % Wb
            if hops > 30:
                break
        longest = max(longest, hops)
        if hops >= 2:
            doubles += 1
        # the document-3 rule: a double hop occurs iff y = -2 mod g and the M-gap at y is 2
        if hops >= 1:
            nxt = (y + 1) % Wb
            d1 = (int(LM[nxt]) + 1)
            pred_double = ((y % g) == (-2) % g) and d1 == 2
            if (hops >= 2) != pred_double:
                rule_bad += 1
        if (x + int(LB[x])) % Wb != yy % Wb:
            rule_bad += 1000000
    return longest, doubles, rule_bad


def main():
    out = {"chain": [], "merge": [], "alternation": [], "hop": [], "removal": [],
           "corridor": [], "oddgap": []}

    for lad in LADDERS:
        for i in range(1, len(lad)):
            M, g = lad[:i], lad[i]
            t, b = chain_law_fast(M, g)
            out["chain"].append({"M": M, "g": g, "tested": t, "bad": b})
            t2, b2 = merge_law(M, g)
            out["merge"].append({"M": M, "g": g, "gaps": t2, "bad": b2})
            r, ba = alternation(M, g)
            out["alternation"].append({"M": M, "g": g, "runs": r, "bad": ba})
            FM = record_scan(np.tile(open_mask(M), 1))
            lg, dbl, rb = hop_chain(M, g)
            out["hop"].append({"M": M, "g": g, "F_M": FM, "g>F+3": g > FM + 3,
                               "longest_chain": lg, "doubles": dbl, "rule_bad": rb})
            print("chain", M, g, t, b, "| merge", t2, b2, "| alt", r, ba,
                  "| hop F_M", FM, "chain", lg, "doubles", dbl, "rulebad", rb, flush=True)

    # removal law
    for gears in [[2, 3, 5, 7, 11], [3, 5, 7, 11, 13], [5, 7, 11, 13, 17], [7, 11, 13, 17, 19],
                  [11, 13, 17, 19, 23]]:
        prev = None
        for k in range(len(gears) - 1):
            G = gears[k:]
            mask = open_mask(G)
            rec = {
                "G": G, "W": prod(G), "open": int(mask.sum()),
                "N1": int((mask & np.roll(mask, -1)).sum()),
                "run": longest_run(mask), "chain": longest_chain(mask),
                "F": record_scan(mask),
            }
            if prev:
                qp = prev["G"][0]
                rec["W_div"] = prev["W"] // rec["W"] == qp and prev["W"] % rec["W"] == 0
                rec["open_div"] = (prev["open"] == rec["open"] * (qp - 2))
                rec["N1_div"] = (prev["N1"] == rec["N1"] * (qp - 4)) if qp > 4 else None
                rec["F_step"] = prev["F"] - rec["F"]
            out["removal"].append(rec)
            print("removal", rec, flush=True)
            prev = rec

    # corridor: the small wheel's own slots
    for small in [[2], [3], [2, 3], [2, 3, 5], [3, 5], [5, 7], [5, 7, 11], [7, 11]]:
        W = prod(small)
        mask = open_mask(small)
        out["corridor"].append({"small": small, "W": W, "slots": int(mask.sum()),
                                "density": int(mask.sum()) / W,
                                "prod_g_minus_2": prod(g - 2 for g in small)})
        print("corridor", out["corridor"][-1], flush=True)

    # the unique odd gap length, and its identification with the mirror-self-paired gap
    for gears in [[2, 3, 5], [2, 5, 7], [2, 3, 5, 7], [3, 5, 7], [3, 5, 7, 11], [3, 7, 11, 13],
                  [5, 7, 11], [5, 7, 11, 13], [7, 11, 13], [11, 13, 17], [13, 17, 19],
                  [2, 3, 5, 7, 11], [3, 11, 13, 17], [5, 11, 13, 17], [7, 11, 13, 17]]:
        W = prod(gears)
        mask = open_mask(gears)
        cen = gap_census(mask)
        odd = sorted(d for d, n in cen.items() if n % 2 == 1)
        # the self-paired gap: endpoints a, a + d with a + (a + d) = -2 mod W
        openpos = np.flatnonzero(mask)
        selfpaired = []
        for i in range(len(openpos)):
            a = int(openpos[i])
            b = int(openpos[(i + 1) % len(openpos)])
            d = (b - a) % W
            if (2 * a + d + 2) % W == 0:
                selfpaired.append(d)
        out["oddgap"].append({"gears": gears, "odd_lengths": odd,
                              "selfpaired_lengths": sorted(set(selfpaired)),
                              "n_selfpaired": len(selfpaired)})
        print("oddgap", out["oddgap"][-1], flush=True)

    with open(__file__.rsplit("s3_")[0] + "results/s3_pairwise.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
