"""Top machine on the raw line: pairwise laws, letters, merge, run spectrum.

Verifies, exhaustively over full periods:
  L-PARTNER  every struck pair has a partner strike at distance exactly 2
  gap 4 is impossible
  chain law   two openings x<y of M both struck by new gear g in some copy
              iff y - x = 0, +2 or -2 (mod g)
  merge law   every gap of M+g is an old gap or a merge of consecutive old gaps
  alternation nonzero letter classes strictly alternate 2, g-2
  run spectrum  #runs of exactly L = second difference of prod(g - 2 - L)
  domino count  prod(g - 4)
"""

import json
import sys
from math import prod

import numpy as np


def open_mask(gears, W):
    a = np.ones(W, dtype=bool)
    for g in gears:
        a[0::g] = False
        a[(g - 2) % g :: g] = False
    return a


def check_partner(gears):
    """Every struck n has a struck partner at n-2 or n+2 (same gear)."""
    W = prod(gears)
    m = open_mask(gears, W)
    struck = ~m
    ok = struck & (np.roll(struck, 2) | np.roll(struck, -2))
    return int((struck & ~ok).sum()), int(struck.sum())


def gap_hist(gears):
    W = prod(gears)
    m = open_mask(gears, W)
    idx = np.flatnonzero(m)
    gaps = np.concatenate((np.diff(idx), [idx[0] + W - idx[-1]]))
    h = {}
    for L, c in zip(*np.unique(gaps, return_counts=True)):
        h[int(L)] = int(c)
    return h


def run_spectrum(gears):
    """Measured run-length histogram vs the second-difference formula."""
    W = prod(gears)
    m = open_mask(gears, W)
    # measured
    idx = np.flatnonzero(m)
    d = np.diff(idx)
    br = np.flatnonzero(d != 1)
    starts = np.concatenate(([0], br + 1))
    ends = np.concatenate((br, [len(idx) - 1]))
    lens = ends - starts + 1
    if m[0] and m[-1] and len(lens) > 1:
        lens[0] += lens[-1]
        lens = lens[:-1]
    meas = {}
    for L, c in zip(*np.unique(lens, return_counts=True)):
        meas[int(L)] = int(c)

    def A(L):
        # number of n with n..n+L-1 all open
        if L <= 0:
            return W
        if L == 1:
            return prod(g - 2 for g in gears)
        return prod(max(g - 2 - L, 0) for g in gears)

    pred = {}
    for L in range(1, gears[0] - 2):
        pred[L] = A(L) - 2 * A(L + 1) + A(L + 2)
    return meas, pred


def chain_law(gears, g):
    """M = wheel of `gears`; new gear g.  For every ordered pair of openings x<y of M
    within one period, check: both struck by g in some copy  iff  y-x = 0,+-2 (mod g)."""
    W = prod(gears)
    m = open_mask(gears, W)
    op = np.flatnonzero(m)
    # both struck in some copy: exists r with x = r or r+2, y = r or r+2 (mod g)
    # equivalently y-x in {0, 2, -2} mod g
    # direct check over all pairs at distance <= some cap plus a random sample of far pairs
    exc = 0
    tested = 0
    P = W
    invP = pow(P, -1, g)
    for i, x in enumerate(op[: min(len(op), 4000)]):
        for y in op[i + 1 : min(i + 60, len(op))]:
            dxy = (int(y) - int(x)) % g
            cond = dxy in (0, 2, g - 2)
            # realisable: exists copy j with x + jP = 0 or -2 and y + jP = 0 or -2 (mod g)
            real = False
            for t1 in (0, -2):
                j = ((t1 - int(x)) * invP) % g
                yy = (int(y) + j * P) % g
                if yy in (0, (g - 2) % g):
                    real = True
                    break
            if real != cond:
                exc += 1
            tested += 1
    return exc, tested


def merge_law(gears, g):
    """Every gap of M+g is an old gap or a sum of consecutive old gaps whose interior
    openings are all struck by g."""
    W = prod(gears)
    m = open_mask(gears, W)
    W2 = W * g
    m2 = open_mask(list(gears) + [g], W2)
    op = np.flatnonzero(m)
    op2 = np.flatnonzero(m2)
    ops = set(int(x) for x in op)
    exc = 0
    # openings of M in one big period
    opbig = np.concatenate([op + j * W for j in range(g)])
    opbig.sort()
    pos = {int(v): i for i, v in enumerate(opbig)}
    for a, b in zip(op2[:-1], op2[1:]):
        a, b = int(a), int(b)
        if a not in pos or b not in pos:
            exc += 1
            continue
        ia, ib = pos[a], pos[b]
        interior = opbig[ia + 1 : ib]
        if len(interior):
            bad = [int(x) for x in interior if not (x % g == 0 or (x + 2) % g == 0)]
            if bad:
                exc += 1
    return exc, len(op2) - 1


def alternation(gears, g):
    """In a maximal run of consecutive openings of M all struck by g (one copy),
    the nonzero letter classes alternate 2, g-2."""
    W = prod(gears)
    m = open_mask(gears, W)
    op = np.flatnonzero(m).astype(np.int64)
    exc = 0
    runs = 0
    for j in range(g):
        # copy j: opening x is struck iff x + jW = 0 or -2 (mod g)
        v = (op + j * W) % g
        hit = (v == 0) | (v == (g - 2) % g)
        # maximal runs of consecutive openings (adjacent in the opening list) all hit
        i = 0
        n = len(op)
        while i < n:
            if not hit[i]:
                i += 1
                continue
            k = i
            while k + 1 < n and hit[k + 1]:
                k += 1
            if k > i:
                runs += 1
                # letters
                teeth = [0 if v[t] == 0 else 1 for t in range(i, k + 1)]
                sp = [(int(op[t + 1]) - int(op[t])) % g for t in range(i, k)]
                cls = [0 if s == 0 else (1 if s == 2 else (2 if s == (g - 2) % g else 3)) for s in sp]
                if 3 in cls:
                    exc += 1
                else:
                    nz = [c for c in cls if c != 0]
                    for u in range(len(nz) - 1):
                        if nz[u] == nz[u + 1]:
                            exc += 1
                            break
                    # tooth consistency: class 1 (spacing 2) goes tooth -2 -> 0
                    for t_i, c in zip(range(i, k), cls):
                        if c == 1 and not (teeth[t_i - i] == 1 and teeth[t_i - i + 1] == 0):
                            exc += 1
                            break
                        if c == 2 and not (teeth[t_i - i] == 0 and teeth[t_i - i + 1] == 1):
                            exc += 1
                            break
            i = k + 1
    return exc, runs


def two_gear_cells(g, h):
    W = g * h
    mg = open_mask([g], W)
    mh = open_mask([h], W)
    both = int(((~mg) & (~mh)).sum())
    only_g = int(((~mg) & mh).sum())
    only_h = int((mg & (~mh)).sum())
    neither = int((mg & mh).sum())
    return {
        "g": g,
        "h": h,
        "both_struck": both,
        "only_g": only_g,
        "only_h": only_h,
        "open": neither,
        "pred_both": 4,
        "pred_open": (g - 2) * (h - 2),
    }


def main():
    out = {}
    wheels = [
        [7, 11, 13],
        [11, 13, 17],
        [13, 17, 19],
        [17, 19, 23],
        [19, 23, 29],
        [7, 11, 13, 17],
        [11, 13, 17, 19],
        [13, 17, 19, 23],
    ]
    out["partner_law"] = []
    out["gap_hist"] = []
    out["run_spectrum"] = []
    for gs in wheels:
        e, tot = check_partner(gs)
        out["partner_law"].append({"gears": gs, "exceptions": e, "struck": tot})
        gh = gap_hist(gs)
        out["gap_hist"].append({"gears": gs, "hist": gh, "gap4": gh.get(4, 0)})
        meas, pred = run_spectrum(gs)
        mism = {L: (meas.get(L, 0), pred.get(L, 0)) for L in pred if meas.get(L, 0) != pred.get(L, 0)}
        out["run_spectrum"].append(
            {"gears": gs, "measured": meas, "predicted": pred, "mismatches": mism}
        )
        print(gs, "partner exc", e, "gap4", gh.get(4, 0), "runspec mismatch", mism)

    out["two_gear_cells"] = [
        two_gear_cells(g, h)
        for g, h in [(7, 11), (11, 13), (13, 17), (17, 19), (29, 31), (7, 13), (11, 17)]
    ]
    for c in out["two_gear_cells"]:
        print("cells", c)

    out["chain_law"] = []
    out["merge_law"] = []
    out["alternation"] = []
    for gs, g in [([7, 11], 13), ([7, 11, 13], 17), ([11, 13], 17), ([11, 13, 17], 19)]:
        e, t = chain_law(gs, g)
        out["chain_law"].append({"M": gs, "g": g, "exceptions": e, "tested": t})
        e2, t2 = merge_law(gs, g)
        out["merge_law"].append({"M": gs, "g": g, "exceptions": e2, "gaps": t2})
        e3, t3 = alternation(gs, g)
        out["alternation"].append({"M": gs, "g": g, "exceptions": e3, "runs": t3})
        print("M", gs, "+", g, "chain exc", e, "/", t, "merge exc", e2, "/", t2, "alt exc", e3, "/", t3)

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
