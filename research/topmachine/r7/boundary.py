"""R7 item 2: THE FREE/LOADED BOUNDARY AS A COROLLARY of the loaded record rule.

Empty core  =>  U = [0, L) and D(L) = ceil(ceil(L/2)/2) + ceil(floor(L/2)/2).
max { L : D(L) <= m } = 2m - (m mod 2): that IS the parity law (L17), now derived.

The threshold: the parity law holds iff [0, P + 1) is not coverable, P = 2m - (m mod 2).
At q' = 2m + 1 exactly one gear is a core gear at that length, and it offers exactly one
parity-crossing piece.  This script measures what that piece leaves behind, at both parities.

usage: uv run python research/topmachine/r7/boundary.py
"""

import json
import os
from math import gcd

from rule import D_empty, F_rule, coverable, domino_cost_set, traces

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def gearset(qp, m):
    """m pairwise-coprime odd gears, the smallest exactly qp."""
    gs = [qp]
    g = qp
    while len(gs) < m:
        g += 2
        if all(gcd(g, h) == 1 for h in gs):
            gs.append(g)
    return gs


def main():
    summary = {}
    say("# R7.2  The free/loaded boundary, as a corollary")
    say()

    # ---------------------------------------------------------------- the empty-core cost
    say("## The empty-core domino cost, and the parity law from it")
    say()
    say("| L | ceil(L/2) even cells | floor(L/2) odd cells | D(L) | 2 floor(L/4) + min(L mod 4,2) |")
    say("|---|---|---|---|---|")
    for L in range(0, 15):
        say(f"| {L} | {(L+1)//2} | {L//2} | {D_empty(L)} | {2*(L//4) + min(L % 4, 2)} |")
    say()
    bad = 0
    rows = []
    for m in range(1, 201):
        best = max(L for L in range(0, 4 * m + 8) if D_empty(L) <= m)
        want = 2 * m - (m % 2)
        rows.append((m, best, want))
        if best != want:
            bad += 1
    say(f"`max {{ L : D(L) <= m }}` against `2m - (m mod 2)`, `m = 1..200`: "
        f"**{bad} mismatches**.")
    say()
    say("Mechanism, exactly.  Write `L = 4k + r`.  Then `D(L) = 2k, 2k+1, 2k+2, 2k+2` for "
        "`r = 0,1,2,3`.  At `m = 2k` the largest `L` with `D(L) <= m` is `4k = 2m`; at "
        "`m = 2k+1` it is `4k+1 = 2m - 1`.  The parity defect is the single odd cell that "
        "cannot be paired inside its own class.")
    say()
    summary["parity_law_from_D_mismatches"] = bad

    # ---------------------------------------------------------------- the threshold
    say("## The sharp threshold, measured with the rule")
    say()
    say("For each `m`, the smallest odd `q'` at which the parity law holds, over pairwise "
        "coprime odd gear sets `{q', ...}` built by taking the next coprime odd number each "
        "time (odd composites included, as `top_machine_5.md` did):")
    say()
    say("| m | P = 2m - (m mod 2) | q' = 2m-1 | q' = 2m+1 | q' = 2m+3 | smallest q' that works "
        "| predicted |")
    say("|---|---|---|---|---|---|---|")
    badth = 0
    thr = {}
    for m in range(2, 10):
        P = 2 * m - (m % 2)
        vals = {}
        for qp in (2 * m - 1, 2 * m + 1, 2 * m + 3):
            if qp < 5:
                vals[qp] = None
                continue
            gs = gearset(qp, m)
            vals[qp] = F_rule(gs)
        works = [qp for qp in (2 * m - 1, 2 * m + 1, 2 * m + 3) if vals.get(qp) == P]
        smallest = min(works) if works else None
        pred = 2 * m + 1 if m % 2 == 0 else 2 * m + 3
        thr[m] = {"P": P, "vals": {str(k): v for k, v in vals.items()},
                  "smallest": smallest, "pred": pred}
        if smallest != pred:
            badth += 1
        say(f"| {m} | {P} | {vals.get(2*m-1)} | {vals.get(2*m+1)} | {vals.get(2*m+3)} | "
            f"{smallest} | {pred}{' **NO**' if smallest != pred else ''} |")
    say()
    say(f"**{badth} mismatches** against `q' >= 2m + 1` (even `m`), `2m + 3` (odd `m`) - "
        "`top_machine_5.md` L55's threshold, here a corollary of the record rule.")
    say()
    summary["threshold_mismatches"] = badth
    summary["threshold"] = thr

    # ---------------------------------------------------------------- the mechanism
    say("## Why odd `m` needs one more: what the single core gear leaves behind")
    say()
    say("At `q' = 2m + 1` and `L = P + 1` the only core gear (`g <= L + 1`) is `q'` itself. "
        "Its best piece, and the cost of what remains:")
    say()
    say("| m | P | L = P+1 | q' | q' vs L | best core piece | cells left | runs left | "
        "D(left) | t = m-1 | coverable |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in range(2, 10):
        P = 2 * m - (m % 2)
        L = P + 1
        qp = 2 * m + 1
        gs = gearset(qp, m)
        core = [g for g in gs if g <= L + 1]
        t = m - len(core)
        best = None
        for tr in traces(qp, L):
            cells = [c for c in range(L) if (tr >> c) & 1]
            left = [c for c in range(L) if not ((tr >> c) & 1)]
            cost = domino_cost_set(left)
            if best is None or cost < best[2]:
                runs = []
                for par in (0, 1):
                    cs = [c for c in left if c % 2 == par]
                    run = []
                    for c in cs:
                        if run and c - run[-1] == 2:
                            run.append(c)
                        else:
                            if run:
                                runs.append(len(run))
                            run = [c]
                    if run:
                        runs.append(len(run))
                best = (cells, left, cost, runs)
        cov = coverable(gs, L)
        rel = "= L" if qp == L else ("= L+1" if qp == L + 1 else f"{qp} vs L={L}")
        say(f"| {m} | {P} | {L} | {qp} | {rel} | {{{','.join(map(str, best[0]))}}} | "
            f"{len(best[1])} | {best[3]} | {best[2]} | {t} | "
            f"{'YES -> parity law FAILS' if cov else 'no -> parity law holds'} |")
    say()
    say("The reading.  At even `m` the core gear sits at `g = L` and its parity-crossing piece "
        "is the wrap pair `{L-2, 0}`; removing it leaves runs of `m` and `m-1` cells, cost "
        "`m/2 + m/2 = m > t = m - 1`.  At odd `m` the core gear sits at `g = L + 1` and its "
        "piece is the end pair `{0, L-1}`; removing it leaves two runs of `m - 1` cells each, "
        "and `m - 1` is EVEN, so the cost is `(m-1)/2 + (m-1)/2 = m - 1 = t` and the cover "
        "closes.  That single parity bit is the whole of the `+2`.")
    say()

    json.dump(summary, open(os.path.join(RES, "boundary.json"), "w"), indent=1)
    with open(os.path.join(RES, "boundary.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
