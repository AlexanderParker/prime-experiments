"""R7 item 1, sharpness of the tail hypothesis, over the whole scannable family.

The rule's core is {g <= L + 1}.  Move the boundary gear g = L + 1 into the tail - i.e. treat
it as if it showed a free distance-2 domino - and the rule breaks.  This sweeps every gear set
of rule.py section B and counts where.

Also the CAPACITY BOUND corollary: L coverable => L <= 2 t(L) + sum over core of 2 ceil(L/g).

usage: uv run python research/topmachine/r7/sharp.py
"""

import itertools
import json
import os
from math import prod

from rule import KNOWN, F_rule, SCAN_CAP, pairwise_coprime, primes_upto

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []
POOL = [5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49]


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def capacity_ok(gears, L):
    return L <= sum(2 if g > L + 1 else 2 * -(-L // g) for g in gears)


def Lcap(gears):
    best = 0
    for L in range(1, 12 * len(gears) + 60):
        if capacity_ok(gears, L):
            best = L
    return best


def main():
    summary = {}
    say("# R7.1b  The tail hypothesis is sharp, and the capacity bound")
    say()

    sets = []
    for m in range(2, 7):
        for s in itertools.combinations(POOL, m):
            if prod(s) <= SCAN_CAP and pairwise_coprime(s):
                sets.append(s)
    say(f"## The wrong boundary, over all {len(sets)} scannable gear sets")
    say()
    diff = []
    for s in sets:
        f1 = F_rule(list(s))
        f0 = F_rule(list(s), boundary=0)
        if f0 != f1:
            diff.append((list(s), f1, f0))
    say(f"Treating `g = L + 1` as a tail gear changes the answer at **{len(diff)} of "
        f"{len(sets)}** gear sets, always downward "
        f"(min {min((b - c for _, b, c in diff), default=0)}, "
        f"max {max((b - c for _, b, c in diff), default=0)} short).")
    say()
    say("| gears | m | true F_top | with the wrong boundary |")
    say("|---|---|---|---|")
    for s, f1, f0 in diff[:15]:
        say(f"| {','.join(map(str, s))} | {len(s)} | {f1} | {f0} |")
    say()
    say("The gear at `g = L + 1` shows the **end pair** `{0, L-1}` and the gear at `g = L` the "
        "**wrap pair** `{L-2, 0}`; both join the two ends of the window, both cross parity, and "
        "neither is a distance-2 domino.  That is why the tail is `g > L + 1` and not `g > L`.")
    say()
    summary["wrong_boundary_sets"] = len(sets)
    summary["wrong_boundary_differences"] = len(diff)
    summary["wrong_boundary_examples"] = diff[:40]

    # ---------------------------------------------------------------- capacity bound
    say("## The capacity bound (a closed-form upper bound, no enumeration)")
    say()
    say("| gears | m | F_top | Lcap | slack |")
    say("|---|---|---|---|---|")
    bad = 0
    n = 0
    worst = 0
    rows = []
    allp = primes_upto(200)
    fam = list(KNOWN.keys()) + [tuple([p for p in allp if p >= qp][:m])
                                for qp in (7, 11, 13, 17, 19, 23, 29, 37)
                                for m in (4, 5, 6)]
    for gs in fam:
        f = F_rule(list(gs))
        lc = Lcap(list(gs))
        n += 1
        if lc < f:
            bad += 1
        worst = max(worst, lc - f)
        rows.append({"gears": list(gs), "F": f, "Lcap": lc})
        if len(gs) >= 5 or gs in KNOWN:
            say(f"| {','.join(map(str, gs))} | {len(gs)} | {f} | {lc} | {lc - f} |")
    say()
    say(f"**{bad} violations** of `F_top <= Lcap` over {n} gear sets; worst slack **{worst}**.")
    say()
    summary["capacity_violations"] = bad
    summary["capacity_sets"] = n
    summary["capacity_worst_slack"] = worst

    json.dump({"summary": summary, "capacity": rows},
              open(os.path.join(RES, "sharp.json"), "w"), indent=1)
    with open(os.path.join(RES, "sharp.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
