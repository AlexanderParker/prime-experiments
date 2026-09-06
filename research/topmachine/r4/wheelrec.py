"""The wheel record as core plus tail (P1, P2).

L16 (top_machine_1.md): F_top(G) is the largest L such that [0, L) can be covered by giving
each gear a phase.  A gear g > L + 1 shows exactly one domino {x, x + 2} inside the window
(or a single cell when its partner falls outside); a gear g <= L + 1 shows a longer, g-periodic
trace that must be taken as it is.  So:

    core  = {g in G : g <= L + 1}       traces enumerated exactly
    tail  = {g in G : g >  L + 1}       t free dominoes, all in one parity class each

    D(U) = sum over maximal step-2 runs of the uncovered set U, inside each parity class,
           of ceil(run length / 2)                          (the domino cost)

    F_top(G) = max { L : min over core phase vectors of D(U) <= t } .

usage: uv run python research/topmachine/r4/wheelrec.py
"""

import json
import os
from math import prod

import numpy as np

from zone import primes_upto

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []
NODES = [0]


def say(s=""):
    print(s)
    OUT.append(str(s))


def domino_cost(unc, L):
    """D(U): the number of tail dominoes needed to cover the uncovered set U."""
    cost = 0
    for par in (0, 1):
        cells = [c for c in unc if c % 2 == par]
        if not cells:
            continue
        run = 1
        for a, b in zip(cells, cells[1:]):
            if b - a == 2:
                run += 1
            else:
                cost += (run + 1) // 2
                run = 1
        cost += (run + 1) // 2
    return cost


def traces(g, L):
    """All distinct traces of gear g inside [0, L): masks as frozensets."""
    out = []
    seen = set()
    for a in range(g):
        cells = tuple(sorted(set([c for c in range(a % g, L, g)]
                                 + [c for c in range((a + g - 2) % g, L, g)])))
        if cells not in seen:
            seen.add(cells)
            out.append(cells)
    return out


def feasible(gears, L, node_cap=300_000):
    """Is [0, L) coverable?  Exact when the search completes; None if the cap is hit."""
    core = [g for g in gears if g <= L + 1]
    t = len([g for g in gears if g > L + 1])
    if not core:
        # pure tail: the parity law's covering count
        return domino_cost(list(range(L)), L) <= t
    tr = [traces(g, L) for g in core]
    # cheapest first in the search: gears with the most cells first
    order = sorted(range(len(core)), key=lambda i: -max(len(c) for c in tr[i]))
    tr = [tr[i] for i in order]
    sizes = [max(len(c) for c in tr[i]) for i in range(len(tr))]
    suffix = [0] * (len(tr) + 1)
    for i in range(len(tr) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sizes[i]
    full = set(range(L))
    NODES[0] = 0

    def dfs(i, covered):
        NODES[0] += 1
        if NODES[0] > node_cap:
            raise TimeoutError
        unc = sorted(full - covered)
        if not unc:
            return True
        if i == len(tr):
            return domino_cost(unc, L) <= t
        # valid bound: covering c more cells leaves at least ceil((u - c)/2) dominoes
        if (len(unc) - suffix[i] + 1) // 2 > t:
            return False
        for cells in tr[i]:
            if dfs(i + 1, covered | set(cells)):
                return True
        return False

    try:
        return dfs(0, set())
    except TimeoutError:
        return None


def F_top_engine(gears, lo=1, hi=None):
    """Largest L that is coverable; None if a decision was not reached."""
    if hi is None:
        hi = 3 * len(gears) + 40
    best = 0
    L = lo
    undecided = []
    while L <= hi:
        f = feasible(gears, L)
        if f is None:
            undecided.append(L)
            L += 1
            continue
        if f:
            best = L
        else:
            # feasibility is monotone in L in every case checked; keep going a little
            if L > best + 6:
                break
        L += 1
    return best, undecided


def F_top_scan(gears):
    """Exact record by a full-period scan (small wheels only)."""
    W = prod(gears)
    a = np.ones(W, dtype=bool)
    idx = np.arange(W)
    for g in gears:
        r = idx % g
        a &= (r != 0) & (r != g - 2)
    pos = np.flatnonzero(a)
    d = np.diff(pos)
    wrap = pos[0] + W - pos[-1]
    return int(max(d.max(), wrap)) - 1


KNOWN = {
    (7, 11, 13): 6, (11, 13, 17): 5, (13, 17, 19): 5, (17, 19, 23): 5, (19, 23, 29): 5,
    (7, 11, 13, 17): 9, (11, 13, 17, 19): 8, (13, 17, 19, 23): 8, (17, 19, 23, 29): 8,
    (11, 13, 17, 19, 23): 10,
    (7, 11, 13, 17, 19, 23, 29, 31): 32,
    (13, 17, 19, 23, 29, 31, 37, 41): 18,
    (19, 23, 29, 31, 37, 41, 43, 47): 16,
}


def main():
    say("# The wheel record as core plus tail")
    say()
    say("## P1  the covering engine against every independently known F_top")
    say()
    say("| gears | m | known F_top | engine | core (g <= F+1) | tail t | agrees |")
    say("|---|---|---|---|---|---|---|")
    bad = 0
    for gs, known in KNOWN.items():
        f, und = F_top_engine(list(gs))
        core = [g for g in gs if g <= f + 1]
        say(f"| {','.join(map(str, gs))} | {len(gs)} | {known} | {f} | "
            f"{'{' + ','.join(map(str, core)) + '}' if core else 'empty'} | "
            f"{len(gs) - len(core)} | {'yes' if f == known else '**NO**'} |")
        if f != known:
            bad += 1
    say()
    say(f"**{bad} mismatches** over {len(KNOWN)} independently known records.")
    say()

    # a wider sweep: consecutive prime gear sets
    say("## P1/P2  consecutive gear sets {q'..Q}: the core, the tail and the two forms")
    say()
    say("| q' | m | gears | F_top | core | t | 2t - (t mod 2) | core's own record | "
        "additive form | additive ok |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    allp = [int(p) for p in primes_upto(400)]
    for qp in (7, 11, 13, 17, 19, 23, 29, 31, 37, 41):
        base = [p for p in allp if p >= qp]
        for m in range(3, 13):
            gs = base[:m]
            f, und = F_top_engine(gs)
            if und:
                continue
            core = [g for g in gs if g <= f + 1]
            t = m - len(core)
            fcore = F_top_engine(core)[0] if len(core) >= 2 else (0 if not core else 1)
            add = fcore + 2 * t - (t % 2)
            rows.append({"q": qp, "m": m, "gears": gs, "F": f, "core": core, "t": t,
                         "F_core": fcore, "additive": add})
            say(f"| {qp} | {m} | {gs[0]}..{gs[-1]} | **{f}** | "
                f"{'{' + ','.join(map(str, core)) + '}' if core else 'empty'} | {t} | "
                f"{2*t - (t % 2)} | {fcore} | {add} | {'yes' if add == f else 'no'} |")
    say()
    nadd = sum(1 for r in rows if r["additive"] == r["F"])
    say(f"The additive form (core's own record + 2t - parity) agrees at **{nadd} of "
        f"{len(rows)}** sets.")
    say()

    json.dump(rows, open(os.path.join(RES, "wheelrec.json"), "w"), indent=1)
    with open(os.path.join(RES, "wheelrec.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
