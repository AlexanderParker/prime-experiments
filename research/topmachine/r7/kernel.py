"""R7 item 4: THE KERNEL SHAPE OF L22 - the gap census law as three separable lemmas.

L22 (top_machine_2.md):  N_d(G) = sum over S subset of [1, d-1] of (-1)^{|S|}
                                  prod_g (g - |E_g(S)|),
                         E_g(S) = ({0,-2,-d,-d-2} u {-i, -(i+2) : i in S}) mod g.

The three lemmas the Formalist needs, each checked here on its own:

 K1 (local characterisation)  n and n+d are CONSECUTIVE open pairs
       <=>  (a) every gear avoids {0,-2,-d,-d-2} at n   and
            (b) every interior i in [1,d-1] is struck: some gear has n = -i or -(i+2) mod g.
 K2 (inclusion-exclusion)  counting (a) and (b) over n mod W equals
       sum over S of (-1)^{|S|} T(S),  T(S) = #{n : (a) and NO gear strikes n+i for i in S}.
 K3 (CRT product)  T(S) = prod_g (g - |E_g(S)|), because "(a) and no gear strikes n+i for
       i in S" is the per-gear condition "n mod g avoids E_g(S)".

usage: uv run python research/topmachine/r7/kernel.py
"""

import itertools
import json
import os
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []

WHEELS = [(7, 11, 13), (11, 13, 17), (5, 7, 11, 13), (7, 11, 13, 17), (9, 11, 13, 17)]
DMAX = 10


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def struck_array(gears, W):
    st = np.zeros(W, dtype=bool)
    for g in gears:
        st[0::g] = True
        st[(g - 2) % g::g] = True
    return st


def main():
    summary = {}
    say("# R7.4  The kernel shape of L22, verified lemma by lemma")
    say()
    say("| wheel | W | d range | K1 mismatches | K2 mismatches | K3 mismatches | "
        "L22 vs scan |")
    say("|---|---|---|---|---|---|---|")
    tot = {"K1": 0, "K2": 0, "K3": 0, "L22": 0}
    rows = []
    for gears in WHEELS:
        W = prod(gears)
        st = struck_array(gears, W)
        openidx = np.flatnonzero(~st)
        # direct census: gaps between consecutive open pairs, cyclically
        nxt = np.roll(openidx, -1)
        dist = (nxt - openidx) % W
        census = {}
        for v in dist:
            census[int(v)] = census.get(int(v), 0) + 1

        b1 = b2 = b3 = bl = 0
        for d in range(1, DMAX + 1):
            # ---- K1: the local characterisation, against the direct census
            a_ok = ~st & ~np.roll(st, -d)            # n open and n + d open
            if d == 1:
                interior = np.ones(W, dtype=bool)
            else:
                interior = np.ones(W, dtype=bool)
                for i in range(1, d):
                    interior &= np.roll(st, -i)
            k1 = int(np.count_nonzero(a_ok & interior))
            if k1 != census.get(d, 0):
                b1 += 1

            # ---- K2 / K3: inclusion-exclusion over the interior positions
            total = 0
            for r in range(0, d):
                for S in itertools.combinations(range(1, d), r):
                    cond = a_ok.copy()
                    for i in S:
                        cond &= ~np.roll(st, -i)      # no gear strikes n + i
                    T_scan = int(np.count_nonzero(cond))
                    # K3: the CRT product
                    p = 1
                    for g in gears:
                        E = set()
                        for v in (0, -2, -d, -d - 2):
                            E.add(v % g)
                        for i in S:
                            E.add((-i) % g)
                            E.add((-i - 2) % g)
                        p *= g - len(E)
                    if T_scan != p:
                        b3 += 1
                    total += (-1) ** len(S) * T_scan
            if total != k1:
                b2 += 1

            # ---- the assembled law against the scan
            law = 0
            for r in range(0, d):
                for S in itertools.combinations(range(1, d), r):
                    p = 1
                    for g in gears:
                        E = set()
                        for v in (0, -2, -d, -d - 2):
                            E.add(v % g)
                        for i in S:
                            E.add((-i) % g)
                            E.add((-i - 2) % g)
                        p *= g - len(E)
                    law += (-1) ** r * p
            if law != census.get(d, 0):
                bl += 1

        tot["K1"] += b1
        tot["K2"] += b2
        tot["K3"] += b3
        tot["L22"] += bl
        rows.append({"gears": list(gears), "W": W, "K1": b1, "K2": b2, "K3": b3, "L22": bl,
                     "census": {str(k): v for k, v in sorted(census.items())}})
        say(f"| {','.join(map(str, gears))} | {W:,} | 1..{DMAX} | {b1} | {b2} | {b3} | {bl} |")
    say()
    say(f"**K1 {tot['K1']}, K2 {tot['K2']}, K3 {tot['K3']}, assembled L22 {tot['L22']} "
        f"mismatches** over {len(WHEELS)} wheels x {DMAX} gap lengths "
        f"({sum(2 ** (d - 1) for d in range(1, DMAX + 1)) * len(WHEELS):,} subset terms).")
    say()
    say("Censuses (gap length: count per period), for the record:")
    say()
    for r in rows:
        items = ", ".join(f"{k}:{v}" for k, v in list(r["census"].items())[:12])
        say(f"- `{{{','.join(map(str, r['gears']))}}}` (W = {r['W']:,}): {items}")
    say()
    summary["totals"] = tot
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(RES, "kernel.json"), "w"), indent=1)
    with open(os.path.join(RES, "kernel.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
