# Branch 5g: why the gears that are NOT at coverage maximum are not.
#
# For each record stretch of m13..m31 and each gear g: g's actual coverage c_g, its maximum
# m_g, its SOLE columns (offsets only g strikes), and the test that matters -
#   is there ANY phase of g that attains m_g and still strikes all of g's sole offsets?
# If not, g's deficit is FORCED by the sole-column requirement: the record buys placement with
# bulk.  If yes, the deficit is a free choice and the "greedy from the bottom" reading fails.
#
# Self-contained; numpy only.  Run: uv run python research/anchor235/r36/e1_deficit.py
import os
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "e1_deficit.txt")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


LAD = [5, 7, 11, 13, 17, 19, 23, 29, 31]
FK = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
BIG = {29: [200906186, 877375978],
       31: [1468940243, 11582483683, 21844264616, 31957808056]}


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


def hits(s, L, g):
    a, b = teeth(g)
    return {j for j in range(L) if (s + j) % g in (a, b)}


say("Branch 5g: the deficit of a below-maximum gear, and whether the sole columns force it.")
say("")
for q in (13, 17, 19, 23, 29, 31):
    gears = [g for g in LAD if g <= q]
    F, L = FK[q], FK[q] - 1
    if q in BIG:
        starts = BIG[q]
    else:
        P = prod(gears)
        w = np.ones(P, bool)
        for g in gears:
            for t in teeth(g):
                w[t::g] = False
        op = np.flatnonzero(w)
        d = np.diff(np.concatenate([op, [op[0] + P]]))
        starts = [int(op[j]) + 1 for j in np.flatnonzero(d == F)]
    s = starts[0]
    H = {g: hits(s, L, g) for g in gears}
    sole = {g: {j for j in H[g] if sum(1 for h in gears if j in H[h]) == 1} for g in gears}
    say(f"  m{q}   L = {L}   record start {s}   (one of {len(starts)})")
    say("    gear  c_g  m_g  deficit  sole  waste  a maximal phase covering the sole columns?")
    for g in gears:
        a, b = teeth(g)
        m = max(len(hits(r, L, g)) for r in range(g))
        # phases attaining m that also strike every sole offset of g
        good = [r for r in range(g)
                if len(hits(r, L, g)) == m and sole[g] <= hits(r, L, g)]
        say(f"    {g:>4}  {len(H[g]):>3}  {m:>3}  {m-len(H[g]):>7}  {len(sole[g]):>4}  "
            f"{len(H[g])-len(sole[g]):>5}  "
            f"{'yes (' + str(len(good)) + ' phases)' if good else 'NO - the deficit is forced'}")
    say("")

say("Reading: 'NO' means no phase of that gear both attains its coverage maximum and keeps the")
say("columns only it strikes, so its deficit is bought by placement.  'yes' means the gear could")
say("have had both and the record does not use the extra strike.")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written", OUT)
