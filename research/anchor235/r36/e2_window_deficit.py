# Branch 5g: does the window's longest blocked stretch obey the same law as the period record -
# "every gear is at its coverage maximum subject to keeping the columns only it strikes"?
#
# At every prime rung 23..1999, for the window's longest blocked stretch and for every gear with
# m_g >= 2 (below that the ratio is vacuous), classify the gear:
#   AT MAX                       c_g = m_g
#   FORCED DEFICIT               c_g < m_g and no phase of g attains m_g while striking all of
#                                g's sole columns of the stretch
#   FREE DEFICIT                 c_g < m_g and such a phase exists  -> the law FAILS here
# Also the same over all record stretches and runner-up stretches of m13..m31.
#
# Self-contained; numpy only.  Run: uv run python research/anchor235/r36/e2_window_deficit.py
import os
import time
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "e2_window_deficit.txt")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


PR = primes_upto(2100)
NXT = {PR[i]: PR[i + 1] for i in range(len(PR) - 1)}


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


def hitset(delta, L, g):
    """offsets j in [0,L) struck by a copy of gear g whose teeth sit at a+delta, b+delta."""
    a, b = teeth(g)
    out = set()
    for t in (a, b):
        off = (t + delta) % g
        out.update(range(off, L, g))
    return out


def stretch_hits(s, L, g):
    """the actual hits of gear g in the stretch starting at column s: delta = -s."""
    return hitset((-s) % g, L, g)


def maxcov(L, g):
    """max over phases; attained at a window starting on a tooth (gated in b1_hinge.py)."""
    a, b = teeth(g)
    return max(len(hitset((-a) % g, L, g)), len(hitset((-b) % g, L, g)))


def classify(L, cg, mg, soleset, g):
    if cg == mg:
        return "max"
    for delta in range(g):
        h = hitset(delta, L, g)
        if len(h) == mg and soleset <= h:
            return "free"
    return "forced"


# ------------------------------------------------------------------ period side
say("Branch 5g: coverage maximum SUBJECT TO the sole columns - the law, over all extremal")
say("stretches of the full period and over the window's longest stretch at every rung.")
say("")
LAD = [5, 7, 11, 13, 17, 19, 23, 29, 31]
FK = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
BIG = {29: [(42, [200906186, 877375978]),
            (39, None)],
       31: [(57, [1468940243, 11582483683, 21844264616, 31957808056]),
            (54, None)]}
say("=== period side: every RECORD stretch of m13..m31 (all of them), gear by gear")
say("  machine  stretches  gear-cells  at max  forced deficit  FREE deficit (law fails)")
for q in (13, 17, 19, 23, 29, 31):
    gears = [g for g in LAD if g <= q]
    F, L = FK[q], FK[q] - 1
    if q in BIG:
        starts = BIG[q][0][1]
    else:
        P = prod(gears)
        w = np.ones(P, bool)
        for g in gears:
            for t in teeth(g):
                w[t::g] = False
        op = np.flatnonzero(w)
        d = np.diff(np.concatenate([op, [op[0] + P]]))
        starts = [int(op[j]) + 1 for j in np.flatnonzero(d == F)]
    tally = {"max": 0, "forced": 0, "free": 0}
    freecells = []
    for s in starts:
        H = {g: stretch_hits(s, L, g) for g in gears}
        for g in gears:                       # gate: the shifted copy is the real one
            assert H[g] == {j for j in range(L) if (s + j) % g in teeth(g)}
        sole = {g: {j for j in H[g] if sum(1 for h in gears if j in H[h]) == 1} for g in gears}
        for g in gears:
            mg = maxcov(L, g)
            assert mg == max(len(hitset(dd, L, g)) for dd in range(g))
            c = classify(L, len(H[g]), mg, sole[g], g)
            tally[c] += 1
            if c == "free":
                freecells.append((s, g))
    say(f"  m{q:<7}  {len(starts):<9}  {sum(tally.values()):<10}  {tally['max']:<6}  "
        f"{tally['forced']:<14}  {tally['free']}  {freecells[:4]}")

# ------------------------------------------------------------------ window side
say("")
say("=== window side: the longest blocked stretch of every window, prime rungs 23..1999")
say("  only gears with m_g >= 2 are classified (m_g = 1 makes the ratio vacuous).")
t0 = time.time()
TOT = {"max": 0, "forced": 0, "free": 0}
perrung = []
for q in PR:
    if q < 23 or q > 1999:
        continue
    qq = NXT[q]
    lo, hi = q // 6 + 1, (qq * qq - 1) // 6
    n = hi - lo + 1
    gears = [g for g in PR if 5 <= g <= q]
    cnt = np.zeros(n, np.int16)
    gsum = np.zeros(n, np.int64)
    for g in gears:
        for t in teeth(g):
            cnt[(t - lo) % g::g] += 1
            gsum[(t - lo) % g::g] += g
    op = np.flatnonzero(cnt == 0) + lo
    if len(op) < 4:
        continue
    d = np.diff(op)
    j = int(d.argmax())
    L, s = int(d[j]) - 1, int(op[j]) + 1
    i0 = s - lo
    c_ = cnt[i0:i0 + L]
    gs = gsum[i0:i0 + L]
    solemap = {}
    for t in np.flatnonzero(c_ == 1):
        solemap.setdefault(int(gs[t]), set()).add(int(t))
    tally = {"max": 0, "forced": 0, "free": 0}
    freecells = []
    for g in gears:
        mg = maxcov(L, g)
        if mg < 2:
            continue
        H = stretch_hits(s, L, g)
        cls = classify(L, len(H), mg, solemap.get(g, set()), g)
        tally[cls] += 1
        if cls == "free":
            freecells.append(g)
    for k in tally:
        TOT[k] += tally[k]
    perrung.append((q, L, tally, freecells))
    del cnt, gsum, op
say(f"  [{time.time()-t0:.1f}s over {len(perrung)} rungs]")
say(f"  gear-cells: at max {TOT['max']}, forced deficit {TOT['forced']}, "
    f"FREE deficit {TOT['free']}  (share free {TOT['free']/sum(TOT.values()):.3f})")
say(f"  rungs with at least one free deficit: "
    f"{sum(1 for _,_,t,_ in perrung if t['free'] > 0)} of {len(perrung)}")
say("  focus rungs:")
for q, L, t, fc in perrung:
    if q in (59, 173, 499, 997, 1999):
        say(f"    q = {q:<5} L = {L:<4} at max {t['max']:<4} forced {t['forced']:<4} "
            f"free {t['free']:<4} free gears {fc[:12]}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written", OUT)
