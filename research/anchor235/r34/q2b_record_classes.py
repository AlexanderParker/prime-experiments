# Branch 7d, question 2 follow-up: the structure of the SET of record stretches beyond the mirror.
# q2 found start differences divisible by 5g far above chance (e.g. m17: 117, 502, 5642, 6027 = 117 + {0, 385, 5525, 5910}).
# Here: for each level, every record stretch's kill map (which relative columns each gear strikes, exclusive kills marked),
# the residue-agreement structure between stretches (gears at which two starts agree mod g), and the orbit decomposition.
# Self-contained; numpy only.  Run: uv run python research/anchor235/r34/q2b_record_classes.py
import numpy as np, os
from math import prod
from itertools import combinations

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "q2b_record_classes.txt")
lines = []
def say(s=""):
    print(s); lines.append(s)

def openings(gears, P):
    w = np.ones(P, bool)
    for g in gears:
        u = pow(6, -1, g); w[u::g] = False; w[g - u::g] = False
    return np.flatnonzero(w)

ladder = [7, 11, 13, 17, 19, 23]
say("Q2b: the set of record stretches - kill maps and residue coincidences.  Relative column j = 1..F-1 of the stretch after opening x.")
for i, q in enumerate(ladder):
    gears = [5] + ladder[:i + 1]; P = prod(gears)
    op = openings(gears, P); gaps = np.diff(np.concatenate([op, [op[0] + P]]))
    F = int(gaps.max()); starts = [int(op[j]) for j in np.flatnonzero(gaps == F)]
    say(f"\nM = {{5..{q}}}, F = {F}, {len(starts)} record stretches")
    maps = {}
    for x in starts:
        cols = np.arange(x + 1, x + F)
        km = {}
        nstrike = np.zeros(F - 1, np.int32)
        for g in gears:
            u = pow(6, -1, g); hit = np.isin(cols % g, (u, g - u)); km[g] = hit; nstrike += hit
        desc = []
        for g in gears:
            offs = np.flatnonzero(km[g]) + 1
            excl = np.flatnonzero(km[g] & (nstrike == 1)) + 1
            desc.append(f"{g}:{list(map(int, offs))}" + (f"*{list(map(int, excl))}" if len(excl) else ""))
        key = tuple(tuple(np.flatnonzero(km[g])) for g in gears)
        maps.setdefault(key, []).append(x)
        say(f"  x={x:>9} (frac {(x + 1) / P:.4f}, residues " + ",".join(f"{(x + 1) % g}" for g in gears) + ")  kills per gear (relative cols; * = exclusive): " + " ".join(desc))
    say(f"  distinct kill maps: {len(maps)} for {len(starts)} stretches (a kill map fixes every residue, so each map occurs once; the mirror reverses maps)")
    # residue agreement structure
    say("  pairs of stretches and the gears at which their starts agree mod g (i.e. the same phase; the kills of those gears coincide relative to the stretch):")
    agree = {}
    for a, b in combinations(starts, 2):
        S = tuple(g for g in gears if (a - b) % g == 0)
        if len(S) >= 2 or (len(S) == 1 and S[0] != 5):
            agree.setdefault(S, []).append((a, b))
    for S, prs in sorted(agree.items(), key=lambda t: -len(t[0])):
        say(f"    agree at {S}: {len(prs)} pairs, e.g. {prs[:3]}  (chance rate per pair 1/{prod(S)} -> expected {len(starts) * (len(starts) - 1) / 2 / prod(S):.2f})")
    # translation classes: group starts by residues at the small gears {5,7,11}
    for cut in range(1, len(gears)):
        low = gears[:cut + 1]
        cls = {}
        for x in starts: cls.setdefault(tuple((x + 1) % g for g in low), []).append(x)
        sizes = sorted(len(v) for v in cls.values())
        say(f"  classes by phase at {low}: {len(cls)} classes, sizes {sizes}")
with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
print("written", OUT)
