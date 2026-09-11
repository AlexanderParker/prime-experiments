"""og_brute.py -- second measurement of d(p) (og_nearest.py) by brute force over the WHOLE tooth family at
p = 17 (1,440 members) and p = 29 (1,995,840 members): every member's kill status and its Hamming distance
from the real teeth; reports the killer count (gate: 15 and 6,030, first_realisation.md 3.5), the minimum
distance among killers, and the killers at that distance. Independent of og_nearest.py's subset search.
Output: results/brute.json
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from og_common import finer_section, real_teeth

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def run(p):
    a, b, cols, gears = finer_section(p)
    L = len(cols)
    full = (1 << L) - 1
    rt = real_teeth(gears)
    per = []
    for g in gears:
        rows = []
        for v in range(1, (g - 1) // 2 + 1):
            m = 0
            for i, j in enumerate(cols):
                if (j % g) in (v, g - v):
                    m |= 1 << i
            rows.append((v, m))
        per.append(rows)
    t0 = time.time()
    killers = 0
    best = None
    best_members = []
    members = 0
    # iterate: nested loops via itertools.product over masks, tracking distance
    for combo in itertools.product(*per):
        members += 1
        m = 0
        for v, mm in combo:
            m |= mm
        if m == full:
            killers += 1
            d = sum(1 for (v, _), w in zip(combo, rt) if v != w)
            if best is None or d < best:
                best = d
                best_members = [[v for v, _ in combo]]
            elif d == best:
                best_members.append([v for v, _ in combo])
    return dict(p=p, members=members, killers=killers, min_distance=best, killers_at_min=best_members[:10],
                n_killers_at_min=len(best_members), seconds=round(time.time() - t0, 1))


def main():
    out = []
    for p in (17, 29):
        r = run(p)
        print(r, flush=True)
        out.append(r)
    with open(os.path.join(RES, "brute.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
