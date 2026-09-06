"""pc_corrcap.py -- P4: CORRCAP_3(c), the small-alphabet cap, for every class c mod 210 coprime
to 210, with gears {5,7}; the PSORD gate (docs/proofs/12's table must come back); and the same
cap with gears {5,7,11} (classes mod 2310) and {5,7,11,13} (mod 30030) to see what the next
gears take off.

Usage: uv run python research/anchor235/r67/pc_corrcap.py
"""
import json
import os
import sys
import time
from collections import Counter
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_core import OUT, corrcap3, psord

PSORD_TABLE = {1: [11, 13, 17, 19, 41, 43, 47, 71, 73, 79, 101, 103, 107, 109, 131, 137, 139, 163,
                   167, 169, 191, 193, 197, 199],
               2: [29, 59, 151, 181],
               3: [1, 23, 31, 61, 67, 89, 97, 113, 121, 143, 149, 179, 187, 209],
               5: [37, 53, 83, 127, 157, 173]}


def main():
    t0 = time.time()
    out = {"psord": {}, "corrcap3_57": {}, "witness_57": {}, "corrcap3_5711": {}, "corrcap3_571113": {}}
    classes = [c for c in range(1, 210) if gcd(c, 210) == 1]
    gate_ok = True
    for c in classes:
        p, _ = psord(c)
        out["psord"][c] = p
        want = [k for k, v in PSORD_TABLE.items() if c in v][0]
        if p != want:
            gate_ok = False
            print(f"PSORD GATE FAIL at class {c}: {p} vs {want}")
        v, w = corrcap3(c)
        out["corrcap3_57"][c] = "inf" if v is None else v
        out["witness_57"][c] = w
    print("PSORD gate:", "OK (48 of 48)" if gate_ok else "FAIL")
    dist = Counter(out["corrcap3_57"].values())
    print("CORRCAP_3 with gears 5,7 by class mod 210:", dict(sorted(dist.items(), key=str)))
    for c in classes:
        print(f"  c={c:3d} PSORD={out['psord'][c]} CORRCAP_3={out['corrcap3_57'][c]} "
              f"witness={out['witness_57'][c]}")
    # gears 5, 7, 11: classes mod 2310
    classes2 = [c for c in range(1, 2310) if gcd(c, 2310) == 1]
    d2 = {}
    for c in classes2:
        v, _ = corrcap3(c, gears=(5, 7, 11))
        d2[c] = "inf" if v is None else v
    out["corrcap3_5711"] = d2
    agg = {}
    for c, v in d2.items():
        agg.setdefault(c % 210, []).append(v)
    print("CORRCAP_3 with gears 5,7,11: distribution", dict(sorted(Counter(d2.values()).items(), key=str)))
    print("  per class mod 210, max over the 10 lifts:",
          {c: max([x for x in agg[c] if x != "inf"], default="inf") if "inf" not in agg[c] else "inf"
           for c in classes})
    out["corrcap3_5711_max_by_210"] = {c: (max(agg[c]) if "inf" not in agg[c] else "inf") for c in classes}
    # gears 5, 7, 11, 13: classes mod 30030 (5760 classes)
    classes3 = [c for c in range(1, 30030) if gcd(c, 30030) == 1]
    d3 = Counter()
    worst = {}
    for c in classes3:
        v, _ = corrcap3(c, gears=(5, 7, 11, 13))
        key = "inf" if v is None else v
        d3[key] += 1
        c0 = c % 210
        worst[c0] = max(worst.get(c0, 0), v if v is not None else 10 ** 6)
    out["corrcap3_571113_dist"] = dict(d3)
    out["corrcap3_571113_max_by_210"] = worst
    print("CORRCAP_3 with gears 5,7,11,13: distribution", dict(sorted(d3.items(), key=str)))
    print("  max by class mod 210:", worst)
    with open(os.path.join(OUT, "corrcap3.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
