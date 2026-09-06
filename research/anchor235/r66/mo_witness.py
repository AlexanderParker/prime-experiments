"""mo_witness.py -- the WITNESSES: for each candidate functional that fails, the exact merge on
the refuting rung that does it.

For a rung y -> q' the openings of M + q' are the openings of M (over q' copies of M's period)
minus those struck by the new gear, so any stretch of M + q' pulls back to a stretch of M and the
merge that made it is read off directly.  Everything here is exact.

Usage: uv run python research/anchor235/r66/mo_witness.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mo_core import OUT, PRIMES, u_of
from lc_core import gaps_of, sieve_machine


def machine(top):
    gears = [p for p in PRIMES if p <= top]
    P, O = sieve_machine(gears)
    return P, O, gaps_of(P, O)


def widest_run(g, J):
    """(span, index) of the widest cyclic run of J consecutive gaps."""
    n = g.size
    gg = np.concatenate([g, g[:J]]).astype(np.int64)
    pre = np.concatenate([[0], np.cumsum(gg)])
    s = pre[J:J + n] - pre[:n]
    i = int(s.argmax())
    return int(s[i]), i


def pullback(y, q, J):
    """The widest J-run of M + q', and the stretch of M it fuses."""
    P, O, g = machine(y)
    Obig = np.concatenate([O + j * P for j in range(q)])
    u = u_of(q)
    keep = ((Obig % q) != (u % q)) & ((Obig % q) != ((-u) % q))
    Onew = Obig[keep]
    gnew = gaps_of(q * P, Onew)
    span, i = widest_run(gnew, J)
    lo = Onew[i]
    hi = Onew[(i + J) % Onew.size] + (q * P if i + J >= Onew.size else 0)
    sel = (Obig >= lo) & (Obig <= hi)
    old = Obig[sel]
    oldgaps = [int(x) for x in np.diff(old)]
    newgaps = [int(x) for x in np.diff(np.concatenate([Onew[i:i + J + 1]]))] if i + J < Onew.size \
        else None
    struck = [bool(not k) for k in keep[sel]]
    # which old gaps each new gap is made of
    pieces, cur = [], []
    for a, b in zip(oldgaps, struck[1:]):
        cur.append(a)
        if not b:
            pieces.append(cur)
            cur = []
    return {"machine": f"m{y}", "gear": q, "J": J, "span": span,
            "new_gaps": newgaps, "old_stretch": oldgaps, "n_old_gaps": len(oldgaps),
            "fusion_pieces": pieces,
            "orders": [len(p) for p in pieces]}


def main():
    out = {}
    for (y, q, J) in [(19, 23, 6), (19, 23, 5), (17, 19, 3), (13, 17, 3)]:
        r = pullback(y, q, J)
        out[f"F_{J}_{y}to{q}"] = r
        print(f"F_{J}: m{y} -> {q}: span {r['span']} from {r['n_old_gaps']} gaps of m{y}; "
              f"new gaps {r['new_gaps']}; orders {r['orders']}", flush=True)
        print(f"    old stretch {r['old_stretch']}", flush=True)
    # top-3 spectrum values per machine, for candidate (b')
    top = {}
    rungs = json.load(open(os.path.join(OUT, "rungs.json")))
    for r in rungs:
        sp = {int(k): v for k, v in r["spec_full"].items()}
        top[f"m{r['machine']}"] = sorted(sp)[-3:]
    lad = os.path.join(OUT, "ladder_K15_base23.json")
    if os.path.exists(lad):
        for r in json.load(open(lad)):
            sp = {int(k): v for k, v in r["spec_full"].items()}
            top[f"m{r['machine']}"] = sorted(sp)[-3:]
    out["top3_values"] = top
    print("top-3 realised values:", top, flush=True)
    with open(os.path.join(OUT, "witness.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
