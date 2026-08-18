"""Round 12 lateral: the FIRING LAW - which fuel sites become real chains.

THE LAW (derived, then verified exactly): in a k-chain of gear q', after an
R-kill the next kill is +s (s = 2u mod q', u = 6^-1 mod q'), after an L-kill
+(q'-s). So the spacing word's FIRST entry fixes the orientation and hence a
SINGLE firing residue:

    word starts with s      =>  R-first  =>  fires iff p = -u (mod q')
    word starts with q'-s   =>  L-first  =>  fires iff p = +u (mod q')

One residue per site (not two): ensemble firing fraction = 1/q', HALF the
naive 2/q'. Corollary (mirror): partner site has the reversed word and fires
iff p* = u-type residue; both members of a mirror pair can fire iff
P_M = span (mod q') - a per-step constant, usually violated.

Verified here on every censused step with N3 > 0:
  19->23 (62 sites), 29->31 (13000 + 4), 31->37 (70964 + 216, period 3.34e10).
Cross-validated against mechanic's fuel_census.csv counts.

Run: uv run python research/firing_ratio.py [ymax]   (repo root; numpy)
"""
import sys
from collections import Counter
from math import prod

import numpy as np

from split_gap_law import primes
from topgap_corridor import chunk_openings

STEPS = [(19, 23), (29, 31), (31, 37)]

def words_for(qp):
    u = pow(6, -1, qp)
    s = (2 * u) % qp
    out = {}
    for k in (3, 4):
        w_R = tuple((s if i % 2 == 0 else qp - s) for i in range(k - 1))
        w_L = tuple((qp - s if i % 2 == 0 else s) for i in range(k - 1))
        out[k] = [(w_R, (qp - u) % qp, 'R-first'), (w_L, u, 'L-first')]
    return out, u, s

def scan_step(y, qp, chunk=20_000_000):
    gears = primes(5, y)
    P = prod(gears)
    W, u, s = words_for(qp)
    found = {k: {w[0]: [] for w in ws} for k, ws in W.items()}
    carry = None
    a = 0
    while a < P:
        S = min(chunk, P - a)
        ops = chunk_openings(gears, a, S)
        ext = ops if carry is None else np.concatenate((carry, ops))
        d = np.diff(ext)
        for k, ws in W.items():
            n = k - 1
            if len(d) < n:
                continue
            for w, res, tag in ws:
                m = d[:len(d) - n + 1] == w[0]
                for j in range(1, n):
                    m &= d[j:len(d) - n + 1 + j] == w[j]
                for i in np.flatnonzero(m):
                    found[k][w].append(int(ext[i]))
        carry = ext[-8:]
        a += S
    print(f"STEP {y}->{qp}: period {P}, u={u}, s={s}, P mod {qp} = {P % qp}")
    for k, ws in W.items():
        tot = fired = wrong = 0
        for w, res, tag in ws:
            ps = sorted(set(found[k][w]))
            if not ps:
                continue
            f = [p for p in ps if p % qp == res]
            other = (qp - res) % qp
            wr = [p for p in ps if p % qp == other]
            tot += len(ps)
            fired += len(f)
            wrong += len(wr)
            span = sum(w)
            dbl = "possible" if P % qp == span % qp else "impossible"
            print(f"  k={k} word {w} ({tag}, fire-residue {res}): sites {len(ps)}, "
                  f"FIRED {len(f)} at {f[:6]}{'...' if len(f) > 6 else ''}; "
                  f"opposite-residue {len(wr)} (law: these CANNOT fire); "
                  f"mirror double-fire {dbl} (span {span})")
        if tot:
            print(f"  k={k} TOTAL: {tot} sites, fired {fired} "
                  f"({fired/tot:.4f} vs 1/q' = {1/qp:.4f}); "
                  f"wrong-orientation residue count {wrong} (info only)")
    return found

if __name__ == "__main__":
    ymax = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    for y, qp in STEPS:
        if y <= ymax:
            scan_step(y, qp)
