"""Round 12 lateral, part 2: (a) EXACTNESS of the firing law, checked against
the actual kills of gear q'; (b) the REALIZED merge spectrum by k, and the
graded constant recomputed on realized rather than potential chains.

Firing law (derived): consecutive kills of q' inside a chain sit at the two
teeth {u, -u} alternately, so a kill at u is followed by a step of
(-u - u) = -2u = q'-s and a kill at -u by a step of +2u = s (s = 2u mod q').
The spacing word's FIRST entry therefore fixes the whole orientation:

    word starts with s      ->  chain starts at residue -u   (one residue)
    word starts with q'-s   ->  chain starts at residue +u   (one residue)

so a fuel site fires iff p lies in ONE residue class mod q': ensemble firing
fraction 1/q', not 2/q'.

Checks below: every site's actual kill-set recomputed from gear q' directly
(predicted-fired == actually-fired, elementwise); residue uniformity of the
site population; realized merge sizes by k vs the F_j spectrum.
"""
import sys
from collections import Counter, defaultdict
from math import prod

import numpy as np

from split_gap_law import primes
from topgap_corridor import chunk_openings

def analyse(y, qp, chunk=20_000_000):
    gears = primes(5, y)
    P = prod(gears)
    u = pow(6, -1, qp)
    s = (2 * u) % qp
    teeth = {u, (qp - u) % qp}
    words = {}
    for k in (3, 4, 5):
        words[(k, 'start_s')] = (tuple((s if i % 2 == 0 else qp - s)
                                       for i in range(k - 1)), (qp - u) % qp)
        words[(k, 'start_sb')] = (tuple((qp - s if i % 2 == 0 else s)
                                        for i in range(k - 1)), u)
    sites = defaultdict(list)
    resid = defaultdict(Counter)
    merges = defaultdict(list)     # k -> realized new-gap sizes
    Fnew = 0
    Fold = 0
    carry = None
    a = 0
    while a < P:
        S = min(chunk, P - a)
        ops = chunk_openings(gears, a, S)
        ext = ops if carry is None else np.concatenate((carry, ops))
        d = np.diff(ext)
        if len(d):
            Fold = max(Fold, int(d.max()))
        keep = ~np.isin(ext % qp, list(teeth))      # survivors in M_{q'}
        surv = ext[keep]
        if len(surv) > 1:
            Fnew = max(Fnew, int(np.diff(surv).max()))
        for (k, tag), (w, fire) in words.items():
            n = k - 1
            if len(d) < n:
                continue
            m = d[:len(d) - n + 1] == w[0]
            for j in range(1, n):
                m &= d[j:len(d) - n + 1 + j] == w[j]
            for i in np.flatnonzero(m):
                p = int(ext[i])
                sites[(k, tag)].append(p)
                resid[(k, tag)][p % qp] += 1
                chain = ext[i:i + k]
                killed = np.isin(chain % qp, list(teeth))
                if killed.all():                    # a real k-chain
                    lo = ext[:i + 1][keep[:i + 1]]
                    hi = ext[i + k:][keep[i + k:]]
                    if len(lo) and len(hi):
                        merges[k].append(int(hi[0] - lo[-1]))
                    assert p % qp == fire, (
                        f"LAW VIOLATION at {p}: fired but residue "
                        f"{p % qp} != {fire}")
                elif killed.any() and p % qp == fire:
                    raise AssertionError(f"predicted fire but partial at {p}")
        carry = ext[-8:]
        a += S
    print(f"STEP {y}->{qp}: period {P}, u={u}, s={s}, F_old={Fold}, F_new={Fnew}")
    for key in sorted(sites):
        k, tag = key
        w, fire = words[key]
        ps = sorted(set(sites[key]))
        if not ps:
            continue
        fired = [p for p in ps if p % qp == fire]
        exp = len(ps) / qp
        rc = resid[key]
        spread = f"min {min(rc.values())} max {max(rc.values())}" if len(rc) > 3 else str(dict(rc))
        print(f"  k={k} word {w} fire-residue {fire}: sites {len(ps)}, "
              f"FIRED {len(fired)} (expected {exp:.2f} at 1/q'); "
              f"site-residue occupancy: {spread}")
    for k in sorted(merges):
        v = merges[k]
        print(f"  realized k={k} chains: {len(v)}, merge sizes max {max(v)}, "
              f"mean {sum(v)/len(v):.1f}; increment (max merge - F_old)/q' = "
              f"{(max(v)-Fold)/qp:+.3f}")
    print(f"  ACTUAL increment (F_new - F_old)/q' = {(Fnew-Fold)/qp:.3f}")
    return Fold, Fnew

if __name__ == "__main__":
    for y, qp in [(19, 23), (29, 31)]:
        if y <= (int(sys.argv[1]) if len(sys.argv) > 1 else 99):
            analyse(y, qp)
