"""xm_family.py -- the counter-construction: on the tooth family of the engine {5..p} (same gears,
teeth +-v_g with 1 <= v_g <= (g-1)/2, every combination; the real engine is the member with
v_g = min(u_g, g - u_g)), which members strike EVERY column of the section at the cut p?

Section at the cut p: columns a+1 .. b-1, a = (p^2-1)/6, b = (p'^2-1)/6 (first_realisation.md
0.1).  A member kills the section iff the union of its gears' strikes covers all l_p - 1 columns;
for the real member this is the negation of step 8 at the cut.  Exhaustive over the family
(sizes 2*3 = 6 at p = 7 ... 29,937,600 at p = 31); the struck set of the section is held as two
64-bit words per member.  Reports the killer count and share, and exhibits the first killer's
teeth at the smallest cut with a killer.
"""
import json
import os
import sys
import time

import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
NEXT = {PRIMES[i]: PRIMES[i + 1] for i in range(len(PRIMES) - 1)}


def gears_of(p):
    return [g for g in PRIMES if g <= p]


def u_of(g):
    return pow(6, -1, g)


def gear_masks(g, cols):
    """(nv, 2) uint64: for each tooth v = 1..(g-1)/2 the bitmask (two words) of the columns in
    `cols` struck by the gear with teeth +-v."""
    nv = (g - 1) // 2
    out = np.zeros((nv, 2), dtype=np.uint64)
    for v in range(1, nv + 1):
        m0 = 0
        m1 = 0
        for i, k in enumerate(cols):
            if (k % g) in (v, g - v):
                if i < 64:
                    m0 |= 1 << i
                else:
                    m1 |= 1 << (i - 64)
        out[v - 1, 0] = np.uint64(m0)
        out[v - 1, 1] = np.uint64(m1)
    return out


def run_cut(p):
    pn = NEXT[p]
    a = (p * p - 1) // 6
    b = (pn * pn - 1) // 6
    cols = list(range(a + 1, b))
    n = len(cols)
    assert n <= 128
    full0 = np.uint64((1 << min(n, 64)) - 1)
    full1 = np.uint64((1 << max(n - 64, 0)) - 1)
    gears = gears_of(p)
    masks = np.zeros((1, 2), dtype=np.uint64)
    sizes = []
    for g in gears:
        gm = gear_masks(g, cols)
        sizes.append(gm.shape[0])
        masks = (masks[:, None, :] | gm[None, :, :]).reshape(-1, 2)
    killers = np.flatnonzero((masks[:, 0] == full0) & (masks[:, 1] == full1))
    total = masks.shape[0]
    # the real member's index (mixed radix, first gear most significant)
    idx = 0
    real = []
    for g, nv in zip(gears, sizes):
        u = u_of(g)
        v = min(u % g, (-u) % g)
        real.append(v)
        idx = idx * nv + (v - 1)
    real_kills = bool((masks[idx, 0] == full0) and (masks[idx, 1] == full1))

    def decode(i):
        vs = []
        for nv in reversed(sizes):
            vs.append(int(i % nv) + 1)
            i //= nv
        return list(reversed(vs))

    first = decode(int(killers[0])) if killers.size else None
    return {"p": p, "p_next": pn, "a": a, "b": b, "section_columns": n, "gears": gears,
            "family": int(total), "killers": int(killers.size),
            "share": float(killers.size / total), "real_teeth": real,
            "real_kills": real_kills, "first_killer": first,
            "first_killer_index": int(killers[0]) if killers.size else None}


def main():
    ps = [int(v) for v in sys.argv[1:]] or [7, 11, 13, 17, 19, 23, 29, 31]
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for p in ps:
        t0 = time.time()
        r = run_cut(p)
        r["secs"] = time.time() - t0
        out.append(r)
        print(f"p = {p:2d} -> {r['p_next']:2d}: section {r['section_columns']:3d} columns "
              f"({r['a'] + 1}..{r['b'] - 1}); family {r['family']:>10,d}; killers {r['killers']:>8,d} "
              f"({100 * r['share']:.3f} %); real teeth {r['real_teeth']} kills: {r['real_kills']}; "
              f"first killer teeth {r['first_killer']}; {r['secs']:.1f}s", flush=True)
    with open(os.path.join(here, "results", "xm_family.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
