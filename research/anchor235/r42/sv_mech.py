"""R2.a.i.a.1.b - the mechanism: which gears' coordinates carry the difference.

At a fixed (d, gear set) the per-gear island-strike sets are packed as uint64 bitmasks, once for
the locally-square law and once for the unrestricted law, on the SAME random stream (paired /
common random numbers) so that a single-gear swap is a paired comparison.  Then:

  * HYBRID H(G*)  : square phases at gears <= G*, unrestricted above.  Sweeps G*.
  * SWAP-IN  g    : unrestricted everywhere except gear g, which is square.
  * SWAP-OUT g    : square everywhere except gear g, which is unrestricted.
  * SWAP-IN/OUT >100 : the same with the whole tail g > 100 switched.

Reported: failure rate (no open island in [1,d)) and mean open-island count for each.

Usage: uv run python research/anchor235/r42/sv_mech.py [--N 300000]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

Q0S = [(191, 1000000), (491, 600000), (463, 600000), (1571, 600000)]
SWAPS = [11, 13, 17, 19, 23, 29, 31, 37, 101, 251]
GSTARS = [7, 11, 13, 17, 19, 23, 29, 31, 43, 61, 101, 251, 1000000]


def sieve_np(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def build(q0, gl, ul, n, rng, square):
    """(n, ng) uint64 matrix of per-gear island bitmasks; islands, d, m returned too."""
    d = (2 * pow(6, -1, q0)) % q0
    isl = np.array([i for i in range(1, d) if i % 35 in (5, 10, 12, 17)], dtype=np.int64)
    m = len(isl)
    assert m <= 64, m
    posl = np.full(d, -1, dtype=np.int64)
    posl[isl] = np.arange(m)
    ng = len(gl)
    M = np.zeros((n, ng), dtype=np.uint64)
    rows = np.arange(n)
    one = np.uint64(1)
    for gi in range(ng):
        g, u = int(gl[gi]), int(ul[gi])
        dg = (2 * u) % g
        s = rng.integers(1, g, n)
        r = (s * s) % g if square else s
        b = (-r * u) % g
        a = (b + dg) % g
        col = M[:, gi]
        for base in (a, b):
            k = 0
            while True:
                vals = base + k * g
                sel = vals < d
                if not sel.any():
                    break
                v = vals[sel]
                j = posl[v]
                ok = j >= 0
                if ok.any():
                    ridx = rows[sel][ok]
                    col[ridx] |= (one << j[ok].astype(np.uint64))
                k += 1
        M[:, gi] = col
    return M, d, m


POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def stats(total, m):
    full = np.uint64((1 << m) - 1)
    openmask = full & ~total
    nopen = POP[openmask.view(np.uint8).reshape(-1, 8)].sum(axis=1)
    return float((nopen == 0).mean()), float(nopen.mean())


def orall(M, cols):
    t = np.zeros(M.shape[0], dtype=np.uint64)
    for c in cols:
        t |= M[:, c]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=0)
    a = ap.parse_args()
    fl = sieve_np(6000)
    allp = np.flatnonzero(fl).astype(np.int64)
    lines, out = [], []
    for q0, NQ in Q0S:
        NQ = a.N or NQ
        gl = allp[(allp > 7) & (allp <= q0)]
        ul = np.array([pow(6, -1, int(g)) for g in gl], dtype=np.int64)
        ng = len(gl)
        rngs = np.random.default_rng(424242)
        rngr = np.random.default_rng(424242)     # same stream -> paired
        MS, d, m = build(q0, gl, ul, NQ, rngs, True)
        MR, _, _ = build(q0, gl, ul, NQ, rngr, False)
        hdr = "=== q0 %d  d %d  m %d  gears %d (11..%d) ===" % (q0, d, m, ng, gl[-1])
        lines.append(hdr)
        print(hdr, flush=True)

        def rec(tag, total):
            f, o = stats(total, m)
            lines.append("   %-28s fail %.6f   open mean %8.4f" % (tag, f, o))
            print(lines[-1], flush=True)
            out.append(dict(q0=int(q0), d=int(d), m=int(m), tag=tag, fail=f, open_mean=o, n=NQ))
            return f

        allc = list(range(ng))
        fLS = rec("LS  (square everywhere)", orall(MS, allc))
        fRND = rec("RND (free everywhere)", orall(MR, allc))
        for G in GSTARS:
            cols_s = [c for c in allc if gl[c] <= G]
            cols_r = [c for c in allc if gl[c] > G]
            rec("HYB square<=%d" % min(G, int(gl[-1])), orall(MS, cols_s) | orall(MR, cols_r))
        for g in SWAPS:
            w = np.flatnonzero(gl == g)
            if not len(w):
                continue
            c = int(w[0])
            rec("SWAP-IN  gear %d square" % g, orall(MR, [x for x in allc if x != c]) | MS[:, c])
            rec("SWAP-OUT gear %d free" % g, orall(MS, [x for x in allc if x != c]) | MR[:, c])
        tail = [c for c in allc if gl[c] > 100]
        head = [c for c in allc if gl[c] <= 100]
        if tail:
            rec("SWAP-IN  all g>100 square", orall(MR, head) | orall(MS, tail))
            rec("SWAP-OUT all g>100 free", orall(MS, head) | orall(MR, tail))
        lines.append("   (LS - RND) fail difference: %+.6f  relative %+.3f%%"
                     % (fLS - fRND, 100 * (fLS - fRND) / fRND))
        print(lines[-1], flush=True)
    open(os.path.join(OUT, "sv_mech.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(OUT, "sv_mech.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
