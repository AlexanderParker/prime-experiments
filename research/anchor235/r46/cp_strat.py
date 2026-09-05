"""Branch 2f.i - stratify the FULL tooth-counterfactual family by every candidate
arithmetic property of the separations, and read off the chain-violation rate.

Properties per member (gears G = old gears + q', separations s_q = 2 v_q mod q):
  k        core size = max over admissible rationals (r,c) of #{q : r s_q = +-c mod q}
  I        incompatible pairs = C(n,2) - C(k,2)
  Tmin     min over gears of min(s_q, q - s_q)      ((T) is Tmin >= 2)
  tmin     min over gears of min(s_q, q-s_q)/q      (tooth distance, W3 N-S4; real ~ 1/3)
  ttail    min over TAIL gears q >= 17 of the same
  rmax     min over gear pairs of max(d+, d-)/gh    (N-S3 gauge-free: real >= 1/3?)
  rmin     min over gear pairs of min(d+, d-)/gh
where d+ = folded CRT(s_g, s_h) and d- = folded CRT(s_g, -s_h) modulo gh; {d+, d-} is the
pair of diagonals of the four residues the two gears strike together, and is invariant under
the per-gear sign gauge.

Usage: uv run python research/anchor235/r46/cp_strat.py 19 30
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_compat import (OUT, PROOF, admissible, gears_of, incompat, next_prime,  # noqa: E402
                       real_tooth)


def diag_table(g, h):
    """t[(vg,vh)] = (d+, d-) folded, for the pair (g,h)."""
    gh = g * h
    ig, ih = pow(h, -1, g), pow(g, -1, h)
    t = {}
    for vg in range(1, (g - 1) // 2 + 1):
        sg = (2 * vg) % g
        for vh in range(1, (h - 1) // 2 + 1):
            sh = (2 * vh) % h
            dp = (sg * h * ig + sh * g * ih) % gh
            dm = (sg * h * ig + (-sh % h) * g * ih) % gh
            t[(vg, vh)] = (min(dp, gh - dp), min(dm, gh - dm))
    return t


def main(y=19, B=30):
    import numpy as np
    gears = gears_of(y)
    q1 = next_prime(y)
    gs = gears + [q1]
    n = len(gs)
    rats = admissible(gs, B)
    R = len(rats)
    print("=== m%d + q'=%d : gears %s ; %d admissible rationals at B=%d ===" % (y, q1, gs, R, B))

    coh = []
    for q in gs:
        t = {}
        for v in range(1, (q - 1) // 2 + 1):
            s = (2 * v) % q
            t[v] = np.array([1 if (r * s) % q in (c % q, (-c) % q) else 0
                             for (r, c) in rats], dtype=np.int16)
        coh.append(t)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    dtab = {(i, j): diag_table(gs[i], gs[j]) for (i, j) in pairs}
    ghs = {(i, j): gs[i] * gs[j] for (i, j) in pairs}
    tail = [i for i, q in enumerate(gs[:-1]) if q >= 17]

    real = [real_tooth(q) for q in gs]

    def resonance(teeth, a):
        """number of OLD gears whose separation matches the incoming letter: a = +-2 v_q mod q.
        Stepping by a then maps one tooth of q onto the other."""
        return sum(1 for q, v in zip(gs[:-1], teeth[:-1])
                   if a % q in ((2 * v) % q, (-2 * v) % q))

    def stats(teeth):
        acc = coh[0][teeth[0]].copy()
        for gi in range(1, n):
            acc += coh[gi][teeth[gi]]
        k = int(acc.max())
        s = [(2 * v) % q for q, v in zip(gs, teeth)]
        td = [min(si, q - si) for q, si in zip(gs, s)]
        rmax = 2.0
        rmin = 2.0
        for (i, j) in pairs:
            dp, dm = dtab[(i, j)][(teeth[i], teeth[j])]
            gh = ghs[(i, j)]
            rmax = min(rmax, max(dp, dm) / gh)
            rmin = min(rmin, min(dp, dm) / gh)
        return dict(k=k, I=incompat(n, k), Tmin=min(td),
                    tmin=min(t / q for t, q in zip(td, gs)),
                    ttail=min(td[i] / gs[i] for i in tail),
                    rmax=rmax, rmin=rmin)

    rs = stats(real)
    print("REAL machine: %s" % rs)

    with open(os.path.join(PROOF, "chain_teeth_r33_fam_m%d.json" % y)) as f:
        rows = json.load(f)
    print("%d rows" % len(rows))

    def add(d, key, viol):
        e = d.setdefault(key, [0, 0])
        e[0] += 1
        e[1] += viol

    byk, byT, bytm, bytt, byrx, byrn = {}, {}, {}, {}, {}, {}
    bykT = {}
    byres = {}
    byresk = {}
    worst = {"rmax": (2.0, None), "tmin": (2.0, None), "rmin": (2.0, None)}
    best_v = {"rmax": (-1, None), "tmin": (-1, None), "rmin": (-1, None), "k": (-1, None)}
    nv = 0
    for row in rows:
        teeth = list(row["teeth"]) + [row["v1"]]
        st = stats(teeth)
        v = 1 if row["viol"] else 0
        nv += v
        add(byk, st["k"], v)
        add(byT, min(st["Tmin"], 6), v)
        add(bytm, int(st["tmin"] * 20), v)
        add(bytt, int(st["ttail"] * 20), v)
        add(byrx, int(st["rmax"] * 20), v)
        add(byrn, int(st["rmin"] * 20), v)
        add(bykT, (st["k"], st["Tmin"] >= 2), v)
        res = resonance(teeth, row["a"])
        add(byres, res, v)
        add(byresk, (res, st["k"]), v)
        if v:
            for key in ("rmax", "tmin", "rmin"):
                if st[key] > best_v[key][0]:
                    best_v[key] = (st[key], teeth)
            if st["k"] > best_v["k"][0]:
                best_v["k"] = (st["k"], teeth)
    print("chain violators: %d" % nv)
    print()
    for name, d in (("k (core size)", byk), ("Tmin (min tooth gap)", byT)):
        print("%s: value  members  violators  rate%%" % name)
        for key in sorted(d):
            t, v = d[key]
            print("   %-6s %8d %6d   %.4f" % (key, t, v, 100.0 * v / t))
        print()
    print("k x (T):  (k, T holds)  members  violators  rate%")
    for key in sorted(bykT, key=lambda x: (x[0], x[1])):
        t, v = bykT[key]
        print("   %-12s %8d %6d   %.4f" % (str(key), t, v, 100.0 * v / t))
    print()
    print("resonance  #old gears with a = +-2 v_q mod q")
    for key in sorted(byres):
        t, v = byres[key]
        print("   res=%-2d %8d %6d   %.4f%%" % (key, t, v, 100.0 * v / t))
    print("resonance x core size k (cells with a violator):")
    for key in sorted(byresk):
        t, v = byresk[key]
        if v:
            print("   (res=%d,k=%d) %8d %6d   %.4f%%" % (key[0], key[1], t, v, 100.0 * v / t))
    print()
    for name, d, rv in (("tmin  min_q sep/q", bytm, rs["tmin"]),
                        ("ttail min_{q>=17} sep/q", bytt, rs["ttail"]),
                        ("rmax  min_pairs max(d+,d-)/gh", byrx, rs["rmax"]),
                        ("rmin  min_pairs min(d+,d-)/gh", byrn, rs["rmin"])):
        print("%s   (real machine = %.4f)" % (name, rv))
        for key in sorted(d):
            t, v = d[key]
            print("   [%.2f,%.2f) %8d %6d   %.4f%%" % (key / 20, (key + 1) / 20, t, v,
                                                       100.0 * v / t))
        print()
    print("largest value of each statistic reached BY A VIOLATOR (the threshold test):")
    for key in ("k", "rmax", "rmin", "tmin"):
        print("   %-5s max at a violator = %s   teeth %s" % (key, best_v[key][0], best_v[key][1]))
    with open(os.path.join(OUT, "strat_m%d_B%d.json" % (y, B)), "w") as f:
        json.dump({"real": rs, "nviol": nv, "byk": {str(a): b for a, b in byk.items()},
                   "byT": {str(a): b for a, b in byT.items()},
                   "bykT": {str(a): b for a, b in bykT.items()},
                   "bytm": {str(a): b for a, b in bytm.items()},
                   "bytt": {str(a): b for a, b in bytt.items()},
                   "byrx": {str(a): b for a, b in byrx.items()},
                   "byrn": {str(a): b for a, b in byrn.items()},
                   "byres": {str(a): b for a, b in byres.items()},
                   "byresk": {str(a): b for a, b in byresk.items()},
                   "best_v": {k: [v[0], v[1]] for k, v in best_v.items()}}, f)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 19,
         int(sys.argv[2]) if len(sys.argv) > 2 else 30)
