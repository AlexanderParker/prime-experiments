"""mf_family.py -- the merge forest of 20 counterfactual family members at m13 and m17.

Family member = teeth at +-v_g with v_g uniform in 1..(g-1)/2 (alignment-rules section 5, the
same family as r46/gl_family.py).  The real machine is scored the same way.  The question: does
the forest's shape depend on the teeth at all, and if so where?

Writes results/mf_family.txt.
"""
import os, random
import numpy as np
from mf_core import build_levels, u_of

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def score(gears, vs):
    L = build_levels(gears, vs)
    lv, old = L[-1], L[-2]
    q = lv.q
    bc = np.bincount(lv.order, minlength=8)
    i = int(np.argmax(lv.size))
    t = int(lv.newpos[i])
    J = int(lv.order[i])
    w = [int(x) for x in old.size[(np.arange(t, t + J)) % old.N]]
    thr = (lv.F + 1) // 2
    big = np.flatnonzero(lv.size >= thr)
    t0 = lv.newpos[big]
    e = big + 1
    t1 = lv.newpos[e % lv.N] + (e // lv.N) * q * old.N
    closed = 0
    fr = []
    for a, b in zip(t0.tolist(), t1.tolist()):
        sz = old.size[(np.arange(a, b)) % old.N]
        closed += int((sz >= old.F / 3).all())
        fr.append(sz.max() / old.F)
    return dict(F=lv.F, Fold=old.F, meanorder=float(lv.order.mean()), exact=q / (q - 2),
                maxorder=int(lv.order.max()), ge3=int(bc[3:].sum()),
                n2=int(bc[2]), n3=int(bc[3]), n4=int(bc[4]),
                recJ=J, recpieces=w, recmaxfrac=max(w) / old.F,
                nbig=int(big.size), closed=closed / max(1, big.size),
                meanmaxfrac=float(np.mean(fr)))


def main():
    lines = []
    W = lines.append
    rng = random.Random(20260906)
    for gears in ([5, 7, 11, 13], [5, 7, 11, 13, 17]):
        real = [u_of(g) for g in gears]
        real = [min(v, g - v) for v, g in zip(real, gears)]
        W(f"=== family at m{gears[-1]} (gears {gears}); real teeth v = {real} ===")
        W("member | v | F | mean order | q/(q-2) | max order | n_2 | n_3 | n_4 | record J | "
          "record pieces | max piece/F_old | #big | frac all-pieces-top-third | mean max frac")
        rows = []
        r = score(gears, real)
        rows.append(("REAL", real, r))
        seen = {tuple(real)}
        while len(rows) < 21:
            vs = [rng.randrange(1, (g - 1) // 2 + 1) for g in gears]
            if tuple(vs) in seen:
                continue
            seen.add(tuple(vs))
            rows.append((f"F{len(rows):02d}", vs, score(gears, vs)))
        for name, vs, r in rows:
            W(f"{name} | {vs} | {r['F']} | {r['meanorder']:.6f} | {r['exact']:.6f} | "
              f"{r['maxorder']} | {r['n2']} | {r['n3']} | {r['n4']} | {r['recJ']} | "
              f"{r['recpieces']} | {r['recmaxfrac']:.3f} | {r['nbig']} | {r['closed']:.4f} | "
              f"{r['meanmaxfrac']:.4f}")
        rr = rows[0][2]
        fam = [r for _, _, r in rows[1:]]
        W("")
        W(f"  mean order exact for {sum(1 for _, _, r in rows if abs(r['meanorder']-r['exact'])<1e-12)}"
          f" of {len(rows)} members (identity, teeth-free)")
        for key in ("F", "maxorder", "n3", "recJ", "recmaxfrac", "closed", "meanmaxfrac"):
            vals = sorted(r[key] for r in fam)
            below = sum(1 for v in vals if v < rr[key])
            eq = sum(1 for v in vals if v == rr[key])
            W(f"  {key}: real {rr[key]}, family min {vals[0]}, median {vals[len(vals)//2]}, "
              f"max {vals[-1]}; real percentile {(below + 0.5*eq)/len(vals):.3f}")
        W("")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "mf_family.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
