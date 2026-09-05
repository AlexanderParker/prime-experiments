"""mf_lineage.py -- the top rungs read as a forest: the m29 and m31 records' full lineages, the
order profiles and recruitment at m29 and m31, and the per-rung record summary completed.
Writes results/mf_lineage.txt.
"""
import os, json, time
import numpy as np
from mf_core import build_levels, u_of, PRIMES
from mf_top import Base

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def word_of(lv, s, c):
    return lv.size[(np.arange(s, s + c)) % lv.N]


def orders_of(lv, s, c):
    return lv.order[(np.arange(s, s + c)) % lv.N]


def births_of(lv, s, c):
    return lv.birth[(np.arange(s, s + c)) % lv.N]


def descend(L, k, start, count):
    """the layer words of a contiguous run of `count` gaps of level k starting at index `start`."""
    out = {k: [int(x) for x in word_of(L[k], start, count)]}
    while k > 0:
        lv, old = L[k], L[k - 1]
        s = start % lv.N
        t0 = int(lv.newpos[s])
        e = s + count
        t1 = int(lv.newpos[e % lv.N]) + (e // lv.N) * lv.q * old.N
        start, count = t0 % old.N, t1 - t0
        k -= 1
        out[k] = [int(x) for x in word_of(L[k], start, count)]
    return out


def main():
    lines = []
    W = lines.append
    L = build_levels()
    base = Base(L)
    N23 = base.N
    lv23 = L[6]
    big29 = np.load(os.path.join(OUT, "big29.npy"))
    spec29 = np.load(os.path.join(OUT, "spec29.npy"))
    spec31 = np.load(os.path.join(OUT, "spec31.npy"))
    N29 = int(spec29.sum())
    cum23 = np.concatenate([[0], np.cumsum(np.bincount(lv23.size, minlength=64))]) / lv23.N
    cum29 = np.concatenate([[0], np.cumsum(spec29)]) / N29
    F23, F29, F31 = 34, 43, 58

    W("=== the top rungs as a forest ===")
    W("")
    W("--- m29 (rung 23 -> 29), gaps of size >= 22 = F/2 ---")
    t0 = big29[:, 0]
    od = big29[:, 1]
    sz = big29[:, 2]
    W(f"  {big29.shape[0]} gaps; order histogram {np.bincount(od).tolist()}")
    prof = np.stack([base.anc_count(k, t0, t0 + od) for k in range(7)] + [od], axis=1)
    up = np.unique(prof, axis=0)
    W(f"  distinct order profiles (k_5..k_23, J): {up.shape[0]}")
    # pieces
    pieces, pmass, closed, low, mxdep0, allpc = [], [], 0, 0, 0, []
    maxfr, minfr = [], []
    for i in range(big29.shape[0]):
        a, J = int(t0[i]), int(od[i])
        w = lv23.size[(np.arange(a, a + J)) % N23]
        b = lv23.birth[(np.arange(a, a + J)) % N23]
        o = lv23.order[(np.arange(a, a + J)) % N23]
        allpc.append(w)
        closed += int((w >= F23 / 3).all())
        low += int((w < F23 / 4).any())
        k = int(np.argmax(w))
        mxdep0 += int(o[k] > 1)
        maxfr.append(w.max() / F23)
        minfr.append(w.min() / F23)
    allpc = np.concatenate(allpc)
    W(f"  piece sizes: mean {allpc.mean():.3f}, max {allpc.max()}, mean size/F23 {allpc.mean()/F23:.4f}")
    W(f"  mean piece mass rank in m23 {cum23[allpc].mean():.6f}, min {cum23[allpc].min():.6f}")
    W(f"  all pieces >= F23/3: {closed/big29.shape[0]:.4f}; some piece < F23/4: {low/big29.shape[0]:.4f}; "
      f"largest piece born at rung 23 (depth 0): {mxdep0/big29.shape[0]:.4f}")
    W(f"  max piece fraction: mean {np.mean(maxfr):.4f}, max {max(maxfr):.4f}; "
      f"min piece fraction: mean {np.mean(minfr):.4f}")
    W("")
    W("--- the m29 record ---")
    ri = np.flatnonzero(sz == F29)
    for i in ri.tolist():
        a, J = int(t0[i]), int(od[i])
        j, ii = a // N23, a % N23
        x = int(lv23.O[ii]) + j * lv23.P
        w = [int(v) for v in lv23.size[(np.arange(a, a + J)) % N23]]
        b = [int(v) for v in lv23.birth[(np.arange(a, a + J)) % N23]]
        W(f"  x = {x} (copy {j} of the m23 period), order {J}, pieces {w}")
        W(f"      piece birth gears {[PRIMES[v] for v in b]}, depths {[6 - v for v in b]}, "
          f"size/F23 {[f'{v/F23:.3f}' for v in w]}, mass rank {[f'{cum23[v]:.6f}' for v in w]}")
        d = descend(L, 6, ii, J)
        for k in sorted(d, reverse=True):
            W(f"      layer {L[k].q}: k={len(d[k])} word {d[k]}")
    W("")
    W("--- the m31 record (rung 29 -> 31) ---")
    top = json.load(open(os.path.join(OUT, "m31_top.json")))["top"]
    for r in top:
        size, tstart, J, w, pcs, ts, pos29 = r
        if size != F31:
            continue
        W(f"  m29 position {pos29}, order {J}, pieces (m29 gaps) {pcs}, "
          f"size/F29 {[f'{v/F29:.3f}' for v in pcs]}, mass rank in m29 {[f'{cum29[v]:.8f}' for v in pcs]}")
        pb, po = [], []
        for k in range(J):
            a, bnd = ts[k], ts[k + 1]
            o29 = bnd - a
            po.append(o29)
            pb.append(0 if o29 > 1 else 6 - int(lv23.birth[a % N23]))
        W(f"      piece orders at rung 29 {po}, piece depths {pb} "
          f"(0 = born at rung 29)")
        # full descent: the m23 layer word of the whole gap
        a0, a1 = ts[0], ts[-1]
        d = descend(L, 6, a0 % N23, a1 - a0)
        W(f"      layer 29: k={J} word {pcs}")
        for k in sorted(d, reverse=True):
            W(f"      layer {L[k].q}: k={len(d[k])} word {d[k]}")
        W("")
    W("--- record summary, all rungs (item 2 completed) ---")
    W("rung | F | order | pieces | max piece/F_old | min piece/F_old | mean piece depth | "
      "max-piece depth | mass rank of max piece")
    rows = []
    for n in range(1, 7):
        lvn, old = L[n], L[n - 1]
        i = int(np.argmax(lvn.size))
        t = int(lvn.newpos[i])
        J = int(lvn.order[i])
        w = [int(v) for v in old.size[(np.arange(t, t + J)) % old.N]]
        b = [int(v) for v in old.birth[(np.arange(t, t + J)) % old.N]]
        cum = np.concatenate([[0], np.cumsum(np.bincount(old.size, minlength=64))]) / old.N
        mx = max(w)
        rows.append((lvn.q, lvn.F, J, w, mx / old.F, min(w) / old.F,
                     np.mean([n - 1 - v for v in b]), n - 1 - b[w.index(mx)], cum[mx]))
    # m29
    i = int(ri[0])
    a, J = int(t0[i]), int(od[i])
    w = [int(v) for v in lv23.size[(np.arange(a, a + J)) % N23]]
    b = [int(v) for v in lv23.birth[(np.arange(a, a + J)) % N23]]
    o = [int(v) for v in lv23.order[(np.arange(a, a + J)) % N23]]
    mx = max(w)
    rows.append((29, 43, J, w, mx / F23, min(w) / F23, np.mean([6 - v for v in b]),
                 6 - b[w.index(mx)], cum23[mx]))
    # m31
    for r in top:
        if r[0] != F31:
            continue
        size, tstart, J, wgt, pcs, ts, pos29 = r
        po = [ts[k + 1] - ts[k] for k in range(J)]
        pb = [0 if po[k] > 1 else 6 - int(lv23.birth[ts[k] % N23]) for k in range(J)]
        mx = max(pcs)
        rows.append((31, 58, J, pcs, mx / F29, min(pcs) / F29, np.mean(pb),
                     pb[pcs.index(mx)], cum29[mx]))
        break
    for r in rows:
        W(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.2f} | {r[7]} | {r[8]:.6f}")
    W("")
    W("--- m31 big-gap piece mass ranks (from the weighted piece histogram) ---")
    txt = open(os.path.join(OUT, "mf_top.txt")).read()
    for ln in txt.splitlines():
        if ln.strip().startswith("piece size histogram"):
            hist = {}
            for tok in ln.split(")")[-1].split():
                if ":" in tok:
                    a_, b_ = tok.split(":")
                    hist[int(a_)] = int(b_)
            tot = sum(hist.values())
            mean_mr = sum(c * cum29[v] for v, c in hist.items()) / tot
            mean_fr = sum(c * v for v, c in hist.items()) / tot / F29
            W(f"  {tot} pieces; mean size/F29 {mean_fr:.4f}; mean mass rank in m29 {mean_mr:.6f}; "
              f"min size {min(hist)}, max size {max(hist)}")
    txt2 = "\n".join(lines)
    open(os.path.join(OUT, "mf_lineage.txt"), "w").write(txt2)
    print(txt2)


if __name__ == "__main__":
    main()
