"""mf_partA.py -- the merge forest at m5..m23 on full periods: identities, orders, the record's
lineage, the top of the spectrum, recruitment.  Writes results/mf_partA.txt.
"""
import os, sys, time
import numpy as np
from mf_core import build_levels, parents, layer_words, PRIMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
L = []


def massrank(lv, s):
    """fraction of gaps of lv strictly smaller than size s."""
    bc = np.bincount(lv.size, minlength=lv.F + 2)
    return float(bc[:s].sum()) / lv.N


def main():
    global L
    lines = []
    W = lines.append
    t0 = time.time()
    L = build_levels()
    W("=== PART A: the merge forest on full periods, m5 .. m23 ===")
    W("")
    W("--- 1. count and length conservation (item 5) ---")
    W("rung | q' | N_old | N_new=(q'-2)N | sum order=q'N | sum(order-1)=2N | mean order | q'/(q'-2) | sum size | P")
    for n in range(1, len(L)):
        lv, old = L[n], L[n - 1]
        q = lv.q
        so = int(lv.order.sum())
        W(f"{old.q}->{q} | {q} | {old.N} | {lv.N} ({'OK' if lv.N == (q-2)*old.N else 'FAIL'}) | "
          f"{so} ({'OK' if so == q*old.N else 'FAIL'}) | {so - lv.N} "
          f"({'OK' if so - lv.N == 2*old.N else 'FAIL'}) | {so/lv.N:.6f} | {q/(q-2):.6f} | "
          f"{int(lv.size.sum())} | {lv.P} ({'OK' if int(lv.size.sum()) == lv.P else 'FAIL'})")
    W("")
    W("--- 2. branching: the order distribution over all gaps born at each rung (item 3) ---")
    W("rung | n_1 (survivals) | n_2 | n_3 | n_4 | n_5 | max | mean | #order>=3 | merge fraction | 2/(q'-2)")
    for n in range(1, len(L)):
        lv, old = L[n], L[n - 1]
        bc = np.bincount(lv.order)
        cnt = [int(bc[j]) if j < bc.size else 0 for j in range(1, 6)]
        ge3 = int(bc[3:].sum()) if bc.size > 3 else 0
        mf = 1.0 - cnt[0] / lv.N
        W(f"{old.q}->{lv.q} | " + " | ".join(str(c) for c in cnt) +
          f" | {int(lv.order.max())} | {lv.order.mean():.6f} | {ge3} | {mf:.6f} | {2/(lv.q-2):.6f}")
    W("")
    W("  check sum_{J>=2} (J-1) n_J = 2 N_old:")
    for n in range(1, len(L)):
        lv, old = L[n], L[n - 1]
        bc = np.bincount(lv.order)
        s = sum((j - 1) * int(bc[j]) for j in range(2, bc.size))
        W(f"    {old.q}->{lv.q}: {s} vs 2N_old = {2*old.N}  {'OK' if s == 2*old.N else 'FAIL'}")
    W("")
    W("--- 3. the record's lineage (item 2) ---")
    for n in range(1, len(L)):
        lv, old = L[n], L[n - 1]
        recs = np.flatnonzero(lv.size == lv.F)
        W(f"rung {lv.q}: F = {lv.F}, {recs.size} record gaps, positions "
          f"{[int(lv.O[i]) for i in recs[:6]]}{' ...' if recs.size > 6 else ''}")
        for i in recs[:4]:
            i = int(i)
            pa = parents(L, n, i)
            szs = [int(old.size[a]) for a in pa]
            bs = [int(old.birth[a]) for a in pa]
            dps = [n - 1 - b for b in bs]
            frac = [s / old.F for s in szs]
            mr = [massrank(old, s) for s in szs]
            W(f"  gap at x={int(lv.O[i])}: order {len(pa)}, depth {n - int(lv.birth[i])}, "
              f"pieces {szs} (sum {sum(szs)})")
            W(f"      piece birth rungs (gear) {[PRIMES[b] for b in bs]}, depths {dps}, "
              f"size/F_old {[f'{f:.3f}' for f in frac]}")
            W(f"      mass rank of each piece among gaps of M_old: {[f'{r:.6f}' for r in mr]}")
            lw = layer_words(L, n, i)
            for k in sorted(lw, reverse=True):
                W(f"      layer {L[k].q}: k={len(lw[k][1])} word {lw[k][1]}")
            break
    W("")
    W("--- 4. the top of the spectrum: profiles, closure, recruitment (item 4) ---")
    W("rung | #gaps >= F/2 | #distinct order profiles | #distinct layer words | max order there | "
      "frac all pieces >= F_old/3 | min piece frac | mean piece frac")
    for n in range(2, len(L)):
        lv, old = L[n], L[n - 1]
        thr = (lv.F + 1) // 2
        big = np.flatnonzero(lv.size >= thr)
        prof = np.stack([lv.anc[k][big] for k in sorted(lv.anc)], axis=1)
        up = np.unique(prof, axis=0)
        # pieces of each big gap
        t0i = lv.newpos[big]
        t1i = np.where(big + 1 < lv.N, lv.newpos[np.minimum(big + 1, lv.N - 1)],
                       lv.newpos[0] + lv.q * old.N)
        allmin, allmax, tot, cnt, closed = [], [], 0, 0, 0
        pfracs = []
        for a, b in zip(t0i.tolist(), t1i.tolist()):
            sz = old.size[np.arange(a, b) % old.N]
            pfracs.append(sz / old.F)
            closed += int((sz >= old.F / 3).all())
            cnt += 1
        pf = np.concatenate(pfracs) if pfracs else np.array([0.0])
        # distinct layer words: hash the full word tuple
        W(f"{lv.q} | {big.size} | {up.shape[0]} | (see below) | {int(lv.order[big].max())} | "
          f"{closed/cnt:.4f} | {pf.min():.4f} | {pf.mean():.4f}")
    W("")
    W("--- 5. forest node counts (item 5) ---")
    top = L[-1]
    W(f"per period of {{5..{top.q}}} (P = {top.P}):")
    tot = 0
    for k, lv in enumerate(L):
        c = top.P // lv.P * lv.N
        tot += c
        W(f"  layer {lv.q}: {c} nodes ({'leaves' if k == 0 else 'internal'})")
    W(f"  total nodes {tot}; leaves {top.P // 5 * 3}; roots {top.N}; "
      f"leaves/roots {(top.P // 5 * 3)/top.N:.4f}")
    W("")
    W(f"[{time.time()-t0:.1f}s]")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "mf_partA.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
