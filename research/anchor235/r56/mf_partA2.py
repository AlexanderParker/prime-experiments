"""mf_partA2.py -- part A continued: every record class's lineage, lineage signatures, the
recruitment profile, and the closure test at every rung m7..m23.  Writes results/mf_partA2.txt.
"""
import os, time
import numpy as np
from mf_core import build_levels, PRIMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def start_chain(L, n, idx):
    """For gaps idx at level n, return dict k -> (start index at level k, count) arrays.
    Ancestors at every layer form a CONTIGUOUS run of that layer's gaps (mod N_k), because the
    merge law merges consecutive gaps; so a lineage is a start plus a count at each layer."""
    out = {n: (idx.copy(), np.ones(idx.size, dtype=np.int64))}
    cur = idx
    for k in range(n, 0, -1):
        lv, old = L[k], L[k - 1]
        st = lv.newpos[cur] % old.N
        cnt = out[k][1] * 0
        # count at level k-1 = number of level-(k-1) ancestors
        cnt = lv.anc[k - 1][cur] if (k - 1) in lv.anc else lv.order[cur]
        out[k - 1] = (st.astype(np.int64), cnt.astype(np.int64))
        cur = st
    return out


def word_of(lv, s, c):
    return lv.size[(np.arange(s, s + c)) % lv.N]


def sig_of(L, n, chain, j):
    parts = []
    for k in range(n, -1, -1):
        s, c = int(chain[k][0][j]), int(chain[k][1][j])
        parts.append(tuple(int(x) for x in word_of(L[k], s, c)))
    return tuple(parts)


def main():
    lines = []
    W = lines.append
    t0 = time.time()
    L = build_levels()
    W("=== PART A2: record classes, lineage signatures, recruitment ===")
    W("")
    W("--- every record gap's immediate decomposition (item 2) ---")
    W("rung | x | order | pieces | birth gears | depths | size/F_old | mass rank | max frac")
    for n in range(1, len(L)):
        lv, old = L[n], L[n - 1]
        bc = np.bincount(old.size, minlength=old.F + 2)
        cum = np.concatenate([[0], np.cumsum(bc)]).astype(np.float64) / old.N
        recs = np.flatnonzero(lv.size == lv.F)
        ch = start_chain(L, n, recs)
        seen = {}
        for j, i in enumerate(recs.tolist()):
            s, c = int(ch[n - 1][0][j]), int(ch[n - 1][1][j])
            sz = word_of(old, s, c)
            key = tuple(int(x) for x in sz)
            if key in seen:
                seen[key] += 1
                continue
            seen[key] = 1
            b = old.birth[(np.arange(s, s + c)) % old.N]
            W(f"{lv.q} | {int(lv.O[i])} | {c} | {list(key)} | "
              f"{[PRIMES[int(x)] for x in b]} | {[n-1-int(x) for x in b]} | "
              f"{[f'{x/old.F:.3f}' for x in key]} | {[f'{cum[x]:.6f}' for x in key]} | "
              f"{max(key)/old.F:.3f}")
        W(f"   ({lv.q}: {recs.size} record gaps in {len(seen)} distinct parent words)")
    W("")
    W("--- record summary per rung (item 2, item 3) ---")
    W("rung | F | F/F_old | order | max piece | max piece/F_old | min piece/F_old | mean piece depth"
      " | max-piece depth | J_max(rung) | J_max x maxfrac")
    for n in range(1, len(L)):
        lv, old = L[n], L[n - 1]
        i = int(np.argmax(lv.size))
        ch = start_chain(L, n, np.array([i]))
        s, c = int(ch[n - 1][0][0]), int(ch[n - 1][1][0])
        sz = [int(x) for x in word_of(old, s, c)]
        b = old.birth[(np.arange(s, s + c)) % old.N]
        dep = [n - 1 - int(x) for x in b]
        mx = max(sz)
        mdep = dep[sz.index(mx)]
        Jm = int(lv.order.max())
        W(f"{lv.q} | {lv.F} | {lv.F/old.F:.3f} | {c} | {mx} | {mx/old.F:.3f} | "
          f"{min(sz)/old.F:.3f} | {np.mean(dep):.2f} | {mdep} | {Jm} | {Jm*mx/old.F:.3f}")
    W("")
    W("--- lineage signatures at the top of the spectrum (item 4) ---")
    W("rung | thr F/2 | #gaps | #distinct full lineages | #distinct parent words | "
      "#distinct order profiles | record lineage multiplicity")
    for n in range(2, len(L)):
        lv, old = L[n], L[n - 1]
        thr = (lv.F + 1) // 2
        big = np.flatnonzero(lv.size >= thr)
        ch = start_chain(L, n, big)
        sigs = {}
        pw = set()
        for j in range(big.size):
            sg = sig_of(L, n, ch, j)
            sigs[sg] = sigs.get(sg, 0) + 1
            pw.add(sg[1])
        prof = np.stack([lv.anc[k][big] for k in sorted(lv.anc)], axis=1)
        nprof = np.unique(prof, axis=0).shape[0]
        ri = int(np.argmax(lv.size))
        chr_ = start_chain(L, n, np.array([ri]))
        rsig = sig_of(L, n, chr_, 0)
        W(f"{lv.q} | {thr} | {big.size} | {len(sigs)} | {len(pw)} | {nprof} | "
          f"{sigs.get(rsig, 0)} of {big.size}")
    W("")
    W("--- recruitment: where the pieces of a big gap come from (item 4) ---")
    W("rung | mean piece mass-rank | min piece mass-rank | frac pieces in old top third by size |"
      " frac gaps all-pieces-top-third | frac gaps with a piece below F_old/4")
    for n in range(2, len(L)):
        lv, old = L[n], L[n - 1]
        bc = np.bincount(old.size, minlength=old.F + 2)
        cum = np.concatenate([[0], np.cumsum(bc)]).astype(np.float64) / old.N
        thr = (lv.F + 1) // 2
        big = np.flatnonzero(lv.size >= thr)
        ch = start_chain(L, n, big)
        s0, c0 = ch[n - 1]
        allmr, allsz, closed, low = [], [], 0, 0
        for j in range(big.size):
            sz = word_of(old, int(s0[j]), int(c0[j]))
            allsz.append(sz)
            allmr.append(cum[sz])
            closed += int((sz >= old.F / 3).all())
            low += int((sz < old.F / 4).any())
        mr = np.concatenate(allmr)
        sz = np.concatenate(allsz)
        W(f"{lv.q} | {mr.mean():.6f} | {mr.min():.6f} | {(sz >= old.F/3).mean():.4f} | "
          f"{closed/big.size:.4f} | {low/big.size:.4f}")
    W("")
    W(f"[{time.time()-t0:.1f}s]")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "mf_partA2.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
