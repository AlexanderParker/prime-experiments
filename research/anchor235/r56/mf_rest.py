"""mf_rest.py -- THE REST STATISTIC and the (largest piece, rest) frontier.

The forest writes every gap as  size = largest piece + rest,  rest = the sum of the other pieces.
Since the largest piece is a gap of the machine below, size <= F(M) + rest, so

    rest(G) <= q'  for every gap G of M + q'     ==>    the budget inequality F(M+q') <= F(M)+q'.

This script measures max rest at every rung and the whole frontier  a -> max{rest : largest
piece = a}, which is where the budget inequality actually lives.  Writes results/mf_rest.txt.
"""
import os
import numpy as np
from mf_core import build_levels
from mf_top import Base

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    lines = []
    W = lines.append
    L = build_levels()
    W("=== the rest statistic:  size = largest piece + rest ===")
    W("")
    W("rung | q' | F | max rest | rest <= q'? | argmax (largest piece, rest, size) | "
      "rest at largest piece = F_old | rest of the record")
    for n in range(1, len(L)):
        lv, old = L[n], L[n - 1]
        q = lv.q
        t0 = lv.newpos
        t1 = np.empty(lv.N, dtype=np.int64)
        t1[:-1] = lv.newpos[1:]
        t1[-1] = lv.newpos[0] + q * old.N
        mx = np.zeros(lv.N, dtype=np.int64)
        Jm = int(lv.order.max())
        for k in range(Jm):
            act = lv.order > k
            idx = np.flatnonzero(act)
            psz = old.size[(t0[idx] + k) % old.N]
            mx[idx] = np.maximum(mx[idx], psz)
        rest = lv.size - mx
        j = int(np.argmax(rest))
        atF = rest[mx == old.F]
        ri = int(np.argmax(lv.size))
        W(f"{old.q}->{q} | {q} | {lv.F} | {int(rest.max())} | "
          f"{'yes' if rest.max() <= q else 'NO'} | ({int(mx[j])}, {int(rest[j])}, {int(lv.size[j])}) | "
          f"{int(atF.max()) if atF.size else '-'} | {int(rest[ri])}")
        # frontier
        fr = {}
        for a, r in zip(mx.tolist(), rest.tolist()):
            if r > fr.get(a, -1):
                fr[a] = r
        W("    frontier a->maxrest(sum): " +
          " ".join(f"{a}->{fr[a]}({a+fr[a]})" for a in sorted(fr)))
    W("")
    W("--- rung 23 -> 29 (gaps of size >= 22; the frontier is exact wherever the sum is >= 22) ---")
    base = Base(L)
    lv23 = L[6]
    big29 = np.load(os.path.join(OUT, "big29.npy"))
    t0, od, sz = big29[:, 0], big29[:, 1], big29[:, 2]
    mx = np.zeros(big29.shape[0], dtype=np.int64)
    for k in range(int(od.max())):
        act = od > k
        idx = np.flatnonzero(act)
        mx[idx] = np.maximum(mx[idx], lv23.size[(t0[idx] + k) % lv23.N])
    rest = sz - mx
    fr = {}
    for a, r in zip(mx.tolist(), rest.tolist()):
        if r > fr.get(a, -1):
            fr[a] = r
    j = int(np.argmax(rest))
    atF = rest[mx == 34]
    W(f"23->29 | 29 | 43 | max rest {int(rest.max())} | {'yes' if rest.max() <= 29 else 'NO'} | "
      f"argmax ({int(mx[j])}, {int(rest[j])}, {int(sz[j])}) | rest at largest piece = 34: "
      f"{int(atF.max()) if atF.size else '-'} | rest of the record {int(rest[sz == 43][0])}")
    W("    frontier a->maxrest(sum): " +
      " ".join(f"{a}->{fr[a]}({a+fr[a]})" for a in sorted(fr)))
    W("")
    W("--- rung 29 -> 31 (from mf_top.txt, gaps of size >= 29) ---")
    for ln in open(os.path.join(OUT, "mf_top.txt")).read().splitlines():
        if "frontier (largest piece" in ln or "REST histogram" in ln:
            W("  " + ln.strip())
    W("")
    W("--- the trade-off, stated ---")
    W("At every rung the sum (largest piece + rest) peaks at a MIDDLING largest piece and collapses")
    W("at largest piece = F_old: a gap that swallows the old record whole cannot be fused with much")
    W("else.  That trade-off, not the branching bound, is what keeps F growing slowly.")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "mf_rest.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
