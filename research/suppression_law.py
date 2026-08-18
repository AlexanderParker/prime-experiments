"""Round 19b: THE SUPPRESSION LAW - turning "luck" into a predictive relation.

If qualifying is independent of window sum (r19a: luck-plausible), then the
qualifying maximum is the maximum over a p-fraction of windows, and for an
exponential-tailed sum distribution P(S > s) ~ exp(-s/lambda):

    suppression  :=  F_j - qualmax_j  ~  lambda_j * ln(1/p_j).

This is a CONSTRUCT, not a fit: lambda from the window-sum tail, p from the
qualifying rate, both computed from M alone - no reference to the merge. If it
holds, the merged maximum is PREDICTED by M's gap distribution:

    merged_max(j)  ~  F_j - lambda_j ln(1/p_j),

and (D) becomes a statement about lambda and p rather than about extremes:
    (D) <==  F_j - lambda_j ln(1/p_j)  <=  F + q'   for every depth j.
That is checkable at every depth INCLUDING the deep ones where plain flatness
failed - the deeper the window, the smaller p, the bigger the suppression.
"""
import numpy as np
import sys
sys.path.insert(0, "research")
from fuel_bound import gapword
from word_ceiling import FK

CH = 30_000_000
CASES = [(19, 23), (23, 29), (29, 31)]


def run(y, q1, depths=(3, 4, 5, 6)):
    g = gapword(y).astype(np.int16)
    n = len(g)
    c = pow(6, -1, q1)
    Q = np.array(sorted({0, (2 * c) % q1, (-2 * c) % q1}))
    F = FK[y]
    print(f"\n=== machine {y}  q'={q1}  F={F}  budget F+q'={F+q1}")
    print("  j   F_j  qualmax  suppr  p_qual        lambda  pred_suppr  "
          "pred_merged  need<=  ok")
    for j in depths:
        gmax = qmax = 0
        nq = ntot = 0
        hist = {}
        for lo in range(0, n - j, CH):
            hi = min(lo + CH, n - j)
            idx = np.arange(lo, hi)
            s = np.zeros(len(idx), np.int32)
            for t in range(j):
                s += g[idx + t]
            ok = np.ones(len(idx), bool)
            for t in range(1, j - 1):
                ok &= np.isin(g[idx + t] % q1, Q)
            ntot += len(idx)
            nq += int(ok.sum())
            gmax = max(gmax, int(s.max()))
            if ok.any():
                qmax = max(qmax, int(s[ok].max()))
            u, cts = np.unique(s, return_counts=True)
            for a, b in zip(u.tolist(), cts.tolist()):
                hist[a] = hist.get(a, 0) + b
        # lambda from the upper tail: log-count slope over the top decade
        vals = np.array(sorted(hist))
        cnts = np.array([hist[v] for v in vals], float)
        tail = np.cumsum(cnts[::-1])[::-1]          # #windows with sum >= v
        m = (tail >= 5) & (tail <= ntot * 1e-3)
        if m.sum() >= 3:
            lam = -1.0 / np.polyfit(vals[m], np.log(tail[m]), 1)[0]
        else:
            lam = float("nan")
        p = nq / ntot
        pred_s = lam * np.log(1 / p) if p > 0 else float("inf")
        pred_m = gmax - pred_s
        ok_s = "yes" if (p > 0 and pred_m <= F + q1) else ("n/a" if p == 0 else "NO")
        print(f"  {j}  {gmax:4d}  {qmax:6d}  {gmax-qmax:5d}  {p:.3e}  "
              f"{lam:6.2f}  {pred_s:10.1f}  {pred_m:11.1f}  {F+q1:6d}  {ok_s}")


if __name__ == "__main__":
    for y, q1 in CASES:
        run(y, q1)
