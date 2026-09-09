"""The top-run share tau in valve_existence.md table 2's own sense, at every prime cut to PMAX.

At the cut p (the step q -> p, q = prevprime(p)) the top column of the prefix of {5..q} is
W = (p^2 - 1)/6.  Under {5..q} that column is OPEN when p^2 - 2 is prime (its other member p^2 is
struck only by p, which is not a gear of {5..q}) - reduction (R)'s "plus W(q) when q'^2 - 2 is
prime".  So the run of {5..q} ending at the top of the prefix has length

    L_top = 0                    if p^2 - 2 is prime,
    L_top = W - k0               otherwise (k0 = the last twin column below W),

and tau = L_top / (W - q//6), the share of the prefix's columns (those with 6k - 1 > q) it takes.
The straddling run of {5..p} is unaffected: under {5..p} the column W is always struck by p.

Usage: uv run python research/stack/r6/tau_prefix.py results/straddle_<tag>.npz
"""
import os
import sys

import gmpy2
import numpy as np


def main(path):
    A = np.load(path)["rows"]
    A = A[np.argsort(A[:, 0])]
    p, W, k0, k1, d0, q = (A[:, i].astype(np.int64) for i in range(6))
    isp = gmpy2.is_prime
    top_open = np.array([bool(isp(int(x) * int(x) - 2)) for x in p])
    L_top = np.where(top_open, 0, W - k0)
    tau = L_top / (W - q // 6)
    L_1 = k1 - W
    L_s = k1 - k0 - 1
    print(f"cuts: {len(p)}; cuts where p^2 - 2 is prime (top of the prefix open under 5..q): "
          f"{int(top_open.sum())} = {top_open.mean():.4f}")
    print()
    print("band of the rung q      cuts   max tau (at p)        max L_top   L_top = 0   median tau")
    for a, b in ((23, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10000), (10000, 20011),
                 (20011, 10 ** 5), (10 ** 5, 10 ** 6), (10 ** 6, 10 ** 7 + 1)):
        m = (q >= a) & (q < b)
        if not m.any():
            continue
        j = int(np.flatnonzero(m)[int(tau[m].argmax())])
        print(f"[{a}, {b})".ljust(22) + f"{int(m.sum()):7d}   {tau[j]:.4e} (p={int(p[j])})".ljust(24)
              + f"  {int(L_top[m].max()):9d}   {int((L_top[m] == 0).sum()):9d}   {np.median(tau[m]):.2e}")
    print()
    print(f"max tau over all cuts: {tau.max():.6f} at p={int(p[int(tau.argmax())])} (rung q={int(q[int(tau.argmax())])})")
    for lo in (23, 41, 20011):
        m = q >= lo
        print(f"  max tau over rungs q >= {lo}: {tau[m].max():.6e} at p={int(p[m][int(tau[m].argmax())])}")
    print(f"cuts with tau > 0.0833: {int((tau > 0.0833).sum())}")
    print()
    C = 4.625
    lhs = (1 - tau) * W
    rhs = C * (tau * W + L_1 + 1)
    fail = p[lhs < rhs]
    print(f"the review's 2.1d condition (1-tau)W >= {C}(tau W + L_1 + 1) with this tau: "
          f"{len(fail)} failures {[int(x) for x in fail][:20]}")
    m = p >= 41
    print(f"  among p >= 41: {int((lhs[m] < rhs[m]).sum())}; tightest LHS/RHS = {float((lhs[m] / rhs[m]).min()):.4f} "
          f"at p={int(p[m][int((lhs[m] / rhs[m]).argmin())])}")
    print(f"  the condition in its reduced form L_1 + 1 <= {(1 - C * 0.0833) / C:.5f} W with tau <= 0.0833: "
          f"failures {int((L_1 + 1 > (1 - C * 0.0833) / C * W).sum())}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    a = sys.argv[1]
    main(a if os.path.isabs(a) else os.path.join(here, a))
