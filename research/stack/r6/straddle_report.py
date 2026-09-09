"""Read the straddle scan's rows and report the frontier floor's two computable quantities.

Rows: p, W, k0, k1, d0, q(=prevprime(p)).  Derived per cut p:
    x_s = k0 + 1                       start of the straddling run (columns)
    L_s = k1 - k0 - 1                  its length (columns)
    ratio = x_s / L_s
    L_top = W - k0                     its part inside the prefix of {5..q}
    tau = L_top / (W - q//6)           the top-run share at the rung q (valve_existence table 2)
    L_1 = k1 - W                       the first-twin offset above p^2 (columns)
    arc = (2p + 1)//3                  the long-arc bound the first-twin scan measured
    bites iff L_s >= d0
The review's 2.1d sufficient condition, checked exactly per cut:
    (1 - tau) W >= 4.625 (tau W + L_1 + 1)

Usage: uv run python research/stack/r6/straddle_report.py results/straddle_<tag>.npz [...]
"""
import os
import sys

import numpy as np

C = 4.625


def main(paths):
    A = np.concatenate([np.load(p)["rows"] for p in paths])
    A = A[np.argsort(A[:, 0])]
    p, W, k0, k1, d0, q = (A[:, i].astype(np.int64) for i in range(6))
    x_s = k0 + 1
    L_s = k1 - k0 - 1
    ratio = x_s / L_s
    L_top = W - k0
    wincols = W - q // 6
    tau = L_top / wincols
    L_1 = k1 - W
    arc = (2 * p + 1) // 3
    bites = L_s >= d0
    print(f"cuts p in [{p.min()}, {p.max()}]: {len(p)} primes")
    print()
    print("-- the straddling run --")
    j = int(ratio.argmin())
    print(f"min ratio x_s/L_s over ALL cuts: {ratio[j]:.4f} at p={p[j]}  (x_s={x_s[j]}, L_s={L_s[j]}, d_0={d0[j]})")
    for lo in (11, 19, 23, 41, 118):
        m = p >= lo
        jj = int(np.flatnonzero(m)[int(ratio[m].argmin())])
        print(f"  min ratio over p >= {lo}: {ratio[jj]:.4f} at p={p[jj]}")
    print(f"cuts below the floor {C} (all cuts): {int((ratio < C).sum())}"
          f"  {[(int(a), round(float(b), 3)) for a, b in zip(p[ratio < C], ratio[ratio < C])][:20]}")
    print(f"max L_s = {int(L_s.max())} at p={int(p[int(L_s.argmax())])}; max L_1 = {int(L_1.max())} at p={int(p[int(L_1.argmax())])}")
    print()
    print("-- where the condition bites (L_s >= d_0) --")
    nb = int(bites.sum())
    print(f"biting cuts: {nb} of {len(p)}; largest biting p = {int(p[bites].max()) if nb else None}")
    if nb:
        rb = ratio[bites]
        jb = int(rb.argmin())
        print(f"min ratio over biting cuts: {rb[jb]:.4f} at p={int(p[bites][jb])}")
        print("  biting cuts (p, x_s, L_s, d_0, ratio):")
        for a, b, c, d, e in zip(p[bites], x_s[bites], L_s[bites], d0[bites], rb):
            print(f"    {int(a):6d} {int(b):9d} {int(c):5d} {int(d):5d}  {e:9.3f}")
        below = p[bites][rb < C]
        print(f"  biting cuts with ratio < {C}: {len(below)} {[int(x) for x in below]}")
    print()
    print("-- the top-run share tau --")
    jt = int(tau.argmax())
    print(f"max tau = {tau[jt]:.6f} at p={int(p[jt])} (rung q={int(q[jt])}, L_top={int(L_top[jt])} of {int(wincols[jt])} columns)")
    print(f"cuts with tau > 0.0833: {int((tau > 0.0833).sum())}; tau > 0.05: {int((tau > 0.05).sum())}; tau > 0.01: {int((tau > 0.01).sum())}")
    for a, b in ((7, 100), (100, 1000), (1000, 10 ** 4), (10 ** 4, 10 ** 5), (10 ** 5, 10 ** 6), (10 ** 6, 10 ** 8)):
        m = (p >= a) & (p < b)
        if m.any():
            jm = int(np.flatnonzero(m)[int(tau[m].argmax())])
            jr = int(np.flatnonzero(m)[int(ratio[m].argmin())])
            print(f"  p in [{a}, {b}): {int(m.sum()):7d} cuts | max tau {tau[jm]:.3e} (p={int(p[jm])}) "
                  f"| min ratio {ratio[jr]:.4g} (p={int(p[jr])}) | max L_s {int(L_s[m].max())} | max L_1 {int(L_1[m].max())}")
    print()
    print("-- the first-twin arc bound (the r3 scan's object, recomputed here) --")
    bad = p[L_1 >= arc]
    print(f"cuts with L_1 >= (2p+1)/3: {len(bad)} {[int(x) for x in bad][:20]}")
    print(f"max L_1/arc = {float((L_1 / arc).max()):.4f} at p={int(p[int((L_1 / arc).argmax())])}")
    print()
    print("-- the review's 2.1d sufficient condition, exact per cut --")
    lhs = (1 - tau) * W
    rhs = C * (tau * W + L_1 + 1)
    fail = p[lhs < rhs]
    print(f"cuts failing (1-tau)W >= {C}(tau W + L_1 + 1): {len(fail)} {[int(x) for x in fail][:30]}")
    m = p >= 41
    fail41 = p[m][lhs[m] < rhs[m]]
    print(f"  among p >= 41: {len(fail41)} {[int(x) for x in fail41][:20]}")
    slack = lhs / rhs
    m = p >= 41
    js = int(np.flatnonzero(m)[int(slack[m].argmin())])
    print(f"  tightest cut with p >= 41: p={int(p[js])}, LHS/RHS = {slack[js]:.4f} "
          f"(tau={tau[js]:.3e}, L_1={int(L_1[js])}, W={int(W[js])})")
    print(f"  max tau over rungs q >= 23: {tau[q >= 23].max():.6f} at p={int(p[q >= 23][int(tau[q >= 23].argmax())])}")
    print(f"  max tau over cuts p >= 41: {tau[p >= 41].max():.3e} at p={int(p[p >= 41][int(tau[p >= 41].argmax())])}")
    print(f"  max L_s/d_0 over cuts p > 487: {float((L_s / d0)[p > 487].max()):.4f} at "
          f"p={int(p[p > 487][int((L_s / d0)[p > 487].argmax())])}  (the condition bites at >= 1)")
    print()
    print("-- verdict --")
    for lo in (7, 11, 23):
        m = bites & (p >= lo)
        ok = bool((ratio[m] >= C).all()) if m.any() else True
        print(f"  over biting cuts p >= {lo:2d} ({int(m.sum())} cuts): floor {C} "
              f"{'CARRIES' if ok else 'FAILS'} to p = {int(p.max())}; min ratio {float(ratio[m].min()):.4f}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main([a if os.path.isabs(a) else os.path.join(here, a) for a in sys.argv[1:]])
