"""R2.a.i.a.1.a.2 - second moment over q.  Does q's own factorisation shift the open count?

The CRT ensemble behind mu(q) is q coprime to every gear.  A composite q has DIVISOR GEARS
(primes g > 7 with g | q); by the divisor rule (N-I1) such a gear does not strike at the rate
2 chi_g(i)/(g-1) but strikes exactly the two classes i = 0 and i = 2 u_g (mod g), which are its
own barred classes.  This script measures, on the scan already computed, whether the number of
divisor gears shifts N(q) away from the model, and singles out the twin-prime products
q = p(p+2) that appear among the largest failures.

Usage: uv run python research/anchor235/r50/mom_div.py
"""
import os
from math import exp, isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
XS = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
S2 = 4.0 * exp(-2.0 * 0.5772156649015329)
QTOP = 2 * XS[-1] + 10


def spf_sieve(n):
    s = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, isqrt(n) + 1):
        if s[i] == 0:
            s[i * i:: i][s[i * i:: i] == 0] = i
    for i in range(2, n + 1):
        if s[i] == 0:
            s[i] = i
    return s


def main():
    spf = spf_sieve(QTOP)
    lines = []
    allq, allN, allmu, allom, alltp = [], [], [], [], []
    for X in XS:
        z = np.load(os.path.join(OUT, "mom_scan_X%d.npz" % X))
        q, N, mu = z["q"], z["N"].astype(float), z["mu_hat"] / S2
        om = np.zeros(len(q), dtype=np.int32)
        tp = np.zeros(len(q), dtype=bool)
        for k, qq in enumerate(q):
            x = int(qq)
            ps = []
            while x > 1:
                p = int(spf[x])
                ps.append(p)
                while x % p == 0:
                    x //= p
            om[k] = sum(1 for p in ps if p > 7)
            if len(ps) == 2 and ps[1] - ps[0] == 2:
                y = int(qq)
                if y == ps[0] * ps[1]:
                    tp[k] = True
        allq.append(q); allN.append(N); allmu.append(mu); allom.append(om); alltp.append(tp)
    q = np.concatenate(allq); N = np.concatenate(allN)
    mu = np.concatenate(allmu); om = np.concatenate(allom); tp = np.concatenate(alltp)

    lines.append("divisor gears (primes g > 7 dividing q) against the open count, all seven bands")
    lines.append("  omega>7   count      sum N      sum mu*    sum N / sum mu*   fails   fail rate")
    for w in range(0, 5):
        s = om == w
        if s.sum() == 0:
            continue
        lines.append("  %5d %8d %10.1f %11.2f %14.5f %7d   %.4e"
                     % (w, s.sum(), N[s].sum(), mu[s].sum(), N[s].sum() / mu[s].sum(),
                        int((N[s] == 0).sum()), (N[s] == 0).sum() / s.sum()))
    s = om >= 5
    if s.sum():
        lines.append("   >=5 %8d %10.1f %11.2f %14.5f %7d   %.4e"
                     % (s.sum(), N[s].sum(), mu[s].sum(), N[s].sum() / mu[s].sum(),
                        int((N[s] == 0).sum()), (N[s] == 0).sum() / s.sum()))
    lines.append("")
    lines.append("twin-prime products q = p(p+2) against all q, all seven bands")
    for name, s in (("q = p(p+2)", tp), ("all other q", ~tp)):
        lines.append("  %-12s count %6d   sum N/sum mu* = %.5f   fails %3d   fail rate %.4e"
                     % (name, s.sum(), N[s].sum() / mu[s].sum(), int((N[s] == 0).sum()),
                        (N[s] == 0).sum() / s.sum()))
    lines.append("  the p(p+2) values in the sweep: %s" % str([int(x) for x in q[tp]]))
    lines.append("  their N: %s" % str([int(x) for x in N[tp]]))
    lines.append("  their mu*: %s" % str([round(float(x), 2) for x in mu[tp]]))
    lines.append("")
    lines.append("per band, sum N / sum mu* split by whether q is prime (omega>7 = 1 and q prime)")
    lines.append("  band     primes: N/mu*    composites: N/mu*    all: N/mu*")
    off = 0
    for X, qa, Na, mua, oma in zip(XS, allq, allN, allmu, allom):
        isp = np.array([bool(spf[int(x)] == int(x)) for x in qa])
        lines.append("  %6d  %14.5f %20.5f %14.5f"
                     % (X, Na[isp].sum() / mua[isp].sum(),
                        Na[~isp].sum() / mua[~isp].sum(), Na.sum() / mua.sum()))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "mom_div.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
