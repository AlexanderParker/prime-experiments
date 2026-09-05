"""R2.a.i.a.1.a.2 - second moment over q.  Are twin-prime products q = p(p+2) poor machines?

The two largest failures of the one-class witness in [1000, 128000] are q = 10403 = 101*103 and
q = 11663 = 107*109, and over the whole sweep the 16 values q = p(p+2) have sum N / sum mu* =
0.862 against 1.018 for every other q.  Sixteen values is not a finding.  This script tests the
same statistic at a scale where a 15% deficit would be tens of sigma:

  every q = p(p+2) with p, p+2 prime and q <= QMAX, against matched controls (the nearest
  integers coprime to 30 with the same q mod 6, hence the same arc), and against the cousin
  products q = p(p+4) as a second family.

Reports sum N / sum mu_hat per family, where mu_hat = m prod_{11<=g<=q}(1 - 2/g).

Usage: uv run python research/anchor235/r50/mom_twin.py [--QMAX 10000000] [--WORKERS 4]
"""
import argparse
import os
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
G = {}


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def init(qmax):
    if "gears" in G:
        return
    fl = sieve(qmax + 10)
    gears = np.array([p for p in range(11, qmax + 1) if fl[p]], dtype=np.int64)
    G["fl"] = fl
    G["gears"] = gears
    G["u"] = np.array([pow(6, -1, int(g)) for g in gears], dtype=np.int64)
    G["i35"] = np.array([pow(35, -1, int(g)) for g in gears], dtype=np.int64)
    G["rate"] = np.cumprod(1.0 - 2.0 / gears)


def one_q(q):
    gears = G["gears"]
    ng = int(np.searchsorted(gears, q, side="right"))
    d = (2 * pow(6, -1, q)) % q
    m = (d - 13) // 35 + 1
    gl = gears[:ng]
    r = (q * q) % gl
    c1 = ((-r) * G["u"][:ng]) % gl
    c2 = ((2 - r) * G["u"][:ng]) % gl
    i35 = G["i35"][:ng]
    t1 = ((c1 - 12) * i35) % gl
    t2 = ((c2 - 12) * i35) % gl
    struck = np.zeros(m, dtype=bool)
    k0 = int(np.searchsorted(gl, m, side="left"))
    for j in range(k0):
        g = int(gl[j])
        struck[int(t1[j])::g] = True
        struck[int(t2[j])::g] = True
    if k0 < ng:
        for t in (t1[k0:], t2[k0:]):
            tt = t[t < m]
            if tt.size:
                struck[tt] = True
    N = int(m - int(struck.sum()))
    return N, m * float(G["rate"][ng - 1])


def work(args):
    qmax, batch = args
    init(qmax)
    return [(lab, q, ) + one_q(q) for lab, q in batch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--QMAX", type=int, default=10000000)
    ap.add_argument("--WORKERS", type=int, default=4)
    ap.add_argument("--NCTRL", type=int, default=6)
    args = ap.parse_args()
    P = isqrt(args.QMAX) + 10
    fl = sieve(P + 10)
    jobs = []
    seen = set()
    twins = []
    cousins = []
    balanced = []
    for p in range(11, P):
        if fl[p] and fl[p + 2] and p * (p + 2) <= args.QMAX and (p * (p + 2)) % 5:
            twins.append(p * (p + 2))
        if fl[p] and fl[p + 4] and p * (p + 4) <= args.QMAX and (p * (p + 4)) % 5:
            cousins.append(p * (p + 4))
        if fl[p]:
            r = p + 50
            while r < len(fl) and not fl[r]:
                r += 1
            if r < len(fl) and p * r <= args.QMAX and (p * r) % 5 and (p * r) % 2 and (p * r) % 3:
                balanced.append(p * r)
    for q in twins:
        jobs.append(("twin", q))
        seen.add(q)
    for q in cousins:
        if q not in seen:
            jobs.append(("cousin", q))
            seen.add(q)
    for q in balanced:
        if q not in seen:
            jobs.append(("balanced", q))
            seen.add(q)
    tw = set(twins) | set(cousins) | set(balanced)
    for q in twins:
        got = 0
        x = q
        while got < args.NCTRL:
            x += 2
            if x % 3 and x % 5 and x % 7 and x % 6 == q % 6 and x not in tw and x not in seen:
                jobs.append(("ctrl", x))
                seen.add(x)
                got += 1
        got = 0
        x = q
        while got < args.NCTRL:
            x -= 2
            if x % 3 and x % 5 and x % 7 and x % 6 == q % 6 and x not in tw and x not in seen:
                jobs.append(("ctrl", x))
                seen.add(x)
                got += 1
    from multiprocessing import Pool
    nb = args.WORKERS * 8
    batches = [(args.QMAX, jobs[i::nb]) for i in range(nb)]
    with Pool(args.WORKERS) as pool:
        res = [r for c in pool.map(work, batches) for r in c]
    L = ["twin-prime products q = p(p+2) against matched controls, q <= %d" % args.QMAX]
    L.append("  family    count      sum N      sum mu_hat   sum N/sum mu_hat   z vs control")
    agg = {}
    for lab, q, N, mh in res:
        a = agg.setdefault(lab, [0, 0.0, 0.0])
        a[0] += 1
        a[1] += N
        a[2] += mh
    base = agg["ctrl"][1] / agg["ctrl"][2]
    for lab in ("twin", "cousin", "balanced", "ctrl"):
        c, sN, sM = agg[lab]
        exp = base * sM
        z = (sN - exp) / (exp ** 0.5)
        L.append("  %-8s %7d %11.0f %14.1f %17.5f %13.2f" % (lab, c, sN, sM, sN / sM, z))
    L.append("  (z is against the control family's ratio, Poisson error on the total count)")
    # per-decade breakdown for the twins
    L.append("")
    L.append("  twin products by size    count   sum N/sum mu_hat   control ratio in the same range")
    edges = [(1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e7), (1e7, 1e8)]
    for lo, hi in edges:
        t = [(N, mh) for lab, q, N, mh in res if lab == "twin" and lo <= q < hi]
        c = [(N, mh) for lab, q, N, mh in res if lab == "ctrl" and lo <= q < hi]
        if not t:
            continue
        L.append("  %8.0e - %8.0e %7d %18.5f %20.5f"
                 % (lo, hi, len(t), sum(x[0] for x in t) / sum(x[1] for x in t),
                    (sum(x[0] for x in c) / sum(x[1] for x in c)) if c else float("nan")))
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "mom_twin.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
