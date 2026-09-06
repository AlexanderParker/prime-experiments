"""The first pure charge above Q: t_1(Q) = the first twin prime pair with lower member > Q,
against p_1 = nextprime(Q) and the q-smooth numbers near Q.

usage: uv run python research/valves/scratch/first_pure.py
"""
import json, math, os
import numpy as np
from sympy import isprime, nextprime, primerange

here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)


def smooth(n, q):
    for p in primerange(2, q + 1):
        while n % p == 0:
            n //= p
    return n == 1


def mopen_num(n, Q, q):
    """manifold-open in the quiet zone: q-smooth times at most one prime above Q.
    For n < Q^2 this is: no prime factor in (q, Q]."""
    m = n
    for p in primerange(2, q + 1):
        while m % p == 0:
            m //= p
    return m == 1 or (m > Q and isprime(m))


def first_twin_above(Q):
    p = nextprime(Q)
    while not isprime(p + 2):
        p = nextprime(p)
    return p


rows = []
# Q = q# for q = 5 .. 47, base q
qs = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
Q = 1
for q in qs:
    Q *= q
    if q == 5:
        Q = 30
    p1 = nextprime(Q)
    t1 = first_twin_above(Q)
    # charges strictly before t1: pairs (n, n+2) with Q < n < t1 both manifold-open
    burnt_before = []
    for n in range(Q + 1, t1):
        if mopen_num(n, Q, q) and mopen_num(n + 2, Q, q):
            burnt_before.append(n)
    L = math.log(Q)
    rows.append({"Q": f"{q}#", "Qval": Q, "q": q, "p1_minus_Q": p1 - Q, "t1_minus_Q": t1 - Q, "t1_eq_p1": t1 == p1,
                 "charges_before_t1": len(burnt_before), "first_charges": burnt_before[:5],
                 "t1_gap_over_log2": (t1 - Q) / L ** 2, "p1_gap_over_log": (p1 - Q) / L,
                 "Q_pm1_twin": isprime(Q - 1) and isprime(Q + 1)})

# Q = 10^k, k = 3..12, q = 5 and q = 13
for k in range(3, 13):
    Q = 10 ** k
    p1 = nextprime(Q); t1 = first_twin_above(Q)
    L = math.log(Q)
    for q in (5, 13):
        burnt_before = [n for n in range(Q + 1, t1) if mopen_num(n, Q, q) and mopen_num(n + 2, Q, q)]
        rows.append({"Q": f"10^{k}", "Qval": Q, "q": q, "p1_minus_Q": p1 - Q, "t1_minus_Q": t1 - Q, "t1_eq_p1": t1 == p1,
                     "charges_before_t1": len(burnt_before), "first_charges": burnt_before[:5],
                     "t1_gap_over_log2": (t1 - Q) / L ** 2, "p1_gap_over_log": (p1 - Q) / L,
                     "Q_pm1_twin": isprime(Q - 1) and isprime(Q + 1)})

# statistics over many Q at each scale: P(t1 = p1), mean (t1 - Q), and correlation with (p1 - Q)
stats = []
for scale in (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7):
    lo, hi = scale, scale + 10 ** 5
    top = hi + 10 ** 5
    is_p = np.ones(top + 3, dtype=bool); is_p[:2] = False
    for i in range(2, int(top ** 0.5) + 2):
        if is_p[i]:
            is_p[i * i::i] = False
    primes = np.flatnonzero(is_p)
    twins = primes[is_p[primes + 2]]
    Qs = np.arange(lo, hi, dtype=np.int64)
    p1 = primes[np.searchsorted(primes, Qs + 1)]
    t1 = twins[np.searchsorted(twins, Qs + 1)]
    d_p = (p1 - Qs).astype(float); d_t = (t1 - Qs).astype(float)
    L = math.log(scale)
    stats.append({"scale": scale, "samples": int(len(Qs)), "P_t1_eq_p1": float(np.mean(t1 == p1)),
                  "mean_t1_minus_Q": float(d_t.mean()), "mean_t1_minus_Q_over_log2": float(d_t.mean() / L ** 2),
                  "mean_p1_minus_Q": float(d_p.mean()), "corr_dp_dt": float(np.corrcoef(d_p, d_t)[0, 1]),
                  "max_t1_minus_Q": int(d_t.max()), "median_t1_minus_Q": float(np.median(d_t)),
                  # exponential check: var/mean^2 of d_t (1 for a memoryless gap)
                  "var_over_mean2": float(d_t.var() / d_t.mean() ** 2)})

json.dump({"rows": rows, "stats": stats}, open(os.path.join(outdir, "first_pure.json"), "w"), indent=1, default=int)
for r in rows:
    print(r["Q"], "q=", r["q"], "p1-Q=", r["p1_minus_Q"], "t1-Q=", r["t1_minus_Q"], "t1=p1:", r["t1_eq_p1"],
          "charges before t1:", r["charges_before_t1"], r["first_charges"], "t1/log2=%.3f" % r["t1_gap_over_log2"], "Q+-1 twin:", r["Q_pm1_twin"])
for s in stats:
    print(s)
