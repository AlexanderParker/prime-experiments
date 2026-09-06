"""Item 2: CUTMONO - the cut sequence, and the exact elementary form the proof needs.

cut_0 = q, cut_1 = q#, cut_{k+1} = prod of the primes in (cut_{k-1}, cut_k].

Measured here:
 (a) the cuts for q = 2, 3, 5, 7, 11, 13 - cut_2 exactly (big int), cut_3 as a digit count from
     Chebyshev's theta computed by a segmented sieve where cut_2 is small enough (q = 5), and from
     theta(x) ~ x otherwise;
 (b) the ratio cut_{k+1} / cut_k, which is what the induction needs;
 (c) the SHARP threshold of the elementary lemma: for each a, the largest b with
     prod{p prime : a < p <= b} <= b.  The claim to be proved is that this largest b is below 4a
     for every a >= 5, so that "b >= 4a" is a safe hypothesis;
 (d) the exact cause of the failure at q = 2, 3: the first ratio cut_1 / cut_0 = prod of the
     primes BELOW q, which is 1, 2, 6, 30, 210, 2310 at q = 2, 3, 5, 7, 11, 13 - below 4 exactly
     at q = 2 and q = 3.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import primes_in, primes_upto  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
sys.set_int_max_str_digits(200000)


def theta_upto(X, seg=10 ** 7):
    """Chebyshev theta(X) = sum of log p over primes p <= X, by a segmented sieve."""
    base = primes_upto(int(math.isqrt(X)) + 1)
    total = 0.0
    count = 0
    lo = 2
    while lo <= X:
        hi = min(X, lo + seg - 1)
        mask = np.ones(hi - lo + 1, dtype=bool)
        for p in base:
            p = int(p)
            start = max(p * p, ((lo + p - 1) // p) * p)
            if start > hi:
                continue
            mask[start - lo:: p] = False
        if lo <= 1:
            mask[:2 - lo] = False
        idx = np.flatnonzero(mask).astype(np.int64) + lo
        total += float(np.log(idx.astype(np.float64)).sum())
        count += idx.size
        lo = hi + 1
    return total, count


def sharp_threshold(amax):
    """For each a, the largest b with prod of the primes in (a, b] <= b (searching b <= 8a)."""
    pr = primes_upto(8 * amax + 10)
    logp = np.log(pr.astype(np.float64))
    cum = np.concatenate([[0.0], np.cumsum(logp)])
    worst = []
    for a in range(2, amax + 1):
        i = int(np.searchsorted(pr, a, side="right"))
        best = 0
        for b in range(a + 1, 8 * a + 1):
            j = int(np.searchsorted(pr, b, side="right"))
            s = cum[j] - cum[i]
            if s <= math.log(b) + 1e-12:
                best = b
        worst.append((a, best, best / a))
    return worst


def main():
    out = {}
    print("=== (a) the cuts ===")
    rows = []
    for q in [2, 3, 5, 7, 11, 13]:
        below = math.prod(primes_in(1, q - 1)) if q > 2 else 1
        cut1 = math.prod(primes_in(1, q))
        t2 = primes_in(q, cut1)
        cut2 = math.prod(t2)
        d2 = len(str(cut2))
        row = {"q": q, "prod_primes_below_q": below, "cut0": q, "cut1": cut1,
               "n_tier2": len(t2), "cut2_digits": d2,
               "cut2": cut2 if d2 <= 12 else None,
               "cut1_over_cut0": cut1 / q,
               "cut2_ge_cut1": cut2 >= cut1, "cut2_over_cut1": (cut2 / cut1) if d2 < 300 else "huge",
               "monotone_first_two_steps": (q <= cut1) and (cut1 <= cut2)}
        rows.append(row)
        print(f"q={q:3d}  prod primes below q = {below:6d}  cut1 = q# = {cut1:8d}  "
              f"tier2 has {len(t2):5d} gears  cut2 has {d2} digits  "
              f"cut1<=cut2: {cut1 <= cut2}")
    out["cuts"] = rows

    print("\n=== (b) cut_3 at q = 5, exactly, from theta ===")
    cut2_5 = math.prod(primes_in(5, 30))
    th, cnt = theta_upto(cut2_5)
    th30 = math.log(math.prod(primes_in(1, 30)))
    L = th - th30
    d3 = int(L / math.log(10)) + 1
    print(f"cut_2 = {cut2_5}; pi(cut_2) = {cnt}; theta(cut_2) = {th:.6f}")
    print(f"log cut_3 = theta(cut_2) - theta(30) = {L:.6f}  "
          f"(theta(30) = {th30:.6f}); cut_3 has {d3} digits")
    print(f"theta(cut_2)/cut_2 = {th / cut2_5:.6f}   (Chebyshev: -> 1)")
    print(f"cut_3 / cut_2 is e^({L:.3f}) / {cut2_5}, i.e. astronomically above 1")
    out["cut3_q5"] = {"cut2": cut2_5, "pi_cut2": cnt, "theta_cut2": th,
                      "theta_30": th30, "log_cut3": L, "cut3_digits": d3,
                      "theta_over_x": th / cut2_5}

    print("\n=== theta-only estimates for q >= 7 ===")
    est = []
    for q in [7, 11, 13]:
        cut1 = math.prod(primes_in(1, q))
        cut2 = math.prod(primes_in(q, cut1))
        d2 = len(str(cut2))
        # log cut_3 = theta(cut_2) - theta(cut_1) ~ cut_2, so cut_3 has ~ cut_2 / ln 10 digits
        d3digits_digits = d2 - int(math.log10(math.log(10)))  # digits of the digit count
        est.append({"q": q, "cut2_digits": d2, "cut3_digit_count_digits": d3digits_digits})
        print(f"q={q}: cut_2 has {d2} digits; log cut_3 = theta(cut_2) - theta(cut_1) ~ cut_2, "
              f"so cut_3 has about cut_2 / ln 10 digits - a digit count with {d3digits_digits} "
              f"digits of its own")
    out["cut3_estimates"] = est

    print("\n=== (c) the sharp threshold of the elementary lemma ===")
    w = sharp_threshold(3000)
    bad5 = [(a, b, r) for a, b, r in w if a >= 5 and b >= 4 * a]
    mx = max((r for a, b, r in w if a >= 5), default=0)
    arg = [(a, b) for a, b, r in w if a >= 5 and r == mx]
    print(f"a = 2..3000: largest b with prod primes in (a,b] <= b.")
    print(f"  for a >= 5 the largest such b never reaches 4a: max b/a = {mx:.4f} at {arg[:5]}")
    print(f"  violations of 'b < 4a' for a >= 5: {len(bad5)}")
    for a, b, r in w[:8]:
        print(f"    a={a:3d}  largest failing b = {b:4d}  b/a = {r:.3f}")
    out["threshold"] = {"amax": 3000, "max_ratio_a_ge_5": mx,
                        "argmax": arg[:5], "violations_b_ge_4a": len(bad5),
                        "small_a": w[:10]}

    print("\n=== (d) the exact cause of the q = 2, 3 failure ===")
    for q in [2, 3, 5, 7]:
        below = math.prod(primes_in(1, q - 1)) if q > 2 else 1
        cut1 = math.prod(primes_in(1, q))
        cut2 = math.prod(primes_in(q, cut1))
        t = int(math.floor(math.log2(cut1 / q))) if cut1 > q else 0
        print(f"q={q}: cut_1/cut_0 = prod primes below q = {below}; dyadic intervals available "
              f"t = {t}; cut_2 = {cut2}; cut_1 = {cut1}; cut_2 >= cut_1: {cut2 >= cut1}")
    out["failure_cause"] = "cut_1/cut_0 = prod of the primes below q; < 4 exactly at q = 2, 3"

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e2_cutmono.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
