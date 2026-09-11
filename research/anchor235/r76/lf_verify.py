"""lf_verify.py -- second measurements for the length-face lane.

(1) The parity twin's record gaps reported by lf_twin.py (results/twin.json) are re-verified
    member by member with sympy.factorint: every open column strictly inside the reported O+
    gap has sigma = -1, every open column inside the reported O- gap has sigma = +1, and the two
    ends are in the right set.  (Maximality over [0, X] rests on the scan alone.)
(2) The sign sums of the record runs of m23, m29, m31, m37 (results/census.json, computed by
    factorint) are recomputed by the division sieve of lf_twin.py on the run's own segment.
(3) The first O- column at the cuts q = 41, 43, 47, 53 by walking the columns from b with
    factorint (lf_twin.py stops at q = 37), and the check that it is the column (r^2 - 2, r^2)
    of a prime r with r^2 - 2 prime.
(4) A compact print of the m23 record run: per column the gear parts and the signs.

Usage: uv run python lf_verify.py
"""
import json
import os
import sys

import numpy as np
from sympy import factorint, isprime, nextprime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lf_common import gears_of, primes_upto  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def lam(n):
    return (-1) ** sum(factorint(int(n)).values())


def rough(n, q):
    f = factorint(int(n))
    return all(p > q for p in f)


def is_open(k, q):
    return rough(6 * k - 1, q) and rough(6 * k + 1, q)


def sigma(k):
    return lam(6 * k - 1) * lam(6 * k + 1)


def omega_segment(lo, hi):
    """Omega(n) for n in [lo, hi) by repeated division (the lf_twin.py method)."""
    n = hi - lo
    rem = np.arange(lo, hi, dtype=np.int64)
    om = np.zeros(n, dtype=np.int64)
    for p in primes_upto(int(hi ** 0.5) + 1):
        pk = int(p)
        while pk < hi:
            start = (-lo) % pk
            if start < n:
                sl = slice(start, None, pk)
                rem[sl] //= int(p)
                om[sl] += 1
            pk *= int(p)
    om += (rem > 1)
    return om


def main():
    with open(os.path.join(RES, "twin.json")) as f:
        tw = json.load(f)
    with open(os.path.join(RES, "census.json")) as f:
        ce = json.load(f)
    print("(1) parity-twin record gaps re-verified by factorint")
    for q in [11, 13, 17, 19, 23, 29, 31, 37]:
        r = tw[f"m{q}"]
        checks = []
        # O+ gap: from gap_Oplus_at, length gap_Oplus: the open columns strictly between have sigma = -1
        for name, at, gap, sign_inside in [("O+", r["gap_Oplus_at"], r["gap_Oplus"], -1),
                                           ("O-", r["gap_Ominus_excl_at"], r["gap_Ominus_excl_initial"], +1)]:
            a, b_ = at, at + gap
            inside = [k for k in range(a + 1, b_) if is_open(k, q)]
            ok_inside = all(sigma(k) == sign_inside for k in inside)
            ok_ends = is_open(a, q) and is_open(b_, q) and sigma(a) == -sign_inside and sigma(b_) == -sign_inside
            checks.append((name, at, gap, len(inside), ok_inside and ok_ends))
        # O- initial gap: no O- column below first_Ominus, and first_Ominus is in O-
        fm = r["first_Ominus"]["k"]
        no_minus_below = all(not (is_open(k, q) and sigma(k) == -1) for k in range(1, fm))
        ok_first = is_open(fm, q) and sigma(fm) == -1
        print(f"  m{q}: " + "; ".join(f"{nm} gap {g} at {a:,} ({ni} openings of the other sign inside): {'OK' if ok else 'FAIL'}" for (nm, a, g, ni, ok) in checks)
              + f"; first O- column {fm} (b = {r['b']}): {'OK' if (no_minus_below and ok_first) else 'FAIL'}")

    print("\n(2) record-run sign sums by the division sieve (census.json values by factorint in brackets)")
    for q, (x, L) in {23: (12_694_429, 33), 29: (200_906_186, 42), 31: (1_468_940_243, 57), 37: (90_816_580_903, 87)}.items():
        lo, hi = 6 * x - 1, 6 * (x + L - 1) + 2
        om = omega_segment(lo, hi)
        lam_ = 1 - 2 * (om & 1)
        ks = np.arange(x, x + L)
        l1 = lam_[6 * ks - 1 - lo]
        l2 = lam_[6 * ks + 1 - lo]
        s_all = int(l1.sum() + l2.sum())
        s_sig = int((l1 * l2).sum())
        rec = ce[f"m{q}"]["table"]
        print(f"  m{q}: sum lambda over members {s_all:+d} [{rec['sum_lam_all']['REC'][0]:+d}], sum sigma {s_sig:+d} [{rec['sum_sigma']['REC'][0]:+d}]"
              f"  {'OK' if (s_all == rec['sum_lam_all']['REC'][0] and s_sig == rec['sum_sigma']['REC'][0]) else 'MISMATCH'}")

    print("\n(3) the first O- column at the larger cuts")
    for q in [41, 43, 47, 53]:
        qp = int(nextprime(q))
        b = (qp * qp - 1) // 6
        k = b
        while not (is_open(k, q) and sigma(k) == -1):
            k += 1
        n1, n2 = 6 * k - 1, 6 * k + 1
        f1, f2 = factorint(n1), factorint(n2)
        sq = None
        for n, f in ((n1, f1), (n2, f2)):
            if len(f) == 1 and list(f.values())[0] == 2:
                sq = list(f.keys())[0]
        print(f"  m{q}: b = {b}, first O- column {k} = b + {k - b}: {n1} = {dict(f1)}, {n2} = {dict(f2)}"
              + (f"; the square column of r = {sq} with r^2 - 2 = {sq * sq - 2} {'prime' if isprime(sq * sq - 2) else 'composite'}" if sq else "; not a square column"))

    print("\n(4) the m23 record run, column by column: k | gear parts of 6k-1, lambda, Omega(rough) | same for 6k+1 | sigma")
    ex = ce["m23"]["exhibit"]
    for row in ex:
        print(f"  {row['k']:>10,d} | {row['s-']:>5d} {row['lam-']:+d} {row['omr-']} | {row['s+']:>5d} {row['lam+']:+d} {row['omr+']} | {row['lam-'] * row['lam+']:+d}")


if __name__ == "__main__":
    main()
