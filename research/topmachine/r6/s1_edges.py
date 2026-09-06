"""Item 1: THE EDGES.  Which definition of the zone is the right one.

Tests, exactly, for q in {5,7,11,13,17} and Q from 50 to 10,000:

  Z1  n admissible  <=>  n = s * P  (s q-smooth, P = 1 or one prime > Q)  on all of [1, Q^2]
  Z2  the handover: the first admissible n that is not q-smooth   (predicted nextprime(Q))
      and whether anything at all happens at Q - sqrt(Q)  or at g0 = first gear with g0^2 > Q
  Z3  where the rule dies above Q^2                                (predicted nextprime(Q)^2)
  Z4  the stratification: the largest q-smooth cofactor of an admissible n at height x
      (predicted floor(x / nextprime(Q)))

usage: uv run python research/topmachine/r6/s1_edges.py
"""

import os
import sys
from math import isqrt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (admissible_by_rule, admissible_by_sieve, open_pairs,  # noqa: E402
                    primes_upto, smooth_numbers)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

QS = [50, 100, 200, 316, 500, 1000, 2000, 3162, 10000]
QQ = [5, 7, 11, 13, 17]
XCAP = 160_000_000

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def main():
    say("# s1 THE EDGES: the zone rule, its handover and its death")
    say()
    say("| q | Q | X scanned | rule vs sieve on [1,Q^2] | handover (first non-smooth "
        "admissible) | nextprime(Q) | first rule failure > Q^2 | nextprime(Q)^2 |")
    say("|---|---|---|---|---|---|---|---|")

    rows = []
    for Q in QS:
        X = min(3 * Q * Q // 2, XCAP)
        primes = primes_upto(X)
        p1 = int(primes[primes > Q][0])
        prQ = primes[primes <= Q]
        for q in QQ:
            if q >= Q:
                continue
            adm = admissible_by_sieve(q, Q, X, prQ)
            rule = admissible_by_rule(q, Q, X, primes)
            top = min(Q * Q, X)
            bad = np.flatnonzero(adm[1:top + 1] != rule[1:top + 1]) + 1
            nbad = int(len(bad))
            # handover: first admissible n that is not q-smooth
            sm = np.asarray(smooth_numbers(q, X), dtype=np.int64)
            issm = np.zeros(X + 1, dtype=bool)
            issm[sm] = True
            hand = int(np.flatnonzero(adm[1:] & ~issm[1:])[0] + 1)
            # first failure above Q^2
            if X > Q * Q:
                d = np.flatnonzero(adm[Q * Q + 1:] != rule[Q * Q + 1:])
                first_fail = int(d[0] + Q * Q + 1) if len(d) else -1
            else:
                first_fail = -1
            rows.append((q, Q, X, nbad, hand, p1, first_fail, p1 * p1))
            say(f"| {q} | {Q} | {X:,} | {nbad} | {hand:,} | {p1:,} | "
                f"{first_fail:,} | {p1 * p1:,} |")
            del adm, rule, issm
        del primes

    say()
    say(f"rule-vs-sieve exceptions on [1, Q^2], total over {len(rows)} machines: "
        f"{sum(r[3] for r in rows)}")
    say(f"handover = nextprime(Q): {sum(1 for r in rows if r[4] == r[5])} of {len(rows)}")
    ff = [r for r in rows if r[6] > 0]
    say(f"first failure above Q^2 = nextprime(Q)^2: "
        f"{sum(1 for r in ff if r[6] == r[7])} of {len(ff)}")

    # ---------------------------------------------------------------- the two edges
    say()
    say("## The candidate lower edges, measured")
    say()
    say("| q | Q | Q-sqrt(Q) | open pairs in (Q-sqrtQ, Q] | g0 | g0^2 | open pairs in "
        "(g0^2/2, g0^2] | last open pair <= Q | first open pair > Q | p1 |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for Q in QS:
        X = min(3 * Q * Q // 2, XCAP)
        primes = primes_upto(X)
        p1 = int(primes[primes > Q][0])
        prQ = primes[primes <= Q]
        for q in QQ:
            if q >= Q:
                continue
            adm = admissible_by_sieve(q, Q, X, prQ)
            op = open_pairs(adm)
            lo = Q - isqrt(Q)
            n1 = int(op[lo + 1:Q + 1].sum())
            cand = [int(g) for g in prQ if g > q and g * g > Q]
            g0 = cand[0] if cand else 0
            n2 = int(op[g0 * g0 // 2:g0 * g0 + 1].sum()) if g0 else 0
            below = np.flatnonzero(op[:Q + 1])
            above = np.flatnonzero(op[Q + 1:])
            lastb = int(below[-1]) if len(below) else -1
            firsta = int(above[0] + Q + 1) if len(above) else -1
            say(f"| {q} | {Q} | {lo} | {n1} | {g0} | {g0 * g0} | {n2} | {lastb} | "
                f"{firsta:,} | {p1:,} |")
            del adm, op
        del primes

    # ------------------------------------------------------- Z4 the stratification
    say()
    say("## Z4  the stratification: largest q-smooth cofactor of an admissible n by height")
    say()
    say("q=5, Q=10,000, p1=10,007.  Strata (2^k Q, 2^(k+1) Q].")
    say()
    say("| stratum | floor(x_hi/p1) | max cofactor s observed (P > 1) | admissible | "
        "open pairs | open density |")
    say("|---|---|---|---|---|---|")
    Q = 10000
    X = Q * Q
    primes = primes_upto(X)
    prQ = primes[primes <= Q]
    p1 = int(primes[primes > Q][0])
    q = 5
    adm = admissible_by_sieve(q, Q, X, prQ)
    op = open_pairs(adm)
    sm = np.asarray(smooth_numbers(q, X), dtype=np.int64)
    # smooth part of every admissible n, by repeated division on the admissible set only
    k = 0
    while True:
        lo, hi = Q << k, min(Q << (k + 1), X)
        if lo >= X:
            break
        idx = np.flatnonzero(adm[lo:hi]).astype(np.int64) + lo
        v = idx.copy()
        cof = np.ones(len(idx), dtype=np.int64)
        for p in (2, 3, 5):
            while True:
                m = (v % p) == 0
                if not m.any():
                    break
                v[m] //= p
                cof[m] *= p
        big = cof[v > 1]                       # P > 1 only: drop the q-smooth numbers
        nop = int(op[lo:hi].sum())
        say(f"| ({lo:,}, {hi:,}] | {(hi - 1) // p1} | "
            f"{int(big.max()) if len(big) else 0} | "
            f"{len(idx):,} | {nop:,} | {nop / (hi - lo):.4f} |")
        k += 1
    say()
    say(f"(the {len(sm)} 5-smooth numbers <= 10^8 are the P = 1 part)")

    with open(os.path.join(RES, "s1_edges.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
