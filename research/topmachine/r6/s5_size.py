"""Item 4 (how big) and item 5 (the owner's second number), finished.

  - the record's two endpoint families over every tested machine (is the record always
    bounded by two twin primes?)
  - g0 = the first gear with g0^2 > Q: how close g0^2 is to Q, over a long sweep
  - the growth of the zone record with Q at fixed q, against (log Q)^2 and against the
    largest prime gap in (Q, 2Q]

usage: uv run python research/topmachine/r6/s5_size.py
"""

import os
import sys
from math import isqrt, log

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (admissible_by_sieve, gap_report, largest_prime_gap,  # noqa: E402
                    open_pairs, primes_upto, smooth_numbers)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def cof(n, ps):
    s = 1
    for p in ps:
        while n % p == 0:
            n //= p
            s *= p
    return s, n


def main():
    say("# s5 HOW BIG: the record's endpoints, its growth, and g0^2 against Q")
    say()

    # --------------------------------------------------- g0^2 against Q, long sweep
    say("## Z14  the first gear g0 with g0^2 > Q, and where g0^2 falls")
    say()
    say("| Q | sqrt(Q) | g0 | g0^2 | (g0^2 - Q)/Q | nextprime(Q) | (g0^2 - Q)/sqrt(Q) |")
    say("|---|---|---|---|---|---|---|")
    pr = primes_upto(200000)
    for Q in [100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000, 10000000]:
        g0 = int(pr[pr > max(isqrt(Q), 5)][0])
        if g0 * g0 <= Q:
            g0 = int(pr[pr > g0][0])
        p1 = int(pr[pr > Q][0]) if Q < 200000 else 0
        say(f"| {Q:,} | {isqrt(Q)} | {g0} | {g0 * g0:,} | {(g0 * g0 - Q) / Q:.4f} | "
            f"{p1:,} | {(g0 * g0 - Q) / isqrt(Q):.2f} |")

    # ------------------------------------------------- endpoints of the record block
    say()
    say("## Z8  the record block's two endpoints, every machine")
    say()
    say("| q | Q | record | at | lower endpoint | family | upper endpoint | family | "
        "both (1,1)? |")
    say("|---|---|---|---|---|---|---|---|---|")
    QS = [200, 316, 500, 1000, 2000, 3162, 10000]
    QQ = [5, 7, 11, 13, 17, 37]
    hits, tot = 0, 0
    lastQ, primes = None, None
    growth = {}
    for Q in QS:
        X = Q * Q
        if Q != lastQ:
            primes = primes_upto(X)
            lastQ = Q
        prQ = primes[primes <= Q]
        for q in QQ:
            if q >= Q:
                continue
            ps = [int(p) for p in prQ if p <= q]
            adm = admissible_by_sieve(q, Q, X, prQ)
            op = open_pairs(adm)
            pos, ln = gap_report(op, Q + 1, len(op))
            i = int(np.argmax(ln))
            rec, at = int(ln[i]), int(pos[i])
            a0, b0 = at - 1, at + rec
            f0 = (cof(a0, ps)[0], cof(a0 + 2, ps)[0])
            f1 = (cof(b0, ps)[0], cof(b0 + 2, ps)[0])
            ok = f0 == (1, 1) and f1 == (1, 1)
            hits += ok
            tot += 1
            growth.setdefault(q, []).append((Q, rec, at))
            say(f"| {q} | {Q} | {rec} | {at:,} | {a0:,} | {f0} | {b0:,} | {f1} | "
                f"{'yes' if ok else 'no'} |")
            del adm, op
    say()
    say(f"record block bounded by two family-(1,1) open pairs (two primes two apart): "
        f"**{hits} of {tot}**")

    # ------------------------------------------------------------------- growth
    say()
    say("## Z11  growth of the zone record with Q at fixed q")
    say()
    for q in QQ:
        rows = growth.get(q, [])
        if not rows:
            continue
        say(f"### q = {q}")
        say()
        say("| Q | zone record | at | at/Q | (log Q)^2 | record/(log Q)^2 | "
            "largest prime gap in (Q,2Q] | record/that | [1,Q] record Q-2-s(q) |")
        say("|---|---|---|---|---|---|---|---|---|")
        for Q, rec, at in rows:
            primes = primes_upto(2 * Q + 10)
            pg, _ = largest_prime_gap(primes, Q, 2 * Q)
            sm = smooth_numbers(q, Q + 2)
            sset = set(sm)
            sk = max([n for n in sm if n <= Q - 2 and n + 2 in sset], default=0)
            lg = log(Q) ** 2
            say(f"| {Q:,} | {rec} | {at:,} | {at / Q:.2f} | {lg:.1f} | {rec / lg:.2f} | "
                f"{pg} | {rec / max(pg, 1):.2f} | {Q - 2 - sk} |")
        say()

    with open(os.path.join(RES, "s5_size.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
