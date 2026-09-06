"""Item 3 and 4: THE ALIGNMENTS and HOW BIG.

  Z7   the proved lower bound: a prime gap in (Q, 2Q] with no q-smooth number inside is a
       run of struck pairs.  Alignments always occur.
  Z8   the record's position (bottom stratum?) and the family of its two endpoints
  Z10  cross-check against A(q, N) of top_machine_4.md
  Z11  growth of the zone record with Q at fixed q, against the [1, Q] record

usage: uv run python research/topmachine/r6/s3_gaps.py
"""

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (admissible_by_sieve, gap_report, open_pairs,  # noqa: E402
                    primes_upto, smooth_numbers)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

QS = [100, 200, 316, 500, 1000, 2000, 3162, 10000]
QQ = [5, 7, 11, 13, 17, 37]

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


def proved_lower_bound(primes, Q, smoothset):
    """max (p' - p - 1) over consecutive primes p < p' in (Q, 2Q] with no q-smooth in (p, p')."""
    sel = [int(p) for p in primes[(primes > Q) & (primes <= 2 * Q)]]
    best, at = 0, 0
    for a, b in zip(sel, sel[1:]):
        if any(x in smoothset for x in range(a + 1, b)):
            continue
        if b - a - 1 > best:
            best, at = b - a - 1, a + 1
    return best, at


def main():
    say("# s3 THE ALIGNMENTS: gaps, the record, its position and its size")
    say()
    say("## The zone record, its position, and the proved prime-gap lower bound")
    say()
    say("| q | Q | zone (Q, Q^2] | record | position | position/Q | position/Q^2 | "
        "proved bound (prime gap in (Q,2Q]) | at | record in (Q,2Q] | [1,Q] record |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")

    table = []
    lastQ, primes = None, None
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
            if not len(ln):
                continue
            i = int(np.argmax(ln))
            rec, at = int(ln[i]), int(pos[i])
            # bottom stratum record
            m = pos <= 2 * Q
            recb = int(ln[m].max()) if m.any() else 0
            # [1, Q] record
            pos0, ln0 = gap_report(op, 0, Q + 1)
            rec0 = int(ln0.max()) if len(ln0) else 0
            smoothset = set(smooth_numbers(q, 2 * Q))
            pb, pbat = proved_lower_bound(primes, Q, smoothset)
            table.append((q, Q, rec, at, recb, rec0, pb))
            say(f"| {q} | {Q} | ({Q:,}, {X:,}] | **{rec}** | {at:,} | {at / Q:.2f} | "
                f"{at / X:.2e} | {pb} | {pbat:,} | {recb} | {rec0} |")
            del adm, op

    say()
    say(f"record >= proved prime-gap bound: "
        f"{sum(1 for r in table if r[2] >= r[6])} of {len(table)}; "
        f"ratio record/bound min {min(r[2] / max(r[6], 1) for r in table):.2f}, "
        f"max {max(r[2] / max(r[6], 1) for r in table):.2f}")
    say(f"record made in the bottom stratum (Q, 2Q]: "
        f"{sum(1 for r in table if r[2] == r[4])} of {len(table)}")
    say(f"record position <= 10 Q: {sum(1 for r in table if r[3] <= 10 * r[1])} "
        f"of {len(table)}")

    # ------------------------------------------------- record by stratum + endpoints
    say()
    say("## The record by stratum, and the family of the record block's endpoints")
    say()
    for q, Q in [(5, 10000), (7, 10000), (11, 10000), (17, 10000), (37, 10000),
                 (5, 3162), (5, 1000)]:
        X = Q * Q
        primes = primes_upto(X)
        prQ = primes[primes <= Q]
        ps = [int(p) for p in prQ if p <= q]
        adm = admissible_by_sieve(q, Q, X, prQ)
        op = open_pairs(adm)
        pos, ln = gap_report(op, Q + 1, len(op))
        say(f"### q = {q}, Q = {Q:,}")
        say()
        say("| stratum | length | open pairs | record there | at | at/Q |")
        say("|---|---|---|---|---|---|")
        k = 0
        while (Q << k) < X:
            lo, hi = Q << k, min(Q << (k + 1), X)
            m = (pos >= lo) & (pos < hi)
            if m.any():
                j = int(np.argmax(ln[m]))
                say(f"| ({lo:,}, {hi:,}] | {hi - lo:,} | {int(op[lo:hi].sum()):,} | "
                    f"{int(ln[m][j])} | {int(pos[m][j]):,} | {int(pos[m][j]) / Q:.2f} |")
            k += 1
        i = int(np.argmax(ln))
        rec, at = int(ln[i]), int(pos[i])
        lo_pair, hi_pair = at - 1, at + rec
        lab = []
        for n in (lo_pair, hi_pair):
            a, ra = cof(n, ps)
            b, rb = cof(n + 2, ps)
            lab.append((a, b, "prime" if ra > 1 else "smooth",
                        "prime" if rb > 1 else "smooth"))
        say()
        say(f"record {rec} at {at:,} ({at / Q:.2f} Q); lower endpoint open pair {lo_pair:,} "
            f"family (s,s') = ({lab[0][0]}, {lab[0][1]}), upper endpoint {hi_pair:,} "
            f"family ({lab[1][0]}, {lab[1][1]})")
        # the ten largest gaps and their endpoint families
        order = np.argsort(-ln)[:10]
        say()
        say("| rank | gap | at | at/Q | lower endpoint family | upper endpoint family |")
        say("|---|---|---|---|---|---|")
        for r, j in enumerate(order, 1):
            a0, b0 = int(pos[j]) - 1, int(pos[j]) + int(ln[j])
            f0 = (cof(a0, ps)[0], cof(a0 + 2, ps)[0])
            f1 = (cof(b0, ps)[0], cof(b0 + 2, ps)[0])
            say(f"| {r} | {int(ln[j])} | {int(pos[j]):,} | {int(pos[j]) / Q:.2f} | "
                f"{f0} | {f1} |")
        say()
        # the gap spectrum
        c = Counter(ln.tolist())
        tot = len(ln)
        say(f"gap spectrum: {tot:,} gaps, lengths 1 to {int(ln.max())}; "
            f"most common {c.most_common(6)}")
        say("small-length spectrum (length = struck pairs; distance = length + 1):")
        say("| length | " + " | ".join(str(i) for i in range(1, 11)) + " |")
        say("|---|" + "---|" * 10)
        say("| count | " + " | ".join(f"{c.get(i, 0):,}" for i in range(1, 11)) + " |")
        say(f"L4 (distance 4 impossible, i.e. length 3): count = **{c.get(3, 0)}**")
        say()
        del adm, op

    with open(os.path.join(RES, "s3_gaps.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
