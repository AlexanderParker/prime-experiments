"""Item 4 (the walk) and item 5 (the owner's two numbers).

  Z12  the walk in the zone as a closed form:
         nextadm(x) = min ( least q-smooth >= x,
                            min over q-smooth s of s * nextprime(max(Q, ceil(x/s))) )
       and the next open pair by iterating it.  Checked against the sieve, exhaustively.
  Z13  Q - sqrt(Q): is anything there?
  Z14  g0 = the first gear with g0^2 > Q, and g0^2: is anything there?
       plus the striker composition (how many gears strike, and how big) by height.

usage: uv run python research/topmachine/r6/s4_walk.py
"""

import os
import sys
from math import isqrt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (admissible_by_sieve, open_pairs, primes_upto,  # noqa: E402
                    smooth_numbers)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def nextadm_formula(xs, q, Q, primes, sm_all, sm_cof):
    """The least admissible n >= x, for every x in xs, from the rule alone (no scan).

    sm_all: every q-smooth number in range (the P = 1 part).
    sm_cof: the q-smooth numbers that can be a cofactor, s * (Q + 1) <= X."""
    big = primes[primes > Q]
    best = np.full(len(xs), np.iinfo(np.int64).max, dtype=np.int64)
    smarr = np.asarray(sm_all, dtype=np.int64)
    j = np.searchsorted(smarr, xs, side="left")
    ok = j < len(smarr)
    best[ok] = np.minimum(best[ok], smarr[j[ok]])
    for s in sm_cof:
        need = np.maximum(-(-xs // s), Q + 1)    # ceil(x / s), at least Q + 1
        k = np.searchsorted(big, need, side="left")
        ok = k < len(big)
        if not ok.any():
            continue
        cand = s * big[np.minimum(k, len(big) - 1)]
        best[ok] = np.minimum(best[ok], cand[ok])
    return best


def main():
    say("# s4 THE WALK IN THE ZONE, and the owner's two numbers")
    say()
    say("## Z12  the next admissible number from the rule, against the sieve")
    say()
    say("| q | Q | x range tested | positions | mismatches | terms in the min |")
    say("|---|---|---|---|---|---|")

    for q, Q in [(5, 1000), (11, 1000), (17, 1000), (5, 3162), (11, 3162)]:
        X = Q * Q
        primes = primes_upto(X)
        prQ = primes[primes <= Q]
        adm = admissible_by_sieve(q, Q, X, prQ)
        hi = X // 2
        # true next admissible >= x, by a backward scan (the thing the formula replaces)
        nxt = np.full(X + 2, X + 1, dtype=np.int64)
        idx = np.flatnonzero(adm).astype(np.int64)
        nxt[idx] = idx
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
        xs = np.arange(Q + 1, hi + 1, dtype=np.int64)
        sm_all = smooth_numbers(q, X)
        sm_cof = [s for s in sm_all if s * (Q + 1) <= X]
        got = nextadm_formula(xs, q, Q, primes, sm_all, sm_cof)
        mism = int((got != nxt[xs]).sum())
        say(f"| {q} | {Q} | ({Q}, {hi:,}] | {len(xs):,} | **{mism}** | "
            f"{len(sm_cof)} + 1 |")
        del adm, nxt

    # ------------------------------------------------ the pair walk from the rule
    say()
    say("## Z12  the next OPEN PAIR from the rule, by iterating nextadm")
    say()
    say("| q | Q | positions tested | mismatches | mean iterations | max iterations |")
    say("|---|---|---|---|---|---|")
    for q, Q in [(5, 1000), (11, 1000), (5, 3162)]:
        X = Q * Q
        primes = primes_upto(X)
        prQ = primes[primes <= Q]
        adm = admissible_by_sieve(q, Q, X, prQ)
        op = open_pairs(adm)
        nxtop = np.full(X + 2, X + 1, dtype=np.int64)
        idx = np.flatnonzero(op).astype(np.int64)
        nxtop[idx] = idx
        nxtop = np.minimum.accumulate(nxtop[::-1])[::-1]
        sm_all = smooth_numbers(q, X)
        sm_cof = [s for s in sm_all if s * (Q + 1) <= X]
        rng = np.random.default_rng(7)
        xs = rng.integers(Q + 1, X // 2, size=200000, dtype=np.int64)
        cur = xs.copy()
        ans = np.zeros(len(xs), dtype=np.int64)
        iters = np.zeros(len(xs), dtype=np.int64)
        live = np.arange(len(xs))
        while len(live):
            iters[live] += 1
            y = nextadm_formula(cur[live], q, Q, primes, sm_all, sm_cof)
            z = nextadm_formula(y + 2, q, Q, primes, sm_all, sm_cof)
            done = z == y + 2
            ans[live[done]] = y[done]
            cur[live[~done]] = y[~done] + 1
            live = live[~done]
        mism = int((ans != nxtop[xs]).sum())
        say(f"| {q} | {Q} | {len(xs):,} | **{mism}** | {iters.mean():.2f} | "
            f"{int(iters.max())} |")
        del adm, op, nxtop

    # ------------------------------------------------------ Z13 / Z14 the two numbers
    say()
    say("## Z13 / Z14  Q - sqrt(Q), g0 and g0^2: the profile through the whole range")
    say()
    q, Q = 5, 10000
    X = Q * Q
    primes = primes_upto(X)
    prQ = primes[primes <= Q]
    p1 = int(primes[primes > Q][0])
    g0 = int([g for g in prQ if g > q and g * g > Q][0])
    adm = admissible_by_sieve(q, Q, X, prQ)
    op = open_pairs(adm)
    say(f"q = {q}, Q = {Q:,}: Q - sqrt(Q) = {Q - isqrt(Q):,}, g0 = {g0}, "
        f"g0^2 = {g0 * g0:,}, p1 = nextprime(Q) = {p1:,}, largest smooth pair s(q) = 160")
    say()
    say("| window | open pairs | density | mean strikers per struck pair | "
        "largest striker | smallest striker |")
    say("|---|---|---|---|---|---|")
    wins = [(1, 200), (150, 350), (g0 * g0 - 500, g0 * g0 + 500),
            (Q - 3 * isqrt(Q), Q - isqrt(Q)), (Q - isqrt(Q), Q),
            (Q, p1), (p1, p1 + 1000), (2 * Q, 2 * Q + 1000),
            (10 * Q, 10 * Q + 1000), (X // 2, X // 2 + 1000)]
    for lo, hi in wins:
        lo, hi = int(lo), int(hi)
        n = hi - lo
        cnt = np.zeros(n, dtype=np.int32)
        big = np.zeros(n, dtype=np.int64)
        small = np.full(n, 0, dtype=np.int64)
        for g in prQ:
            g = int(g)
            if g <= q:
                continue
            for t in (0, -2):
                st = (t - lo) % g
                cnt[st::g] += 1
                big[st::g] = g
                sl = small[st::g]
                small[st::g] = np.where(sl == 0, g, sl)
        struck = cnt > 0
        nop = int(op[lo:hi].sum())
        say(f"| [{lo:,}, {hi:,}) | {nop} | {nop / n:.4f} | "
            f"{cnt[struck].mean():.2f} | {int(big[struck].max()) if struck.any() else 0} | "
            f"{int(small[struck].min()) if struck.any() else 0} |")

    say()
    say("## the record's mechanism either side of the candidate edges")
    say()
    say("open pairs in [1, Q]: the q-smooth pairs only (L46).  For q = 5, Q = 10,000 they are")
    sm = smooth_numbers(q, Q + 2)
    sset = set(sm)
    say("  " + ", ".join(str(n) for n in sm if n <= Q - 2 and n + 2 in sset))
    say()
    say(f"so [1, Q] carries {len([n for n in sm if n <= Q - 2 and n + 2 in sset])} open pairs, "
        f"all below 161, and the whole of (160, {Q:,}] is one struck block of "
        f"{Q - 2 - 160} pairs: Q - sqrt(Q) = {Q - isqrt(Q):,}, g0 = {g0} and "
        f"g0^2 = {g0 * g0:,} all lie strictly inside it.")

    with open(os.path.join(RES, "s4_walk.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
