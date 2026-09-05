"""Branch 5d.ii.i.a, item 2: the DOMINO-MATCHING IDENTITY, verified exactly.

At level L (a run of L consecutive struck columns 0..L-1) split a gear set G:

    small(L) = {g : g - a_g <= L-1}   (the long arc fits; the gear is a periodic tiler)
    big(L)   = {g : g - a_g >= L}     (the long arc does not fit; a bare domino of size a_g)

MATCHING FORM.  G covers L columns iff some phase assignment of small(L) leaves a hole
set H that big(L) can finish, where each big gear may take

    * one hole on its own (ALWAYS possible: a gear of arc a covers hole i by the
      singleton {i} when i < a, and otherwise by the pair {i-a, i} whose left column
      i - a >= 0 is already covered), or
    * two holes at distance exactly a_g.

So the condition on H is purely combinatorial:

    maxpairs(H, arcs(big)) >= |H| - |big|,

maxpairs = the largest number of disjoint hole pairs whose distances form a
sub-multiset of the big gears' arcs.

The brief's caveat - a big gear placing its two strikes at distance g - a_g instead - is
VOID: g - a_g >= L > L - 1 is the definition of big, so the long arc never fits in the run.

Usage: uv run python research/anchor235/r50/arc_match.py
"""
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import arc, F_of, coverable, RESULTS  # noqa: E402


def split(G, L):
    small = [g for g in G if g - arc(g) <= L - 1]
    big = [g for g in G if g - arc(g) >= L]
    return small, big


def maxpairs(H, bigarcs):
    """Largest number of disjoint pairs of holes whose distances are a sub-multiset
    of bigarcs."""
    H = sorted(H)
    n = len(H)
    from collections import Counter
    best = [0]

    def rec(i, used, avail, cur):
        if cur + (n - bin(used).count("1") - i) // 2 <= best[0]:
            pass
        if i >= n:
            best[0] = max(best[0], cur)
            return
        if used >> i & 1:
            rec(i + 1, used, avail, cur)
            return
        # option: leave H[i] unpaired
        rec(i + 1, used | (1 << i), avail, cur)
        for j in range(i + 1, n):
            if used >> j & 1:
                continue
            d = H[j] - H[i]
            if avail.get(d, 0) > 0:
                avail[d] -= 1
                rec(i + 1, used | (1 << i) | (1 << j), avail, cur + 1)
                avail[d] += 1

    rec(0, 0, Counter(bigarcs), 0)
    return best[0]


def matching_form(L, G, phase_cap=4_000_000):
    """The matching-form answer, or None if the phase enumeration is too big."""
    if L <= 0:
        return True
    small, big = split(G, L)
    if not big:
        # nothing to match: the matching form IS the direct search, no information
        return None
    prod = 1
    for g in small:
        prod *= g
    if prod > phase_cap:
        return None
    bigarcs = [arc(g) for g in big]
    full = (1 << L) - 1
    masks = []
    for g in small:
        d = pow(3, -1, g)
        ms = []
        for o in range(g):
            m = 0
            for i in range(o, L, g):
                m |= 1 << i
            for i in range((o + d) % g, L, g):
                m |= 1 << i
            ms.append(m & full)
        masks.append(sorted(set(ms)))
    nb = len(big)
    for combo in itertools.product(*masks):
        cov = 0
        for m in combo:
            cov |= m
        holes = [i for i in range(L) if not (cov >> i & 1)]
        if len(holes) == 0:
            return True
        if len(holes) > 2 * nb:
            continue
        if len(holes) <= nb:
            return True
        if maxpairs(holes, bigarcs) >= len(holes) - nb:
            return True
    return False


def main():
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, "arc_match.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    SETS = [
        ("A(4) opt", [5, 7, 11, 17]),
        ("A(4) opt", [5, 7, 11, 19]),
        ("A(5) opt", [5, 7, 11, 23, 29]),
        ("A(5) opt", [5, 7, 11, 23, 31]),
        ("A(6) opt", [5, 7, 11, 17, 23, 37]),
        ("A(6) opt", [5, 7, 11, 13, 19, 47]),
        ("m11", [5, 7, 11]),
        ("m13", [5, 7, 11, 13]),
        ("m17", [5, 7, 11, 13, 17]),
        ("m19", [5, 7, 11, 13, 17, 19]),
        ("init K=4", [5, 7, 11, 13]),
        ("detwinned K=4", [5, 11, 17, 23]),
        ("detwinned K=5", [5, 11, 17, 23, 29]),
        ("big-heavy", [5, 7, 29, 31, 41]),
        ("big-heavy", [5, 7, 11, 41, 43, 47]),
    ]
    say("set                 gears                 F   L range tested  "
        "instances  disagreements  levels with big part")
    total = 0
    disagree = []
    for name, G in SETS:
        F = F_of(G)
        n_ok = 0
        n_big = 0
        skipped = 0
        for L in range(1, F + 1):
            direct = coverable(L, G)
            mf = matching_form(L, G)
            if mf is None:
                skipped += 1
                continue
            small, big = split(G, L)
            if big:
                n_big += 1
            total += 1
            if direct != mf:
                disagree.append((name, tuple(G), L, direct, mf))
            else:
                n_ok += 1
        say(f"{name:18s} {str(G):22s} {F:3d}  1..{F:<3d}  {n_ok:6d}     "
            f"{len(disagree):3d}          {n_big:3d}   (skipped {skipped})")
    say("")
    say(f"TOTAL instances checked: {total};  disagreements: {len(disagree)}")
    for d in disagree[:20]:
        say("   DISAGREE " + str(d))

    # The waste caveat, stated as a check: is there any gear set, level and big gear
    # whose long arc fits inside the run?  By definition of big there cannot be.
    bad = 0
    for _n, G in SETS:
        for L in range(1, F_of(G) + 1):
            for g in split(G, L)[1]:
                if g - arc(g) <= L - 1:
                    bad += 1
    say(f"big gears whose LONG arc fits inside the run: {bad}  (the brief's caveat)")

    json.dump({"total": total, "disagreements": disagree},
              open(os.path.join(RESULTS, "arc_match.json"), "w"))
    log.close()


if __name__ == "__main__":
    main()
