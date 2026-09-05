"""Branch 5d.ii.i.a, item 1: A(K) exactly at K = 7 and 8, and the arc multisets of the
optimal sets at K = 4..8.

Exhaustive over ALL primes >= 5 by the type reduction of arc_core (a gear whose long arc
does not fit in the run is described by its short arc alone).

Usage: uv run python research/anchor235/r50/arc_adv.py [KMAX]
"""
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import Level, arc, is_prime, RESULTS  # noqa: E402

SEED = {4: 15, 5: 20, 6: 26, 7: 35, 8: 43, 9: 55}


def cover_multiset(L, sel):
    """Can this exact multiset of item types cover 0..L-1?  sel = list of (kind, key)."""
    lv = Level(L, len(sel))
    # rebuild the level restricted to the chosen multiset
    items, mult = [], []
    for it in sorted(set(sel)):
        items.append(it)
        mult.append(sel.count(it))
    lv.items = [(k, key, m) for (k, key), m in zip(items, mult)]
    lv.cap = []
    for kind, key, _m in lv.items:
        from arc_core import masks_for
        ms = masks_for(kind, key, L)
        lv.cap.append(max(bin(m).count("1") for m in ms))
    lv.K = len(sel)
    return lv.coverable()


def gears_of(kind, key, L):
    """Which real primes realise this item type at level L."""
    if kind == 'p':
        return [key]
    if kind == 'd':
        return [g for g in (3 * key - 1, 3 * key + 1)
                if g >= 5 and is_prime(g) and arc(g) == key and g - key >= L]
    return ['any prime with arc >= %d' % L]


def arcs_of(sel, L):
    out = []
    for kind, key in sel:
        out.append(arc(key) if kind == 'p' else (key if kind == 'd' else 999))
    return sorted(out)


def job(args):
    L, sel = args
    return sel, cover_multiset(L, sel)


def enumerate_optima(K, L, procs=4):
    """Every K-multiset of item types that covers L columns.

    Candidates are pruned by the exact capacity bound: an item can strike at most
    cap(item) of the L columns, so a covering multiset needs sum of caps >= L.
    """
    lv = Level(L, K)
    types = [(k, key, min(m, K)) for k, key, m in lv.items]
    cap = list(lv.cap)
    n = len(types)
    # suffix maximum capacity, for pruning
    sufmax = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        sufmax[i] = max(cap[i], sufmax[i + 1])
    combos = []

    def rec(i, chosen, tot, left):
        if left == 0:
            if tot >= L:
                combos.append(tuple(chosen))
            return
        if i == n:
            return
        if tot + sufmax[i] * left < L:
            return
        for take in range(min(types[i][2], left), -1, -1):
            rec(i + 1, chosen + [(types[i][0], types[i][1])] * take,
                tot + cap[i] * take, left - take)

    rec(0, [], 0, K)
    winners = []
    with Pool(procs) as pl:
        for sel, ok in pl.imap_unordered(job, [(L, list(s)) for s in combos],
                                         chunksize=200):
            if ok:
                winners.append(tuple(sel))
    return combos, winners


def main():
    KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, "arc_adv.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    A = {}
    for K in range(4, KMAX + 1):
        L = SEED.get(K, 1)
        t = time.time()
        while True:
            lv = Level(L, K)
            ok = lv.coverable()
            say(f"K={K} L={L}: {'cover' if ok else 'NO COVER'}  "
                f"{lv.nodes} nodes, {len(lv.items)} item types, {time.time()-t:.1f}s")
            if not ok:
                A[K] = L
                break
            L += 1
        say(f"*** A({K}) = {A[K]}  (longest blocked run {A[K]-1} columns), "
            f"{time.time()-t:.1f}s")

    say("")
    say("=== optimal sets and their arc multisets ===")
    OPT = {}
    for K in range(4, KMAX + 1):
        L = A[K] - 1
        t = time.time()
        combos, winners = enumerate_optima(K, L)
        say(f"K={K}: {len(winners)} optimal multisets out of {len(combos)} "
            f"candidates at L={L}, {time.time()-t:.1f}s")
        rows = []
        for w in sorted(winners, key=lambda s: (arcs_of(s, L), s)):
            am = arcs_of(w, L)
            gr = [gears_of(k, key, L) for k, key in w]
            rows.append({"items": [[k, key] for k, key in w], "arcs": am,
                         "gears": gr})
        OPT[K] = rows
        for r in rows[:40]:
            say("    arcs " + ",".join(map(str, r["arcs"])) + "   gears " +
                " ".join("/".join(map(str, g)) for g in r["gears"]))
        if len(rows) > 40:
            say(f"    ... and {len(rows)-40} more")
        dup = sum(1 for r in rows if len(set(r["arcs"])) < len(r["arcs"]))
        say(f"    optima with a repeated arc: {dup}/{len(rows)}")

    json.dump({"A": A, "opt": OPT}, open(os.path.join(RESULTS, "arc_adv.json"), "w"))
    log.close()


if __name__ == "__main__":
    main()
