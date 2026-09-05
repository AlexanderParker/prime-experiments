"""Branch 5d.ii.i (prover, round 48): the MECHANISM behind the adversarial ladder.

The lattice showed that the best K-gear sub-machine is NOT the K smallest gears.
This script asks what decides it.

  A. The arc table.  a_g = (g -+ 1)/3 is gear g's short arc; two gears share a short
     arc iff they are 3a - 1 and 3a + 1, i.e. iff they are a TWIN PRIME PAIR.  So the
     number of distinct arcs among the gears of {5..q} is (gears) - (twin pairs).
  B. The one-gear sweep: F({5,7,11,g}) and F({5,7,11,17,g}) over every prime g, so the
     choice can be read against g and against a_g.
  C. Witness covers: the actual tiling of the longest blocked run for the winning sets
     and for the initial segments of the same size, with the waste (strikes - columns).
  D. Over the whole m31 lattice: max F at fixed K among subsets with all short arcs
     DISTINCT against subsets with a repeated arc.

Usage: uv run python research/anchor235/r48/mech.py
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cover_core import F_of, arcs  # noqa: E402

OUT = os.path.join(HERE, "results")


def primes_upto(n):
    b = [True] * (n + 1)
    b[0] = b[1] = False
    for i in range(2, int(n ** .5) + 1):
        if b[i]:
            b[i * i::i] = [False] * len(b[i * i::i])
    return [i for i in range(2, n + 1) if b[i]]


def witness(L, gears):
    """A cover of columns 0..L-1: {gear: sorted list of columns it strikes}, or None."""
    gears = sorted(gears)
    full = (1 << L) - 1
    masks, dsep = [], []
    for g in gears:
        d = pow(3, -1, g)
        dsep.append(d)
        ms = []
        for o in range(g):
            m = 0
            for i in range(o, L, g):
                m |= 1 << i
            for i in range((o + d) % g, L, g):
                m |= 1 << i
            ms.append(m)
        masks.append(ms)
    sol = {}

    def rec(covered, avail):
        if covered == full:
            return True
        u = ~covered & full
        pos = (u & -u).bit_length() - 1
        a = avail
        while a:
            b = a & -a
            i = b.bit_length() - 1
            a ^= b
            g, d = gears[i], dsep[i]
            for o in {pos % g, (pos - d) % g}:
                sol[g] = o
                if rec(covered | masks[i][o], avail ^ b):
                    return True
                del sol[g]
        return False

    if not rec(0, (1 << len(gears)) - 1):
        return None
    out = {}
    for g, o in sol.items():
        d = pow(3, -1, g)
        cols = sorted({i for i in range(o % g, L, g)} |
                      {i for i in range((o + d) % g, L, g)})
        out[g] = cols
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "mechanism.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    pool = [p for p in primes_upto(199) if p >= 5]

    say("A. THE ARC TABLE.  a_g = (g -+ 1)/3; two gears share a short arc iff they are")
    say("   3a-1 and 3a+1, i.e. iff they are a twin prime pair.")
    bya = {}
    for g in pool:
        bya.setdefault(arcs(g)[1], []).append(g)
    say("   arc: gears  -> " + "; ".join(f"{a}:{v}" for a, v in sorted(bya.items())))
    dup = [v for v in bya.values() if len(v) > 1]
    say(f"   arcs carried by two gears (= twin pairs) up to 199: {dup}")
    say("   gears of {5..q}, distinct arcs, and the difference (twin pairs):")
    for q in [23, 31, 47, 61, 101, 199]:
        gs = [g for g in pool if g <= q]
        na = len({arcs(g)[1] for g in gs})
        say(f"     q={q:4d}  gears={len(gs):3d}  distinct arcs={na:3d}  "
            f"twin pairs={len(gs)-na:3d}")

    say("")
    say("B. ONE-GEAR SWEEP.  F({5,7,11,g}) and F({5,7,11,17,g}) against g and a_g.")
    say("   g    a_g   F({5,7,11,g})   F({5,7,11,17,g})")
    for g in pool:
        if g in (5, 7, 11):
            continue
        f4 = F_of([5, 7, 11, g])
        f5 = "-" if g == 17 else F_of([5, 7, 11, 17, g])
        say(f"  {g:4d} {arcs(g)[1]:5d} {f4:12d} {str(f5):>18}")
        if g > 101:
            break

    say("")
    say("C. WITNESS COVERS at the winning sets and at the initial segments.")
    for gs in ([5, 7, 11, 13], [5, 7, 11, 17], [5, 7, 11, 13, 17], [5, 7, 11, 23, 29],
               [5, 7, 11, 13, 17, 19], [5, 7, 11, 17, 23, 37]):
        F = F_of(gs)
        L = F - 1
        w = witness(L, gs)
        strikes = sum(len(v) for v in w.values())
        say(f"  gears {gs}  arcs {[arcs(g)[1] for g in gs]}  F={F}  "
            f"run L={L}  strikes={strikes}  waste={strikes - L}  "
            f"capacity sum(2/g)={sum(2 / g for g in gs):.3f}")
        for g in sorted(w):
            say(f"      gear {g:3d} (arcs {arcs(g)[1]},{arcs(g)[2]}): {w[g]}")

    say("")
    say("D. DISTINCT ARCS AGAINST REPEATED ARCS over the whole {5..31} lattice.")
    lat = json.load(open(os.path.join(OUT, "lattice.json")))
    rows = {}
    for key, F in lat.items():
        gs = [int(x) for x in key.split(",")]
        K = len(gs)
        aa = [arcs(g)[1] for g in gs]
        distinct = len(set(aa)) == len(aa)
        d = rows.setdefault(K, {True: (-1, None), False: (-1, None)})
        if F > d[distinct][0]:
            d[distinct] = (F, gs)
    say("   K   best F, arcs all distinct        best F, some arc repeated")
    for K in sorted(rows):
        a, b = rows[K][True], rows[K][False]
        say(f"  {K:2d}   {a[0]:4d}  {a[1]}"
            f"        {b[0]:4d}  {b[1]}")
    log.close()


if __name__ == "__main__":
    main()
