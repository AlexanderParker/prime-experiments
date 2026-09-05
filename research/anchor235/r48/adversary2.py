"""Branch 5d.ii.i (prover, round 48), item 3: the adversarial ladder A(K), part two.

A(K) = the longest span any K gears (any primes >= 5, each with its own fixed
separation d_g = 3^{-1} mod g, one phase per gear) can block.

Two regimes, both rigorous:
  * K <= 6: EXHAUSTIVE over every K-subset of the primes 5..POOL, seeded with a lower
    bound so most subsets cost one refutation.  A(K) exact (relative to the pool).
  * K >= 7: LOWER BOUNDS only, certified by exhibiting a cover.  A cover of L columns
    by K gears is a positive certificate; the search is a hill-climb over gear sets,
    with the coverability test capped in nodes (a cap can only lose covers, never
    invent one), so every value printed is a proved A(K) >= L.

Usage: uv run python research/anchor235/r48/adversary2.py [POOL] [KMAX]
"""
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cover_core import F_of, arcs  # noqa: E402

OUT = os.path.join(HERE, "results")
LADDER = {2: 5, 3: 7, 4: 11, 5: 18, 6: 25, 7: 34, 8: 43, 9: 58, 10: 88, 11: 91,
          12: 103, 13: 118, 14: 145, 15: 161}
SEED = {4: 16, 5: 22, 6: 28, 7: 37, 8: 45, 9: 58}      # known lower bounds


def primes_upto(n):
    b = [True] * (n + 1)
    b[0] = b[1] = False
    for i in range(2, int(n ** .5) + 1):
        if b[i]:
            b[i * i::i] = [False] * len(b[i * i::i])
    return [i for i in range(2, n + 1) if b[i]]


def cover_capped(L, gears, cap):
    """(found, complete).  found=True means a cover EXISTS (a proof).
    complete=False means the node budget ran out with no cover found."""
    if L <= 0:
        return True, True
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
        del ms
    capn = [max(bin(m).count("1") for m in mm) for mm in masks]
    fail = set()
    budget = [cap]
    exhausted = [False]

    def rec(covered, avail):
        if covered == full:
            return True
        if budget[0] <= 0:
            exhausted[0] = True
            return False
        budget[0] -= 1
        key = (covered, avail)
        if key in fail:
            return False
        u = ~covered & full
        todo = bin(u).count("1")
        tot, a = 0, avail
        while a:
            b = a & -a
            tot += capn[b.bit_length() - 1]
            a ^= b
        if tot < todo:
            fail.add(key)
            return False
        pos = (u & -u).bit_length() - 1
        a = avail
        while a:
            b = a & -a
            i = b.bit_length() - 1
            a ^= b
            g, d = gears[i], dsep[i]
            for o in {pos % g, (pos - d) % g}:
                if rec(covered | masks[i][o], avail ^ b):
                    return True
        fail.add(key)
        return False

    ok = rec(0, (1 << len(gears)) - 1)
    return ok, not exhausted[0]


def job(arg):
    gs, lo = arg
    return gs, F_of(gs, lo=lo)


def main():
    POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 149
    KMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    CAP = 400_000
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, f"adversary2_{POOL}_{KMAX}.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    pool = [p for p in primes_upto(POOL) if p >= 5]
    say(f"pool = primes 5..{POOL} ({len(pool)} gears)")
    A, best = {}, {}

    with Pool(4) as pl:
        for K in (4, 5, 6):
            lo = SEED.get(K, 1)
            subs = [(s, lo) for s in itertools.combinations(pool, K)]
            t = time.time()
            bf, bs = lo, []
            for gs, f in pl.imap_unordered(job, subs, chunksize=64):
                if f > bf:
                    bf, bs = f, [gs]
                elif f == bf:
                    bs.append(gs)
            A[K], best[K] = bf, bs
            say(f"K={K:2d}  A(K)={bf:4d}  F ladder={LADDER[K]:4d}  "
                f"ratio={bf/LADDER[K]:5.2f}  EXACT over {len(subs)} subsets, "
                f"{time.time()-t:.1f}s")
            say("        argmax x%d: " % len(bs) +
                "; ".join("{" + ",".join(map(str, s)) + "}" for s in bs[:6]))

    # K >= 7: certified lower bounds by hill-climbing, positive certificates only
    cur = list(best[6][0]) if best.get(6) else [5, 7, 11, 17, 23, 37]
    curL = A[6] - 1
    for K in range(7, KMAX + 1):
        t = time.time()
        # add the gear that lets the run grow most
        bestset, bestL = None, curL
        for g in pool:
            if g in cur:
                continue
            cand = sorted(cur + [g])
            L = bestL
            while True:
                ok, _ = cover_capped(L + 1, cand, CAP)
                if not ok:
                    break
                L += 1
            if L > bestL:
                bestL, bestset = L, cand
        if bestset is None:
            bestset = sorted(cur + [next(g for g in pool if g not in cur)])
        cur = bestset
        curL = bestL
        # hill-climb by swapping one gear
        improved = True
        while improved:
            improved = False
            for i in range(K):
                for g in pool:
                    if g in cur:
                        continue
                    cand = sorted(cur[:i] + cur[i + 1:] + [g])
                    ok, _ = cover_capped(curL + 1, cand, CAP)
                    if ok:
                        cur, curL = cand, curL + 1
                        improved = True
                        break
                if improved:
                    break
        A[K] = curL + 1
        best[K] = [tuple(cur)]
        lad = LADDER.get(K, 0)
        say(f"K={K:2d}  A(K)>={curL+1:4d}  F ladder={lad:4d}  "
            f"ratio>={(curL+1)/lad if lad else 0:5.2f}  hill climb, "
            f"{time.time()-t:.1f}s   set {{" + ",".join(map(str, cur)) + "}")

    say("")
    say("The covering form of the root (wall W2): A(K) at K = the gear count of {5..q},")
    say("against the window W(q) = (q'^2-1)/6 and against the machine record F({5..q}).")
    say("   K    q     q'      W(q)     A(K)    A/W     F      F/W")
    ps = primes_upto(500)
    for K in sorted(A):
        q, qn = ps[K + 1], ps[K + 2]
        W = (qn * qn - 1) // 6
        Fq = LADDER.get(K, 0)
        say(f"  {K:3d} {q:5d} {qn:5d} {W:9d} {A[K]:7d}  {A[K]/W:6.3f} {Fq:6d}  "
            f"{Fq/W if Fq else 0:6.3f}")
    json.dump({"A": A, "best": {k: [list(s) for s in v] for k, v in best.items()}},
              open(os.path.join(OUT, f"adversary2_{POOL}_{KMAX}.json"), "w"))
    log.close()


if __name__ == "__main__":
    main()
