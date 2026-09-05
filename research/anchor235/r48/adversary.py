"""Branch 5d.ii.i (prover, round 48), item 3 continued: the UNRESTRICTED adversarial
ladder A(K) = the longest span any K gears (any primes >= 5, each at its own fixed
separation d_g = 3^{-1} mod g, one phase per gear) can block.

The lattice inside {5..31} already refutes "initial segments are optimal", so A(K) is
NOT the F ladder read backwards.  This script measures A(K) properly:

  exhaustive over every K-subset of the pool of primes 5..POOL for K <= KEX;
  steepest-ascent swap search (from several starts) for KEX < K <= KMAX, which gives
  a LOWER bound on A(K) and the best set found.

Reported against the F ladder F({5..p_K}) and against the window W(q) = (q'^2-1)/6 at
the machine with K gears: the covering form of the root (the wall's W2) needs
A(pi(q)-2) < W(q).

Usage: uv run python research/anchor235/r48/adversary.py [KEX] [KMAX] [POOL]
"""
import itertools
import json
import os
import random
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cover_core import F_of, coverable, arcs  # noqa: E402

OUT = os.path.join(HERE, "results")
LADDER = {2: 5, 3: 7, 4: 11, 5: 18, 6: 25, 7: 34, 8: 43, 9: 58, 10: 88, 11: 91,
          12: 103, 13: 118, 14: 145, 15: 161}     # F({5..p_K}) by gear count K


def primes_upto(n):
    b = [True] * (n + 1)
    b[0] = b[1] = False
    for i in range(2, int(n ** .5) + 1):
        if b[i]:
            b[i * i::i] = [False] * len(b[i * i::i])
    return [i for i in range(2, n + 1) if b[i]]


def job(gs):
    return gs, F_of(gs, lo=1)


def better(gs, cur):
    """Is F(gs) > cur?  Only asks the single question, so it is cheap."""
    return coverable(cur, gs)


def main():
    KEX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    KMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    POOL = int(sys.argv[3]) if len(sys.argv) > 3 else 101
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, f"adversary_{KEX}_{KMAX}_{POOL}.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    pool = [p for p in primes_upto(POOL) if p >= 5]
    say(f"pool = primes 5..{POOL} ({len(pool)} gears): {pool}")
    say("short arcs: " + ", ".join(f"{g}:{arcs(g)[1]}" for g in pool))
    say("")
    A = {}
    best = {}
    with Pool(4) as pl:
        for K in range(1, KEX + 1):
            subs = list(itertools.combinations(pool, K))
            t = time.time()
            bf, bs = -1, []
            for gs, f in pl.imap_unordered(job, subs, chunksize=64):
                if f > bf:
                    bf, bs = f, [gs]
                elif f == bf:
                    bs.append(gs)
            A[K], best[K] = bf, bs
            init = tuple(pool[:K])
            say(f"K={K:2d}  A(K)={bf:4d}  F(init={init[-1]})={LADDER.get(K, 0):4d}  "
                f"ratio={bf / max(LADDER.get(K, 1), 1):5.2f}  "
                f"exhaustive over {len(subs)} subsets in {time.time()-t:.1f}s")
            say(f"        argmax x{len(bs)}: " +
                "; ".join("{" + ",".join(map(str, s)) + "}" for s in bs[:8]) +
                ("..." if len(bs) > 8 else ""))

    # steepest-ascent swaps for the larger K
    rng = random.Random(20480905)
    for K in range(KEX + 1, KMAX + 1):
        t = time.time()
        starts = [tuple(pool[:K])]
        if K - 1 in best:
            for s in best[K - 1][:4]:
                for extra in pool:
                    if extra not in s:
                        starts.append(tuple(sorted(s + (extra,))))
                        break
        for _ in range(6):
            starts.append(tuple(sorted(rng.sample(pool, K))))
        gbest, gset = -1, None
        for st in starts:
            cur = list(st)
            f = F_of(cur, lo=1)
            improved = True
            while improved:
                improved = False
                for i in range(K):
                    for g in pool:
                        if g in cur:
                            continue
                        cand = sorted(cur[:i] + cur[i + 1:] + [g])
                        if coverable(f, cand):          # F(cand) > f
                            cur = cand
                            f = F_of(cand, lo=f + 1)
                            improved = True
                            break
                    if improved:
                        break
            if f > gbest:
                gbest, gset = f, tuple(cur)
        A[K], best[K] = gbest, [gset]
        say(f"K={K:2d}  A(K)>={gbest:4d}  F(init)={LADDER.get(K, 0):4d}  "
            f"ratio>={gbest / max(LADDER.get(K, 1), 1):5.2f}  "
            f"swap search, {len(starts)} starts, {time.time()-t:.1f}s")
        say("        best set: {" + ",".join(map(str, gset)) + "}")

    say("")
    say("The covering form of the root (W2): A(pi(q)-2) against W(q) = (q'^2-1)/6")
    say("  K   q     q'     W(q)     A(K)   A/W    F({5..q})  F/W")
    ps = primes_upto(400)
    for K in sorted(A):
        idx = K + 1                                   # p_1 = 5 is primes[2]
        if idx + 1 >= len(ps):
            break
        q, qn = ps[idx], ps[idx + 1]
        W = (qn * qn - 1) // 6
        Fq = LADDER.get(K, 0)
        say(f"  {K:2d} {q:5d} {qn:5d} {W:8d}  {A[K]:6d}  {A[K]/W:5.3f}   "
            f"{Fq:6d}   {Fq/W if Fq else 0:5.3f}")
    json.dump({"A": A, "best": {k: [list(s) for s in v] for k, v in best.items()}},
              open(os.path.join(OUT, f"adversary_{KEX}_{KMAX}_{POOL}.json"), "w"))
    log.close()


if __name__ == "__main__":
    main()
