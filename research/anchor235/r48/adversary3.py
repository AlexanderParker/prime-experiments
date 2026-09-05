"""Branch 5d.ii.i (prover, round 48), item 3: certified LOWER bounds on A(K), K >= 7.

A(K) >= L is proved by exhibiting one cover of L consecutive columns by K gears, so
every number here is a proof, never an estimate.  The search is a hill climb over gear
sets seeded with (i) the initial segment {5..p_K}, (ii) the best K-subsets of {5..31}
found by the exhaustive lattice, (iii) the best (K-1)-set plus one gear.  The
coverability test is capped in nodes, which can only lose covers, never invent one.

Usage: uv run python research/anchor235/r48/adversary3.py [POOL] [KMAX]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adversary2 import cover_capped, primes_upto  # noqa: E402

OUT = os.path.join(HERE, "results")
LADDER = {7: 34, 8: 43, 9: 58, 10: 88, 11: 91, 12: 103, 13: 118, 14: 145, 15: 161}
# best K-subsets of {5..31} from lattice.py (exact inside that machine)
LATTICE = {7: [5, 7, 11, 13, 17, 19, 31], 8: [5, 7, 11, 13, 17, 19, 29, 31],
           9: [5, 7, 11, 13, 17, 19, 23, 29, 31]}
SEED6 = [5, 7, 11, 17, 23, 37]


def climb(seed, pool, L0, cap):
    cur, L = sorted(seed), L0
    while True:
        ok, _ = cover_capped(L + 1, cur, cap)
        if not ok:
            break
        L += 1
    improved = True
    while improved:
        improved = False
        for i in range(len(cur)):
            for g in pool:
                if g in cur:
                    continue
                cand = sorted(cur[:i] + cur[i + 1:] + [g])
                ok, _ = cover_capped(L + 1, cand, cap)
                if ok:
                    cur, L, improved = cand, L + 1, True
                    break
            if improved:
                break
    return cur, L


def main():
    POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 71
    KMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    CAP = 150_000
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, f"adversary3_{POOL}_{KMAX}.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    pool = [p for p in primes_upto(POOL) if p >= 5]
    say(f"pool = primes 5..{POOL} ({len(pool)} gears); node cap {CAP}")
    A, best = {6: 28}, {6: SEED6}
    for K in range(7, KMAX + 1):
        t = time.time()
        seeds = [pool[:K]]
        if K in LATTICE:
            seeds.append(LATTICE[K])
        seeds.append(sorted(best[K - 1] + [g for g in pool
                                           if g not in best[K - 1]][:1]))
        seeds.append(sorted(best[K - 1] + [g for g in pool
                                           if g not in best[K - 1]][-1:]))
        bL, bset = -1, None
        for s in seeds:
            c, L = climb(s, pool, max(A[K - 1] - 1, 1), CAP)
            if L > bL:
                bL, bset = L, c
        A[K], best[K] = bL + 1, bset
        lad = LADDER.get(K, 0)
        say(f"K={K:2d}  A(K)>={bL+1:4d}  F ladder={lad:4d}  "
            f"ratio>={(bL+1)/lad if lad else 0:5.2f}  {time.time()-t:.1f}s  "
            "set {" + ",".join(map(str, bset)) + "}")
    json.dump({"A": A, "best": best},
              open(os.path.join(OUT, f"adversary3_{POOL}_{KMAX}.json"), "w"))

    say("")
    say("The covering form of the root (wall W2): A(K) at K = the gear count of {5..q}")
    say("against the window W(q) = (q'^2-1)/6 and the machine record F({5..q}).")
    say("   K    q     q'      W(q)    A(K)    A/W     F      F/W")
    ps = primes_upto(500)
    full = {2: 5, 3: 7, 4: 16, 5: 22, 6: 28}
    full.update(A)
    for K in sorted(full):
        q, qn = ps[K + 1], ps[K + 2]
        W = (qn * qn - 1) // 6
        Fq = {2: 5, 3: 7, 4: 11, 5: 18, 6: 25}.get(K, LADDER.get(K, 0))
        say(f"  {K:3d} {q:5d} {qn:5d} {W:9d} {full[K]:6d}  {full[K]/W:6.3f} {Fq:6d}  "
            f"{Fq/W if Fq else 0:6.3f}")
    log.close()


if __name__ == "__main__":
    main()
