"""Branch 5d.ii.i.a, item 3 (the twin tax) and item 6 (the feedback reading).

(a) The ladder: at each rung q the real machine {5..q} with K = pi(q) - 2 gears, against
    the free adversary's A(K) and against the DE-TWINNED machine D(K).

    De-twinning, literally as the brief defines it: walk the gears upward; whenever a
    gear's short arc is already used (i.e. it is the larger member of a twin pair),
    replace it by the next prime whose arc is new.  Carried out, this is exactly
    D(K) = one gear per arc, the smallest prime realising each of the K smallest arcs:
    5, 11, 17, 23, 29, 37, 41, 47, 53, 59, 67, 71, 79, 83, 89 (arcs 2, 4, ..., 30).

(b) The feedback test: at FIXED gear count K, does F fall as the number of duplicated
    arcs (twin pairs) in the set rises?  Exhaustive over all K-subsets of a prime pool.

Usage: uv run python research/anchor235/r50/arc_tax.py
"""
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import arc, F_of, primes_upto, RESULTS  # noqa: E402

# the project's recorded F ladder, machine {5..q}: K = pi(q) - 2 gears
FLADDER = {2: 5, 3: 7, 4: 11, 5: 18, 6: 25, 7: 34, 8: 43, 9: 58, 10: 88, 11: 91,
           12: 103, 13: 118, 14: 145, 15: 161}
QOF = {2: 7, 3: 11, 4: 13, 5: 17, 6: 19, 7: 23, 8: 29, 9: 31, 10: 37, 11: 41,
       12: 43, 13: 47, 14: 53, 15: 59}


def detwin(gears):
    """Replace the larger member of every twin pair by the next prime with a new arc."""
    pool = [p for p in primes_upto(500) if p >= 5]
    out, used = [], set()
    for g in gears:
        a = arc(g)
        if a not in used and g not in out:
            out.append(g)
            used.add(a)
            continue
        top = out[-1] if out else 5
        for p in pool:
            if p > top and p not in out and arc(p) not in used:
                out.append(p)
                used.add(arc(p))
                break
    return out


def twin_pairs(gears):
    seen = defaultdict(int)
    for g in gears:
        seen[arc(g)] += 1
    return sum(v - 1 for v in seen.values())


def job(args):
    gs, lo = args
    return gs, F_of(list(gs), lo=lo)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, "arc_tax.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    A = {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28}
    jf = os.path.join(RESULTS, "arc_adv.json")
    if os.path.exists(jf):
        A.update({int(k): v for k, v in json.load(open(jf))["A"].items()})

    say("=== (a) the ladder: real machine, adversary, de-twinned machine ===")
    say("  K    q   pi_2  arcs  F_real   A(K)  F_real/A   D(K)                     "
        "F_D   F_real/F_D")
    rows = []
    for K in range(4, 13):
        q = QOF[K]
        real = [p for p in primes_upto(q) if p >= 5]
        tp = twin_pairs(real)
        D = detwin(real)
        t = time.time()
        try:
            FD = F_of(D)
        except (MemoryError, RecursionError):
            FD = None
        Freal = FLADDER[K]
        AK = A.get(K)
        say(f" {K:2d} {q:4d}   {tp:2d}   {K-tp:3d}  {Freal:5d}  "
            f"{(str(AK) if AK else '  ? '):>5s}  "
            f"{(f'{Freal/AK:6.3f}' if AK else '     ?'):>7s}  {str(D):24s} "
            f"{(str(FD) if FD else '?'):>4s}  "
            f"{(f'{Freal/FD:6.3f}' if FD else '     ?'):>8s}   [{time.time()-t:.0f}s]")
        rows.append({"K": K, "q": q, "pi2": tp, "F_real": Freal, "A": AK,
                     "D": D, "F_D": FD})

    say("")
    say("=== (b) F at fixed gear count, by number of duplicated arcs (twin pairs) ===")
    fam = {}
    for K, POOL in ((4, 47), (5, 47), (6, 47)):
        pool = [p for p in primes_upto(POOL) if p >= 5]
        subs = list(itertools.combinations(pool, K))
        t = time.time()
        by = defaultdict(list)
        with Pool(3) as pl:
            for gs, f in pl.imap_unordered(job, [(s, 1) for s in subs], chunksize=32):
                by[twin_pairs(gs)].append((f, gs))
        say(f"K={K}, pool primes 5..{POOL} ({len(pool)} gears, {len(subs)} subsets), "
            f"{time.time()-t:.0f}s")
        say("    dup arcs   sets     max F   argmax                    mean F")
        for d in sorted(by):
            v = by[d]
            mx = max(v)
            say(f"      {d:3d}    {len(v):6d}    {mx[0]:5d}   {str(list(mx[1])):24s} "
                f"{sum(x[0] for x in v)/len(v):7.2f}")
        fam[K] = {str(d): {"n": len(v), "max": max(v)[0],
                           "argmax": list(max(v)[1]),
                           "mean": sum(x[0] for x in v) / len(v)}
                  for d, v in by.items()}

    json.dump({"ladder": rows, "family": fam},
              open(os.path.join(RESULTS, "arc_tax.json"), "w"))
    log.close()


if __name__ == "__main__":
    main()
