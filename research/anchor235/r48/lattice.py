"""Branch 5d.ii.i (prover, round 48), item 1 and item 3.

F(A) for EVERY non-empty subset A of the gears {5,7,11,13,17,19,23,29,31}, exact,
by the phase-covering search of cover_core.py (no period scan).  From the lattice:

  S_max^M(K) = max{F(A) : A subset of M's gears, |A| = K}   for every machine
               M = m11..m31 and every K;
  h_M(S)     = min{K : S_max^M(K) >= S}                      (the exact inverse);
  f_M(S)     = #{g in M : g - a_g < S + 2}                   (umbrella, forced);
  the argmax subsets at every K, so the question "are initial segments optimal?"
  is answered by exhibition.

Subsets are processed in increasing size; each one starts its search at the best
value of its own subsets one smaller (F is monotone under adding a gear), so only
the last, failing L is ever proved from scratch.

Usage: uv run python research/anchor235/r48/lattice.py
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

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31]
OUT = os.path.join(HERE, "results")
LADDER = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}


def job(arg):
    idxs, lo = arg
    gs = [GEARS[i] for i in idxs]
    t = time.time()
    return idxs, F_of(gs, lo=lo), time.time() - t


def main():
    os.makedirs(OUT, exist_ok=True)
    n = len(GEARS)
    F = {}
    log = open(os.path.join(OUT, "lattice.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say("gears " + " ".join(map(str, GEARS)))
    say("arcs (g: d_g, short a_g, long g-a_g): " +
        "; ".join(f"{g}: {arcs(g)[0]},{arcs(g)[1]},{arcs(g)[2]}" for g in GEARS))

    with Pool(4) as pool:
        for k in range(1, n + 1):
            subsets = list(itertools.combinations(range(n), k))
            args = []
            for s in subsets:
                lo = 1
                if k > 1:
                    lo = max(F[tuple(x for x in s if x != drop)] for drop in s)
                args.append((s, lo))
            t0 = time.time()
            for idxs, f, dt in pool.imap_unordered(job, args, chunksize=1):
                F[idxs] = f
            say(f"size {k}: {len(subsets)} subsets, {time.time()-t0:.1f}s")

    # gate: initial segments must reproduce the recorded ladder
    say("")
    say("GATE  initial segments against the recorded ladder")
    for k in range(1, n + 1):
        s = tuple(range(k))
        q = GEARS[k - 1]
        say(f"  F({{5..{q}}}) = {F[s]:4d}   record {LADDER[q]:4d}   "
            f"{'OK' if F[s] == LADDER[q] else 'MISMATCH'}")

    # per machine: S_max(K), the argmax subsets, h and f
    say("")
    for m in range(2, n + 1):                       # machines m7 .. m31
        q = GEARS[m - 1]
        gears = GEARS[:m]
        say(f"=== machine {{5..{q}}}  n = {m} gears, F = {F[tuple(range(m))]}")
        smax = {}
        best = {}
        for k in range(1, m + 1):
            bestf, bestsets = -1, []
            for s in itertools.combinations(range(m), k):
                f = F[s]
                if f > bestf:
                    bestf, bestsets = f, [s]
                elif f == bestf:
                    bestsets.append(s)
            smax[k] = bestf
            best[k] = bestsets
            init = tuple(range(k))
            tag = "initial segment optimal" if init in bestsets else "INITIAL SEGMENT BEATEN"
            witnesses = "; ".join("{" + ",".join(str(GEARS[i]) for i in s) + "}"
                                  for s in bestsets[:6])
            say(f"  K={k:2d}  S_max={bestf:4d}  F(init)={F[init]:4d}  {tag}"
                f"   argmax x{len(bestsets)}: {witnesses}"
                + ("..." if len(bestsets) > 6 else ""))
        # h and f as functions of the span
        Fmax = smax[m]
        rows = []
        for S in range(2, Fmax + 1):
            h = min((k for k in range(1, m + 1) if smax[k] >= S), default=None)
            f = sum(1 for g in gears if arcs(g)[2] < S + 2)   # long arc below S+2
            rows.append((S, h, f))
        say(f"  span S | h_M(S) | f_M(S)   (S = 2..F = {Fmax})")
        # only print the spans where h or f changes
        prev = None
        for S, h, f in rows:
            if (h, f) != prev:
                say(f"    S={S:4d}  h={h}  f={f}")
                prev = (h, f)
        say(f"    S={Fmax:4d}  h={rows[-1][1]}  f={rows[-1][2]}  (the record)")
        json.dump({"q": q, "n": m, "F": Fmax,
                   "S_max": smax,
                   "best": {k: [[GEARS[i] for i in s] for s in best[k]] for k in best},
                   "h_f": rows},
                  open(os.path.join(OUT, f"machine_{q}.json"), "w"))
    json.dump({",".join(str(GEARS[i]) for i in s): v for s, v in F.items()},
              open(os.path.join(OUT, "lattice.json"), "w"))
    log.close()


if __name__ == "__main__":
    main()
