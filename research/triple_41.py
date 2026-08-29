"""Round 27 (constructor): THE TRIPLE INEQUALITY AT 41 -> 43, EXACTLY.

research/increment_law.py tests the manager's depth-3 reduction of the
increment law

    max over adjacent gap triples (g_L, w, g_R) of M with LEGAL middle w
    of  g_L + w + g_R   <=   F_2(M) + s_min(q')

exactly at every step with a full-period dictionary (11->13 .. 37->41).  At
41 -> 43 the only dictionary is Mechanic's TRANSFER SUPERSET, which is
inflated at arity >= 2 (R72: 12 of 12 sampled superset-YES arity-4 tuples
were CRT-refuted), so the superset row (144 against the budget 117) is a
phantom, not a violation.  This script decides the row.

METHOD, two filters then the decider, all exact:
  (0) the superset is a sound source of CANDIDATES: every realised triple is
      in it, so sweeping it downward by sum finds the true maximum.
  (1) MIRROR HALVING (Lateral, round 25): #occ(w) = #occ(reverse w), so only
      g_L <= g_R needs deciding.
  (2) PHASE SATURATION (Mechanic, round 26), free: if no translate of the
      prefix-sum set X = {0, g_L, g_L+w, g_L+w+g_R} fits inside the exposed
      set of some gear of M, the triple is unrealised with no search.
  (3) research/crt_dict.py decide_cover on the survivors - the exact
      realisability CSP, scan-free.
Sweeping DESCENDING by sum and stopping at the first realised triple gives
the exact maximum, with every larger candidate refuted.

Usage:
  .venv/Scripts/python.exe research/triple_41.py [--y 41] [--q 43]
                                                 [--floor 117] [--workers 6]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")

import crt_dict  # noqa: E402


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def gears_of(y):
    return [p for p in range(5, y + 1) if is_prime(p)]


def exposed(g):
    c = pow(6, -1, g)
    return frozenset(r for r in range(g) if r != c % g and r != (-c) % g)


def ps_refuted(X, gears):
    for g in gears:
        Eg = exposed(g)
        xs = {x % g for x in X}
        if len(xs) > g - 2:
            return g
        if not any(all((t + x) % g in Eg for x in xs) for t in range(g)):
            return g
    return None


def job(args):
    y, tup = args
    t0 = time.time()
    try:
        ok = crt_dict.realised(y, tup, node_budget=60_000_000)
        return tup, ok, None, time.time() - t0
    except Exception as e:                                  # Budget etc.
        return tup, None, type(e).__name__, time.time() - t0


def main():
    args = sys.argv[1:]

    def opt(name, dflt):
        return type(dflt)(args[args.index(name) + 1]) if name in args else dflt

    y = opt("--y", 41)
    q1 = opt("--q", 43)
    floor = opt("--floor", 117)
    top = opt("--top", 10 ** 6)     # resume a stopped descent at a level
    workers = opt("--workers", 6)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    gears = gears_of(y)

    src = os.path.join(DDIR, "gap_tuples_%d_4_transfer.csv" % y)
    arr = np.loadtxt(src, delimiter=",", skiprows=1, dtype=np.int64)
    T = np.unique(np.concatenate([arr[:, 0:3], arr[:, 1:4]]), axis=0)
    print("machine %d, q' = %d, letters (%d, %d), s_min = %d"
          % (y, q1, a, b, min(a, b)))
    print("superset triples: %d  (source %s)" % (len(T), os.path.basename(src)))

    r = T[:, 1] % q1
    kinds = [k for k in ("LITERAL", "PADDED")
             if "--kinds" not in args or k.lower()[:3] in
             args[args.index("--kinds") + 1]]
    for kind, mask in [kv for kv in
                       (("LITERAL", (r == a) | (r == b)), ("PADDED", r == 0))
                       if kv[0] in kinds]:
        sel = T[mask]
        sel = sel[sel[:, 0] <= sel[:, 2]]              # mirror halving
        s = sel.sum(axis=1)
        order = np.argsort(-s)
        sel, s = sel[order], s[order]
        cand = [(int(s[i]), tuple(int(v) for v in sel[i]))
                for i in range(len(sel)) if floor < s[i] <= top]
        print("\n=== %s middles: %d mirror-halved candidates with sum > %d "
              "(superset max %d)" % (kind, len(cand), floor, int(s.max())))
        # phase-saturation prefilter
        live, killed = [], 0
        for tot, t in cand:
            X = [0, t[0], t[0] + t[1], tot]
            if ps_refuted(X, gears) is not None:
                killed += 1
            else:
                live.append((tot, t))
        print("    phase saturation refutes %d of %d for free; %d go to CRT"
              % (killed, len(cand), len(live)))
        if not live:
            print("    => every candidate above %d is REFUTED; max <= %d"
                  % (floor, floor))
            continue
        t0 = time.time()
        found = None
        done = 0
        undecided = []
        # decide in descending-sum blocks so the sweep can stop early
        blocks = {}
        for tot, t in live:
            blocks.setdefault(tot, []).append(t)
        with Pool(workers) as pool:
            for tot in sorted(blocks, reverse=True):
                res = list(pool.imap_unordered(
                    job, [(y, t) for t in blocks[tot]], chunksize=1))
                done += len(res)
                yes = [t for t, ok, err, dt in res if ok]
                und = [t for t, ok, err, dt in res if ok is None]
                undecided += und
                worst = max(dt for t, ok, err, dt in res)
                print("      sum %3d : %3d tuples, %d realised, %d undecided,"
                      " worst %.1f s   [%.0f s elapsed]"
                      % (tot, len(res), len(yes), len(und), worst,
                         time.time() - t0))
                if yes:
                    found = (tot, yes[0])
                    break
        if found:
            print("    => MAX = %d, witness %s (every larger candidate "
                  "refuted)" % found)
        else:
            print("    => all %d decided candidates above %d REFUTED "
                  "(undecided: %d) => max <= %d"
                  % (done, floor, len(undecided), floor))
        print("    CRT time %.0f s on %d workers" % (time.time() - t0, workers))


if __name__ == "__main__":
    main()
