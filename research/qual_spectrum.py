"""Round 26 (constructor), CROSS-LANE: THE QUALIFYING SPECTRUM OF MACHINE 37
AT FLOOR 14 - the one input Formalist's eighth rung is blocked on.

Formalist's stratified-dictionary vehicle consumes F_2(M) and the QUALIFYING
SPECTRUM Q_J(M; v) = max span of a J-gap window of M whose J-2 MIDDLE gaps
are all >= v.  At the eighth rung the machine is M = 37, the added gear is
q' = 41 and the floor is v = 2u'(41) = 14.  A 1.24e12-slot scan is out; this
computes it from the realised-tuple dictionary instead.

TWO LAYERS OF ANSWER, both sound, stated separately because they have
different strengths.

  (1) UPPER BOUNDS AT EVERY DEPTH, from one closure.  Build A_4 over machine
      M's realised 3- and 4-tuples with the size floor as the edge predicate
      (the middle gap of the edge must be >= v).  Every real qualifying
      J-window maps to a walk of J-2 edges with the same weight, so
          Q_J(M; v)  <=  layer_{J-2} of that closure,
      and the closure TERMINATES, which caps the depth rigorously: no walk of
      length k means Q_{k+2} = 0.  This is exactly the vehicle R49 uses, with
      one predicate swapped, and it needs no enumeration at all.

  (2) EXACT VALUES, by descending-span search seeded at (1)'s bound, over
      A_4 walks (branch and bound with the layer potentials as an admissible
      heuristic), each candidate window then DECIDED by the realisability
      oracle - the exact full-period census at arity <= 4, the scan-free CRT
      set-cover decider deeper.

HONEST SIZING (the lane asked for this explicitly).  Costs measured at m37 in
round 25: an arity-2 refutation is 5.8 s, an arity-4 refutation 43 ms, and
DEEPER tuples are CHEAPER, not dearer - more open points means smaller gear
domains.  The queries here are arity 5..7, the cheap end.  What is expensive
is the ENUMERATION, not the oracle, and that is what the branch and bound
with an admissible heuristic is for.

Usage:  python research/qual_spectrum.py 37 --floor 14 --depth 7
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_a4                                          # noqa: E402
import chain_dict_oracle                                 # noqa: E402
import crt_dict                                          # noqa: E402

NEG = -(1 << 40)


def floor_closure(D, floor, maxlay=16):
    """A_4 with the SIZE-FLOOR edge predicate.  Returns (layers, h_k table,
    graph) where layers[k] is a sound upper bound on Q_{k+2}(M; floor)."""
    d3 = sorted(D[3])
    idx = {t: i for i, t in enumerate(d3)}
    S = len(d3)
    esrc, edst, ew = [], [], []
    for t in sorted(D[4]):
        if t[2] < floor:                 # the gap this edge consumes
            continue
        i = idx.get(t[:3])
        j = idx.get(t[1:])
        if i is None or j is None:
            continue
        esrc.append(i)
        edst.append(j)
        ew.append(t[2])
    esrc = np.array(esrc, np.int64)
    edst = np.array(edst, np.int64)
    ew = np.array(ew, np.int64)
    Rs = np.array([t[-1] for t in d3], np.int64)
    Ls = np.array([t[-2] for t in d3], np.int64)
    lay, hs = [], []
    cur = Rs.copy()
    for _ in range(maxlay):
        hs.append(cur.copy())
        lay.append(int((Ls + cur).max()))
        nxt = np.full(S, NEG, np.int64)
        if len(esrc):
            np.maximum.at(nxt, esrc, ew + cur[edst])
        if nxt.max() <= NEG // 2:
            break
        cur = nxt
    return lay, hs, (d3, idx, S, esrc, edst, ew, Rs, Ls)


def exact_depth(y, D, floor, J, upper, hs, G, oracle, nodecap=4_000_000):
    """Exact Q_J by branch and bound over A_4 walks of J-2 edges."""
    d3, idx, S, esrc, edst, ew, Rs, Ls = G
    if J - 2 >= len(hs):
        return 0, None
    order = np.argsort(esrc, kind="stable")
    es, ed, ww = esrc[order], edst[order], ew[order]
    usrc, starts = np.unique(es, return_index=True)
    ends = np.append(starts[1:], len(es))
    where = {int(s): i for i, s in enumerate(usrc.tolist())}
    k = J - 2
    best = [0, None]
    nodes = [0]

    def rec(st, depth, span, path):
        nodes[0] += 1
        if nodes[0] > nodecap:
            raise RuntimeError("enumeration budget")
        if depth == k:
            tot = span + int(Rs[st])
            if tot <= best[0]:
                return
            win = tuple([int(Ls[path[0]])] + [int(w) for w in path[1]] +
                        [int(Rs[st])])
            if oracle(win):
                best[0], best[1] = tot, win
            return
        # admissible bound: the best completion from st in k-depth steps
        h = hs[k - depth]
        if int(h[st]) <= NEG // 2:
            return
        if span + int(h[st]) <= best[0]:
            return
        gi = where.get(st)
        if gi is None:
            return
        lo, hi = starts[gi], ends[gi]
        cand = sorted(range(lo, hi), key=lambda e: -(int(ww[e]) +
                                                     int(hs[k - depth - 1]
                                                         [ed[e]])))
        for e in cand:
            rec(int(ed[e]), depth + 1, span + int(ww[e]),
                (path[0], path[1] + [int(ww[e])]))

    for st in range(S):
        if int(Ls[st]) <= NEG // 2:
            continue
        if int(Ls[st]) + int(hs[k][st]) <= best[0]:
            continue
        rec(st, 0, 0, (st, []))
    return best[0], best[1]


class Oracle:
    def __init__(self, y, D):
        self.y, self.D = y, D
        self.n = self.ncrt = 0
        self.secs = 0.0

    def __call__(self, t):
        self.n += 1
        if len(t) in self.D:
            return t in self.D[len(t)]
        t0 = time.time()
        r = crt_dict.realised(self.y, t, 20_000_000)
        self.secs += time.time() - t0
        self.ncrt += 1
        return r


def main():
    args = sys.argv[1:]
    y = int(args[0]) if args and args[0].isdigit() else 37
    floor = int(args[args.index("--floor") + 1]) if "--floor" in args else 14
    dmax = int(args[args.index("--depth") + 1]) if "--depth" in args else 7
    q1 = chain_a4.next_prime(y)
    D, F, F2 = chain_dict_oracle.load_exact_dict(y)
    budget = F + q1
    print("machine %d (exact full-period 4-tuple census), q' = %d, "
          "floor v = %d, budget F + q' = %d" % (y, q1, floor, budget))
    print("  |D_1..D_4| = %s,  F = %d, F_2 = %d"
          % ([len(D[m]) for m in sorted(D)], F, F2))
    t0 = time.time()
    lay, hs, G = floor_closure(D, floor)
    print("  floor-%d A_4: %d states, %d edges;  LAYERS (sound upper bounds "
          "on Q_{k+2}) = %s   (%.0fs)"
          % (floor, G[2], len(G[3]), lay, time.time() - t0))
    print("  depth cap: Q_J = 0 for J > %d (the closure has no walk of "
          "length %d)" % (len(lay) + 1, len(lay)))
    oracle = Oracle(y, D)
    rows = []
    for J in range(2, min(dmax, len(lay) + 1) + 1):
        t1 = time.time()
        up = lay[J - 2]
        try:
            v, wit = exact_depth(y, D, floor, J, up, hs, G, oracle)
            tag = "EXACT"
        except RuntimeError:
            v, wit, tag = up, None, "UPPER BOUND ONLY (enumeration budget)"
        rows.append((J, up, v, wit, tag))
        print("    Q_%d(%d; %d) = %3d   %-34s  bound %3d   witness %s "
              " (%.0fs)" % (J, y, floor, v, tag, up, wit, time.time() - t1),
              flush=True)
    mx = max(r[2] for r in rows)
    print("\n  max_J Q_J(%d; %d) = %d   vs budget %d   %s"
          % (y, floor, mx, budget, "CERTIFIES (D)" if mx <= budget
             else "FAILS"))
    print("  oracle: %d calls (%d by CRT, %.0f s)"
          % (oracle.n, oracle.ncrt, oracle.secs))
    print("\n  FOR FORMALIST - the stratified qualifying spectrum of machine "
          "%d at floor %d:" % (y, floor))
    print("    Q_J = %s   for J = 2..%d, and 0 above"
          % ([r[2] for r in rows], rows[-1][0]))
    assert mx <= budget, "the qualifying criterion FAILS at this step"
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
