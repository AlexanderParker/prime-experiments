"""Round 23 (constructor): HOW MANY FACTS ABOUT THE MACHINE DOES (D) NEED?

research/machinefree_cert.py shows that the purely machine-free abstraction
MF_4 (corridor-admissible gap 4-tuples, values 1..F) is far too loose - 125
against a budget of 74 at 29 -> 31.  research/kleene_history.py shows that
A_4, the same system with "corridor-admissible" replaced by "REALISED in the
period", is exact (58).  The gap between them is a set of yes/no facts:
"is this 4-tuple of consecutive gaps realised by machine M?"  MF_4 has 140,471
candidate edges at that step and A_4 has 3,513 realised ones, so answering all
of them is hopeless.

This script measures how many of them a COUNTEREXAMPLE-GUIDED refinement
actually needs:

    start from MF_4 (machine-free);
    repeat: close it; if the bound <= F + q' STOP - (D) is certified;
            otherwise read off a maximising abstract walk, ASK THE ORACLE
            whether each of its gap 4-tuples is realised, and delete every
            4-tuple that is not (deleting a value tuple removes it at every
            corridor phase at once - sound, since an unrealised tuple cannot
            occur anywhere).

Deleting only unrealised tuples is sound, so the bound stays an upper bound on
F(M+q') throughout, and the number of ORACLE QUERIES is the honest size of the
proof obligation: that many CRT zero-certificates (R43) would replace the
period scan.  The oracle here is the realised-tuple dump from the machine-29
pass, so this MEASURES the query count; it does not yet avoid the scan.

Usage: uv run python research/cegar_cert.py [tuple-dump] [--mod 35] [--m 4]
"""
import sys
import time

import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from machinefree_cert import build_mf_edges, NEG          # noqa: E402


def close_with(S, esrc, edst, ew, Rs, Ls, cap=64):
    hh = Rs.copy()
    for _ in range(cap):
        new = hh.copy()
        if len(esrc):
            np.maximum.at(new, esrc, ew + hh[edst])
        if np.array_equal(new, hh):
            break
        hh = new
    else:
        return None, None
    return hh, int((Ls + hh).max())


def walk_from(hh, Ls, Rs, esrc, edst, ew, S):
    """One maximising abstract walk: the sequence of edge indices."""
    st = int(np.argmax(Ls + hh))
    out = []
    order = np.argsort(esrc, kind="stable")
    ss = esrc[order]
    for _ in range(40):
        lo = int(np.searchsorted(ss, st, "left"))
        hi = int(np.searchsorted(ss, st, "right"))
        if lo == hi:
            break
        cand = order[lo:hi]
        gains = ew[cand] + hh[edst[cand]]
        j = int(cand[int(np.argmax(gains))])
        if int(ew[j] + hh[edst[j]]) < int(hh[st]):
            break
        out.append(j)
        st = int(edst[j])
    return out


def main():
    skip = set()
    for i, a in enumerate(sys.argv):
        if a.startswith("--"):
            skip.add(i)
            skip.add(i + 1)
    args = [a for i, a in enumerate(sys.argv) if i and i not in skip]
    dump = args[0] if args else "research/data/tuples4_29.txt"
    Mod = 35
    if "--mod" in sys.argv:
        Mod = int(sys.argv[sys.argv.index("--mod") + 1])
    F, q1, exact = 43, 31, 58
    budget = F + q1
    oracle = set()
    for line in open(dump):
        oracle.add(tuple(int(x) for x in line.split()))
    print("oracle: %d realised gap 4-tuples of machine %d (from the "
          "full-period pass)" % (len(oracle), 29))
    t0 = time.time()
    S, esrc, edst, ew, Rs, Ls, tup = build_mf_edges(F, q1, Mod, 4)
    print("MF_4 mod %d: %d states, %d candidate edges (%d distinct value "
          "4-tuples)" % (Mod, S, len(esrc), len(set(map(tuple, tup)))))
    # ONE extra machine fact, a single number rather than a dictionary:
    # F_2(M), which is lemma 1's left-hand side.  A state whose flank+base
    # pair exceeds it is unrealisable, so dropping it is sound.
    F2 = 0
    if "--f2" in sys.argv:
        F2 = int(sys.argv[sys.argv.index("--f2") + 1])
    if F2:
        okst = (Ls + Rs) <= F2
        keep = okst[esrc] & okst[edst]
        esrc, edst, ew, tup = esrc[keep], edst[keep], ew[keep], tup[keep]
        Ls = np.where(okst, Ls, NEG)
        Rs = np.where(okst, Rs, NEG)
        print("with F_2(M) = %d given: %d of %d states survive, %d edges"
              % (F2, int(okst.sum()), S, len(esrc)))
    t64 = tup.astype(np.int64)
    key = (t64[:, 0] * 64 ** 3 + t64[:, 1] * 64 ** 2
           + t64[:, 2] * 64 + t64[:, 3])
    bykey = {}
    for i, k in enumerate(key.tolist()):
        bykey.setdefault(k, []).append(i)
    live = np.ones(len(esrc), bool)
    asked = set()
    killed = set()
    it = 0
    while True:
        it += 1
        hh, bnd = close_with(S, esrc[live], edst[live], ew[live], Rs, Ls)
        if hh is None:
            print("  iteration %d: CYCLIC" % it)
            break
        if bnd <= budget:
            print("  iteration %4d: bound %3d <= budget %d  -> (D) CERTIFIED"
                  % (it, bnd, budget))
            break
        w = walk_from(hh, Ls, Rs, esrc[live], edst[live], ew[live],
                      S)
        idx = np.flatnonzero(live)[w]
        new_bad = []
        for e in idx.tolist():
            t = tuple(int(x) for x in tup[e])
            asked.add(t)
            if t not in oracle:
                new_bad.append(t)
        if not new_bad:
            print("  iteration %4d: bound %3d and the maximising walk is "
                  "entirely REALISED - the abstraction is tight here, no "
                  "further refinement possible" % (it, bnd))
            break
        for t in new_bad:
            if t in killed:
                continue
            killed.add(t)
            k = (int(t[0]) * 64 ** 3 + int(t[1]) * 64 ** 2
                 + int(t[2]) * 64 + int(t[3]))
            live[bykey.get(k, [])] = False
        if it % 2000 == 0 or it < 4:
            print("  iteration %4d: bound %3d, asked %5d tuples, deleted %5d,"
                  " %6d edges live  (%.0fs)"
                  % (it, bnd, len(asked), len(killed), int(live.sum()),
                     time.time() - t0))
        if it > 200000:
            print("  ITERATION CAP reached at bound %d" % bnd)
            break
    print("\nRESULT: %d refinement iterations, %d ORACLE QUERIES, %d tuples "
          "deleted, %d of %d candidate edges left, %.0f s"
          % (it, len(asked), len(killed), int(live.sum()), len(esrc),
             time.time() - t0))
    print("       (a query is one 'is this 4-tuple realised?' - the unit a "
          "CRT zero-certificate would supply)")


if __name__ == "__main__":
    main()
