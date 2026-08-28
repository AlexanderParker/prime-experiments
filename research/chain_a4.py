"""Round 25 (constructor): THE CHAIN'S ENGINE - F(M + q') from machine M's
SCAN-FREE dictionary.

R49: the history abstraction A_m of the Kleene generator (R46) has, as its
entire machine input, the dictionary of realised gap m-tuples of M; A_4 is
EXACT at all seven scannable steps.  Round 24 got that dictionary from a
full-period scan.  research/scanfree_dict.py builds it from the gear list by
CRT, so this script closes the loop:

    gears of M   ->  D_3(M), D_4(M)   ->  A_4 closure  =  F(M + q')

with no period anywhere.  That makes the ladder self-propelling: the exact
F needed to size the next rung is an OUTPUT of this rung, not an input.

A_4 is a SOUND upper bound at every step by R49's argument (every real chain
maps to an abstract walk of the same weight), and measured EXACT at 11->13 ..
31->37.  So even where exactness fails the number certifies (D).

Usage: python research/chain_a4.py 19 23 29 [--m 4] [--workers 6]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                        # noqa: E402
import scanfree_dict                                   # noqa: E402

NEG = -(1 << 40)
KNOWN_FNEW = {11: 11, 13: 18, 17: 25, 19: 34, 23: 43, 29: 58, 31: 88,
              37: 91, 41: 103}


def next_prime(y):
    n = y + 1
    while any(n % d == 0 for d in range(2, int(n ** 0.5) + 1)):
        n += 1
    return n


def cls_of(v, q1, a, b):
    r = v % q1
    if r == 0:
        return 0
    if r == a:
        return 1
    if r == b:
        return -1
    return 9


def a_m_closure(y, D, m, verbose=True):
    """A_m over the realised (m-1)-tuple states and realised m-tuple edges."""
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    states = {}                       # (hist tuple, tooth) -> index
    slist = []

    def sid(h, s):
        k = (h, s)
        i = states.get(k)
        if i is None:
            i = len(slist)
            states[k] = i
            slist.append(k)
        return i

    for h in D[m - 1]:
        sid(h, 0)
        sid(h, 1)
    esrc, edst = [], []
    for t in D[m]:
        h1, h2 = t[:m - 1], t[1:]
        di = h1[-1]
        c = cls_of(di, q1, a, b)
        if c == 9:
            continue
        for s in (0, 1):
            if c == 0:
                land = s
            elif c == 1:
                if s != 0:
                    continue
                land = 1
            else:
                if s != 1:
                    continue
                land = 0
            i = states.get((h1, s))
            j = states.get((h2, land))
            if i is None or j is None:
                continue
            esrc.append(i)
            edst.append(j)
    S = len(slist)
    esrc = np.array(esrc, np.int64)
    edst = np.array(edst, np.int64)
    Rs = np.array([h[-1] for h, _ in slist], np.int64)
    Ls = np.array([h[-2] for h, _ in slist], np.int64)
    ew = Rs[esrc] if len(esrc) else np.zeros(0, np.int64)
    lay = []
    cur = Rs.copy()
    for _ in range(14):
        lay.append(int((Ls + cur).max()))
        nxt = np.full(S, NEG, np.int64)
        if len(esrc):
            np.maximum.at(nxt, esrc, ew + cur[edst])
        if nxt.max() <= NEG // 2:
            break
        cur = nxt
    hh = Rs.copy()
    cyclic = True
    for _ in range(S + 2):
        new = hh.copy()
        if len(esrc):
            np.maximum.at(new, esrc, ew + hh[edst])
        if np.array_equal(new, hh):
            cyclic = False
            break
        hh = new
    bound = None if cyclic else int((Ls + hh).max())
    if verbose:
        print("  A_%d over the scan-free dictionary: %d states, %d edges  "
              "-> %s   layers %s"
              % (m, S, len(esrc), "CYCLIC" if cyclic else bound, lay))
    return bound, S, len(esrc), lay


def main():
    args = sys.argv[1:]
    m = int(args[args.index("--m") + 1]) if "--m" in args else 4
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args \
        else 6
    ys = []
    skip = False
    for i, x in enumerate(args):
        if skip:
            skip = False
            continue
        if x.startswith("--"):
            skip = True
            continue
        if x.isdigit():
            ys.append(int(x))
    if not ys:
        print(__doc__)
        return
    rows = []
    for y in ys:
        q1 = next_prime(y)
        print("\n=== machine %d  (q' = %d)" % (y, q1), flush=True)
        t0 = time.time()
        cap = crt_dict.KNOWN_F.get(y)
        D, Fj, und = scanfree_dict.build(y, m, workers,
                                         cap=(cap + 20) if cap else None)
        assert not und, ("undecided dictionary queries", und[:5])
        tdict = time.time() - t0
        t1 = time.time()
        bound, S, E, lay = a_m_closure(y, D, m)
        want = KNOWN_FNEW.get(y)
        tag = ""
        if want is not None:
            assert bound is not None and bound >= want, (y, bound, want)
            tag = ("EXACT" if bound == want else "loose by +%d" % (bound - want))
            print("  F(%d + %d) = %s   corpus %s   %s"
                  % (y, q1, bound, want, tag))
        budget = Fj[1] + q1
        print("  (D) at alpha = 3:  %d <= F(M) + q' = %d + %d = %d   %s"
              % (bound, Fj[1], q1, budget,
                 "CERTIFIES" if bound <= budget else "FAILS"))
        print("  dictionary %.0f s, closure %.0f s, F_1..F_%d = %s"
              % (tdict, time.time() - t1, m, [Fj[j] for j in sorted(Fj)]))
        rows.append((y, q1, Fj, bound, budget, tag, S, E, tdict))
    print("\nTHE A_4 CHAIN (each rung consumes only machine M's own gears)")
    print("  M    q'   F(M)  F_2(M)  A_%d -> F(M+q')  budget  verdict" % m)
    for y, q1, Fj, bound, budget, tag, S, E, td in rows:
        print("  %-4d %-4d %4d  %6d  %14s  %6d  %s %s"
              % (y, q1, Fj[1], Fj[2], bound, budget,
                 "CERTIFIES" if bound <= budget else "FAILS", tag))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
