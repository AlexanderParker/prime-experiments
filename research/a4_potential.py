"""Round 25 (formalist): the A_m QUALIFYING-TAIL POTENTIAL, and whether it
certifies a (D) rung over an EXPLICIT gap-tuple dictionary.

WHY.  `proofs/Potential.lean` already has, kernel-checked,

    merged_le_of_potential :
      (C1) forall i, g i <= h i
      (C2) forall i, 2u <= g i -> g i + h (i+1) <= h i
      (C3) forall i, g i + h (i+1) <= F + q
      ->  forall a l, (forall i < l, 2u <= g (a+1+i)) ->
            g a + windowSum g (a+1) l + g (a+l+1) <= F + q

i.e. a POTENTIAL on the old machine's gap sequence discharges BOTH inputs of
`MergeLaw.newgap_le_step` (SpectrumBound at depth 2 is the l = 0 case, and
QualBound at every depth j >= 3 is the general case).  A potential is a
PER-STEP certificate: no depth quantifier, no fixpoint, no closure.

The point of this script: if `h i` is a function of the A_m STATE at i (the
next m-1 gaps), then (C1),(C2),(C3) become checks over the realised gap
m-tuple DICTIONARY - a finite explicit list - and nothing about the machine's
period is used except `hE : every realised m-tuple is in E`, which is
verdict 15's shape.

The canonical potential is the QUALIFYING TAIL: h(i) = the largest sum
g i + g(i+1) + ... + g(i+r) over runs where g i .. g(i+r-1) all qualify.
Over the A_m abstraction it is the longest path in the digraph restricted to
qualifying source gaps, which is what this script computes (with explicit
cycle detection - an infinite value means the abstraction is too coarse).

    H(s)  =  s[0]                            if s[0] <  floor
             s[0] + max{ H(t) : s -> t }     if s[0] >= floor
    V     =  max over edges s -> t of  s[0] + H(t)          (this is C3)

V is then the certified bound: max(F_2, max_j Q_j) <= V, and the rung needs
V <= F(M) + q'.

Usage:  python research/a4_potential.py
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

INF = float("inf")


def load(path):
    rows = open(path).read().strip().split("\n")
    if rows[0][0].isalpha():
        rows = rows[1:]
    return [tuple(int(x) for x in r.replace(" ", ",").split(",") if x != "")
            for r in rows]


def induced(tuples, m):
    out = set()
    M = len(tuples[0])
    for t in tuples:
        for i in range(M - m + 1):
            out.add(t[i:i + m])
    return sorted(out)


def potential(edges, floor):
    """edges: list of m-tuples.  States are (m-1)-tuples.  Returns (H, V,
    cyclic) where H maps state -> qualifying-tail value and V = max over
    edges of s[0] + H(t)."""
    m = len(edges[0])
    out = defaultdict(list)
    states = set()
    for e in edges:
        s, t = e[:m - 1], e[1:]
        out[s].append(t)
        states.add(s)
        states.add(t)
    H = {}
    # iterative DFS with colour marking (0 unvisited, 1 on stack, 2 done)
    colour = {}
    sys.setrecursionlimit(100000)
    cyclic = [False]

    def visit(s):
        stack = [(s, iter(out.get(s, ())))]
        colour[s] = 1
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if colour.get(nxt, 0) == 0:
                    colour[nxt] = 1
                    stack.append((nxt, iter(out.get(nxt, ()))))
                    advanced = True
                    break
                elif colour[nxt] == 1 and node[0] >= floor:
                    cyclic[0] = True
            if advanced:
                continue
            stack.pop()
            colour[node] = 2
            if node[0] < floor:
                H[node] = node[0]
            else:
                best = 0
                for nxt in out.get(node, ()):
                    v = H.get(nxt, 0)
                    if v > best:
                        best = v
                H[node] = node[0] + best

    for s in sorted(states):
        if colour.get(s, 0) == 0:
            visit(s)
    V = 0
    argmax = None
    for e in edges:
        s, t = e[:m - 1], e[1:]
        v = s[0] + H.get(t, 0)
        if v > V:
            V, argmax = v, e
    return H, V, cyclic[0], argmax


CASES = [
    # name, dictionary file, floor 2u' of the NEXT gear, budget F(M)+q', truth
    ("19 -> 23", "tuples4_19.txt", 8, 25 + 23, 47),
    ("23 -> 29", "gap_tuples_23_4.csv", 10, 34 + 29, 60),
    ("29 -> 31", "gap_tuples_29_4.csv", 10, 43 + 31, 71),
    ("31 -> 37", "gap_tuples_31_4.csv", 12, 58 + 37, None),
]


def main():
    print("A_m QUALIFYING-TAIL POTENTIAL over realised gap-tuple dictionaries")
    print("V = certified max(F_2, max_j Q_j); rung needs V <= budget\n")
    for name, fn, floor, budget, truth in CASES:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            print(f"{name}: MISSING {fn}")
            continue
        tup = load(path)
        print(f"{name}   ({fn}, {len(tup):,} 4-tuples, floor {floor}, "
              f"budget {budget}, truth max(F2,maxQ) = {truth})")
        for m in (2, 3, 4):
            ed = induced(tup, m) if m < 4 else sorted(tup)
            H, V, cyc, am = potential(ed, floor)
            tag = "CYCLIC (infinite)" if cyc else f"V = {V}"
            verdict = "" if cyc else (
                "  CERTIFIES" if V <= budget else "  too loose")
            print(f"    A_{m}: {len(ed):,} edges, "
                  f"{len(set(e[:m-1] for e in ed) | set(e[1:] for e in ed)):,}"
                  f" states, {tag}{verdict}   worst edge {am}")
        print()


if __name__ == "__main__":
    main()
