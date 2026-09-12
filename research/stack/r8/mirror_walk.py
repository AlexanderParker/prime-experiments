"""The mirror walk (owner's idea, 2026-09-13), built as described.

Nodes: columns (n, n+2) with n = 5 mod 6, n any integer (negative territory allowed).
Home: the column (-1, 1), open to every gear of every machine.
Moves: a flip about the axis k M / 2 for M the product of a subset S of the machine's gears that
contains 2 and 3 (so columns go to columns) and k any positive integer: n -> k M - n - 2.
A flip carries the residues of the gears in S with it: for g in S, the new column is open to g
iff the old one was. Gears outside S are not carried. Intermediate steps need no openness.
Goal: the last flip has an axis carrying all of q's gears (M = q#, any k) and its source must be
open to all of q's gears; the target is a column in the window (q, q^2].

Two readings of "open at a node":
  transported: the gears whose openness the walk itself certifies from home (cut down to S at
               every flip; a node reached with the full gear set certified is proved open);
  computed:    openness checked directly at the node (the walk as a search, at the mercy of
               nothing, since the machine's gears are all we test).
Outputs, per machine: (1) the reach of one flip from home; (2) BFS over flips with transported
certification, the nodes certified open to all gears; (3) the walk as a search: for every window
twin, its sources under the last flip (k q# - n - 2 for k = 1, 2, 3), whether each is open to all
gears (computed) and whether it is itself a twin; (4) the converse count: all-gear-open columns
in the mirror stretch (k q# - q^2 - 2, k q# - q - 2) against the window's twins.
Usage: uv run python mirror_walk.py q [depth]
"""
import sys
from itertools import combinations
from collections import deque
from sympy import primerange, isprime


def main():
    q = int(sys.argv[1]); depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    gears = list(primerange(2, q + 1)); P = 1
    for g in gears: P *= g
    others = [g for g in gears if g > 3]
    subsets = [(2, 3) + c for r in range(len(others) + 1) for c in combinations(others, r)]
    def prod(S):
        m = 1
        for g in S: m *= g
        return m
    def open_to(n, g): return bool(n % g) and bool((n + 2) % g)
    def all_open(n): return all(open_to(n, g) for g in gears)
    win = [n for n in range(q + 1, q * q) if n % 6 == 5]
    twins = [n for n in win if isprime(n) and isprime(n + 2)]
    print(f"machine {q}: gears {gears}, q# = {P}, window ({q}, {q*q}], window twins (left members) {twins}")

    # (1) one flip from home
    home = -1
    one = set()
    for S in subsets:
        M = prod(S)
        for k in range(1, (q * q) // M + 2):
            one.add(k * M - home - 2)
    print(f"(1) columns reachable from home in ONE flip inside the window: {sum(1 for n in win if n in one)} of {len(win)} (the axis M = 6k reaches every column)")

    # (2) BFS with transported certification
    bound = 3 * P
    start = (home, frozenset(gears))
    seen = {start: 0}; dq = deque([start]); full = set()
    while dq:
        n, cert = dq.popleft(); d = seen[(n, cert)]
        if cert == frozenset(gears): full.add(n)
        if d >= depth: continue
        for S in subsets:
            M = prod(S); c2 = cert & frozenset(S)
            kmax = (bound + n + 2) // M + 1
            for k in range(-kmax, kmax + 1):
                m = k * M - n - 2
                if abs(m) > bound: continue
                key = (m, c2)
                if key not in seen: seen[key] = d + 1; dq.append(key)
    fw = sorted(n for n in full if q < n <= q * q)
    cls = sorted(set(n % P for n in full))
    print(f"(2) BFS to depth {depth}, |n| <= 3 q#: nodes visited {len(seen)}; columns certified open to ALL gears by transport: {len(full)}, their residues mod q#: {cls} (i.e. the home class -1 only); certified columns inside the window: {fw}")

    # (3) the walk as a search: sources of the window twins under the last flip
    print("(3) last flip about k q#/2 into the window: window twin n | source s = k q# - n - 2 for k = 1, 2, 3 | source open to all gears (computed) | source is a twin")
    for n in twins:
        row = []
        for k in (1, 2, 3):
            s = k * P - n - 2
            row.append(f"{s}: open {all_open(s)}, twin {isprime(s) and isprime(s + 2)}")
        print(f"   {n} | " + " ; ".join(row))
    # non-twin window columns: their sources are never all-open
    bad = [n for n in win if n not in twins and any(all_open(k * P - n - 2) for k in (1, 2, 3))]
    print(f"    non-twin window columns with an all-open source: {bad} (expected none: the flip is a symmetry)")

    # (4) converse: all-open columns in the mirror stretch
    for k in (1, 2):
        lo, hi = k * P - q * q - 2, k * P - q - 2
        srcs = [s for s in range(lo, hi + 1) if s % 6 == 5 and all_open(s)]
        print(f"(4) k = {k}: all-gear-open columns in the mirror stretch [{lo}, {hi}]: {len(srcs)} = window twins {len(twins)}: {len(srcs) == len(twins)}; their images k q# - s - 2 = {sorted(k * P - s - 2 for s in srcs)}")


if __name__ == "__main__":
    main()
