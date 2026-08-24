"""Round 22 (constructor): exact pattern counts at machine 41 (q' = 43) -
the arity probe one machine beyond any scan.

Machine 41's period is 5*7*...*41 = 5.07e13 slots; no census exists.  The
round-21 exact pattern counter (qualrun_zerocert.pattern_count, pure CRT
inclusion-exclusion with hereditary-zero pruning) counts a NAMED word's
occurrences per period exactly, with no scan, at a cost that grows with the
word's SPAN only.  This script counts the words that decide the two arities
at machine 41:

  A_relax  = min m such that some m-window of the infinite alternating word
             ...a b a b... is unrealized      -> needs (a,b,a) and (b,a,b)
  A_kill   = k_max = 1 + max killable run     -> needs the killable 2-words
             (k_max >= 3) and the killable 3-words (k_max = 3)

a = 2u' = 14, b = q'-2u' = 29, padded letters 43 and 86.
Usage: uv run python research/arity_probe41.py [--maxspan S]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qualrun_zerocert import pattern_count, primes

Y, QP = 41, 43
A, B, PAD = 14, 29, 43


def cls(v):
    """T3 class of a letter: +1 = +2c, -1 = -2c, 0 = padded (transparent)."""
    r = v % QP
    if r == 0:
        return 0
    if r == A:
        return 1
    if r == B:
        return -1
    return None


def killable(word):
    """T3: nonzero classes strictly alternate (padded transparent)."""
    last = 0
    for v in word:
        c = cls(v)
        if c is None:
            return False
        if c == 0:
            continue
        if c == last:
            return False
        last = c
    return True


def count_word(gears, word, tag=""):
    X = [0]
    for v in word:
        X.append(X[-1] + v)
    span = X[-1]
    Y_ = [t for t in range(1, span) if t not in set(X)]
    t0 = time.time()
    cnt, nodes = pattern_count(gears, X, Y_)
    dt = time.time() - t0
    st = "BUDGET" if cnt is None else f"{cnt:,}"
    print(f"  {tag}word {tuple(word)} span {span}: {st}  "
          f"({nodes:,} nodes, {dt:.0f}s)", flush=True)
    return cnt


def main():
    maxspan = 90
    if "--maxspan" in sys.argv:
        maxspan = int(sys.argv[sys.argv.index("--maxspan") + 1])
    gears = primes(5, Y)
    print(f"machine {Y}: gears {gears}, q' = {QP}, a = {A}, b = {B}, "
          f"pad = {PAD}; maxspan {maxspan}", flush=True)
    # letters: qualifying values <= F(41) = 91
    F41 = 91
    letters = [v for v in range(1, F41 + 1) if cls(v) is not None]
    print(f"  qualifying letters <= F(41)={F41}: {letters}", flush=True)

    print("\n[1] A_relax probe: the two 3-windows of the infinite "
          "alternating word", flush=True)
    relax = {}
    for w in ((A, B, A), (B, A, B)):
        relax[w] = count_word(gears, w, "relax ")

    print("\n[2] killable 2-words (k_max >= 3 test)", flush=True)
    two = {}
    for x in letters:
        for y in letters:
            if not killable((x, y)):
                continue
            if x + y > maxspan:
                continue
            two[(x, y)] = count_word(gears, (x, y))

    print("\n[3] killable 3-words (k_max = 3 test)", flush=True)
    three = {}
    for x in letters:
        for y in letters:
            for z in letters:
                if not killable((x, y, z)):
                    continue
                if x + y + z > maxspan:
                    continue
                if (x, y, z) in relax:
                    continue
                three[(x, y, z)] = count_word(gears, (x, y, z))

    print("\n=== SUMMARY machine 41 (q'=43)", flush=True)
    n2 = sum(v for v in two.values() if v)
    print(f"  killable 2-words counted (span <= {maxspan}): "
          f"{len(two)}, total {n2:,}, "
          f"undecided {sum(1 for v in two.values() if v is None)}")
    allthree = dict(three)
    allthree.update({k: v for k, v in relax.items() if killable(k)})
    n3 = sum(v for v in allthree.values() if v)
    print(f"  killable 3-words counted (span <= {maxspan}): "
          f"{len(allthree)}, total {n3:,}, "
          f"undecided {sum(1 for v in allthree.values() if v is None)}")
    print(f"  A_relax(41) = 3  iff  min(relax counts) == 0 : "
          f"{ {str(k): v for k, v in relax.items()} }")


if __name__ == "__main__":
    main()
