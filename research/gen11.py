"""Round 25 (formalist): THE SURVIVOR GENERATOR AT 11 -> 13, in the exact
finite form a Lean kernel can decide.

Constructor's survivor identity (docs/novel/survivor-generator.md) says the
whole low spectrum of machine M + q' is a max-plus language over machine M.
At M = 11, q' = 13 the object is small enough to write out completely:

  * machine 11's cyclic gap word has 135 letters over a period of 385 slots;
  * gear 13 kills the slot residues 2 and 11 (6*2 = 12 = 13 - 1,
    6*11 = 66 = 5*13 + 1), so relative to a base opening at slot residue
    c mod 13 the k-th following machine-11 opening is killed iff
    (c + off_k) mod 13 is 2 or 11;
  * a gap of machine 13 is a window of machine-11 openings whose interiors
    are ALL killed (the plain Kleene generator, F);
  * a TWO-gap window of machine 13 is one in which exactly ONE interior
    survives (Constructor's SIGMA letter - the survivor identity, F_2).

Everything is then a bounded search over 135 bases x 13 phases x windows,
with no reference to machine 13's own period (5005) at all.  This script
computes both values, checks them against the corpus, and emits the machine-11
gap word for transcription into Lean.

Usage: python research/gen11.py
"""
from math import prod

GEARS = [5, 7, 11]
P = prod(GEARS)                      # 385
KILL13 = {2, 11}                     # gear 13's teeth on slot residues
SPANCAP = 30


def openings():
    ex = [False] * P
    for q in GEARS:
        u = pow(6, -1, q)
        for t in (u % q, (-u) % q):
            for k in range(t, P, q):
                ex[k] = True
    return [k for k in range(P) if not ex[k]]


def main():
    ops = openings()
    n = len(ops)
    gw = [ops[(i + 1) % n] - ops[i] + (P if i == n - 1 else 0)
          for i in range(n)]
    assert sum(gw) == P, sum(gw)
    assert n == prod(q - 2 for q in GEARS) == 135, n
    print(f"machine 11: period {P}, {n} openings, gap word sum {sum(gw)}")
    print(f"  F(11) = {max(gw)}  (corpus 7)")
    assert max(gw) == 7

    def off(i, k):
        return sum(gw[(i + t) % n] for t in range(k))

    # the generator: nsurv = number of surviving interior openings.
    # nsurv = 0 gives F(13); nsurv = 1 gives F_2(13) (the SIGMA letter).
    def gen(nsurv, spancap=SPANCAP):
        best, wit = 0, None
        for i in range(n):
            for c in range(13):
                if c in KILL13:
                    continue                     # the base must survive
                d, surv = 0, 0
                k = 0
                while True:
                    k += 1
                    d += gw[(i + k - 1) % n]
                    if d > spancap:
                        break
                    if (c + d) % 13 not in KILL13:      # this one survives
                        if surv == nsurv and d > best:
                            best, wit = d, (i, c, k)
                        surv += 1
                        if surv > nsurv:
                            break
        return best, wit

    for ns, name, corpus in ((0, "F(13)", 11), (1, "F_2(13)", 16)):
        v, w = gen(ns)
        print(f"  generator with {ns} surviving interior(s): {name} = {v}"
              f"   (corpus {corpus})   witness base/phase/len {w}")
        assert v == corpus, (ns, v, corpus)
    # the search is complete at this span cap only if no window is truncated
    # by the cap while still below the value found; report the margin.
    for cap in (20, 25, 30, 40, 60):
        print(f"    span cap {cap}: F={gen(0, cap)[0]}, F_2={gen(1, cap)[0]}")
    # minimum span of j consecutive machine-11 openings (Lean guard)
    for j in (8, 9, 10, 11, 12):
        print(f"    min span of {j} consecutive m11 gaps = "
              f"{min(off(i, j) for i in range(n))}")
    print("\nLean literal (135 gaps):")
    body = ",\n   ".join(", ".join(str(x) for x in gw[i:i + 20])
                         for i in range(0, n, 20))
    print("  [" + body + "]")
    print("\n=> GENERATOR GATE GREEN: F(13) = 11 and F_2(13) = 16 both "
          "computed from machine 11's word alone")


if __name__ == "__main__":
    main()
