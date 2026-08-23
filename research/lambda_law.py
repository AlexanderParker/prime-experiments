"""Round 20 lateral: THE ADJACENT-GAP EXCLUSION LAW - stated, proved, scoped,
and cross-checked against the Mechanic's independent census.

LAW. Let A_5 = {0,2,3} be the residues mod 5 exposed by gear 5 (teeth at 1,4).
Three consecutive openings with gaps (g1, g2) sit at offsets 0, g1, g1+g2, so
they require some phase r with r, r+g1, r+g1+g2 all in A_5. Whenever no such r
exists the configuration is IMPOSSIBLE - in every machine containing gear 5,
at every scale, forever.

    FORBIDDEN: (g1 mod 5, g2 mod 5) in
               {(1,1), (1,3), (2,4), (3,1), (4,2), (4,4)}   -  6 of 25 classes.

This is a PROOF, not a measurement: a phase failing mod 5 fails outright, and
CRT cannot rescue it. The census zeros are a CHECK on the proof, not evidence
for it.

SCOPE, precisely. Lambda(g1,g2) = 0 iff c_q = 0 for some gear q, and by the
round-17 completeness lemma an n-point shape can only be blocked by gears
q <= 2n. For three points that is q <= 6, so ONLY GEAR 5 can ever do it -
gear 7 onward never blocks an adjacent gap pair. Hence the law above is
complete: those 6 classes are ALL the forbidden adjacent pairs, for every
machine. And it applies to ADJACENT gaps only (lag 1): at separation j >= 2
the intervening openings are free, the offsets are not determined, and no
exclusion follows.
"""
import csv
from collections import defaultdict

A5 = {0, 2, 3}
FORBIDDEN = {(g1, g2) for g1 in range(5) for g2 in range(5)
             if not any((r in A5) and ((r + g1) % 5 in A5)
                        and ((r + g1 + g2) % 5 in A5) for r in range(5))}

print("=" * 78)
print("PART A: the law, derived")
print(f"  exposed mod 5: {sorted(A5)}; forbidden (g1,g2) mod 5: "
      f"{sorted(FORBIDDEN)}  ({len(FORBIDDEN)}/25 = {100*len(FORBIDDEN)/25:.0f}%)")
print("  only gear 5 can block a 3-point shape (completeness lemma: q <= 2n = 6),")
print("  so this list is COMPLETE for adjacent gap pairs in every machine.")

print("=" * 78)
print("PART B: cross-check against mechanic's independent census")
rows = defaultdict(list)
with open("research/data/gap_pair_joint.csv") as f:
    for r in csv.DictReader(f):
        rows[(int(r["y"]), int(r["lag"]))].append(
            (int(r["gu"]), int(r["gv"]), int(r["count"])))
print(f"  {'y':>4} {'lag':>4} {'cells':>7} {'in forbidden class':>19} "
      f"{'total count there':>18}  verdict")
for (y, lag) in sorted(rows):
    cells = rows[(y, lag)]
    viol = [(u, v, c) for u, v, c in cells if (u % 5, v % 5) in FORBIDDEN]
    tot = sum(c for _, _, c in viol)
    if lag == 1:
        verdict = "LAW HOLDS" if not viol else f"VIOLATION x{len(viol)}"
    else:
        verdict = ("law does not apply (lag>=2); populated as expected"
                   if viol else "law does not apply; none present anyway")
    print(f"  {y:>4} {lag:>4} {len(cells):>7} {len(viol):>19} {tot:>18}  {verdict}")
