"""Round 11 lateral: the near-top word-SHAPE family - machine-independent
formalization, permitted-family count, and the finiteness reduction.

SHAPE (machine-independent, u'-free): for a near-top gap (t, G),
    Lword = gaps between consecutive openings reading LEFT from t, within
            window W = 20 (nearest flank first);
    Rword = same reading RIGHT from t+G;
    shape = (Lword, Rword);  pinning input = (shape, G mod 5005).
All u'-dependence lives in the INTERIOR merge word (spacing types
{2u'_q, q-2u'_q}), which the pinning never sees - the interior enters only
through G mod 5005.

TWO grammars, two finiteness questions:
  (a) INTERIOR grammar (one gear step): merge word = k interior kills,
      side-alternating, spacing types in {sigma, sigma-bar} alternating ->
      family size <= sum_k 2 * (#part classes)^(k+1): FINITE IFF k_max
      bounded. (The Constructor's fuel bound - the clean reduction.)
  (b) BOUNDARY grammar (this file): at fixed window W the family is finite
      A PRIORI (parts <= W, compositions of <= W). The open content is not
      finiteness but SELECTION: which of the ~10^6 a-priori shapes are
      (i) CRT-admissible (openings avoid all {5,7,11,13} teeth for some
      phase - computable, enumerated here) and (ii) actually near-top
      (extreme-value selection - measured here via cross-machine recurrence).

Tests: shape census of all near-top gaps (0.9F strata, y = 13..29);
cross-machine shape recurrence (does the observed family stabilize?);
mirror closure; part-size anatomy by depth; full enumeration of the
CRT-admissible half-shape family vs the observed one.

Run: uv run python research/word_shapes.py    (repo root; numpy)
"""
from collections import Counter, defaultdict

import numpy as np

from address_drift import FKNOWN, thresh_gaps
from topgap_nesting import local_openings

W = 20
GEARS = (5, 7, 11, 13)

def halves(y, t, G):
    ops = local_openings(y, t - W - 1, t + G + W + 2)
    left = sorted(int(t - o) for o in ops if t - W <= o <= t)      # 0 = t itself
    right = sorted(int(o - (t + G)) for o in ops if t + G <= o <= t + G + W)
    Lword = tuple(np.diff(left).tolist())      # nearest-first gaps
    Rword = tuple(np.diff(right).tolist())
    return Lword, Rword

def collect():
    shapes = {}
    for y in (13, 17, 19, 23, 29):
        F = FKNOWN[y]
        gaps, P = thresh_gaps(y, int(np.ceil(0.9 * F)))
        shapes[y] = [(t, G, *halves(y, t, G)) for t, G in gaps]
    return shapes

def admissible_half(word):
    """Openings at offsets 0, -c1, -c1-c2, ... : some phase avoids all teeth?"""
    offs = [0]
    for c in word:
        offs.append(offs[-1] - c)
    for q in GEARS:
        u = pow(6, -1, q)
        bad = {(u - s) % q for s in offs} | {(-u - s) % q for s in offs}
        if len(bad) == q:
            return False
    return True

def enumerate_admissible(maxsum=W):
    """Count all CRT-admissible half-shapes (compositions, sum <= maxsum)."""
    total = adm = 0
    stack = [((), 0)]
    admset = set()
    while stack:
        word, s = stack.pop()
        if word:
            total += 1
            if admissible_half(word):
                adm += 1
                admset.add(word)
            else:
                continue          # prune: extensions add openings, never help
        for c in range(1, maxsum - s + 1):
            stack.append((word + (c,), s + c))
    return total, adm, admset

def main():
    shapes = collect()
    print("=" * 72)
    print("PART A: shape census and cross-machine recurrence (W = 20)")
    seen_before = set()
    allL = Counter()
    partdepth = defaultdict(Counter)
    for y in (13, 17, 19, 23, 29):
        rows = shapes[y]
        S = {(L, R) for _, _, L, R in rows}
        mirror_ok = all((R, L) in S for L, R in S)
        rec = sum(1 for s in S if s in seen_before)
        halfset = {L for L, R in S} | {R for L, R in S}
        maxpart = max((max(w) for w in halfset if w), default=0)
        for _, _, L, R in rows:
            allL[L] += 1
            allL[R] += 1
            for d, c in enumerate(L):
                partdepth[d + 1][c] += 1
            for d, c in enumerate(R):
                partdepth[d + 1][c] += 1
        print(f"  y={y:>2}: gaps {len(rows):>3}, distinct shapes {len(S):>3}, "
              f"mirror-closed {mirror_ok}, recur-from-earlier {rec}/{len(S)}, "
              f"distinct half-shapes {len(halfset):>3}, max part {maxpart}")
        seen_before |= S
    print(f"  part-size distribution by flank depth (all machines):")
    for d in sorted(partdepth)[:4]:
        c = partdepth[d]
        tot = sum(c.values())
        top = ", ".join(f"{v}:{n}" for v, n in c.most_common(6))
        print(f"    depth {d}: n={tot:>3}  {top}  (max {max(c)})")
    print("=" * 72)
    print("PART B: the permitted family vs the observed family")
    total, adm, admset = enumerate_admissible()
    obs = set(allL)
    inadm = [w for w in obs if w and not admissible_half(w)]
    print(f"  admissible-prefix tree size (pruned enumeration): {total}; "
          f"true a-priori composition count = 2^{W} - 1 = {2**W - 1}")
    print(f"  CRT-admissible under gears {GEARS}: {adm} "
          f"({adm/total:.4f} of a priori)")
    print(f"  observed distinct half-shapes (all machines, all near-top): "
          f"{len(obs)}")
    print(f"  observed but CRT-inadmissible (MUST be 0): {len(inadm)}")
    print(f"  observed / admissible = {len(obs)/adm:.5f} - the cut below "
          f"CRT level is extreme-value selection, not corridor arithmetic")

if __name__ == "__main__":
    main()
