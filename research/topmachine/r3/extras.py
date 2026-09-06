"""Q16 refinement (is the record block covered exactly once?) and Q3 at m = 5."""

import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from core import open_mask_pair, walk_lengths_fast

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def cover_multiplicity(sets):
    say("### Q16 refinement  how many gears strike each cell of a record block")
    say()
    say("| gears | m | F_top | record blocks | blocks with every cell struck once |"
        " cells struck twice or more (min over blocks) |")
    say("|---|---|---|---|---|---|")
    for gears in sets:
        W = prod(gears)
        idx = np.arange(W)
        cnt = np.zeros(W, dtype=np.int8)
        for g in gears:
            r = idx % g
            cnt += ((r == 0) | (r == g - 2)).astype(np.int8)
        mask = cnt == 0
        L = walk_lengths_fast(mask)
        F = int(L.max())
        starts = np.flatnonzero(L == F)
        # a record block starts at x where L(x) = F and x-1 is open
        good = 0
        minmulti = 10 ** 9
        for s in starts:
            block = cnt[(np.arange(s, s + F)) % W]
            excess = int((block - 1).sum())
            if excess == 0:
                good += 1
            minmulti = min(minmulti, excess)
        say(f"| {','.join(map(str,gears))} | {len(gears)} | {F} | {len(starts):,} | "
            f"{good:,} | {minmulti} |")
    say()


def triple_m5():
    say("### Q3 at m = 5  the twin-candidate record on a five-gear wheel")
    say()
    gears = (19, 23, 29, 31, 37)
    W = prod(gears)
    m = len(gears)
    mask = np.ones(W, dtype=bool)
    for g in gears:
        for t in (0, 1, 2):
            mask[(-t) % g::g] = False
    R = walk_lengths_fast(mask)
    say(f"gears {gears}, W = {W:,}, every gear >= 3m+3 = {3*m+3}: {min(gears) >= 3*m+3}")
    say(f"starts of a twin candidate = {int(mask.sum()):,} = prod(g-3) = "
        f"{prod(g-3 for g in gears):,}")
    say(f"max R = {int(R.max())}   (predicted 3m = {3*m})")
    say()


if __name__ == "__main__":
    cover_multiplicity([(7, 11, 13), (11, 13, 17), (13, 17, 19), (17, 19, 23),
                        (19, 23, 29), (7, 11, 13, 17), (11, 13, 17, 19),
                        (13, 17, 19, 23), (17, 19, 23, 29), (11, 13, 17, 19, 23)])
    triple_m5()
    with open("research/topmachine/r3/results/extras.out", "w") as f:
        f.write("\n".join(OUT))
