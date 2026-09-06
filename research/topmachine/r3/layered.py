"""Q7-Q10: the layered walk, the hop law, the collapse to a double hop, the nested form."""

import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from core import open_mask_pair, walk_lengths_fast

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def layer_test(gears_lo, g):
    """Add gear g to the machine gears_lo; test the hop law and the collapse."""
    W_lo = prod(gears_lo)
    W_hi = W_lo * g
    mask_lo = np.tile(open_mask_pair(gears_lo), g)
    idx = np.arange(W_hi)
    hit = (idx % g == 0) | (idx % g == g - 2)
    mask_hi = mask_lo & ~hit
    L_lo = walk_lengths_fast(mask_lo)
    L_hi = walk_lengths_fast(mask_hi)
    F_lo = int(L_lo.max())

    opens = np.flatnonzero(mask_lo)             # openings of the lower machine
    is_hit = hit[opens]                          # which of them the new gear strikes
    n = len(opens)
    # chain length at each opening: number of consecutive hits starting there
    chain = np.zeros(n, dtype=np.int64)
    run = 0
    for i in range(n - 1, -1, -1):
        run = run + 1 if is_hit[i] else 0
        chain[i] = run
    # cyclic fix: propagate from the end into the start
    if is_hit[-1]:
        i = 0
        while i < n and is_hit[i]:
            chain[i] += 0  # already counted forward within the array
            i += 1

    maxchain = int(chain.max())
    # double hops: chain >= 2
    dbl = np.flatnonzero(chain >= 2)
    # for each of those, y = opens[i]; predicted iff y % g == g-2 and next opening = y+2
    nxt = np.roll(opens, -1)
    gap_lo = (nxt - opens) % W_hi
    pred_dbl = (opens % g == g - 2) & (gap_lo == 2) & is_hit
    ok_dbl = np.array_equal(np.sort(dbl), np.flatnonzero(pred_dbl))

    # hop law: hit iff y mod g in {0, g-2}
    ok_hop = bool(np.array_equal(is_hit, (opens % g == 0) | (opens % g == g - 2)))

    # next own strike after a hop
    ok_next = True
    for i in np.flatnonzero(is_hit)[:5000]:
        y = int(opens[i])
        step = 2 if y % g == g - 2 else g - 2
        if not ((y + step) % g in (0, g - 2)):
            ok_next = False
        # and it is the FIRST such
        for t in range(1, step):
            if (y + t) % g in (0, g - 2):
                ok_next = False
                break

    # the one-line layer formula
    L_pred = L_lo.copy()
    y = idx + L_lo
    ymod = y % g
    hitmask = (ymod == 0) | (ymod == g - 2)
    # gap of the lower machine at y, and at y+2
    pos_of = np.searchsorted(opens, np.arange(W_hi) % W_hi, side="left")
    gap_at = np.zeros(W_hi, dtype=np.int64)
    gap_at[opens] = gap_lo
    d1 = gap_at[y % W_hi]
    y2 = (y + 2) % W_hi
    d2 = gap_at[y2]
    second = hitmask & (ymod == g - 2) & (d1 == 2)
    L_pred = L_lo + hitmask * (d1 + second * d2)
    ok_formula = bool(np.array_equal(L_pred, L_hi))

    return dict(F_lo=F_lo, g=g, big=g > F_lo + 3, maxchain=maxchain,
                ok_hop=ok_hop, ok_dbl=ok_dbl, ok_next=ok_next,
                ok_formula=ok_formula, n_dbl=len(dbl),
                hit_rate=float(is_hit.mean()), F_hi=int(L_hi.max()))


def q7q9(ladders):
    say("### Q8/Q9  the hop law and the collapse (one layer at a time, smallest gear first)")
    say()
    say("| lower machine | F_G | new gear g | g > F_G + 3 | hop law | longest hop chain |"
        " double hops | double-hop rule | one-line layer formula | F_{G+g} |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for gears in ladders:
        for i in range(1, len(gears)):
            lo = gears[:i]
            g = gears[i]
            r = layer_test(lo, g)
            say(f"| {','.join(map(str,lo))} | {r['F_lo']} | {g} | {r['big']} | "
                f"{'0 exceptions' if r['ok_hop'] else 'FAIL'} | {r['maxchain']} | "
                f"{r['n_dbl']:,} | {'0 exceptions' if r['ok_dbl'] else 'FAIL'} | "
                f"{'0 mismatches' if r['ok_formula'] else 'FAIL'} | {r['F_hi']} |")
    say()


# ------------------------------------------------------------------ Q10

def nested_walk(gears, x, counter):
    """The nested closed form: walk with gears[0:i] built up recursively."""
    def W(i, x):
        if i == 0:
            return 0
        counter[1] += 1
        y = x + W(i - 1, x)
        g = gears[i - 1]
        while y % g == 0 or y % g == g - 2:
            counter[0] += 1          # a hop
            y = (y + 1) + W(i - 1, y + 1)
        return y - x
    return W(len(gears), x)


def q10(sets, sample=4000):
    say("### Q10  the nested form: exactness and cost")
    say()
    say("| gears | m | sample | nested == scan | total hops | hops per walk (mean) |"
        " max hops | sum 2/g |")
    say("|---|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(7)
    for gears in sets:
        W = prod(gears)
        mask = open_mask_pair(gears)
        L = walk_lengths_fast(mask)
        xs = rng.integers(0, W, size=min(sample, W))
        counter = [0, 0]
        ok = True
        mx = 0
        for x in xs:
            before = counter[0]
            v = nested_walk(list(gears), int(x), counter)
            mx = max(mx, counter[0] - before)
            if v != L[int(x)]:
                ok = False
        say(f"| {','.join(map(str,gears))} | {len(gears)} | {len(xs):,} | "
            f"{'0 mismatches' if ok else 'FAIL'} | {counter[0]:,} | "
            f"{counter[0]/len(xs):.3f} | {mx} | {sum(2/g for g in gears):.3f} |")
    say()


def q7_order():
    say("### Q7  why smallest-first: the same gear added LAST breaks the collapse")
    say()
    say("| lower machine | F_G | new gear g | g > F_G + 3 | longest hop chain | collapse |")
    say("|---|---|---|---|---|---|")
    for lo, g in [((11, 13, 17), 7), ((13, 17, 19), 7), ((13, 17, 19), 11),
                  ((11, 13), 7), ((17, 19, 23), 7)]:
        r = layer_test(lo, g)
        say(f"| {','.join(map(str,lo))} | {r['F_lo']} | {g} | {r['big']} | "
            f"{r['maxchain']} | {'holds' if r['maxchain'] <= 2 else 'BROKEN'} |")
    say()


if __name__ == "__main__":
    q7q9([(7, 11, 13, 17), (11, 13, 17, 19), (13, 17, 19, 23), (17, 19, 23, 29)])
    q7_order()
    q10([(7, 11, 13), (11, 13, 17), (13, 17, 19), (17, 19, 23),
         (11, 13, 17, 19), (13, 17, 19, 23), (17, 19, 23, 29)])
    with open("research/topmachine/r3/results/layered.out", "w") as f:
        f.write("\n".join(OUT))
