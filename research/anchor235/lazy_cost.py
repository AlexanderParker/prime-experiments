"""Static size versus evaluation cost of the nested form, and the scan form.

The unrolled nested form has prod (1 + D_g) bottom terms (exponential in layers). But a
hit term is only entered when its landing is on the teeth, so evaluated lazily the number
of W_5 evaluations at a slot s is not prod (1 + D_g) but 1 + (number of hops on the walk
from s at the layers above 5), since W_5 absorbs the hops of gear 5 itself. Pre-registered
as W(s) + 1 (the hop identity); that was wrong by the gear-5 hops and is corrected here:
lazy bottom evaluations = 1 + (crossed slots not on 5's teeth), checked at every slot of
every machine up to {5..19}. The mean and the maximum over the period are printed.

The scan form  W(s) = sum_{j>=1} prod_{i<j} B(s + i),  B = blocked indicator, is exact with
F + 1 terms of pi(q) residue tests each; lazily it costs (W(s) + 1) x (index of the smallest
blocker of each crossed slot) residue tests. Mean residue tests per slot are printed.
"""
from math import prod

import numpy as np

PR = [5, 7, 11, 13, 17, 19]
DEPTH = {7: 2, 11: 1, 13: 2, 17: 2, 19: 2, 23: 3}


def on_teeth(g, x):
    u = pow(6, -1, g)
    return (x % g == u) | (x % g == g - u)


def walk(gears):
    P = prod(gears)
    k = np.arange(P, dtype=np.int64)
    w = np.ones(P, dtype=bool)
    for g in gears:
        w &= ~on_teeth(g, k)
    idx = np.flatnonzero(np.concatenate([w, w]))
    return idx[np.searchsorted(idx, k)] - k, w


def lazy_count(gears, s):
    """cleaner: recursive count per slot via explicit recursion on index sets."""
    cnt = np.zeros(len(s), dtype=np.int64)

    def rec(gs, ss, ids):
        if len(gs) == 1:
            cnt[ids] += 1
            return on_teeth(5, ss).astype(np.int64)
        low, g = gs[:-1], gs[-1]
        x = ss + rec(low, ss, ids)
        total = x - ss
        h = on_teeth(g, x)
        cur = x
        for _ in range(DEPTH[g]):
            j = np.flatnonzero(h)
            if len(j) == 0:
                break
            step = np.zeros(len(ss), dtype=np.int64)
            step[j] = 1 + rec(low, cur[j] + 1, ids[j])
            total = total + np.where(h, step, 0)
            cur = np.where(h, cur + step, cur)
            h = h & on_teeth(g, cur)
        return total

    return rec(gears, s, np.arange(len(s))), cnt


def main():
    for n in range(2, len(PR) + 1):
        gears = PR[:n]
        P = prod(gears)
        s = np.arange(P, dtype=np.int64)
        Wt, w = walk(gears)
        Wf, cnt = lazy_count(gears, s)
        assert np.array_equal(Wt, Wf)
        # residue tests in the layered scan: each crossed slot costs the index of its smallest blocker
        k = s
        smallest = np.full(P, len(gears), dtype=np.int64)
        for i, g in reversed(list(enumerate(gears))):
            smallest[on_teeth(g, k)] = i + 1
        # tests along the walk from s: sum over crossed slots s..s+W-1 of smallest, plus the open landing costs len(gears)
        cs = np.concatenate([[0], np.cumsum(np.concatenate([smallest, smallest]))])
        tests = cs[s + Wt] - cs[s] + len(gears)
        above = (smallest > 1).astype(np.int64)
        ca = np.concatenate([[0], np.cumsum(np.concatenate([above, above]))])
        hops_above = ca[s + Wt] - ca[s]
        print(f"{'+'.join(map(str, gears))}: static terms {prod(1 + DEPTH[g] for g in gears[1:])}; "
              f"lazy W_5 evaluations = W(s) + 1: {bool(np.array_equal(cnt, Wt + 1))} (refuted); = 1 + hops above gear 5: {bool(np.array_equal(cnt, 1 + hops_above))}; "
              f"mean {cnt.mean():.4f} (mean W + 1 = {Wt.mean() + 1:.4f}); max {int(cnt.max())} (F + 1 = {int(Wt.max()) + 1}); "
              f"layered scan residue tests per slot: mean {tests.mean():.3f}, max {int(tests.max())} "
              f"(flat scan {len(gears)} x (W + 1): mean {len(gears) * (Wt.mean() + 1):.3f}, max {len(gears) * (int(Wt.max()) + 1)})")


if __name__ == "__main__":
    main()
