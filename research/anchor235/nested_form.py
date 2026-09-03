"""The walk as a nested residue formula, verified exactly.

Layer g on top of the machine M = {5..g-}: with x = s + W_M(s) (the lower landing) and the
lower gap sequence after x, the hop of g is the sum of the leading lower gaps whose
prefix sums keep landing on g's teeth, i.e. the first m gaps with every landing = +-u_g
(mod g); m is bounded by the layer's chain depth D_g (2, 1, 2, 2, 2, 3 for g = 7..23).

    W_g(s) = W_M(s) + h1 (1 + W_M(x+1)) + h1 h2 (1 + W_M(x1+1)) + ... (D_g terms)
    h1 = [x on teeth], x1 = x + 1 + W_M(x+1), h2 = [x1 on teeth], ...

Verifies the capped formula against the true walk over the full period of every machine up
to {5..19}, counts the W_5 evaluations the unrolled formula needs (prod (1 + D_g)), prints
W_5 and the period-35 table of W_{5,7}, and the gap-sequence form: the gap sequence of layer
g is the lower gap sequence with consecutive gaps merged at every prefix sum = +-u_g mod g.
"""
from math import prod

import numpy as np

PR = [5, 7, 11, 13, 17, 19, 23]
DEPTH = {7: 2, 11: 1, 13: 2, 17: 2, 19: 2, 23: 3}


def teeth(g):
    u = pow(6, -1, g)
    return u, g - u


def on_teeth(g, x):
    u, v = teeth(g)
    return (x % g == u) | (x % g == v)


def W5(s):
    return on_teeth(5, s).astype(np.int64)  # s on a tooth of 5 -> next slot open (teeth 1, 4 mod 5)


EVALS = [0]


def W(gears, s):
    """nested formula, depth-capped; s any int64 array."""
    if len(gears) == 1:
        EVALS[0] += 1
        return W5(s)
    low, g = gears[:-1], gears[-1]
    x = s + W(low, s)
    total = x - s
    h = on_teeth(g, x)
    cur = x
    for _ in range(DEPTH[g]):
        step = 1 + W(low, cur + 1)
        total = total + np.where(h, step, 0)
        cur = np.where(h, cur + step, cur)
        h = h & on_teeth(g, cur)
    return total


def true_walk(gears):
    P = prod(gears)
    k = np.arange(P, dtype=np.int64)
    w = np.ones(P, dtype=bool)
    for g in gears:
        w &= ~on_teeth(g, k)
    ww = np.concatenate([w, w])
    idx = np.flatnonzero(ww)
    nxt = idx[np.searchsorted(idx, np.arange(P))]
    return nxt - np.arange(P), w


def main():
    print("W_5(s) = [s = 1 or 4 mod 5]")
    s = np.arange(35)
    t = W([5, 7], s)
    print("W_{5,7}(s), s = 0..34: " + " ".join(map(str, t.tolist())))
    import sys
    for n in range(2, int(sys.argv[1]) if len(sys.argv) > 1 else 7):
        gears = PR[:n]
        Wt, w = true_walk(gears)
        EVALS[0] = 0
        Wf = W(gears, np.arange(prod(gears), dtype=np.int64))
        ok = np.array_equal(Wt, Wf)
        # residual if depth were one less at the top layer
        print(f"{'+'.join(map(str, gears))}: formula exact over the full period {prod(gears)}: {ok}; "
              f"W_5 evaluations in the unrolled formula {EVALS[0]} = prod(1 + D_g) = {prod(1 + DEPTH[g] for g in gears[1:])}")
        # gap-sequence form: merge at prefix sums on the teeth
        low = gears[:-1]; g = gears[-1]
        _, wl = true_walk(low)
        Pl = prod(low)
        X = np.flatnonzero(wl)
        gaps = np.diff(np.concatenate([X, [X[0] + Pl]]))
        kill = on_teeth(g, X)
        # merged gap sequence over one lower period repeated g times = one full period
        Xg = np.concatenate([X + j * Pl for j in range(g)])
        killg = on_teeth(g, Xg)
        surv = Xg[~killg]
        mg = np.diff(np.concatenate([surv, [surv[0] + prod(gears)]]))
        print(f"   gap-sequence form: {len(X)} lower gaps per lower period (max {int(gaps.max())}), "
              f"{int(killg.sum())} of {len(Xg)} lower openings per full period on {g}'s teeth (= 2/{g} exactly: {int(killg.sum()) * g == 2 * len(Xg)}); "
              f"merged: {len(surv)} gaps per period (= prod(g - 2) = {prod(h - 2 for h in gears)}), "
              f"max merged gap {int(mg.max())} = F + 1 = {int(Wt.max()) + 1}")


if __name__ == "__main__":
    main()
