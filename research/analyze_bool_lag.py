"""Round 20 (mechanic): Markov-ORDER test on the 8-point boolean pattern
census (bool_lag_census.py).  All predictions are exact factorisations of
the same census's own marginal counts - zero fitted parameters.

For order k: pred(x_0..x_7) = N * P(x_0..x_{k-1}) *
             prod_{t=k}^{7} P(x_t | x_{t-k}..x_{t-1}),
with every P a ratio of position-aggregated m-gram counts.

Reported per (machine, floor):
  TV_k     = total-variation distance sum|obs-pred|/2N for k = 1, 2, 3, 4
  the all-ones pattern (the run event) obs vs pred_k
  lag-j correlation ratios E[b0 bj]/E[b]^2, measured vs the order-k chain
Usage: uv run python research/analyze_bool_lag.py y [y ...] [--floor a]
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")
W = 8  # override with --W


def load(y):
    out = {}
    with open(os.path.join(DDIR, f"bool_lag_{y}.csv" if W == 8 else f"bool_lag{W}_{y}.csv")) as f:
        next(f)
        for line in f:
            yy, cov, a, pat, c = line.strip().split(",")
            d = out.setdefault(int(a), np.zeros(1 << W, np.int64)); d[int(pat)] += int(c)
    return out


def grams(counts, m):
    """Position-aggregated m-gram counts from the 8-pattern counts."""
    g = np.zeros(1 << m, np.float64)
    for pat in range(1 << W):
        c = counts[pat]
        if not c:
            continue
        for p in range(W - m + 1):
            g[(pat >> p) & ((1 << m) - 1)] += c
    return g


def first_gram(counts, m):
    """Marginal of bits 0..m-1 (exact, position 0 only)."""
    g = np.zeros(1 << m, np.float64)
    for pat in range(1 << W):
        g[pat & ((1 << m) - 1)] += counts[pat]
    return g


def order_k_pred(counts, k):
    N = counts.sum()
    init = first_gram(counts, k) / N
    gk1 = grams(counts, k + 1)
    gk = np.zeros(1 << k, np.float64)
    for w in range(1 << k + 1):
        gk[w & ((1 << k) - 1)] += gk1[w]
    pred = np.zeros(1 << W, np.float64)
    for pat in range(1 << W):
        p = init[pat & ((1 << k) - 1)]
        for t in range(k, W):
            prev = (pat >> (t - k)) & ((1 << k) - 1)
            nxt = (pat >> (t - k)) & ((1 << (k + 1)) - 1)
            p *= gk1[nxt] / gk[prev] if gk[prev] else 0.0
        pred[pat] = p * N
    return pred


def lag_ratios(counts):
    N = counts.sum()
    p1 = sum(counts[pat] for pat in range(1 << W) if pat & 1) / N
    out = []
    for j in range(1, W):
        both = sum(counts[pat] for pat in range(1 << W)
                   if (pat & 1) and (pat >> j) & 1)
        out.append(both / N / p1 ** 2)
    return p1, out


def analyse(y, floors=None):
    data = load(y)
    for a in sorted(data):
        if floors and a not in floors:
            continue
        counts = data[a]
        N = int(counts.sum())
        p1, lr = lag_ratios(counts)
        if p1 == 0 or N < 10000:
            continue
        print(f"\n=== machine {y}, floor a = {a}: N = {N:,} windows, "
              f"p1 = {p1:.4f}")
        print("  lag-j ratio E[b0 bj]/p1^2, measured:   "
              + "  ".join(f"{r:6.3f}" for r in lr))
        preds = {}
        for k in (1, 2, 3, 4):
            pred = order_k_pred(counts, k)
            preds[k] = pred
            tv = float(np.abs(counts - pred).sum()) / (2 * N)
            ones = counts[(1 << W) - 1]
            pones = pred[(1 << W) - 1]
            print(f"  order-{k} Markov: TV = {tv:.5f}   "
                  f"all-ones obs {ones:,} pred {pones:,.0f} "
                  f"ratio {ones/pones if pones else float('nan'):.3f}")
        # worst patterns for order-2
        dev = np.abs(counts - preds[2])
        worst = np.argsort(-dev)[:5]
        print("  worst order-2 patterns (bits lag0..lag7, 1 = gap >= a):")
        for pat in worst:
            print(f"    {format(pat, '0%db' % W)[::-1]}  obs {counts[pat]:>12,} "
                  f" pred2 {preds[2][pat]:>14,.0f}  "
                  f"ratio {counts[pat]/preds[2][pat] if preds[2][pat] else float('inf'):.3f}")


def main():
    args = sys.argv[1:]
    floors = None
    if "--floor" in args:
        i = args.index("--floor")
        floors = {int(args[i + 1])}
        del args[i:i + 2]
    if "--W" in args:
        i = args.index("--W")
        globals()["W"] = int(args[i + 1])
        del args[i:i + 2]
    for y in [int(x) for x in args]:
        analyse(y, floors)


if __name__ == "__main__":
    main()
