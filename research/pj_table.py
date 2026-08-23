"""Round 20 (mechanic): p_j deficit table for Constructor.

Reads the (deduped) gap_pair_hist.csv and emits, per machine at its own
qualifying floor a = 2u'(next prime), the exact ratios
p_m / p_1^m  for m = 2..6, where p_m = P(all m consecutive gaps >= a).
Writes research/data/pj_deficits.csv. All numbers are ratios of exact
census counts (machine 37's block is the 12.9% prefix, labeled by its
coverage in the source CSV).

Usage: uv run python research/pj_table.py
"""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")
V = 128
FLOOR = {13: 4, 17: 6, 19: 6, 23: 8, 29: 10, 31: 12, 37: 14}


def main():
    hist, minh = {}, {}
    with open(os.path.join(DDIR, "gap_pair_hist.csv")) as f:
        next(f)
        for line in f:
            fl = line.strip().split(",")
            y = int(fl[0])
            if fl[2] == "ghist":
                hist.setdefault(y, np.zeros(V, np.int64))[int(fl[4])] += \
                    int(fl[5])
            else:
                minh.setdefault((y, int(fl[3])),
                                np.zeros(V, np.int64))[int(fl[4])] += \
                    int(fl[5])
    print("p_m/p_1^m at each machine's own floor (exact count ratios)")
    print("machine  a    p1       m=2     m=3     m=4     m=5     m=6")
    rows = []
    for y in sorted(FLOOR):
        if y not in hist:
            continue
        a = FLOOR[y]
        N = int(hist[y].sum())
        p1 = hist[y][a:].sum() / N
        out = f"  {y:5d}  {a:2d}  {p1:.4f}"
        for m in range(2, 7):
            key = (y, m)
            if key not in minh:
                continue
            Nw = int(minh[key].sum())
            r = (minh[key][a:].sum() / Nw) / p1 ** m
            out += f"  {r:7.4f}"
            rows.append((y, a, m, int(minh[key][a:].sum()), float(p1),
                         float(r)))
        print(out)
    with open(os.path.join(DDIR, "pj_deficits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y", "floor", "m", "count_all_ge_a", "p1",
                    "ratio_obs_indep"])
        w.writerows(rows)
    # anchor assertions (full-period machines, exact)
    assert int(hist[29].sum()) == 214708724
    assert int(hist[31].sum()) == 6226553024
    assert FLOOR[29] == 10 and int(hist[29][10:].sum()) == 25507880
    print("assertions passed; wrote pj_deficits.csv")


if __name__ == "__main__":
    main()
