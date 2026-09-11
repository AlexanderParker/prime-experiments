"""xm_table.py -- the tables of first_realisation.md from results/xm_scan_m{p}.json.

For every engine p: the staircase x_min(p, l) (one row per record-breaking run), the section
row (l_p = (p'^2 - p^2)/6, the position b = (p'^2 - 1)/6 of the section's end, the ratio), the
prefix rows (x_min below b: the twin gaps of the prefix), and the two pre-registered laws:

  X1 (first-hit floor)   x_min(p, l) * rho_p(l) >= 1/4,  rho_p(l) = S_p(l) / X = the density of
                         columns at which a run of length >= l starts, measured on the scanned
                         range X (S_p(l) = sum_{r >= l} (r - l + 1) hist[r]);
  X2 (quadratic floor)   x_min(p, l) >= (3/8) l^2, all l; and restricted to l >= (2p+2)/3.
"""
import json
import os
import sys

import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
NEXT = {PRIMES[i]: PRIMES[i + 1] for i in range(len(PRIMES) - 1)}


def load(here, p):
    fn = os.path.join(here, "results", f"xm_scan_m{p}.json")
    if not os.path.exists(fn):
        return None
    with open(fn) as f:
        return json.load(f)


def xmin_of(stair, l):
    for (x, r) in stair:
        if r >= l:
            return x
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ps = [int(v) for v in sys.argv[1:]] or [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    summary = {}
    for p in ps:
        sc = load(here, p)
        if sc is None:
            print(f"m{p}: no scan")
            continue
        stair = [tuple(e) for e in sc["stair"]]
        hist = np.array(sc["hist"], dtype=np.int64)
        X = sc["scanned_to"] - 1
        P = sc["P"]
        pn = NEXT[p]
        a = (p * p - 1) // 6
        b = (pn * pn - 1) // 6
        lp = b - a
        gmax = sc["gmax"]
        # S(l) = number of run starts of length >= l in the scanned range
        r_idx = np.arange(hist.size)
        S = {}
        for l in range(1, gmax + 1):
            S[l] = int(((r_idx - l + 1) * hist * (r_idx >= l)).sum())
        print(f"\n=== m{p}: P = {P:,d}; scanned X = {X:,d} ({X / P:.4f} P); done {sc['done']}; "
              f"F = {gmax + 1 if sc['done'] else '>= ' + str(gmax + 1)}; "
              f"section l_p = {lp} (interior {lp - 1}), a = {a}, b = {b} ===")
        print(f"{'l..':>8} {'x_min':>16} {'x/P':>10} {'x/b':>12} {'S(l)/X':>12} {'x*rho':>8} "
              f"{'x/(3/8 l^2)':>11}")
        rows = []
        prev_r = 0
        x1_min = None
        x1_cells = []
        x2_min_all = None
        x2_min_restricted = None
        for (x, r) in stair:
            for l in range(prev_r + 1, r + 1):
                rho = S[l] / X
                x1 = x * rho
                x1_cells.append((x1, l))
                x2 = x / ((3 / 8) * l * l)
                if x1_min is None or x1 < x1_min[0]:
                    x1_min = (x1, l, x)
                if x2_min_all is None or x2 < x2_min_all[0]:
                    x2_min_all = (x2, l, x)
                if 3 * l >= 2 * p + 2 and (x2_min_restricted is None or x2 < x2_min_restricted[0]):
                    x2_min_restricted = (x2, l, x)
            l = prev_r + 1
            rho = S[l] / X
            lab = f"{l}..{r}" if r > l else f"{l}"
            print(f"{lab:>8} {x:16,d} {x / P:10.6f} {x / b:12.3f} {rho:12.3e} {x * rho:8.3f} "
                  f"{x / ((3 / 8) * l * l):11.3f}")
            rows.append({"l_from": l, "l_to": r, "x": x})
            prev_r = r
        xs = xmin_of(stair, lp)
        xs1 = xmin_of(stair, lp - 1)
        med = float(np.median([c[0] for c in x1_cells])) if x1_cells else None
        below = sum(1 for c in x1_cells if c[0] < 0.25)
        print(f"section: x_min(p, l_p = {lp}) = {xs if xs is not None else '> ' + f'{X:,d}'}"
              f"{'' if xs is None else f'  = {xs / b:.2f} b = {xs / a:.2f} a'};  "
              f"x_min(p, l_p - 1 = {lp - 1}) = {xs1 if xs1 is not None else '> ' + f'{X:,d}'}")
        print(f"X1: min x*rho = {x1_min[0]:.4f} at l = {x1_min[1]} (x = {x1_min[2]:,d}); median "
              f"{med:.3f}; cells below 1/4: {below} of {len(x1_cells)}")
        print(f"X2: min x/(3/8 l^2) = {x2_min_all[0]:.4f} at l = {x2_min_all[1]} (x = {x2_min_all[2]:,d})"
              + (f"; restricted to l >= (2p+2)/3: {x2_min_restricted[0]:.4f} at l = {x2_min_restricted[1]}"
                 f" (x = {x2_min_restricted[2]:,d})" if x2_min_restricted else "; restricted: no cell"))
        summary[p] = {"P": P, "X": X, "done": sc["done"], "gmax": gmax, "l_p": lp, "a": a, "b": b,
                      "x_min_lp": xs, "x_min_lp_minus_1": xs1, "rows": rows,
                      "X1_min": x1_min, "X1_median": med, "X1_below_quarter": below,
                      "X1_cells": len(x1_cells), "X2_min_all": x2_min_all,
                      "X2_min_restricted": x2_min_restricted,
                      "S": {str(l): S[l] for l in S}}
    with open(os.path.join(here, "results", "xm_table.json"), "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()
