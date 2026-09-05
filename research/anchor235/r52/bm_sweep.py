"""bm_sweep.py - the exceptionless sweep: the exact block budget at beta = L = W(q) for every
machine {5..q}, q prime, 5 <= q <= 599, against 1, against sum 4/g^2 and against r51's exact
fibre budget on the same window.

Outputs (results/, untracked): bm_sweep.txt
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bm_exact import ALLP, gears, run_blocks, window
from bm_envelope import fibre_terms

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def fibre_exact(k0, L, gs):
    """r51 dm_budget part B: exact fibre budget under the uniform measure on the window"""
    import numpy as np
    tot = 0.0
    Q = 1
    for g in gs:
        u = pow(6, -1, g)
        teeth = {u % g, (-u) % g}
        if Q > L:
            hit = sum(1 for j in range(L) if (k0 + j) % g in teeth)
            tot += hit / L
        else:
            fib_t = {}
            fib_h = {}
            for j in range(L):
                k = k0 + j
                r = k % Q
                fib_t[r] = fib_t.get(r, 0) + 1
                if k % g in teeth:
                    fib_h[r] = fib_h.get(r, 0) + 1
            m2 = 0.0
            for r, t in fib_t.items():
                a = fib_h.get(r, 0) / t
                m2 += (t / L) * a * a
            tot += m2
        Q *= g
    return tot


def main():
    say("=" * 100)
    say("THE EXACT BLOCK BUDGET AT beta = L = W(q), EVERY MACHINE q = 5..599")
    say("  eta_B  = sum_i E[alpha_i^2] with one block (the whole window), real teeth")
    say("  ideal  = sum 4/g^2;  eta_I = r51's exact fibre budget on the same window")
    say("=" * 100)
    say(f"  {'q':>5} {'L=W(q)':>8} {'gears':>6} {'eta_B':>9} {'ideal':>9} {'eta_B-ideal':>12} "
        f"{'eta_I fibre':>12} {'eta_I/eta_B':>12}")
    bad = 0
    n = 0
    worst_excess = (0.0, None)
    worst_ratio = (1e9, None)
    for q in [p for p in ALLP if 5 <= p <= 599]:
        gs = gears(q)
        k0, _, L = window(q)
        eta, _, _, per = run_blocks(k0, L, gs, lambda i, g, Q, lQ, b=L: b)
        ideal = sum(4.0 / (g * g) for g in gs)
        ef = fibre_exact(k0, L, gs)
        n += 1
        if eta >= 1.0:
            bad += 1
        if eta - ideal > worst_excess[0]:
            worst_excess = (eta - ideal, q)
        if ef / eta < worst_ratio[0]:
            worst_ratio = (ef / eta, q)
        if q in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 97, 149, 199,
                 251, 307, 353, 401, 449, 499, 557, 599):
            say(f"  {q:>5} {L:>8} {len(gs):>6} {eta:>9.5f} {ideal:>9.5f} {eta-ideal:>12.5f} "
                f"{ef:>12.5f} {ef/eta:>12.3f}")
    say()
    say(f"  machines tested: {n};  eta_B >= 1: {bad};  largest excess over sum 4/g^2: "
        f"{worst_excess[0]:.5f} at q = {worst_excess[1]}")
    say(f"  smallest ratio fibre/block: {worst_ratio[0]:.3f} at q = {worst_ratio[1]}")
    say()


if __name__ == "__main__":
    main()
    with open(os.path.join(OUT, "bm_sweep.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nwritten:", os.path.join(OUT, "bm_sweep.txt"))
