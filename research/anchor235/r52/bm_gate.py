"""bm_gate.py - thresholds done properly (the a priori block budget is NOT monotone in beta)
and the phase-adversarial validity gate.

The budget as a function of the block length beta jumps whenever some ceil(beta/g) increases, so
bisection is unsound; every threshold here is a scan.  Candidate block lengths: every integer up
to CAP, plus the transition points Q_{<i} g_i (+-1) of the CRT branch, plus a geometric grid.

Validity gate.  A(K) (arc_multiset.md R1) is the longest OPENING-FREE STRETCH, i.e. the gap
between consecutive openings, so K primes can cover A(K) - 1 consecutive columns.  Any envelope
that claims an interval of A(K) - 1 columns cannot be covered is FALSE.  Same for the machine's
own record: F(M) - 1 columns are covered.

Outputs (results/, untracked): bm_gate.txt
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bm_exact import ALLP, gears, window
from bm_envelope import block_terms, fibre_terms, pi_prefix

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


A_K = [2, 5, 7, 16, 22, 28, 37, 45, 68, 88, 101, 115]
GEARS_ALL = [p for p in ALLP if p >= 5]
CAP = 4000


def budget(gs, beta, mode):
    return sum(block_terms(gs, beta, mode))


def limit_budget(gs, mode):
    """beta -> infinity"""
    P = pi_prefix(gs)
    if mode == "CRT+LD":
        return sum(4.0 / (g * g) for g in gs)
    if mode == "CAP":
        return float("inf") if sum(2.0 / g for g in gs) >= 1.0 else \
            sum(min(1.0, (2.0 / g) / (1.0 - sum(2.0 / h for h in gs[:i]))) ** 2
                for i, g in enumerate(gs))
    return sum(min(1.0, (2.0 / g) / P[i]) ** 2 for i, g in enumerate(gs))


def candidates(gs, cap=CAP):
    xs = set(range(1, cap + 1))
    Q = 1
    for g in gs:
        for t in (Q * g - 1, Q * g, Q * g + 1):
            if 0 < t < 10 ** 300:
                xs.add(int(t))
        Q *= g
        if Q > 10 ** 300:
            break
    x = float(cap)
    while x < 1e299:
        xs.add(int(x))
        x *= 1.05
    return sorted(xs)


def beta_star(gs, mode, cap=CAP):
    """1 + the largest block length at which the budget is >= 1 (None if it never settles)"""
    if limit_budget(gs, mode) >= 1.0:
        return None
    worst = 0
    for b in candidates(gs, cap):
        if budget(gs, b, mode) >= 1.0:
            worst = max(worst, b)
    return worst + 1


def claims_below(gs, mode, limit):
    """the block lengths <= limit at which the envelope CLAIMS non-coverability"""
    return [b for b in range(1, int(limit) + 1) if budget(gs, b, mode) < 1.0]


def main():
    say("=" * 100)
    say("A.  THE PHASE-ADVERSARIAL VALIDITY GATE ON THE ADVERSARIAL LADDER")
    say("    K primes >= 5, two classes each at any phase, cover A(K) - 1 consecutive columns")
    say("    (A(K) is the gap between openings; arc_multiset.md R1, exact to K = 12).")
    say("    An envelope is REFUTED if it claims an interval of A(K) - 1 columns is uncoverable.")
    say("    Gears: the K smallest (the worst K-set for the budget).")
    say("=" * 100)
    say(f"  {'K':>3} {'A(K)':>5} {'covered':>8} " +
        " ".join(f"{m:>26}" for m in ["LD", "CAP", "CRT+LD"]))
    for K in range(1, 13):
        gs = GEARS_ALL[:K]
        A = A_K[K - 1]
        cov = A - 1
        cells = []
        for mode in ["LD", "CAP", "CRT+LD"]:
            bad = claims_below(gs, mode, cov)
            bs = beta_star(gs, mode)
            tag = f"REFUTED at beta={bad[-1]}" if bad else "ok"
            cells.append(f"{tag}, b*={'-' if bs is None else bs}")
        say(f"  {K:>3} {A:>5} {cov:>8} " + " ".join(f"{c:>26}" for c in cells))
    say()
    say("  The refuting instance is the smallest one: K primes covering `covered` columns while")
    say("  the envelope's budget at beta = `covered` is below 1.")
    for K in range(1, 13):
        gs = GEARS_ALL[:K]
        cov = A_K[K - 1] - 1
        for mode in ["LD", "CAP", "CRT+LD"]:
            if cov >= 1 and budget(gs, cov, mode) < 1.0:
                say(f"    K={K} gears={gs} mode={mode}: {cov} columns ARE covered, "
                    f"budget = {budget(gs, cov, mode):.5f} < 1")
    say()

    say("=" * 100)
    say("B.  THE SAME GATE ON THE REAL MACHINE'S OWN RECORD")
    say("    F(M) - 1 consecutive columns are covered by the machine {5..q}.")
    say("=" * 100)
    FLAD = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
            41: 91, 43: 103, 47: 118}
    say(f"  {'m':>4} {'F(M)':>5} {'covered':>8} " +
        " ".join(f"{m:>26}" for m in ["LD", "CAP", "CRT+LD"]))
    for q, F in FLAD.items():
        gs = gears(q)
        cov = F - 1
        cells = []
        for mode in ["LD", "CAP", "CRT+LD"]:
            bad = claims_below(gs, mode, cov)
            bs = beta_star(gs, mode)
            tag = f"REFUTED at beta={bad[-1]}" if bad else "ok"
            cells.append(f"{tag}, b*={'-' if bs is None else f'{bs:.3e}'}")
        say(f"  {q:>4} {F:>5} {cov:>8} " + " ".join(f"{c:>26}" for c in cells))
    say()

    say("=" * 100)
    say("C.  THE THRESHOLDS, SCANNED (not bisected): least beta above which the a priori")
    say("    block budget stays below 1, against the window and against r51's fibre L*.")
    say("=" * 100)
    say(f"  {'q':>5} {'W(q)':>9} {'beta* LD':>12} {'beta* CAP':>12} {'beta* CRT+LD':>14} "
        f"{'fibre L* (r51)':>16} {'CRT+LD / L*':>13}")
    for q in [59, 97, 199, 499, 997]:
        gs = gears(q)
        _, _, W = window(q)
        b1 = beta_star(gs, "LD")
        b2 = beta_star(gs, "CAP")
        b3 = beta_star(gs, "CRT+LD")
        # r51's fibre threshold, scanned the same way
        lim = sum(fibre_terms(gs, 1e300))
        fs = None
        if lim < 1.0:
            worst = 0
            for b in candidates(gs, CAP):
                if sum(fibre_terms(gs, b)) >= 1.0:
                    worst = max(worst, b)
            fs = worst + 1
        say(f"  {q:>5} {W:>9} {('none' if b1 is None else f'{b1:.4e}'):>12} "
            f"{('none' if b2 is None else f'{b2:.4e}'):>12} "
            f"{('none' if b3 is None else f'{b3:.4e}'):>14} "
            f"{('none' if fs is None else f'{fs:.4e}'):>16} "
            f"{(f'{b3/fs:.3e}' if (b3 and fs) else '-'):>13}")
    say()


if __name__ == "__main__":
    main()
    with open(os.path.join(OUT, "bm_gate.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nwritten:", os.path.join(OUT, "bm_gate.txt"))
