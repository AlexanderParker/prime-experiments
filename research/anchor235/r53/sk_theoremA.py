"""r53 sk_theoremA - the certificates for Theorem A and for the ladder A(K), K <= 10.

Two certificates per K.

POSITIVE (a cover):  an explicit set of K primes and an explicit phase for each, verified by
engine (1) of sk_core (a direct search over the phases of that explicit set), showing that
K primes DO cover A(K) - 1 consecutive columns.  Nothing type-reduced enters.

NEGATIVE (no cover): infeasibility of the exact 0/1 program

    binary x[i,o] = "one gear of item type i is used at option o"
    sum_o x[i,o] <= mult_i          (type i is offered by mult_i primes)
    sum_{i,o} x[i,o] <= K           (K gears)
    sum_{(i,o): c in mask(i,o)} x[i,o] >= 1   for every column c of 0..L-1

over the type-reduced item list of sk_core at level L, which by the type lemma quantifies
over ALL primes >= 5.  Infeasible at L means no K primes cover L columns.  Since a cover of
L columns restricts to a cover of L-1, coverability is monotone decreasing in L, so
infeasibility at L = A(K) gives infeasibility at every L >= A(K), in particular at W(K).

Run:  uv run python research/anchor235/r53/sk_theoremA.py [KMAX] [--window]
      --window also runs the program directly at L = W(K) (the literal Theorem A).
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from sk_core import (RESULTS, Level, arc, cover_set, coverable_any, masks_for,
                     primes_upto, sep)

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


A_K = {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28, 7: 37, 8: 45, 9: 68, 10: 88}
OPT = {1: [5], 2: [5, 7], 3: [5, 7, 11], 4: [5, 7, 11, 17], 5: [5, 7, 11, 23, 29],
       6: [5, 7, 11, 17, 23, 37], 7: [5, 7, 11, 13, 17, 19, 31],
       8: [5, 7, 11, 13, 19, 29, 31, 83], 9: [5, 7, 11, 13, 17, 23, 31, 37, 47],
       10: [5, 7, 11, 13, 17, 19, 23, 29, 37, 79]}
PR = primes_upto(1000)


def window(K):
    """W(K) = (p_{K+1}^2 - 1)/6 with p_j the j-th prime above 3."""
    g = [p for p in PR if p >= 5]
    return (g[K] ** 2 - 1) // 6


def build_program(L, K):
    lv = Level(L, K)
    cols, types = [], []
    for idx, (kind, key, _m) in enumerate(lv.items):
        for msk in masks_for(kind, key, L):
            if msk:
                cols.append(msk)
                types.append(idx)
    n = len(cols)
    nt = len(lv.items)
    rows, ca, va = [], [], []
    for j, msk in enumerate(cols):
        mm = msk
        while mm:
            b = mm & -mm
            rows.append(b.bit_length() - 1)
            ca.append(j)
            va.append(1.0)
            mm ^= b
    for j, t in enumerate(types):
        rows.append(L + t)
        ca.append(j)
        va.append(1.0)
    for j in range(n):
        rows.append(L + nt)
        ca.append(j)
        va.append(1.0)
    A = csr_matrix((va, (rows, ca)), shape=(L + nt + 1, n))
    lb = np.concatenate([np.ones(L), np.full(nt, -np.inf), [-np.inf]])
    ub = np.concatenate([np.full(L, np.inf),
                         np.array([min(m, K) for _k, _key, m in lv.items], dtype=float),
                         [float(K)]])
    return A, lb, ub, n, lv, cols, types


def program_feasible(L, K, timeout=None):
    A, lb, ub, n, lv, cols, types = build_program(L, K)
    t = time.time()
    res = milp(c=np.zeros(n), constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": timeout} if timeout else None)
    sol = None
    if res.success and res.x is not None:
        sol = [(lv.items[types[j]][0], lv.items[types[j]][1],
                [b for b in range(L) if cols[j] >> b & 1])
               for j in range(n) if res.x[j] > 0.5]
    return res.success, res.status, sol, n, time.time() - t


def explicit_cover(S, L):
    """Engine (1): the explicit phases of an explicit prime set covering L columns."""
    ok, w = cover_set(S, L, want_witness=True)
    return ok, w


def main():
    os.makedirs(RESULTS, exist_ok=True)
    KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    do_window = "--window" in sys.argv

    say("=" * 96)
    say("THEOREM A AND THE LADDER: certificates for K = 1..%d" % KMAX)
    say("  A(K)  = the least L that no K primes >= 5 cover")
    say("  W(K)  = (p_{K+1}^2 - 1)/6")
    say("=" * 96)
    say(f"{'K':>3} {'A(K)':>5} {'W(K)':>5} {'A<W':>4} {'cover of A-1':>13} "
        f"{'no cover of A':>14} {'binaries':>9} {'secs':>7}")
    rec = {}
    for K in range(1, KMAX + 1):
        A = A_K[K]
        W = window(K)
        ok_pos, wit = explicit_cover(OPT[K], A - 1) if A > 1 else (True, ())
        ok, status, sol, n, dt = program_feasible(A, K)
        rec[K] = dict(A=A, W=W, cover=bool(ok_pos), witness=wit, nocover=(not ok),
                      status=int(status), binaries=n, secs=round(dt, 1))
        say(f"{K:>3} {A:>5} {W:>5} {'yes' if A < W else 'NO':>4} "
            f"{('YES' if ok_pos else 'FAILED'):>13} {('YES' if not ok else 'FAILED'):>14} "
            f"{n:>9} {dt:>7.1f}")

    say()
    say("positive certificates (explicit primes and phases; phase = the column of the low")
    say("tooth, i.e. the gear strikes k iff k = phase or phase + 3^{-1} (mod g)):")
    for K in range(1, KMAX + 1):
        if rec[K]["witness"]:
            say(f"  K={K}, L={A_K[K]-1}: " +
                ", ".join(f"{g}@{ph}" for g, ph in rec[K]["witness"]))

    if do_window:
        say()
        say("=" * 96)
        say("the literal Theorem A: the same program run directly at L = W(K)")
        say("=" * 96)
        for K in range(1, KMAX + 1):
            W = window(K)
            ok, status, sol, n, dt = program_feasible(W, K, timeout=900)
            say(f"  K={K:>2} L=W={W:>4}: {'COVER (!)' if ok else 'NO COVER'} "
                f"(status {status}, {n} binaries, {dt:.1f}s)")
            rec[K]["window_nocover"] = (not ok)
            rec[K]["window_status"] = int(status)

    with open(os.path.join(RESULTS, "sk_theoremA.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    with open(os.path.join(RESULTS, "sk_theoremA.json"), "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
