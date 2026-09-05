"""mf_chain.py -- THE BRANCHING IDENTITY.

A gap of M+q' of order J has exactly J-1 struck interior openings, and they are consecutive
openings of M.  So it contains exactly max(J-r, 0) chains of r consecutive struck openings, and
every such chain (counted once per copy in which it is struck) lies inside exactly one gap.
Hence, writing C_r for the number of (copy, chain) pairs with r consecutive openings of M all
struck by q' in that copy,

        sum_J max(J - r, 0) n_J = C_r          (r >= 0)
        n_J = C_{J-1} - 2 C_J + C_{J+1}        (second difference; the inversion)

with C_0 = q' N (every old gap lies in one new gap in every copy) and C_1 = 2 N (each opening is
struck in exactly two copies).  This script computes C_r INDEPENDENTLY from the chain law - a
chain is struck in 2 copies if all its residues coincide, 1 if they take two values differing by
+-d, 0 otherwise - and checks the inversion against the order histogram built by mf_core.

Writes results/mf_chain.txt.
"""
import os, time
import numpy as np
from mf_core import build_levels, u_of, PRIMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def chain_counts(O, P, q, rmax=10):
    """C_r for r = 0..rmax: number of (copy, r consecutive openings all struck) pairs."""
    N = O.size
    d = (2 * u_of(q)) % q
    ext = np.concatenate([O, O[:rmax + 2] + P])
    res = (ext % q).astype(np.int64)
    C = {0: q * N, 1: 2 * N}
    idx = np.arange(N, dtype=np.int64)
    lo = res[idx].copy()          # candidate phase r (the lower of the two classes)
    hi = (res[idx] - d) % q       # the other candidate
    va = np.ones(N, dtype=bool)
    vb = np.ones(N, dtype=bool)
    for r in range(2, rmax + 1):
        y = res[idx + r - 1]
        va &= (y == lo) | (y == (lo + d) % q)
        vb &= (y == hi) | (y == (hi + d) % q)
        w = va.astype(np.int64) + vb.astype(np.int64)
        C[r] = int(w.sum())
        alive = va | vb
        if not alive.any():
            for rr in range(r + 1, rmax + 2):
                C[rr] = 0
            break
        idx, lo, hi, va, vb = idx[alive], lo[alive], hi[alive], va[alive], vb[alive]
    else:
        C[rmax + 1] = 0
    if rmax + 1 not in C:
        C[rmax + 1] = 0
    return C


def main():
    lines = []
    W = lines.append
    L = build_levels()
    W("=== THE BRANCHING IDENTITY: the order distribution is the second difference of the")
    W("    chain-count sequence,  n_J = C_(J-1) - 2 C_J + C_(J+1)  ===")
    W("")
    W("rung | C_0 = q'N | C_1 = 2N | C_2 | C_3 | C_4 | C_5 | n_1 n_2 n_3 n_4 n_5 (from C) | "
      "n from the built forest | agree")
    for n in range(1, len(L)):
        old, lv = L[n - 1], L[n]
        q = lv.q
        C = chain_counts(old.O, old.P, q, rmax=8)
        pred = [C[J - 1] - 2 * C[J] + C[J + 1] for J in range(1, 6)]
        bc = np.bincount(lv.order, minlength=8)
        act = [int(bc[J]) for J in range(1, 6)]
        W(f"{old.q}->{q} | {C[0]} | {C[1]} | {C[2]} | {C[3]} | {C[4]} | {C[5]} | "
          f"{pred} | {act} | {'OK' if pred == act else 'FAIL'}")
    # rung 23 -> 29 uses the m23 openings, which we have
    W("")
    W("--- rung 23 -> 29 (the same computation on the m23 period) ---")
    old = L[6]
    C = chain_counts(old.O, old.P, 29, rmax=8)
    pred = [C[J - 1] - 2 * C[J] + C[J + 1] for J in range(1, 6)]
    W(f"  C = {[C[r] for r in range(6)]}")
    W(f"  predicted n_1..n_5 = {pred}")
    W(f"  built (mf_top pass 1) order histogram = [199048197, 15416706, 243822, 0, 0]")
    W(f"  agree: {'OK' if pred[:3] == [199048197, 15416706, 243822] and pred[3:] == [0, 0] else 'FAIL'}")
    W("")
    W("--- the closed form of C_2 (the first non-trivial chain count) ---")
    W("C_2 = 2 A_0 + A_d, A_0 = #gaps of M divisible by q', A_d = #gaps = +-d (mod q')")
    for n in range(1, len(L)):
        old, lv = L[n - 1], L[n]
        q = lv.q
        d = (2 * u_of(q)) % q
        r = old.size % q
        A0 = int((r == 0).sum())
        Ad = int(((r == d) | (r == (q - d) % q)).sum())
        C = chain_counts(old.O, old.P, q, rmax=3)
        W(f"  {old.q}->{q}: A_0 = {A0}, A_d = {Ad}, 2A_0 + A_d = {2*A0+Ad}, C_2 = {C[2]}  "
          f"{'OK' if 2*A0+Ad == C[2] else 'FAIL'}")
    d = (2 * u_of(29)) % 29
    r = L[6].size % 29
    A0 = int((r == 0).sum())
    Ad = int(((r == d) | (r == (29 - d) % 29)).sum())
    C = chain_counts(L[6].O, L[6].P, 29, rmax=3)
    W(f"  23->29: A_0 = {A0}, A_d = {Ad}, 2A_0 + A_d = {2*A0+Ad}, C_2 = {C[2]}  "
      f"{'OK' if 2*A0+Ad == C[2] else 'FAIL'}")
    W("")
    W("--- the maximum order is 1 + (largest r with C_r > 0) ---")
    for n in range(1, len(L)):
        old, lv = L[n - 1], L[n]
        C = chain_counts(old.O, old.P, lv.q, rmax=8)
        D = max(r for r in C if C[r] > 0)
        W(f"  {old.q}->{lv.q}: chain depth D = {D}, 1 + D = {D+1}, max order = {int(lv.order.max())} "
          f"{'OK' if D + 1 == int(lv.order.max()) else 'FAIL'}")
    C = chain_counts(L[6].O, L[6].P, 29, rmax=8)
    D = max(r for r in C if C[r] > 0)
    W(f"  23->29: chain depth D = {D}, 1 + D = {D+1}, max order = 3 "
      f"{'OK' if D + 1 == 3 else 'FAIL'}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "mf_chain.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
