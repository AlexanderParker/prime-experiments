"""Round 26 (mechanic): THE SMALL-GEAR PHASE-SATURATION OBSTRUCTION.

A k-chain word w at step M -> q' occupies exposed offsets
X = {0, w1, w1+w2, ..., sum(w)} (k of them).  In the CRT/COV encoding every
gear q <= M blocks the pair {a, a + s_q} mod q for a FREE phase a, with
s_q = -2 * 6^{-1} (mod q).  The word can occur only if EVERY gear has at
least one phase avoiding all of X, i.e. only if for every q

    FREE_q(X) := Z_q \ ( (X mod q) u (X - s_q mod q) )   is NON-EMPTY.

If some gear's FREE set is empty that gear must block an exposed slot, and
the word is ZERO BY THEOREM - no SAT call.  This is exactly the argument
that killed (18,35,18,35,18) at 47->53 (C23) with no solver; here it is
applied systematically to the ALTERNATING chain (s, q'-s, s, ...), whose
realisability round 25 made the predictor of fuel arity.

usage: python research/alt_obstruct_r26.py
"""
from math import prod


def primes_upto(n):
    return [p for p in range(2, n + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def free_phases(X, q):
    s = (-2 * pow(6, -1, q)) % q
    bad = set()
    for x in X:
        bad.add(x % q)
        bad.add((x - s) % q)
    return [a for a in range(q) if a not in bad]


def alt_offsets(qp, k):
    """exposed offsets of the k-chain pure alternation (s, q'-s, s, ...)."""
    s = (2 * pow(6, -1, qp)) % qp
    X = [0]
    for i in range(k - 1):
        X.append(X[-1] + (s if i % 2 == 0 else qp - s))
    return X


def first_dead_gear(X, M):
    for q in [p for p in primes_upto(M) if p >= 5]:
        if not free_phases(X, q):
            return q
    return None


print("THE ALTERNATION LADDER, decided by pure arithmetic (no SAT):")
print("  step        s   q'-s   k: obstruction (gear that must block an "
      "exposed slot)")
for M, qp in [(31, 37), (37, 41), (41, 43), (43, 47), (47, 53), (53, 59),
              (59, 61), (61, 67)]:
    s = (2 * pow(6, -1, qp)) % qp
    row = []
    kdead = None
    for k in range(2, 9):
        X = alt_offsets(qp, k)
        q = first_dead_gear(X, M)
        row.append(f"k={k}:{'gear %d' % q if q else 'free'}")
        if q and kdead is None:
            kdead = k
    print(f"  {M}->{qp:<4}  {s:3d}  {qp-s:4d}   " + "  ".join(row))
    print(f"        => the pure alternation is ZERO BY THEOREM from k = "
          f"{kdead} on, so it supplies chains of length at most {kdead-1}")


# ---------------------------------------------------------------------------
# FULL-LEVEL SCREEN: apply the same obstruction to EVERY legal word of a level.
# The word list is a_kill.py's (residue legality + prefix-sum window validity
# + span caps), so a level all of whose words are obstructed has N_k = 0 BY
# THEOREM, with no SAT call anywhere.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_kill as A                                            # noqa: E402

print("\nFULL-LEVEL OBSTRUCTION SCREEN (every legal word of the level):")
for M, qp, caps in [(37, 41, A.DEFAULT_CAPS[37]), (41, 43, [91, 103, 110]),
                    (43, 47, A.DEFAULT_CAPS[43]), (47, 53, A.DEFAULT_CAPS[47]),
                    (53, 59, [145])]:
    print(f"  --- {M} -> {qp}  (caps {caps}) ---")
    prev = None
    for k in range(3, 8):
        _, _, _, words = A.enumerate_words(M, qp, k - 1, caps)
        if prev is not None:
            words = [w for w in words if A.sub_ok(w, prev)]
        killed, alive = [], []
        for w in words:
            X, acc = [0], 0
            for g in w:
                acc += g
                X.append(acc)
            (killed if first_dead_gear(X, M) else alive).append(w)
        print(f"    k={k}: {len(words):5d} legal words -> {len(killed):5d} "
              f"ZERO BY THEOREM, {len(alive):5d} need SAT"
              + ("   ==> N_%d = 0 BY THEOREM" % k if words and not alive
                 else ""))
        if alive[:6]:
            print(f"          survivors (first 6): {alive[:6]}")
        if not words:
            break
        # for the screen, treat every non-obstructed word as possibly realised
        prev = {w: (0 if first_dead_gear([sum(w[:i]) for i in
                                          range(len(w) + 1)], M) else 1)
                for w in words}
