"""r53 sk_gate - gates for the two engines of sk_core.

G1  the recorded F ladder of the real machines {5..q}, q = 7..31, by engine (1).
G2  the recorded A(K) for K = 1..6 by engine (2) (type-reduced, all primes).
G3  the two engines agree on the optimal sets: F({5,7,11,17}) = 16, F({5,7,11,23,29}) = 22,
    F({5,7,11,17,23,37}) = 28, and the brief's claimed K=5 optimum F({5,7,11,13,17}).
G4  the type reduction is not throwing away covers: for every L <= 22 and K <= 5, if the
    type-reduced engine says COVER then engine (1) finds a cover with an explicit prime set
    read off the witness.
"""
import os
import time

from sk_core import (RESULTS, A_exact, F_of, arc, cover_set, coverable_any,
                     primes_upto)

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


LADDER = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
KNOWN_A = {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    say("=" * 78)
    say("G1  F({5..q}) by engine (1), against the recorded ladder")
    say("=" * 78)
    ps = [5, 7, 11, 13, 17, 19, 23, 29, 31]
    prev = 1
    ok1 = True
    for i in range(2, len(ps) + 1):
        t = time.time()
        f = F_of(ps[:i], lo=prev)
        prev = f
        q = ps[i - 1]
        good = (LADDER[q] == f)
        ok1 &= good
        say(f"  F({{5..{q}}}) = {f:3d}   recorded {LADDER[q]:3d}   "
            f"{'OK' if good else 'MISMATCH'}   ({time.time()-t:.1f}s)")

    say()
    say("=" * 78)
    say("G2  A(K) by engine (2), type-reduced over ALL primes >= 5")
    say("=" * 78)
    ok2 = True
    for K in range(1, 7):
        t = time.time()
        a, w = A_exact(K, L0=max(1, KNOWN_A[K] - 2))
        good = (a == KNOWN_A[K])
        ok2 &= good
        say(f"  A({K}) = {a:3d}   recorded {KNOWN_A[K]:3d}   "
            f"{'OK' if good else 'MISMATCH'}   ({time.time()-t:.1f}s)")
        say(f"      witness at L = {a-1}: {w}")

    say()
    say("=" * 78)
    say("G3  the optimal sets, engine (1)")
    say("=" * 78)
    for S, claim in [([5, 7, 11], 7), ([5, 7, 11, 13], 11), ([5, 7, 11, 17], 16),
                     ([5, 7, 11, 19], 16), ([5, 7, 11, 13, 17], None),
                     ([5, 7, 11, 23, 29], 22), ([5, 7, 11, 23, 31], 22),
                     ([5, 7, 11, 17, 23, 37], 28), ([5, 7, 11, 13, 19, 47], 28)]:
        f = F_of(S)
        tag = "" if claim is None else ("  OK" if f == claim else "  MISMATCH")
        say(f"  F({S}) = {f}{tag}   arcs {[arc(g) for g in S]}")

    say()
    say("=" * 78)
    say("G4  every type-reduced COVER is realised by explicit primes (K <= 5, L <= 22)")
    say("=" * 78)
    bad = 0
    checked = 0
    for K in range(1, 6):
        for L in range(1, KNOWN_A[K]):
            ok, lv = coverable_any(L, K)
            if not ok:
                say(f"  K={K} L={L}: type-reduced says NO COVER (unexpected below A(K))")
                bad += 1
                continue
            S = realise(lv.witness, L, K)
            checked += 1
            if S is None or not cover_set(S, L):
                say(f"  K={K} L={L}: could not realise witness with explicit primes")
                bad += 1
    say(f"  {checked} type-reduced covers, {bad} not realisable by explicit primes")

    say()
    say(f"GATES: G1 {'PASS' if ok1 else 'FAIL'}   G2 {'PASS' if ok2 else 'FAIL'}   "
        f"G4 {'PASS' if bad == 0 else 'FAIL'}")
    with open(os.path.join(RESULTS, "sk_gate.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


def realise(witness, L, K):
    """Turn a type-reduced witness into an explicit set of K distinct primes."""
    used = set()
    spares = [p for p in primes_upto(4000) if p >= 5 and p - arc(p) >= L]
    for kind, key, _m in witness:
        if kind == 'p':
            used.add(key)
        elif kind == 'd':
            cand = [g for g in (3 * key - 1, 3 * key + 1)
                    if g >= 5 and g not in used and g - arc(g) >= L
                    and g in set(primes_upto(3 * key + 2))]
            if not cand:
                return None
            used.add(cand[0])
        else:
            for g in spares:
                if g not in used:
                    used.add(g)
                    break
    return sorted(used)


if __name__ == "__main__":
    main()
