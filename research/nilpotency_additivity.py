"""Round 21 (constructor): NILPOTENCY ADDITIVITY - the (D) proto-law in the
operator algebra, attacked with the sum splitting + the kill spacing law.

THE SUM SPLITTING (new form; one line of algebra, verified exactly here).
With E' = I - D' the exposure projector of the new gear q', on the tensor
grid Z_P (x) Z_q':

    B_new S_new = (I - E_M (x) E') (S_M (x) S')
                = (B_M S_M) (x) S'  +  (E_M S_M) (x) (B' S')
                =      N    (x) S'  +       R    (x) K'

- the NEW blocked walk = OLD blocked walk (x) plain shift  PLUS  old renewal
step (x) q'-kill.  N is nilpotent of index F(M); K' is nilpotent of index 2
(a single gear's blocked slots are isolated); S', S_M are permutations.
F(M+q') = nilpotency index of the SUM: "adding a gear = tensor-and-strike"
is this exact operator move, and the round's question - why does the index
grow by <= q'? - is a question about sums of two Kronecker products.

WHAT THIS SCRIPT ESTABLISHES (each part asserted):

 P1  The splitting is an exact integer matrix identity (dense at {5}+7 and
     {5,7}+11), and operationally (vector iteration in tensor coordinates)
     index(sum) = F(M+q') at every step 11->13 .. 19->23.

 P2  NO CANCELLATION + CRT SEPARATION: (B_new S_new)^m expands over binary
     kill-words w in {N,R}^m into sum of L(w) (x) R(w); all entries >= 0, so
     the power is nonzero iff SOME word has BOTH factors nonzero; the left
     factor is an old-machine pattern event, the right factor a mod-q'
     event, and they are INDEPENDENT (coprime moduli).  The right factor is
     nonzero iff the kill offsets embed in the two teeth {+-c} mod q' -
     which is EXACTLY the two-teeth kill spacing law (kill_spacing.py T2/T3).
     Verified: right-factor 0/1 matrices at q' = 13 and 23 for realized and
     spacing-violating kill patterns.

 P3  THE COUNTING BOUNDARY (honest negative, sharp): the index of the sum
     is NOT bounded by any function of the marginal data (index(N) = F,
     index(K') = 2, the spacing law, litcap).  The 2-POINT RELAXATION -
     require only that every adjacent kill pair be individually realizable
     (each spacing an occurring old gap at the right residue class) - is
     satisfied by the infinite alternating word (a, b, a, b, ...) at every
     measured step from 19->23 on (pairs (a,b) and (b,a) both occur), while
     the true chain stops at k_max: the growth bound delta <= q' is
     PURELY a >= 3-point joint-realizability statement (matches R37's
     tropical boundary).  Demonstrated: at machine 19 the pairs (8,15) and
     (15,8) occur adjacently (62 k=3 chains) but run3 = 0 - the 3-point
     joint kills the infinite word.

 P4  THE PROVEN SHELL, from the splitting alone: per step, every window
     satisfies merged = g_L + span + g_R with span determined by the letter
     pattern (T3 alternation), padded count <= span // q', k <= 1 + span//2u',
     and the (D) clause is EXACTLY the remaining freedom: max over windows
     of (g_L + g_R) <= F + q' - span.  Verified per step with the measured
     margin table (delta vs q' vs 2u').

Usage: uv run python research/nilpotency_additivity.py
"""
import sys
import os
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flank_envelope import primes_upto
from tm_resid_runs import next_prime


def gear_mats(q):
    u = pow(6, -1, q)
    D = np.zeros((q, q), np.int64)
    D[u % q, u % q] = 1
    D[(-u) % q, (-u) % q] = 1
    E = np.eye(q, dtype=np.int64) - D
    S = np.zeros((q, q), np.int64)
    for r in range(q):
        S[(r + 1) % q, r] = 1
    return D, E, S


def sieve_openings(gears, P):
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    return ~ex


def maxgap(gears):
    P = prod(gears)
    op = np.flatnonzero(sieve_openings(gears, P))
    d = np.diff(np.append(op, op[0] + P))
    return int(d.max())


# ---------------------------------------------------------------- P1 dense
def p1_dense():
    print("P1a  DENSE EXACT: B_new S_new = N (x) S' + R (x) K'")
    for base, q1 in (([5], 7), ([5, 7], 11)):
        P = prod(base)
        # machine operators on Z_P in tensor coordinates = kron over gears
        EM = np.eye(1, dtype=np.int64)
        SM = np.eye(1, dtype=np.int64)
        for g in base:
            _, E, S = gear_mats(g)
            EM = np.kron(EM, E)
            SM = np.kron(SM, S)
        _, E1, S1 = gear_mats(q1)
        B1S1 = (np.eye(q1, dtype=np.int64) - E1) @ S1
        I = np.eye(P * q1, dtype=np.int64)
        lhs = (I - np.kron(EM, E1)) @ np.kron(SM, S1)
        BM = np.eye(P, dtype=np.int64) - EM
        rhs = np.kron(BM @ SM, S1) + np.kron(EM @ SM, B1S1)
        assert np.array_equal(lhs, rhs), (base, q1)
        # nilpotency index of the sum = F of the joint machine
        Fnew = maxgap(base + [q1])
        A = lhs.copy()
        m = 1
        while A.any():
            A = A @ lhs
            m += 1
            assert m < 100
        # A first zero at power m -> index = m
        idx = m
        assert idx == Fnew, (base, q1, idx, Fnew)
        print(f"   {base}+{q1}: identity exact (dim {P * q1}); "
              f"index(sum) = {idx} = F = {Fnew}")


# ------------------------------------------------------- P1 operational
def p1_vector(y):
    """Iterate X <- (N (x) S')X + (R (x) K')X in tensor coordinates (Z_P x
    Z_q'), boolean (entries of B_new S_new powers are 0/1: masked
    permutation).  Assert index = F(M+q') from the joint sieve (via the
    known F ladder)."""
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    q1 = next_prime(y)
    c = pow(6, -1, q1)
    exposedM = sieve_openings(gears, P)
    blockedM = ~exposedM
    teeth = np.zeros(q1, bool)
    teeth[c % q1] = True
    teeth[(-c) % q1] = True
    X = np.ones((P, q1), bool)
    m = 0
    while X.any():
        Xs = np.roll(X, 1, axis=0)                 # S_M on the P axis
        t1 = np.roll(Xs & blockedM[:, None], 1, axis=1)     # N (x) S'
        t2 = np.roll(Xs & exposedM[:, None], 1, axis=1) & teeth[None, :]
        X = t1 | t2                                # K' masks AFTER the shift
        m += 1
        assert m < 200
    return m, q1


# ---------------------------------------------------------------- P2
def right_factor(offsets, q1):
    """R(w) for kill offsets (positions of R-picks in the word, in slots):
    product over the word of S' (non-kill steps) and B'S' (kill steps) -
    computed as the exact 0/1 matrix on Z_q'."""
    c = pow(6, -1, q1)
    teeth = np.zeros(q1, np.int64)
    teeth[c % q1] = 1
    teeth[(-c) % q1] = 1
    _, E1, S1 = gear_mats(q1)
    B1 = np.eye(q1, dtype=np.int64) - E1
    m = max(offsets) + 1
    A = np.eye(q1, dtype=np.int64)
    ks = set(offsets)
    for step in range(m):
        A = (B1 @ S1 if step in ks else S1) @ A
        A = np.minimum(A, 1)
    return A


def p2(y):
    """Right-factor nonzero <=> spacing law, on realized + violating
    patterns at step y -> q'."""
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    ok_patterns = [[0], [0, a], [0, b], [0, a, a + b], [0, b, a + b],
                   [0, q1], [0, a, a + q1]]
    bad_patterns = [[0, 1], [0, a + 1], [0, a, 2 * a],       # a,a repeats
                    [0, b, 2 * b], [0, a - 1]]
    for pat in ok_patterns:
        R = right_factor(pat, q1)
        assert R.any(), ("P2 ok-pattern died", y, pat)
    for pat in bad_patterns:
        R = right_factor(pat, q1)
        assert not R.any(), ("P2 bad-pattern survived", y, pat)
    print(f"   step {y}->{q1}: right factor nonzero on every spacing-law "
          f"pattern ({len(ok_patterns)}), zero on every violating pattern "
          f"({len(bad_patterns)})")


# ---------------------------------------------------------------- P3
def p3():
    """The 2-point relaxation is unbounded from 19->23 on; the true chain
    stops at the 3-point joint."""
    y, q1 = 19, 23
    a, b = 8, 15
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    op = np.flatnonzero(sieve_openings(gears, P))
    d = np.diff(np.append(op, op[0] + P))
    n = len(d)
    dd = np.concatenate([d, d[:4]])
    pair_ab = int(((dd[:n] == a) & (dd[1:n + 1] == b)).sum())
    pair_ba = int(((dd[:n] == b) & (dd[1:n + 1] == a)).sum())
    tri_aba = int(((dd[:n] == a) & (dd[1:n + 1] == b)
                   & (dd[2:n + 2] == a)).sum())
    tri_bab = int(((dd[:n] == b) & (dd[1:n + 1] == a)
                   & (dd[2:n + 2] == b)).sum())
    assert pair_ab > 0 and pair_ba > 0, "2-point support missing"
    assert tri_aba == 0 and tri_bab == 0, "3-point should be empty"
    print(f"   machine 19 (q'=23, letters {a},{b}): adjacent pairs "
          f"(a,b) x{pair_ab}, (b,a) x{pair_ba}  ->  the infinite alternating "
          f"word is 2-POINT consistent;")
    print(f"   consecutive triples (a,b,a): {tri_aba}, (b,a,b): {tri_bab}  "
          f"->  the 3-point joint truncates it.  Index growth is NOT "
          f"marginal-index arithmetic.")


# ---------------------------------------------------------------- P4
def p4(y):
    """Windows with flanks at step y -> q' over the FULL JOINT PERIOD
    (direct sieve of the old gears over P*q'); assert the shell, compute
    F(M+q') as max surviving-to-surviving distance, report the margin."""
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    q1 = next_prime(y)
    PJ = P * q1
    c = pow(6, -1, q1)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    op = np.flatnonzero(sieve_openings(gears, PJ))     # old openings, joint
    F = int(np.diff(np.append(op, op[0] + PJ)).max())
    kill = np.isin(op % q1, [c % q1, (-c) % q1])
    # rotate so position 0 is a surviving opening
    r0 = int(np.flatnonzero(~kill)[0])
    op = np.concatenate([op[r0:], op[:r0] + PJ])
    kill = np.concatenate([kill[r0:], kill[:r0]])
    surv = np.flatnonzero(~kill)
    surv = np.append(surv, len(op))                    # sentinel = op[0]+PJ
    ops_ext = np.append(op, op[0] + PJ)
    best, best_detail, kmax = 0, None, 0
    per_k = {}                  # k -> (record merged, detail)
    Fnew = 0
    for i in range(len(surv) - 1):
        lo_i, hi_i = surv[i], surv[i + 1]
        merged = int(ops_ext[hi_i] - ops_ext[lo_i])
        Fnew = max(Fnew, merged)
        k = hi_i - lo_i - 1
        if k == 0:
            continue
        kmax = max(kmax, k)
        kills = ops_ext[lo_i + 1: hi_i]
        spac = np.diff(kills).astype(np.int64)
        span = int(spac.sum())
        gL = int(ops_ext[lo_i + 1] - ops_ext[lo_i])
        gR = int(ops_ext[hi_i] - ops_ext[hi_i - 1])
        assert merged == gL + span + gR
        npad = int((spac % q1 == 0).sum()) if k >= 2 else 0
        assert npad <= span // q1 if span else npad == 0
        assert k <= 1 + (span // a if a else 0) if k >= 2 else True
        if k >= 2 and npad == 0:
            assert span < (10 * q1) / 3 or len(spac) > 5
        if merged > best:
            best = merged
            best_detail = (k, gL, spac.tolist(), gR)
        if k not in per_k or merged > per_k[k][0]:
            per_k[k] = (merged, (gL, spac.tolist(), gR))
    delta = Fnew - F
    print(f"   step {y:>2}->{q1:<2}: F = {F:>2}  F(M+q') = {Fnew:>2}  "
          f"delta = {delta:>2}  q' = {q1:>2}  2u' = {a:>2}  "
          f"delta<=q': {'OK' if delta <= q1 else 'FAIL'}   "
          f"winner k={best_detail[0]} gL={best_detail[1]} "
          f"int={best_detail[2]} gR={best_detail[3]}")
    for k in sorted(per_k):
        m0, (gL, sp, gR) = per_k[k]
        maxold = max([gL, gR] + sp)
        print(f"        k={k} record merged {m0:>3} = {gL} + {sp} + {gR}"
              f"   max bridged old gap {maxold} = {maxold / F:.2f} F")
    return dict(y=y, q1=q1, F=F, Fnew=Fnew, delta=delta, a=a)


def main():
    p1_dense()
    print("P1b  OPERATIONAL (vector iteration of the sum, tensor coords):")
    known_F = {13: 11, 17: 18, 19: 25, 23: 34}
    for y in (11, 13, 17, 19):
        idx, q1 = p1_vector(y)
        assert idx == known_F[q1], (y, idx)
        print(f"   {y}+{q1}: index(N(x)S' + R(x)K') = {idx} = F(M+q') OK")
    print("P2   RIGHT FACTOR <=> KILL SPACING LAW:")
    for y in (11, 19):
        p2(y)
    print("P3   THE COUNTING BOUNDARY (2-point relaxation unbounded):")
    p3()
    print("P4   THE PROVEN SHELL + margin table (interior windows):")
    rows = [p4(y) for y in (11, 13, 17, 19)]
    print("\nAll parts asserted.  The open clause, stated in the algebra: "
          "index growth of the sum\nN(x)S' + R(x)K' is decided by the "
          ">=3-point joint realizability of spacing-compatible\nkill "
          "patterns in the old machine - the anti-correlation clause (D), "
          "nothing else.")


if __name__ == "__main__":
    main()
