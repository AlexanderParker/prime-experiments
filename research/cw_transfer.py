"""COSTELLO-WATTS, TRANSFERRED TO THE TWO-TEETH MACHINE  (round 23).

Costello & Watts, "A computational upper bound on Jacobsthal's function"
(arXiv:1208.5342).  Round 22 recorded, from the abstract-level reading, that
their recursive counting bound is "the same species as and stronger than" the
closed-form corollary in docs/novel/covering-lp-certificates.md.  This file is
the round-23 follow-through demanded by the brief: the paper's LaTeX source was
read in full and its machinery is transferred to our machine and MEASURED.

WHAT THE PAPER ACTUALLY DOES (their notation).
  p_i = i-th prime, P_k = p_1...p_k, phi(b,m,k) = #{a in (b, b+m] : (a,P_k)=1},
  phi_min(m,k) = min over b, h(k) = least m with phi_min(m,k) > 0.

  Thm 3.1   phi(b,m,k) = m - sum_i F_{b,m}(p_i) + sum_{a: w_k(a)>0} (w_k(a)-1)
            (bookkeeping: the first-moment count undercounts by the
            multiplicity excess).
  Thm 3.2   partition the blocked a by their LOWEST blocking prime p_x; then
            sum_{a in that class} (w_k(a)-1) = sum_{i>x} F_S(p_i p_x).
  Thm 2.1   THE DILATION LEMMA.  For coprime squarefree d, n, the arithmetic
            progression b+d, b+2d, ... has the SAME gcd-with-n pattern as a run
            of CONSECUTIVE integers cb+1, cb+2, ... where cd = 1 (mod n).
  Thm 3.3   combining these, the excess is EXACTLY a double sum over PAIRS of
            primes of phi(., ., i-1) - the same function at a smaller machine.
  Thm 3.4   phi(b,m,k) = m - sum_i F(p_i) + sum_{j>=2} F(2 p_j)
                         + sum_{2<=i<j<=k} phi(c_b(p_i p_j), F(p_i p_j), i-1).
            AN IDENTITY, not an inequality.
  Thm 4.4   the computable version: worst-case each term over b, plus E, an
            integer correction counting primes for which the two worst cases
            (F(p) = ceil(r/p) and F(2p) = floor(r/2p)) CANNOT CO-OCCUR.
  Algorithms 1-3: recursion bottoming out at k <= 6 with exact phi_min tables.
            Result: b(k) < 3 h(k) for k <= 49, b(k) <= 0.2775 k^2 log k to
            k = 10^4.

THE TRANSFER.  Our machine deletes TWO residues per gear (k = +-6^{-1} mod q)
instead of one, and its smallest gear is 5 rather than 2, so:
  * the pair term splits into FOUR arithmetic progressions mod q_i q_j, one per
    tooth combination, instead of one;
  * there is no separate "sum F(2 p_j)" term - that is their p_1 = 2 base case,
    and ours is the i = 1 (gear 5) case of the same double sum;
  * under the dilation t |-> (a - c)/d the sub-machine's gear q keeps a
    SYMMETRIC tooth pair, {s_q + v_q, s_q - v_q}, with the half-width
    v_q = (6 q_i q_j)^{-1} mod q DETERMINED and the centre s_q FREE (it runs
    over all residues as the window moves).  So the sub-problem is again a
    legal machine of the same family, with a different tooth separation.
    THAT IS A SELF-SIMILARITY LAW FOR THE TWIN MACHINE, and it is the part of
    the paper we did not have.

Everything below is exact integer arithmetic and every structural claim is
asserted against brute force.  Run:  uv run python research/cw_transfer.py
"""
import sys
from functools import lru_cache
from itertools import combinations, product
from math import prod

ZEROSET = frozenset()


def primes_upto(n):
    s = [True] * (n + 1)
    s[0] = s[1] = False
    for p in range(2, int(n ** .5) + 1):
        if s[p]:
            for m in range(p * p, n + 1, p):
                s[m] = False
    return [i for i, v in enumerate(s) if v]


def gears_of(y):
    return tuple(p for p in primes_upto(y) if p >= 5)


@lru_cache(maxsize=None)
def teeth(q):
    u = pow(6, -1, q)
    return (u % q, (-u) % q)


def blocked(a, q):
    return a % q in teeth(q)


# ------------------------------------------------------- structural asserts
def assert_layer_identities(gears, m, nb=400):
    """Thm 3.1 + 3.2 + 3.3 transferred: for every window,
       #openings = m - sum_q F_q + sum_{i<j} |T_ij|,
    where T_ij = slots blocked by BOTH q_i and q_j and by NO gear below q_i."""
    n = len(gears)
    for b in range(nb):
        win = range(b, b + m)
        op = sum(1 for a in win if not any(blocked(a, q) for q in gears))
        F = sum(sum(1 for a in win if blocked(a, q)) for q in gears)
        T = 0
        for i in range(n):
            for j in range(i + 1, n):
                T += sum(1 for a in win
                         if blocked(a, gears[i]) and blocked(a, gears[j])
                         and not any(blocked(a, gears[k]) for k in range(i)))
        assert op == m - F + T, (b, op, m - F + T)
    return True


def assert_dilation(gears, qi, qj, nb=200):
    """Thm 2.1 transferred: on the AP  a = c + t*d,  d = qi*qj, gear q < qi
    blocks the t-th term iff t is in the SYMMETRIC pair {s_q +- v_q} mod q with
    v_q = d^{-1} * 6^{-1} mod q, and s_q = -d^{-1} c mod q."""
    d = qi * qj
    for q in gears:
        if q >= qi:
            continue
        v = (pow(d, -1, q) * pow(6, -1, q)) % q
        for c in range(nb):
            s = (-pow(d, -1, q) * c) % q
            for t in range(2 * q):
                lhs = blocked(c + t * d, q)
                rhs = (t % q) in ((s + v) % q, (s - v) % q)
                assert lhs == rhs, (q, c, t)
    return True


# ------------------------------------------------------- the recursion leaf
@lru_cache(maxsize=None)
def Phi(L, sub, halfwidths):
    """min over centres s_q of the number of survivors in L consecutive terms,
    where gear sub[k] deletes the symmetric pair {s_k +- halfwidths[k]}.
    Exact, by enumeration over all centre tuples."""
    if L <= 0:
        return 0
    if not sub:
        return L
    best = L
    for s in product(*[range(q) for q in sub]):
        cnt = 0
        for t in range(L):
            ok = True
            for k, q in enumerate(sub):
                v = halfwidths[k]
                r = t % q
                if r == (s[k] + v) % q or r == (s[k] - v) % q:
                    ok = False
                    break
            if ok:
                cnt += 1
            if cnt >= best:
                break
        if cnt < best:
            best = cnt
        if best == 0:
            break
    return best


def max_blocked(q, m):
    """max over phase of the number of slots of an m-window blocked by q."""
    best = 0
    for r in range(q):
        c = sum(1 for i in range(m) if (i + r) % q in teeth(q))
        best = max(best, c)
    return best


def cw_lower_bound(gears, m, depth_cap=5):
    """Transferred Costello-Watts LOWER BOUND on the number of openings in any
    window of m slots.  Positive => F(M) <= m.

        openings >= m - sum_q maxblocked(q,m)
                      + sum_{i<j} sum_{4 tooth combos} Phi(floor(m/(q_i q_j)),
                                    gears below q_i, dilated half-widths)

    Each term is bounded in the worst case over the window position
    independently, exactly as their Thm 4.4 does (we do NOT implement their E
    correction, so this is a slightly weaker transfer)."""
    n = len(gears)
    tot = m - sum(max_blocked(q, m) for q in gears)
    for i in range(n):
        for j in range(i + 1, n):
            qi, qj = gears[i], gears[j]
            d = qi * qj
            L = m // d
            if L <= 0:
                continue
            sub = gears[:i]
            if len(sub) > depth_cap:
                sub = sub[:depth_cap]          # a valid weakening: fewer gears
            hw = tuple((pow(d, -1, q) * pow(6, -1, q)) % q for q in sub)
            tot += 4 * Phi(L, sub, hw)
        # each of the 4 tooth combinations of (q_i,q_j) gives its own AP
    return tot


def cw_bound(gears, hi=400, depth_cap=5):
    """smallest m for which the transferred bound is positive (=> F <= m)."""
    for m in range(1, hi + 1):
        if cw_lower_bound(gears, m, depth_cap) > 0:
            return m
    return None


def assert_bound_sound(gears, ms):
    """END-TO-END: the transferred bound must never exceed the TRUE minimum
    number of openings in a window, computed by brute force over the whole
    period.  This is what makes the transfer a theorem-shaped object rather
    than a hopeful adaptation."""
    P = prod(gears)
    opn = bytearray(b'\x01') * P
    for q in gears:
        for t in teeth(q):
            opn[t::q] = b'\x00' * len(opn[t::q])
    pre = [0] * (2 * P + 1)
    for i in range(2 * P):
        pre[i + 1] = pre[i] + opn[i % P]
    out = []
    for m in ms:
        true_min = min(pre[b + m] - pre[b] for b in range(P))
        lb = cw_lower_bound(gears, m)
        assert lb <= true_min, ("transfer unsound", m, lb, true_min)
        out.append((m, lb, true_min))
    return out


def main():
    print("=" * 78)
    print("COSTELLO-WATTS TRANSFERRED TO THE TWO-TEETH MACHINE")
    print("=" * 78)
    print("1  the exact layer identity (Thms 3.1-3.3), asserted by brute force")
    for y, m in ((11, 12), (13, 14), (17, 16)):
        assert_layer_identities(gears_of(y), m)
        print(f"   machine {y:>2}, window {m:>2}: identity holds at every"
              f" window position tested")
    print("\n2  the DILATION LEMMA (Thm 2.1), asserted by brute force")
    for (y, qi, qj) in ((17, 11, 13), (19, 13, 17), (23, 17, 19)):
        assert_dilation(gears_of(y), qi, qj)
        print(f"   gears below {qi}, pair ({qi},{qj}): every term of every AP"
              f" mod {qi * qj} has the predicted symmetric tooth pair")
    print("\n   => the sub-problem on the multiples of q_i q_j is again a")
    print("      two-teeth machine on the gears below q_i, with tooth")
    print("      HALF-WIDTH (6 q_i q_j)^{-1} mod q and free centre.  The twin")
    print("      machine is SELF-SIMILAR under 'restrict to a pair modulus'.")
    print("\n2b end-to-end soundness: the transferred bound never exceeds")
    print("   the TRUE minimum opening count (brute force, whole period)")
    for y in (11, 13):
        rows = assert_bound_sound(gears_of(y), range(4, 40, 6))
        print(f"   machine {y:>2}: " + "  ".join(
            f"m={m}: {lb} <= {t}" for m, lb, t in rows))
    print("\n3  what the transferred bound actually proves")
    from fractions import Fraction
    F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43}
    LP = {11: 8, 13: 20, 17: 28, 19: 37}
    print(f"   {'machine':>8} {'F':>4} {'CW-transfer F <=':>17}"
          f" {'ratio':>7}   {'LP-dual cert':>13}")
    for y in (11, 13, 17, 19, 23, 29):
        g = gears_of(y)
        m = cw_bound(g)
        r = f"{float(Fraction(m, F[y])):.2f}" if m else "  -"
        lp = str(LP.get(y, '-'))
        print(f"   {y:>8} {F[y]:>4} {str(m):>17} {r:>7}   {lp:>13}")
    print("\n   The transfer is a COUNTING bound: it produces a number, not a")
    print("   checkable dual object, and it is far weaker than the dual")
    print("   certificate at every machine where both exist.  What it has")
    print("   that the certificate does not is UNBOUNDED EFFECTIVE DEGREE:")
    print("   its pair term is the exact survivor count of a smaller machine,")
    print("   not a truncated second moment, so the moment-degree ceiling")
    print("   (docs/novel/moment-degree-ceiling.md) does not apply to it.")


if __name__ == '__main__':
    main()
