"""Exact finite-y kappa(L) in the k-frame for small L, at any y, plus the
closed-form proof conditions for h(L) >= d at L = 1, 2, 3.

Method: N(L) = sum_{S subset [0,L)} (-1)^|S| X(S), X(S) = prod_q (q - r_q(S)),
r_q(S) = #{(u-s) mod q, (-u-s) mod q : s in S}, u = 6^{-1} mod q.
For q > 3L+1 there are no collisions, so r_q(S) = 2|S| and those primes
contribute a product depending only on |S|. Everything exact (ints/Fractions).

Findings recorded in docs/review-2026-08-17.md:
  - kappa(2) drifts down monotonically to the limit 2 - (11/3) C = 0.54477;
    it does not settle near 0.68.
  - h(2) >= d for all y >= 7, since it is equivalent to (11/3) C_y + E_y <= 2
    with C_y = prod (1 - 4/(q-2)^2), E_y = prod (q-4)/(q-2), both strictly
    decreasing, value 1.9111 at y = 7.
  - h(3) >= d for all y >= 7: sufficient condition (17/3) C_y + (14/3) E_y <= 3
    holds from y = 17; y = 7, 11, 13 verified exactly in integers here.
"""
from fractions import Fraction
from math import prod

def primes_upto(n):
    s = bytearray([1])*(n+1); s[0:2] = b'\x00\x00'
    for i in range(2, int(n**0.5)+1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [i for i in range(n+1) if s[i]]

def N_exact(y, Lmax, qs):
    """dict L -> N(L), exact, for L = 1..Lmax+1."""
    small = [q for q in qs if q <= 3*(Lmax+1)+1]
    big   = [q for q in qs if q >  3*(Lmax+1)+1]
    us = {q: pow(6, -1, q) for q in small}
    out = {}
    for L in range(1, Lmax+2):
        tail = [prod(q - 2*j for q in big) for j in range(L+1)]
        total = 0
        for mask in range(1 << L):
            S = [s for s in range(L) if mask >> s & 1]
            term = tail[len(S)]
            for q in small:
                u = us[q]
                rs = set()
                for s in S:
                    rs.add((u - s) % q); rs.add((-u - s) % q)
                term *= (q - len(rs))
            total += term if len(S) % 2 == 0 else -term
        out[L] = total
    return out

def kappa_table(y, Lmax):
    qs = [q for q in primes_upto(y) if q >= 5]
    d = Fraction(prod(q-2 for q in qs), prod(qs))
    N = N_exact(y, Lmax, qs)
    rows = []
    for L in range(1, Lmax+1):
        if N[L] == 0: break
        h = 1 - Fraction(N[L+1], N[L])
        rows.append((L, float((h/d - 1)/d)))
    return rows, float(d)

def proof_conditions(y):
    """Exact values of the monotone products in the L=2,3 proofs."""
    qs = [q for q in primes_upto(y) if q >= 5]
    P = prod(qs); A = prod(q-2 for q in qs); B1 = prod(q-4 for q in qs)
    Cy = Fraction(P*B1, A*A)
    Ey = Fraction(B1, A)
    return float(Fraction(11,3)*Cy + Ey), float(Fraction(17,3)*Cy + Fraction(14,3)*Ey)

if __name__ == "__main__":
    Lmax = 11
    for y in [23, 47, 101, 199, 401, 997, 2003, 5003, 10007]:
        rows, d = kappa_table(y, Lmax)
        mn = min(k for _, k in rows)
        c2, c3 = proof_conditions(y)
        print(f"y={y:6d} d={d:.5f} min kappa={mn:.4f} | "
              + "  ".join(f"{k:.4f}" for _, k in rows))
        print(f"         L=2 condition (<=2): {c2:.4f}   L=3 condition (<=3): {c3:.4f}")
