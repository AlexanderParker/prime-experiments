"""R2.a.i.a.1.b reading (b) - which gears can reach island i, as a condition on g mod M.

Gear g reaches offset i iff -6i or 2-6i is a nonzero quadratic residue mod g.  For a fixed i the
two Legendre symbols (-6i/g) and ((2-6i)/g) are, by quadratic reciprocity, characters in g of
conductor dividing 4|s| where s is the squarefree kernel (sign included) of the argument, and
dividing |s| when s = 1 (mod 4).  So reachability of island i is a union of arithmetic
progressions of gears modulo M(i) = lcm of the two conductors.  This script computes M(i) exactly,
the exact class list, and verifies it against every prime gear up to 200,000.

Usage: uv run python research/anchor235/r42/sv_bar.py
"""
import os
from math import gcd, isqrt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def squarefree_kernel(n):
    """n = s * k^2 with s squarefree (sign kept on s)."""
    sgn = -1 if n < 0 else 1
    n = abs(n)
    s = 1
    d = 2
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d
            e += 1
        if e % 2:
            s *= d
        d += 1 if d == 2 else 2
    s *= n
    return sgn * s


def conductor(n):
    """minimal M such that the Kronecker symbol (n/.) is a character mod M on odd g."""
    s = squarefree_kernel(n)
    return abs(s) if s % 4 == 1 else 4 * abs(s)


def jacobi(a, n):
    """Jacobi symbol (a|n), n odd positive; equals the Legendre symbol when n is prime."""
    a %= n
    r = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                r = -r
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            r = -r
        a %= n
    return r if n == 1 else 0


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    r = pow(a, (p - 1) // 2, p)
    return 1 if r == 1 else -1


def reaches(i, p):
    """does prime gear p reach offset i (some q^2 = -6i or 2-6i mod p with the value a nonzero QR)"""
    return legendre(-6 * i, p) == 1 or legendre(2 - 6 * i, p) == 1


def main():
    fl = sieve(200000)
    primes = [p for p in range(11, 200001) if fl[p]]
    lines = []
    for i in (12, 47, 82):
        n1, n2 = -6 * i, 2 - 6 * i
        s1, s2 = squarefree_kernel(n1), squarefree_kernel(n2)
        c1, c2 = conductor(n1), conductor(n2)
        M = c1 * c2 // gcd(c1, c2)
        # classify every residue class mod M that can hold a gear
        cls_reach, cls_bar, undecided = [], [], 0
        seen = {}
        for p in primes:
            if n1 % p == 0 or n2 % p == 0:
                continue
            c = p % M
            r = reaches(i, p)
            if c in seen and seen[c] != r:
                undecided += 1
            seen[c] = r
        for c in sorted(seen):
            (cls_reach if seen[c] else cls_bar).append(c)
        tot = len(cls_reach) + len(cls_bar)
        lines.append("island i = %d :  -6i = %d = %d * square, 2-6i = %d = %d * square" % (i, n1, s1, n2, s2))
        lines.append("   conductors %d and %d  ->  M(%d) = %d ; classes met by a prime gear <= 200000: %d"
                     % (c1, c2, i, M, tot))
        lines.append("   reachable classes: %d (%.4f) ; BARRED classes: %d (%.4f) ; inconsistencies: %d"
                     % (len(cls_reach), len(cls_reach) / tot, len(cls_bar), len(cls_bar) / tot, undecided))
        lines.append("   first 12 barred classes mod %d: %s" % (M, cls_bar[:12]))
        lines.append("   first 12 reachable classes mod %d: %s" % (M, cls_reach[:12]))
        # exact class census by Jacobi symbols over every class mod M coprime to 2M
        er, eb = 0, 0
        for c in range(1, M, 2):
            if gcd(c, M) != 1:
                continue
            ok = (jacobi(s1 % c, c) == 1) or (jacobi(s2 % c, c) == 1)
            if ok:
                er += 1
            else:
                eb += 1
        lines.append("   EXACT census over classes mod %d coprime to 2M: reachable %d, barred %d (%.6f barred)"
                     % (M, er, eb, eb / (er + eb)))
        lines.append("")
    # offset 0 for reference (the classical +-1 mod 8 statement)
    b8 = {}
    for p in primes:
        b8.setdefault(p % 8, set()).add(legendre(2, p) == 1)
    lines.append("offset 0 (needs q^2 = 2): reaches iff g = +-1 mod 8 -> %s" % ({k: sorted(v) for k, v in sorted(b8.items())}))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "sv_bar.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
