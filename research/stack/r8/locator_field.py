"""Machine-readable probes of the locator field (docs/locator_field.html): rows = gears h,
columns = natural numbers m, row h painted at m where h strikes the candidate column
(m^2 + 6i - 2, m^2 + 6i). Row h is periodic in m with period h; its teeth are the square roots
of -(6i - 2) (left member) and -6i (right member) mod h.
Reported per offset i: the teeth count distribution over gears h up to hmax (0 = blind, 1 = a
root at 0 only, 2, 3, 4), the mean teeth per gear against the twin machine's 2, the mirror of
every row about the multiples of h (teeth come in pairs r, h - r), and the survivors of the
locator machine among the primes m in (sqrt q, q] for a few q against the survivors of the
twin machine among the columns of (q, q^2] (the two sieves side by side).
Usage: uv run python locator_field.py hmax
"""
import sys
from collections import Counter
from sympy import primerange, isprime, sqrt_mod


def teeth(h, i):
    L = set(sqrt_mod((-(6 * i - 2)) % h, h, all_roots=True) or [])
    R = set(sqrt_mod((-6 * i) % h, h, all_roots=True) or [])
    return L, R


def main():
    hmax = int(sys.argv[1]); gears = list(primerange(5, hmax + 1))
    for i in (2, 5, 10, 12, 17):
        dist = Counter(); tot = 0; mirror_ok = True
        for h in gears:
            L, R = teeth(h, i); T = L | R; dist[len(T)] += 1; tot += len(T)
            if any((h - r) % h not in T for r in T): mirror_ok = False
        n = len(gears)
        print(f"offset {i}: gears to {hmax}: teeth 0 (blind) {dist[0]/n:.3f}, 1 {dist[1]/n:.3f}, 2 {dist[2]/n:.3f}, 3 {dist[3]/n:.3f}, 4 {dist[4]/n:.3f}; mean teeth {tot/n:.3f} (twin machine: 2); every row mirror-symmetric about the multiples of h: {mirror_ok}")
    print("the two sieves side by side: q | primes in (sqrt q, q] | locator survivors there (offset 10) | columns in (q, q^2] | twin survivors there")
    for q in (101, 307, 1009, 3001):
        ps = [m for m in primerange(2, q + 1) if m * m > q]
        hits = sum(1 for m in ps if isprime(m * m + 58) and isprime(m * m + 60))
        cols = (q * q - q) // 6
        tw = sum(1 for n in range(q + 1, q * q - 1) if n % 6 == 5 and isprime(n) and isprime(n + 2))
        print(f"  {q} | {len(ps)} | {hits} ({hits/len(ps):.3f}) | {cols} | {tw} ({tw/cols:.4f})")


if __name__ == "__main__":
    main()
