"""The mirror walk on real mirror axes (owner, 2026-09-13): an axis is a multiple k M of the
product M of a chosen set S of gears (2 and 3 always in S). The pattern of the gears in S is
symmetric about every multiple of M, so a flip about k M, n -> 2 k M - n - 2 (column to
column), carries the openness of every gear in S. No offsets: the landing is wherever the
mirror puts it.

From home (-1, 1), open to every gear, one flip about k M lands on (2 k M - 1, 2 k M + 1):
open to every gear of S for every k. The remaining gears h of the machine (h <= q, h not in S)
strike that landing iff h divides 2 k M - 1 or 2 k M + 1, i.e. k = -(2M)^-1 or +(2M)^-1 mod h:
two classes of k per remaining gear, read off the roots. The rule: take the smallest k whose
landing lies in the window (q, q^2] and whose class avoids those two teeth for every remaining
gear. If such a k exists the landing is a twin: its members are below q^2 and no gear up to q
divides them (the S gears by the mirror, the rest by the choice of k).
Reported, machines 11 .. qmax, for S = {2,3}, {2,3,5}, {2,3,5,7}, {2,3,5,7,11}: machines with a
landing, the largest k tried, where in the window the landing sits (2kM / q^2, mean and max),
certification of every landing afterwards, and the failures.
Usage: uv run python true_mirror_walk.py qmax
"""
import sys
from sympy import primerange, isprime


def main():
    qmax = int(sys.argv[1]); primes = list(primerange(2, qmax + 1))
    qs = [q for q in primes if q >= 11]
    for S in ([2, 3], [2, 3, 5], [2, 3, 5, 7], [2, 3, 5, 7, 11]):
        M = 1
        for g in S: M *= g
        found = 0; kmax = 0; ksum = 0; pos = []; fails = []; none = []
        for q in qs:
            rest = [h for h in primes if h <= q and h not in S and h >= 5]
            k = max(1, (q + 1) // (2 * M) + 1); hit = None
            while 2 * k * M + 1 <= q * q:
                n = 2 * k * M - 1
                if n > q and all(n % h and (n + 2) % h for h in rest): hit = k; break
                k += 1
            if hit is None: none.append(q); continue
            found += 1; kmax = max(kmax, hit); ksum += hit; n = 2 * hit * M - 1; pos.append(2 * hit * M / (q * q))
            if not (isprime(n) and isprime(n + 2)): fails.append((q, n))
        print(f"S = {S}, M = {M}: landings found {found} of {len(qs)}; no landing at {none[:8]}; largest k {kmax}, mean k {ksum/max(found,1):.1f}; landing position 2kM/q^2 mean {sum(pos)/max(len(pos),1):.4f}, max {max(pos) if pos else None:.4f}; landings not twins {fails[:3]}")


if __name__ == "__main__":
    main()
