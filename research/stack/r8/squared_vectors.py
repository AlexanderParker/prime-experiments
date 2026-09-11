"""Which squared residue vectors leave a blind offset within a stretch of L = (g'^2 - g^2)/6?

Step g -> g': gears h in [5, g]; the origin's phase vector is phi_h = g^2 mod h; offset i (a column
above the square) is struck by h iff (phi_h + 6 i) mod h in {0, 2}; a blind offset is one no
gear strikes; the stretch is i = 1 .. L with L = (g'^2 - g^2)/6 (the new range).
For each prime g up to gmax, three vectors are tested for "leaves a blind offset in 1..L":
  real     : phi_h = (g mod h)^2            (the prime's own squared vector)
  sqrand   : phi_h = (r_h)^2, r_h uniform in 1..h-1 (a squared vector of a random non-zero residue vector)
  freerand : phi_h uniform in 0..h-1        (an arbitrary phase vector, no square structure)
with N samples each. Reported per g: L, the real vector's first blind offset, the fraction of
sqrand and freerand vectors with a blind offset in 1..L, and the mean number of blind offsets.
Usage: uv run python squared_vectors.py gmax N
"""
import sys, random
from sympy import primerange, nextprime

random.seed(11)


def first_blind(phi, gears, L):
    for i in range(1, L + 1):
        if all((phi[h] + 6 * i) % h not in (0, 2) for h in gears):
            return i
    return None


def count_blind(phi, gears, L):
    return sum(1 for i in range(1, L + 1) if all((phi[h] + 6 * i) % h not in (0, 2) for h in gears))


def main():
    gmax, N = int(sys.argv[1]), int(sys.argv[2])
    print("g | g' | L | real: first blind, count | sqrand: fraction with a blind offset, mean count | freerand: fraction, mean count")
    for g in primerange(5, gmax + 1):
        gp = nextprime(g); L = (gp * gp - g * g) // 6; gears = list(primerange(5, g + 1))
        real = {h: (g * g) % h for h in gears}
        fb = first_blind(real, gears, L); cb = count_blind(real, gears, L)
        sq_ok = sq_cnt = fr_ok = fr_cnt = 0
        for _ in range(N):
            sq = {h: (random.randrange(1, h) ** 2) % h for h in gears}
            fr = {h: random.randrange(0, h) for h in gears}
            c1 = count_blind(sq, gears, L); c2 = count_blind(fr, gears, L)
            sq_ok += c1 > 0; sq_cnt += c1; fr_ok += c2 > 0; fr_cnt += c2
        print(f"{g} | {gp} | {L} | {fb}, {cb} | {sq_ok/N:.3f}, {sq_cnt/N:.2f} | {fr_ok/N:.3f}, {fr_cnt/N:.2f}")


if __name__ == "__main__":
    main()
