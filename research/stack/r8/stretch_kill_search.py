"""Round 94 / loop entry 100: can a stretch be killed at all, with the strike law's own constraint?

The target is now one stretch wide (entry 99): for consecutive gears p < q, can the gears up to p
cover every column of (p^2, q^2] except the square's?  The stretch starts at p^2, so the phase of
every gear h at its start is fixed by the residue r = p mod h: the start column is congruent to
(r^2 + 5) inv(6) modulo h.  This is the strike law of proof_skeleton section 12 - a gear's teeth
fall i columns after p^2 exactly when h divides r^2 + 6i or r^2 + 6i - 2 - and it means the
adversary at a stretch is not free: its shift at each gear is a SQUARE residue, one of (h-1)/2
values, not one of h.

This searches exactly over the square-residue shifts: for each p, is there ANY assignment of
residues r_h (1 <= r_h < h) whose induced rigid tooth pairs cover the whole stretch (p^2, q^2]
apart from the square column?  If none exists, no prime p of that size could open a twin-free
stretch, whatever its residues; if one exists, the protection lies in which residue vectors real
primes carry.

usage: uv run python research/stack/r8/stretch_kill_search.py <p_max>
"""

import sys
import time


def primes_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def search_stretch(p, q):
    gears = [h for h in primes_to(p) if 5 <= h <= p]
    L = (q * q - p * p) // 6  # columns in (p^2, q^2]
    # the square column of q is the last one; the adversary need not cover it
    full = (1 << L) - 1
    sq_col = L - 1  # column whose upper member is q^2
    full &= ~(1 << sq_col)
    masks = {}
    cap = {}
    for h in gears:
        inv6 = pow(6, -1, h)
        seen = {}
        for r in range(1, h):
            c = ((r * r + 5) * inv6) % h  # start column mod h
            m = 0
            for tooth in (inv6 % h, (-inv6) % h):
                t = (tooth - c) % h
                while t < L:
                    m |= 1 << t
                    t += h
            seen[m] = True
        masks[h] = list(seen.keys())
        cap[h] = max(bin(m).count("1") for m in masks[h])
    fixed = {h: False for h in gears}
    best = [L]
    nodes = [0]

    def search(unc):
        nodes[0] += 1
        cnt = bin(unc).count("1")
        if cnt < best[0]:
            best[0] = cnt
        if unc == 0:
            return True
        if sum(cap[h] for h in gears if not fixed[h]) < cnt:
            return False
        u = (unc & -unc).bit_length() - 1
        for h in gears:
            if fixed[h]:
                continue
            fixed[h] = True
            for m in masks[h]:
                if (m >> u) & 1:
                    if search(unc & ~m):
                        return True
            fixed[h] = False
        return False

    t0 = time.time()
    found = search(full)
    return len(gears), L, found, best[0], nodes[0], time.time() - t0


def main(argv):
    pmax = int(argv[0]) if argv else 60
    ps = [x for x in primes_to(pmax + 200) if x >= 5]
    print("exact search over square-residue shifts: can the gears up to p cover the stretch (p^2, q^2]?")
    print("     p     q   gears   columns   coverable by some residue vector?   fewest uncovered   nodes      s")
    for p, q in zip(ps, ps[1:]):
        if p > pmax:
            break
        n, L, found, best, nodes, dt = search_stretch(p, q)
        print("  %4d  %4d  %6d  %8d   %-33s  %6d  %10d  %6.1f"
              % (p, q, n, L, "YES - a killer stretch exists" if found else "no", best, nodes, dt))


if __name__ == "__main__":
    main(sys.argv[1:])
