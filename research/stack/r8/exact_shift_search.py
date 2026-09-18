"""Round 90 / loop entry 96: what the adversarial residue vector must contain - and the exact search
over the vectors that actually exist.

The strike pattern of the gears on the column line is fixed by arithmetic; the residue vector of
the window's start only says WHERE the window sits in that pattern.  For each gear h it fixes one
number, the offset c_h of the window's start from the gear's first tooth, and the gear's second
tooth then sits at a FIXED distance from the first: the two teeth are the classes
m = +inv(6) and m = -inv(6) modulo h, which differ by 2 inv(6) = inv(3) modulo h.

So an adversarial vector has one free residue per gear, not two: the shift c_h, with the pair
(c_h, c_h + inv(3)) rigid.  The exact searches of entries 88 and 89 allowed both classes free and
were therefore generous to the adversary.  This is the exact search over the vectors that can
exist: for machine q, is there ANY choice of shifts (c_h) for the gears 5..q whose rigid tooth
pairs cover every column of the window?  If not, no integer at all - small or large - can start a
twin-free window for that machine's gear set.

Method: branch on the lowest uncovered column; some gear not yet fixed must cover it, through one
of its two teeth, which fixes that gear's shift.  Capacity bound on the unfixed gears.

usage: uv run python research/stack/r8/exact_shift_search.py <q>
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


def main(argv):
    q = int(argv[0]) if argv else 29
    gears = [h for h in primes_to(q) if h >= 5]
    lo = q // 6 + 2
    hi = (q * q - 1) // 6
    n = hi - lo + 1
    full = (1 << n) - 1
    masks = {}
    cap = {}
    inv3 = {}
    for h in gears:
        inv3[h] = pow(3, -1, h)
        ms = []
        for c in range(h):
            m = 0
            for r in (c, (c + inv3[h]) % h):
                k = r
                while k < n:
                    m |= 1 << k
                    k += h
            ms.append(m)
        masks[h] = ms
        cap[h] = max(bin(m).count("1") for m in ms)
    fixed = {h: False for h in gears}
    best = [n]
    nodes = [0]
    t0 = time.time()

    def search(unc):
        nodes[0] += 1
        cnt = bin(unc).count("1")
        if cnt < best[0]:
            best[0] = cnt
        if unc == 0:
            return True
        remaining = sum(cap[h] for h in gears if not fixed[h])
        if remaining < cnt:
            return False
        u = (unc & -unc).bit_length() - 1
        for h in gears:
            if fixed[h]:
                continue
            fixed[h] = True
            for c in {u % h, (u - inv3[h]) % h}:
                if search(unc & ~masks[h][c]):
                    return True
            fixed[h] = False
        return False

    found = search(full)
    dt = time.time() - t0
    print("q = %d: %d gears, %d columns, one free shift per gear with the tooth pair rigid" % (q, len(gears), n))
    print("exact answer: %s" % ("some residue vector KILLS the window" if found else "NO residue vector kills the window"))
    print("fewest uncovered columns met: %d;  nodes %d in %.1f s" % (best[0], nodes[0], dt))


if __name__ == "__main__":
    main(sys.argv[1:])
