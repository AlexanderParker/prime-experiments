"""Round 64 / loop entry 69: the carry wall, measured.

A mirror switches off exactly the gears dividing its product, and a landing at 2 M k inside the
window forces 2 M <= q^2.  Since every gear is at least 2, the carried gears number at most
log2(q^2) (kernel: `carried_le_log`, `mirror_in_window_carries_le`, proofs/MirrorWalkCarry.lean).

This measures the wall on the real gear sets: the largest primorial mirror that fits, how many
gears it carries against how many the machine has, how many periods the window then affords, and
how far the remaining gears are from the free regime of `keeping_move_free` (which needs every
uncarried gear above twice their number).
"""

import math
import sys


def primes_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def main():
    QS = [1009, 5003, 20011, 100003, 1000003, 10000019]
    allp = primes_to(10000019)
    print("largest primorial mirror that fits the window, and what is left to dodge")
    print("      q   gears  mirror B  carried  log2(q^2)  periods afforded  uncarried  free regime needs")
    for q in QS:
        gears = [p for p in allp if p <= q]
        M = 1
        B = 2
        carried = 0
        for p in gears:
            if 2 * M * p > q * q:
                break
            M *= p
            B = p
            carried += 1
        n_unc = len(gears) - carried
        nxt = next(p for p in gears if p > B)
        periods = (q * q - q) // (2 * M)
        print(
            "%8d  %6d  %8d  %7d  %9d  %16s  %9d  gears above %d (have %d)"
            % (q, len(gears), B, carried, int(math.log2(q * q)), "{:,}".format(periods), n_unc, 2 * n_unc, nxt)
        )
    print()
    print("the same trade at every primorial, q = 1000003")
    q = 1000003
    gears = [p for p in allp if p <= q]
    print("    B   carried   uncarried   mirror product 2M      periods afforded   free regime needs gears above")
    M = 1
    carried = 0
    for p in gears:
        if 2 * M * p > q * q:
            break
        M *= p
        carried += 1
        if p > 60:
            continue
        n_unc = len(gears) - carried
        print(
            "%5d  %8d  %10d  %20s  %20s  %10d"
            % (p, carried, n_unc, "%.6g" % (2 * M), "{:,}".format((q * q - q) // (2 * M)), 2 * n_unc)
        )


if __name__ == "__main__":
    sys.exit(main())
