"""Round 83 / loop entry 89: exact search - can ANY free configuration cover a window?

The annealing adversary (round 82) never reached zero uncovered columns.  Annealing is not
exhaustive.  This is: for a small machine, decide exactly whether two classes per gear, chosen
freely, can cover every column of the window (q, q^2].

Method: branch on the smallest uncovered column - some gear with a free slot must take the class
through it - with a capacity bound (remaining slots times the most any slot can cover must reach
the uncovered count).  Columns are bitmasks.

usage: uv run python research/stack/r8/exact_cover_search.py <q>
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
    classmask = {}
    percap = {}
    for h in gears:
        masks = []
        for r in range(h):
            m = 0
            k = lo + ((r - lo) % h)
            while k <= hi:
                m |= 1 << (k - lo)
                k += h
            masks.append(m)
        classmask[h] = masks
        percap[h] = max(bin(m).count("1") for m in masks)
    slots = {h: 2 for h in gears}
    nodes = [0]
    t0 = time.time()
    best = [n]

    def search(unc):
        nodes[0] += 1
        cnt = bin(unc).count("1")
        if cnt < best[0]:
            best[0] = cnt
        if unc == 0:
            return True
        cap = sum(slots[h] * percap[h] for h in gears)
        if cap < cnt:
            return False
        u = (unc & -unc).bit_length() - 1  # lowest uncovered column index
        col = lo + u
        for h in gears:
            if slots[h] == 0:
                continue
            r = col % h
            slots[h] -= 1
            if search(unc & ~classmask[h][r]):
                return True
            slots[h] += 1
        return False

    found = search(full)
    dt = time.time() - t0
    print("q = %d: %d gears, %d columns" % (q, len(gears), n))
    print("exact answer: %s" % ("a free configuration COVERS the window" if found else "NO free configuration covers the window"))
    print("fewest uncovered columns met during the search: %d" % best[0])
    print("nodes searched: %d in %.1f s" % (nodes[0], dt))


if __name__ == "__main__":
    main(sys.argv[1:])
