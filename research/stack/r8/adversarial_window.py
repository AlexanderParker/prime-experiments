"""Round 81 / loop entry 87: can the gears be ARRANGED to kill every twin slot in a window?

The counterexample hunt, asked structurally.  A machine fails if every column of its window
(q, q^2] is struck.  Each gear h strikes two classes of columns - one per member of the pair - and
in the real machine those two classes are fixed by arithmetic.  This asks the adversarial version:
if the two classes of every gear could be CHOSEN, could they cover the whole window?

If yes, the machine's protection is not structural: nothing about the shape of the gear set
forbids a twin-free window, and whatever forbids it is the arithmetic of the actual residues.
If no, there is a structural obstruction and it is worth finding.

Method: greedy adversary, gears in increasing order, each gear taking the two classes that cover
the most still-uncovered columns.  Greedy is a lower bound on what an adversary can do, so a
complete cover found by greedy settles the question for that machine.
"""

import sys


def primes_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def adversarial_cover(q, gears):
    """Greedy: each gear picks its two best classes. Returns columns left uncovered."""
    lo = q // 6 + 1
    hi = (q * q - 1) // 6
    n = hi - lo + 1
    covered = bytearray(n)
    for h in gears:
        for _ in range(2):
            best_r, best_gain = 0, -1
            for r in range(h):
                gain = 0
                k = lo + ((r - lo) % h)
                while k <= hi:
                    if not covered[k - lo]:
                        gain += 1
                    k += h
                if gain > best_gain:
                    best_gain, best_r = gain, r
            k = lo + ((best_r - lo) % h)
            while k <= hi:
                covered[k - lo] = 1
                k += h
    return n, n - sum(covered)


def true_open(q, sieve):
    lo = q // 6 + 1
    hi = (q * q - 1) // 6
    cnt = 0
    for m in range(lo, hi + 1):
        if sieve[6 * m - 1] and sieve[6 * m + 1]:
            cnt += 1
    return hi - lo + 1, cnt


def main():
    QS = [29, 37, 47, 59, 71, 101, 149]
    top = max(q * q for q in QS) + 10
    sieve = bytearray([1]) * (top + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(top ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))

    print("adversarial arrangement of the gears' teeth against the real arithmetic")
    print("     q   columns in window   gears   uncovered if teeth are CHOSEN   twins actually there")
    for q in QS:
        gears = [h for h in primes_to(q) if h >= 5]
        n, left = adversarial_cover(q, gears)
        n2, twins = true_open(q, sieve)
        print(
            "%6d  %17d  %6d  %30d  %20d"
            % (q, n, len(gears), left, twins)
        )
    print()
    print("uncovered = 0 means an adversary could kill every twin slot in that window,")
    print("so the machine's protection there is arithmetic and not structural.")


if __name__ == "__main__":
    sys.exit(main())
