"""Round 75 / loop entry 80: the longest struck run inside the window, against what a proof needs.

`window_of_column_gap` (proofs/JacobsthalWindow.lean) turns a bound on the longest run of struck
columns into the window statement.  In the machine's units the input needed is

    the longest run of columns with no open column, inside (q, q^2],  <  the window's length,

which for the full periodic pattern is the paired Jacobsthal function j2(q#).  The project's
ladder proves j2(p_n#) << p_n^(4.266+eps); the window needs exponent 2.

This measures the truth: the longest run of consecutive struck columns actually occurring inside
each window, against the window's length and against (ln q)^2.
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
    QS = [101, 251, 503, 1009, 2003]
    top = max(q * q for q in QS) + 10
    sieve = bytearray([1]) * (top + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(top ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))

    print("longest run of struck columns inside the window, against the window and (ln q)^2")
    print("      q   window columns   longest struck run   run / (ln q)^2   open columns in window")
    for q in QS:
        lo = q // 6 + 1
        hi = (q * q - 1) // 6
        run = 0
        best = 0
        opens = 0
        for m in range(lo, hi + 1):
            a, b = 6 * m - 1, 6 * m + 1
            if sieve[a] and sieve[b]:
                opens += 1
                if run > best:
                    best = run
                run = 0
            else:
                run += 1
        if run > best:
            best = run
        lg = math.log(q) ** 2
        print(
            "%7d  %15s  %19d  %15.2f  %22s"
            % (q, "{:,}".format(hi - lo + 1), best, best / lg, "{:,}".format(opens))
        )

    print()
    print("what a proof needs: a bound on the run of the order of the window, i.e. exponent 2 in q")
    print("what the ladder proves: exponent 4.266 (fundamental lemma); the parity floor is 4")


if __name__ == "__main__":
    sys.exit(main())
