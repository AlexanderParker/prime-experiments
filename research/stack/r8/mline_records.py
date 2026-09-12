"""The termination statement on the multiples of 6, as a field and as records (owner: both).

Landing family from home: (12 m - 1, 12 m + 1), m >= 1. Gear h strikes it iff 12 m = -+1 mod h,
i.e. m = -+12^-1 mod h: two teeth per gear, symmetric about the multiples of h. The m-line
field: rows the gears, columns m, row h painted at its two classes. Termination of the walk
for machine q: some m with q < 12 m - 1 and 12 m + 1 <= q^2 is unpainted in every row h <= q.
Measured here, exactly, on the m-line up to q^2 / 12 for primes q to qmax: the number of open
m in the window (q/12, q^2/12), the first open m above q/12 (the first landing), and the
record R(q) = the longest run of consecutive struck m below q^2 / 12 (the wheel's longest
closed stretch on this line), against the window length (q^2 - q) / 12; the walk terminates
for q as long as R(q) < window length. Also 12 R(q) / q^2, the record as a share of the window.
Usage: uv run python mline_records.py qmax
"""
import sys
import numpy as np
from sympy import primerange


def main():
    qmax = int(sys.argv[1])
    print("q | open m in the window | first landing m (12m-1) | record R(q): longest struck run of m below q^2/12 | window length in m | 12 R / q^2")
    worst = 0
    for q in primerange(11, qmax + 1):
        top = q * q // 12 + 1
        struck = np.zeros(top + 1, dtype=bool)
        for h in primerange(5, q + 1):
            inv = pow(12, -1, h)
            for t in (inv % h, (-inv) % h):
                struck[t::h] = True
                if t == 0: struck[0] = True
        struck[0] = True
        m = np.arange(top + 1)
        inwin = (12 * m - 1 > q) & (12 * m + 1 <= q * q)
        open_m = np.nonzero(inwin & ~struck)[0]
        # record: longest run of struck m among 1 .. top
        s = struck[1:top + 1]; run = 0; best = 0
        for v in s:
            run = run + 1 if v else 0
            if run > best: best = run
        wl = (q * q - q) // 12
        share = 12 * best / (q * q); worst = max(worst, share)
        if q in (11, 13, 31, 53, 101, 211, 401, 1009, 2003, 3001) or q == max(primerange(11, qmax + 1)):
            print(f"{q} | {len(open_m)} | {open_m[0] if len(open_m) else None} ({12 * open_m[0] - 1 if len(open_m) else None}) | {best} | {wl} | {share:.4f}")
    print(f"largest record share 12 R / q^2 over all machines to {qmax}: {worst:.4f} (termination needs it below 1 - 1/q)")


if __name__ == "__main__":
    main()
