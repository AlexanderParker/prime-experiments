"""rr_holes.py -- why the top of the spectrum has holes: the capacity margin and the shortest
uncoverable window, for the sizes just below the record.

For each size v near F(M): the gears' admissible classes (those leaving both ends of a v-gap
open), the total capacity  sum_g max_lam |{interior columns g strikes}|  against the demand v-1,
and -- for the EMPTY sizes -- the shortest sub-interval W of (0, v) that already cannot be covered
(the local-certificate question of pinned_arithmetic.md 3.4, asked at the spectrum's holes).

Usage: uv run python rr_holes.py <top gear> <vlo> <vhi>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr_record import PRIMES, gear_options, Enum          # noqa: E402


def window_feasible(gears, v, lo, hi):
    """can the columns [lo, hi) of the interior be covered, with 0 and v kept open?"""
    cols = list(range(lo, hi))
    opts = {}
    for g in gears:
        seen = set()
        for m, lam in gear_options(g, v):
            mm = 0
            for i, j in enumerate(cols):
                if (m >> (j - 1)) & 1:
                    mm |= 1 << i
            seen.add(mm)
        opts[g] = sorted(seen)
    full = (1 << len(cols)) - 1
    order = sorted(gears, key=lambda g: len(opts[g]))
    reach = [0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        r = 0
        for m in opts[order[i]]:
            r |= m
        reach[i] = reach[i + 1] | r

    def dfs(i, cov):
        if cov == full:
            return True
        if (full & ~cov) & ~reach[i]:
            return False
        if i == len(order):
            return False
        for m in opts[order[i]]:
            if dfs(i + 1, cov | m):
                return True
        return False
    return dfs(0, 0)


def main():
    top = int(sys.argv[1])
    vlo, vhi = int(sys.argv[2]), int(sys.argv[3])
    gears = [g for g in PRIMES if g <= top]
    print(f"M = {{5..{top}}}")
    print("| v | classes per gear | capacity sum_g max strikes | demand v-1 | margin | m(v) | "
          "shortest uncoverable window |")
    print("|---|---|---|---|---|---|---|")
    for v in range(vlo, vhi + 1):
        opts = {g: gear_options(g, v) for g in gears}
        cap = {g: max((bin(m).count("1") for m, _ in opts[g]), default=0) for g in gears}
        tot = sum(cap.values())
        e = Enum(gears, v, 2_000_000_000)
        sols, _ = e.run()
        short = "-"
        if not sols:
            best = None
            for L in range(2, v):
                for lo in range(1, v - L + 1):
                    if not window_feasible(gears, v, lo, lo + L):
                        best = (lo, lo + L, L)
                        break
                if best:
                    break
            short = f"[{best[0]}, {best[1]}) of {v - 1}, length {best[2]}" if best else "none"
        print(f"| {v} | {[cap and len(opts[g]) for g in gears]} | {tot} | {v - 1} | "
              f"{tot - (v - 1):+d} | {len(sols)} | {short} |")


if __name__ == "__main__":
    main()
