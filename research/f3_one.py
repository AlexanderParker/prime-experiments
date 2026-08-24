"""Round 21 (mechanic): single-instance window solver for parallel F_j/Q_j
refutation sweeps.  Solves ONE (y, j, S) occurrence question and exits, so
independent S values can run concurrently (cov_sat.fjone descends serially).

Usage:
  uv run --with python-sat python research/f3_one.py y j S [a]
    a > 0 adds the qualifying middle-gap floor (Q_j instead of F_j).

Prints exactly one result line; witness (if SAT) is CRT'd back and
machine-verified by assert inside cov_sat.solve_window.
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
from cov_sat import gears_of, solve_window  # noqa: E402


def main():
    y, j, S = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    a = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    qs = gears_of(y)
    t0 = time.time()
    r, k, sp = solve_window(S, j, qs, min_middle=a)
    dt = time.time() - t0
    tag = f"y={y} j={j} S={S}" + (f" a={a}" if a else "")
    if r:
        gaps = [b - x for x, b in zip([0] + sp, sp + [S])]
        print(f"{tag} SAT k={k} openings=+{sp} gaps={gaps} ({dt:.0f}s)",
              flush=True)
    else:
        print(f"{tag} UNSAT ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
