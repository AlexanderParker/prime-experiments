"""mo_excursion.py -- candidate (e): the maximum excursion of the opening walk.

X(M) := max over cyclic runs of gaps of ( span  -  (number of gaps) * mean gap )
      = max_j S_j - min_j S_j,   S_j = op(j) - j * P / N   (the centred opening walk).

It is exactly `max_J ( F_J(M) - J * mu(M) )`, the Legendre-type transform of the F_J curve at the
machine's own mean rate, and it BOUNDS the record: F(M) <= X(M) + mu(M) (take J = 1).  So it is
the first candidate on the list that is column-valued, bounds F, and is not itself one of the
three statements.  Everything below is exact: the walk is stored as the integer
N * S_j = N * op(j) - j * P, so no floating point enters the extremes.

Also reported: F_J - J*mu at each J (the profile whose maximum X is), and the argmax J.

Usage: uv run python research/anchor235/r66/mo_excursion.py
"""
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mo_core import OUT, base_gaps, fj_from_period


def excursion(gaps, P):
    """X(M) as an exact Fraction, plus the argmax/argmin positions of the integer walk."""
    N = gaps.size
    op = np.concatenate([[0], np.cumsum(gaps.astype(np.int64))[:-1]])   # openings from op(0) = 0
    Sn = N * op - P * np.arange(N, dtype=np.int64)                      # = N * S_j, integer
    jmax, jmin = int(Sn.argmax()), int(Sn.argmin())
    return Fraction(int(Sn[jmax]) - int(Sn[jmin]), N), jmax, jmin, int(Sn.max()), int(Sn.min())


def main():
    log = []
    for y in [5, 7, 11, 13, 17, 19, 23]:
        t0 = time.time()
        P, g = base_gaps(y)
        N = g.size
        X, jmax, jmin, smax, smin = excursion(g, P)
        mu = Fraction(P, N)
        # the profile F_J - J mu for J = 1 .. 40, and the J at which the excursion is attained
        fjs = fj_from_period(g, min(N, 40))
        prof = [(J, fjs[J - 1], float(Fraction(fjs[J - 1]) - J * mu))
                for J in range(1, len(fjs) + 1) if fjs[J - 1] is not None]
        argJ = max(prof, key=lambda t: t[2])
        # the run length that realises X (cyclic distance from jmin to jmax)
        runlen = (jmax - jmin) % N
        row = {"machine": y, "P": P, "N": N, "F": int(g.max()),
               "mu": float(mu), "mu_frac": [mu.numerator, mu.denominator],
               "X": float(X), "X_frac": [X.numerator, X.denominator],
               "F_le_X_plus_mu": float(X + mu) >= int(g.max()),
               "run_length_of_X": runlen,
               "span_of_X": int(np.int64(g.astype(np.int64)[np.arange(jmin, jmin + runlen) % N].sum())) if runlen else 0,
               "best_J_within_40": argJ[0], "best_val_within_40": argJ[2],
               "profile": [{"J": J, "Fj": f, "excess": e} for J, f, e in prof[:20]],
               "secs": round(time.time() - t0, 1)}
        log.append(row)
        print(f"m{y}: N={N:,} mu={float(mu):.5f} F={int(g.max())} | X={float(X):.4f} "
              f"(run of {runlen} gaps, span {row['span_of_X']}) | F <= X+mu: "
              f"{row['F_le_X_plus_mu']} (X+mu={float(X+mu):.3f}) | argmax J<=40: {argJ[0]} "
              f"[{row['secs']}s]", flush=True)
        with open(os.path.join(OUT, "excursion.json"), "w") as f:
            json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
