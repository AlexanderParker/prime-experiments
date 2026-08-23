"""Round 20 (mechanic): the TRANSFER-MATRIX SPECTRUM of the gap process -
the matrix-frame directive measured as exact events.

From the full-period gap-pair census (gap_pair_joint.csv, gap_pair_hist.csv)
build, per machine:

  T[u][v] = pair1[u][v] / ghist[u]   (measured lag-1 transition matrix on
                                      gap values - row-stochastic, exact
                                      ratio of two census counts)

and measure:

  1. its eigenvalue spectrum: Perron = 1; the SUBLEADING eigenvalue lam2.
     The observed lag-structure (deficit at lags 1-3, EXCESS at lags 4-5)
     is an oscillation; if lam2 is complex its argument predicts the
     oscillation period 2*pi/arg - a spectral statement, checked against
     the measured lag at which obs/indep first crosses 1.
  2. the Markov (one-step-memory) predictions of the lag-j joints:
     pred_pair_j = diag(ghist) @ T^j, compared to measured pair[j] on the
     threshold events "both >= a" - the exact share of the measured
     deficit that one-step memory explains.
  3. the Markov predictions of run events "all m consecutive >= a":
     start * (restricted chain)^{m-1}, vs measured minhist - how much of
     the run-suppression (the x26/x6.7/x1400 family) is one-step memory.

Every number is a census count or an exact function of census counts -
no fits.  Writes research/data/transfer_spectrum.csv.

Usage: uv run python research/transfer_spectrum.py [y ...]
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")
V = 128
FLOORS = {13: 4, 17: 6, 19: 6, 23: 8, 29: 10, 31: 12, 37: 14}


def load(y):
    ghist = np.zeros(V, np.int64)
    minh = np.zeros((7, V), np.int64)
    with open(os.path.join(DDIR, "gap_pair_hist.csv")) as f:
        next(f)
        for line in f:
            yy, cov, kind, idx, v, c = line.strip().split(",")
            if int(yy) != y:
                continue
            if kind == "ghist":
                ghist[int(v)] += int(c)
            else:
                minh[int(idx)][int(v)] += int(c)
    pair = np.zeros((6, V, V), np.int64)
    with open(os.path.join(DDIR, "gap_pair_joint.csv")) as f:
        next(f)
        for line in f:
            yy, cov, j, u, v, c = line.strip().split(",")
            if int(yy) != y:
                continue
            pair[int(j)][int(u)][int(v)] += int(c)
    return ghist, pair, minh


def analyse(y, out):
    ghist, pair, minh = load(y)
    if ghist.sum() == 0:
        print(f"machine {y}: no data")
        return
    ng = int(ghist.sum())
    vals = np.flatnonzero(ghist)
    F = int(vals[-1])
    a = FLOORS[y]
    # row-stochastic T on the support
    sup = vals
    P1 = pair[1][np.ix_(sup, sup)].astype(float)
    rows = P1.sum(axis=1)
    T = P1 / rows[:, None]
    pi = ghist[sup] / ng                      # stationary (exact shares)
    ev = np.linalg.eigvals(T)
    ev = ev[np.argsort(-np.abs(ev))]
    lam2 = ev[1]
    per = 2 * np.pi / abs(np.angle(lam2)) if abs(np.angle(lam2)) > 1e-9 \
        else np.inf
    print(f"\n=== machine {y}: F = {F}, {ng:,} gaps, support {len(sup)} "
          f"values, floor a = {a}")
    print(f"  eigenvalues by modulus: 1.0000, "
          + ", ".join(f"{abs(l):.4f}@{np.degrees(np.angle(l)):+.0f}deg"
                      for l in ev[1:5]))
    print(f"  |lam2| = {abs(lam2):.4f}  arg = "
          f"{np.degrees(np.angle(lam2)):+.1f} deg  -> predicted "
          f"oscillation period {per:.1f} lags")
    big = sup >= a
    p1 = float(pi[big].sum())
    # measured lag-j threshold correlations vs Markov T^j vs independent
    Tj = np.eye(len(sup))
    first_cross_obs, first_cross_mk = None, None
    print(f"  lag j:   obs/indep      markov/indep   (both gaps >= {a}; "
          f"indep = p1^2, p1 = {p1:.4f})")
    for j in range(1, 6):
        Tj = Tj @ T
        mk = float((pi[big, None] * Tj[np.ix_(big, big)]).sum()) / p1 ** 2
        npairs = int(pair[j].sum())
        obs = float(pair[j][np.ix_(sup[big], sup[big])].sum()) / npairs \
            / p1 ** 2
        print(f"    {j}      {obs:8.4f}       {mk:8.4f}")
        if first_cross_obs is None and obs > 1:
            first_cross_obs = j
        if first_cross_mk is None and mk > 1:
            first_cross_mk = j
        out.append((y, "lagratio", j, a, obs, mk))
    print(f"  first lag with obs/indep > 1: measured {first_cross_obs}, "
          f"markov-predicted {first_cross_mk}, eigen-period/2 = {per/2:.1f}")
    # run events: all m >= a, Markov prediction with the value-level chain
    # restricted to big values: start pi_big, propagate T[big,big]
    print(f"  run m:   observed        markov          indep     "
          f"obs/markov  (all m gaps >= {a})")
    w = pi[big].copy()
    for m in range(2, 7):
        w = w @ T[np.ix_(big, big)]
        nwin = int(minh[m].sum())
        mk_p = float(w.sum())                 # P(first >= a, all next >= a)
        obs_c = int(minh[m][a:].sum())
        mk_c = mk_p * nwin
        ind_c = p1 ** m * nwin
        r = obs_c / mk_c if mk_c else float("nan")
        print(f"    {m}    {obs_c:>12,}   {mk_c:>12,.0f}   {ind_c:>12,.0f}"
              f"   {r:8.4f}")
        out.append((y, "run", m, a, obs_c, mk_c))
    out.append((y, "lam2", 0, a, abs(lam2), np.degrees(np.angle(lam2))))


def main():
    ys = [int(x) for x in sys.argv[1:]] or [13, 17, 19, 23, 29, 31]
    out = []
    for y in ys:
        analyse(y, out)
    p = os.path.join(DDIR, "transfer_spectrum.csv")
    with open(p, "w") as f:
        f.write("y,kind,index,floor,observed,markov\n")
        for row in out:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
