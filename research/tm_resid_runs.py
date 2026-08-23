"""Round 20 (constructor): exact census of RESIDUE-QUALIFYING gap runs.

The (D)-relevant p_j (suppression_law.py, R31/R32) uses the residue condition:
an interior gap g qualifies iff g mod q' in {0, +-2c} (c = 6^{-1} mod q') -
exactly the merge law's tooth-difference set.  This census computes, at FULL
period with the cyclic seam handled exactly:

  ngaps          number of gaps (= openings) per period
  nrun[m]        # of positions where m consecutive gaps ALL qualify, m=1..4
                 (a j-window's interior is j-2 gaps, so nrun[j-2] is p_j's
                 numerator for j = 3..6)
  F_j            max sum of j consecutive gaps (cross-check vs spectra CSV)
  qualmax_j      max sum of j consecutive gaps whose j-2 interiors all qualify

Every window is counted exactly once (counted at its rightmost gap; the seam
is stitched by wrapping the head openings after the last segment).

Usage: uv run python research/tm_resid_runs.py y1 [y2 ...] [--seg N]
Appends research/data/tm_resid_runs.csv.
"""
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
from flank_envelope import primes_upto

MAXJ = 6            # deepest window
CTX = 2 * MAXJ + 4  # openings kept across segment boundaries


def next_prime(y):
    p = y + 1
    while True:
        if all(p % d for d in range(2, int(p ** 0.5) + 1)):
            return p
        p += 1


def run(y, seg=64_000_000, verbose=True):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    q1 = next_prime(y)
    c = pow(6, -1, q1)
    Qres = np.array(sorted({0, (2 * c) % q1, (-2 * c) % q1}))
    uvals = [pow(6, -1, g) for g in gears]
    nrun = np.zeros(MAXJ + 1, np.int64)      # nrun[m], m = 1..MAXJ-2 used
    Fj = np.zeros(MAXJ + 1, np.int64)        # Fj[j], j = 1..MAXJ
    qualmax = np.zeros(MAXJ + 1, np.int64)   # qualmax[j], j = 3..MAXJ
    ngaps = 0
    tail = None
    head = None
    t0 = time.time()

    def eat(ops, lo_new, caps=None):
        """Process a stretch of consecutive openings; count each gap/window
        exactly once: only those whose RIGHTMOST gap ends at slot >= lo_new
        (and, in the seam pass, at most caps(L) for an item spanning L gaps -
        an item of L gaps ending later than the L-th wrapped opening was
        already counted in the first segment)."""
        nonlocal ngaps
        d = np.diff(ops)
        n = len(d)
        if n == 0:
            return

        def newmask(L):
            m = ops[1:] >= lo_new
            if caps is not None:
                m = m & (ops[1:] <= caps(L))
            return m

        new1 = newmask(1)                          # gap i is new
        ngaps += int(new1.sum())
        qual = np.isin(d % q1, Qres)
        # runs of m consecutive qualifying gaps
        for m in range(1, MAXJ - 1):
            if n < m:
                break
            ok = qual[: n - m + 1].copy()
            for t in range(1, m):
                ok &= qual[t: n - m + 1 + t]
            nrun[m] += int((ok & newmask(m)[m - 1:]).sum())
        # window sums, unrestricted and interior-qualifying
        for j in range(2, MAXJ + 1):
            if n < j:
                break
            s = d[: n - j + 1].astype(np.int64).copy()
            for t in range(1, j):
                s += d[t: n - j + 1 + t]
            sel = newmask(j)[j - 1:]
            if sel.any():
                Fj[j] = max(Fj[j], int(s[sel].max()))
            iok = np.ones(n - j + 1, bool)
            for t in range(1, j - 1):
                iok &= qual[t: n - j + 1 + t]
            iok &= sel
            if iok.any():
                qualmax[j] = max(qualmax[j], int(s[iok].max()))
        Fj[1] = max(Fj[1], int(d[new1].max())) if new1.any() else Fj[1]

    for lo in range(0, P, seg):
        hi = min(P, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        if head is None:
            head = op[:CTX].copy()
        ops = op if tail is None else np.concatenate([tail, op])
        eat(ops, lo)
        tail = ops[-CTX:].copy()
        if verbose and (lo // seg) % 4 == 0:
            print(f"  seg to {hi:.4g} ({100 * hi / P:.1f}%) "
                  f"{time.time() - t0:.0f}s", flush=True)
    # cyclic seam: wrap the head after the end.  An item spanning L gaps is
    # owned by the seam only if its rightmost opening is one of the first L
    # wrapped openings (later ones were countable inside the first segment).
    eat(np.concatenate([tail, head + P]), P,
        caps=lambda L: P + int(head[min(L - 1, len(head) - 1)]))
    secs = time.time() - t0
    return dict(y=y, q1=q1, P=P, gears=gears, Qres=Qres.tolist(), ngaps=ngaps,
                nrun=nrun, Fj=Fj, qualmax=qualmax, secs=secs)


def report_write(r):
    y, q1, ngaps = r["y"], r["q1"], r["ngaps"]
    print(f"\n=== machine {y}  q'={q1}  Qres={r['Qres']}  period {r['P']:,}"
          f"  ngaps {ngaps:,}  ({r['secs']:.0f}s)")
    p1 = r["nrun"][1] / ngaps
    print(f"  p_1V = {p1:.6g}  ({r['nrun'][1]:,} qualifying gaps)")
    for m in range(1, MAXJ - 1):
        c = int(r["nrun"][m])
        ind = ngaps * p1 ** m
        print(f"  run m={m} (j={m + 2}): {c:>12,}   indep {ind:>14,.2f}   "
              + (f"obs/indep {c / ind:8.5f}" if ind > 0 else ""))
    print("  j:      " + "  ".join(f"{j:>5}" for j in range(2, MAXJ + 1)))
    print("  F_j:    " + "  ".join(f"{int(r['Fj'][j]):>5}"
                                   for j in range(2, MAXJ + 1)))
    print("  qmax_j: " + "  ".join(f"{int(r['qualmax'][j]):>5}"
                                   for j in range(3, MAXJ + 1)).rjust(0))
    os.makedirs(DDIR, exist_ok=True)
    p = os.path.join(DDIR, "tm_resid_runs.csv")
    new = not os.path.exists(p) or os.path.getsize(p) == 0
    with open(p, "a") as f:
        if new:
            f.write("y,qp,ngaps,run1,run2,run3,run4,"
                    "F1,F2,F3,F4,F5,F6,qm3,qm4,qm5,qm6\n")
        f.write(f"{y},{q1},{ngaps},"
                + ",".join(str(int(r["nrun"][m])) for m in range(1, 5)) + ","
                + ",".join(str(int(r["Fj"][j])) for j in range(1, 7)) + ","
                + ",".join(str(int(r["qualmax"][j])) for j in range(3, 7))
                + "\n")
    print(f"  appended {p}")


def main():
    args = sys.argv[1:]
    seg = 64_000_000
    if "--seg" in args:
        i = args.index("--seg")
        seg = int(float(args[i + 1]))
        del args[i:i + 2]
    for a in args:
        r = run(int(a), seg=seg, verbose=True)
        report_write(r)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
