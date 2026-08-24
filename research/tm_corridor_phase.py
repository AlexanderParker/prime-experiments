"""Round 21 (constructor): THE CORRIDOR-PHASE TRANSFER CHAIN.

Mechanic's corridor resonance (docs/novel/corridor-resonance.md) says the gap
process's memory is CORRIDOR PHASE (mod 35), not last-gap value: the
value-level one-step chain (R36) over-predicts deep qualifying runs by x49
(machine 29 depth 3) and predicts NO deficit at lags 2-5 where the census
shows 0.51-0.68.  This script rebuilds the transfer chain with state =
LEFT-ENDPOINT RESIDUE MOD 35 (the 15 exposed classes E) and tests it against
full-period exact censuses taken in the same pass:

  (1) lag-j autocorrelation R(j) = E[b_0 b_j]/E[b]^2, j = 1..15
  (2) deep runs (m consecutive qualifying gaps), m = 1..6
for two indicators: b = residue-qualifying (g mod q' in {0, +-2c}) and
b = size floor (g >= 2u').

Three nested models, every one built from exact full-period counts:
  VALUE  chain: state = last gap value               (R36 baseline, rebuilt)
  PHASE  chain: state = left endpoint mod 35         (the corridor ask)
  HYBRID chain: state = (phase mod 35, gap value)    (both memories)

Model predictions are floats (labeled MODEL); every census column is an
exact integer count.  Asserted cross-checks: the phase marginal is exactly
stationary for the phase chain (closed cycle); all three models reproduce
the m=1 marginal; V-run counts match research/data/tm_resid_runs.csv rows.

Usage: uv run python research/tm_corridor_phase.py y1 [y2 ...] [--seg N]
                                                   [--mod 35|385|...]
--mod extends the phase to E mod (product of the smallest gears): 35 is the
corridor ask; 385 tests whether the residual memory is gear-11 phase.
"""
import csv
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
from flank_envelope import primes_upto
from tm_resid_runs import next_prime

VMAX = 128          # gap values < VMAX
LAGMAX = 15         # deepest lag for R(j)
RUNMAX = 6          # deepest run
CTX = LAGMAX + RUNMAX + 8   # openings carried across segment boundaries


def census(y, seg=64_000_000, verbose=True, mod=35):
    """One full-period pass: exact counts of
       Cjoint[r, v]   gaps of value v with left endpoint r mod 35
       Ctrip[r, u, v] consecutive gap pairs (u then v), left endpoint of u = r
       pairs[b][j]    # (b_i and b_{i+j}),  j = 1..LAGMAX
       runs[b][m]     # positions with m consecutive b-gaps, m = 1..RUNMAX
    for b in {V (residue-qualifying), A (size >= 2u')}.  Every item counted
    exactly once (rightmost-gap ownership; cyclic seam stitched by wrapping
    the head openings - tm_resid_runs's caps rule)."""
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    q1 = next_prime(y)
    c = pow(6, -1, q1)
    Qres = np.zeros(q1, bool)
    for t in (0, (2 * c) % q1, (-2 * c) % q1):
        Qres[t] = True
    a_floor = 2 * round(q1 / 6)
    uvals = [pow(6, -1, g) for g in gears]

    Cjoint = np.zeros(mod * VMAX, np.int64)
    Ctrip = np.zeros(mod * VMAX * VMAX, np.int64)
    pairs = {"V": np.zeros(LAGMAX + 1, np.int64),
             "A": np.zeros(LAGMAX + 1, np.int64)}
    runs = {"V": np.zeros(RUNMAX + 1, np.int64),
            "A": np.zeros(RUNMAX + 1, np.int64)}
    ngaps = 0
    tail = None
    head = None
    t0 = time.time()

    def eat(ops, lo_new, caps=None):
        nonlocal ngaps, Cjoint, Ctrip
        d = np.diff(ops)
        n = len(d)
        if n == 0:
            return
        assert int(d.max()) < VMAX, "gap exceeds VMAX"
        r35 = (ops[:-1] % mod).astype(np.int64)         # left phase of gap i
        bV = Qres[d % q1]
        bA = d >= a_floor

        def newmask(L):
            m = ops[1:] >= lo_new
            if caps is not None:
                m = m & (ops[1:] <= caps(L))
            return m

        new1 = newmask(1)
        ngaps += int(new1.sum())
        sel = np.flatnonzero(new1)
        Cjoint += np.bincount(r35[sel] * VMAX + d[sel],
                              minlength=mod * VMAX).astype(np.int64)
        if n >= 2:
            s2 = np.flatnonzero(newmask(2)[1:])
            Ctrip += np.bincount(
                (r35[s2] * VMAX + d[s2]) * VMAX + d[s2 + 1],
                minlength=mod * VMAX * VMAX).astype(np.int64)
        for j in range(1, LAGMAX + 1):
            if n <= j:
                break
            mj = newmask(j + 1)[j:]
            pairs["V"][j] += int((bV[:-j] & bV[j:] & mj).sum())
            pairs["A"][j] += int((bA[:-j] & bA[j:] & mj).sum())
        for key, b in (("V", bV), ("A", bA)):
            for m in range(1, RUNMAX + 1):
                if n < m:
                    break
                ok = b[: n - m + 1].copy()
                for t in range(1, m):
                    ok &= b[t: n - m + 1 + t]
                runs[key][m] += int((ok & newmask(m)[m - 1:]).sum())

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
        if verbose and (lo // seg) % 32 == 0:
            print(f"  [census {y}] to {hi:.4g} ({100 * hi / P:.1f}%) "
                  f"{time.time() - t0:.0f}s", flush=True)
    eat(np.concatenate([tail, head + P]), P,
        caps=lambda L: P + int(head[min(L - 1, len(head) - 1)]))
    secs = time.time() - t0
    return dict(y=y, q1=q1, P=P, ngaps=ngaps, a_floor=a_floor, mod=mod,
                gears=gears,
                Cjoint=Cjoint.reshape(mod, VMAX),
                Ctrip=Ctrip.reshape(mod, VMAX, VMAX),
                pairs=pairs, runs=runs, Qres=Qres, secs=secs)


def analyse(r):
    y, q1, ngaps = r["y"], r["q1"], r["ngaps"]
    Cjoint, Ctrip = r["Cjoint"], r["Ctrip"]
    mod = r["mod"]
    print(f"\n=== machine {y}  q' = {q1}  period {r['P']:,}  "
          f"ngaps {ngaps:,}  phase mod {mod}  ({r['secs']:.0f}s census)")
    Ephase = np.flatnonzero(Cjoint.sum(1) > 0)
    nEexp = 1
    for g in r["gears"]:
        if mod % g == 0:
            nEexp *= g - 2
    assert len(Ephase) == nEexp, (len(Ephase), nEexp)
    assert Cjoint.sum() == ngaps and Ctrip.sum() == ngaps, "count/seam error"
    assert np.array_equal(Ctrip.sum(2), Cjoint), "pair/gap seam inconsistency"
    supp = np.flatnonzero(Cjoint.sum(0) > 0)
    F = int(supp[-1])
    Vset = np.array([v for v in supp if r["Qres"][v % q1]])
    print(f"  F = {F}   V(q') = {Vset.tolist()}   size floor a = {r['a_floor']}")
    # cross-check V runs against tm_resid_runs.csv if the row exists
    p = os.path.join(DDIR, "tm_resid_runs.csv")
    if os.path.exists(p):
        with open(p) as f:
            for row in csv.DictReader(f):
                if int(row["y"]) == y and int(row["qp"]) == q1:
                    assert int(row["ngaps"]) == ngaps, "ngaps mismatch"
                    for m in range(1, 5):
                        assert int(row[f"run{m}"]) == int(r["runs"]["V"][m]), \
                            (m, row[f"run{m}"], r["runs"]["V"][m])
                    print("  cross-check vs tm_resid_runs.csv: ngaps + "
                          "run1..4 EXACT MATCH")
                    break

    nE, nS = len(Ephase), len(supp)
    J = Cjoint[np.ix_(Ephase, supp)].astype(float)
    piPhase = J.sum(1) / ngaps
    Pv_r = J / J.sum(1, keepdims=True)                  # P(v | phase r)
    tgt = (Ephase[:, None] + supp[None, :]) % mod       # phase after the gap
    tix = np.minimum(np.searchsorted(Ephase, tgt), nE - 1)
    valid = Ephase[tix] == tgt
    # every OCCURRING (phase, value) pair must land back in E
    occ = Cjoint[np.ix_(Ephase, supp)] > 0
    assert valid[occ].all(), "an occurring transition left E - impossible"
    # zero-count combos may point outside E; their weight is 0 everywhere

    bvecs = {"V": np.isin(supp, Vset), "A": supp >= r["a_floor"]}

    # ---- PHASE chain matrices: M (full), Amat[b] (b-restricted) ----
    M = np.zeros((nE, nE))
    Amat = {k: np.zeros((nE, nE)) for k in bvecs}
    for iv in range(nS):
        col = tix[:, iv]
        for i in range(nE):
            M[i, col[i]] += Pv_r[i, iv]
            for k, b in bvecs.items():
                if b[iv]:
                    Amat[k][i, col[i]] += Pv_r[i, iv]
    assert np.abs(piPhase @ M - piPhase).max() < 1e-12, "stationarity"
    pvec = {k: Amat[k].sum(1) for k in bvecs}
    ev = np.linalg.eigvals(M)
    ev = ev[np.argsort(-np.abs(ev))]
    print(f"  phase-chain spectrum: lambda_1 = {np.abs(ev[0]):.6f}  "
          f"|lambda_2| = {np.abs(ev[1]):.6f}  (value {ev[1]:.6f})  "
          f"[phi/3 = 0.539345]")

    # ---- VALUE chain (R36 baseline, from this same exact pass) ----
    Cpair = Ctrip.sum(0)[np.ix_(supp, supp)].astype(float)
    Tval = Cpair / np.maximum(Cpair.sum(1, keepdims=True), 1e-300)
    piVal = Cjoint.sum(0)[supp].astype(float) / ngaps

    # ---- HYBRID chain: state (r, u); step via Ctrip probabilities ----
    TT = Ctrip[np.ix_(Ephase, supp, supp)].astype(float)
    rowsum = TT.sum(2)
    piHyb = TT.sum(2) / ngaps                            # approx state law

    def hstep(X, restrict=None):
        """One hybrid step: X[r, u] -> Xn[r', v], r' = r + u mod 35."""
        Xn = np.zeros((nE, nS))
        for iu in range(nS):
            w = X[:, iu]
            if not w.any():
                continue
            rs = rowsum[:, iu]
            probs = TT[:, iu, :] / np.where(rs > 0, rs, 1)[:, None]
            contrib = w[:, None] * probs
            if restrict is not None:
                contrib = contrib * restrict[None, :]
            np.add.at(Xn, tix[:, iu], contrib)
        return Xn

    # ---------------- deep runs ----------------
    print("\n  DEEP RUNS (exact = integer census; models = float)")
    print("   b   m        exact         indep     VALUE-ch     PHASE-ch"
          "    HYBRID-ch      val/ex     ph/ex     hy/ex")
    for key in ("V", "A"):
        b = bvecs[key]
        pb = r["runs"][key][1] / ngaps
        xv = piVal * b                       # value chain, after m=1 emit
        vph = piPhase @ Amat[key]            # phase chain, after m=1 emit
        Xh = piHyb * b[None, :]              # hybrid, current value is b
        for m in range(1, RUNMAX + 1):
            exact = int(r["runs"][key][m])
            ind = ngaps * pb ** m
            predval = ngaps * xv.sum()
            predph = ngaps * (piPhase @ pvec[key] if m == 1 else prevph)
            predhy = ngaps * Xh.sum()
            rat = (lambda x: f"{x / exact:9.3f}" if exact else "      inf" if x > 0.5 else "        -")
            print(f"   {key}  {m}  {exact:>12,}  {ind:>12.2f} {predval:>12.2f}"
                  f" {predph:>12.2f} {predhy:>12.2f}  {rat(predval)} "
                  f"{rat(predph)} {rat(predhy)}")
            # advance to m+1
            xv = (xv @ Tval) * b
            prevph = vph @ pvec[key]
            vph = vph @ Amat[key]
            Xh = hstep(Xh, restrict=b)
        # marginal sanity: all three m=1 predictions equal the census
        assert abs(ngaps * (piVal @ b) - r["runs"][key][1]) < 1e-6 * ngaps
        assert abs(ngaps * (piPhase @ pvec[key]) - r["runs"][key][1]) \
            < 1e-6 * ngaps

    # ---------------- lag correlations ----------------
    print("\n  LAG CORRELATIONS R(j) = E[b0 bj]/E[b]^2 "
          "(exact vs the three chains)")
    for key in ("V", "A"):
        b = bvecs[key]
        nb = int(r["runs"][key][1])
        pb = nb / ngaps
        print(f"   {key}  (p = {pb:.5g})")
        print("      j    exact     VALUE     PHASE    HYBRID")
        vval = (piVal * b) @ Tval
        vph = piPhase @ Amat[key]
        Xj = hstep(piHyb * b[None, :])
        for j in range(1, LAGMAX + 1):
            exact = int(r["pairs"][key][j]) / ngaps / pb ** 2
            predval = (vval @ b) / pb ** 2
            predph = (vph @ pvec[key]) / pb ** 2
            predhy = (Xj * b[None, :]).sum() / pb ** 2
            print(f"    {j:>3}  {exact:8.4f}  {predval:8.4f}  {predph:8.4f}"
                  f"  {predhy:8.4f}")
            vval = vval @ Tval
            vph = vph @ M
            Xj = hstep(Xj)


def main():
    args = sys.argv[1:]
    seg = 64_000_000
    mod = 35
    if "--seg" in args:
        i = args.index("--seg")
        seg = int(float(args[i + 1]))
        del args[i:i + 2]
    if "--mod" in args:
        i = args.index("--mod")
        mod = int(args[i + 1])
        del args[i:i + 2]
    for a in args:
        r = census(int(a), seg=seg, mod=mod)
        analyse(r)
        sys.stdout.flush()
    print("\nDone.")


if __name__ == "__main__":
    main()
