"""lf_twin.py -- the parity twin of the engine's opening set, on the columns 1 .. X.

Constructions (research/proof/length_face.md section 3):
  O(q)   = the columns k in [0, X] open under {5..q} (column 0 is open under every engine);
  sigma(k) = lambda(6k - 1) lambda(6k + 1);
  O+(q)  = {k in O : sigma(k) = +1} together with column 0;   O-(q) = {k in O : sigma(k) = -1}.
  Below q'^2 (q' the next prime) every open column is a twin prime pair, so sigma = +1 there and
  O- has no element below b = (q'^2 - 1)/6: its first gap is at least b, which exceeds q^2/6.

Measured here, per engine q in 11..37, on [0, X]:
  the record gap of O (should be F(q) when the record's first realisation lies below X);
  the record gap of O+ and of O-; the first element of O- with its members' factorisations;
  the record gap of O- EXCLUDING its initial gap; the record gaps of two one-sided twins
  (thin O by lambda(6k-1) alone, and by lambda(6k+1) alone); a random-thinning control (each
  opening kept independently with probability 1/2, 20 seeds); the share of O with sigma = -1 in
  dyadic bands of height above b.

Method: segmented sieve of Omega(n) (with multiplicity, by repeated division) and of the least
gear factor lpf5(n) in {5, 7, ..., 37} for n < 6X + 2; then per column k the two members.

Usage: uv run python lf_twin.py [--X 100000000] [--seg 20000000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sympy import factorint, nextprime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lf_common import primes_upto  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
F_KNOWN = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
X_REC = {11: 151, 13: 123, 17: 118, 19: 111, 23: 12_694_429, 29: 200_906_186, 31: 1_468_940_243, 37: 90_816_580_903}


def sieve_columns(X, seg):
    """Returns per-column arrays for k = 0..X: lam1, lam2 (int8, +-1), lpf1, lpf2 (int8; 0 = no
    gear factor <= 37) for the members 6k - 1 and 6k + 1 (column 0: members -1 -> treated as 1,
    and 1)."""
    N = 6 * X + 2
    lam1 = np.zeros(X + 1, dtype=np.int8)
    lam2 = np.zeros(X + 1, dtype=np.int8)
    lpf1 = np.zeros(X + 1, dtype=np.int8)
    lpf2 = np.zeros(X + 1, dtype=np.int8)
    primes = primes_upto(int(N ** 0.5) + 1)
    lo = 0
    t0 = time.time()
    while lo < N:
        hi = min(lo + seg, N)
        n = hi - lo
        rem = np.arange(lo, hi, dtype=np.int64)
        om = np.zeros(n, dtype=np.int8)
        lpf = np.zeros(n, dtype=np.int8)
        for p in primes:
            pk = int(p)
            while pk < hi:
                start = (-lo) % pk
                if start < n:
                    sl = slice(start, None, pk)
                    rem[sl] //= p
                    om[sl] += 1
                pk *= p
        om += (rem > 1).astype(np.int8)
        for g in reversed(GEARS):
            start = (-lo) % g
            lpf[start::g] = g
        lam = (1 - 2 * (om & 1)).astype(np.int8)
        # members 6k - 1 (n = 5 mod 6) and 6k + 1 (n = 1 mod 6) inside [lo, hi)
        # n = 6k - 1: k = (n + 1) / 6
        first5 = lo + ((5 - lo) % 6)
        if first5 < hi:
            idx = np.arange(first5, hi, 6)
            ks = (idx + 1) // 6
            lam1[ks] = lam[idx - lo]
            lpf1[ks] = lpf[idx - lo]
        first1 = lo + ((1 - lo) % 6)
        if first1 < hi:
            idx = np.arange(first1, hi, 6)
            ks = (idx - 1) // 6
            lam2[ks] = lam[idx - lo]
            lpf2[ks] = lpf[idx - lo]
        lo = hi
        print(f"  sieved to {hi:,} of {N:,}  ({time.time() - t0:.0f}s)", flush=True)
    # column 0: members -1 and 1 -> both units: lambda = +1, no gear factor
    lam1[0] = 1
    lpf1[0] = 0
    return lam1, lam2, lpf1, lpf2


def record_gap(idx):
    if idx.size < 2:
        return None, None
    d = np.diff(idx)
    j = int(np.argmax(d))
    return int(d[j]), int(idx[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", type=int, default=100_000_000)
    ap.add_argument("--seg", type=int, default=20_000_000)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    lam1, lam2, lpf1, lpf2 = sieve_columns(a.X, a.seg)
    # gate on the sieve: lambda by factorint at 200 random columns
    rng = np.random.default_rng(1)
    for k in rng.integers(1, a.X, 200):
        k = int(k)
        for n, l in ((6 * k - 1, lam1[k]), (6 * k + 1, lam2[k])):
            om = sum(factorint(n).values())
            assert (-1) ** om == int(l), (n, om, l)
            f = factorint(n)
            small = [q for q in f if q in GEARS]
            expect = min(small) if small else 0
            arr = lpf1 if n % 6 == 5 else lpf2
            assert int(arr[k]) == expect, (n, expect, int(arr[k]))
    print(f"lambda and lpf gates: 400 members agree with factorint  ({time.time() - t0:.0f}s)")
    sigma = (lam1.astype(np.int16) * lam2.astype(np.int16)).astype(np.int8)
    out = {"X": a.X}
    for q in [11, 13, 17, 19, 23, 29, 31, 37]:
        qp = int(nextprime(q))
        b = (qp * qp - 1) // 6
        open_ = ((lpf1 == 0) | (lpf1 > q)) & ((lpf2 == 0) | (lpf2 > q))
        O = np.flatnonzero(open_)
        sg = sigma[O]
        Op = O[sg == 1]
        Om = O[sg == -1]
        gO, atO = record_gap(O)
        gP, atP = record_gap(np.concatenate([[0], Op]) if Op.size and Op[0] != 0 else Op)
        # O- with column 0 prepended (so the initial gap is measured from the origin)
        Om0 = np.concatenate([[0], Om])
        gM, atM = record_gap(Om0)
        gM_ex, atM_ex = record_gap(Om) if Om.size >= 2 else (None, None)
        first_m = int(Om[0]) if Om.size else None
        fm = None
        if first_m is not None:
            fm = {"k": first_m, "6k-1": dict((int(p_), int(e)) for p_, e in factorint(6 * first_m - 1).items()),
                  "6k+1": dict((int(p_), int(e)) for p_, e in factorint(6 * first_m + 1).items())}
        # one-sided twins
        one1 = O[lam1[O] == -1]
        one2 = O[lam2[O] == -1]
        g1, _ = record_gap(np.concatenate([[0], one1]))
        g2, _ = record_gap(np.concatenate([[0], one2]))
        # random thinning
        thin = []
        for s in range(a.seeds):
            r = np.random.default_rng(1000 + s)
            keep = r.random(O.size) < 0.5
            keep[0] = True
            gg, _ = record_gap(O[keep])
            thin.append(gg)
        # sigma = -1 share by dyadic band above b
        bands = []
        lo = b
        while lo < a.X:
            hi = min(2 * lo, a.X)
            m = (O >= lo) & (O < hi)
            n = int(m.sum())
            nm = int((sg[m] == -1).sum())
            bands.append((int(lo), int(hi), n, nm, (nm / n if n else None)))
            lo = hi
        row = {"q": q, "q_next": qp, "b": b, "F_known": F_KNOWN[q], "x_rec": X_REC[q], "record_below_X": X_REC[q] <= a.X,
               "n_open": int(O.size), "n_plus": int(Op.size), "n_minus": int(Om.size),
               "gap_O": gO, "gap_O_at": atO, "gap_Oplus": gP, "gap_Oplus_at": atP,
               "gap_Ominus_from_origin": gM, "gap_Ominus_at": atM, "first_Ominus": fm,
               "gap_Ominus_excl_initial": gM_ex, "gap_Ominus_excl_at": atM_ex,
               "gap_one_sided_minus_lam1": g1, "gap_one_sided_minus_lam2": g2,
               "thin_records": thin, "thin_mean": float(np.mean(thin)), "thin_min": int(min(thin)), "thin_max": int(max(thin)),
               "bands": bands}
        out[f"m{q}"] = row
        print(f"\nm{q}: b = {b}, F = {F_KNOWN[q]} (record first at {X_REC[q]:,}, below X: {X_REC[q] <= a.X}); open {O.size:,} (+ {Op.size:,}, - {Om.size:,})")
        print(f"  record gap of O on [0, X]: {gO} at {atO:,};  O+: {gP} at {atP:,};  O- from the origin: {gM} at {atM:,} (initial gap = first O- column {first_m:,} = b + {first_m - b});  O- excluding the initial gap: {gM_ex} at {atM_ex:,}")
        print(f"  first O- column {first_m:,}: 6k-1 = {6 * first_m - 1} = {fm['6k-1']}, 6k+1 = {6 * first_m + 1} = {fm['6k+1']}")
        print(f"  one-sided twins (lambda(6k-1) = -1 / lambda(6k+1) = -1): records {g1}, {g2};  random half-thinning of O, {a.seeds} seeds: mean {np.mean(thin):.1f}, min {min(thin)}, max {max(thin)}")
        print("  share of O with sigma = -1 by band: " + "; ".join(f"[{lo:,},{hi:,}): {nm}/{n}" + (f" = {fr:.3f}" if fr is not None else "") for (lo, hi, n, nm, fr) in bands))
    with open(os.path.join(RES, "twin.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwritten results/twin.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
