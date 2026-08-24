"""Round 21 (constructor): THE TWO-TEETH KILL SPACING LAW, reproduced.

Found live with the human 2026-08-24 (round-20 session); this script is its
reproduction with assertions, over the FULL JOINT PERIOD P*q' of every step
11->13 .. 29->31.

Setting: step M -> M+q'.  A merge window is a maximal run of consecutive
M-openings all killed by q' (opening k killed iff k = +-c mod q',
c = 6^{-1} mod q').  The window's k kills have k-1 interior spacings s_i
(= the merged window's interior gaps).

DERIVED PARTS (elementary arithmetic - asserted as theorems on every window):
  T1 (letters identity)  {2c mod q', -2c mod q'} = {2u', q'-2u'},
     u' = round(q'/6): the tooth-difference set IS the literal alphabet.
  T2 (residue law)       s_i mod q' in {0, 2c, -2c}  (endpoints in {+-c}).
  T3 (sign alternation)  a spacing = +2c mod q' moves the kill residue
     -c -> +c, = -2c moves +c -> -c, = 0 keeps it; so with padded spacings
     transparent the +- signs STRICTLY ALTERNATE, and within one window
     |#a - #b| <= 1 (non-padded spacings alternate the two letters).
  T4 (minimum)           every nonzero-class spacing >= 2u'; padded >= q'.
  T5 (FUEL-SPAN LAW)     k <= 1 + span/(2u') <= 1 + 3*span/(q'-1)
     (span = sum s_i): fuel is capped by the merged span in closed form -
     at most ~3L/q' kills in a window of interior span L, no census needed.

MEASURED PART (the live observation - checked, violations reported):
  M1 (value law)  every realized spacing VALUE is exactly 2u', q'-2u', or
     q' - never 2u'+q', 2q', ...: the residue classes admit those larger
     values; the machine does not realize them.

Method: the joint period = q' copies of the old machine's opening sequence;
copy t kills the openings with o mod q' in {+-c - t*P}.  Runs are found per
copy with explicit carry across copy boundaries and the cyclic seam, so
every window of the joint period is seen exactly once.

Usage: uv run python research/kill_spacing.py [y1 y2 ...]  (default 11..29)
"""
import sys
import time
import numpy as np
from math import prod
import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
from flank_envelope import primes_upto
from tm_resid_runs import next_prime


def old_machine(y, seg=64_000_000):
    """Return (d, om, N, P): cyclic gap array d (uint8), opening mod-q'
    array om (uint8), opening count N, old period P."""
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    q1 = next_prime(y)
    uvals = [pow(6, -1, g) for g in gears]
    ds, oms = [], []
    prev = None
    first = None
    for lo in range(0, P, seg):
        hi = min(P, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        if first is None:
            first = int(op[0])
        if prev is not None:
            ds.append(np.array([op[0] - prev], np.uint8))
        dd = np.diff(op)
        assert dd.max() < 256
        ds.append(dd.astype(np.uint8))
        oms.append((op % q1).astype(np.uint8))
        prev = int(op[-1])
    ds.append(np.array([first + P - prev], np.uint8))   # cyclic wrap gap
    d = np.concatenate(ds)
    om = np.concatenate(oms)
    return d, om, len(om), P


def run(y):
    t0 = time.time()
    q1 = next_prime(y)
    c = pow(6, -1, q1)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    assert {(2 * c) % q1, (-2 * c) % q1} == {a, b}, "T1 letters identity"
    plus, minus = (2 * c) % q1, (-2 * c) % q1

    d, om, N, P = old_machine(y)
    spacing_hist = np.zeros(3 * q1, np.int64)     # value -> count (v < 3q')
    big_spacings = []                             # any value >= 3q' (report)
    win_by_k = {}
    total_kills = 0
    carry = None          # trailing run: list of spacing values (k-1 of them)
    first_run0 = None     # leading run of copy 0 if it starts at index 0
    max_span = 0

    def check_run(spac, counted=False):
        """T2-T5 + M1 on one window's interior spacings.  counted=True when
        the run's k-tally and pair histogram were already handled by the
        vectorized path (deep interior runs); boundary runs come here
        uncounted and get everything."""
        nonlocal max_span
        k = len(spac) + 1
        if not counted:
            win_by_k[k] = win_by_k.get(k, 0) + 1
        if not spac:
            return
        s = np.asarray(spac, np.int64)
        res = s % q1
        assert np.isin(res, [0, plus, minus]).all(), ("T2", s.tolist())
        assert (s >= a).all(), ("T4a", s.tolist())
        assert (s[res == 0] >= q1).all(), ("T4pad", s.tolist())
        signs = np.where(res == plus, 1, np.where(res == minus, -1, 0))
        nz = signs[signs != 0]
        if len(nz) >= 2:
            assert (nz[1:] != nz[:-1]).all(), ("T3", s.tolist())
        assert abs(int((s == a).sum()) - int((s == b).sum())) <= 1, \
            ("T3count", s.tolist())
        span = int(s.sum())
        max_span = max(max_span, span)
        assert k <= 1 + span // a, ("T5", k, span)
        assert k <= 1 + 3 * span / (q1 - 1) + 1e-9, ("T5b", k, span)
        if not counted:
            for v in s.tolist():
                if v < 3 * q1:
                    spacing_hist[v] += 1
                else:
                    big_spacings.append(v)

    plus_c, minus_c = (2 * c) % q1, (-2 * c) % q1

    def bulk_check(sp):
        """Vectorized T2/T4/M1 + histogram on an array of spacings."""
        nonlocal max_span
        if len(sp) == 0:
            return
        res = sp % q1
        assert np.isin(res, [0, plus_c, minus_c]).all(), "T2"
        assert (sp >= a).all(), "T4a"
        pad = sp[res == 0]
        if len(pad):
            assert (pad >= q1).all(), "T4pad"
        small = sp[sp < 3 * q1]
        spacing_hist[:] = spacing_hist + np.bincount(
            small, minlength=len(spacing_hist))[:len(spacing_hist)]
        big = sp[sp >= 3 * q1]
        big_spacings.extend(big.tolist())

    for t in range(q1):
        r1 = (c - t * P) % q1
        r2 = (-c - t * P) % q1
        kb = (om == r1) | (om == r2)
        Ki = np.flatnonzero(kb)
        total_kills += len(Ki)
        if len(Ki) == 0:
            if carry is not None:
                check_run(carry)
                carry = None
            continue
        dif = np.diff(Ki)
        breaks = np.flatnonzero(dif != 1)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [len(Ki) - 1]])
        lens = ends - starts + 1
        nruns = len(starts)
        lead = bool(Ki[0] == 0)
        trail = bool(Ki[-1] == N - 1)
        assert not (lead and trail and nruns == 1), "whole copy killed"
        # --- boundary runs handled exactly, in the small carry logic ---
        lead_sp = [int(x) for x in d[Ki[0]:Ki[ends[0]]]] if lead else None
        trail_sp = [int(x) for x in d[Ki[starts[-1]]:Ki[-1]]] if trail \
            else None
        if carry is not None:
            if lead:
                lead_sp = carry + [int(d[N - 1])] + lead_sp
            else:
                check_run(carry)
            carry = None
        if t == 0 and lead:
            first_run0 = lead_sp          # defer for the final cyclic join
        elif lead:
            check_run(lead_sp)
        if trail:
            carry = trail_sp
        # --- interior runs, vectorized ---
        imask = np.ones(nruns, bool)
        if lead:
            imask[0] = False
        if trail:
            imask[-1] = False
        ilens = lens[imask]
        bc = np.bincount(ilens) if len(ilens) else np.zeros(1, np.int64)
        for k, n in enumerate(bc):
            if k >= 1 and n:
                win_by_k[k] = win_by_k.get(k, 0) + int(n)
        # pairs of consecutive kills inside interior runs
        adj = np.flatnonzero(dif == 1)          # pair (i, i+1) in Ki space
        pm = np.ones(len(adj), bool)
        if lead:
            pm &= adj >= ends[0]                # drop pairs in leading run
        if trail:
            pm &= adj < starts[-1]              # drop pairs in trailing run
        sp = d[Ki[adj[pm]]].astype(np.int64)
        bulk_check(sp)
        if len(sp):
            max_span = max(max_span, int(sp.max()))   # k=2 span = spacing
        # deep interior runs (k >= 3): full per-run checks (rare)
        for idx in np.flatnonzero(imask & (lens >= 3)):
            s0, e0 = Ki[starts[idx]], Ki[ends[idx]]
            check_run([int(x) for x in d[s0:e0]], counted=True)
        # interior k=2 runs already counted in win_by_k and bulk-checked

    # final cyclic seam: trailing carry of copy q'-1 joins copy 0's lead run
    if carry is not None and first_run0 is not None:
        check_run(carry + [int(d[N - 1])] + first_run0)
    else:
        if carry is not None:
            check_run(carry)
        if first_run0 is not None:
            check_run(first_run0)

    assert total_kills == 2 * N, (total_kills, 2 * N)
    secs = time.time() - t0
    kmax = max(win_by_k) if win_by_k else 0
    nwin = sum(win_by_k.values())
    print(f"\n=== step {y} -> {q1}   joint period {P * q1:,}   ({secs:.0f}s)")
    print(f"  letters a = 2u' = {a}, b = q'-2u' = {b}; kills {total_kills:,} "
          f"(= 2N exactly)")
    print(f"  windows {nwin:,}; by k: "
          + "  ".join(f"k={k}: {n:,}" for k, n in sorted(win_by_k.items())))
    vals = np.flatnonzero(spacing_hist)
    print(f"  spacing values realized: "
          + "  ".join(f"{v}: {spacing_hist[v]:,}" for v in vals)
          + (f"  PLUS BIG {sorted(set(big_spacings))}" if big_spacings else ""))
    ok_M1 = set(vals.tolist()) <= {a, b, q1} and not big_spacings
    print("  M1 value law {a, b, q'}: "
          + ("HOLDS - every spacing is exactly 2u', q'-2u' or q'" if ok_M1
             else f"VIOLATED: extra values "
                  f"{sorted(set(vals.tolist()) - {a, b, q1})}"))
    print(f"  max interior span {max_span}; k_max = {kmax}; "
          f"fuel-span cap 1 + {max_span}//{a} = {1 + max_span // a}; "
          f"T1-T5 asserted on every window")
    # cross-check k_max - 1 <= deepest V-run of the old machine
    p = os.path.join(DDIR, "tm_resid_runs.csv")
    if os.path.exists(p):
        with open(p) as f:
            for row in csv.DictReader(f):
                if int(row["y"]) == y and int(row["qp"]) == q1:
                    deepest = max([m for m in range(1, 5)
                                   if int(row[f"run{m}"]) > 0], default=0)
                    assert kmax - 1 <= deepest, (kmax, deepest)
                    print(f"  cross-check: k_max-1 = {kmax - 1} <= deepest "
                          f"V-run {deepest} (tm_resid_runs.csv) OK")
    return dict(y=y, q1=q1, a=a, b=b, win_by_k=win_by_k, kmax=kmax,
                ok_M1=ok_M1)


def main():
    ys = [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19, 23, 29]
    res = [run(y) for y in ys]
    print("\nSUMMARY  step   k_max   M1(value law)")
    for r in res:
        print(f"        {r['y']:>3}->{r['q1']:<3}  {r['kmax']}      "
              f"{'holds' if r['ok_M1'] else 'VIOLATED'}")
    print("Two-teeth kill spacing law: derived parts T1-T5 asserted at full "
          "joint period; measured part M1 as printed.")


if __name__ == "__main__":
    main()
