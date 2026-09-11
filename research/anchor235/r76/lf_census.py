"""lf_census.py -- the Liouville / Omega-parity census of the members 6k +- 1 inside the record
runs of the engines m23..m37, against two controls, and the sieve-data discrepancy of the
parity-twisted sequence on the prefix and on the record runs.

Constructions (research/proof/length_face.md):
  member n of column k: n in {6k - 1, 6k + 1};
  smooth part s(n) = the product of the prime powers of n with prime <= p (the engine's part);
  rough part r(n) = n / s(n);  n is STRUCK iff s(n) > 1;
  Omega(n) = the number of prime factors with multiplicity; lambda(n) = (-1)^Omega(n) (Liouville);
  sigma(k) = lambda(6k - 1) lambda(6k + 1), the column's parity sign;
  the sieve-visible parity of a column: (-1)^{Omega(s(6k-1)) + Omega(s(6k+1))}.

Sets measured:
  REC   = the first record run of the engine (first_realisation.md 3.3);
  RAND  = stretches of the same length L at uniformly random columns of [1, P/2] (unconditioned);
  RUNS  = fully struck runs of length >= Lmin found in a scan window (conditioned, non-record).

Sieve data on an interval I for d = a product of distinct gears:
  A_d = #{k in I : every gear of d strikes k},  S_d = sum over those k of sigma(k).
  The plain remainder r_d = A_d - 2^{omega(d)} |I| / d.  The parity twin's remainder is S_d / 2.

Usage: uv run python lf_census.py [--rand 300] [--runs 200]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from itertools import combinations

import numpy as np
from sympy import factorint, isprime, nextprime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lf_common import gears_of, period, struck_segment, runs_from_struck, u_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

RECORDS = {
    17: (118, 17),
    19: (111, 24),
    23: (12_694_429, 33),
    29: (200_906_186, 42),
    31: (1_468_940_243, 57),
    37: (90_816_580_903, 87),
}
# scan windows for the RUNS control and the minimum run length kept
RUN_WINDOWS = {
    17: (1, 40_000, 9),
    19: (1, 800_000, 12),
    23: (1, 18_000_000, 20),
    29: (1, 200_000_000, 28),
    31: (1_000_000_000, 1_100_000_000, 36),
    37: (1_000_000_000, 1_100_000_000, 45),
}

_FCACHE = {}


def fac(n):
    n = int(n)
    if n not in _FCACHE:
        _FCACHE[n] = factorint(n)
    return _FCACHE[n]


def member_stats(n, p):
    f = fac(n)
    om = sum(f.values())
    om_s = sum(e for q, e in f.items() if q <= p)
    om_r = om - om_s
    s = 1
    for q, e in f.items():
        if q <= p:
            s *= q ** e
    r = n // s
    ndist_s = sum(1 for q in f if 5 <= q <= p)
    return {"n": n, "om": om, "om_s": om_s, "om_r": om_r, "s": s, "r": r, "lam": (-1) ** om,
            "lam_r": (-1) ** om_r, "lam_s": (-1) ** om_s, "struck": s > 1, "ndist": ndist_s}


def stretch_stats(p, x, L):
    """Per-member and per-column facts of the columns x .. x+L-1."""
    cols = []
    for k in range(x, x + L):
        a = member_stats(6 * k - 1, p)
        b = member_stats(6 * k + 1, p)
        cols.append((k, a, b))
    st = {}
    mem = [m for (_, a, b) in cols for m in (a, b)]
    struck = [m for m in mem if m["struck"]]
    rough = [m for m in mem if not m["struck"]]
    st["L"] = L
    st["members"] = len(mem)
    st["n_struck"] = len(struck)
    st["n_rough"] = len(rough)
    st["both_struck_cols"] = sum(1 for (_, a, b) in cols if a["struck"] and b["struck"])
    st["open_cols"] = sum(1 for (_, a, b) in cols if not a["struck"] and not b["struck"])
    st["sum_lam_all"] = sum(m["lam"] for m in mem)
    st["sum_lam_struck"] = sum(m["lam"] for m in struck)
    st["sum_lam_rough"] = sum(m["lam"] for m in rough)
    st["sum_lam_r_struck"] = sum(m["lam_r"] for m in struck if m["r"] > 1)
    st["n_r_gt1_struck"] = sum(1 for m in struck if m["r"] > 1)
    st["sum_sigma"] = sum(a["lam"] * b["lam"] for (_, a, b) in cols)
    st["sum_sieve_parity"] = sum(a["lam_s"] * b["lam_s"] for (_, a, b) in cols)
    st["sum_striker_parity"] = sum((-1) ** (a["ndist"] + b["ndist"]) for (_, a, b) in cols)
    st["mult_mean"] = float(np.mean([a["ndist"] + b["ndist"] for (_, a, b) in cols]))
    st["odd_mult_cols"] = sum(1 for (_, a, b) in cols if (a["ndist"] + b["ndist"]) % 2 == 1)
    # Omega of the rough parts among the rough members (the sieve-invisible part)
    hist = {}
    for m in rough:
        hist[m["om_r"]] = hist.get(m["om_r"], 0) + 1
    st["rough_omega_hist"] = hist
    hist2 = {}
    for m in struck:
        if m["r"] > 1:
            hist2[m["om_r"]] = hist2.get(m["om_r"], 0) + 1
    st["struck_rough_omega_hist"] = hist2
    return st, cols


def z(sum_, n):
    return sum_ / math.sqrt(n) if n > 0 else 0.0


def sieve_data(p, cols_sigma, maxdim=None):
    """cols_sigma: list of (k, sigma). Returns per-d table over squarefree products of gears."""
    gears = gears_of(p)
    ks = np.array([k for k, _ in cols_sigma], dtype=np.int64)
    sg = np.array([s for _, s in cols_sigma], dtype=np.int64)
    N = len(ks)
    hit = {}
    for g in gears:
        u = u_of(g)
        hit[g] = (ks % g == u % g) | (ks % g == (-u) % g)
    rows = []
    mx = len(gears) if maxdim is None else maxdim
    for r in range(0, mx + 1):
        for D in combinations(gears, r):
            m = np.ones(N, dtype=bool)
            d = 1
            for g in D:
                m &= hit[g]
                d *= g
            A = int(m.sum())
            S = int(sg[m].sum())
            rem = A - (2 ** r) * N / d
            rows.append((d, r, A, S, rem))
    return rows


def summarise_sieve(rows):
    A1 = [rw for rw in rows if rw[2] >= 1]
    worst = max(A1, key=lambda rw: abs(rw[3]) / math.sqrt(rw[2]))
    big_rem = max(rows, key=lambda rw: abs(rw[4]))
    n_A_pos = len(A1)
    ratio_over_2 = sum(1 for rw in A1 if abs(rw[3]) / math.sqrt(rw[2]) > 2.0)
    ratio_over_3 = sum(1 for rw in A1 if abs(rw[3]) / math.sqrt(rw[2]) > 3.0)
    return {
        "d_count": len(rows), "d_with_A_pos": n_A_pos,
        "max_|S|/sqrtA": abs(worst[3]) / math.sqrt(worst[2]), "at_d": worst[0], "A_there": worst[2], "S_there": worst[3],
        "n_ratio_gt2": ratio_over_2, "n_ratio_gt3": ratio_over_3,
        "max_|plain_rem|": abs(big_rem[4]), "at_d_rem": big_rem[0], "omega_rem": big_rem[1],
        "S_1": rows[0][3], "A_1": rows[0][2],
        "max_|S_d|": max(abs(rw[3]) for rw in rows), "max_|S_d|_d": max(rows, key=lambda rw: abs(rw[3]))[0],
    }


def prefix_sigma(p):
    """sigma(k) for the prefix 1 <= k < b, b = (p'^2 - 1)/6, with the open columns and their signs."""
    pp = int(nextprime(p))
    b = (pp * pp - 1) // 6
    out = []
    opens = []
    for k in range(1, b):
        a = member_stats(6 * k - 1, p)
        c = member_stats(6 * k + 1, p)
        s = a["lam"] * c["lam"]
        out.append((k, s))
        if not a["struck"] and not c["struck"]:
            opens.append((k, s, isprime(6 * k - 1) and isprime(6 * k + 1)))
    return b, out, opens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rand", type=int, default=300)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--engines", default="17,19,23,29,31,37")
    ap.add_argument("--prefix-engines", default="23,29,31,37,41,43,47,53")
    ap.add_argument("--seed", type=int, default=76)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.time()
    report = {}

    # ---- part A: the census inside the record runs and the two controls
    for p in [int(s) for s in a.engines.split(",")]:
        x, L = RECORDS[p]
        P = period(p)
        rec, rec_cols = stretch_stats(p, x, L)
        # exhibit: the kill map and signs of the record run
        exhibit = []
        for (k, m1, m2) in rec_cols:
            exhibit.append({"k": k, "n-": m1["n"], "s-": m1["s"], "lam-": m1["lam"], "omr-": m1["om_r"],
                            "n+": m2["n"], "s+": m2["s"], "lam+": m2["lam"], "omr+": m2["om_r"]})
        # RAND control
        rand_rows = []
        for _ in range(a.rand):
            xr = rng.randrange(1, P // 2 - L)
            st, _ = stretch_stats(p, xr, L)
            rand_rows.append(st)
        # RUNS control: fully struck runs of length >= Lmin in the window, excluding the record
        w0, w1, Lmin = RUN_WINDOWS[p]
        found = []
        chunk = 1 << 24
        xx = w0
        while xx < w1 and len(found) < 20 * a.runs:
            n = min(chunk, w1 - xx)
            arr = struck_segment(p, xx - 1, n + 2048)
            for (s, ln) in runs_from_struck(arr, xx - 1):
                if xx <= s < xx + n and ln >= Lmin and not (s == x):
                    found.append((s, ln))
            xx += n
        rng.shuffle(found)
        found = found[:a.runs]
        run_rows = []
        for (s, ln) in found:
            st, _ = stretch_stats(p, s, ln)
            run_rows.append(st)

        def pooled(rows, key, nkey):
            S = sum(r[key] for r in rows)
            N = sum(r[nkey] for r in rows)
            return S, N, (S / N if N else 0.0), (S / math.sqrt(N) if N else 0.0)

        keys = [("sum_lam_all", "members"), ("sum_lam_struck", "n_struck"), ("sum_lam_rough", "n_rough"),
                ("sum_lam_r_struck", "n_r_gt1_struck"), ("sum_sigma", "L"), ("sum_sieve_parity", "L"),
                ("sum_striker_parity", "L")]
        summary = {"p": p, "x": x, "L": L, "P": P, "record": rec, "n_rand": len(rand_rows), "n_runs": len(run_rows),
                   "runs_lengths": sorted(r["L"] for r in run_rows)}
        table = {}
        for key, nk in keys:
            row = {"REC": (rec[key], rec[nk], z(rec[key], rec[nk]))}
            row["RAND"] = pooled(rand_rows, key, nk)
            row["RUNS"] = pooled(run_rows, key, nk)
            # per-stretch mean distribution of the controls, and the record's percentile in it
            rr = [r[key] / r[nk] for r in run_rows if r[nk] > 0]
            rv = rec[key] / rec[nk] if rec[nk] else 0.0
            pct = (sum(1 for v in rr if v < rv) + 0.5 * sum(1 for v in rr if v == rv)) / len(rr) if rr else None
            row["REC_pct_in_RUNS"] = pct
            table[key] = row
        summary["table"] = table
        summary["rec_rough_omega_hist"] = rec["rough_omega_hist"]
        summary["rec_struck_rough_omega_hist"] = rec["struck_rough_omega_hist"]
        summary["rec_mult_mean"] = rec["mult_mean"]
        summary["rec_both_struck_cols"] = rec["both_struck_cols"]
        summary["rec_odd_mult_cols"] = rec["odd_mult_cols"]
        summary["rand_both_struck_mean"] = float(np.mean([r["both_struck_cols"] for r in rand_rows]))
        summary["rand_open_cols_mean"] = float(np.mean([r["open_cols"] for r in rand_rows]))
        summary["runs_mult_mean"] = float(np.mean([r["mult_mean"] for r in run_rows])) if run_rows else None
        summary["runs_both_struck_frac"] = float(np.mean([r["both_struck_cols"] / r["L"] for r in run_rows])) if run_rows else None
        # rough omega histogram pooled over RUNS and RAND
        def pool_hist(rows, key):
            h = {}
            for r in rows:
                for kk, v in r[key].items():
                    h[kk] = h.get(kk, 0) + v
            return h
        summary["runs_rough_omega_hist"] = pool_hist(run_rows, "rough_omega_hist")
        summary["rand_rough_omega_hist"] = pool_hist(rand_rows, "rough_omega_hist")
        summary["exhibit"] = exhibit
        # sieve data of the record run (all d) and of the parity twist
        sd = sieve_data(p, [(k, m1["lam"] * m2["lam"]) for (k, m1, m2) in rec_cols])
        summary["record_sieve"] = summarise_sieve(sd)
        report[f"m{p}"] = summary
        print(f"\n=== m{p}: record run of {L} at {x:,} (P = {P:,}); RAND {len(rand_rows)} stretches, RUNS {len(run_rows)} runs of length >= {Lmin}  t = {time.time() - t0:.0f}s")
        print(f"  struck members {rec['n_struck']} of {rec['members']}; both-struck columns {rec['both_struck_cols']} (RAND mean {summary['rand_both_struck_mean']:.1f}, RAND open cols mean {summary['rand_open_cols_mean']:.1f}); mean multiplicity {rec['mult_mean']:.3f} (RUNS {summary['runs_mult_mean']})")
        print(f"  {'statistic':22s} {'REC sum/N (z)':>22s} {'RAND mean (z)':>22s} {'RUNS mean (z)':>22s} {'REC pct in RUNS':>16s}")
        for key, nk in keys:
            r = table[key]
            print(f"  {key:22s} {r['REC'][0]:+5d}/{r['REC'][1]:<5d} ({r['REC'][2]:+.2f}) "
                  f"{r['RAND'][2]:+.4f}/{r['RAND'][1]:<6d} ({r['RAND'][3]:+.2f}) "
                  f"{r['RUNS'][2]:+.4f}/{r['RUNS'][1]:<6d} ({r['RUNS'][3]:+.2f}) "
                  f"{(r['REC_pct_in_RUNS'] if r['REC_pct_in_RUNS'] is not None else float('nan')):16.2f}")
        print(f"  rough-part Omega histogram of the rough members: REC {rec['rough_omega_hist']}  RUNS {summary['runs_rough_omega_hist']}  RAND {summary['rand_rough_omega_hist']}")
        rs = summary["record_sieve"]
        print(f"  record-run sieve data: {rs['d_count']} d, {rs['d_with_A_pos']} with A_d > 0; max |S_d|/sqrt(A_d) = {rs['max_|S|/sqrtA']:.2f} at d = {rs['at_d']} (A = {rs['A_there']}, S = {rs['S_there']}); ratios > 2: {rs['n_ratio_gt2']}, > 3: {rs['n_ratio_gt3']}; S_1 = {rs['S_1']} over A_1 = {rs['A_1']}; max |plain remainder| = {rs['max_|plain_rem|']:.2f} at d = {rs['at_d_rem']} (omega {rs['omega_rem']})")

    # ---- part B: the prefix [1, b): the parity twin's data
    pref = {}
    for p in [int(s) for s in a.prefix_engines.split(",")]:
        b, cs, opens = prefix_sigma(p)
        n_open = len(opens)
        n_minus = sum(1 for (_, s, _) in opens if s == -1)
        all_twin = all(t for (_, _, t) in opens)
        sd = sieve_data(p, cs, maxdim=min(len(gears_of(p)), 6))
        sm = summarise_sieve(sd)
        pref[f"m{p}"] = {"b": b, "columns": b - 1, "open": n_open, "open_sigma_minus": n_minus, "all_open_are_twin_primes": all_twin, "sieve": sm}
        print(f"\nprefix m{p}: b = {b} ({b - 1} columns), open columns {n_open}, with sigma = -1: {n_minus}, all open are twin primes: {all_twin}; "
              f"S_1 = {sm['S_1']} over {sm['A_1']}; max |S_d|/sqrt(A_d) = {sm['max_|S|/sqrtA']:.2f} at d = {sm['at_d']} (A = {sm['A_there']}, S = {sm['S_there']}); "
              f"ratios > 2: {sm['n_ratio_gt2']} of {sm['d_with_A_pos']}, > 3: {sm['n_ratio_gt3']}; max |plain rem| {sm['max_|plain_rem|']:.2f} (omega {sm['omega_rem']}), max |S_d| = {sm['max_|S_d|']} at d = {sm['max_|S_d|_d']}  t = {time.time() - t0:.0f}s")
    report["prefix"] = pref
    with open(os.path.join(RES, "census.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\nwritten results/census.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
