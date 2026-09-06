"""Item 4: THE EXHAUST'S OWN RECORD (O-X4), and where the record of a tier lives.

A tier used on a range [1, N] has the two zones of that range: the SMOOTH ZONE [1, Q] and the
QUIET ZONE (Q, Q^2], with Q the largest gear used.  For tier 3 at base q, "smooth" means
q#-smooth, and by L59 the bottom stratum (Q, 2Q] of the quiet zone contains only primes and
q#-smooth numbers, so its open pairs are the tier's family (1, 1) - twin primes above Q - together
with pairs having a q#-smooth member.  This script measures:

  * the record on [1, N] and its position (smooth zone or quiet zone) - a sweep in N;
  * the smooth-zone record from the finite list of q#-smooth pairs (L47's form);
  * the record of the quiet zone and its dyadic strata (L64's U-profile);
  * the proved prime-gap lower bound of L63 on the bottom stratum, and the ratio truth / bound;
  * the composition of the bottom stratum's open pairs: twin primes vs a smooth member.
"""

import json
import os
import sys
from math import isqrt, prod

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (admissible_range, longest_false_run, primes_in, primes_upto,  # noqa: E402
                    range_open, smooth_numbers)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def machine(q, N):
    cut1 = prod(primes_in(1, q))
    gears = primes_in(cut1, isqrt(N))
    return cut1, gears, gears[-1]


def record_sweep(q, N):
    cut1, gears, Q = machine(q, N)
    op = range_open(gears, N)
    op[0] = False
    F, at = longest_false_run(op[:N + 1])
    smset = set(smooth_numbers(cut1, Q + 2))
    S = [n for n in range(1, Q - 1) if n in smset and n + 2 in smset]
    gaps = [S[i + 1] - S[i] - 1 for i in range(len(S) - 1)]
    nxt = S[-1] + 1
    while nxt <= N and not op[nxt]:
        nxt += 1
    Fz = max(max(gaps), nxt - S[-1] - 1)
    Fq, atq = longest_false_run(op[Q:min(N, Q * Q) + 1])
    return {"q": q, "N": N, "cut1": cut1, "Q": Q, "m": len(gears),
            "F_range": F, "F_at": at, "in_smooth_zone": bool(at + F <= Q),
            "F_smooth_zone": Fz, "n_smooth_pairs": len(S), "s_k": S[-1],
            "F_quiet_zone": Fq, "F_quiet_at": atq + Q,
            "F_over_Q": round(F / Q, 5)}


def strata_and_bound(q, N):
    """The quiet zone (Q, Q^2] in dyadic strata, and the L63 prime-gap bound on (Q, 2Q]."""
    cut1, gears, Q = machine(q, N)
    top = min(N, Q * Q)
    op = range_open(gears, N)
    op[0] = False
    adm = admissible_range(gears, N)
    smset = set(smooth_numbers(cut1, top + 2))
    pm = np.zeros(top + 3, dtype=bool)
    pm[primes_upto(top + 2)] = True

    # L59: in (Q, 2Q] admissible <=> prime or q#-smooth
    bad = 0
    for n in range(Q + 1, min(2 * Q, top) + 1):
        pred = bool(pm[n]) or (n in smset)
        if bool(adm[n]) != pred:
            bad += 1
    l59_exc = bad

    # the bottom stratum's open pairs, classified
    twin = smoothy = 0
    for n in range(Q + 1, min(2 * Q, top) - 1):
        if op[n]:
            if pm[n] and pm[n + 2]:
                twin += 1
            else:
                smoothy += 1

    # L63: consecutive primes p < p' in (Q, 2Q] with no q#-smooth number strictly between
    pr = [int(p) for p in primes_upto(min(2 * Q, top)) if p > Q]
    sm_sorted = sorted(x for x in smset if Q < x <= min(2 * Q, top))
    bound = 0
    for i in range(len(pr) - 1):
        p, pp = pr[i], pr[i + 1]
        if not any(p < s < pp for s in sm_sorted):
            bound = max(bound, pp - p - 1)
    # the truth on the bottom stratum
    Fb, atb = longest_false_run(op[Q:min(2 * Q, top) + 1])

    strata = []
    lo = Q
    while lo < top:
        hi = min(2 * lo, top)
        f, a = longest_false_run(op[lo:hi + 1])
        strata.append({"lo": int(lo), "hi": int(hi), "record": int(f), "at": int(a + lo)})
        lo = hi
    return {"q": q, "N": N, "Q": Q, "quiet_top": top,
            "L59_exceptions": l59_exc,
            "bottom_open_twin": twin, "bottom_open_with_smooth_member": smoothy,
            "L63_bound": bound, "bottom_record": int(Fb), "bottom_record_at": int(atb + Q),
            "truth_over_bound": (round(Fb / bound, 3) if bound else None),
            "strata": strata}


def main():
    out = {}
    print("=== the record of the loaded tier 3, sweep in N ===")
    print(f"{'q':>3} {'N':>10} {'Q':>7} {'m':>6} {'F':>6} {'at':>10} {'zone':>7} "
          f"{'F_smooth':>9} {'F_quiet':>8} {'F/Q':>8}")
    rows = []
    for q in [5, 7]:
        for N in [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8]:
            cut1, gears, _ = (prod(primes_in(1, q)), primes_in(prod(primes_in(1, q)),
                                                              isqrt(N)), None)
            if len(gears) < 5:
                print(f"{q:>3} {N:>10}   (only {len(gears)} gears above q# = {cut1}: skipped)")
                continue
            r = record_sweep(q, N)
            rows.append(r)
            print(f"{q:>3} {N:>10} {r['Q']:>7} {r['m']:>6} {r['F_range']:>6} {r['F_at']:>10} "
                  f"{'smooth' if r['in_smooth_zone'] else 'quiet':>7} "
                  f"{r['F_smooth_zone']:>9} {r['F_quiet_zone']:>8} {r['F_over_Q']:>8}")
    out["sweep"] = rows

    print("\n=== the quiet zone: strata, the L59 rule, and the L63 prime-gap bound ===")
    st = []
    for q, N in [(5, 10 ** 6), (5, 10 ** 7), (7, 10 ** 6), (7, 10 ** 7)]:
        r = strata_and_bound(q, N)
        st.append(r)
        print(f"\nq={q} N={N} Q={r['Q']} quiet zone ({r['Q']}, {r['quiet_top']}]")
        print(f"   L59 on the bottom stratum: {r['L59_exceptions']} exceptions; "
              f"open pairs there = {r['bottom_open_twin']} twin primes + "
              f"{r['bottom_open_with_smooth_member']} with a smooth member")
        print(f"   L63 prime-gap bound {r['L63_bound']}, truth {r['bottom_record']} at "
              f"{r['bottom_record_at']}, ratio {r['truth_over_bound']}")
        print("   strata records:", [s["record"] for s in r["strata"]])
    out["strata"] = st

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e4_record.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
