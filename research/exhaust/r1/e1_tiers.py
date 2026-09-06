"""Item 1: THE TIERS AS OBJECTS.

Part A - tier 3 as a WHEEL, exact over a full period, at raised splits q = 5, 7, 11.
        A sub-wheel of tier 3's first gears is a machine with q' = nextprime(q#) and m gears, and
        every wheel law of top_machine_1.md is stated in (q', m) only.  Laws checked: L1/L2 (two
        teeth, arcs g-3 and 1, the shield), L3 (partner law), L4 (no gap 4), L5 (count prod(g-2)),
        L6 (origin clump 2(q'-3)+1, the always-open pair), L7 (mirror), L9 (mirror parity of the
        gap census), L10 (run ceiling q'-3, chain ceiling q'-2, and the two start-counts),
        L15 (dominoes prod(g-4) and prod(g-3)).

Part B - the LOADED EXHAUST ON A RANGE: tier 3's gears that act on [1, N], i.e. the primes in
        (q#, Q] with Q the largest gear used.  Zone law L46 with "smooth" meaning q#-smooth, zone
        rule L57 on [1, Q^2], edges L58, the range record against the smooth-zone record L47, and
        the ceilings of L4 and L10 on the range.

Exact everywhere; no sampling.
"""

import json
import os
import sys
from math import prod

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (admissible_range, gap_census, longest_false_run, primes_in,  # noqa: E402
                    primes_upto, range_open, runs_of_true, smooth_numbers, wheel_open)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def nextprimes(x, k):
    got, p = [], x
    while len(got) < k:
        p += 1
        if p > 2 and all(p % r for r in range(2, int(p ** 0.5) + 1)):
            got.append(p)
    return got


def run_start_counts(a, step, cyclic=False):
    """hist[L] = #{x : x, x+step, ..., x+step(L-1) all open}.

    On a full period W (odd) the step-2 orbit is a single cycle 0, 2, ..., W-1, 1, 3, ..., W-2,
    so the two parity classes must be CONCATENATED, not treated separately; on a range they are
    two independent sequences.  a[0] is always False on a period, so no wrap correction is needed.
    """
    if step == 1:
        seqs = [a]
    elif cyclic:
        seqs = [np.concatenate([a[0::2], a[1::2]])]
    else:
        seqs = [a[0::2], a[1::2]]
    mx = 0
    cs = []
    for s in seqs:
        _, lens = runs_of_true(s)
        c = np.bincount(lens) if lens.size else np.zeros(1, dtype=np.int64)
        cs.append(c)
        mx = max(mx, c.size)
    c = np.zeros(mx, dtype=np.int64)
    for x in cs:
        c[:x.size] += x
    ell = np.arange(mx, dtype=np.int64)
    # suffix sums: C[L] = sum_{l>=L} c[l];  S[L] = sum_{l>=L} c[l]*l
    C = np.cumsum(c[::-1])[::-1]
    S = np.cumsum((c * ell)[::-1])[::-1]
    hist = {}
    for L in range(1, mx):
        v = int(S[L] - (L - 1) * C[L])
        if v:
            hist[L] = v
    return hist


# ------------------------------------------------------------------ Part A: wheels

def wheel_laws(gears, label):
    g0, m = gears[0], len(gears)
    W = prod(gears)
    a = wheel_open(gears)
    res = {"label": label, "gears": gears, "q_prime": g0, "m": m, "W": W}
    exc = {}

    # L5 the wheel count
    res["open_count"] = int(a.sum())
    res["open_count_pred"] = prod(g - 2 for g in gears)
    exc["L5_count"] = 0 if res["open_count"] == res["open_count_pred"] else 1

    # L1/L2 two teeth, g-2 slots, arcs of lengths g-3 and 1, singleton arc at -1 (the shield)
    bad = 0
    for g in gears:
        struck = {n for n in range(g) if n % g == 0 or (n + 2) % g == 0}
        if struck != {0, g - 2}:
            bad += 1
            continue
        if g - len(struck) != g - 2:
            bad += 1
        arcs, cur = [], []
        for n in [(i) % g for i in range(g)]:
            if n in struck:
                if cur:
                    arcs.append(cur)
                cur = []
            else:
                cur.append(n)
        if cur:
            arcs.append(cur)
        if sorted(len(x) for x in arcs) != sorted([g - 3, 1]):
            bad += 1
        if [x[0] for x in arcs if len(x) == 1] != [(g - 1) % g]:
            bad += 1
    exc["L1_L2_arcs_shield"] = bad

    # L3 partner law: the two teeth of one gear are at cyclic distance exactly 2
    exc["L3_partner"] = sum(1 for g in gears if ((g - 2) + 2) % g != 0)

    # L4 / L9: the cyclic gap census
    idx = np.flatnonzero(a)
    dists = np.concatenate([np.diff(idx), [W - int(idx[-1]) + int(idx[0])]])
    vals, cnt = np.unique(dists, return_counts=True)
    census = {int(v): int(c) for v, c in zip(vals, cnt)}
    res["gap_census_head"] = {k: census[k] for k in sorted(census)[:8]}
    exc["L4_gap4"] = census.get(4, 0)
    exc["L9_mirror_parity"] = sum(1 for k, v in census.items() if k != 1 and v % 2)

    # L6 the origin clump and the always-open pair
    exc["L6_shield"] = 0 if a[(-1) % W] else 1
    clump_bad = sum(1 for n in range(-(g0 - 1), g0 - 2)
                    if n not in (0, -2) and not a[n % W])
    exc["L6_clump_open"] = clump_bad
    res["clump_pred"] = 2 * (g0 - 3) + 1
    res["clump_obs"] = sum(1 for n in range(-(g0 - 1), g0 - 2) if a[n % W])
    exc["L6_clump_size"] = 0 if res["clump_obs"] == res["clump_pred"] else 1
    exc["L6_antipode"] = (0 if a[2 % W] else 1) + (0 if a[(-4) % W] else 1)

    # L7 the mirror n -> -n-2  (b[n] = a[(W-2-n) mod W])
    b = np.concatenate([a[W - 2::-1], a[W - 1:]])
    exc["L7_mirror"] = int((b != a).sum())

    # L10 run and chain ceilings, with the start-counts
    _, lens = runs_of_true(a)
    res["max_run"] = int(lens.max())
    res["max_run_pred"] = g0 - 3
    exc["L10_run_ceiling"] = 0 if res["max_run"] == g0 - 3 else 1
    hr = run_start_counts(a, 1)
    hc = run_start_counts(a, 2, cyclic=True)
    res["max_chain"] = max(hc) if hc else 0
    res["max_chain_pred"] = g0 - 2
    exc["L10_chain_ceiling"] = 0 if res["max_chain"] == g0 - 2 else 1
    runbad = sum(1 for L in range(2, g0 - 2)
                 if hr.get(L, 0) != prod(g - 2 - L for g in gears))
    chainbad = sum(1 for L in range(1, g0 - 1)
                   if hc.get(L, 0) != prod(g - 1 - L for g in gears))
    exc["L10_run_counts_L>=2"] = runbad
    exc["L10_chain_counts_L>=1"] = chainbad
    res["L10_run_count_L1_obs"] = hr.get(1, 0)
    res["L10_run_count_L1_formula"] = prod(g - 3 for g in gears)

    # L15 dominoes
    res["adjacent"] = int((a & np.roll(a, -1)).sum())
    res["adjacent_pred"] = prod(g - 4 for g in gears)
    res["shared"] = int((a & np.roll(a, -2)).sum())
    res["shared_pred"] = prod(g - 3 for g in gears)
    exc["L15_adjacent"] = 0 if res["adjacent"] == res["adjacent_pred"] else 1
    exc["L15_shared"] = 0 if res["shared"] == res["shared_pred"] else 1

    res["exceptions"] = exc
    res["total_exceptions"] = sum(exc.values())
    return res


# ------------------------------------------------------------------ Part B: the range

def loaded_exhaust(q, N):
    cut1 = prod(primes_in(1, q))
    Qn = int(np.sqrt(N))
    gears = primes_in(cut1, Qn)
    Qtop = gears[-1]
    a = admissible_range(gears, N)
    op = range_open(gears, N)
    op[0] = False
    res = {"q": q, "cut1": cut1, "N": N, "Q_top_gear": Qtop, "n_gears": len(gears),
           "q_prime": gears[0]}

    smlist = smooth_numbers(cut1, N + 2)
    res["n_smooth_upto_N"] = len(smlist)
    sm = set(smlist)

    # L46: for n <= Qtop - 2 the pair n is open iff n and n+2 are both cut1-smooth
    bad = 0
    S = []
    for n in range(1, Qtop - 1):
        pred = (n in sm) and (n + 2 in sm)
        if pred:
            S.append(n)
        if bool(op[n]) != pred:
            bad += 1
    res["L46_range"] = [1, Qtop - 2]
    res["L46_exceptions"] = bad
    res["n_smooth_pairs"] = len(S)

    # L57 on [1, min(N, Qtop^2)]; L58 edges
    lim = min(N, Qtop * Qtop)
    pr = primes_upto(lim)
    smarr = np.zeros(lim + 1, dtype=bool)
    sml = [s for s in smlist if s <= lim]
    smarr[np.array(sml, dtype=np.int64)] = True
    rule = smarr.copy()
    lo = int(np.searchsorted(pr, Qtop, side="right"))
    for s in sml:
        hi = lim // s
        if hi <= Qtop:
            continue
        j = int(np.searchsorted(pr, hi, side="right"))
        if j > lo:
            rule[s * pr[lo:j]] = True
    rule[0] = False
    diff = np.flatnonzero(rule != a[:lim + 1])
    res["L57_range"] = [1, lim]
    res["L57_exceptions"] = int(diff.size)
    res["p1"] = int(pr[lo])
    res["p1_sq"] = res["p1"] ** 2
    ns = np.flatnonzero(a[:lim + 1] & ~smarr)
    res["L58_first_admissible_nonsmooth"] = int(ns[0]) if ns.size else None
    res["L58_edge1_ok"] = (res["L58_first_admissible_nonsmooth"] == res["p1"])

    # the range record against the smooth-zone record
    F, at = longest_false_run(op[:N + 1])
    res["F_range"], res["F_at"] = F, at
    res["F_in_smooth_zone"] = bool(at + F <= Qtop)
    gaps = [S[i + 1] - S[i] - 1 for i in range(len(S) - 1)]
    sk = S[-1]
    nxt = sk + 1
    while nxt <= N and not op[nxt]:
        nxt += 1
    res["s_k"] = sk
    res["next_open_after_s_k"] = int(nxt)
    res["F_zone_maxgap"] = max(gaps)
    res["F_zone_tail"] = int(nxt - sk - 1)
    res["F_zone"] = max(res["F_zone_maxgap"], res["F_zone_tail"])
    res["L47_lower_bound_ok"] = bool(F >= res["F_zone"])
    res["F_equals_zone_record"] = bool(F == res["F_zone"])
    res["F_over_Qtop"] = round(F / Qtop, 4)

    # L4 and L10 on the range
    cens = gap_census(op[:N + 1])
    res["gap4_count"] = cens.get(4, 0)
    res["gap_census_head"] = {int(k): int(v) for k, v in sorted(cens.items())[:8]}
    _, lens = runs_of_true(op[:N + 1])
    res["max_run_on_range"] = int(lens.max()) if lens.size else 0
    res["max_run_ceiling"] = gears[0] - 3
    hc = run_start_counts(op[:N + 1], 2)
    res["max_chain_on_range"] = max(hc) if hc else 0
    res["max_chain_ceiling"] = gears[0] - 2
    return res


def main():
    out = {}
    for q, k in [(5, 4), (7, 3), (11, 2)]:
        cut1 = prod(primes_in(1, q))
        g = nextprimes(cut1, k)
        r = wheel_laws(g, f"tier3(q={q})")
        r["q_base"] = q
        out.setdefault("wheels", []).append(r)
        print(f"q={q} tier-3 wheel {g} W={r['W']} open={r['open_count']}"
              f" (pred {r['open_count_pred']}) run={r['max_run']}/{r['max_run_pred']}"
              f" chain={r['max_chain']}/{r['max_chain_pred']}"
              f" adj={r['adjacent']}/{r['adjacent_pred']}"
              f" shared={r['shared']}/{r['shared_pred']}"
              f" clump={r['clump_obs']}/{r['clump_pred']}"
              f" EXC={r['total_exceptions']}")
        print("   ", {k2: v for k2, v in r["exceptions"].items() if v})
        print("    L=1 run starts:", r["L10_run_count_L1_obs"],
              "formula prod(g-3):", r["L10_run_count_L1_formula"])

    r = wheel_laws([7, 11, 13], "tier2(q=5)")
    r["q_base"] = 5
    out.setdefault("wheels", []).append(r)
    print(f"q=5 tier-2 wheel [7,11,13] W={r['W']} EXC={r['total_exceptions']}",
          {k2: v for k2, v in r["exceptions"].items() if v})

    for q, N in [(5, 10 ** 6), (5, 10 ** 7), (7, 10 ** 6), (7, 10 ** 7)]:
        r = loaded_exhaust(q, N)
        out.setdefault("range", []).append(r)
        print(f"\nq={q} N={N}: gears ({r['cut1']}, {r['Q_top_gear']}], m={r['n_gears']},"
              f" q'={r['q_prime']}, smooth numbers <= N: {r['n_smooth_upto_N']}")
        print(f"   L46 on [1,{r['L46_range'][1]}]: {r['L46_exceptions']} exceptions;"
              f" {r['n_smooth_pairs']} smooth pairs, s_k={r['s_k']}")
        print(f"   L57 on [1,{r['L57_range'][1]}]: {r['L57_exceptions']} exceptions;"
              f" p1={r['p1']} p1^2={r['p1_sq']};"
              f" first admissible non-smooth={r['L58_first_admissible_nonsmooth']}"
              f" (edge ok {r['L58_edge1_ok']})")
        print(f"   F_range={r['F_range']} at {r['F_at']} (smooth zone: {r['F_in_smooth_zone']},"
              f" F/Q={r['F_over_Qtop']}); F_zone={r['F_zone']}"
              f" (maxgap {r['F_zone_maxgap']}, tail {r['F_zone_tail']});"
              f" equal: {r['F_equals_zone_record']}")
        print(f"   gap-4 count {r['gap4_count']}; max run {r['max_run_on_range']}"
              f" <= {r['max_run_ceiling']}; max chain {r['max_chain_on_range']}"
              f" <= {r['max_chain_ceiling']}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e1_tiers.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
