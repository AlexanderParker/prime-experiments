"""og_census.py -- the strike census of the real sections in the dilation form (P2, P3, P4, P6).

For every column j of a section (members 6j - 1, 6j + 1) under the engine {5..q}:
  the strikers (gears dividing a member); the least striker g0 (= lpf of the struck member); the
  struck member n; the quotient m = n / g0; whether m is prime; the depth Omega(n); whether n < g0^3.
Aggregates per section: columns, open (twin) columns, both-struck columns, least-striker histogram,
prime-quotient shares (all; n < g0^3, which must be 1.0 by the two-prime lemma; n >= g0^3), depth
histogram and maximum, and the nesting identity P6 at member level:
  {(column, side) of members with lpf = g} == {(g i + s k_g, s eps_g) : m = 6i + s in R_g, g m in section}.
Sections: the finer sections at q = 11..53 and the construction's sections (bases 3, 5, 7, 11, 13 link 1;
bases 3, 5, 7 link 2; base 11 link 2 with --big).
Output: results/census.json, results/census.log
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from og_common import (construction_sections, finer_section, gears_of, k_of, nextprime, spf_table)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def omega_vec(n, spf):
    """Omega for an int64 array n (values within the spf table)"""
    n = n.copy()
    om = np.zeros(n.shape, dtype=np.int32)
    while True:
        act = n > 1
        if not act.any():
            break
        f = spf[n[act]]
        n[act] //= f
        om[act] += 1
    return om


def census(lo, hi, q, spf, label, do_p6=True):
    """columns whose members lie strictly inside (lo, hi): members n with lo < n < hi, n = +-1 mod 6.
    lo and hi are prime squares (lo = p^2 struck by p; hi = p'^2 not struck), so the section's columns are
    a+1 .. b-1 with 6a + 1 = lo, 6b + 1 = hi."""
    a = (lo - 1) // 6
    b = (hi - 1) // 6
    cols = np.arange(a + 1, b, dtype=np.int64)
    left = 6 * cols - 1
    right = 6 * cols + 1
    sl = spf[left].astype(np.int64)
    sr = spf[right].astype(np.int64)
    struck_l = sl <= q
    struck_r = sr <= q
    open_col = ~struck_l & ~struck_r
    both = struck_l & struck_r
    # least striker per column: min of the struck members' lpf
    INF = np.int64(10 ** 12)
    gl = np.where(struck_l, sl, INF)
    gr = np.where(struck_r, sr, INF)
    use_left = gl <= gr  # ties impossible (g | 2)
    g0 = np.where(use_left, gl, gr)
    n = np.where(use_left, left, right)
    struck = ~open_col
    g0s = g0[struck]
    ns = n[struck]
    ms = ns // g0s
    m_prime = spf[ms] == ms  # m = 1 counts as not prime (spf[1] = 1 == 1 -> True); handle
    m_one = ms == 1
    m_prime = m_prime & ~m_one
    below_cube = ns < g0s ** 3
    depth = omega_vec(ns, spf)
    gears = gears_of(q)
    hist = {int(g): int((g0s == g).sum()) for g in gears}
    hist = {g: c for g, c in hist.items() if c}
    out = dict(
        label=label, lo=int(lo), hi=int(hi), q=int(q), a=int(a), b=int(b), columns=int(cols.size),
        open_cols=int(open_col.sum()), open_list=[int(x) for x in cols[open_col][:40]],
        both_struck=int(both.sum()), struck=int(struck.sum()),
        least_striker_hist=hist,
        prime_quotient_all=int(m_prime.sum()), quotient_one=int(m_one.sum()),
        below_cube=int(below_cube.sum()), prime_quotient_below_cube=int((m_prime & below_cube).sum()),
        above_cube=int((~below_cube).sum()), prime_quotient_above_cube=int((m_prime & ~below_cube).sum()),
        depth_hist={int(d): int((depth == d).sum()) for d in np.unique(depth)},
        max_depth=int(depth.max()) if depth.size else 0,
        max_depth_members=[int(x) for x in ns[depth == depth.max()][:5]] if depth.size else [],
        # tail gears (above the section's length) and their strikes
        section_length=int(b - a - 1),
    )
    # the least striker restricted to the tail gears g > section length: how many columns, and their quotients
    L = b - a - 1
    tail = g0s > L
    out["tail_least_struck_cols"] = int(tail.sum())
    out["tail_least_prime_quotient"] = int((m_prime & tail).sum())
    # P6 nesting identity at member level
    if do_p6:
        mism = 0
        checked = 0
        for g in gears:
            eps = 1 if g % 6 == 1 else -1
            kg = k_of(g)
            # A_g: members with lpf = g
            A = set()
            for side, arr, sp in ((-1, left, sl), (1, right, sr)):
                sel = sp == g
                for j in cols[sel]:
                    A.add((int(j), side))
            # B_g: images of m in R_g with g m in (lo, hi)
            B = set()
            m_lo = lo // g + 1
            m_hi = (hi - 1) // g
            for m in range(m_lo, m_hi + 1):
                r = m % 6
                if r == 5:
                    s = -1
                    i = (m + 1) // 6
                elif r == 1:
                    s = 1
                    i = (m - 1) // 6
                else:
                    continue
                if m != 1 and spf[m] < g:
                    continue
                j = g * i + s * kg
                if j < a + 1 or j > b - 1:
                    # the member p'^2 - 2 sits in column b, outside the section's columns a+1..b-1
                    continue
                B.add((int(j), s * eps))
            checked += len(A)
            if A != B:
                mism += len(A ^ B)
        out["p6_members_checked"] = checked
        out["p6_mismatches"] = mism
    return out


def main():
    big = "--big" in sys.argv
    t0 = time.time()
    results = []
    log = []

    def emit(r):
        results.append(r)
        pq = r["prime_quotient_all"] / max(r["struck"], 1)
        pqa = r["prime_quotient_above_cube"] / max(r["above_cube"], 1)
        pqb = r["prime_quotient_below_cube"] / max(r["below_cube"], 1)
        line = (f"{r['label']:<28} q={r['q']:<5} cols={r['columns']:<9} open={r['open_cols']:<6} "
                f"both={r['both_struck']:<7} pq_all={pq:.3f} pq_below_cube={pqb:.3f} ({r['below_cube']}) "
                f"pq_above_cube={pqa:.3f} ({r['above_cube']}) maxdepth={r['max_depth']} "
                f"depth={r['depth_hist']} tail_cols={r['tail_least_struck_cols']} "
                f"p6={r.get('p6_mismatches', 'na')}/{r.get('p6_members_checked', 'na')}")
        print(line, flush=True)
        log.append(line)

    # finer sections
    spf = spf_table(3600)
    for q in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        a, b, cols, gears = finer_section(q)
        qn = nextprime(q)
        emit(census(q * q, qn * qn, q, spf, f"finer p={q}"))
    # construction sections
    plan = [(3, 2), (5, 2), (7, 2), (11, 2 if big else 1), (13, 1)]
    top = 0
    for base, links in plan:
        for sec in construction_sections(base, links):
            top = max(top, sec["hi"])
    print(f"spf table to {top} ...", flush=True)
    spf = spf_table(top)
    print(f"  built in {time.time() - t0:.1f}s", flush=True)
    for base, links in plan:
        for sec in construction_sections(base, links):
            do_p6 = sec["b"] - sec["a"] < 200000
            emit(census(sec["lo"], sec["hi"], sec["q"], spf, f"base {base} link {sec['k']}", do_p6))
    with open(os.path.join(RES, "census.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(RES, "census.log"), "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
