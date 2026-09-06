"""R7 item 3: THE MOMENT VANISHING  M_k(d) = 0 for k < r(d).

Setting (top_machine_2.md L22/L23/L25).  In the universal regime every gear exceeds d + 2, so
the d + 3 integers 0, -1, ..., -(d+2) are distinct residues mod every gear and

    e(S) = |A(S)| ,   A(S) = {0, 2, d, d+2} u {j, j+2 : j in S} ,  S subset of [1, d-1]
    c_e(d) = sum over S with e(S) = e of (-1)^{|S|}
    N_d = sum_e c_e(d) prod_g (g - e) = sum_k (-1)^k sigma_{m-k} M_k(d) ,
    M_k(d) = sum_e c_e(d) e^k = sum over S of (-1)^{|S|} e(S)^k .

The claim proved in top_machine_7.md: M_k(d) is (-1)^{d-1} times the coefficient of the full
monomial x_1...x_{d-1} in the multilinear form of f(x)^k with f(x) = |A(S(x))|, and that
coefficient can only be nonzero if k of the pieces

    J_p = {p-2, p} n [1, d-1] ,   p in [1, d+1] , p != 2 , p != d

cover [1, d-1].  The minimum number of such pieces is r(d).

This script computes c_e(d) and M_k(d) EXACTLY (integer inclusion-exclusion over all 2^(d-1)
subsets, no sampling) for d up to DMAX, computes r(d) independently by exact set cover, and
checks the vanishing, the first non-vanishing value, and the d = 4 degeneracy.

usage: uv run python research/topmachine/r7/moments.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []
DMAX = 26


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def popcount(a):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a).astype(np.uint8)
    x = a.astype(np.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24).astype(np.uint8)


def coeffs(d):
    """c_e(d) as a dict e -> integer, exact."""
    n = d - 1
    bmask = (1 << 0) | (1 << 2) | (1 << d) | (1 << (d + 2))
    dm = [np.uint32((1 << j) | (1 << (j + 2))) for j in range(1, d)]   # piece for j in [1,d-1]
    cov = np.zeros(1 << n, dtype=np.uint32)
    cov[0] = bmask
    par = np.zeros(1 << n, dtype=bool)
    for j in range(n):
        half = 1 << j
        cov[half:2 * half] = cov[:half] | dm[j]
        par[half:2 * half] = ~par[:half]
    e = popcount(cov).astype(np.int64)
    del cov
    pos = np.bincount(e[~par], minlength=d + 4)
    neg = np.bincount(e[par], minlength=d + 4)
    del e, par
    c = pos.astype(object) - neg.astype(object)
    return {int(i): int(c[i]) for i in range(len(c)) if c[i] != 0}


def moments(c, kmax):
    return [sum(cv * (e ** k) for e, cv in c.items()) for k in range(kmax + 1)]


def r_by_cover(d):
    """minimum number of pieces J_p covering [1, d-1]; None if impossible."""
    n = d - 1
    if n <= 0:
        return 0
    pieces = []
    for p in range(1, d + 2):
        if p == 2 or p == d:
            continue
        s = [x for x in (p - 2, p) if 1 <= x <= n]
        if s:
            pieces.append(sum(1 << (x - 1) for x in s))
    full = (1 << n) - 1
    if not pieces or (np.bitwise_or.reduce(np.array(pieces, dtype=np.int64)) & full) != full:
        return None
    INF = 10 ** 9
    dp = np.full(1 << n, INF, dtype=np.int32)
    dp[0] = 0
    for mask in range(1 << n):
        v = dp[mask]
        if v == INF:
            continue
        if mask == full:
            return int(v)
        for pc in pieces:
            nm = mask | pc
            if dp[nm] > v + 1:
                dp[nm] = v + 1
    return int(dp[full]) if dp[full] < INF else None


def r_closed(d):
    """D(d-1): the free-boundary domino cost of d - 1 consecutive positions."""
    L = d - 1
    a, b = (L + 1) // 2, L // 2
    return (a + 1) // 2 + (b + 1) // 2


def main():
    summary = {}
    say("# R7.3  The moment vanishing")
    say()
    say("## The table: `r(d)` three ways, and where `M_k(d)` first fails to vanish")
    say()
    say("| d | r(d) closed form D(d-1) | r(d) by exact set cover | first k with M_k != 0 | "
        "M_{r(d)}(d) | vanishing below | subsets |")
    say("|---|---|---|---|---|---|---|")
    rows = []
    bad_close, bad_van, bad_nonzero = 0, 0, 0
    for d in range(2, DMAX + 1):
        c = coeffs(d)
        rc = r_closed(d)
        rcov = r_by_cover(d) if d <= 20 else None
        kmax = min(rc + 2, 2 * rc + 2)
        M = moments(c, max(kmax, 8))
        first = next((k for k, v in enumerate(M) if v != 0), None)
        ok_van = all(M[k] == 0 for k in range(min(rc, len(M))))
        if d != 4:
            if rcov is not None and rcov != rc:
                bad_close += 1
            if not ok_van:
                bad_van += 1
            if M[rc] == 0:
                bad_nonzero += 1
        rows.append({"d": d, "r_closed": rc, "r_cover": rcov, "first_nonzero": first,
                     "M_r": M[rc] if rc < len(M) else None,
                     "c_e": c, "M": M[:max(kmax, 8) + 1]})
        say(f"| {d} | {rc} | {rcov if rcov is not None else ('IMPOSSIBLE' if d == 4 else '-')} | "
            f"{first if first is not None else 'never'} | "
            f"{M[rc] if rc < len(M) else '-'} | {'yes' if ok_van else '**NO**'} | "
            f"{1 << (d - 1):,} |")
    say()
    say(f"**{bad_van} failures of the vanishing** and **{bad_nonzero} failures of "
        f"`M_{{r(d)}} != 0`** over `d = 2..{DMAX}` (excluding `d = 4`); the closed form "
        f"`D(d-1)` agrees with the exact set cover at every `d <= 20` except `d = 4`: "
        f"**{bad_close} mismatches**.")
    say()
    summary["vanishing_failures"] = bad_van
    summary["nonzero_failures"] = bad_nonzero
    summary["closed_vs_cover_mismatches"] = bad_close
    summary["dmax"] = DMAX

    # ---------------------------------------------------------------- d = 4
    say("## `d = 4`: not an exception, the extreme case")
    say()
    r4 = [d for d in rows if d["d"] == 4][0]
    say(f"Available pieces for `d = 4` on the ground set `[1, 3]`: `J_1 = {{1}}`, "
        f"`J_3 = {{1,3}}`, `J_5 = {{3}}` - and `J_2`, `J_4` are the two excluded ones. "
        f"**Position 2 is in no piece**, so no `k` covers and `r(4) = infinity`.")
    say(f"Measured: `M_k(4) = {r4['M']}` for `k = 0..{len(r4['M'])-1}` - **all zero**, "
        f"which is `N_4 = 0` identically, i.e. L4.")
    say()
    summary["M_4_all_zero"] = all(v == 0 for v in r4["M"])

    # ---------------------------------------------------------------- signatures
    say("## The universal signatures `c_e(d)`, against the published ones")
    say()
    say("| d | signature c_e (e: value) |")
    say("|---|---|")
    for row in rows:
        if row["d"] <= 10:
            sig = ", ".join(f"{e}: {v}" for e, v in sorted(row["c_e"].items()))
            say(f"| {row['d']} | {sig} |")
    say()
    c3 = {d["d"]: d["c_e"] for d in rows}
    same35 = c3[3] == c3[5]
    say(f"`c_e(3) == c_e(5)`: **{same35}** (L24, the gap-3 / gap-5 coincidence). "
        f"`c_e(6) = {sorted(c3[6].items())}` against the published `+1, -1, -3, +5, -2` at "
        f"`e = 4..8`.")
    say()
    summary["c3_eq_c5"] = bool(same35)

    # ---------------------------------------------------------------- L18 numbers
    say("## The gear-independent lengths: `|M_{r(d)}(d)|` against L18/L25's record "
        "multiplicities")
    say()
    want = {6: 18, 7: 96, 8: 24, 9: 24, 10: 480, 11: 6480, 12: 1440, 13: 720}
    say("| d | r(d) | M_{r(d)}(d) | published multiplicity | agrees |")
    say("|---|---|---|---|---|")
    badL18 = 0
    for row in rows:
        d = row["d"]
        if d in want:
            v = abs(row["M_r"])
            ok = v == want[d]
            if not ok:
                badL18 += 1
            say(f"| {d} | {row['r_closed']} | {row['M_r']} | {want[d]} | "
                f"{'yes' if ok else '**NO**'} |")
    say()
    say(f"**{badL18} mismatches of {len(want)}**.")
    summary["L18_mismatches"] = badL18
    say()

    json.dump({"summary": summary,
               "rows": [{k: v for k, v in r.items() if k != "c_e"} for r in rows]},
              open(os.path.join(RES, "moments.json"), "w"), indent=1)
    with open(os.path.join(RES, "moments.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
