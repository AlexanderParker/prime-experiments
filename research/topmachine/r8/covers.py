"""R8 section 1: THE NON-CANCELLATION OF THE TOP MOMENT, and the signature as a cover polynomial.

Setting (top_machine_7.md L73): in the universal regime (every gear > d + 2)

    e(S) = |A(S)|, A(S) = {0, 2, d, d+2} u {j, j+2 : j in S},  S subset of [1, d-1]
    c_e(d) = sum over S with e(S) = e of (-1)^{|S|},   M_k(d) = sum_e c_e(d) e^k .

Pieces: J_p = {p-2, p} n [1, d-1] for p in [1, d+1], p != 2, p != d.  A cover is a subfamily
whose union is [1, d-1]; C_t(d) = number of covers with exactly t pieces; r(d) = min t.

Claims tested (pre-registered in top_machine_8.md):
  P1  M_{r(d)}(d) = (-1)^r r! C_r(d)                         d = 2..26 (incl. d = 4)
  P2  C_r(d) closed form by d mod 4; the eight L18 multiplicities
  P3  every minimum cover P has mu(P) = sum_x (-1)^{|x|} prod_{p in P} u_p(x) = (-1)^r,  d <= 14
  P4  sum_e c_e(d) z^e = sum_t C_t(d) (1-z)^t z^{d+3-t}      d = 2..26; transfer matrix = brute
  P5  N_d(G) = sum_t C_t(d) (-1)^t Delta^t P_G(d+3-t)         against L22 and scans
  P6  N_d = r! C_r in a wheel of r(d) gears all > d+2, by scan
  P7  M_{r+1}(d) = (-1)^r (r+1)!/2 [C_r (2d+6-r) - 2 C_{r+1}]  d = 3..26
  P8  the signature by transfer matrix, tables to d = 60

usage: uv run python research/topmachine/r8/covers.py
"""

import itertools
import json
import os
import sys
from math import factorial, prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r7"))
from moments import coeffs as exact_signature  # noqa: E402  (R7's exact 2^(d-1) enumeration)

OUT = []
DMAX = 26


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


# ---------------------------------------------------------------- pieces and covers

def pieces(d):
    """[(p, mask)] with bit (x-1) for position x in [1, d-1]."""
    n = d - 1
    out = []
    for p in range(1, d + 2):
        if p == 2 or p == d:
            continue
        m = 0
        for x in (p - 2, p):
            if 1 <= x <= n:
                m |= 1 << (x - 1)
        if m:
            out.append((p, m))
    return out


def r_closed(d):
    L = d - 1
    a, b = (L + 1) // 2, L // 2
    return (a + 1) // 2 + (b + 1) // 2


def C_closed(d):
    m = d % 4
    if m == 0:
        return d // 4 - 1
    if m == 1:
        return 1
    if m == 2:
        return (d + 6) // 4
    return ((d + 1) // 4) ** 2


def covers_bruteforce(d):
    """C_t(d) for all t by enumerating every subfamily of the pieces (2^#pieces)."""
    pc = pieces(d)
    n = d - 1
    full = (1 << n) - 1
    k = len(pc)
    cov = np.zeros(1 << k, dtype=np.int64)
    size = np.zeros(1 << k, dtype=np.int8)
    for j, (_, m) in enumerate(pc):
        half = 1 << j
        cov[half:2 * half] = cov[:half] | m
        size[half:2 * half] = size[:half] + 1
    ok = cov == full
    return {int(t): int(c) for t, c in enumerate(np.bincount(size[ok], minlength=k + 1)) if c}


def class_cover_poly(ell, sing_left, sing_right):
    """Cover polynomial (list of int coefficients in w) of a step-2 run of ell cells by its
    ell-1 adjacent pairs and the available end singletons (multiplicities sing_left at cell 1,
    sing_right at cell ell; if ell == 1 both act on the same cell).  Two-state DP."""
    if ell == 0:
        return [1]
    # state: polynomial for "cell i is already covered by the pair (i-1, i)" (True/False)
    def padd(a, b):
        n = max(len(a), len(b))
        return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]

    def pshift(a, k):
        return [0] * k + a

    state = {False: [1], True: []}
    for i in range(1, ell + 1):
        sing = 0
        if i == 1:
            sing += sing_left
        if i == ell:
            sing += sing_right
        new = {False: [], True: []}
        for covered, poly in state.items():
            if not poly:
                continue
            for e_i in ((0, 1) if i < ell else (0,)):
                for s in range(sing + 1):          # s of the sing singletons chosen
                    ways = 1
                    # binomial(sing, s)
                    from math import comb
                    ways = comb(sing, s)
                    if not (covered or e_i or s > 0):
                        continue
                    term = pshift([ways * c for c in poly], e_i + s)
                    new[bool(e_i)] = padd(new[bool(e_i)], term)
        state = new
    return padd(state[False], state[True])


def covers_transfer(d):
    """C_t(d) for all t by the product of the two classes' cover polynomials."""
    n = d - 1
    if n == 0:
        return {0: 1}
    if n == 1:
        return {int(t): c for t, c in enumerate(class_cover_poly(1, 1, 1)) if c}
    l_odd, l_even = (n + 1) // 2, n // 2
    odd = class_cover_poly(l_odd, 1, 1 if n % 2 == 1 else 0)
    even = class_cover_poly(l_even, 0, 1 if n % 2 == 0 else 0)
    outp = [0] * (len(odd) + len(even) - 1)
    for i, a in enumerate(odd):
        for j, b in enumerate(even):
            outp[i + j] += a * b
    return {int(t): c for t, c in enumerate(outp) if c}


def signature_from_covers(d, C):
    """c_e(d) from sum_t C_t (1-z)^t z^{d+3-t}: dict e -> coefficient."""
    from math import comb
    c = {}
    for t, ct in C.items():
        for j in range(t + 1):
            e = d + 3 - t + j
            c[e] = c.get(e, 0) + ct * comb(t, j) * (-1) ** j
    return {e: v for e, v in c.items() if v}


def moments_from_signature(c, kmax):
    return [sum(v * e ** k for e, v in c.items()) for k in range(kmax + 1)]


# ---------------------------------------------------------------- the census on a wheel

def census_scan(gears, dmax):
    W = prod(gears)
    a = np.ones(W, dtype=bool)
    for g in gears:
        a[0::g] = False
        a[(g - 2) % g::g] = False
    pos = np.flatnonzero(a)
    dd = np.diff(pos)
    wrap = pos[0] + W - pos[-1]
    dd = np.append(dd, wrap)
    h = np.bincount(dd, minlength=dmax + 1)
    return [int(h[d]) for d in range(dmax + 1)]


def census_L22(gears, d):
    """N_d by the gap census law (any gears)."""
    n = d - 1
    total = 0
    base = [0, 2, d, d + 2]
    for S in range(1 << n):
        A = list(base)
        for j in range(1, d):
            if (S >> (j - 1)) & 1:
                A += [j, j + 2]
        term = 1
        for g in gears:
            term *= g - len({(-x) % g for x in A})
        total += (-1) ** bin(S).count("1") * term
    return total


def census_cover_sum(gears, d, C):
    """N_d = sum_t C_t(d) (-1)^t Delta^t P(d+3-t),  P(e) = prod (g - e)."""
    from math import comb

    def P(e):
        return prod(g - e for g in gears)

    total = 0
    for t, ct in C.items():
        a = d + 3 - t
        delta = sum((-1) ** (t - j) * comb(t, j) * P(a + j) for j in range(t + 1))
        total += ct * (-1) ** t * delta
    return total


def main():
    summary = {}
    say("# R8.1  The non-cancellation, the minimum covers, and the cover polynomial")
    say()

    # ---------------------------------------------------------------- exact signatures (R7)
    sig = {}
    for d in range(2, DMAX + 1):
        sig[d] = exact_signature(d)
    say(f"Exact signatures `c_e(d)` recomputed by R7's `moments.py` for `d = 2..{DMAX}` "
        f"(up to {1 << (DMAX - 1):,} subsets, integer arithmetic).")
    say()

    # ---------------------------------------------------------------- covers: brute vs transfer
    say("## P4 (part): cover counts `C_t(d)` - brute force against the transfer matrix")
    say()
    say("| d | r(d) | C_t(d) (t: count), transfer matrix | brute force agrees |")
    say("|---|---|---|---|")
    Ccov = {}
    bad_tm = 0
    for d in range(1, 61):
        Ccov[d] = covers_transfer(d)
        line = ""
        if d <= 22:
            bf = covers_bruteforce(d)
            ok = bf == Ccov[d]
            if not ok:
                bad_tm += 1
            line = "yes" if ok else "**NO**"
        else:
            line = "-"
        if d <= 26:
            say(f"| {d} | {r_closed(d) if d != 4 else 'inf'} | "
                f"{', '.join(f'{t}: {c}' for t, c in sorted(Ccov[d].items()))} | {line} |")
    say()
    say(f"Transfer matrix against brute force over every subfamily, `d = 2..22`: "
        f"**{bad_tm} mismatches**.")
    say()
    summary["transfer_vs_brute_mismatches"] = bad_tm

    # ---------------------------------------------------------------- P1, P2
    say("## P1, P2: `M_{r(d)}(d) = (-1)^r r! C_r(d)` and the closed form of `C_r(d)`")
    say()
    say("| d | d mod 4 | r(d) | C_r(d) counted | C_r(d) closed form | (-1)^r r! C_r | M_{r(d)}(d) exact | agree |")
    say("|---|---|---|---|---|---|---|---|")
    bad1 = bad2 = 0
    for d in range(2, DMAX + 1):
        r = r_closed(d)
        Cr = Ccov[d].get(r, 0)
        Cc = C_closed(d)
        M = moments_from_signature(sig[d], r + 1)
        pred = (-1) ** r * factorial(r) * Cr
        ok1 = pred == M[r]
        ok2 = Cr == Cc
        bad1 += not ok1
        bad2 += not ok2
        say(f"| {d} | {d % 4} | {r if d != 4 else '(inf) 2'} | {Cr} | {Cc} | {pred} | {M[r]} | "
            f"{'yes' if ok1 and ok2 else '**NO**'} |")
    say()
    say(f"P1: **{bad1} mismatches** of {DMAX - 1} (`d = 4` included: `C = 0`, `M = 0`).  "
        f"P2: **{bad2} mismatches** between the counted `C_r(d)` and the closed form.")
    say()
    summary["P1_mismatches"] = bad1
    summary["P2_mismatches"] = bad2

    want = {6: 18, 7: 96, 8: 24, 9: 24, 10: 480, 11: 6480, 12: 1440, 13: 720}
    say("The eight published multiplicities as `r! C_r`:")
    say()
    say("| d | r | C_r | r! C_r | published |")
    say("|---|---|---|---|---|")
    bad18 = 0
    for d, v in want.items():
        r = r_closed(d)
        val = factorial(r) * C_closed(d)
        bad18 += val != v
        say(f"| {d} | {r} | {C_closed(d)} | {val} | {v} |")
    say()
    say(f"**{bad18} mismatches of 8.**")
    say()
    summary["L18_mismatches"] = bad18

    # ---------------------------------------------------------------- P3: signs cover by cover
    say("## P3: the sign of every minimum cover, by brute force")
    say()
    say("| d | r | minimum covers | mu(P) values seen | all equal (-1)^r |")
    say("|---|---|---|---|---|")
    bad3 = 0
    total_covers = 0
    for d in range(2, 15):
        if d == 4:
            continue
        r = r_closed(d)
        pc = pieces(d)
        n = d - 1
        full = (1 << n) - 1
        mus = set()
        cnt = 0
        for P in itertools.combinations(pc, r):
            if np.bitwise_or.reduce([m for _, m in P]) != full:
                continue
            cnt += 1
            # mu(P) = sum_x (-1)^{|x|} prod_{p in P} [x hits J_p]
            mu = 0
            for x in range(1 << n):
                if all(x & m for _, m in P):
                    mu += (-1) ** bin(x).count("1")
            mus.add(mu)
            if mu != (-1) ** r:
                bad3 += 1
        total_covers += cnt
        say(f"| {d} | {r} | {cnt} | {sorted(mus)} | {'yes' if mus == {(-1) ** r} else '**NO**'} |")
    say()
    say(f"**{bad3} exceptions** over {total_covers} minimum covers, `d = 2..14`.")
    say()
    summary["P3_exceptions"] = bad3
    summary["P3_covers"] = total_covers

    # ---------------------------------------------------------------- P4: the whole signature
    say("## P4: the whole signature is the cover polynomial")
    say()
    say("| d | coefficients c_e(d), exact | from sum_t C_t (1-z)^t z^{d+3-t} | agree |")
    say("|---|---|---|---|")
    bad4 = 0
    for d in range(2, DMAX + 1):
        pred = signature_from_covers(d, Ccov[d])
        ok = pred == sig[d]
        bad4 += not ok
        if d <= 12 or not ok:
            say(f"| {d} | {sorted(sig[d].items())} | {sorted(pred.items())} | "
                f"{'yes' if ok else '**NO**'} |")
    say()
    say(f"**{bad4} mismatches** over `d = 2..{DMAX}`, every coefficient compared.")
    say()
    summary["P4_mismatches"] = bad4

    # the d = 3 / d = 5 coincidence as an identity of cover polynomials
    say(f"`C_t(3) = {Ccov[3]}`, `C_t(5) = {Ccov[5]}`: "
        f"`(1-z)^2 z^6 + 2 (1-z)^3 z^5 + (1-z)^4 z^4 = (1-z)^2 z^4 (z + (1-z))^2 = (1-z)^2 z^4` - "
        f"L24's coincidence is `(z + (1 - z))^2 = 1`.")
    say()

    # ---------------------------------------------------------------- P7: the second moment
    say("## P7: the second moment `M_{r+1}(d)`")
    say()
    say("| d | r | C_r | C_{r+1} | formula | M_{r+1}(d) exact | agree |")
    say("|---|---|---|---|---|---|---|")
    bad7 = 0
    for d in range(3, DMAX + 1):
        if d == 4:
            continue
        r = r_closed(d)
        Cr, Cr1 = Ccov[d].get(r, 0), Ccov[d].get(r + 1, 0)
        M = moments_from_signature(sig[d], r + 1)
        num = factorial(r + 1) * (Cr * (2 * d + 6 - r) - 2 * Cr1)
        assert num % 2 == 0
        pred = (-1) ** r * num // 2
        ok = pred == M[r + 1]
        bad7 += not ok
        say(f"| {d} | {r} | {Cr} | {Cr1} | {pred} | {M[r + 1]} | {'yes' if ok else '**NO**'} |")
    say()
    say(f"**{bad7} mismatches**, `d = 3..{DMAX}`, `d != 4`.")
    say()
    summary["P7_mismatches"] = bad7

    # ---------------------------------------------------------------- P5, P6: the census
    say("## P5, P6: the universal census as a cover sum, and the bijection count")
    say()
    say("| d | gears | all g > d+2 | N_d scan | N_d by L22 | N_d cover sum | r(d) | r! C_r | agree |")
    say("|---|---|---|---|---|---|---|---|---|")
    bad5 = bad6 = 0
    cases = [
        ((11, 13, 17), range(1, 9)),
        ((13, 17, 19, 23), range(1, 11)),
        ((11, 13, 17, 19), range(6, 9)),
        ((13, 17, 19, 23, 29), range(9, 11)),
    ]
    for gears, ds in cases:
        sc = census_scan(list(gears), max(ds) + 1)
        for d in ds:
            univ = all(g > d + 2 for g in gears)
            n22 = census_L22(gears, d)
            ncs = census_cover_sum(gears, d, Ccov[d])
            r = r_closed(d) if d != 4 else None
            bij = factorial(r) * C_closed(d) if (r is not None and r == len(gears)) else None
            ok5 = (ncs == n22 == sc[d]) if univ else True
            bad5 += not ok5
            if bij is not None:
                bad6 += bij != sc[d]
            say(f"| {d} | {','.join(map(str, gears))} | {'yes' if univ else 'no'} | {sc[d]} | "
                f"{n22} | {ncs if univ else '-'} | {r if r else 'inf'} | "
                f"{bij if bij is not None else '-'} | {'yes' if ok5 else '**NO**'} |")
    # the six-gear cases by L22 (period too large to scan)
    G6 = (17, 19, 23, 29, 31, 37)
    for d in (11, 12, 13):
        n22 = census_L22(G6, d)
        ncs = census_cover_sum(G6, d, Ccov[d])
        r = r_closed(d)
        bij = factorial(r) * C_closed(d)
        ok = n22 == ncs == bij
        bad5 += n22 != ncs
        bad6 += bij != n22
        say(f"| {d} | {','.join(map(str, G6))} | yes | (period 2.5e8, not scanned) | {n22} | "
            f"{ncs} | {r} | {bij} | {'yes' if ok else '**NO**'} |")
    say()
    say(f"P5: **{bad5} mismatches** of the cover sum against L22 and the scans in the universal "
        f"regime.  P6: **{bad6} mismatches** of `r! C_r` against the census where `r(d) = m`.")
    say()
    summary["P5_mismatches"] = bad5
    summary["P6_mismatches"] = bad6

    # ---------------------------------------------------------------- P8: tables to 60
    say("## P8: the signature beyond the enumeration - `r(d)`, `C_r(d)`, `M_{r(d)}(d)` to `d = 60`")
    say()
    say("| d | r(d) | C_r(d) | M_{r(d)}(d) | number of nonzero c_e(d) | sum_e |c_e(d)| |")
    say("|---|---|---|---|---|---|")
    rows8 = []
    for d in range(27, 61):
        r = r_closed(d)
        Cr = Ccov[d].get(r, 0)
        c = signature_from_covers(d, Ccov[d])
        M = (-1) ** r * factorial(r) * Cr
        rows8.append({"d": d, "r": r, "C_r": Cr, "M_r": M})
        if d <= 34 or d % 10 == 0:
            say(f"| {d} | {r} | {Cr} | {M} | {len(c)} | {sum(abs(v) for v in c.values())} |")
    say()
    c27 = signature_from_covers(27, Ccov[27])
    say(f"`c_e(27)` in full (never enumerated; `2^26` subsets would be needed): "
        f"{sorted(c27.items())}.")
    say()
    say(f"Cost: the transfer matrix runs in `O(d^2)`; `d = 60` takes milliseconds against "
        f"`2^59` subsets for the enumeration.")
    summary["rows_to_60"] = rows8

    json.dump(summary, open(os.path.join(RES, "covers.json"), "w"), indent=1)
    with open(os.path.join(RES, "covers.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
