"""R2.a.i.a.1.a.2 - second moment over q.  The exact pairwise joint density.

Four parts, all exact:

A) the overlap o_g(i, j) = # classes of q mod g striking both islands, exhaustive over every
   gear 11..GMAX and every residue pair (i, j) mod g: verifies o_g in {0, 2, 4}, never odd, and
   that o_g > 0 only when g | delta, g | 3 delta - 1 or g | 3 delta + 1  (delta = j - i).

B) the joint-density formula
       rho_2(i,j) = prod_g ( 1 - a_g(i) - a_g(j) + o_g(i,j)/(g-1) ),   a_g(i) = 2 chi_g(i)/(g-1)
   against brute force over a FULL period of the gear product, gear sets {11,13}, {11,13,17},
   {11,13,17,19} (q ranging over the classes coprime to the product).

C) the per-gear log correction L_g(i,j) = log[(1-a_i-a_j+o/(g-1)) / ((1-a_i)(1-a_j))], its mean
   over all island pairs, gear by gear: tests whether the disjointness deficit -4/g^2 is
   cancelled by the coincidence families (g | delta, g | 3 delta -+ 1).

D) the exact first and second CRT moments mu(q) = sum_i rho_1(i), Var_model(q) = sum_ij rho_2 - mu^2
   at sampled q, against the measured N(q).

Usage: uv run python research/anchor235/r50/mom_pair.py [--PART ABCD]
"""
import argparse
import os
from math import isqrt, log

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
XS = [1000, 2000, 4000, 8000, 16000, 32000, 64000]


def sieve_flags(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def primes_upto(n):
    fl = sieve_flags(n)
    return [i for i in range(2, n + 1) if fl[i]]


def spf_sieve(n):
    """smallest prime factor table."""
    s = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, isqrt(n) + 1):
        if s[i] == 0:
            s[i * i:: i][s[i * i:: i] == 0] = i
    for i in range(2, n + 1):
        if s[i] == 0:
            s[i] = i
    return s


def factor_over7(x, spf):
    """distinct prime factors of x that exceed 7."""
    out = []
    while x > 1:
        p = int(spf[x])
        if p > 7:
            out.append(p)
        while x % p == 0:
            x //= p
    return out


# ---------------------------------------------------------------- A


def part_A(GMAX=2000):
    lines = []
    bad_odd = bad_val = bad_supp = cells = 0
    diag_bad = 0
    ngear = 0
    space = 0
    nz = 0
    for g in primes_upto(GMAX):
        if g < 11:
            continue
        ngear += 1
        space += g * g
        u = pow(6, -1, g)
        cnt = {}
        chi = np.zeros(g, dtype=np.int64)
        for r in range(1, g):
            s = (r * r) % g
            i1 = (-s * u) % g
            i2 = ((2 - s) * u) % g
            chi[i1] += 1
            chi[i2] += 1
            if i1 != i2:
                cnt[(i1, i2)] = cnt.get((i1, i2), 0) + 1
                cnt[(i2, i1)] = cnt.get((i2, i1), 0) + 1
        # chi[i] here counts striking classes = 2 chi_g(i); diagonal overlap
        if (chi % 2).any() or chi.max() > 4:
            diag_bad += 1
        nz += len(cnt)
        for (i, j), o in cnt.items():
            cells += 1
            if o % 2:
                bad_odd += 1
            if o not in (0, 2, 4):
                bad_val += 1
            dl = (j - i) % g
            if not ((3 * dl) % g in (1, g - 1)):
                bad_supp += 1
        cells += g  # the g diagonal cells
    lines.append("A) overlap o_g(i,j), exhaustive over gears 11..%d and every residue pair" % GMAX)
    lines.append("   gears inspected                   : %d" % ngear)
    lines.append("   (gear, ordered residue pair) cells: %d" % space)
    lines.append("   off-diagonal cells with o > 0     : %d" % nz)
    lines.append("   cells with ODD o                  : %d" % bad_odd)
    lines.append("   cells with o not in {0,2,4}       : %d" % bad_val)
    lines.append("   cells with o>0 and 3*delta != +-1 : %d" % bad_supp)
    lines.append("   gears whose diagonal 2*chi is odd or > 4 : %d" % diag_bad)
    return lines


# ---------------------------------------------------------------- B


def chi_of(g, i):
    """(chi, qr_of_first_target) for offset i at gear g."""
    t1 = (-6 * i) % g
    t2 = (2 - 6 * i) % g
    q1 = 1 if (t1 and pow(t1, (g - 1) // 2, g) == 1) else 0
    q2 = 1 if (t2 and pow(t2, (g - 1) // 2, g) == 1) else 0
    return q1 + q2, q1, q2


def overlap(g, i, j):
    u = pow(6, -1, g)
    d = (j - i) % g
    if d == 0:
        c, _, _ = chi_of(g, i)
        return 2 * c
    t = (3 * d) % g
    if t == 1:
        x = (-6 * i) % g
    elif t == g - 1:
        x = (2 - 6 * i) % g
    else:
        return 0
    return 2 if (x and pow(x, (g - 1) // 2, g) == 1) else 0


def part_B():
    lines = ["B) joint-density formula against brute force over a full gear-product period"]
    sets = [[11, 13], [11, 13, 17], [11, 13, 17, 19]]
    worst = 0.0
    checks = 0
    for S in sets:
        P = 1
        for g in S:
            P *= g
        rs = np.arange(P)
        ok = np.ones(P, dtype=bool)
        for g in S:
            ok &= (rs % g) != 0
        rs = rs[ok]
        sq = (rs.astype(np.int64) ** 2)
        strike = {}
        for g in S:
            u = pow(6, -1, g)
            strike[g] = (sq % g, u)
        islands = [12 + 35 * k for k in range(0, 20)]
        for ai in range(len(islands)):
            for bj in range(ai, len(islands)):
                i, j = islands[ai], islands[bj]
                both = np.ones(len(rs), dtype=bool)
                for g in S:
                    r, u = strike[g]
                    si = (r == ((-6 * i) % g)) | (r == ((2 - 6 * i) % g))
                    sj = (r == ((-6 * j) % g)) | (r == ((2 - 6 * j) % g))
                    both &= ~(si | sj)
                meas = both.sum() / len(rs)
                pred = 1.0
                for g in S:
                    ci, _, _ = chi_of(g, i)
                    cj, _, _ = chi_of(g, j)
                    o = overlap(g, i, j)
                    pred *= 1.0 - (2 * ci + 2 * cj - o) / (g - 1)
                worst = max(worst, abs(meas - pred))
                checks += 1
        lines.append("   gears %-16s period %6d   checks %4d   worst |measured - formula| = %.2e"
                     % (str(S), P, checks, worst))
    lines.append("   total checks %d, worst deviation over all sets %.2e" % (checks, worst))
    return lines


# ---------------------------------------------------------------- shared machinery


def build_chi(m, gears):
    """chi (m x G) int8 and qr1 (m x G) int8 for islands i = 12 + 35 t, t = 0..m-1."""
    ii = 12 + 35 * np.arange(m, dtype=np.int64)
    G = len(gears)
    chi = np.zeros((m, G), dtype=np.int8)
    qr1 = np.zeros((m, G), dtype=np.int8)
    for k, g in enumerate(gears):
        g = int(g)
        half = (g + 1) // 2
        tab = np.zeros(g, dtype=np.int8)
        tab[(np.arange(1, half, dtype=np.int64) ** 2) % g] = 1
        t1 = (-6 * ii) % g
        t2 = (2 - 6 * ii) % g
        a1 = tab[t1]
        a2 = tab[t2]
        qr1[:, k] = a1
        chi[:, k] = a1 + a2
    return chi


def part_C(q=65003, GMAX=1000):
    """mean over island pairs of L_g, gear by gear."""
    lines = ["C) mean over island pairs of the per-gear log correction, q = %d" % q]
    d = (2 * pow(6, -1, q)) % q
    m = (d - 13) // 35 + 1
    gears = [p for p in primes_upto(GMAX) if p >= 11]
    chi = build_chi(m, gears)
    spf = spf_sieve(3 * 35 * m + 10)
    lines.append("   arc d = %d, islands m = %d, ordered pairs = %d" % (d, m, m * (m - 1)))
    lines.append("   gear   mean L_g        mean L_g * g^3   generic-only mean * g^2")
    rows = []
    for k, g in enumerate(gears):
        c = chi[:, k].astype(np.int64)
        a = 2.0 * c / (g - 1)
        at = a / (1.0 - a)
        # generic part: sum over ordered pairs i != j of log(1 - at_i at_j)
        vals = np.array([0.0, 2.0 / (g - 1), 4.0 / (g - 1)])
        atv = vals / (1 - vals)
        n = np.array([(c == 0).sum(), (c == 1).sum(), (c == 2).sum()], dtype=np.float64)
        T = np.log(1.0 - np.outer(atv, atv))
        tot = float((np.outer(n, n) * T).sum()) - float((np.log(1 - at * at)).sum())
        gen = tot
        # corrections at the coincidence gears
        corr = 0.0
        for kk in range(1, m):
            delta = 35 * kk
            sp = set(factor_over7(delta, spf)) | set(factor_over7(3 * delta - 1, spf)) | \
                 set(factor_over7(3 * delta + 1, spf))
            if g not in sp:
                continue
            idx = np.arange(0, m - kk)
            ai = a[idx]
            aj = a[idx + kk]
            if delta % g == 0:
                o = 2.0 * c[idx]
            else:
                t = (3 * delta) % g
                ii = 12 + 35 * idx
                if t == 1:
                    x = (-6 * ii) % g
                else:
                    x = (2 - 6 * ii) % g
                half = (g + 1) // 2
                tab = np.zeros(g, dtype=np.int8)
                tab[(np.arange(1, half, dtype=np.int64) ** 2) % g] = 1
                o = 2.0 * tab[x]
            true = np.log((1 - ai - aj + o / (g - 1)) / ((1 - ai) * (1 - aj)))
            genp = np.log(1 - (ai / (1 - ai)) * (aj / (1 - aj)))
            corr += 2.0 * float((true - genp).sum())
        tot += corr
        mean = tot / (m * (m - 1))
        genmean = gen / (m * (m - 1))
        rows.append((g, mean, mean * g ** 3, genmean * g ** 2))
    for g, mean, s3, s2 in rows:
        if g <= 100 or g % 7 == 3 or g > 900:
            lines.append("   %5d  %+.6e   %+9.4f        %+9.4f" % (g, mean, s3, s2))
    arr = np.array([r[2] for r in rows])
    arr2 = np.array([r[3] for r in rows])
    lines.append("   over gears 11..%d : max |mean L_g| * g^3 = %.4f ; "
                 "generic-only mean * g^2 in [%.4f, %.4f]"
                 % (GMAX, np.abs(arr).max(), arr2.min(), arr2.max()))
    lines.append("   sum over gears of mean L_g = %+.6e  (the mean pair log-correlation)"
                 % sum(r[1] for r in rows))
    return lines


def moments(q, gears_all, chunk=1024):
    """exact mu(q) and Var_model(q) for the CRT ensemble at q."""
    m, rho, M = pair_matrix(q, gears_all, chunk)
    mu = float(rho.sum())
    off = float(M.sum() - np.trace(M))
    var = mu + off - mu * mu
    cov = off - (mu * mu - float((rho * rho).sum()))
    return m, mu, var, cov


def pair_matrix(q, gears_all, chunk=1024):
    """m, rho_1 vector, and the m x m matrix rho_2(i, j) (diagonal = rho_1 rho_1 e^0, unused)."""
    d = (2 * pow(6, -1, q)) % q
    m = (d - 13) // 35 + 1
    ng = int(np.searchsorted(gears_all, q, side="right"))
    gears = gears_all[:ng]
    chi = build_chi(m, gears)
    # rho_1
    rho = np.ones(m)
    small = [k for k, g in enumerate(gears) if g <= 100]
    large = np.array([k for k, g in enumerate(gears) if g > 100], dtype=np.int64)
    for k in range(ng):
        g = int(gears[k])
        rho *= 1.0 - 2.0 * chi[:, k] / (g - 1)
    mu = float(rho.sum())
    LR = np.zeros((m, m))
    for k in small:
        g = int(gears[k])
        vals = np.array([0.0, 2.0 / (g - 1), 4.0 / (g - 1)])
        atv = vals / (1 - vals)
        T = np.log(1.0 - np.outer(atv, atv))
        c = chi[:, k].astype(np.int64)
        LR += T[np.ix_(c, c)] if False else T[c[:, None], c[None, :]]
    S1 = np.zeros((m, m))
    S2 = np.zeros((m, m))
    S3 = np.zeros((m, m))
    for s in range(0, len(large), chunk):
        cols = large[s:s + chunk]
        gg = gears[cols].astype(np.float64)
        A = 2.0 * chi[:, cols] / (gg - 1.0)
        At = A / (1.0 - A)
        S1 += At @ At.T
        A2 = At * At
        S2 += A2 @ A2.T
        A3 = A2 * At
        S3 += A3 @ A3.T
    LR += -S1 - S2 / 2.0 - S3 / 3.0
    del S1, S2, S3
    # coincidence corrections
    spf = spf_sieve(3 * 35 * m + 10)
    gpos = {int(g): k for k, g in enumerate(gears)}
    qrtab = {}
    for kk in range(1, m):
        delta = 35 * kk
        sp = set(factor_over7(delta, spf)) | set(factor_over7(3 * delta - 1, spf)) | \
             set(factor_over7(3 * delta + 1, spf))
        idx = np.arange(0, m - kk)
        for g in sp:
            k = gpos.get(g)
            if k is None:
                continue
            c = chi[:, k].astype(np.int64)
            a = 2.0 * c / (g - 1)
            ai = a[idx]
            aj = a[idx + kk]
            if delta % g == 0:
                o = 2.0 * c[idx]
            else:
                if g not in qrtab:
                    half = (g + 1) // 2
                    tab = np.zeros(g, dtype=np.int8)
                    tab[(np.arange(1, half, dtype=np.int64) ** 2) % g] = 1
                    qrtab[g] = tab
                tab = qrtab[g]
                t = (3 * delta) % g
                ii = 12 + 35 * idx
                x = ((-6 * ii) % g) if t == 1 else ((2 - 6 * ii) % g)
                o = 2.0 * tab[x]
            true = np.log((1 - ai - aj + o / (g - 1)) / ((1 - ai) * (1 - aj)))
            genp = np.log(1 - (ai / (1 - ai)) * (aj / (1 - aj)))
            dv = true - genp
            LR[idx, idx + kk] += dv
            LR[idx + kk, idx] += dv
    M = np.outer(rho, rho) * np.exp(LR)
    return m, rho, M


def part_D(nper=6):
    lines = ["D) exact CRT moments at sampled q against the measured N(q)"]
    lines.append("   band       q  arc     m     mu(q)   Var_model   Var/mu    sumC/mu^2   N(q)")
    gears_all = np.array([p for p in primes_upto(2 * XS[-1] + 10) if p >= 11], dtype=np.int64)
    rows = []
    for X in XS:
        z = np.load(os.path.join(OUT, "mom_scan_X%d.npz" % X))
        qs, Ns = z["q"], z["N"]
        fl = sieve_flags(2 * X + 10)
        cand = [(int(q), int(n)) for q, n in zip(qs, Ns) if fl[int(q)]]
        shortq = [c for c in cand if c[0] % 6 == 5]
        longq = [c for c in cand if c[0] % 6 == 1]
        pick = []
        for lst in (shortq, longq):
            for f in (0.1, 0.5, 0.9):
                pick.append(lst[int(f * (len(lst) - 1))])
        for q, N in pick:
            m, mu, var, cov = moments(q, gears_all)
            rows.append((X, q, "short" if q % 6 == 5 else "long", m, mu, var, var / mu,
                         cov / mu ** 2, N))
            lines.append("   %6d %7d %-6s %5d %9.4f %11.4f %8.4f %12.6f %6d"
                         % rows[-1])
            print(lines[-1], flush=True)
    arr = np.array([r[6] for r in rows])
    arr2 = np.array([r[7] for r in rows])
    lines.append("   Var_model/mu over %d sampled q: min %.4f max %.4f" % (len(rows), arr.min(), arr.max()))
    lines.append("   sum_{i!=j} C(i,j)/mu^2: min %+.6f max %+.6f" % (arr2.min(), arr2.max()))
    nn = np.array([r[8] for r in rows], dtype=float)
    mm = np.array([r[4] for r in rows])
    lines.append("   mean N/mu over the sample = %.4f (the s=2 handicap 1/(4e^-2g) = 0.79246)"
                 % (nn / mm).mean())
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--PART", default="ABCD")
    args = ap.parse_args()
    out = []
    if "A" in args.PART:
        out += part_A()
    if "B" in args.PART:
        out += part_B()
    if "C" in args.PART:
        out += part_C()
    if "D" in args.PART:
        out += part_D()
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(OUT, "mom_pair_%s.txt" % args.PART), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
