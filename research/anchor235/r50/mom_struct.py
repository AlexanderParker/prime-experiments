"""R2.a.i.a.1.a.2 - second moment over q.  Where the negative pair correlation lives.

E) end-to-end brute force: the EXACT distribution of N over a full period of a small gear
   product, its variance, against the pairwise-density prediction.  Validates the whole
   Var_model machinery, not just the pair formula.

F) the exact first moment mu(q) = sum_i rho_1(i) against the mean-rate proxy
   mu_hat(q) = m prod (1 - 2/g), and both against the measured N.

G) the covariance C(i,j) = rho_2 - rho_1 rho_1 aggregated by island separation delta = j - i:
   which separations carry the sub-Poisson deficit.

Usage: uv run python research/anchor235/r50/mom_struct.py [--PART EFG]
"""
import argparse
import os

import numpy as np

import mom_pair as MP

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
XS = [1000, 2000, 4000, 8000, 16000, 32000, 64000]


def part_E():
    lines = ["E) exact distribution of N over a full gear-product period, against the pair model"]
    for S, T in ((["11", "13"], 30), ([11, 13, 17], 30), ([11, 13, 17, 19], 40),
                 ([11, 13, 17, 19], 80)):
        S = [int(x) for x in S]
        P = 1
        for g in S:
            P *= g
        rs = np.arange(P, dtype=np.int64)
        ok = np.ones(P, dtype=bool)
        for g in S:
            ok &= (rs % g) != 0
        rs = rs[ok]
        sq = rs * rs
        islands = np.array([12 + 35 * k for k in range(T)], dtype=np.int64)
        N = np.zeros(len(rs), dtype=np.int32)
        for i in islands:
            open_i = np.ones(len(rs), dtype=bool)
            for g in S:
                r = sq % g
                open_i &= (r != ((-6 * int(i)) % g)) & (r != ((2 - 6 * int(i)) % g))
            N += open_i
        Em, Vm = N.mean(), N.var()
        rho = np.ones(T)
        for k, i in enumerate(islands):
            for g in S:
                c, _, _ = MP.chi_of(g, int(i))
                rho[k] *= 1.0 - 2.0 * c / (g - 1)
        mu = rho.sum()
        off = 0.0
        for k1 in range(T):
            for k2 in range(T):
                if k1 == k2:
                    continue
                i, j = int(islands[k1]), int(islands[k2])
                p = 1.0
                for g in S:
                    ci, _, _ = MP.chi_of(g, i)
                    cj, _, _ = MP.chi_of(g, j)
                    o = MP.overlap(g, i, j)
                    p *= 1.0 - (2 * ci + 2 * cj - o) / (g - 1)
                off += p
        var = mu + off - mu * mu
        lines.append("   gears %-18s period %6d islands %3d :  brute E=%.6f Var=%.6f | "
                     "model mu=%.6f Var=%.6f | dE=%.2e dV=%.2e"
                     % (str(S), P, T, Em, Vm, mu, var, abs(Em - mu), abs(Vm - var)))
    return lines


def part_F():
    lines = ["F) exact first moment against the mean-rate proxy, and both against N"]
    lines.append("   band       q  arc     m   mu_exact   mu_hat   mu_ex/mu_hat   N    N/mu_ex  N/mu_hat")
    gears_all = np.array([p for p in MP.primes_upto(2 * XS[-1] + 10) if p >= 11], dtype=np.int64)
    rat1, rat2, rat3 = [], [], []
    for X in XS:
        z = np.load(os.path.join(OUT, "mom_scan_X%d.npz" % X))
        qs, Ns, mus = z["q"], z["N"], z["mu_hat"]
        fl = MP.sieve_flags(2 * X + 10)
        cand = [(int(q), int(n), float(u)) for q, n, u in zip(qs, Ns, mus) if fl[int(q)]]
        shortq = [c for c in cand if c[0] % 6 == 5]
        longq = [c for c in cand if c[0] % 6 == 1]
        for lst in (shortq, longq):
            for f in (0.1, 0.5, 0.9):
                q, N, mh = lst[int(f * (len(lst) - 1))]
                d = (2 * pow(6, -1, q)) % q
                m = (d - 13) // 35 + 1
                ng = int(np.searchsorted(gears_all, q, side="right"))
                gears = gears_all[:ng]
                chi = MP.build_chi(m, gears)
                rho = np.ones(m)
                for k in range(ng):
                    rho *= 1.0 - 2.0 * chi[:, k] / (int(gears[k]) - 1)
                mu = float(rho.sum())
                rat1.append(mu / mh)
                rat2.append(N / mu)
                rat3.append(N / mh)
                lines.append("   %6d %7d %-6s %5d %9.4f %8.4f %13.5f %5d %9.4f %9.4f"
                             % (X, q, "short" if q % 6 == 5 else "long", m, mu, mh,
                                mu / mh, N, N / mu, N / mh))
                print(lines[-1], flush=True)
    lines.append("   mean mu_exact/mu_hat = %.5f   mean N/mu_exact = %.5f   mean N/mu_hat = %.5f"
                 % (np.mean(rat1), np.mean(rat2), np.mean(rat3)))
    lines.append("   (the s=2 opening handicap is 1/(4 e^-2gamma) = 0.79246)")
    return lines


def part_G(qs=(30307, 60727, 15259)):
    lines = ["G) the covariance C(i,j) by island separation delta = 35 k"]
    gears_all = np.array([p for p in MP.primes_upto(2 * XS[-1] + 10) if p >= 11], dtype=np.int64)
    for q in qs:
        d = (2 * pow(6, -1, q)) % q
        m = (d - 13) // 35 + 1
        ng = int(np.searchsorted(gears_all, q, side="right"))
        gears = gears_all[:ng]
        chi = MP.build_chi(m, gears)
        rho = np.ones(m)
        for k in range(ng):
            rho *= 1.0 - 2.0 * chi[:, k] / (int(gears[k]) - 1)
        mu = float(rho.sum())
        # reuse the LR machinery from mom_pair.moments by recomputing it here
        m2, mu2, var, cov = MP.moments(q, gears_all)
        lines.append("   q = %d  arc %s  m = %d  mu = %.4f  Var_model = %.4f  Var/mu = %.4f"
                     % (q, "short" if q % 6 == 5 else "long", m, mu, var, var / mu))
        # rebuild the covariance matrix cheaply for the delta profile
        _mm, rho, M = MP.pair_matrix(q, gears_all)
        Cm = M - np.outer(rho, rho)
        np.fill_diagonal(Cm, 0.0)
        tot = Cm.sum()
        lines.append("      total sum_{i!=j} C = %+.6f   (= %.4f x mu)" % (tot, tot / mu))
        lines.append("      k   delta   #pairs    sum C over that separation   cumulative / total")
        cum = 0.0
        for k in list(range(1, 13)) + [16, 20, 25, 33, 50, 100]:
            if k >= m:
                break
            s = float(np.trace(Cm, offset=k) * 2)
            cum += s
            lines.append("     %3d %6d %8d   %+14.6f   %8.4f" % (k, 35 * k, m - k, s, cum / tot))
        allk = np.array([float(np.trace(Cm, offset=k) * 2) for k in range(1, m)])
        top = np.argsort(allk)[:8] + 1
        lines.append("      the 8 most negative separations: k = %s" % str(sorted(top.tolist())))
        lines.append("      sum over k <= 12 = %.4f of the total; k <= 100 = %.4f"
                     % (allk[:12].sum() / tot, allk[:100].sum() / tot))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--PART", default="EFG")
    args = ap.parse_args()
    out = []
    if "E" in args.PART:
        out += part_E()
    if "F" in args.PART:
        out += part_F()
    if "G" in args.PART:
        out += part_G()
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(OUT, "mom_struct_%s.txt" % args.PART), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
