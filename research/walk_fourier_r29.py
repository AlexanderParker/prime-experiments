"""
LATERAL round 29, BLOCK C - THE ANCHOR-235 FLOOR AS A CHARACTER SUM.

anchor-235.md 9g names the floor: "below the scan, a form would have to compute
the first integer outside a union of 2 pi(q) arithmetic progressions from the
pi(q) residues of s alone; none is known here and none was found."  W(s) is the
distance from slot s to the next OPEN slot; the scan form
W(s) = sum_{j>=1} prod_{i<j} B(s+i) is exact with F+1 terms of 2 pi(q) residue
tests.  This script writes W in additive characters, PRICES every exact form by
term count, tests the two natural character-sum bounds exactly, and says what is
left.

WHAT IS DERIVED HERE (all verified numerically in this script):

 1. THE OPEN INDICATOR'S TRANSFORM IS CLOSED FORM AND FACTORISES (lateral item
    29a, re-gated here): Shat(m) = sum_{open o} e(-mo/P) = prod_q hat_q(m),
    hat_q(0 mod q) = q-2, hat_q(j) = -2 cos(2 pi j v_q / q).  REAL, pi(q)
    multiplications per frequency, no scan.

 2. THE WALK'S TRANSFORM HAS A POLE FACTOR AND ONE HARD FACTOR.  From the exact
    recursion W(s) = 1 + B(s+1) W(s+1),

        What(m) (1 - e(m/P))  =  - e(m/P) Ghat(m),      m != 0,
        Ghat(m) := sum_{open o} g(o) e(-mo/P)   (the GAP-WEIGHTED opening sum).

    The pole factor 1/(1 - e(m/P)) is exactly the shape of lateral's round-21
    pole-phase law H_p(k) = [omega/(1-omega)] B: THE POLE-PHASE LAW IS THE
    WALK'S OWN FOURIER TRANSFORM.

 3. THE SPLIT.  Ghat = lambda Shat + Dhat with lambda = P/N the mean gap; the
    first term is CLOSED FORM, the second is the gap-FLUCTUATION transform.
    Parseval gives the exact energy shares lambda^2 : Var(g).

 4. THE TWO NATURAL BOUNDS, PRICED EXACTLY.  For the count of openings in a
    window of length L, N_L(s) = (1/P) sum_m Shat(m) D_L(m) e(ms/P) with
    D_L(m) = sum_{i=1..L} e(mi/P):
      * L1 (large-sieve shape): |N_L - L N/P| <= (1/P) sum_{m!=0} |Shat| |D_L|.
      * L2 (Chebyshev): #{s : N_L(s) = 0} <= P Var(N_L) / mu^2, with Var exactly
        closed form from c(d) = prod_q c_q(d), c_q(0) = q-2, c_q(d) = q-3 if
        d = +-2v_q (mod q), q-4 otherwise (lateral item 21).

Usage: python walk_fourier_r29.py [--upto 19] [--closed-upto 23]
"""
import argparse
import math
import sys

import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23]
NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def build(gears):
    P = 1
    for q in gears:
        P *= q
    v = {q: min(pow(6, -1, q) % q, (-pow(6, -1, q)) % q) for q in gears}
    blocked = np.zeros(P, dtype=bool)
    for q in gears:
        blocked[v[q] % q::q] = True
        blocked[(-v[q]) % q::q] = True
    chi = (~blocked)
    op = np.flatnonzero(chi).astype(np.int64)
    return P, chi, op, v


def walk_fast(chi, P):
    """Vectorised W: for each slot, the next open slot minus itself."""
    op = np.flatnonzero(chi).astype(np.int64)
    # next opening strictly greater than s
    idx = np.searchsorted(op, np.arange(P, dtype=np.int64), side="right")
    nxt = np.where(idx < op.size, op[np.minimum(idx, op.size - 1)], op[0] + P)
    return nxt - np.arange(P, dtype=np.int64)


def cq_closed(d, gears, v):
    """c(d) = #{s : s and s+d both open}, closed form (lateral item 21)."""
    out = 1
    for q in gears:
        r = d % q
        u2 = (2 * v[q]) % q
        if r == 0:
            out *= q - 2
        elif r == u2 or r == (q - u2) % q:
            out *= q - 3
        else:
            out *= q - 4
    return out


def shat_closed(m, gears, v, P):
    """prod_q hat_q(m c_q) - the machine DFT in closed form (lateral item 29a).

    CRT makes the additive character mod P a PRODUCT of characters mod q, but
    the frequency each gear sees is m * c_q with c_q = (P/q)^{-1} mod q, not m
    itself: s = sum_q s_q E_q (mod P) with E_q = (P/q) c_q, so
    m s / P = sum_q m c_q s_q / q (mod 1).
    """
    out = 1.0
    for q in gears:
        cq = pow(P // q, -1, q)
        j = (m * cq) % q
        if j == 0:
            out *= (q - 2)
        else:
            out *= -2.0 * math.cos(2.0 * math.pi * j * v[q] / q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=19)
    ap.add_argument("--closed-upto", type=int, default=23)
    a = ap.parse_args()

    print("=" * 78)
    print("PART 1-3: the exact transforms, at machines with a computable period")
    print("=" * 78)
    rows = []
    for gi in range(2, len(PRIMES)):
        y = PRIMES[gi]
        if y > a.upto:
            break
        gears = PRIMES[:gi + 1]
        P, chi, op, v = build(gears)
        N = op.size
        Nref = int(np.prod([q - 2 for q in gears]))
        gate(N == Nref, "m%d: N = prod(q-2) = %d" % (y, N))
        W = walk_fast(chi, P)
        g = np.empty(N, dtype=np.int64)
        g[:-1] = op[1:] - op[:-1]
        g[-1] = P - op[-1] + op[0]
        F = int(g.max())
        gate(int(W.max()) == F, "m%d: max_s W(s) = F = %d" % (y, F))
        lam = P / N
        print("\n----- machine {5..%d}: P = %d, N = %d, F = %d, lambda = %.6f -----"
              % (y, P, N, F, lam))

        # --- 1. the closed-form machine DFT ---
        Shat = np.fft.fft(chi.astype(np.float64))          # sum_s chi(s) e(-2pi i m s/P)
        cf = np.array([shat_closed(m, gears, v, P) for m in range(min(P, 4000))])
        err = float(np.max(np.abs(Shat[:cf.size] - cf)))
        gate(err < 1e-6,
             "m%d: Shat(m) = prod_q hat_q(m) (closed form) to %.2e on the first "
             "%d frequencies" % (y, err, cf.size))
        gate(float(np.max(np.abs(Shat.imag))) < 1e-6,
             "m%d: Shat is REAL (max |Im| = %.2e)"
             % (y, float(np.max(np.abs(Shat.imag)))))

        # --- 2. the pole identity ---
        What = np.fft.fft(W.astype(np.float64))
        Gs = np.zeros(P, dtype=np.float64)
        Gs[op] = g
        Ghat = np.fft.fft(Gs)
        m = np.arange(P)
        z = np.exp(-2j * np.pi * m / P)      # e(-m/P); fft uses e(-2pi i m s/P)
        # W(s) = 1 + B(s+1) W(s+1)  ==>  What(m)(1 - conj(z)) = -conj(z) Ghat(m)
        eplus = np.exp(2j * np.pi * m / P)
        lhs = What * (1 - eplus)
        rhs = -eplus * Ghat
        e2 = float(np.max(np.abs(lhs[1:] - rhs[1:])))
        rel = e2 / float(np.max(np.abs(rhs[1:])))
        gate(rel < 1e-9,
             "m%d: POLE IDENTITY What(m)(1-e(m/P)) = -e(m/P) Ghat(m) at all "
             "%d nonzero frequencies (max rel err %.2e)" % (y, P - 1, rel))
        gate(abs(Ghat[0].real - P) < 1e-6, "m%d: Ghat(0) = P = %d" % (y, P))
        gate(abs(What[0].real - float(np.sum(g * (g + 1) // 2))) < 1e-3,
             "m%d: What(0) = sum_g W_1(g) g(g+1)/2" % y)

        # --- 3. the split and its energy shares (Parseval, exact) ---
        Dhat = Ghat - lam * Shat
        eS = float(np.sum(np.abs(lam * Shat) ** 2))
        eD = float(np.sum(np.abs(Dhat) ** 2))
        varg = float(np.mean((g - lam) ** 2))
        share = lam * lam / (lam * lam + varg)
        gate(abs(eS / (eS + eD) - share) < 1e-9,
             "m%d: Parseval - closed-form energy share = lambda^2/(lambda^2+Var g)"
             " = %.6f" % (y, share))
        m2 = float(np.mean(g.astype(np.float64) ** 2))
        print("   mean gap %.4f, Var(g) %.4f, E[g^2] %.4f -> CLOSED-FORM SHARE of "
              "the walk's Fourier energy = %.4f" % (lam, varg, m2, share))
        rows.append((y, P, N, F, lam, varg, share))

        # --- 4a. the L1 (large-sieve) price ---
        for L in (F - 1, F):
            mu = L * N / P
            k = np.arange(1, P)
            DL = np.abs(np.sin(np.pi * L * k / P) / np.sin(np.pi * k / P))
            l1 = float(np.sum(np.abs(Shat[1:]) * DL)) / P
            print("   L = %-3d  main term L*N/P = %10.4f   L1 error bound = "
                  "%14.4f   ratio %.3e" % (L, mu, l1, l1 / mu))
        # closed-form L1 mass, for the record
        mass = 1.0
        for q in gears:
            s = (q - 2) + sum(abs(2.0 * math.cos(2 * math.pi * j * v[q] / q))
                              for j in range(1, q))
            mass *= s / q
        print("   sum_m |Shat(m)| / P (closed form, prod over gears) = %.4f "
              "vs N/P = %.6f  -> L1 mass exceeds the density by %.1fx"
              % (mass, N / P, mass / (N / P)))

        # --- 4b. the L2 / Chebyshev price ---
        for L in (F - 1, F):
            mu = L * N / P
            tot = 0
            for d in range(-(L - 1), L):
                tot += (L - abs(d)) * cq_closed(d, gears, v)
            var = tot / P - mu * mu
            emptytrue = int(np.sum(W > L))
            bound = P * var / (mu * mu)
            print("   L = %-3d  mu = %8.4f  Var(N_L) = %10.4f (closed form)   "
                  "Chebyshev bound on #empty = %12.1f   TRUE #empty = %-8d  "
                  "vacuity factor %s"
                  % (L, mu, var, bound, emptytrue,
                     ("%.1fx" % (bound / emptytrue)) if emptytrue else
                     "bound>0 while truth is 0 - NO CERTIFICATE"))
        # gate the closed-form pair correlation against the direct count
        for d in (1, 2, 3, 7, 11):
            direct = int(np.sum(chi & np.roll(chi, -d)))
            gate(direct == cq_closed(d, gears, v),
                 "m%d: c(%d) = prod_q c_q(%d) = %d (closed form == direct count)"
                 % (y, d, d, direct))

    print("\n" + "=" * 78)
    print("PART 5: TERM COUNTS of the exact forms (no asymptotics)")
    print("=" * 78)
    print("  %-6s %-12s %-10s %-6s %-14s %-16s %-16s"
          % ("y", "P", "N", "F", "scan tests", "flat/DFT terms", "IE subsets"))
    for gi in range(2, len(PRIMES)):
        y = PRIMES[gi]
        if y > a.closed_upto:
            break
        gears = PRIMES[:gi + 1]
        P, chi, op, v = build(gears)
        N = op.size
        g = np.diff(np.concatenate([op, [op[0] + P]]))
        F = int(g.max())
        npi = len(gears)
        print("  %-6d %-12d %-10d %-6d %-14d %-16d %-16s"
              % (y, P, N, F, 2 * npi * (F + 1), P, "2^%d = %.3e" % (F + 1, 2.0 ** (F + 1))))
    part6()
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


def part6():
    """WHAT THE TWO BOUNDS CAN SEE, tested on the counterfactual tooth family.

    The family (lateral round 27) keeps the gears, the period and the survivor
    count and moves the teeth; F varies by 1.6-1.9x across it.  So any bound
    that is CONSTANT on the family provably cannot determine F.
    """
    import itertools
    print("\n" + "=" * 78)
    print("PART 6: WHAT THE TWO BOUNDS CAN SEE - on the COUNTERFACTUAL FAMILY")
    print("=" * 78)
    for gi in range(2, len(PRIMES)):
        y = PRIMES[gi]
        if y > 17:
            break
        gears = PRIMES[:gi + 1]
        P = 1
        for q in gears:
            P *= q
        N = int(np.prod([q - 2 for q in gears]))
        space = [list(range(1, (q - 1) // 2 + 1)) for q in gears]
        vecs = list(itertools.product(*space))
        Fs, L1s, L2c = [], [], []
        for vv in vecs:
            vd = {q: vv[i] for i, q in enumerate(gears)}
            blocked = np.zeros(P, dtype=bool)
            for q in gears:
                blocked[vd[q] % q::q] = True
                blocked[(-vd[q]) % q::q] = True
            op = np.flatnonzero(~blocked).astype(np.int64)
            gg = np.diff(np.concatenate([op, [op[0] + P]]))
            Fs.append(int(gg.max()))
            mass = 1.0
            for q in gears:
                sq = (q - 2) + sum(abs(2.0 * math.cos(2 * math.pi * j * vd[q] / q))
                                   for j in range(1, q))
                mass *= sq / q
            L1s.append(mass)
            # T(L) = sum_{|d|<L} (L-|d|) c(|d|) = 2 sum_{j<=L} S(j) - L c(0),
            # S(j) = sum_{d<j} c(d).  O(LMAX) with two cumsums instead of O(L^2).
            LMAX = 20000
            cv = np.array([cq_closed(d, gears, vd) for d in range(LMAX)],
                          dtype=np.float64)
            S = np.cumsum(cv)
            Ls = np.arange(1, LMAX + 1, dtype=np.float64)
            T = 2.0 * np.cumsum(S) - Ls * cv[0]
            mu = Ls * N / P
            var = T / P - mu * mu
            bnd = P * var / (mu * mu)
            hit = np.flatnonzero(bnd < 1.0)
            L2c.append(int(hit[0]) + 1 if hit.size else -1)
        Fs = np.array(Fs, dtype=float)
        L1s = np.array(L1s)
        L2c = np.array(L2c, dtype=float)
        gate(float(L1s.max() - L1s.min()) < 1e-9,
             "m%d: the L1 mass sum_m |Shat|/P is IDENTICAL (%.6f) at ALL %d "
             "counterfactual tooth vectors, while F ranges over [%d, %d] - the "
             "L1 character bound is PROVABLY BLIND TO THE TEETH"
             % (y, float(L1s[0]), len(vecs), int(Fs.min()), int(Fs.max())))
        rk = lambda x: np.argsort(np.argsort(x)).astype(float)
        sp = float(np.corrcoef(rk(Fs), rk(L2c))[0, 1])
        print("  m%-3d |V| = %-6d  F in [%d, %d] (spread %.2fx)   L2/Chebyshev "
              "certifying length in [%d, %d] (spread %.2fx)   median L2cert/F "
              "= %.1fx   spearman(F, L2cert) = %+.3f"
              % (y, len(vecs), int(Fs.min()), int(Fs.max()), Fs.max() / Fs.min(),
                 int(L2c.min()), int(L2c.max()), L2c.max() / L2c.min(),
                 float(np.median(L2c / Fs)), sp))


if __name__ == "__main__":
    sys.exit(main())
