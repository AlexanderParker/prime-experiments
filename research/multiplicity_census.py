"""Cross-root multiplicity distribution vs independence nulls (round 5).

Slot k's pair-hit multiplicity mu(k) = # gear pairs {q,q'} with q dividing
one member and q' the other. Slot-cap (no gear hits both members) makes
    mu(k) = omega_G(mL) * omega_G(mR)
exactly (omega_G = distinct gear divisors, gears = primes in [5, y]); each
unordered pair is counted once. mu >= 1 iff the slot is a double; sum of mu
over slots = S_pair (round 4's schedule - cross-validated here).

Real distribution: histogram of mu over the full window (degree sieve).
Boundary: a first-slot member <= y that is prime (the gear y itself)
counts omega = 0, consistent with rounds 1-4.

NULL 1 (coordinator's): each pair's 2 nontrivial cross classes mod qq' are
placed by CRT (exact per-pair window count c_i), but pairs are INDEPENDENT:
mu_null = Poisson-binomial(p_i = c_i/W). Exact pmf by DFT of the PGF
(N=128 points; aliasing negligible). Same mean as real BY CONSTRUCTION
(sum p_i = S_pair/W up to boundary convention) - all differences are
higher-moment.

NULL 2 (decomposition): keep the PRODUCT structure, break only the
arithmetic: omega'L, omega'R independent Poisson-binomials over per-gear
class counts (1 class/gear/member side), mu' = omega'L * omega'R with
independent sides. Isolates how much of the real compression is the
product structure alone.

Outputs (append): research/data/multiplicity_hist.csv (y, mu, real count,
null1 and null2 expected counts), multiplicity_summary.csv (moments).
Usage: uv run python research/multiplicity_census.py [y...]
Default ladder: 503 2003 10007 50021.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime

NDFT = 128


def real_hist(y, seg=16_000_000):
    gears = [q for q in primes_upto(y) if q >= 5]
    k_lo = -((-(y - 1)) // 6)
    k_hi = (y * y + 1) // 6
    uvals = [pow(6, -1, q) for q in gears]
    hist = np.zeros(NDFT, dtype=np.int64)
    n0 = 0
    for a in range(k_lo, k_hi + 1, seg):
        b = min(k_hi + 1, a + seg)
        n = b - a
        cntL = np.zeros(n, np.int16)
        cntR = np.zeros(n, np.int16)
        for q, u in zip(gears, uvals):
            cntL[(u - a) % q::q] += 1
            cntR[(-u - a) % q::q] += 1
        if a == k_lo:
            for arr, m in ((cntL, 6 * k_lo - 1), (cntR, 6 * k_lo + 1)):
                if m <= y and is_prime(m):
                    arr[0] = 0
        n0 += int(((cntL == 0) & (cntR == 0)).sum())
        mu = cntL.astype(np.int32) * cntR
        hist += np.bincount(mu, minlength=NDFT)[:NDFT]
    return gears, uvals, k_lo, k_hi, hist, n0


def pair_probs(gears, uvals, k_lo, k_hi, chunk=2_000_000):
    """exact per-pair window class counts -> probabilities p_i."""
    W = k_hi - k_lo + 1
    G = len(gears)
    ps = []
    Ms, Rs = [], []

    def flush():
        if not Ms:
            return
        M = np.array(Ms, dtype=np.int64)
        r1 = np.array(Rs, dtype=np.int64)
        c = np.zeros(len(M), dtype=np.int64)
        for r in (r1, (-r1) % M):
            c += (k_hi - r) // M - (k_lo - 1 - r) // M
        ps.append(c / W)
        Ms.clear()
        Rs.clear()
    for j in range(1, G):
        qj, uj = gears[j], uvals[j]
        for i in range(j):
            qi, ui = gears[i], uvals[i]
            Ms.append(qi * qj)
            Rs.append((ui + qi * (((-uj - ui) * pow(qi, -1, qj)) % qj))
                      % (qi * qj))
            if len(Ms) >= chunk:
                flush()
        if len(Ms) >= chunk:
            flush()
    flush()
    return np.concatenate(ps) if ps else np.array([])


def poisson_binomial_pmf(p, nchunk=100_000):
    """exact pmf of sum of independent Bernoulli(p_i), via PGF at NDFT
    roots of unity."""
    om = np.exp(2j * np.pi * np.arange(NDFT) / NDFT)
    logs = np.zeros(NDFT, dtype=np.complex128)
    for lo in range(0, len(p), nchunk):
        pc = p[lo:lo + nchunk, None]
        logs += np.log(1.0 - pc * (1.0 - om[None, :])).sum(axis=0)
    pgf = np.exp(logs)
    pmf = np.fft.fft(pgf).real / NDFT
    return np.clip(pmf, 0.0, None)


def gear_product_pmf(gears, uvals, k_lo, k_hi):
    """null 2: mu' = product of independent per-side omega'."""
    W = k_hi - k_lo + 1
    sides = []
    for sgn in (1, -1):
        p = []
        for q, u in zip(gears, uvals):
            r = (sgn * u) % q
            p.append(((k_hi - r) // q - (k_lo - 1 - r) // q) / W)
        sides.append(poisson_binomial_pmf(np.array(p)))
    pmf = np.zeros(NDFT)
    wa, wb = sides
    for a in range(NDFT):
        if wa[a] < 1e-15:
            continue
        for b_ in range(NDFT):
            m = a * b_
            if m >= NDFT:
                break
            pmf[m] += wa[a] * wb[b_]
    return pmf, float(wa[0] * wb[0])


def moments(hist_or_pmf, W=None):
    h = np.asarray(hist_or_pmf, dtype=float)
    tot = h.sum()
    p = h / tot
    mu = np.arange(len(h))
    mean = (p * mu).sum()
    var = (p * mu * mu).sum() - mean ** 2
    p0 = p[0]
    cond = mean / (1 - p0) if p0 < 1 else float("nan")
    t4 = p[4:].sum()
    t9 = p[9:].sum()
    return mean, var, p0, cond, t4, t9


def main():
    ys = [int(a) for a in sys.argv[1:]] or [503, 2003, 10007, 50021]
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)

    def opencsv(name, header):
        path = os.path.join(ddir, name)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a")
        if new:
            f.write(header + "\n")
        return f
    fh = opencsv("multiplicity_hist.csv",
                 "y,mu,count_real,expected_null_pairs,expected_null_gearprod")
    fs = opencsv("multiplicity_summary.csv",
                 "y,W,S_pair_check,n2_check,n0,mean,"
                 "P0_real,P0_null1,P0_null2,twin_share_real,both0_null2,"
                 "cond_real,cond_null1,cond_null2,"
                 "var_real,var_null1,var_null2,"
                 "tail4_real,tail4_null1,tail4_null2,"
                 "tail9_real,tail9_null1,tail9_null2")
    for y in ys:
        t0 = time.time()
        gears, uvals, k_lo, k_hi, hist, n0 = real_hist(y)
        W = k_hi - k_lo + 1
        p = pair_probs(gears, uvals, k_lo, k_hi)
        pmf1 = poisson_binomial_pmf(p)
        pmf2, both0 = gear_product_pmf(gears, uvals, k_lo, k_hi)
        S_pair = int((hist * np.arange(NDFT)).sum())
        n2 = int(hist[1:].sum())
        mR, vR, p0R, cR, t4R, t9R = moments(hist)
        m1, v1, p01, c1, t41, t91 = moments(pmf1)
        m2, v2, p02, c2, t42, t92 = moments(pmf2)
        for m in range(NDFT):
            if hist[m] or pmf1[m] * W > 1e-3 or pmf2[m] * W > 1e-3:
                fh.write(f"{y},{m},{hist[m]},{pmf1[m]*W:.3f},"
                         f"{pmf2[m]*W:.3f}\n")
        fs.write(f"{y},{W},{S_pair},{n2},{n0},{mR:.6f},"
                 f"{p0R:.6f},{p01:.6f},{p02:.6f},{n0/W:.6f},{both0:.6f},"
                 f"{cR:.4f},{c1:.4f},{c2:.4f},"
                 f"{vR:.4f},{v1:.4f},{v2:.4f},"
                 f"{t4R:.6f},{t41:.6f},{t42:.6f},"
                 f"{t9R:.6f},{t91:.6f},{t92:.6f}\n")
        print(f"y={y} W={W} mean={mR:.4f} (null1 {m1:.4f}, null2 {m2:.4f}) "
              f"S_pair={S_pair} n2={n2}")
        print(f"   P0: real {p0R:.4f} null1 {p01:.4f} null2 {p02:.4f} | "
              f"twin share: real {n0/W:.4f} null2-both0 {both0:.4f} | "
              f"cond mean: real {cR:.4f} null1 {c1:.4f} null2 {c2:.4f}")
        print(f"   var: real {vR:.3f} null1 {v1:.3f} null2 {v2:.3f} | "
              f"tail>=4: {t4R:.4f}/{t41:.4f}/{t42:.4f} "
              f"tail>=9: {t9R:.4f}/{t91:.4f}/{t92:.4f} "
              f"| {time.time()-t0:.0f}s")
        sys.stdout.flush()
    fh.close()
    fs.close()
    print("wrote multiplicity_hist.csv, multiplicity_summary.csv")


if __name__ == "__main__":
    main()
