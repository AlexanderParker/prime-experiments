"""Round 20 (mechanic): CHARACTER-SUM EVENTS - the complex-numbers frame
measured against census values, exactly.

(1) POWER SPECTRUM OF THE EXPOSED SET.  For gear q with teeth {u, -u}
    (u = 6^{-1} mod q), the DFT of the exposed-set indicator is
        hat1_A(t) = (q-2) delta_{t=0} - 2 cos(2 pi u t / q)  (t != 0)
    and Wiener-Khinchin gives lateral's closed form c_q(g) = q-2/q-3/q-4
    as a two-line Fourier computation whose three delta terms ARE the
    three tooth-relationships.  Verified here for every gear 5..53 at
    every lag against the brute-force census (exact integer match after
    rounding; tolerance 1e-9 before rounding).

(2) THE CORRIDOR MOD 35: hat1_{A35}(s,t) factorises as the product of the
    gear DFTs, and the inverse transform at lag g reproduces
    c_5(g) * c_7(g) - the admissible-phase count - at all 35 lags.

(3) DFT OF THE GAP-VALUE HISTOGRAM (the unexplained r17 residue law).
    For each machine's full-period hist_M[v] and each small gear p:
        H_p(t) = sum_v hist[v] e(2 pi i t v / p).
    Reported: |H_p(1)|/H_p(0) (the strength of the mod-p ripple) and
    arg H_p(1) compared against the phase 2 pi s_p / p that concentration
    at v = +-s_p (s_p = 2u_p, the tooth separation = the letter) would
    produce.  This turns 'richest classes are +-s' into one complex
    number per (machine, gear).

Usage: uv run python research/dft_events.py
"""
import os
import sys
import cmath
import math

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")


def primes_upto(n):
    s = list(range(n + 1))
    for i in range(2, int(n ** 0.5) + 1):
        if s[i] == i:
            for j in range(i * i, n + 1, i):
                if s[j] == j:
                    s[j] = i
    return [i for i in range(2, n + 1) if s[i] == i]


def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q


def c_brute(q, g):
    t = set(teeth(q))
    return sum(1 for r in range(q)
               if r not in t and (r + g) % q not in t)


def part1():
    print("(1) exposed-set power spectrum vs census, gears 5..53:")
    bad = 0
    for q in [p for p in primes_upto(53) if p >= 5]:
        u = pow(6, -1, q)
        for g in range(q):
            # Fourier evaluation of c_q(g)
            s = (q - 2) ** 2
            for t in range(1, q):
                s += (4 * math.cos(2 * math.pi * u * t / q) ** 2
                      * math.cos(2 * math.pi * g * t / q))
            four = s / q
            cb = c_brute(q, g)
            if abs(four - cb) > 1e-9:
                bad += 1
                print(f"    MISMATCH q={q} g={g}: fourier {four} vs "
                      f"census {cb}")
    print(f"    all gears, all lags: {'ZERO mismatches' if bad == 0 else f'{bad} mismatches'}"
          f" (sum of {sum(q for q in primes_upto(53) if q>=5)} lag checks)")


def part2():
    print("(2) corridor mod 35: product DFT vs admissible-phase census:")
    t5, t7 = set(teeth(5)), set(teeth(7))
    A35 = [r for r in range(35) if r % 5 not in t5 and r % 7 not in t7]
    bad = 0
    for g in range(35):
        census = sum(1 for r in A35 if (r + g) % 35 in A35)
        prod_c = c_brute(5, g % 5) * c_brute(7, g % 7)
        # direct 35-point DFT route
        four = 0
        for t in range(35):
            h = sum(cmath.exp(-2j * math.pi * r * t / 35) for r in A35)
            four += abs(h) ** 2 * cmath.exp(2j * math.pi * g * t / 35)
        four = four.real / 35
        if abs(census - prod_c) > 0 or abs(four - census) > 1e-6:
            bad += 1
            print(f"    MISMATCH g={g}: census {census} product {prod_c} "
                  f"fourier {four:.6f}")
    print(f"    35 lags: {'ZERO mismatches (census = c5*c7 = inverse-DFT '
          'of |product spectrum|^2)' if bad == 0 else f'{bad} mismatches'}")


def load_ghist(y):
    h = {}
    p = os.path.join(DDIR, "gap_pair_hist.csv")
    with open(p) as f:
        next(f)
        for line in f:
            yy, cov, kind, idx, v, c = line.strip().split(",")
            if int(yy) == y and kind == "ghist" and float(cov) == 1.0:
                h[int(v)] = h.get(int(v), 0) + int(c)
    return h


def part3():
    print("(3) DFT of the full-period gap-value histogram (residue-law "
          "phases):")
    print("    machine  gear p  |H(1)|/H(0)   arg H(1) (deg)   "
          "phase of +-s ripple: 0 deg by construction; s_p = 2u_p")
    for y in [13, 17, 19, 23, 29, 31, 37]:
        h = load_ghist(y)
        if not h:
            continue
        for p in (5, 7):
            u = pow(6, -1, p)
            s = 2 * u % p
            H0 = sum(h.values())
            H1 = sum(c * cmath.exp(2j * math.pi * v / p)
                     for v, c in h.items())
            # rotate so that concentration at +-s (equal weight) -> arg 0
            rot = H1 * cmath.exp(0j)          # raw
            # expected arg if ripple centred at +-s: H1 real positive
            # after multiplying by e(-i*0) since cos(2pi(v-s)/p)+
            # cos(2pi(v+s)/p) = 2cos(2pi v/p)cos(2pi s/p) - real axis.
            amp = abs(H1) / H0
            ang = math.degrees(cmath.phase(H1))
            pred = 0.0
            print(f"      {y:5d}   {p}      {amp:8.4f}      {ang:+9.2f}"
                  f"        (s_{p} = {s}; cos-ripple at +-s -> arg 0 or "
                  f"180)")


if __name__ == "__main__":
    part1()
    part2()
    part3()
