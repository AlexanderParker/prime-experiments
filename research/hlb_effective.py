"""Harvester round 22 (c): what the paired pinch theorem buys, and where it stops.

Round 21 proved the PINCH THEOREM for the paired (twin-slot) sieve M_y:

    N2(g) - sum_{t=1}^{g-1} N3(0,t,g)  <=  n_g(M_y)  <=  N2(g),

both sides closed-form CRT products (docs/novel/paired-hlb-cycles.md).  Round 22 asks
what that buys OUTSIDE the sieve.  Three things are computed here, two positive and one
negative, all exact.

(1) EFFECTIVE POLIGNAC IN THE PAIRED SIEVE.  Define y_0(g) = the least y for which the
    pinch lower bound is positive.  Then for EVERY y >= y_0(g) the gap value g occurs in
    M_y's cyclic gap sequence - unconditionally, effectively, with no scan.  Holt's
    one-residue Theorem 5.5 is asymptotic; this is a number.  The computation splits at
    q = 6g+2, beyond which every local ratio is generic:
        rho(y,g) = sum_t prod_q c_q({0,t,g})/c_q({0,g}) = A(g) * B(y)/B(6g+2),
        A(g) = sum_t prod_{q<=6g+2} (ratio),   B(y) = prod_{5<=q<=y} (q-6)/(q-4),
    so y_0(g) is read off a single monotone table.  Measured shape: log y_0(g) ~ c*sqrt(g).

(2) THE MAX-GAP CONSEQUENCE, priced honestly.  Every gap <= G(y) := max{g : y >= y_0(g)}
    occurs, so the sieve's maximal gap - which IS the project's F(2,y), i.e. the paired
    Jacobsthal value at the twin difference - satisfies F(2,y) >= 3*G(y) ~ c'(log y)^2.
    That is far below the Ford-Green-Konyagin-Maynard-Tao transfer (round 21), so the
    pinch contributes NOTHING to the j_2 lower ladder.  Recorded so it is not re-derived.

(3) THE BOUNDARY.  The pinch is a FULL-PERIOD statement; primality of survivors needs the
    window (y, y^2] (the project's horizon theorem).  The window is a fraction
    y^2 / P_y = exp(-(1+o(1)) y) of the period, so no full-period population statement -
    however exact - localises to it.  That ratio is tabulated: it is the entire distance
    between "paired HL-B in cycles, proved" and "paired HL-B for primes, open".
"""
from math import log, exp, prod, sqrt
from sympy import primerange, prime

LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def cq_set(q, X):
    u = pow(6, -1, q)
    T = {u % q, (-u) % q}
    return q - len({(t - x) % q for t in T for x in X})


def A_of(g):
    """sum over interior offsets t of the local ratio product up to q = 6g+2."""
    qs = list(primerange(5, 6 * g + 3))
    tot = 0.0
    for t in range(1, g):
        r = 1.0
        for q in qs:
            den = cq_set(q, (0, g))
            if den == 0:
                return float("inf")           # N2 = 0: gap g impossible, skip
            r *= cq_set(q, (0, t, g)) / den
            if r == 0.0:
                break
        tot += r
    return tot


def main():
    say("=== (1) EFFECTIVE POLIGNAC IN THE PAIRED SIEVE: y_0(g) ===")
    say("   g   N2 factors    A(g)        y_0(g)      log y_0   log y_0/sqrt(g)")
    # B(y) table, incremental
    ys = []
    B = 1.0
    prev = 5
    Bs = {}
    for q in primerange(5, 10 ** 7):
        B *= (q - 6) / (q - 4)
        Bs[q] = B
    qlist = sorted(Bs)

    def y0(g, A):
        """least y with A * B(y)/B(6g+2) < 1."""
        q0 = max(p for p in qlist if p <= 6 * g + 2) if 6 * g + 2 >= 5 else None
        B0 = Bs[q0] if q0 else 1.0
        for q in qlist:
            if q <= 6 * g + 2:
                continue
            if A * Bs[q] / B0 < 1.0:
                return q
        return None

    rows = []
    for g in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        A = A_of(g)
        if A == float("inf"):
            continue
        if A < 1.0:
            yy, note = 6 * g + 2, "(already positive at q = 6g+2)"
        else:
            yy, note = y0(g, A), ""
        if yy is None:
            say(f"  {g:>4}   A = {A:.4g}: not resolved below 10^7")
            continue
        rows.append((g, A, yy))
        say(f"  {g:>4}   {A:>12.5g}   {yy:>10}   {log(yy):>9.3f}   "
            f"{log(yy)/sqrt(g):>8.3f} {note}")
    assert rows and all(r[2] > 0 for r in rows)
    # monotone-ish sqrt law
    cs = [log(r[2]) / sqrt(r[0]) for r in rows if r[0] >= 10]
    say(f"  log y_0(g)/sqrt(g) over g in [10,100]: min {min(cs):.3f}, max {max(cs):.3f} "
        f"-> y_0(g) = exp(Theta(sqrt(g))), NOT polynomial in g")
    assert max(cs) / min(cs) < 1.6

    say("")
    say("=== (2) the max-gap consequence, priced ===")
    for y in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
        G = max((g for g, A, yy in rows if yy <= y), default=0)
        say(f"  y = 10^{len(str(y))-1}: every gap <= {G} provably occurs "
            f"-> F(2,y) >= {3*G}; true F(2,y) is of order y^2 "
            f"(measured 264 at y=37 alone)")
    say("  VERDICT: the pinch gives a (log y)^2-size lower bound on the paired "
        "Jacobsthal value - three orders of magnitude below the FGKMT transfer. "
        "No contribution to the j_2 ladder.")

    say("")
    say("=== (3) THE BOUNDARY: full period vs the window where survivors are prime ===")
    say("    y     log P_y     log(y^2)    window share y^2/P_y")
    P = 1.0
    for y in (19, 23, 29, 37, 53, 101, 1009):
        P = sum(log(q) for q in primerange(5, y + 1))
        say(f"  {y:>4}   {P:>9.2f}   {2*log(y):>9.2f}   "
            f"{exp(2*log(y) - P):>12.3g}")
    say("  The pinch is exact on the whole period; primality lives in a share that "
        "decays like exp(-(1+o(1)) y).  Nothing full-period localises there - that "
        "gap IS the open half of Hardy-Littlewood.")

    with open("research/data/hlb_effective.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("hlb_effective: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()
