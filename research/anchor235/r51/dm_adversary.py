"""dm_adversary.py - the localised distortion budget against the adversarial ladder A(K).

A(K) = the longest run of consecutive columns that K primes >= 5, each with two classes at
its own fixed separation and any phase, can block (research/proof/arc_multiset.md R1, exact
to K = 12).  The open lemma is A(K) < (p_{K+1}^2 - 1)/6.

The localised budget of BBMST Theorem 3.1 (see dm_budget.py) on an interval of length L is
    eta(L) = sum_i E[alpha_i^2],  alpha_i = fibre proportion struck by gear i,
and eta < 1 is the method's hypothesis.  eta is decreasing in L (longer interval -> longer
fibres -> smaller second moments), so there is a threshold L*(K): the shortest interval the
method can say anything about at all.  Anything shorter, and the method is silent.

For a K-gear adversary the worst gear set for the budget is the K SMALLEST gears (both
4/g^2 and 2/g fall with g), so L*(K) computed on {5, 7, ..., p_{K+2}} is the method's best
possible adversarial threshold.

Also tabulated: Stevens' 1977 one-class bound H(r) <= 2 r^(2 + 2e log r), the only printed
upper bound on "r arbitrary primes, one class each, blocking an interval".
"""

import os
from math import log, exp, e

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:: i] = bytearray(len(sieve[i * i:: i]))
    return [i for i in range(n + 1) if sieve[i]]


P = primes_upto(500)
GEARS = [p for p in P if p >= 5]

A_K = [2, 5, 7, 16, 22, 28, 37, 45, 68, 88, 101, 115]          # arc_multiset.md R1, exact
F_LADDER = [2, 5, 7, 11, 18, 25, 34, 43, 58, 88, 91, 103, 118]  # F({5..p_K})


def eta_avg(gs, L):
    """OPTIMISTIC model: every gear strikes its average share 2/g of the interval and the
    teeth fall in a fibre independently.  E[alpha^2] = 4/g^2 (m >= g) or 2/(mg) + 2/g^2."""
    tot = 0.0
    Q = 1
    for g in gs:
        m = max(1.0, L / Q) if Q <= L else 1.0
        t = 4.0 / (g * g) if m >= g else 2.0 / (m * g) + 2.0 / (g * g)
        tot += min(2.0 / g, t)
        Q *= g
    return tot


def eta_max(gs, L):
    """RIGOROUS model: the largest E[alpha^2] the phases can produce.
      * E[alpha] <= (2*ceil(L/g))/L <= 2/g + 2/L         (a gear strikes at most 2 per period)
      * alpha <= 2/m on a fibre of m columns, so E[alpha^2] <= (2/m) E[alpha] <= 4/(g m)
      * E[alpha^2] >= (E[alpha])^2 = 4/g^2               (Cauchy-Schwarz, the floor)
    so the term is max(4/g^2, min(2/g + 2/L, 4/(g m))).  At m >= g this is 4/g^2 (whole
    classes per fibre); at m = 1 (a collapsed gear) it is 2/g + 2/L, the first moment."""
    tot = 0.0
    Q = 1
    for g in gs:
        m = max(1.0, L / Q) if Q <= L else 1.0
        sup_a = min(1.0, 2.0 / m)                 # alpha <= (teeth in a fibre)/m <= 2/m, <= 1
        e_a = min(1.0, 2.0 / g + 2.0 / L)         # E[alpha] <= 2*ceil(L/g)/L
        t = max(4.0 / (g * g), sup_a * e_a)       # E[alpha^2] in [ (E a)^2 , sup_a * E a ]
        tot += t
        Q *= g
    return tot


def threshold(gs, f=None):
    f = f or eta_max
    hi = 1e300
    if f(gs, hi) >= 1.0:
        return None
    lo = 1.0
    for _ in range(400):
        mid = exp((log(lo) + log(hi)) / 2)
        if f(gs, mid) < 1.0:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0000001:
            break
    return hi


eta_model = eta_max


def collapsed(gs, L):
    """number of gears whose fibre inside an interval of length L is a single column"""
    c = 0
    Q = 1
    for g in gs:
        if Q > L:
            c += 1
        Q *= g
    return c


if __name__ == "__main__":
    say("=" * 92)
    say("THE LOCALISED DISTORTION BUDGET AGAINST THE ADVERSARIAL LADDER A(K)")
    say("  L*(K)  = shortest interval the localised budget can speak about (eta < 1),")
    say("           computed on the K smallest gears (the worst K-set for the budget)")
    say("  W      = (p_{K+1}^2 - 1)/6, the window the open lemma asks A(K) to stay below")
    say("  coll@W = gears whose fibre inside an interval of length W is one column")
    say("           (for those the method's second moment equals its first: the collapse)")
    say("=" * 92)
    say(f"{'K':>3} {'gears':>19} {'A(K)':>5} {'W':>6} {'L*max':>10} {'L*/W':>9} "
        f"{'L*avg':>9} {'avg<A?':>7} {'coll@W':>6} {'eta@W':>7}")
    for K in list(range(1, 13)) + [14, 16, 18, 21, 24, 28, 32]:
        gs = GEARS[:K]
        pk1 = P[P.index(gs[-1]) + 1]
        W = (pk1 * pk1 - 1) // 6
        Ls = threshold(gs, eta_max)
        La = threshold(gs, eta_avg)
        A = A_K[K - 1] if K <= 12 else 0
        e_at_W = eta_max(gs, W)
        gstr = ",".join(str(g) for g in gs)
        if len(gstr) > 19:
            gstr = gstr[:16] + "..."
        bad = ("REFUTED" if (La is not None and La <= A) else "ok") if A else "-"
        say(f"{K:>3} {gstr:>19} {A:>5} {W:>6} "
            f"{(f'{Ls:.3e}' if Ls else 'none'):>10} "
            f"{(f'{Ls / W:.2e}' if Ls else '-'):>9} "
            f"{(f'{La:.2e}' if La else 'none'):>9} {bad:>7} "
            f"{collapsed(gs, W):>6} {e_at_W:>7.3f}")
    say()
    say("L*max: the rigorous localised threshold (the method is silent below it).")
    say("L*avg: the same with average first moments.  Where it is at or below the exact A(K)")
    say("       it asserts something FALSE (K = 4..11), which is the check that the average")
    say("       reading of the localisation is not a valid localisation: on a short interval")
    say("       the phases are chosen, and a chosen phase strikes more than its average share.")
    say()
    say("The budget's tail is the collapsed gears, which contribute their full capacity 2/g.")
    say("Cumulative capacity of the K smallest gears (the union bound the method degrades to):")
    cum = 0.0
    row = []
    for K in range(1, 13):
        cum += 2.0 / GEARS[K - 1]
        row.append(f"K={K}:{cum:.3f}")
    say("  " + "  ".join(row))
    say("  it passes 1 at K = 4 ({5,7,11,13}), so the localised budget can tolerate at most")
    say("  three collapsed gears plus the head; every longer machine needs the interval to be")
    say("  as long as the product of all but its top few gears.")
    say()
    say("=" * 92)
    say("STEVENS 1977, the one printed adversarial interval bound (ONE class per prime):")
    say("  H(r) <= 2 r^(2 + 2e log r)   [Hajdu-Saradha (1.1); Stevens, Math. Ann. 226 (1977)]")
    say("  against the project's two-class A(K).  H and A are different functions; the row is")
    say("  here to price the only bound of this shape that exists in print.")
    say("=" * 92)
    say(f"{'r':>3} {'2 r^(2+2e ln r)':>22} {'A(r) (2-class truth)':>22} {'ratio':>12}")
    for r in range(2, 13):
        b = 2.0 * r ** (2.0 + 2.0 * e * log(r))
        say(f"{r:>3} {b:>22.4e} {A_K[r - 1]:>22} {b / A_K[r - 1]:>12.3e}")
    say()
    with open(os.path.join(OUT, "dm_adversary.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("written:", os.path.join(OUT, "dm_adversary.txt"))
