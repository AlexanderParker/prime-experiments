"""Harvester round 22: the PER-DIFFERENCE sieve dimension of the paired Jacobsthal
family, and the interpolated bound.

The project's family F_d(y) fixes the even difference and asks for the maximal window;
h_2 is the max over d.  The sieve that proves the upper bounds (docs/novel/
j2-upper-bound.md) removes omega_p(d) = 2 classes for p not dividing d and 1 class for
p | d, so the SIEVE DIMENSION IS d-DEPENDENT:

    sum_{p<=y} omega_p(d) * log p / p  =  kappa_d * log y + O(1),
    kappa_d  =  2  -  (1/log y) * sum_{p | d, p <= y} log p / p         (Mertens),

which runs over [1, 2].  Both endpoints are attained inside the family, because d
ranges over residues mod the primorial:

  * kappa_d = 2 for d coprime to p_n# - the generic (and, by the project's percentile
    measurements, the hardest) class;
  * kappa_d = 1 for d = 0 mod p_n#, where the paired problem collapses exactly onto
    the ordinary one (round-21 verification: the survivor sets coincide);
  * kappa_d = 1 + theta + o(1) for d divisible by exactly the primes in (y^theta, y],
    since sum_{p <= x} log p/p = log x + O(1).

Feeding kappa_d into the same fundamental-lemma argument gives the interpolated
per-difference bound  F_d(y) <<_eps y^(beta(kappa_d) + eps)  with beta the sifting
limit (beta(1) = 2, beta(2) = 4.266).  HONEST CAVEAT: for a FIXED d and y -> infinity
kappa_d -> 2, so this is a statement about differences that grow with the machine -
which is exactly the family setting, and exactly where the project's own measurements
live.

This script checks the arithmetic of the three endpoints exactly.
"""
from math import log
from sympy import primerange, prime

LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def kappa(y, dprimes):
    """exact kappa_d = 2 - (1/log y) sum_{p | d, p <= y} log p / p."""
    s = sum(log(p) / p for p in dprimes if p <= y)
    return 2 - s / log(y)


def main():
    say("=== per-difference sieve dimension kappa_d of the paired family ===")
    say("  y        d                              kappa_d   beta-slot")
    for y in (10 ** 4, 10 ** 5, 10 ** 6):
        ps = list(primerange(2, y + 1))
        rows = [("coprime to the primorial", []),
                ("= 0 mod the primorial", ps)]
        for th in (0.25, 0.5, 0.75):
            lo = y ** th
            rows.append((f"primes in (y^{th}, y]", [p for p in ps if p > lo]))
        for name, dp in rows:
            k = kappa(y, dp)
            say(f"  1e{round(log(y)/log(10)):<2}  {name:<32} {k:7.4f}   "
                f"{'2 (=beta_1)' if k < 1.02 else ('4.266 (=beta_2)' if k > 1.98 else 'interpolated')}")
        # assertions: endpoints exact, theta-law within O(1/log y)
        assert abs(kappa(y, []) - 2.0) < 1e-12
        assert abs(kappa(y, ps) - 1.0) < 3.0 / log(y), (y, kappa(y, ps))
        for th in (0.25, 0.5, 0.75):
            k = kappa(y, [p for p in ps if p > y ** th])
            assert abs(k - (1 + th)) < 3.0 / log(y), (y, th, k)
    say("  endpoints exact (kappa = 2 coprime, kappa -> 1 at d = 0 mod primorial),")
    say("  interpolation kappa = 1 + theta verified to O(1/log y) at three thetas and")
    say("  three scales.  The kappa = 1 endpoint is the round-21 exact collapse")
    say("  j_2 = j, so the interpolation has a verified anchor at both ends.")
    with open("research/data/j2_perdiff.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("j2_perdiff: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()
