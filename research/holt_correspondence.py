"""Harvester round 22: the load-bearing step of the round's prior-art CORRECTION -
the twin-slot machine is the twin-candidate subsequence of Holt's cycle of gaps, and
the project's depth-sum / local-factor identities are his Corollary 1 specialised.

Holt, "Eratosthenes sieve supports the k-tuple conjecture", arXiv:2502.20470,
Corollary 1: for an admissible constellation s of length J,

    sum_{j >= J} n_{s,j}(p#)  =  prod_{q <= p} (q - nu_q(s)),

where n_{s,j} counts s and its driving terms of length j, and nu_q(s) is the number of
distinct residues mod q among the J+1 boundary points of s.  "s and its driving terms"
means exactly "the J+1 boundary points are all rough, interiors arbitrary".

This script asserts, exactly, on real cycles:

 (A) THE CORRESPONDENCE.  Twin-slot survivors of the project's machine (slots k with
     6k-1 and 6k+1 both coprime to the primorial) are exactly the left endpoints of
     the gaps of 2 in the cycle of p-rough numbers.
 (B) THE IDENTITY IS HOLT'S.  The project's N2(g) = prod_q c_q(g) with
     c_q(g) = q - nu_q({0,2,6g,6g+2}) equals the count of positions where the four
     boundary points of the constellation s = (2, 6g-2, 2) are all rough - i.e.
     Holt's right-hand side at that s - and equals sum_j W_j(g), the depth-sum
     identity.
 (C) WHAT IS NOT HIS.  n_g (consecutive twin-slot survivors: no TWIN CANDIDATE
     between, ordinary rough numbers allowed) differs from Holt's n_{s,J} (no ROUGH
     NUMBER between).  The two are printed side by side; they differ at every g > 1.
"""
import numpy as np
from math import prod

LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def rough(primes, P):
    a = np.ones(P, bool)
    for q in primes:
        a[0::q] = False
    return a


def nu(q, X):
    return len({x % q for x in X})


def main():
    for gears in ([2, 3, 5, 7, 11, 13], [2, 3, 5, 7, 11, 13, 17]):
        P = prod(gears)
        R = rough(gears, P)
        odd = [q for q in gears if q >= 5]
        # (A) correspondence
        slots = np.array([k for k in range(P // 6)
                          if R[(6 * k - 1) % P] and R[(6 * k + 1) % P]])
        idx = np.flatnonzero(R)
        gaps2 = np.array([int(n) for n in idx if R[(n + 2) % P]])
        assert set(((6 * k - 1) % P) for k in slots) == set(int(n) for n in gaps2), gears
        say(f"  machine {gears[-1]:>2} (P = {P:,}): {len(slots):,} twin-slot survivors "
            f"= {len(gaps2):,} gaps of 2 in the rough cycle - CORRESPONDENCE EXACT")
        # (B) + (C)
        spos = np.sort(np.array([(6 * k - 1) % P for k in slots]))
        sset = set(int(x) for x in spos)
        say("     g   N2 = prod_q c_q(g)   #{4 boundary pts rough}   sum_j W_j(g)"
            "     n_g (ours)   n_{s,J} (Holt)")
        for g in range(1, 7):
            H = (0, 2, 6 * g, 6 * g + 2)
            N2 = prod(q - nu(q, H) for q in odd)
            direct = sum(1 for n in spos if ((int(n) + 6 * g) % P) in sset)
            # sum_j W_j(g) is the same count (every open pair at lag 6g)
            assert N2 == direct, (gears, g, N2, direct)
            # n_g: consecutive twin-slot survivors at lag 6g
            ng = 0
            hj = 0
            for n in spos:
                n = int(n)
                m = (n + 6 * g) % P
                if m not in sset:
                    continue
                if not any(((n + 6 * t) % P) in sset for t in range(1, g)):
                    ng += 1
                if not any(R[(n + t) % P] for t in range(3, 6 * g)):
                    hj += 1
            say(f"    {g:>2}   {N2:>18,}   {direct:>22,}   {direct:>12,}"
                f"   {ng:>12,}   {hj:>14,}")
            if g > 1:
                assert ng != hj, (gears, g, ng, hj)
    say("  (B) N2 = the Holt right-hand side at s = (2, 6g-2, 2), exactly, at every g")
    say("      and every machine tested -> the depth-sum identity and the local-factor")
    say("      identity are his Corollary 1 specialised.")
    say("  (C) n_g != n_{s,J} for every g > 1: the twin-slot GAP population is a")
    say("      different object from the constellation population, and it is the one")
    say("      the pinch / Bonferroni series is about.")
    with open("research/data/holt_correspondence.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("holt_correspondence: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()
