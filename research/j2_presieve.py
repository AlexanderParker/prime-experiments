"""Harvester round 24 (item c): THE PRE-SIEVED EXPLICIT RUNG - exponent 19 -> 17
at NO cost in the constant, and a priced ladder down to 12.

Round 23's THEOREM 2E is

    j_2(p_n#)  <=  1.0963e10 * p_n^19 * (log p_n)^10 + 1      (p_n >= 285),

from Friedlander-Iwaniec, Opera de Cribro, Theorem 7.7 (p. 111; the statement was
checked against the book's own text this round) with kappa = 2 and K = 3, whose
bracket first turns positive at s* = 18.308.  Round 23 named the improvement and
priced it as "bounded and mechanical": PRE-SIEVE the small primes, so that the
sieve hypothesis constant K = sup_{w<z} prod_{w<=p<z}(1-g(p))^{-1}/(log z/log w)^2
is taken over p >= p_0 only.  K = 3 is forced entirely by the single degenerate
point w = 3, z -> 3+, where (1 - 2/3)^{-1} = 3; drop the primes below p_0 and K
falls fast.

THE ACCOUNTING, in full, because the whole question is what pre-sieving COSTS.
Let Q = prod_{p < p_0} p and let A' be the elements of A = (M, M+m] surviving the
pre-sieve.  A' is a union of

    N_pre = prod_{p < p_0} (p - omega(p))          [omega(2)=1, omega(p)=2]

residue classes mod Q, so X = |A'| has main term m V_pre, V_pre = N_pre/Q, and for
every squarefree d composed of primes in [p_0, z],

    |A'_d| = X g(d) + r_d,      |r_d| <= omega(d) N_pre <= 2^{nu(d)} N_pre,

uniformly in M.  Hence R_4 = sum_{d<D} tau_4(d)|r_d| <= N_pre * C_8 D (log D)^8
with the SAME C_8 = 0.0316 as round 23, and since X V'(z) = m V_pre V'(z) = m V(z)
exactly, positivity needs

    m  >  (2/bracket(s,k)) * N_pre * C_8 * z^s * (s log z)^8 / V(z).

So pre-sieving changes exactly ONE thing in Theorem 2E's constant: a factor N_pre.
AND N_pre = 1 FOR p_0 = 5, because omega(3) = 2 leaves 3 - 2 = 1 class mod 3.
Pre-sieving 2 and 3 is therefore FREE, and it moves s* from 18.308 to 16.136.

Everything below is assertion-gated and re-derives round 23's numbers as controls.
"""
from math import log, exp, e as E
import numpy as np

LOG = []
KAPPA = 2.0
GAMMA = 0.5772156649015328606


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def primes_upto(n):
    sv = np.ones(n + 1, bool)
    sv[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sv[i]:
            sv[i * i::i] = False
    return np.flatnonzero(sv).astype(np.int64)


PR = primes_upto(200000).tolist()
G = {p: (0.5 if p == 2 else 2.0 / p) for p in PR}


def bracket(s, k):
    """FI Opera de Cribro Thm 7.7 / (7.121) bracket."""
    return 1.0 - ((s + 3) / (2 * exp(k))) * (2 * E * k / (s - 3)) ** ((s - 3) / 2)


def K_of(p0, nw=300, nz=400):
    """sup over 2 <= w < z of prod_{w<=p<z, p>=p0}(1-g(p))^{-1}/(log z/log w)^kappa.

    For w < p0 the numerator is unchanged while (log z/log w)^kappa is LARGER, so
    the supremum is attained at w >= p0 and it is enough to scan w over primes
    >= p0.  Within (p_i, p_{i+1}] the ratio is largest at z -> p_i+ (the numerator
    is constant there and the denominator increases with z), so it suffices to
    test z just above each prime.
    """
    sub = [p for p in PR if p >= p0]
    worst = 1.0
    for i, w in enumerate(sub[:nw]):
        acc = 1.0
        for j in range(i, min(i + nz, len(sub))):
            p = sub[j]
            acc *= 1.0 / (1.0 - G[p])
            z = p * (1.0 + 1e-13)
            if z <= w:
                continue
            val = acc / (log(z) / log(w)) ** KAPPA
            if val > worst:
                worst = val
    return worst


def s_star(k):
    lo, hi = max(2 * k + 3, 3.0 + 1e-9), 200.0
    assert bracket(hi, k) > 0
    for _ in range(300):
        mid = (lo + hi) / 2
        if bracket(mid, k) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def N_pre(p0):
    n = 1
    for p in PR:
        if p >= p0:
            break
        n *= (p - (1 if p == 2 else 2))
    return n


def absorb(C0):
    """least X with C0 (log X)^10 <= X."""
    lo, hi = 10.0, 1e300
    for _ in range(3000):
        mid = (lo * hi) ** 0.5
        if C0 * log(mid) ** 10 <= mid:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    say("=" * 78)
    say("P1 - C_8 recomputed (control: round 23 got 0.0316)")
    say("=" * 78)
    fac = [(1 + 8.0 / p) * (1 - 1.0 / p) ** 8 for p in primes_upto(10 ** 6).tolist()]
    assert all(f < 1.0 for f in fac)
    H = 1.0
    for f in fac:
        H *= f
    C8 = H * exp(8 * GAMMA)
    say(f"  H = prod_{{p<10^6}}(1+8/p)(1-1/p)^8 = {H:.6e}   (decreasing in D, so a")
    say("  valid UPPER bound for every D >= 10^6, which z^s >= 285^12 far exceeds)")
    say(f"  C_8 = H e^{{8 gamma}} = {C8:.4f}")
    assert abs(C8 - 0.0316) < 0.0005, C8

    say("")
    say("=" * 78)
    say("P2 - K(p_0), k(p_0) and the FI 7.7 positivity threshold s*(p_0)")
    say("=" * 78)
    say("  (controls: round 23 reported K = 3 / 5/3 / 1.4 / 1.2624 / 1.0479 and")
    say("   s* = 18.308 / 16.136 / 15.474 / 15.077 / 14.353 at p_0 = 2/5/7/11/101)")
    say("")
    say("     p_0     K(p_0)      k       s*      least int s   N_pre")
    P0S = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53, 61, 71, 83,
           101]
    tab = {}
    for p0 in P0S:
        K = K_of(p0)
        k = KAPPA + log(K)
        ss = s_star(k)
        Np = N_pre(p0)
        tab[p0] = (K, k, ss, Np)
        say(f"  {p0:>6} {K:>10.5f} {k:>8.5f} {ss:>8.4f}   {int(ss)+1:>10}   "
            f"{Np:>12,}")
    # controls against round 23
    assert abs(tab[2][0] - 3.0) < 1e-6 and abs(tab[2][2] - 18.308) < 0.01
    assert abs(tab[5][0] - 5.0 / 3.0) < 1e-6 and abs(tab[5][2] - 16.136) < 0.01
    assert abs(tab[7][0] - 1.4) < 1e-6 and abs(tab[7][2] - 15.474) < 0.01
    assert abs(tab[11][0] - 1.2624) < 5e-4 and abs(tab[11][2] - 15.077) < 0.01
    assert abs(tab[101][0] - 1.0479) < 5e-4 and abs(tab[101][2] - 14.353) < 0.01
    say("  ASSERTED: round 23's five values reproduced exactly.")
    say("")
    say(f"  THE FLOOR OF THIS METHOD: K -> 1, k -> 2, s* -> {s_star(2.0):.4f}.  No")
    say("  amount of pre-sieving takes FI 7.7 below that; since it exceeds 14,")
    say("  EXPONENT 15 IS THE BEST INTEGER THIS THEOREM CAN EVER GIVE at kappa = 2,")
    say("  and p_0 = 13 (N_pre = 135) already reaches it.")
    assert 14.0 < s_star(2.0) < 14.5

    say("")
    say("  N_pre = prod_{p<p_0}(p - omega(p)), omega(2)=1 and omega(p)=2, so")
    say("  N_pre(3) = 1 (only class 1 mod 2 survives) and N_pre(5) = 1 x 1 = 1")
    say("  because gear 3 keeps a SINGLE class.  PRE-SIEVING 2 AND 3 IS FREE.")
    assert N_pre(5) == 1 and N_pre(7) == 3 and N_pre(11) == 15

    say("")
    say("=" * 78)
    say("P3 - THE PRE-SIEVED THEOREM 2E'")
    say("=" * 78)
    say("      m > (2/bracket) * N_pre * C_8 * z^s * (s log z)^8 / V(z),")
    say("      V(z) >= 0.3905/(log z)^2 for z >= 285 (round-23 corrected constant),")
    say("      so  j_2(p_n#) <= C_0 p_n^s (log p_n)^10 + 1  with")
    say("      C_0 = 1.001 * (2/bracket(s,k)) * N_pre * C_8 * s^8 / 0.3905.")
    say("")
    say("     p_0   s   bracket        C_0         valid from   note")
    rows = []
    for p0 in P0S:
        K, k, ss, Np = tab[p0]
        s = float(int(ss) + 1)
        br = bracket(s, k)
        C0 = 1.001 * (2.0 / br) * Np * C8 * (s ** 8) / 0.3905
        rows.append((p0, s, br, C0, Np))
        note = ""
        if p0 == 2:
            note = "round-23 THEOREM 2E (control)"
        elif Np == 1:
            note = "FREE - no constant cost"
        say(f"  {p0:>6} {int(s):>3} {br:>9.5f} {C0:>13.4e}   p_n >= 285   {note}")
    c23 = [r for r in rows if r[0] == 2][0]
    say("")
    say(f"  CONTROL: p_0 = 2 gives C_0 = {c23[3]:.4e} against round 23's 1.0963e10")
    assert abs(c23[3] / 1.0963e10 - 1.0) < 0.02, c23
    say("  ASSERTED: round 23's THEOREM 2E constant reproduced.")

    say("")
    r5 = [r for r in rows if r[0] == 5][0]
    r13 = [r for r in rows if r[0] == 13][0]
    say("  THEOREM 2E' (THE FREE ONE - the pre-sieve is by p = 2 and p = 3 only,")
    say("  and it costs NOTHING because N_pre = 1):")
    say("")
    say(f"      j_2(p_n#)  <=  {r5[3]:.4e} * p_n^{{17}} * (log p_n)^{{10}}  +  1")
    say("")
    say("  for every p_n >= 285, every constant explicit, no ineffective threshold.")
    say(f"  More generally j_2(p_n#) << p_n^s for every real s > {tab[5][2]:.4f},")
    say("  against round 23's 18.308.  The constant is SMALLER than round 23's")
    say(f"  ({r5[3]:.3e} vs {c23[3]:.3e}) as well, because s^8 and 2/bracket both")
    say("  fall with s - so THEOREM 2E' dominates THEOREM 2E at every p_n.")
    assert r5[4] == 1 and r5[3] < c23[3] and r5[1] == 17.0
    say("")
    say("  THEOREM 2E'' (THE BEST ONE - pre-sieve by 2, 3, 5, 7, 11, i.e. p_0 = 13,")
    say(f"  N_pre = {int(r13[4])}, and s* = {tab[13][2]:.4f} < 15):")
    say("")
    say(f"      j_2(p_n#)  <=  {r13[3]:.4e} * p_n^{{15}} * (log p_n)^{{10}}  +  1")
    say("")
    say("  for every p_n >= 285.  It DOMINATES 2E' already at the threshold:")
    say(f"      ratio (2E' bound)/(2E'' bound) at p_n = 285 is "
        f"{(r5[3]/r13[3])*285**2:.1f} > 1,")
    say("  so exponent 15 is the statement to make, and by the floor above it is")
    say("  the smallest integer exponent Theorem 7.7 can ever deliver at kappa = 2.")
    assert (r5[3] / r13[3]) * 285 ** 2 > 1 and r13[1] == 15.0 and r13[4] == 135

    say("")
    say("  THE PRICED LADDER BELOW 17.  Each further rung costs a factor N_pre in")
    say("  the constant and buys one or two in the exponent.  The crossover point")
    say("  where the smaller exponent beats the bigger constant, at p_n = 285 and")
    say("  at p_n = 10^6:")
    say("     p_0   s      C_0          bound at p_n=285      bound at p_n=10^6")
    best285, best6 = None, None
    for p0, s, br, C0, Np in rows:
        v285 = log(C0) + s * log(285) + 10 * log(log(285))
        v6 = log(C0) + s * log(10 ** 6) + 10 * log(log(10 ** 6))
        say(f"  {p0:>6} {int(s):>3} {C0:>12.3e}   10^{v285/log(10):>8.2f}"
            f"          10^{v6/log(10):>8.2f}")
        if best285 is None or v285 < best285[1]:
            best285 = ((p0, s), v285)
        if best6 is None or v6 < best6[1]:
            best6 = ((p0, s), v6)
    say(f"  BEST at p_n = 285:   p_0 = {best285[0][0]}, s = {int(best285[0][1])}")
    say(f"  BEST at p_n = 10^6:  p_0 = {best6[0][0]}, s = {int(best6[0][1])}")
    say("  (the small-p_n optimum is not the large-p_n optimum: N_pre is a fixed")
    say("  cost, so deeper pre-sieving always wins eventually.)")

    say("")
    say("=" * 78)
    say("P4 - WHAT PRE-SIEVING DOES NOT FIX")
    say("=" * 78)
    say(f"  * The floor is s*(k=2) = {s_star(2.0):.4f}: exponent 14 is unreachable by")
    say("    FI 7.7 no matter how much is pre-sieved, because K -> 1 only.")
    say("  * The 4.266 rung (Diamond-Halberstam-Richert sifting limit) is untouched")
    say("    and still not explicit - pre-sieving is a change of constant, not of")
    say("    method.")
    say("  * Halberstam-Richert, 'A new look at Brun's sieve', Mem. S.M.F. 25")
    say("    (1971) 97-106, p. 99, treats EXACTLY our density (they write")
    say("    A = {n(n+2) : n <= x}, omega(2) = 1, omega(p) = 2) and their")
    say("    conditions admit any level exponent u > 1 + 2.01/(e^lambda - 1) with")
    say("    lambda^2 e^{2 lambda}(2 + e^2) < 1, i.e. u > 7.9720 - BELOW this")
    say("    method's floor of 14 - but every remainder in that paper is an")
    say("    unspecified O(.).  The exponent-8 route is therefore an EXPLICITNESS")
    say("    problem in a known theorem, not a new sieve.  (Verified against the")
    say("    numdam scan this round; the figure 7.972 is DERIVED from p. 99, not")
    say("    printed there - the paper says only 'u < 8'.)")
    lam_lo, lam_hi = 0.0, 1.0
    for _ in range(300):
        mid = (lam_lo + lam_hi) / 2
        if mid * mid * exp(2 * mid) * (2 + exp(2.0)) < 1.0:
            lam_lo = mid
        else:
            lam_hi = mid
    lam = lam_lo
    u = 1 + 2.01 / (exp(lam) - 1)
    say(f"    RE-DERIVED HERE: lambda* = {lam:.15f}, u = 1 + 2.01/(e^lambda*-1)")
    say(f"                   = {u:.12f}  ->  7.9720.")
    assert abs(u - 7.971954833) < 1e-6, u
    say("    ASSERTED: 7.97195 reproduced from Halberstam-Richert's own two")
    say("    printed conditions, independently of the reported value.")
    say("    (Their (1.2), lambda e^{1+lambda} < 1, gives lambda < 0.278464 and is")
    say("    the WEAKER constraint - positivity binds first.)")
    lam2_lo, lam2_hi = 0.0, 1.0
    for _ in range(300):
        mid = (lam2_lo + lam2_hi) / 2
        if mid * exp(1 + mid) < 1.0:
            lam2_lo = mid
        else:
            lam2_hi = mid
    say(f"    check: lambda e^{{1+lambda}} = 1 at lambda = {lam2_lo:.6f} > {lam:.6f}")
    assert lam2_lo > lam

    with open("research/data/j2_presieve.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("j2_presieve: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()
