"""r53 sk_distortion - audit of the localised-distortion claim.

The claim on the record (research/proof/distortion_method.md R7, the_wall.md 5f):
"the localised budget proves the adversarial lemma A(K) < (p_{K+1}^2-1)/6 for every K <= 10,
and fails from K = 11."

The chain of steps behind it:
  S1  BBMST Theorem 3.1: if eta = sum_i min{M1_i, M2_i/(4 d_i(1-d_i))} < 1 then the system does
      not cover Z, where the moments are taken over the fibres of Z_{Q_i} = Z_{Q_{i-1}} x Z_{p_i}
      under the method's own reweighted measures P_{i-1}.
  S2  "Localise": replace Z_Q by an interval I of L columns with the uniform measure, and the
      fibres by (class mod Q_{i-1}) intersected with I.  Assert the same conclusion for I.
  S3  Bound the localised eta above by
          eta_max = sum_g max( 4/g^2 , min(1, 2/m_g) * min(1, 2/g + 2/L) ),   m_g = L/Q_{<g},
      and define L*(S) = the least L with eta_max < 1.
  S4  Take the worst K-set to be the K smallest gears, so A(K) <= L*max(K).
  S5  Read off L*max(K) < W(K) for K <= 10.

This script tests the ARITHMETIC of S3-S5 (does the table reproduce?), the unproved assertion
in S4 (are the K smallest gears really the worst?), and the validity of S3 itself against
exactly computed localised second moments.
"""
import os
from itertools import combinations
from math import exp, log

from sk_core import RESULTS, primes_upto, sep

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


P = primes_upto(2000)
GEARS = [p for p in P if p >= 5]
A_K = [2, 5, 7, 16, 22, 28, 37, 45, 68, 88, 101, 115]


# ------------------------------------------------------------------ the lane's models

def eta_max(gs, L):
    tot = 0.0
    Q = 1
    for g in gs:
        m = max(1.0, L / Q) if Q <= L else 1.0
        sup_a = min(1.0, 2.0 / m)
        e_a = min(1.0, 2.0 / g + 2.0 / L)
        tot += max(4.0 / (g * g), sup_a * e_a)
        Q *= g
    return tot


def eta_avg(gs, L):
    tot = 0.0
    Q = 1
    for g in gs:
        m = max(1.0, L / Q) if Q <= L else 1.0
        t = 4.0 / (g * g) if m >= g else 2.0 / (m * g) + 2.0 / (g * g)
        tot += min(2.0 / g, t)
        Q *= g
    return tot


def threshold(gs, f=eta_max):
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


# ------------------------------------------------- exact localised moments, real interval

def moments(g, phase, Q, L):
    """Exact (M1, M2) for gear g on an interval of L columns whose fibres are the classes
    mod Q, under the UNIFORM measure on the interval.  The gear's low tooth sits at column
    = phase (mod g).  Fibre of class c = {k in [0,L) : k = c (mod Q)}."""
    d = sep(g)
    t1, t2 = phase % g, (phase + d) % g
    m1 = 0.0
    m2 = 0.0
    for c in range(min(Q, L)):
        size = len(range(c, L, Q))
        if size == 0:
            continue
        hit = sum(1 for k in range(c, L, Q) if k % g in (t1, t2))
        a = hit / size
        w = size / L
        m1 += w * a
        m2 += w * a * a
    return m1, m2


def envelope_term(g, Q, L):
    m = max(1.0, L / Q) if Q <= L else 1.0
    return max(4.0 / (g * g), min(1.0, 2.0 / m) * min(1.0, 2.0 / g + 2.0 / L))


def exact_eta(gs, L, adversarial=True):
    """sum_g max_phase E[alpha_g^2] with the fibres of the gears already processed, uniform
    measure on the interval.  Also returns the per-gear comparison with the envelope."""
    rows = []
    tot = 0.0
    env = 0.0
    Q = 1
    for g in gs:
        best = None
        rng = range(g) if adversarial else [pow(6, -1, g)]
        for ph in rng:
            _m1, m2 = moments(g, ph, Q, L)
            if best is None or m2 > best:
                best = m2
        e = envelope_term(g, Q, L)
        rows.append((g, Q, L / Q if Q <= L else 1.0, best, e, best <= e + 1e-12))
        tot += best
        env += e
        Q *= g
    return tot, env, rows


def main():
    os.makedirs(RESULTS, exist_ok=True)

    say("=" * 96)
    say("A.  the recorded table reproduced (S3-S5 arithmetic)")
    say("=" * 96)
    say(f"{'K':>3} {'gears':>16} {'A(K)':>5} {'W':>6} {'L*max':>11} {'L*/W':>9} "
        f"{'L*avg':>10} {'A<W?':>5} {'L*<W?':>6}")
    for K in range(1, 13):
        gs = GEARS[:K]
        pk1 = P[P.index(gs[-1]) + 1]
        W = (pk1 * pk1 - 1) // 6
        Ls = threshold(gs, eta_max)
        La = threshold(gs, eta_avg)
        A = A_K[K - 1]
        gstr = ",".join(str(g) for g in gs)
        if len(gstr) > 16:
            gstr = gstr[:13] + "..."
        say(f"{K:>3} {gstr:>16} {A:>5} {W:>6} {Ls:>11.4e} {Ls/W:>9.3f} "
            f"{(f'{La:.3e}' if La else 'none'):>10} {'yes' if A < W else 'NO':>5} "
            f"{'yes' if Ls < W else 'NO':>6}")

    say()
    say("=" * 96)
    say("B.  S4: are the K smallest gears really the worst K-set for the budget?")
    say("     (the lane asserts it; here it is tested exhaustively over pools)")
    say("=" * 96)
    for K in range(1, 7):
        pool = GEARS[:14]
        best, arg = -1.0, None
        for S in combinations(pool, K):
            t = threshold(list(S), eta_max)
            if t is not None and t > best:
                best, arg = t, S
        base = threshold(GEARS[:K], eta_max)
        say(f"  K={K}: max L* over {len(pool)}-prime pool = {best:.4e} at {arg}; "
            f"K smallest gives {base:.4e}  {'OK' if arg == tuple(GEARS[:K]) else 'DIFFERENT'}")

    say()
    say("=" * 96)
    say("C.  S3: is eta_max an upper bound on the localised second-moment budget?")
    say("     exact E[alpha^2] over the fibres of the interval, uniform measure,")
    say("     maximised over the gear's phase, against the envelope term.")
    say("=" * 96)
    for (gs, L, label) in [([5, 7, 11, 13], 20, "K=4 gears, L=20"),
                           ([5, 7, 11, 13, 17], 60, "K=5 gears, L=W(5)=60"),
                           ([5, 7, 11, 13, 17, 19], 88, "K=6 gears, L=W(6)=88"),
                           ([5, 7, 11, 13, 17, 19, 23, 29, 31, 37], 280,
                            "K=10 gears, L=W(10)=280"),
                           ([5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], 308,
                            "K=11 gears, L=W(11)=308"),
                           ([5, 7, 11, 13], 610, "the real machine's head, L=610")]:
        tot, env, rows = exact_eta(gs, L)
        say(f"  {label}")
        say(f"    {'g':>4} {'Q_<g':>10} {'fibre m':>9} {'max E[a^2]':>11} "
            f"{'envelope':>10} {'env >= true?':>13}")
        for g, Q, m, tru, e, ok in rows:
            say(f"    {g:>4} {Q:>10} {m:>9.2f} {tru:>11.5f} {e:>10.5f} "
                f"{'yes' if ok else 'NO  <-- envelope FAILS':>13}")
        say(f"    exact sum = {tot:.4f}   envelope sum = {env:.4f}   "
            f"{'envelope valid' if env >= tot - 1e-12 else 'ENVELOPE BELOW TRUTH'}")
        say()

    say("=" * 96)
    say("D.  S2: is the localised inequality itself true?  Take a gear set that DOES cover an")
    say("     interval, and evaluate the localised budget on that very interval.")
    say("=" * 96)
    OPT = {1: [5], 2: [5, 7], 3: [5, 7, 11], 4: [5, 7, 11, 17], 5: [5, 7, 11, 23, 29],
           6: [5, 7, 11, 17, 23, 37], 7: [5, 7, 11, 13, 17, 19, 31],
           8: [5, 7, 11, 13, 19, 29, 31, 83], 9: [5, 7, 11, 13, 17, 23, 31, 37, 47],
           10: [5, 7, 11, 13, 17, 19, 23, 29, 37, 79]}
    A_K_D = {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28, 7: 37, 8: 45, 9: 68, 10: 88}
    say(f"{'K':>3} {'covering set':>36} {'L covered':>9} {'exact eta':>10} "
        f"{'eta_max':>9} {'localised claim':>17}")
    for K in range(2, 11):
        gs = sorted(OPT[K])
        L = A_K_D[K] - 1
        tot, env, _rows = exact_eta(gs, L)
        em = eta_max(gs, L)
        verdict = "FALSE (eta<1, covers)" if tot < 1.0 else "silent"
        say(f"{K:>3} {str(gs):>36} {L:>9} {tot:>10.4f} {em:>9.4f} {verdict:>17}")
    say()
    say("     exact eta = sum_g max_phase E[alpha_g^2] on the interval that the set covers,")
    say("     fibres = classes mod the product of the gears already processed, uniform measure.")
    say("     Wherever it is below 1 the localised form of the hypothesis holds on a set that")
    say("     demonstrably covers the interval, so 'eta < 1 implies no cover of I' is false.")
    say()

    say("=" * 96)
    say("E.  what eta_max actually is: capacity on the collapsed gears + 4/g^2 on the head")
    say("=" * 96)
    say(f"{'K':>3} {'L=W(K)':>7} {'head gears':>26} {'head sum':>9} {'tail sum':>9} "
        f"{'eta_max':>8} {'union bound':>11}")
    for K in range(1, 13):
        gs = GEARS[:K]
        pk1 = P[P.index(gs[-1]) + 1]
        L = (pk1 * pk1 - 1) // 6
        head, tail, hg = 0.0, 0.0, []
        Q = 1
        for g in gs:
            m = max(1.0, L / Q) if Q <= L else 1.0
            t1 = 4.0 / (g * g)
            t2 = min(1.0, 2.0 / m) * min(1.0, 2.0 / g + 2.0 / L)
            if t1 >= t2:
                head += t1
                hg.append(g)
            else:
                tail += t2
            Q *= g
        ub = sum(2 * -(-L // g) for g in gs) / L
        say(f"{K:>3} {L:>7} {str(hg):>26} {head:>9.4f} {tail:>9.4f} "
            f"{head+tail:>8.4f} {ub:>11.4f}")
    say()
    say("     The 'tail' terms are the collapsed gears, where the term is the gear's own")
    say("     capacity 2/g + 2/L; summed over ALL gears that is the union bound, a genuine")
    say("     theorem (if sum_g 2*ceil(L/g) < L there is no cover).  The union-bound column")
    say("     shows it is above 1 - vacuous - at every K >= 4.  So everything that makes")
    say("     eta_max < 1 comes from the HEAD, where the capacity 2/g is replaced by 4/g^2.")
    say("     That replacement is the step with no proof behind it.")

    with open(os.path.join(RESULTS, "sk_distortion.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
