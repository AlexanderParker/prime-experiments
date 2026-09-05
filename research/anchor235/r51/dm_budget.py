"""dm_budget.py - the distortion method's budget, over Z and localised to an interval.

BBMST (Inventiones 2022) Theorem 3.1: with delta_i in [0, 1/2],
    eta := sum_i min{ M_i^(1), M_i^(2) / (4 delta_i (1 - delta_i)) }  <  1
implies the system does not cover Z, and P_0(R) >= (1-eta) exp(-2/(1-eta) sum_d nu(d)/d).
M_i^(1) = E_{i-1}[alpha_i], M_i^(2) = E_{i-1}[alpha_i^2], alpha_i(x) = the proportion of the
fibre over x (the class of x mod Q_{i-1}) that gear i strikes.

4 delta (1 - delta) <= 1 with equality at delta = 1/2, so the smallest each term can be made is
min{M^(1), M^(2)} = M^(2) (alpha in [0,1] gives alpha^2 <= alpha).  So the budget is

    eta = sum_i E_{i-1}[ alpha_i^2 ].

Over a full period, CRT makes alpha_i = 2/g_i on every fibre, so eta = sum 4/g^2.
On an interval the fibres are shorter than g_i from the fourth or fifth gear on; when a fibre
holds one point alpha is 0/1-valued, alpha^2 = alpha, and the term collapses to the first moment.

Outputs (results/, untracked):
  dm_budget.txt   the tables this run prints.
No numpy; pure python; seconds.
"""

import os
from math import log, exp

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


ALLP = primes_upto(2000)


def gears(q):
    return [p for p in ALLP if 5 <= p <= q]


def next_prime(q):
    for p in ALLP:
        if p > q:
            return p
    raise ValueError


def window(q):
    """The window of the machine {5..q}: columns k with q < 6k-1 and 6k+1 <= q'^2.
    Returns (k0, k1, L) with the columns k0..k1 inclusive."""
    qp = next_prime(q)
    k0 = (q + 1) // 6 + 1
    k1 = (qp * qp - 1) // 6
    return k0, k1, k1 - k0 + 1


# ---------------------------------------------------------------- part A: over Z

def part_A(qs):
    say("=" * 78)
    say("A.  THE BUDGET OVER Z (full period).  eta_Z = sum_{5<=g<=q} 4/g^2")
    say("    exact uncovered density = prod (1 - 2/g)  [CRT, the truth]")
    say("    BBMST density bound = (1-eta) exp(-2/(1-eta) * sum_prog nu/d), nu = 2 at delta=1/2")
    say("=" * 78)
    say(f"{'q':>5} {'gears':>6} {'sum 2/g':>9} {'eta_Z':>8} {'exact dens':>12} "
        f"{'BBMST bound':>13} {'ratio':>10}")
    for q in qs:
        G = gears(q)
        s1 = sum(2.0 / g for g in G)
        eta = sum(4.0 / (g * g) for g in G)
        dens = 1.0
        for g in G:
            dens *= (1.0 - 2.0 / g)
        # sum over PROGRESSIONS d of nu(d)/d: two progressions per gear, nu = 1/(1-1/2) = 2
        snu = sum(2 * 2.0 / g for g in G)
        bound = (1 - eta) * exp(-2.0 * snu / (1 - eta))
        say(f"{q:>5} {len(G):>6} {s1:>9.4f} {eta:>8.4f} {dens:>12.6e} "
            f"{bound:>13.6e} {dens / bound:>10.1f}x")
    say()
    say("eta_Z < 1 at every q (sum_{p>=5} 4/p^2 = %.4f, the limit), so the hypothesis of"
        % (4 * sum(1.0 / (p * p) for p in ALLP if p >= 5)))
    say("Theorem 3.1 HOLDS for the machine over Z and the theorem applies: the machine does not")
    say("cover Z.  Its density conclusion is weaker than the CRT product by the last column.")
    say()


# ------------------------------------------------- part B: exact budget on an interval

def alpha_moments(k0, k1, g, Qprev, weights=None):
    """Exact M^(1), M^(2) for gear g on the interval of columns [k0, k1], with fibres the
    residue classes mod Qprev.  weights: optional dict k -> weight (else uniform on the
    interval).  Returns (M1, M2, nfib, meanfibre)."""
    L = k1 - k0 + 1
    u = pow(6, -1, g)
    teeth = {u % g, (-u) % g}
    if Qprev > L:
        # every fibre holds at most one column: alpha is 0/1 valued
        tot = 0.0
        hit = 0.0
        for k in range(k0, k1 + 1):
            w = 1.0 if weights is None else weights.get(k, 0.0)
            if w == 0.0:
                continue
            tot += w
            if k % g in teeth:
                hit += w
        if tot == 0:
            return 0.0, 0.0, 0, 1.0
        return hit / tot, hit / tot, L, 1.0
    # group by residue mod Qprev
    fib_tot = {}
    fib_hit = {}
    for k in range(k0, k1 + 1):
        w = 1.0 if weights is None else weights.get(k, 0.0)
        r = k % Qprev
        fib_tot[r] = fib_tot.get(r, 0.0) + w
        if k % g in teeth:
            fib_hit[r] = fib_hit.get(r, 0.0) + w
    tot = sum(fib_tot.values())
    if tot == 0:
        return 0.0, 0.0, len(fib_tot), 0.0
    m1 = 0.0
    m2 = 0.0
    for r, t in fib_tot.items():
        if t == 0:
            continue
        a = fib_hit.get(r, 0.0) / t
        m1 += (t / tot) * a
        m2 += (t / tot) * a * a
    return m1, m2, len(fib_tot), L / Qprev


def part_B(qs):
    say("=" * 78)
    say("B.  THE BUDGET ON THE WINDOW INTERVAL, EXACT, REAL TEETH")
    say("    eta_I = sum_i E[alpha_i^2], fibres = classes mod Q_{i-1} inside the window.")
    say("    'uniform' = expectation under P_0 (uniform on the window).")
    say("    'survivors' = expectation under the uniform measure on the columns still open")
    say("                  after gears 1..i-1 (the method's P_{i-1} is between the two).")
    say("=" * 78)
    for q in qs:
        G = gears(q)
        k0, k1, L = window(q)
        say(f"\n  q = {q}:  window columns {k0}..{k1}, L = {L}, gears = {len(G)}")
        say(f"  {'g':>5} {'Q_<g':>14} {'fibre m':>9} {'M1_unif':>9} {'M2_unif':>9} "
            f"{'M2/M1':>7} {'M2_surv':>9}")
        Q = 1
        eta_u = 0.0
        eta_s = 0.0
        alive = {k: 1.0 for k in range(k0, k1 + 1)}
        for g in G:
            m1u, m2u, nf, mf = alpha_moments(k0, k1, g, Q)
            m1s, m2s, _, _ = alpha_moments(k0, k1, g, Q, weights=alive)
            eta_u += m2u
            eta_s += m2s
            if g <= 41 or Q <= L:
                say(f"  {g:>5} {min(Q, 10**13):>14} {mf:>9.2f} {m1u:>9.5f} {m2u:>9.5f} "
                    f"{(m2u / m1u if m1u else 0):>7.3f} {m2s:>9.5f}")
            u = pow(6, -1, g)
            teeth = {u % g, (-u) % g}
            for k in list(alive):
                if k % g in teeth:
                    del alive[k]
            Q *= g
        say(f"  ... remaining {len([g for g in G if g > 41])} gears: fibres are single columns,")
        say(f"      alpha in {{0,1}}, so M2 = M1 exactly (the collapse).")
        say(f"  eta_I (uniform)   = {eta_u:.4f}   [need < 1]")
        say(f"  eta_I (survivors) = {eta_s:.4f}   [need < 1]")
        say(f"  sum 2/g           = {sum(2.0 / g for g in G):.4f}  (the union bound the method"
            f" degrades to)")
        say(f"  eta_Z             = {sum(4.0 / (g * g) for g in G):.4f}  (the same budget over"
            f" the period)")
    say()


# ------------------------------------------- part C: the length the budget tolerates

def eta_model(q, L, worst=True):
    """Budget on an interval of length L.  Fibre size m = L/Q_{<g}.

    worst=True (the rigorous reading, used for L*): the largest E[alpha^2] the phases allow.
      alpha <= min(1, 2/m) pointwise, E[alpha] <= min(1, 2/g + 2/L), and E[alpha^2] >= 4/g^2
      by Cauchy-Schwarz, so the term is max(4/g^2, min(1,2/m) * min(1, 2/g + 2/L)).
    worst=False (the average reading): E[alpha^2] = 4/g^2 when m >= g (whole classes per
      fibre) and 2/(mg) + 2/g^2 when 1 <= m < g, capped at 2/g."""
    tot = 0.0
    Q = 1
    for g in gears(q):
        m = max(1.0, L / Q) if Q <= L else 1.0
        if worst:
            t = max(4.0 / (g * g), min(1.0, 2.0 / m) * min(1.0, 2.0 / g + 2.0 / L))
        else:
            t = min(2.0 / g, 4.0 / (g * g) if m >= g else 2.0 / (m * g) + 2.0 / (g * g))
        tot += t
        Q *= g
    return tot


def threshold(q):
    """Smallest L with eta_model(q, L) < 1, by bisection on log L."""
    lo, hi = 1.0, 1e300
    if eta_model(q, hi) >= 1.0:
        return None
    for _ in range(400):
        mid = exp((log(lo) + log(hi)) / 2) if lo > 0 else hi / 2
        if eta_model(q, mid) < 1.0:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0000001:
            break
    return hi


def part_C(qs, Fknown):
    say("=" * 78)
    say("C.  THE LENGTH THE LOCALISED BUDGET TOLERATES")
    say("    L*(q) = least L with the modelled eta_I(q, L) < 1.  Compare with the window")
    say("    W(q) = (q'^2-1)/6 and with the measured record F.")
    say("=" * 78)
    say(f"{'q':>5} {'gears':>6} {'W(q)':>9} {'F':>7} {'eta@W max':>10} {'eta@W avg':>10} "
        f"{'L*(q)':>13} {'L*/W':>12} {'cut gear':>9}")
    for q in qs:
        G = gears(q)
        k0, k1, W = window(q)
        e = eta_model(q, W)
        eavg = eta_model(q, W, worst=False)
        Ls = threshold(q)
        # the cut gear: the largest gear whose fibre still holds a whole class at L = L*
        cut = None
        Q = 1
        for g in G:
            if Ls is not None and L_ok(Ls, Q, g):
                cut = g
            Q *= g
        F = Fknown.get(q, None)
        say(f"{q:>5} {len(G):>6} {W:>9} {str(F):>7} {e:>10.4f} {eavg:>10.4f} "
            f"{(f'{Ls:.4e}' if Ls else 'none'):>13} "
            f"{(f'{Ls / W:.4e}' if Ls else '-'):>12} {str(cut):>9}")
    say()


def L_ok(L, Qprev, g):
    return L / Qprev >= g if Qprev <= L else False


# --------------------------------------------- part D: the naive Legendre localisation

def part_D(qs):
    say("=" * 78)
    say("D.  THE NAIVE LOCALISATION, both readings (the pre-registered E6 test)")
    say("    naive-1 ('+1 per gear'):  openings >= L*prod(1-2/g) - 2*(#gears)")
    say("    naive-2 (honest Legendre): openings >= L*prod(1-2/g) - sum_{d|P,d>1} 2^omega(d)")
    say("                               = L*prod(1-2/g) - (3^n - 1),  n = #gears")
    say("=" * 78)
    say(f"{'q':>5} {'n':>4} {'W(q)':>9} {'prod(1-2/g)':>12} {'L naive-1':>11} "
        f"{'L naive-2':>13}")
    for q in qs:
        G = gears(q)
        n = len(G)
        _, _, W = window(q)
        pr = 1.0
        for g in G:
            pr *= (1.0 - 2.0 / g)
        l1 = 2.0 * n / pr           # L needed for main term to beat 2n
        l2 = (3.0 ** n - 1) / pr    # L needed for main term to beat the Legendre error
        say(f"{q:>5} {n:>4} {W:>9} {pr:>12.6f} {l1:>11.1f} {l2:>13.4e}")
    say()
    say("naive-1 says an interval of a few thousand columns already contains an opening -")
    say("that is the illusion: prod(1-2/g) is the density over the PERIOD, and using it on an")
    say("interval is exactly what has to be justified.  naive-2 is the honest cost of that")
    say("justification with no truncation, and it is above the window by the last column.")
    say()


if __name__ == "__main__":
    QS = [59, 97, 199, 499]
    FKNOWN = {59: 161}
    part_A(QS)
    part_B(QS)
    part_C(QS, FKNOWN)
    part_D(QS)
    with open(os.path.join(OUT, "dm_budget.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nwritten:", os.path.join(OUT, "dm_budget.txt"))
