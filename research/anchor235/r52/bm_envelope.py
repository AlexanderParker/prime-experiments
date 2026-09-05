"""bm_envelope.py - the PHASE-ADVERSARIAL block envelope, the per-gear blocks-vs-fibres
comparison, the gear-ordering child, and the adversarial gate against A(K).

A theorem needs an upper bound on alpha_i(B) valid for every phase vector.  Gear g strikes at
most 2*ceil(beta/g) columns of a block of length beta, and the adversary is free to put them all
on survivors, so

    alpha_i(B) <= min(1, 2*ceil(beta/g_i) / s_{i-1}(B)) ,   s = survivors of B before gear i.

Three readings of s:
  (LD)   s = beta * Pi_{<i}          perfect local density - the most generous hypothesis
  (CAP)  s = beta - sum_{j<i} 2*ceil(beta/g_j)    the only unconditional lower bound (counting)
  (CRT)  while beta >= Q_{<i} * g_i the block contains whole classes mod Q_{<i} g_i, so
         alpha_i = 2/g_i EXACTLY and the term is 4/g^2 with no hypothesis at all.

Fibre terms for comparison (r51 dm_budget.eta_model, the rigorous envelope):
    4/g^2 while the fibre holds a whole class; min(1,2/m)*min(1,2/g+2/L) otherwise; 2/g at
    full collapse.

Outputs (results/, untracked): bm_envelope.txt
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bm_exact import ALLP, gears, run_blocks, window

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


A_K = [2, 5, 7, 16, 22, 28, 37, 45, 68, 88, 101, 115]      # arc_multiset.md R1, exact
GEARS_ALL = [p for p in ALLP if p >= 5]


def pi_prefix(gs):
    """Pi_{<i} = prod_{j<i} (1 - 2/g_j), as a list of length len(gs)"""
    out = []
    p = 1.0
    for g in gs:
        out.append(p)
        p *= (1.0 - 2.0 / g)
    return out


def block_terms(gs, beta, mode):
    """per-gear terms of the a priori block budget.  mode in {'LD','CAP','CRT+LD'}"""
    P = pi_prefix(gs)
    out = []
    Q = 1
    cap = float(beta)
    for i, g in enumerate(gs):
        strikes = 2 * math.ceil(beta / g)
        if mode == "CRT+LD" and Q <= beta / g:      # beta >= Q_{<i} * g
            t = 4.0 / (g * g)
        else:
            if mode == "CAP":
                s = cap
            else:
                s = beta * P[i]
            t = 1.0 if s <= 0 else min(1.0, strikes / s) ** 2
        out.append(t)
        cap -= strikes
        Q *= g
    return out


def fibre_terms(gs, L):
    """r51's rigorous fibre envelope, per gear"""
    out = []
    Q = 1
    for g in gs:
        m = max(1.0, L / Q) if Q <= L else 1.0
        t = max(4.0 / (g * g), min(1.0, 2.0 / m) * min(1.0, 2.0 / g + 2.0 / L))
        out.append(t)
        Q *= g
    return out


def threshold_beta(gs, mode, lo=1.0, hi=1e300):
    """least beta with the a priori block budget < 1 (None if never)"""
    if sum(block_terms(gs, hi, mode)) >= 1.0:
        return None
    for _ in range(400):
        mid = math.exp((math.log(lo) + math.log(hi)) / 2)
        if sum(block_terms(gs, mid, mode)) < 1.0:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0000001:
            break
    return hi


def part_A():
    say("=" * 100)
    say("A.  THE A PRIORI (PHASE-ADVERSARIAL) BLOCK BUDGET - the deciding computation")
    say("    Granting the method EVERYTHING: beta -> infinity (the ceiling free) and perfect")
    say("    local density s = beta*Pi_{<i}.  Term = min(1, (2/g)/Pi_{<g})^2.")
    say("=" * 100)
    gs = gears(4000)
    P = pi_prefix(gs)
    say(f"  {'g':>5} {'Pi_<g':>9} {'(2/g)/Pi':>9} {'term':>9} {'cumulative':>11} "
        f"{'fibre 2/g':>10} {'4/g^2':>9}")
    tot = 0.0
    cross = None
    for i, g in enumerate(gs[:14]):
        a = min(1.0, (2.0 / g) / P[i])
        t = a * a
        tot += t
        if cross is None and tot >= 1.0:
            cross = g
        say(f"  {g:>5} {P[i]:>9.5f} {a:>9.5f} {t:>9.5f} {tot:>11.5f} "
            f"{2.0/g:>10.5f} {4.0/(g*g):>9.5f}")
    say(f"  the a priori block budget passes 1 at gear g = {cross} "
        f"(gear number {gs.index(cross)+1}), with NO dependence on beta, L or q.")
    say()
    say("  The same sum with the ceiling paid, at finite beta (mode LD), and the")
    say("  unconditional counting version (mode CAP):")
    say(f"  {'q':>5} {'beta':>10} {'LD budget':>10} {'CAP budget':>11} {'CRT+LD':>9}")
    for q in [59, 97, 199, 499, 997]:
        gq = gears(q)
        _, _, W = window(q)
        for beta in [W, 10 * W, 10 ** 6, 10 ** 12]:
            say(f"  {q:>5} {beta:>10} {sum(block_terms(gq, beta, 'LD')):>10.4f} "
                f"{sum(block_terms(gq, beta, 'CAP')):>11.4f} "
                f"{sum(block_terms(gq, beta, 'CRT+LD')):>9.4f}")
    say()
    say("  threshold beta*(q) = least beta with the a priori block budget < 1:")
    say(f"  {'q':>5} {'W(q)':>9} {'beta* LD':>10} {'beta* CAP':>11} {'beta* CRT+LD':>13}")
    for q in [59, 97, 199, 499, 997]:
        gq = gears(q)
        _, _, W = window(q)
        r = []
        for mode in ["LD", "CAP", "CRT+LD"]:
            t = threshold_beta(gq, mode)
            r.append("none" if t is None else f"{t:.3e}")
        say(f"  {q:>5} {W:>9} {r[0]:>10} {r[1]:>11} {r[2]:>13}")
    say()


def part_B():
    say("=" * 100)
    say("B.  PER-GEAR COMPARISON: blocks against fibres (E4)")
    say("    fibre term  = 4/g^2 while uncollapsed, 2/g after (r51 collapse lemma)")
    say("    block term  = min(1, (2/g)/Pi_<g)^2 (a priori, beta -> infinity)")
    say("=" * 100)
    gs = gears(1000)
    P = pi_prefix(gs)
    say(f"  {'g':>5} {'4/g^2':>9} {'fibre 2/g':>10} {'block adv':>10} {'block/fibre':>12} "
        f"{'winner':>8}")
    first_loss = None
    for i, g in enumerate(gs):
        if g > 60 and g not in (97, 199, 499, 997):
            continue
        b = min(1.0, (2.0 / g) / P[i]) ** 2
        f = 2.0 / g
        if first_loss is None and b > f:
            first_loss = g
        say(f"  {g:>5} {4.0/(g*g):>9.5f} {f:>10.5f} {b:>10.5f} {b/f:>12.3f} "
            f"{('block' if b < f else 'fibre'):>8}")
    say(f"  the block term first exceeds the collapsed fibre term at g = {first_loss}.")
    say()


def part_C():
    say("=" * 100)
    say("C.  MECHANISM: where the exact block budget sits between the two")
    say("    ideal   = sum 4/g^2                (alpha_i = 2/g exactly: the CRT value)")
    say("    exact   = eta_B at beta = L        (real teeth, real survivors)")
    say("    fibre   = r51's rigorous fibre envelope at L = W(q)")
    say("    adv     = sum min(1,(2/g)/Pi)^2    (a priori block, any beta)")
    say("=" * 100)
    say(f"  {'q':>5} {'L=W(q)':>8} {'ideal':>8} {'exact':>8} {'excess':>9} {'fibre':>8} "
        f"{'adv':>8}")
    for q in [59, 97, 199, 499, 997]:
        gq = gears(q)
        k0, _, L = window(q)
        ideal = sum(4.0 / (g * g) for g in gq)
        eta, loss, _, per = run_blocks(k0, L, gq, lambda i, g, Q, lQ, b=L: b)
        fib = sum(fibre_terms(gq, L))
        P = pi_prefix(gq)
        adv = sum(min(1.0, (2.0 / g) / P[i]) ** 2 for i, g in enumerate(gq))
        say(f"  {q:>5} {L:>8} {ideal:>8.5f} {eta:>8.5f} {eta-ideal:>9.5f} {fib:>8.4f} "
            f"{adv:>8.4f}")
    say()
    say("  The per-gear strike rate on the survivors at beta = L, as a multiple of the fair")
    say("  rate 2/g (rho_i = alpha_i g_i / 2).  eta_B = sum (4/g^2) rho_i^2, so the budget")
    say("  is below 1 as long as the rho_i stay under about 1/sqrt(0.365) = 1.655 in the")
    say("  weighted L2 sense.")
    for q in [199, 997]:
        gq = gears(q)
        k0, _, L = window(q)
        eta, loss, _, per = run_blocks(k0, L, gq, lambda i, g, Q, lQ, b=L: b)
        rho = [(g, m1 * g / 2.0) for (g, b, m1, m2, nb) in per]
        rho_sorted = sorted(rho, key=lambda t: -t[1])
        say(f"\n  q = {q}: rho over {len(rho)} gears: min {min(r for _, r in rho):.4f}, "
            f"max {max(r for _, r in rho):.4f}, mean {np.mean([r for _, r in rho]):.4f}")
        say("    largest 8: " + ", ".join(f"g={g}:{r:.3f}" for g, r in rho_sorted[:8]))
        say("    smallest 4: " + ", ".join(f"g={g}:{r:.3f}" for g, r in rho_sorted[-4:]))
        say(f"    head gears 5..23 rho: " +
            ", ".join(f"{g}:{r:.3f}" for g, r in rho if g <= 23))
    say()
    say("  The same, block by block, at a range of beta (q = 199): how much of the excess")
    say("  eta_B - sum 4/g^2 is dead blocks and how much is live-block variance.")
    q = 199
    gq = gears(q)
    k0, _, L = window(q)
    ideal = sum(4.0 / (g * g) for g in gq)
    say(f"  {'beta':>8} {'eta_B':>9} {'excess':>9} {'true loss':>10} "
        f"{'excess-loss':>12}")
    for beta in [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, L]:
        eta, loss, _, _ = run_blocks(k0, L, gq, lambda i, g, Q, lQ, b=beta: b)
        say(f"  {beta:>8} {eta:>9.5f} {eta-ideal:>9.5f} {loss:>10.5f} {eta-ideal-loss:>12.5f}")
    say()


def part_D():
    say("=" * 100)
    say("D.  THE GEAR-ORDERING CHILD")
    say("    fibres: the ordering moves the collapse point, hence L* (a primorial).")
    say("    blocks: there is no collapse point to move.")
    say("=" * 100)

    def fibre_threshold(gs, L_lo=1.0):
        hi = 1e300
        if sum(fibre_terms(gs, hi)) >= 1.0:
            return None
        lo = L_lo
        for _ in range(400):
            mid = math.exp((math.log(lo) + math.log(hi)) / 2)
            if sum(fibre_terms(gs, mid)) < 1.0:
                hi = mid
            else:
                lo = mid
            if hi / lo < 1.0000001:
                break
        return hi

    say(f"  {'q':>5} {'fibre L* inc':>13} {'fibre L* dec':>13} {'ratio':>10} "
        f"{'block eta inc':>14} {'block eta dec':>14}")
    for q in [59, 97, 199, 499, 997]:
        gq = gears(q)
        k0, _, L = window(q)
        li = fibre_threshold(gq)
        ld = fibre_threshold(list(reversed(gq)))
        ei, _, _, _ = run_blocks(k0, L, gq, lambda i, g, Q, lQ, b=L: b)
        ed, _, _, _ = run_blocks(k0, L, list(reversed(gq)), lambda i, g, Q, lQ, b=L: b)
        fi = f"{li:.4e}" if li is not None else "none"
        fd = f"{ld:.4e}" if ld is not None else "none"
        rt = f"{li/ld:.4e}" if (li is not None and ld is not None) else "-"
        say(f"  {q:>5} {fi:>13} {fd:>13} {rt:>10} {ei:>14.5f} {ed:>14.5f}")
    say()
    say("  block budget under three orderings at several beta (q = 199):")
    q = 199
    gq = gears(q)
    k0, _, L = window(q)
    import random
    rnd = list(gq)
    random.Random(7).shuffle(rnd)
    say(f"  {'beta':>8} {'increasing':>11} {'decreasing':>11} {'random':>11}")
    for beta in [4, 16, 64, 256, 1024, L]:
        a, _, _, _ = run_blocks(k0, L, gq, lambda i, g, Q, lQ, b=beta: b)
        b, _, _, _ = run_blocks(k0, L, list(reversed(gq)), lambda i, g, Q, lQ, b=beta: b)
        c, _, _, _ = run_blocks(k0, L, rnd, lambda i, g, Q, lQ, b=beta: b)
        say(f"  {beta:>8} {a:>11.5f} {b:>11.5f} {c:>11.5f}")
    say()


def part_E():
    say("=" * 100)
    say("E.  THE ADVERSARIAL GATE AGAINST A(K)")
    say("    L*_B(K) = shortest interval the a priori block budget can address, on the K")
    say("    smallest gears (the worst K-set).  Validity needs L*_B(K) >= A(K).")
    say("=" * 100)
    say(f"  {'K':>3} {'gears':>16} {'A(K)':>5} {'W':>6} {'L*_B LD':>10} {'L*_B CAP':>10} "
        f"{'L*_B CRT+LD':>12} {'gate':>6}")
    for K in range(1, 13):
        gs = GEARS_ALL[:K]
        pk1 = ALLP[ALLP.index(gs[-1]) + 1]
        W = (pk1 * pk1 - 1) // 6
        A = A_K[K - 1]
        vals = []
        for mode in ["LD", "CAP", "CRT+LD"]:
            t = threshold_beta(gs, mode)
            vals.append(t)
        gate = "ok"
        for t in vals:
            if t is not None and t < A:
                gate = "FAIL"
        gstr = ",".join(str(g) for g in gs)
        if len(gstr) > 16:
            gstr = gstr[:13] + "..."
        say(f"  {K:>3} {gstr:>16} {A:>5} {W:>6} "
            + " ".join(f"{('none' if t is None else f'{t:.3e}'):>10}" for t in vals[:2])
            + f" {('none' if vals[2] is None else f'{vals[2]:.3e}'):>12} {gate:>6}")
    say()
    say("  Same, for the machine's own record: the block bound must not claim an interval")
    say("  shorter than the certified F(M) is uncoverable.")
    FLAD = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
    say(f"  {'m':>4} {'F(M)':>5} {'L*_B LD':>10} {'L*_B CAP':>10} {'L*_B CRT+LD':>12} {'gate':>6}")
    for q, F in FLAD.items():
        gs = gears(q)
        vals = [threshold_beta(gs, m) for m in ["LD", "CAP", "CRT+LD"]]
        gate = "ok"
        for t in vals:
            if t is not None and t <= F:
                gate = "FAIL"
        say(f"  {q:>4} {F:>5} "
            + " ".join(f"{('none' if t is None else f'{t:.3e}'):>10}" for t in vals[:2])
            + f" {('none' if vals[2] is None else f'{vals[2]:.3e}'):>12} {gate:>6}")
    say()


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
    part_D()
    part_E()
    with open(os.path.join(OUT, "bm_envelope.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nwritten:", os.path.join(OUT, "bm_envelope.txt"))
