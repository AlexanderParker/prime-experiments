"""bm_mech.py - the mechanism of the block second moment, term by term, at q = 199 and 997
(and the other machines for the table).

Three costs separate the ideal block term 4/g^2 from what an argument can actually use:

  ceiling cost      (2 ceil(beta/g)/beta)^2 - (2/g)^2      blocks are not unions of classes,
                                                           so a gear's strike count is 2beta/g
                                                           up to +-2
  conditioning cost ((2/g)/Pi_<g)^2 - (2/g)^2              the strikes need not land on the
                                                           survivors in proportion; the adversary
                                                           puts them all there
  measured excess   E[alpha^2] - (2/g)^2                   what the real machine actually costs

and the provable range: while Q_{<g} * g divides beta the block contains every class mod
Q_{<g} g_g equally often and alpha_g = 2/g EXACTLY, with no hypothesis.  Beyond that gear the
equality is a measured fact, not a theorem: this script reports the last gear where it is proved
and the last gear where it is measured true to 1%.

Outputs (results/, untracked): bm_mech.txt
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bm_exact import gears, run_blocks, window
from bm_envelope import pi_prefix

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def main():
    say("=" * 100)
    say("THE THREE COSTS, SUMMED OVER GEARS, AT beta = L = W(q)")
    say("  ideal       sum 4/g^2")
    say("  +ceiling    sum (2 ceil(L/g)/L)^2         (blocks are not unions of classes)")
    say("  +condition  sum min(1,(2/g)/Pi_<g)^2      (strikes need not land on survivors)")
    say("  measured    sum E[alpha^2] (real teeth)")
    say("  fibre       r51's rigorous fibre envelope at the same L")
    say("=" * 100)
    say(f"  {'q':>5} {'L':>8} {'ideal':>8} {'ceiling':>9} {'condition':>10} {'measured':>9} "
        f"{'fibre':>8}")
    for q in [59, 97, 199, 499, 997]:
        gs = gears(q)
        k0, _, L = window(q)
        P = pi_prefix(gs)
        ideal = sum(4.0 / (g * g) for g in gs)
        ceil_c = sum(min(1.0, 2.0 * math.ceil(L / g) / L) ** 2 for g in gs)
        cond = sum(min(1.0, (2.0 / g) / P[i]) ** 2 for i, g in enumerate(gs))
        eta, _, _, per = run_blocks(k0, L, gs, lambda i, g, Q, lQ, b=L: b)
        from bm_envelope import fibre_terms
        fib = sum(fibre_terms(gs, L))
        say(f"  {q:>5} {L:>8} {ideal:>8.5f} {ceil_c:>9.5f} {cond:>10.5f} {eta:>9.5f} "
            f"{fib:>8.4f}")
    say()
    say("  Reading: the CEILING is almost free (it moves the ideal by the fifth decimal at")
    say("  every machine).  The CONDITIONING is what busts the budget, and it busts it at the")
    say("  sixth gear, before any tail exists.")
    say()

    say("=" * 100)
    say("PROVED versus MEASURED: how far up the gears alpha_g = 2/g survives on the window")
    say("  proved: Q_{<g} * g divides beta = L  (the block then holds every class mod")
    say("          Q_{<g} g equally often, so alpha_g = 2/g with no hypothesis)")
    say("  proved-loose: Q_{<g} * g <= L        (the same up to an edge error 1 - L/(Q g)*floor)")
    say("  measured: rho_g = alpha_g g / 2 within 1% of 1 on the real window")
    say("=" * 100)
    say(f"  {'q':>5} {'L':>8} {'last proved':>12} {'last loose':>11} {'1st dev 1/5/20%':>16} "
        f"{'#gears':>7} {'max rho':>8} {'min rho':>8}")
    for q in [59, 97, 199, 499, 997]:
        gs = gears(q)
        k0, _, L = window(q)
        Q = 1
        last_div, last_loose = None, None
        for g in gs:
            if L % (Q * g) == 0:
                last_div = g
            if Q * g <= L:
                last_loose = g
            Q *= g
        eta, _, _, per = run_blocks(k0, L, gs, lambda i, g, Q, lQ, b=L: b)
        rho = [(g, m1 * g / 2.0) for (g, b, m1, m2, nb) in per]
        def first_dev(tol):
            for g, r in rho:
                if abs(r - 1.0) > tol:
                    return g
            return None
        last_ok = f"{first_dev(0.01)}/{first_dev(0.05)}/{first_dev(0.20)}"
        say(f"  {q:>5} {L:>8} {str(last_div):>12} {str(last_loose):>11} {str(last_ok):>16} "
            f"{len(gs):>7} {max(r for _, r in rho):>8.4f} {min(r for _, r in rho):>8.4f}")
    say()
    say("  The gap between 'last loose' and 'last rho~1' is the measured level of")
    say("  distribution: the gears above it strike the survivors at the fair rate as a fact,")
    say("  not as a theorem.")
    say()


    say("=" * 100)
    say("HOW MUCH OF THE BLOCK BUDGET IS UNCONDITIONAL")
    say("  head  = gears with Q_{<g} * g <= L: alpha_g = 2/g by CRT inside the block, so their")
    say("          terms 4/g^2 need no hypothesis at all")
    say("  room  = 1 - head, the budget left for every gear above the head")
    say("  meas  = what those gears actually cost on the real window")
    say("  adv   = what the phase-adversarial envelope charges them")
    say("=" * 100)
    say(f"  {'q':>5} {'#head':>6} {'top head g':>11} {'head':>8} {'room':>8} {'meas tail':>10} "
        f"{'adv tail':>9} {'adv/room':>9}")
    for q in [59, 97, 199, 499, 997]:
        gs = gears(q)
        k0, _, L = window(q)
        P = pi_prefix(gs)
        Q = 1
        head_g = []
        for g in gs:
            if Q * g <= L:
                head_g.append(g)
            Q *= g
        head = sum(4.0 / (g * g) for g in head_g)
        room = 1.0 - head
        eta, _, _, per = run_blocks(k0, L, gs, lambda i, g, Q, lQ, b=L: b)
        meas_tail = sum(m2 for (g, b, m1, m2, nb) in per if g not in head_g)
        adv_tail = sum(min(1.0, (2.0 / g) / P[i]) ** 2
                       for i, g in enumerate(gs) if g not in head_g)
        say(f"  {q:>5} {len(head_g):>6} {head_g[-1]:>11} {head:>8.5f} {room:>8.5f} "
            f"{meas_tail:>10.5f} {adv_tail:>9.5f} {adv_tail/room:>9.3f}")
    say()
    say("  The unconditional head never reaches beyond gear 17 (three to five gears), the")
    say("  adversary is 1.77 to 3.19 times over the room left for everything above it, and")
    say("  the real machine uses about a tenth of that room.  The ratio adv/room is the")
    say("  exact size of the missing input.")
    say()

    say("=" * 100)
    say("THE rho PROFILE, gear by gear, at q = 199 (all 44 gears)")
    say("  rho_g = (fraction of the window's survivors struck by gear g) / (2/g)")
    say("=" * 100)
    q = 199
    gs = gears(q)
    k0, _, L = window(q)
    eta, _, _, per = run_blocks(k0, L, gs, lambda i, g, Q, lQ, b=L: b)
    row = []
    for (g, b, m1, m2, nb) in per:
        row.append(f"{g}:{m1*g/2.0:.3f}")
    for i in range(0, len(row), 6):
        say("    " + "  ".join(row[i:i + 6]))
    say()
    say("  eta_B = sum (4/g^2) rho_g^2 = %.5f; the budget is below 1 while the weighted"
        % eta)
    say("  L2 mean of rho stays below 1/sqrt(sum 4/g^2) = %.4f."
        % (1.0 / math.sqrt(sum(4.0 / (g * g) for g in gs))))
    say()


if __name__ == "__main__":
    main()
    with open(os.path.join(OUT, "bm_mech.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nwritten:", os.path.join(OUT, "bm_mech.txt"))
