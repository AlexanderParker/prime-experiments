"""r55 cl_triples - item 4: triples, and the record's overlap decomposed.

(a) For every triple from {5,7,11,13,17,19}: the three-way block deficit
    c_{g,h,k}(L) = max_g + max_h + max_k - joint_max(triple; L), against the sum of the three
    pairwise deficits; the exact growth law per period; additivity.

(b) For the real machines m11..m23: the record stretch F, the total deficit
    sum_g max_g(F) - F it pays, split into maximality loss and actual overlap, and compared
    with the pairwise collision deficits (all pairs, and the best matching).
"""
import itertools
import os

import numpy as np

from cl_core import (RESULTS, PRIMES, block_deficit, dump, maxstrike, pair_profile,
                     real_sep, say_factory)

LINES = []
say = say_factory(LINES)

SMALL = [5, 7, 11, 13, 17, 19]


def part_a():
    say("=" * 100)
    say("ITEM 4a - triples: the three-way deficit against the sum of the pairwise deficits")
    say("=" * 100)
    say("  growth law predicted for a block B: c_B(L + P) = c_B(L) + sum_g 2P/g "
        "- (P - prod_g (g-2)),  P = prod g")
    say("  for a pair that is +4; for a triple it is 4(g+h+k) - 8")
    say()
    say(f"  {'triple':>14} {'P':>7} {'incr/period':>12} {'4(g+h+k)-8':>11} {'exc':>5} "
        f"{'rate3':>8} {'sum rate2':>10} {'8/ghk':>8} {'L: c3<sum2':>11} {'=':>5} {'>':>4}")
    pc = {}
    for g, h in itertools.combinations(SMALL, 2):
        pc[(g, h)] = pair_profile(g, real_sep(g), h, real_sep(h), 9000)["c"]
    tot_exc = 0
    tot_sub = tot_eq = tot_sup = 0
    for tri in itertools.combinations(SMALL, 3):
        g, h, k = tri
        P = g * h * k
        Lmax = min(2 * P + 4, 9000)
        c3, _ = block_deficit(list(tri), [real_sep(x) for x in tri], Lmax)
        pred = 4 * (g + h + k) - 8
        hi = Lmax - P
        exc = int(np.count_nonzero(c3[1:hi + 1] + pred != c3[1 + P:hi + P + 1]))
        tot_exc += exc
        s2 = (pc[(g, h)][:Lmax + 1] + pc[(g, k)][:Lmax + 1] + pc[(h, k)][:Lmax + 1])
        sub = int(np.count_nonzero(c3[2:] < s2[2:]))
        eq = int(np.count_nonzero(c3[2:] == s2[2:]))
        sup = int(np.count_nonzero(c3[2:] > s2[2:]))
        tot_sub += sub
        tot_eq += eq
        tot_sup += sup
        say(f"  {str(tri):>14} {P:>7} {pred:>12} {pred:>11} {exc:>5} "
            f"{pred / P:>8.5f} {4 / (g * h) + 4 / (g * k) + 4 / (h * k):>10.5f} "
            f"{8 / P:>8.5f} {sub:>11} {eq:>5} {sup:>4}")
    say()
    say(f"  growth law exceptions over the 20 triples: {tot_exc}")
    say(f"  L with c3 < sum of pairwise: {tot_sub};  equal: {tot_eq};  "
        f"GREATER (super-additive): {tot_sup}")
    say("  (rate3 = the triple's deficit per column; sum rate2 = the three pairwise rates;")
    say("   their difference is exactly 8/ghk, the inclusion-exclusion triple term)")
    say()
    say("  the deficits at the run lengths that matter (L = A(K) and L = W(K)):")
    Ls = [16, 22, 28, 37, 45, 48, 60, 68, 88, 140]
    say(f"  {'triple':>14} " + " ".join(f"{('L=%d' % L):>7}" for L in Ls))
    for tri in itertools.combinations(SMALL, 3):
        g, h, k = tri
        P = g * h * k
        c3, _ = block_deficit(list(tri), [real_sep(x) for x in tri], max(Ls))
        say(f"  {str(tri):>14} " + " ".join(f"{int(c3[L]):>7}" for L in Ls))
    say(f"  and the pairwise ones:")
    say(f"  {'pair':>14} " + " ".join(f"{('L=%d' % L):>7}" for L in Ls))
    for g, h in itertools.combinations(SMALL, 2):
        say(f"  {str((g, h)):>14} " + " ".join(f"{int(pc[(g, h)][L]):>7}" for L in Ls))


def machine_record(q):
    """The real machine {5..q}: its period, its record stretch F, and the run's position."""
    gears = [p for p in PRIMES if 5 <= p <= q]
    P = 1
    for g in gears:
        P *= g
    struck = np.zeros(2 * P, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        struck[(u % g)::g] = True
        struck[((-u) % g)::g] = True
    # longest run of struck columns inside [0, 2P), which contains every cyclic run
    idx = np.flatnonzero(~struck)
    gaps = np.diff(idx) - 1
    j = int(np.argmax(gaps))
    F = int(gaps[j])
    start = int(idx[j]) + 1
    return gears, P, F, start, struck


def part_b():
    say()
    say("=" * 100)
    say("ITEM 4b - the record stretch of the real machines m11..m23, overlap decomposed")
    say("=" * 100)
    say("  D = sum_g max_g(F) - F  (the total deficit the record pays)")
    say("  D = M + O with M = sum_g (max_g(F) - n_g)  the maximality loss")
    say("            and  O = sum_g n_g - F           the overlap actually paid")
    say("  C_all = sum over ALL pairs of c(g,h;F);  C_match = best perfect/near-perfect matching")
    say()
    say(f"  {'machine':>9} {'F':>4} {'sum max':>8} {'D':>5} {'M':>4} {'O':>4} "
        f"{'C_all':>6} {'C_match':>8} {'O/C_all':>8} {'D/C_all':>8}")
    rows = []
    for q in (11, 13, 17, 19, 23):
        gears, P, F, start, struck = machine_record(q)
        run = range(start, start + F)
        ng = {}
        for g in gears:
            u = pow(6, -1, g)
            ng[g] = sum(1 for k in run if k % g in (u % g, (-u) % g))
        smax = sum(maxstrike(g, real_sep(g), F) for g in gears)
        D = smax - F
        M = sum(maxstrike(g, real_sep(g), F) - ng[g] for g in gears)
        O = sum(ng.values()) - F
        cs = {}
        for g, h in itertools.combinations(gears, 2):
            cs[(g, h)] = int(pair_profile(g, real_sep(g), h, real_sep(h), F)["c"][F])
        C_all = sum(cs.values())
        # best matching by brute force over pairings of the gear list
        def best_match(gs):
            if len(gs) < 2:
                return 0
            a = gs[0]
            best = best_match(gs[1:])          # leave a unmatched
            for i in range(1, len(gs)):
                b = gs[i]
                rest = gs[1:i] + gs[i + 1:]
                best = max(best, cs[(a, b)] + best_match(rest))
            return best
        C_match = best_match(gears)
        rows.append((q, F, smax, D, M, O, C_all, C_match, cs, ng, gears))
        say(f"  {('m%d' % q):>9} {F:>4} {smax:>8} {D:>5} {M:>4} {O:>4} {C_all:>6} "
            f"{C_match:>8} {O / C_all if C_all else 0:>8.3f} "
            f"{D / C_all if C_all else 0:>8.3f}")
    say()
    say("  per gear at the record (n_g = columns of the record run it strikes; "
        "max = its capacity):")
    for (q, F, smax, D, M, O, C_all, C_match, cs, ng, gears) in rows:
        say(f"    m{q} (F = {F}): " + ", ".join(
            f"{g}:{ng[g]}/{maxstrike(g, real_sep(g), F)}" for g in gears))
    say()
    say("  the pairwise collision deficits at the record length, and the actual pairwise")
    say("  overlaps the record pays (the record's own phases, not the best ones):")
    for (q, F, smax, D, M, O, C_all, C_match, cs, ng, gears) in rows:
        gearsq, P, F2, start, struck = machine_record(q)
        run = list(range(start, start + F))
        act = {}
        for g, h in itertools.combinations(gears, 2):
            ug, uh = pow(6, -1, g), pow(6, -1, h)
            act[(g, h)] = sum(1 for k in run
                              if k % g in (ug % g, (-ug) % g)
                              and k % h in (uh % h, (-uh) % h))
        say(f"    m{q}: c(g,h;F) = " + ", ".join(
            f"{g}-{h}:{cs[(g, h)]}" for (g, h) in sorted(cs)))
        say(f"         actual   = " + ", ".join(
            f"{g}-{h}:{act[(g, h)]}" for (g, h) in sorted(act))
            + f"   (sum {sum(act.values())}, against O = {O})")
    return rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    part_a()
    part_b()
    dump(LINES, "cl_triples.txt")


if __name__ == "__main__":
    main()
