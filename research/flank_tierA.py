"""Constructor round 13: TIER A for the flank bound - the (l+3)-point
machine-free forbidden configurations around a word occurrence.

Structure. A word occurrence with its two flanks is a chain of openings
  p0, p1 = p0+gL, p1+w1, ..., p1+span, p1+span+gR
ALL of which are openings, hence exposed, hence in E mod 35 (gears 5,7 only;
tier A). Interior non-openings give NO tier-A constraint (they are blocked by
higher gears), so tier A content is exactly the endpoint residue system:

  S_m(w) = {r in E_m : every partial sum r + w_1..w_j in E_m}   (word carrier)
  flank pair (gL, gR) TIER-A-FEASIBLE iff exists r in S_m(w) with
      r - gL in E_m  and  r + span + gR in E_m.

This is the (l+3)-point generalisation of no_11_11_chain / A3 (l = 0 gives
A3 exactly). Note gcd(35, q') = 1, so the tooth condition mod q' that defines
COMPATIBILITY is CRT-independent of S(w): tier A and firing never interact.

Computed here per step and per compatible word:
 (1) |S_m(w)| at m = 35, 385, 5005, 85085, 1616615 (gears 5..19) - the tier
     ladder A -> B -> C;
 (2) the TOP-STRATUM FLANK TEST: is a maximal gap (length F) tier-feasible as
     a left/right flank? (derivation target for the measured 0-of-17);
 (3) the fraction of (gL, gR) residue pairs excluded at each modulus;
 (4) the escape check: can a forbidden flank pair be reached by sliding
     sizes by +-1 (the round-9 obstruction, re-tested for flanks).
"""
import numpy as np
from math import prod
import sys
sys.path.insert(0, "research")
from word_ceiling import words, valid_starts, FK, F2K, STEPS

LADDER = [(35, [5, 7]), (385, [5, 7, 11]), (5005, [5, 7, 11, 13]),
          (85085, [5, 7, 11, 13, 17]), (1616615, [5, 7, 11, 13, 17, 19])]


def exposed(gears, m):
    a = np.ones(m, bool)
    for q in gears:
        c = pow(6, -1, q)
        a[c::q] = False
        a[(q - c) % q::q] = False
    return a


EM = {m: exposed(g, m) for m, g in LADDER}


def carrier(w, m):
    """S_m(w) as a boolean mask over residues."""
    E = EM[m]
    s = E.copy()
    acc = 0
    for x in w:
        acc += x
        s &= np.roll(E, -acc % m)
    return s


def feasible(w, gL, gR, m):
    """exists r in S_m(w): r-gL in E and r+span+gR in E."""
    E, s = EM[m], carrier(w, m)
    span = sum(w)
    return bool((s & np.roll(E, gL % m) & np.roll(E, -(span + gR) % m)).any())


def analyse(y, q1):
    F = FK[y]
    print(f"\n=== step {y}->{q1}   F={F}")
    for w in words(q1):
        if not valid_starts(w, q1):
            continue
        span = sum(w)
        row = []
        for m, _ in LADDER:
            s = carrier(w, m)
            row.append(f"{int(s.sum())}/{int(EM[m].sum())}")
        # top-stratum flank feasibility per modulus
        fl = [("L" if feasible(w, F, 1, m) else "-") +
              ("R" if feasible(w, 1, F, m) else "-") for m, _ in LADDER]
        # excluded fraction of (gL,gR) residue pairs at m=35
        E35 = EM[35]
        s35 = carrier(w, 35)
        tot = exc = 0
        for gL in range(35):
            for gR in range(35):
                tot += 1
                if not (s35 & np.roll(E35, gL) &
                        np.roll(E35, -(span + gR) % 35)).any():
                    exc += 1
        print(f"  w={w} span={span}: |S| {row}")
        print(f"      F-flank feasible (L/R) by modulus: {fl}   "
              f"tier-A excluded (gL,gR) residue pairs: {exc}/{tot} "
              f"({100*exc/tot:.0f}%)")


def escape_check():
    """Can a tier-A-forbidden flank pair be reached by sliding sizes +-1?"""
    w = (10,)
    span = 10
    E, s = EM[35], carrier(w, 35)
    forb = [(gL, gR) for gL in range(35) for gR in range(35)
            if not (s & np.roll(E, gL) & np.roll(E, -(span + gR) % 35)).any()]
    worst = 0
    for gL, gR in forb:
        d = min(abs(a) + abs(b) for a in range(-3, 4) for b in range(-3, 4)
                if (s & np.roll(E, (gL + a) % 35) &
                    np.roll(E, -(span + gR + b) % 35)).any())
        worst = max(worst, d)
    print(f"\nescape check (w=(10,)): {len(forb)}/1225 forbidden pairs; "
          f"max L1 slide to a feasible pair = {worst}")


if __name__ == "__main__":
    for m, g in LADDER:
        print(f"modulus {m:>8} (gears {g}): |E| = {int(EM[m].sum())} "
              f"({EM[m].mean():.4f})")
    for y, q1 in STEPS:
        analyse(y, q1)
    escape_check()
