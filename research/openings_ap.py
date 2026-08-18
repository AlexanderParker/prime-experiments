"""Round 18 lateral: the OPENINGS AP THEOREM - what the machine cannot do.

Round 16's AP lemma was stated for difference q' (the gear being added). It
generalises to ARBITRARY difference, and to every gear, and that is a much
stronger statement about the opening set itself.

  THEOREM (openings AP). Gear q kills exactly 2 residues mod q, so openings
  occupy exactly q-2 residues mod q. An arithmetic progression of L terms with
  common difference d, gcd(d,q) = 1, occupies min(L, q) distinct residues mod
  q. So if L > q - 2 the progression must meet a tooth. Hence:

      an AP of L openings has common difference divisible by EVERY gear
      q <= y with q < L + 2.

  Corollaries (the testable form): in the gap word of any machine,
      3 consecutive equal gaps (g,g,g)      require 5 | g
      5 consecutive equal gaps              require 35 | g
      9 consecutive equal gaps              require 385 | g
  and an AP of L >= y+2 openings needs difference divisible by the whole
  primorial P(y), i.e. difference >= P(y).

This is a "machine cannot" statement of the same family as the round-8 32-cap
(gears 5,7 alone cap saturated runs at 32) - unconditional, scale-free, and
about the OPENINGS rather than about gap sizes.

Tests: (1) exhaustive verification on machines 13..31 that no forbidden
repetition occurs; (2) the actual repetition spectrum (longest runs of equal
gaps and their g values); (3) the F(M_y)/y^2 trend, as a diagnostic of whether
the max gap is really "constant x y^2" or creeping.
"""
from math import prod
import numpy as np
from split_gap_law import primes

def openings(y, chunk=100_000_000):
    gears = primes(5, y)
    P = prod(gears)
    out = []
    a = 0
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, bool)
        for q in gears:
            u = pow(6, -1, q)
            for t in (u, q - u):
                killed[(t - a) % q::q] = True
        out.append(np.flatnonzero(~killed).astype(np.int64) + a)
        a += S
    return np.concatenate(out), P

def required_divisor(L, y):
    return prod([q for q in primes(5, y) if q < L + 2]) or 1

print("=" * 74)
print("PART 1: the theorem, and what it forbids")
for L in range(3, 12):
    need = [q for q in (5, 7, 11, 13, 17, 19) if q < L + 2]
    print(f"  AP of L={L:>2} openings: difference must be divisible by "
          f"{need} -> {prod(need) if need else 1}")

print("=" * 74)
print("PART 2: verification on real machines (full periods)")
print(f"  {'y':>3} {'openings':>12} {'F':>4} {'F/y^2':>7} "
      f"{'max equal-run':>14} {'g there':>8} {'5|g?':>5} {'violations':>11}")
for y in (13, 17, 19, 23, 29, 31):
    o, P = openings(y)
    d = np.diff(o)
    F = int(d.max())
    # longest run of equal consecutive gaps
    change = np.flatnonzero(np.diff(d) != 0)
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change, [len(d) - 1]))
    lens = ends - starts + 1
    i = int(lens.argmax())
    runmax = int(lens.max())
    gval = int(d[starts[i]])
    # violations: a run of >= 3 equal gaps g with 5 not dividing g
    viol = 0
    for st, ln in zip(starts, lens):
        if ln >= 3:
            g = int(d[st])
            if g % 5 != 0:
                viol += 1
            if ln >= 5 and g % 35 != 0:
                viol += 1
    print(f"  {y:>3} {len(o):>12} {F:>4} {F/y**2:>7.4f} {runmax:>14} "
          f"{gval:>8} {str(gval%5==0):>5} {viol:>11}")

print("=" * 74)
print("PART 3: the F/y^2 diagnostic (slot units)")
data = [(13, 11), (17, 18), (19, 25), (23, 34), (29, 43), (31, 58), (37, 88)]
print(f"  {'y':>3} {'F':>4} {'F/y^2':>7} {'F/(y^2/ln^2 y)':>16}")
import math
for y, F in data:
    print(f"  {y:>3} {F:>4} {F/y**2:>7.4f} {F/(y**2/math.log(y)**2):>16.4f}")
print("  (if the max gap is ~ c*y^2 the middle column is flat; if it carries")
print("   an extra log^2 - the Cramer/Iwaniec shape - the right column is.)")
