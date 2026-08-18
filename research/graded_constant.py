"""Round 12 lateral, part 3: the graded constant on REALIZED chains.

Corrected firing law (see firing_law_check.py + lateral.md round 12):
a fuel site fires iff its position lies in ONE residue class mod q'. Over one
period of the OLD machine that is a density 1/q'; but the NEW machine's period
is q' * P_old and P_old is invertible mod q', so each site fires at exactly
one of the q' phase windows:

    realized k-chains per NEW period  =  N_k   (exactly, no suppression)
    realized density                  =  N_k / P_new  =  (1/q') x site density

So alignment is a DENSITY factor, never a count factor - it buys the graded
constant nothing. Table below prices every censused step both ways.
"""
F = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
F2 = {13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90}
NK = {(19, 23): (11784, 62, 0), (23, 29): (243816, 0, 0),
      (23, 31): (248058, 276, 0), (29, 31): (8022924, 13000, 4),
      (29, 37): (3286190, 374, 0), (31, 37): (None, 70964, 216)}
STEPS = [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31), (31, 37)]

print(f"{'step':>9} {'q':>3} {'F_old':>5} {'F_new':>5} {'incr/q':>7} "
      f"{'lemma1':>7} {'excess':>7} {'exc/q':>6} {'N3':>7} {'N4':>4} "
      f"{'realized k>=3 /period':>21}")
for y, q in STEPS:
    incr = (F[q] - F[y]) / q
    l1 = (F2[y] - F[y]) / q
    exc = F[q] - F2[y]
    n = NK.get((y, q), (None, 0, 0))
    real = "" if n[1] is None else f"{n[1]} (k=3) + {n[2]} (k=4)"
    print(f"{y:>4}->{q:<3} {q:>3} {F[y]:>5} {F[q]:>5} {incr:>7.3f} "
          f"{l1:>7.3f} {exc:>7} {exc/q:>6.3f} {n[1]:>7} {n[2]:>4} {real:>21}")
print()
print("budget check: tolerance route needs increment <= 2.5 q at every step")
print(f"  max measured increment/q = {max((F[q]-F[y])/q for y, q in STEPS):.3f} "
      f"at 31->37; headroom factor {2.5/max((F[q]-F[y])/q for y, q in STEPS):.1f}x")
print("  excess dominates ONLY at 31->37 (0.541 vs lemma1 0.270) - the step")
print("  with the largest fuel population (N3=70964, N4=216), consistent with")
print("  realized = N_k per period: fuel abundance drives excess, alignment")
print("  does not damp it.")
