"""The residual case that resists, at m29 and m31: the runs where the outer sum is at or near
F_2 and no colouring exists.  Needs only x0 mod g."""
import sys
from gl_glue import gears_of, us_of, cov_pair, solve_cover, strikers, glue_free, shifted_best
from gl_move import legality

CASES = [
    (29, 42157188, 25, 7, 30, "TIGHT: L+R = 55 = F_2(m29) exactly; N(7)=55 is the law's only tight cell"),
    (29, 278620515, 18, 10, 30, "the record run: 43 = F(m31 base) ... N(10)=48=F+5, the run that killed the F+1 law"),
    (29, 19068420, 28, 9, 26, "L+R = 54 = F_2 - 1"),
    (29, 412573910, 32, 6, 15, ""),
    (31, 3407593522, 31, 7, 35, "L+R = 66 = F_2 - 2 at m31"),
    (31, 913998995, 47, 6, 12, ""),
]
F = {29: 43, 31: 58}
F2 = {29: 55, 31: 68}
for (top, x0, L, v, R, note) in CASES:
    gears = gears_of(top); us = us_of(gears)
    x1, x2, x3 = x0 + L, x0 + L + v, x0 + L + v + R
    leg, pad = legality(gears, us, v)
    KL = strikers(gears, us, x1 - v); KR = strikers(gears, us, x2 + v)
    T, h, cL, cR = cov_pair(gears, us, x0, L, R, x2)
    sol = solve_cover(cL, cR, T, h)
    print(f"\nm{top} (L,v,R)=({L},{v},{R}) sum={L+R} (F{L+R-F[top]:+d}, F_2{L+R-F2[top]:+d}) "
          f"x0={x0}  {note}")
    print(f"   Leg(v) (v is a letter of) = {leg};  Pad(v) (g | v) = {pad}")
    print(f"   left shadow x1-v struck by {KL}; right shadow x2+v struck by {KR}; "
          f"movable shadow gears (in Leg) = {sorted(set(KR) & set(leg))}")
    print(f"   C2: {'OK' if sol else 'FAIL'}")
    soleL = [g for g in gears if any(strikers(gears, us, c) == [g] for c in range(x0+1, x1))]
    soleR = [g for g in gears if any(strikers(gears, us, c) == [g] for c in range(x2+1, x3))]
    print(f"   sole strikers on the left flank {soleL}; on the right flank {soleR}; "
          f"both {sorted(set(soleL) & set(soleR))}")
    for f in (1, 2, 3):
        s = glue_free(gears, us, x0, L, v, R, f)
        if s:
            print(f"   C2+{f}: SUCCESS  {dict(zip(gears, s))}")
            break
    else:
        print("   C2+1..3: no")
    sb = shifted_best(gears, us, x0, L, v, R, tmax=12)
    if sb:
        t = sb[0]
        print(f"   Cs overlap t={t}: certifies L+R <= F+1+t = {F[top]+1+t}; "
              f"loss vs F_2 = {max(0, F[top]+1+t-F2[top])}")
    else:
        print("   Cs: no t <= 12")
