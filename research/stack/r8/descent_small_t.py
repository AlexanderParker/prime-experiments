"""(b) Why the first gap sits low: the columns t = 1, 2, ... of each primorial's stripes.

Column t of P_s is the pair (2 t P_s - 1, 2 t P_s + 1).  A gear g above the base paints it iff
g divides a member, i.e. iff t is on one of g's two teeth t_0 = (2 P_s)^{-1} or -t_0 mod g.  For
t below g that means the tooth IS t (a gear whose tooth sits exactly at t); for g <= t the gear
is periodic on the t-line and reaches t if t = +-t_0 mod g.  So for each primorial and each t
up to the first gap: the two members factored, the painting gears, and for each whether it is
a gear with its tooth exactly at t (g > t) or a periodic small gear (g <= t).

usage: uv run python research/stack/r8/descent_small_t.py
"""
from sympy import primerange, factorint, isprime
from pathlib import Path

def main():
    out = [__doc__.strip(), ""]
    P = 1
    for p in primerange(2, 32):
        P *= p
        if P < 30: continue
        base = [x for x in primerange(2, p + 1)]
        t = 1; rows = []
        while True:
            L, R = 2 * t * P - 1, 2 * t * P + 1
            fl, fr = factorint(L), factorint(R)
            if fl.get(L) == 1 and fr.get(R) == 1:
                rows.append(f"   t = {t}: ({L}, {R}) GAP, both prime"); break
            painters = []
            for m, f in ((L, fl), (R, fr)):
                if f.get(m) == 1: continue
                for g in f:
                    if g > p: painters.append((g, 'L' if m == L else 'R', 'tooth at t' if g > t else 'periodic'))
            fs = lambda f: "*".join(f"{x}^{e}" if e > 1 else str(x) for x, e in sorted(f.items()))
            rows.append(f"   t = {t}: ({fs(fl)}, {fs(fr)}); painted by " + ", ".join(f"{g} {m} ({how})" for g, m, how in sorted(painters)))
            t += 1
            if t > 40: rows.append("   ... no gap by t = 40"); break
        out.append(f"P_s = {P} (base to {p}), columns up to the first gap:")
        out += rows
        # the gears with a tooth at t = 1, 2, 3: those dividing 2 t P_s -+ 1
        out.append("   gears whose tooth sits at t = 1: " + str(sorted(g for g in set(factorint(2 * P - 1)) | set(factorint(2 * P + 1)) if g > p)) +
                   "; at t = 2: " + str(sorted(g for g in set(factorint(4 * P - 1)) | set(factorint(4 * P + 1)) if g > p and g > 2)) +
                   "; at t = 3: " + str(sorted(g for g in set(factorint(6 * P - 1)) | set(factorint(6 * P + 1)) if g > p and g > 3)))
        out.append("")
    Path("research/stack/r8/results_descent_small_t.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
