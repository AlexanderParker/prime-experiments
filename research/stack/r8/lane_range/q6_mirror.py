"""Q6 (own rule): the mirror n -> P - n of q's period and T's classes.

q's openings are mirror-symmetric (P = 0 mod every gear of q).  For g in T the mirror sends the
class e c_g (mod g) to the residue P - e c_g, which is a class of g iff P = e d_g (mod g),
i.e. iff g | 3P - e.  So:
  g | 3P-1 : the class +c_g is carried onto itself, the class -c_g off both classes;
  g | 3P+1 : the class -c_g is carried onto itself, the class +c_g off both classes;
  otherwise: the mirror carries every strike of g to a column g does not strike.
(g cannot divide both 3P-1 and 3P+1.)  This is the leg rule applied to the gap P.
Consequence: the mirror image of an opening struck only by gears g not dividing (3P-1)(3P+1)
is an opening those gears do not strike.
"""
import numpy as np
from lane_common import period, c, d, openings, T_primes, factor


def main():
    for q in (7, 11, 13, 17):
        P = period(q)
        O = openings(q)
        Oset = set(O.tolist())
        assert all((P - n) in Oset or (P - n) == 0 for n in Oset), "openings mirror-symmetric"
        T = T_primes(q)
        cols = np.arange(0, P + 1)
        fixed_plus = [g for g in T if (3 * P - 1) % g == 0]
        fixed_minus = [g for g in T if (3 * P + 1) % g == 0]
        assert not set(fixed_plus) & set(fixed_minus)
        print(f"q={q} P={P} 3P-1={3*P-1}={factor(3*P-1)} 3P+1={3*P+1}={factor(3*P+1)}; T={T[0]}..{T[-1]} ({len(T)})")
        print(f"   gears of T fixing class +c_g: {fixed_plus}; fixing class -c_g: {fixed_minus}")
        for g in T:
            cg = c(g)
            A = {+1: set(cols[cols % g == cg].tolist()), -1: set(cols[cols % g == (g - cg) % g].tolist())}
            for e in (+1, -1):
                mir = {P - n for n in A[e]}
                predicted_fixed = (3 * P - e) % g == 0
                if predicted_fixed:
                    assert mir == A[e], (q, g, e)
                    assert not (mir & A[-e])
                else:
                    assert not (mir & (A[1] | A[-1])), (q, g, e)
            # on openings: mirror of a struck opening is an unstruck opening unless g fixes that class
            struck = O[(O % g == cg) | (O % g == (g - cg) % g)]
            mir_struck = np.array([P - n for n in struck if n != P])
            if len(mir_struck):
                back = ((mir_struck % g) == cg) | ((mir_struck % g) == (g - cg) % g)
                if g in fixed_plus or g in fixed_minus:
                    e = 1 if g in fixed_plus else -1
                    cls = (struck % g) == (e * cg) % g
                    cls = cls[struck != P]
                    assert np.array_equal(back, cls)
                else:
                    assert not back.any()
        print(f"   PASS for all g in T: mirror fixes exactly the predicted class, carries every other class off g's classes")
        if fixed_plus or fixed_minus:
            g = (fixed_plus or fixed_minus)[0]
            e = 1 if fixed_plus else -1
            print(f"   e.g. g={g}: c_g={c(g)}, d_g={d(g)}, P mod g = {P % g} = {'+' if e==1 else '-'}d_g mod g; "
                  f"class {'+' if e==1 else '-'}c_g = {(e*c(g)) % g} -> P - {(e*c(g))%g} = {(P - e*c(g)) % g} mod g")


if __name__ == "__main__":
    main()
