"""Round 20: the anti-correlation law - how much is NEEDED, and a rigorous
upper bound on p_j from exposure alone.

TWO RESULTS COMPUTED HERE.

(1) HOW MUCH ANTI-CORRELATION DOES (D) NEED? From the suppression law,
    merged_max(j) ~ F_j - lambda ln(1/p_j) <= F + q'  requires
        ln(1/p_j)  >=  (F_j - F - q') / lambda.
    Compare the REQUIRED p_j against the INDEPENDENT value p_1^(j-2) and the
    MEASURED p_j. If independence already clears the requirement, (D) does not
    need the anti-correlation law at all - only a crude upper bound.

(2) A RIGOROUS UPPER BOUND ON p_j FROM EXPOSURE (dodging Lateral's disjunction
    obstruction). "gap = v" = (both endpoints exposed) AND (no opening
    strictly between). Dropping the second condition only INCREASES the
    probability, so
        p_j  <=  sum over qualifying tuples (v_1..v_{j-2}) of
                 prod_q c_q(0, v_1, v_1+v_2, ...) / q,
    where c_q(offsets) = #{r mod q : r+o avoids both teeth of q, all o}.
    Exposure is a CONJUNCTION, so this factorises by CRT exactly - the
    disjunction survives only as a SUM over a small tuple set (<= 4^(j-2)).
    This is Lateral's c_q(g) generalised to multi-lag, used as a bound.
"""
import numpy as np
from math import prod, log
import sys
sys.path.insert(0, "research")

MACH = {19: ([5, 7, 11, 13, 17, 19], 23, 25, 1.20, [25, 31, 35, 38, 47, 50]),
        23: ([5, 7, 11, 13, 17, 19, 23], 29, 34, 1.59, [34, 39, 50, 58, 65, 77]),
        29: ([5, 7, 11, 13, 17, 19, 23, 29], 31, 43, 2.73, [43, 55, 65, 70, 85, 90])}
PMEAS = {19: {3: 3.112e-2, 4: 6.180e-4}, 23: {3: 3.066e-2, 4: 3.622e-5},
         29: {3: 3.737e-2, 4: 2.072e-4, 5: 3.726e-8}}
NGAPS = {19: 378675, 23: 7952175, 29: 214708725}


def c_q(q, offsets):
    c = pow(6, -1, q)
    teeth = {c % q, (q - c) % q}
    return sum(1 for r in range(q)
               if all(((r + o) % q) not in teeth for o in offsets))


def exposure_prob(gears, offsets):
    return prod(c_q(q, offsets) / q for q in gears)


def qual_values(q1, F):
    c = pow(6, -1, q1)
    Q = {0, (2 * c) % q1, (-2 * c) % q1}
    return [v for v in range(1, F + 1) if v % q1 in Q]


def run():
    print("(1) HOW MUCH ANTI-CORRELATION DOES (D) NEED?")
    print("  mach j   F_j-F-q'   required p_j <=   independent p_1^(j-2)   "
          "measured      indep margin")
    for y, (gears, q1, F, lam, Fj) in MACH.items():
        p1 = PMEAS[y][3]
        for j in (3, 4, 5, 6):
            need_ln = (Fj[j - 1] - F - q1) / lam
            if need_ln <= 0:
                print(f"  {y:4d} {j}   {Fj[j-1]-F-q1:6d}    (no constraint)")
                continue
            req = np.exp(-need_ln)
            ind = p1 ** (j - 2)
            meas = PMEAS[y].get(j)
            print(f"  {y:4d} {j}   {Fj[j-1]-F-q1:6d}    {req:.3e}         "
                  f"{ind:.3e}          "
                  f"{meas if meas is not None else 'none':>9}    "
                  f"x{req/ind:,.0f}")

    print("\n(2) RIGOROUS UPPER BOUND ON p_j FROM EXPOSURE (no disjunction problem)")
    for y, (gears, q1, F, lam, Fj) in MACH.items():
        V = qual_values(q1, F)
        # j=3: one interior
        b3 = sum(exposure_prob(gears, (0, v)) for v in V)
        # j=4: two consecutive interiors
        b4 = sum(exposure_prob(gears, (0, v1, v1 + v2)) for v1 in V for v2 in V)
        # j=5: three
        b5 = sum(exposure_prob(gears, (0, v1, v1 + v2, v1 + v2 + v3))
                 for v1 in V for v2 in V for v3 in V)
        m3, m4 = PMEAS[y][3], PMEAS[y].get(4)
        m5 = PMEAS[y].get(5)
        print(f"  machine {y} (q'={q1}, qualifying values {V}):")
        print(f"    p_3 <= {b3:.3e}   measured {m3:.3e}   slack x{b3/m3:.1f}")
        if m4:
            print(f"    p_4 <= {b4:.3e}   measured {m4:.3e}   slack x{b4/m4:.1f}"
                  f"   [independent-of-bound would be {b3**2:.3e}]")
        if m5:
            print(f"    p_5 <= {b5:.3e}   measured {m5:.3e}   slack x{b5/m5:.1f}")
        # does the BOUND alone clear the requirement?
        for j, b in ((3, b3), (4, b4), (5, b5)):
            need_ln = (Fj[j - 1] - F - q1) / lam
            if need_ln > 0:
                req = np.exp(-need_ln)
                print(f"    j={j}: requirement p_j <= {req:.3e};  bound gives "
                      f"{b:.3e}  -> {'CLEARS' if b <= req else 'insufficient'}")


if __name__ == "__main__":
    run()
