"""rh_omega.py -- Omega^full(n), the rich-interval function of the pullback M^(q') with EVERY gear
of M, exactly, for the rungs m19 -> m53 and n <= NMAX; the corridor values (gears 5, 7) per class
and uniform; the first n where the full value drops below the corridor and the gear that does it;
the sharpened E2 at the corpus rungs (per class and joint two-class); the stacking deficit D(n)
and the smallest gear subset that forces it.

Only the gears g <= 2n of M bite a window of n multipliers (rich_half.md 0.1); the script uses
every gear of M anyway and records that the two agree.

    uv run python research/anchor235/r68/rh_omega.py [NMAX]
"""
import json
import os
import sys
import time
from itertools import combinations
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh_core import (CORPUS_F, CORPUS_L, OUT, arc, gears_of, letter_data, mask_to_set, min_g,
                     next_gear, omega, omega_prefix, twisted_sep)

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RUNGS = [19, 23, 29, 31, 37, 41, 43, 47, 53]


def corridor_uniform(nmax):
    """Omega_{5,7}(n) = max over invertible steps r mod 35 of the largest feasible subset of
    {0, r, 2r, ...} under gears 5 and 7 (pad_cap.md 4.1, recomputed)."""
    out = {}
    for n in range(1, nmax + 1):
        best = 0
        for r in range(1, 35):
            if gcd(r, 35) != 1:
                continue
            b, _, _, _ = omega([m * r for m in range(n)], [5, 7], want_args=False)
            best = max(best, b)
        out[n] = best
    return out


def main():
    t0 = time.time()
    res = {"NMAX": NMAX}
    uni = corridor_uniform(NMAX)
    res["corridor_uniform"] = uni
    print("Omega_{5,7}(n) uniform:", [uni[n] for n in range(1, NMAX + 1)])
    for y in RUNGS:
        q = next_gear(y)
        gears = gears_of(y)
        Ld = letter_data(q)
        tw = {g: twisted_sep(g, q) for g in gears}
        arcs = {g: arc(tw[g], g) for g in gears}
        row = {"q": q, "gears": gears, "twisted_sep": tw, "twisted_arc": arcs, "a": Ld["a"], "b": Ld["b"],
               "omega_full": {}, "omega_57_class": {}, "prefix": {}, "n_argmax": {}, "argmax_example": {},
               "bite_check": {}, "min_g": {}, "D": {}, "forcing_subset": {}}
        print(f"\nm{y} -> q'={q}: twisted arcs {arcs}")
        for n in range(1, NMAX + 1):
            slots = [m * q for m in range(n)]
            chain, best, args, feas, pc = omega_prefix(slots, gears)
            biting = [g for g in gears if g <= 2 * n]
            b2, _, _, _ = omega(slots, biting, want_args=False)
            row["bite_check"][n] = (best == b2)
            row["omega_full"][n] = best
            row["omega_57_class"][n] = chain[1][1]
            row["prefix"][n] = chain
            row["n_argmax"][n] = int(len(args))
            row["argmax_example"][n] = [o // q for o in mask_to_set(int(args[0]), slots)]
            mg = {g: min_g(g, tw[g], n) for g in gears}
            row["min_g"][n] = mg
            D = (n - best) - max(mg.values())
            row["D"][n] = D
            forcing = None
            if D > 0:
                for k in range(2, len(biting) + 1):
                    for S in combinations(biting, k):
                        bS, _, _, _ = omega(slots, list(S), want_args=False)
                        if (n - bS) - max(mg[g] for g in S) > 0:
                            forcing = list(S)
                            break
                    if forcing:
                        break
            row["forcing_subset"][n] = forcing
            print(f"  n={n:2d}: Omega_full={best} (5,7 class {chain[1][1]}, uniform {uni[n]}) "
                  f"prefix={[c[1] for c in chain]} argmax#={len(args)} ex={row['argmax_example'][n]} "
                  f"maxmin_g={max(mg.values())} D={D} forcing={forcing}", flush=True)
        # first drops
        n0_class = next((n for n in range(1, NMAX + 1) if row["omega_full"][n] < row["omega_57_class"][n]), None)
        n0_uni = next((n for n in range(1, NMAX + 1) if row["omega_full"][n] < uni[n]), None)
        dec = None
        if n0_class:
            ch = row["prefix"][n0_class]
            target = row["omega_57_class"][n0_class]
            dec = next(g for g, v in ch if v < target)
        row["n0_class"], row["n0_uniform"], row["deciding_gear"] = n0_class, n0_uni, dec
        # sharpened E2 at the corpus T
        G = CORPUS_F[q]
        T = (G - 2) // q
        cap_full = 2 * row["omega_full"][T + 1] - 1
        joint = {}
        for s in (Ld["a"], Ld["b"]):
            T1 = (G - 2 - s) // q
            slots = [m * q for m in range(T + 1)] + [s + m * q for m in range(T1 + 1)]
            b, args, _, _ = omega(slots, gears)
            joint[s] = {"T1": T1, "omega2": b, "n_argmax": int(len(args)),
                        "example": mask_to_set(int(args[0]), slots)}
        cap_joint = max(v["omega2"] for v in joint.values()) - 1
        row["E2"] = {"G": G, "T": T, "cap_full": cap_full, "cap_joint": cap_joint, "L": CORPUS_L[y],
                     "slack_full": cap_full - CORPUS_L[y], "slack_joint": cap_joint - CORPUS_L[y],
                     "joint": joint}
        print(f"  first drop below corridor: class n0={n0_class} (gear {dec}), uniform n0={n0_uni}; "
              f"E2: G={G} T={T} cap_full={cap_full} cap_joint={cap_joint} L={CORPUS_L[y]} "
              f"slack {cap_full - CORPUS_L[y]} / {cap_joint - CORPUS_L[y]}  joint={joint}")
        res[f"m{y}"] = row
    with open(os.path.join(OUT, "omega.json"), "w") as f:
        json.dump(res, f, indent=1, default=int)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
