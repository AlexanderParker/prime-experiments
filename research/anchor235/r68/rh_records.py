"""rh_records.py -- the record stretches of M + q' in the pullback's coordinate: for each record
fusion, the two tooth windows W_0, W_1 inside the stretch, the openings of M on them (the word's
skeleton), the richest value Omega_slots over all translates, the deficit, the exact rank of the
record's translate among all P(M) translates, and the phase classes the argmax pins.

Inputs: results/positions_*.json from rh_scan.py where available (every occurrence per period),
else the fusions on record (rh_core.RECORD_FUSIONS).

    uv run python research/anchor235/r68/rh_records.py
"""
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh_core import (CORPUS_F, CORPUS_L, OUT, RECORD_FUSIONS, gears_of, letter_data, mask_to_set,
                     next_gear, omega, open_count_distribution, phases_for, u_of)

RUNGS = [13, 17, 19, 23, 29, 31, 37]


def load_positions():
    pos = {}
    for fn in ("positions_small.json", "positions_29.json", "positions_31.json", "positions_37.json"):
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            pos.update(json.load(open(p)))
    return pos


def analyse(y, q, fusion, x0_mod_q=None):
    gears = gears_of(y)
    Ld = letter_data(q)
    d = Ld["d"]
    G = sum(fusion)
    fL, word, fR = fusion[0], list(fusion[1:-1]), fusion[-1]
    offs = [0]
    for w in word:
        offs.append(offs[-1] + w)
    s = next((o % q for o in offs if o % q != 0), None)
    if s is None:
        if x0_mod_q is None:
            raise ValueError("all-pad word needs the position to fix the other tooth")
        u = u_of(q)
        s = (q - d) if x0_mod_q == u % q else d
    lo, hi = -fL, G - fL          # slots strictly inside the stretch, relative to x_0
    W0 = [m * q for m in range(-G // q - 2, G // q + 3) if lo < m * q < hi]
    W1 = [s + m * q for m in range(-G // q - 2, G // q + 3) if lo < s + m * q < hi]
    slots = W0 + W1
    S0 = [o for o in offs if o % q == 0]
    S1 = [o for o in offs if o % q != 0]
    assert set(offs) <= set(slots), (fusion, offs, slots)
    best, args, feas, pc = omega(slots, gears)
    L1 = len(offs)
    dist, P = open_count_distribution(slots, gears)
    ge = sum(c for k, c in dist.items() if k >= L1)
    gt = sum(c for k, c in dist.items() if k > L1)
    at_max = sum(c for k, c in dist.items() if k == best)
    argsets = [sorted(mask_to_set(int(a), slots)) for a in args]
    # does the record's open set extend to a richest set?
    rec_set = set(offs)
    extends = [A for A in argsets if rec_set <= set(A)]
    # phases pinned by the argmax sets, per biting gear (gears that cannot be phased off = those
    # with fewer than g admissible phases)
    pins = []
    for A in argsets:
        pins.append({str(g): phases_for(A, g) for g in gears})
    rec_pins = {str(g): phases_for(sorted(rec_set), g) for g in gears}
    return {"q": q, "G": G, "fusion": list(fusion), "s": s, "word_offsets": offs, "W0": W0, "W1": W1,
            "n0": len(W0), "n1": len(W1), "S0": S0, "S1": S1, "L+1": L1, "omega_slots": best,
            "deficit": best - L1, "n_argmax": len(argsets), "argmax_sets": argsets,
            "record_extends_to_argmax": len(extends),
            "rank_ge": str(Fraction(ge, P)), "rank_ge_float": ge / P, "rank_gt_float": gt / P,
            "frac_at_max": at_max / P, "dist": {str(k): str(c) for k, c in sorted(dist.items())},
            "P": str(P), "argmax_pins": pins, "record_pins": rec_pins}


def main():
    pos = load_positions()
    out = {}
    print("rung  q'  G   fusion               s  n0 n1 |S0| |S1| L+1 Omega def #arg  rank(>=L+1)   rank(>L+1)  frac@max")
    for y in RUNGS:
        q = next_gear(y)
        key = f"{y}->{q}"
        if key in pos:
            recs = pos[key]["records"]
            fusions = {}
            for r in recs:
                fusions.setdefault(tuple(r["fusion"]), []).append(r)
        else:
            fusions = {tuple(f): [] for f in RECORD_FUSIONS[y]}
        out[key] = {}
        for fus, recs in fusions.items():
            x0q = recs[0]["phase"][str(q)] if recs else None
            row = analyse(y, q, fus, x0q)
            row["occurrences_per_period"] = len(recs)
            # consistency of the actual positions with the record's own pins, and whether the
            # actual translate is a richest one (it opens an argmax set) -- it cannot if deficit > 0
            if recs:
                ok = all(int(r["phase"][str(g)]) in row["record_pins"][str(g)] for r in recs for g in gears_of(y))
                row["positions_consistent"] = ok
                # phase vectors of the occurrences
                row["positions"] = [{"c": r["c"], "x0": r["x0"], "phase": r["phase"]} for r in recs]
            out[key][" ".join(map(str, fus))] = row
            print(f"{key:7s} {q:2d} {row['G']:3d} {str(fus):20s} {row['s']:2d} {row['n0']:2d} {row['n1']:2d} "
                  f"{len(row['S0']):3d} {len(row['S1']):4d} {row['L+1']:3d} {row['omega_slots']:4d} "
                  f"{row['deficit']:3d} {row['n_argmax']:3d}  {row['rank_ge_float']:.5f}      "
                  f"{row['rank_gt_float']:.5f}   {row['frac_at_max']:.6f}   occ/period={len(recs)} "
                  f"ext->argmax={row['record_extends_to_argmax']} "
                  f"{'pos ok' if recs and row['positions_consistent'] else ''}")
            print(f"        W0={row['W0']} W1={row['W1']} word={row['word_offsets']} argmax={row['argmax_sets'][:3]}")
            print(f"        dist={row['dist']}")
    # the joint two-class sharpened cap with {5,7} only against all gears, corpus rungs
    print("\njoint two-class Omega^(2) at the corpus T: gears {5,7} vs all gears of M")
    jt = {}
    for y in [19, 23, 29, 31, 37, 41, 43, 47, 53]:
        q = next_gear(y)
        Ld = letter_data(q)
        G = CORPUS_F[q]
        T = (G - 2) // q
        vals = {}
        for s in (Ld["a"], Ld["b"]):
            T1 = (G - 2 - s) // q
            slots = [m * q for m in range(T + 1)] + [s + m * q for m in range(T1 + 1)]
            b57, _, _, _ = omega(slots, [5, 7], want_args=False)
            ball, _, _, _ = omega(slots, gears_of(y), want_args=False)
            vals[s] = (b57, ball)
        cap57 = max(v[0] for v in vals.values()) - 1
        capall = max(v[1] for v in vals.values()) - 1
        jt[f"m{y}"] = {"q": q, "G": G, "T": T, "per_s": {str(k): v for k, v in vals.items()},
                       "cap_57": cap57, "cap_all": capall, "L": CORPUS_L[y]}
        print(f"  m{y} q'={q} G={G} T={T}: per s {vals}; cap {5,7} = {cap57}, cap all = {capall}, L = {CORPUS_L[y]}")
    out["joint_cap"] = jt
    with open(os.path.join(OUT, "records.json"), "w") as f:
        json.dump(out, f, indent=1, default=int)


if __name__ == "__main__":
    main()
