"""pa_forced.py -- which gears are FORCED to strike inside the letter gap, by the arithmetic alone.

For gear g and an a_L-gap with near end in tooth units lam_g = 6x (mod g), the interior columns
struck by g are the offsets 1 <= j <= a_L - 1 with lam_g + 6j = +-1 (mod g), and lam_g must avoid
the four classes that would strike an end.  Over the admissible lam_g this gives a RANGE of
interior strike counts; when its minimum is >= 1 the gear cannot avoid striking inside the letter
gap whatever the phase -- a forced strike, decided by q' mod g alone (through 6 a_L = 2(q' + eps)).

Also tabulated: the endpoint cost c_g(a_L), which by the identity is
    2 if g | q' + eps,   3 if g | q' + 2 eps,   4 otherwise,
i.e. one congruence condition on q' mod g per gear -- the gear's ROLE at the letter.

Outputs results/pa_forced.txt / .json.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import u_of                                  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

RUNGS = [(7, [5]), (11, [5, 7]), (13, [5, 7, 11]), (17, [5, 7, 11, 13]),
         (19, [5, 7, 11, 13, 17]), (23, [5, 7, 11, 13, 17, 19]),
         (29, [5, 7, 11, 13, 17, 19, 23]), (31, [5, 7, 11, 13, 17, 19, 23, 29]),
         (37, [5, 7, 11, 13, 17, 19, 23, 29, 31])]


def unorm(g):
    w = u_of(g)
    return min(w, g - w)


def eps_of(q):
    return 1 if q % 6 == 5 else -1


def main():
    t0 = time.time()
    L = []
    W = L.append
    W("=== FORCED STRIKES INSIDE THE LETTER GAP, FROM THE ARITHMETIC ALONE ===")
    W("rung | a_L | gear | long arc g - d_g | role (c_g) | congruence on q' | admissible lam |"
      " interior strikes min..max | forced?")
    res = []
    for qn, M in RUNGS:
        e = eps_of(qn)
        aL = 2 * unorm(qn)
        sh = 2 * (qn + e)                    # = 6 a_L
        for g in M:
            d = (2 * u_of(g)) % g
            long_arc = g - min(d, g - d)
            if (qn + e) % g == 0:
                role, cong = 2, f"q' = {(-e) % g} (mod {g})"
            elif (qn + 2 * e) % g == 0:
                role, cong = 3, f"q' = {(-2 * e) % g} (mod {g})"
            else:
                role, cong = 4, "-"
            bad = {1 % g, (-1) % g, (1 - sh) % g, (-1 - sh) % g}
            adm = [lam for lam in range(g) if lam % g not in bad]
            counts = []
            for lam in adm:
                counts.append(sum(1 for j in range(1, aL)
                                  if (lam + 6 * j) % g in (1 % g, (g - 1) % g)))
            lo, hi = (min(counts), max(counts)) if counts else (0, 0)
            W(f"{qn:>4} | {aL:>3} | {g:>4} | {long_arc:>16} | {role:>10} | {cong:<16} | "
              f"{len(adm):>14} | {lo}..{hi} | {'YES' if lo >= 1 else 'no'}")
            res.append(dict(q=qn, aL=aL, g=g, long_arc=long_arc, c=role, adm=len(adm),
                            lo=lo, hi=hi, forced=bool(lo >= 1)))
    W("\n--- summary: forced gears per rung ---")
    W("rung | a_L | gears forced to strike inside the letter gap | gears free to miss it entirely")
    for qn, M in RUNGS:
        rows = [r for r in res if r["q"] == qn]
        f = [r["g"] for r in rows if r["forced"]]
        n = [r["g"] for r in rows if not r["forced"]]
        W(f"{qn:>4} | {rows[0]['aL']:>3} | {f or '-'} | {n or '-'}")
    W("  a gear is forced exactly when its long arc (the larger tooth spacing, about 2g/3) is")
    W("  short enough against a_L that no admissible phase fits the whole letter gap between two")
    W("  consecutive teeth -- and a_L = (q'+eps)/3, so the threshold is a statement about q'/g.")
    json.dump(res, open(os.path.join(OUT, "pa_forced.json"), "w"))
    txt = "\n".join(L)
    open(os.path.join(OUT, "pa_forced.txt"), "w").write(txt)
    print(txt[-2600:])
    print(f"[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
