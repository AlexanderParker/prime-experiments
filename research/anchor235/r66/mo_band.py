"""mo_band.py -- ITEM 4 (the motor's gate item 2): the chain statement at J = 3 and J = 4 on the
band of old sizes [15, 36], at the rung 29 -> 31, as an exact finite check.

A gap of M + q' of order J is a run of J consecutive gaps of M whose J - 1 interior openings are
all struck in one phase and whose two endpoints are not (docs/proofs/05 (D), (F)).  So the finite
object to enumerate is: every realised J-window of m29 that fuses at some phase of 31.  D_4^#(m29)
has 45,854 distinct rows, so the enumeration is exact and complete, not a search.

For every such fusion we record its span and its largest piece `a`.  The chain statement on the
band is then the finite check

    max { span : J-fusion at 29 -> 31 with largest piece a in [15, 36] }  <=  F(m29) + 31 = 74.

Usage: uv run python research/anchor235/r66/mo_band.py [K0]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mo_core import OUT, base_gaps, closure_step, dict_from_gaps, letters_of, u_of

Q = 31          # the incoming gear
FOLD = 43       # F(m29)
BAND = (15, 36)


def fusions(win, mult, q, J):
    """Rows of `win` whose first J gaps fuse into ONE gap of M + q' at some phase, with the
    number of phases at which they do (the weight eps_J of branching_identity.md 2.6)."""
    n, K = win.shape
    assert J <= K
    d = (2 * u_of(q)) % q
    off = np.zeros((J + 1, n), dtype=np.int64)
    for i in range(J):
        off[i + 1] = off[i] + win[:, i]
    complete = np.all(win[:, :J] != 0, axis=0) if False else np.all(win[:, :J] != 0, axis=1)
    eps = np.zeros(n, dtype=np.int64)
    for z in range(q):
        hit = ((off + z) % q == 0) | ((off + z) % q == d)
        ok = complete & ~hit[0] & ~hit[J]
        for i in range(1, J):
            ok &= hit[i]
        eps += ok
    return eps, off[J]


def main():
    K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t0 = time.time()
    P, g = base_gaps(23)
    win, mult = dict_from_gaps(g, K0)
    print(f"base m23: |D_{K0}| = {win.shape[0]:,}  [{time.time()-t0:.1f}s]", flush=True)
    t0 = time.time()
    w29, m29, st = closure_step(win, mult, 29, m=4, mode="fixed")
    print(f"m29: |D_4^#| = {w29.shape[0]:,} rows, mass {int(m29.sum()):,}, loss {st['loss']}, "
          f"over0 {st['over0']}  [{time.time()-t0:.1f}s]", flush=True)
    assert st["loss"] == 0 and st["over0"] == 0

    out = {"rung": "29->31", "F_old": FOLD, "q": Q, "budget": FOLD + Q,
           "dict4_rows": int(w29.shape[0]), "dict4_mass": int(m29.sum()), "byJ": {}}
    for J in (2, 3, 4):
        eps, span = fusions(w29, m29, Q, J)
        sel = np.flatnonzero(eps > 0)
        sp = span[sel]
        a = w29[sel, :J].astype(np.int64).max(axis=1)
        mass = m29[sel] * eps[sel]
        Qstar = int(sp.max())
        prof = {}
        for av in range(1, int(a.max()) + 1):
            m_ = a == av
            if m_.any():
                idx = int(np.argmax(np.where(m_, sp, -1)))
                prof[av] = {"max_span": int(sp[m_].max()), "n_windows": int(m_.sum()),
                            "mass": int(mass[m_].sum()),
                            "witness": [int(x) for x in w29[sel[idx], :J]]}
        band = {av: prof[av] for av in range(BAND[0], BAND[1] + 1) if av in prof}
        band_max = max((v["max_span"] for v in band.values()), default=0)
        band_arg = [av for av, v in band.items() if v["max_span"] == band_max]
        # extremal witnesses over the whole J
        top = np.argsort(-sp)[:8]
        out["byJ"][str(J)] = {
            "Qstar": Qstar, "n_fusion_windows": int(sel.size),
            "mass": int(mass.sum()),
            "band_max_span": band_max, "band_argmax_a": band_arg,
            "band_margin": FOLD + Q - band_max,
            "band_windows": int(sum(v["n_windows"] for v in band.values())),
            "band_mass": int(sum(v["mass"] for v in band.values())),
            "profile_a_to_maxspan": {str(k): v for k, v in prof.items()},
            "top_witnesses": [{"window": [int(x) for x in w29[sel[i], :J]],
                               "span": int(sp[i]), "a": int(a[i]),
                               "phases": int(eps[sel[i]]), "mult": int(m29[sel[i]])}
                              for i in top],
        }
        print(f"J={J}: Q*_{J} = {Qstar}; fusion windows {sel.size:,}; "
              f"band a in [{BAND[0]},{BAND[1]}]: max span {band_max} at a={band_arg}, "
              f"margin {FOLD + Q - band_max}", flush=True)
        for av in sorted(band):
            v = band[av]
            print(f"   a={av:2d}: max span {v['max_span']:2d}  windows {v['n_windows']:6,}  "
                  f"witness {v['witness']}", flush=True)
    with open(os.path.join(OUT, "band_29_31.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
