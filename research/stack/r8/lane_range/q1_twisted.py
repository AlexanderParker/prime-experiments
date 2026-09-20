"""Q1: one gear g of T on one period of q.

Claim: the columns of class +c_g that are openings of q are exactly n = c_g + j g with j an
opening of the twisted machine M_g^+ : gears 5..q, teeth at j = a_h +- b_h (mod h),
a_h = -g^{-1} c_g, b_h = g^{-1} c_h, tooth distance 2 b_h = (3g)^{-1} mod h.
The class -c_g hits are n = -c_g + j' g with j' an opening of M_g^- = -M_g^+ (teeth negated):
the - class reads the same twisted cycle backwards from j = 0.
Over a full j-period the twisted openings biject onto q's openings by j -> c_g + j g mod P.
Spacing of consecutive + hits = g * (twisted gap).
"""
import sys
import numpy as np
from lane_common import gears, period, c, d, openings, record, T_primes, strikes


def twisted_openings(q, g, jmax):
    """openings of M_g^+ for j in 0..jmax, plus the per-gear teeth."""
    gs = gears(q)
    m = np.zeros(jmax + 1, dtype=bool)
    teeth = {}
    for h in gs:
        ginv = pow(g, -1, h)
        a = (-ginv * c(g)) % h
        b = (ginv * c(h)) % h
        assert (2 * b) % h == pow(3 * g, -1, h), "tooth distance is (3g)^-1 mod h"
        t1, t2 = (a + b) % h, (a - b) % h
        teeth[h] = (t1, t2)
        m[t1::h] = True
        m[t2::h] = True
    return np.nonzero(~m)[0], teeth


def check(q, verbose_g=None):
    P = period(q)
    O = openings(q)
    Oset = set(O.tolist())
    F = record(q)
    T = T_primes(q)
    print(f"q={q} P={P} |O|={len(O)} F={F} T={T[0]}..{T[-1]} ({len(T)} gears)")
    F_tw = {}
    for g in T:
        cg = c(g)
        plus = O[(O % g) == cg]                    # + class hits on openings
        minus = O[(O % g) == (g - cg) % g]
        # (a) + hits are the twisted openings in j-coordinate
        j_hits = (plus - cg) // g
        assert np.all((plus - cg) % g == 0)
        jmax = (P - cg) // g
        tw, teeth = twisted_openings(q, g, jmax)
        assert np.array_equal(j_hits, tw), f"twisted machine mismatch g={g}"
        # (b) - class: n = -c_g + j' g is an opening iff j' is an opening of M_g^- whose teeth are the
        #     negatives of M_g^+'s teeth, i.e. M_g^- = -M_g^+ (j -> -j).  So the - class reads the same
        #     twisted cycle backwards from j = 0: (-j') mod P is an opening of M_g^+.
        jm = (minus + cg) // g
        assert np.all((minus + cg) % g == 0)
        gs_ = gears(q)
        mneg = np.zeros(int(jm.max()) + 1, dtype=bool)
        for h in gs_:
            t1, t2 = teeth[h]
            mneg[(-t1) % h::h] = True
            mneg[(-t2) % h::h] = True
        mneg[0] = True                                  # j' = 0 gives column -c_g < 1, outside the range
        assert np.array_equal(jm, np.nonzero(~mneg)[0]), f"M_g^- mismatch g={g}"
        # (c) full j-period of the twisted machine bijects onto O (mod P)
        tw_full, _ = twisted_openings(q, g, P - 1)
        img = (cg + tw_full.astype(np.int64) * g) % P
        assert len(set(img.tolist())) == len(tw_full) == len(O)
        assert set(img.tolist()) == set((O % P).tolist()), f"bijection fails g={g}"
        twset = set(tw_full.tolist())
        assert all(((-int(j)) % P) in twset for j in jm), f"- class not the backward read g={g}"
        # the two classes together read M_g^+ over j in (-(P+c_g)/g, (P-c_g)/g]: a 2/g fraction centred at j=0
        assert int(jm.max()) <= (P + cg) // g and int(j_hits.max()) <= (P - cg) // g
        # (d) spacing of consecutive + hits = g * twisted gap
        sp = np.diff(plus)
        assert np.all(sp % g == 0) and np.array_equal(sp // g, np.diff(tw))
        # twisted record (longest struck run of M_g^+ over its full period)
        mfull = np.ones(P, dtype=bool)
        mfull[tw_full] = False
        x = np.concatenate([mfull, mfull]).astype(np.int8)
        dx = np.diff(np.concatenate([[0], x, [0]]))
        F_tw[g] = int((np.nonzero(dx == -1)[0] - np.nonzero(dx == 1)[0]).max())
        if verbose_g and g == verbose_g:
            print(f"  g={g}: c_g={cg} d_g={d(g)} teeth of M_g^+ (mod h): {teeth}")
            print(f"  first + hits on openings: {plus[:14].tolist()}")
            print(f"  their j: {j_hits[:14].tolist()}  twisted gaps: {np.diff(tw)[:13].tolist()}")
            print(f"  spacing/g of + hits: {(sp // g)[:13].tolist()}")
            print(f"  widen-cluster-widen: max twisted gap in window = {int(np.diff(tw).max())} "
                  f"(-> {g}*{int(np.diff(tw).max())} columns with no + hit); "
                  f"runs of twisted gap 1 (hits every {g} columns): "
                  f"{int(((np.diff(tw) == 1)).sum())} adjacent pairs")
    same = [g for g in T if F_tw[g] == F]
    print(f"  twisted records F_tw(g) (full period): "
          f"{sorted(set(F_tw.values()))}; q's record F={F}; "
          f"gears with F_tw=F: {len(same)} of {len(T)} (check count)")
    print(f"  all checks (a)-(d) PASS for all g in T")


if __name__ == "__main__":
    check(11, verbose_g=13)
    check(13, verbose_g=17)
