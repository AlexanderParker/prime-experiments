"""Q2: one gear on one run.

(A) g > span S of a run => g strikes at most two openings of the run.
(B) exactly two => their distance D is d_g or g - d_g, i.e. g | 3D-1 or 3D+1.
(C) hence g > 3S+1 strikes at most one opening of the run; for ADJACENT openings (gap <= F+1)
    g > 3F+4 never strikes both.  Exact threshold is 3F+4, not 3F+3: at q=23, 3F+4 = 103 is prime.
(D) the strong reading 'g > 3F+3 strikes at most one opening of any run' is REFUTED (non-adjacent
    openings of a short run at distance d_g or g-d_g).
"""
import numpy as np
from math import isqrt
from lane_common import gears, period, c, d, openings, record, T_primes, struck_mask, primes_upto


def check(q):
    P = period(q)
    O = openings(q)
    F = record(q)
    T = T_primes(q)
    gaps = np.diff(O)
    print(f"q={q} P={P} |O|={len(O)} F={F} max gap={int(gaps.max())} (=F+1) T={T[0]}..{T[-1]} ({len(T)})")
    thr_brief = 3 * F + 3
    thr_exact = 3 * F + 4
    adj_double = {}
    same_class = []
    refute_strong = None
    for g in T:
        cg = c(g)
        r = O % g
        s = (r == cg) | (r == (g - cg) % g)
        # (B) consecutive struck openings at distance < g: distance in {d_g, g-d_g}
        struck = O[s]
        D = np.diff(struck)
        close = D[D < g]
        dg = d(g)
        assert np.all((close == dg) | (close == g - dg)), f"(B) fails g={g}"
        assert np.all(((3 * close - 1) % g == 0) | ((3 * close + 1) % g == 0))
        # adjacent openings both struck
        both = s[:-1] & s[1:]
        if both.any():
            ds = sorted(set(gaps[both].tolist()))
            adj_double[g] = ds
            for dd in ds:
                # leg rule as stated (g | 3d-1 or 3d+1) fails when d = g (same class, g <= F+1)
                assert (3 * dd - 1) % g == 0 or (3 * dd + 1) % g == 0 or dd % g == 0, "leg rule"
                if dd % g == 0:
                    i0 = np.nonzero(both & (gaps == dd))[0][0]
                    same_class.append((g, dd, int(O[i0]), int(O[i0 + 1])))
            assert g <= thr_exact, f"gear {g} > 3F+4 strikes two adjacent openings"
        # (A) any run of span S < g contains at most two struck openings: windows of O of
        # length < g.  For each struck opening, count struck openings within (n, n+g): <= 1.
        if len(struck) > 2:
            nxt2 = struck[2:] - struck[:-2]
            assert np.all(nxt2 >= g), f"(A) fails g={g}"
        # (D) instance: g > 3F+4 striking two openings of a run with an opening between
        if refute_strong is None and g > thr_exact and len(struck) > 1:
            idx = np.nonzero(D < g)[0]
            for i in idx:
                n1, n2 = int(struck[i]), int(struck[i + 1])
                between = O[(O > n1) & (O < n2)]
                if len(between) > 0:
                    refute_strong = (g, n1, n2, between.tolist())
                    break
    print(f"  (A),(B) PASS for all g in T; gears striking two ADJACENT openings: {adj_double}")
    if same_class:
        print(f"  leg rule 'iff g | 3d-1 or 3d+1' REFUTED for g <= F+1: same-class closures (g, d=g, n, n+g): {same_class}")
    print(f"  all such gears <= 3F+4 = {thr_exact}; largest = {max(adj_double)}; "
          f"3F+3 = {thr_brief}, 3F = {3*F}; leg primes of gap F+1={F+1}: 3(F+1)-1={3*F+2}, 3(F+1)+1={3*F+4}")
    if refute_strong:
        g, n1, n2, b = refute_strong
        print(f"  (D) REFUTES 'one per run above 3F+3': g={g} > {thr_exact} strikes openings {n1} and {n2} "
              f"(distance {n2-n1} = {'d_g' if n2-n1==d(g) else 'g-d_g'}, d_g={d(g)}) with openings {b} between: "
              f"a run of {len(b)+2} consecutive openings, span {n2-n1}, two struck by one gear above 3F+4")


def check_q23_threshold():
    q = 23
    P = period(q)
    m = struck_mask(gears(q), P)
    O = np.nonzero(~m[1:])[0] + 1
    gaps = np.diff(O)
    F = int(gaps.max()) - 1
    print(f"q=23 P={P} |O|={len(O)} F={F}; 3F+2={3*F+2}, 3F+3={3*F+3}, 3F+4={3*F+4}")
    idx = np.nonzero(gaps == F + 1)[0]
    ends = [(int(O[i]), int(O[i + 1])) for i in idx]
    print(f"  gaps of F+1={F+1} between consecutive openings: {len(ends)} (check count); endpoints {ends[:6]}...")
    for g in (3 * F + 2, 3 * F + 4):
        cg = c(g)
        hits = [(a, b) for a, b in ends if (a % g in (cg, g - cg)) and (b % g in (cg, g - cg))]
        print(f"  g={g} (prime={g in primes_upto(g)}): strikes both ends of a gap {F+1} at {hits[:4]} "
              f"-> {'YES: two adjacent openings struck by a gear above 3F+3' if hits else 'no'}")
    # verify no prime g > 3F+4 strikes two adjacent openings (only leg primes can: 3d+-1 <= 3F+4)
    print("  proof: adjacent openings at gap d<=F+1 both struck => g | 3d-1 or 3d+1 <= 3F+4, so g <= 3F+4.")


if __name__ == "__main__":
    for q in (13, 17, 19):
        check(q)
    check_q23_threshold()
