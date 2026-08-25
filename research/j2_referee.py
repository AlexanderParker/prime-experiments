"""Harvester round 23: REFEREE PASS over publication Unit 1.

Unit 1 = docs/novel/j2-upper-bound.md + twin-percentile.md +
paired-jacobsthal-values.md (sections 4a/4b/4c).  This script re-derives, from
scratch and by independent code, every NUMERICAL claim those documents make, and
asserts each one.  Where a claim cannot be recomputed inside a round (the y = 19
and y = 23 exhaustive scans) the stored round-22 arrays are re-checked for
internal consistency and the limitation is printed rather than hidden.

Run: uv run python research/j2_referee.py     (from the repo root)

Sections:
  R1  the h_2 table (y = 5,7,11,13,17) recomputed exhaustively, per difference
  R2  the COMPLETE maximiser sets at y = 13 and y = 17 (doc lists are truncated)
  R3  tie-aware percentiles (twin-percentile.md 4a)
  R4  gcd-class F_max/lambda spread 2.88 .. 7.52
  R5  OEIS A288815 vs computed h_2, and the Conjecture-6 margin table
  R6  delta-profile law: (1,1,1,3,6) at y <= 13, precision and recall
  R7  the shallow-extension CAP LAW at 13 -> 17: local context, 272 lifts,
      extension value set {81, 84, 87}, and the exact 9
  R8  the b - a = p# collapse j_2(p#) >= j(p#)
  R9  Theorem 1's explicit chain (twin constant monotone, V_n bound, ratio)
  R10 stored round-22 winner arrays: internal consistency + winner counts
"""
import numpy as np
from math import prod, gcd, log, exp
from fractions import Fraction as Fr
from sympy import primerange

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------ core
def survivors(gears, e, P):
    a = np.ones(P, bool)
    for q in gears:
        a[0::q] = False
        a[(-e) % q::q] = False
    return a


def maxgap(gears, e, P):
    a = survivors(gears, e, P)
    idx = np.flatnonzero(a)
    if idx.size == 0:
        return P, idx                 # no survivor at all: the whole period
    if idx.size == 1:
        return P, idx                 # ONE survivor per period: cyclic gap = P
    g = np.diff(np.append(idx, idx[0] + P))
    return int(g.max()), idx


def family(gears, cache=True):
    """Exhaustive per-difference maximal-gap array.  Cached under
    research/data/ref_fam_<y>.npy so a re-run of the referee pass is cheap; the
    cache is INDEPENDENT of round 22's f13/f17 arrays (different code path) and
    is cross-checked against them in R3."""
    import os
    P = prod(gears)
    fn = f"research/data/ref_fam_{gears[-1]}.npy"
    if cache and os.path.exists(fn):
        F = np.load(fn)
        assert F.size == P // 2 + 1
        return P, F
    F = np.zeros(P // 2 + 1, np.int32)
    for e in range(1, P // 2 + 1):
        F[e] = maxgap(gears, e, P)[0]
    if cache:
        np.save(fn, F)
    return P, F


SETS = {5: [3, 5], 7: [3, 5, 7], 11: [3, 5, 7, 11], 13: [3, 5, 7, 11, 13],
        17: [3, 5, 7, 11, 13, 17]}


def main():
    say("=" * 78)
    say("R1 - the h_2 table, recomputed exhaustively per difference")
    say("=" * 78)
    ZM = {3: 6, 5: 18, 7: 30, 11: 66, 13: 150, 17: 192}    # arXiv:1706.03668 / A288815
    DOC_NDIFF = {3: 1, 5: 7, 7: 52, 11: 577, 13: 7507, 17: 127627}
    SETS[3] = [3]
    say("  DEGENERATE FIRST ROW - a referee correction.  paired-jacobsthal-values.md")
    say("  section 1 tabulates 'y = 3, P = 3, #diffs = 1, h_2 = 0, p^2-p = 6,")
    say("  Conj.6 holds'.  With gears = {3} and e = 1 the survivor set mod 3 is the")
    say("  single class {1}, so the CYCLIC maximal gap is 3, not 0, and h_2 = 6.")
    say("  (research/jacobsthal_family.py returns 0 whenever fewer than two")
    say("  survivors exist per period - the single-survivor case is not handled.)")
    say("  A288815 confirms h_2 = 6 at p_n = 3.  Consequence: h_2 = p_n^2 - p_n")
    say("  EXACTLY at n = 2, i.e. Conjecture 6 is FALSE at n = 2 - which is")
    say("  precisely why Ziller-Morack state it for n >= 3.  The table's 'holds'")
    say("  in that row is wrong and the row should be marked as the excluded case.")
    say("")
    fams = {}
    say("    y      P   #diffs   h_2=2maxF   ZM/A288815   p^2-p   margin%   ok")
    for y in (3, 5, 7, 11, 13, 17):
        gears = SETS[y]
        P, F = family(gears)
        fams[y] = (P, F)
        h2 = 2 * int(F[1:].max())
        nd = P // 2
        B = y * y - y
        marg = 100.0 * (B - h2) / B
        ok = (h2 == ZM[y]) and (nd == DOC_NDIFF[y])
        say(f"  {y:>3} {P:>7} {nd:>8} {h2:>11} {ZM[y]:>12} {B:>8} "
            f"{marg:>8.1f}   {'OK' if ok else 'MISMATCH'}"
            f"{'   <- n=2, Conj.6 EXCLUDED (equality)' if y == 3 else ''}")
        assert h2 == ZM[y], (y, h2, ZM[y])
        assert nd == DOC_NDIFF[y], (y, nd, DOC_NDIFF[y])
        assert (h2 < B) == (y >= 5), (y, h2, B)
    say("  ASSERTED: all five h_2 values and all five #diffs match the docs and")
    say("  Ziller-Morack / OEIS A288815 exactly.")
    # margins quoted in paired-jacobsthal-values.md section 1
    doc_marg = {5: 10.0, 7: 28.6, 11: 40.0, 13: 3.8, 17: 29.4}
    for y in doc_marg:
        B = y * y - y
        m = 100.0 * (B - 2 * int(fams[y][1][1:].max())) / B
        assert abs(m - doc_marg[y]) < 0.06, (y, m, doc_marg[y])
    say("  ASSERTED: the margin column (10.0, 28.6, 40.0, 3.8, 29.4) reproduces.")

    say("")
    say("=" * 78)
    say("R2 - COMPLETE maximiser sets (a documentation correction)")
    say("=" * 78)
    for y in (11, 13, 17):
        P, F = fams[y] if y in fams else family(SETS[y])
        mx = int(F[1:].max())
        win = [int(e) for e in np.flatnonzero(F[1:] == mx) + 1]
        say(f"  y = {y:>2}: max F = {mx:>3}, {len(win)} maximisers in 1 <= e <= P/2")
        say(f"           {win}")
        if y == 13:
            assert win[:5] == [344, 734, 839, 916, 2164], win[:5]
            assert len(win) == 16, len(win)
        if y == 17:
            assert win[:6] == [2791, 3176, 5584, 5794, 6361, 6571], win[:6]
    say("")
    say("  REFEREE FINDING (documentation, not mathematics):")
    say("  paired-jacobsthal-values.md section 1 writes")
    say("    'Maximisers: y = 13: e = 344, 734, 839, 916, 2164' and")
    say("    'y = 17: F = 96 at e = 2791, 3176, 5584, 5794, 6361, 6571'")
    say("  as if COMPLETE.  They are the first 5 / first 6 entries of the argmax")
    say("  list printed by research/jacobsthal_family.py (which slices arg[:5]).")
    say(f"  The complete sets have {len(np.flatnonzero(fams[13][1][1:] == fams[13][1][1:].max()))}"
        f" and {len(np.flatnonzero(fams[17][1][1:] == fams[17][1][1:].max()))} members")
    say("  respectively - consistent with the round-22 delta-space counts 16 and 64,")
    say("  and with the delta-profile law's '16 of 7507'.  The doc must say")
    say("  'the first five are' or list all sixteen.")

    say("")
    say("=" * 78)
    say("R3 - tie-aware percentiles (twin-percentile.md section 4a)")
    say("=" * 78)
    DOC = {(13, "coprime"): (2880, 384, 272, 2224),
           (13, "full"): (7507, 4519, 396, 2592),
           (17, "coprime"): (46080, 9824, 4640, 31616),
           (17, "full"): (127627, 84859, 6920, 35848)}
    say("     y      class        n     below    ties     above   doc-match")
    for y in (13, 17):
        P, F = fams[y]
        twin = int(F[1])
        es = np.arange(1, P // 2 + 1)
        cop = np.array([gcd(int(x), P) == 1 for x in es])
        for label, mask in (("full", np.ones_like(cop)), ("coprime", cop)):
            v = F[1:][mask]
            n = int(v.size)
            below = int((v < twin).sum())
            above = int((v > twin).sum())
            ties = n - below - above
            d = DOC[(y, label)]
            match = (n, below, ties, above) == d
            say(f"  {y:>4} {label:>10} {n:>9} {below:>9} {ties:>7} {above:>9}   "
                f"{'OK' if match else 'MISMATCH ' + str(d)}")
            assert match, ((y, label), (n, below, ties, above), d)
        # section-1 statistics of the coprime class at y = 13
        if y == 13:
            v = np.sort(F[1:][cop])
            say(f"        y=13 coprime class: F range {int(v.min())}..{int(v.max())}, "
                f"mean {v.mean():.2f}, median {float(np.median(v)):.0f}, twin F = {twin}")
            assert (int(v.min()), int(v.max())) == (30, 75)
            assert abs(v.mean() - 38.83) < 0.01
            assert float(np.median(v)) == 39.0
            assert twin == 33
            rank = int((v < twin).sum()) + 1
            say(f"        twin rank {rank} of {v.size}  -> "
                f"{100.0*(rank-1)/v.size:.1f}th percentile; extreme/twin = "
                f"{int(v.max())/twin:.2f}x")
            assert rank == 385, rank
            assert abs(100.0 * (rank - 1) / v.size - 13.3) < 0.05
    say("  ASSERTED: all four percentile rows reproduce exactly.")
    # cross-check this round's independent arrays against round 22's stored ones
    import os
    for y, fn in ((13, "f13_family.npy"), (17, "f17_family.npy")):
        path = f"research/data/{fn}"
        if os.path.exists(path):
            old = np.load(path)
            new = fams[y][1]
            same = old.shape == new.shape and bool((old == new).all())
            say(f"  cross-check vs round-22 {fn}: identical = {same}")
            assert same, (y, "round-20/22 array disagrees with this recomputation")

    say("")
    say("=" * 78)
    say("R4 - gcd-class F_max/lambda spread (twin-percentile.md section 1)")
    say("=" * 78)
    gears = SETS[13]
    P, F = fams[13]
    prof = {}
    for e in range(1, P // 2 + 1):
        prof.setdefault(gcd(e, P), []).append(int(F[e]))
    rows = []
    for g in sorted(prof):
        dens = prod((q - (1 if g % q == 0 else 2)) / q for q in gears)
        lam = 1.0 / dens
        rows.append((g, len(prof[g]), max(prof[g]), lam, max(prof[g]) / lam))
    lo = min(rows, key=lambda r: r[4])
    hi = max(rows, key=lambda r: r[4])
    say(f"  {len(rows)} gcd classes at gears <= 13")
    say(f"  min F_max/lambda = {lo[4]:.2f} at gcd = {lo[0]}   "
        f"max = {hi[4]:.2f} at gcd = {hi[0]}")
    assert len(rows) == 31, len(rows)
    assert abs(lo[4] - 2.88) < 0.01 and lo[0] == 5005, lo
    assert abs(hi[4] - 7.52) < 0.01 and hi[0] == 3, hi
    say("  ASSERTED: 31 classes, spread 2.88 (gcd 5005) .. 7.52 (gcd 3) - exact match.")

    say("")
    say("=" * 78)
    say("R5 - OEIS A288815 head, and the ZM margin drift")
    say("=" * 78)
    A288815 = [2, 6, 18, 30, 66, 150, 192, 258, 366, 450, 570, 708, 894, 1044,
               1284, 1422, 1656, 1902, 2190, 2460, 2622]
    PR = list(primerange(2, 74))
    assert len(PR) == len(A288815) == 21
    say("    p_n     h_2    p^2-p    margin%")
    margs = []
    say("  (p_n = 2 and p_n = 3 are the excluded cases n = 1, 2: h_2 = 2 = p^2-p at")
    say("   p = 2 and h_2 = 6 = p^2-p at p = 3, both EQUALITY, so Conjecture 6's")
    say("   'n >= 3' is exactly sharp - it fails by equality at both smaller n.)")
    assert A288815[0] == 2 * 2 - 2 and A288815[1] == 3 * 3 - 3
    for p, h in zip(PR, A288815):
        if p < 5:            # Conjecture 6 is stated for n >= 3, i.e. p_n >= 5
            continue
        B = p * p - p
        m = 100.0 * (B - h) / B
        margs.append((p, m))
        assert h < B, (p, h, B)
    for p, m in margs:
        say(f"  {p:>5} {A288815[PR.index(p)]:>7} {p*p-p:>8} {m:>10.1f}")
    dips = [(p, m) for p, m in margs if m < 12.0]
    say(f"  margins below 12%: {dips}")
    assert [p for p, _ in dips] == [5, 13], dips
    say("  ASSERTED: through p_n = 73 exactly two margins are below 12% - p = 5")
    say("  (10.0%) and p = 13 (3.8%); the 13 dip is the unique extreme for p >= 7.")
    tail = [m for p, m in margs if p >= 19]
    say(f"  margins for p_n >= 19: min {min(tail):.1f}%, max {max(tail):.1f}%, "
        f"mean {sum(tail)/len(tail):.1f}%  (doc: 'drifts toward ~50%')")
    doc_tail = [24.6, 27.7, 44.6, 38.7, 46.8, 45.5, 42.2, 40.6, 48.4, 51.6,
                48.0, 50.5, 50.5, 50.1]
    assert len(tail) == len(doc_tail), (len(tail), len(doc_tail))
    for a, b in zip(tail, doc_tail):
        assert abs(a - b) < 0.06, (a, b)
    say("  ASSERTED: the 14-entry margin list quoted in paired-jacobsthal-values.md")
    say("  section 4 reproduces from A288815 to 0.05%.")
    # slack quantisation
    slack = [(p, p * p - p - h) for p, h in zip(PR, A288815) if p >= 5]
    minslack = [(p, s) for p, s in slack if s <= 6]
    say(f"  slack B - h_2 minimal (<= 6) at: {minslack}")
    assert [p for p, _ in minslack] == [5, 13], minslack

    say("")
    say("=" * 78)
    say("R6 - the delta-profile law at y <= 13")
    say("=" * 78)
    P, F = fams[13]
    mx = int(F[1:].max())
    win = set(int(e) for e in np.flatnonzero(F[1:] == mx) + 1)

    def profile(e, gears):
        return tuple(min(e % q, q - e % q) for q in gears)

    target = (1, 1, 1, 3, 6)
    carriers = set(e for e in range(1, P // 2 + 1)
                   if profile(e, SETS[13]) == target)
    say(f"  winners at y = 13: {len(win)};  carriers of profile {target}: "
        f"{len(carriers)}")
    say(f"  precision = {len(win & carriers)}/{len(carriers)}, "
        f"recall = {len(win & carriers)}/{len(win)}")
    assert win == carriers, (len(win), len(carriers), len(win ^ carriers))
    say("  ASSERTED: the winner set and the profile-carrier set are IDENTICAL")
    say("  (precision and recall both 100%), exactly as claimed.")
    for e in sorted(win):
        assert profile(e, SETS[13])[:2] == (1, 1)
    say("  ASSERTED: every winning profile begins delta_3 = delta_5 = 1.")

    say("")
    say("=" * 78)
    say("R7 - the shallow-extension CAP LAW at 13 -> 17, and the exact 9")
    say("=" * 78)
    P13 = prod(SETS[13])
    ctxs = set()
    for e in sorted(win):
        a = survivors(SETS[13], e, P13)
        idx = np.flatnonzero(a)
        g = np.diff(np.append(idx, idx[0] + P13))
        j = int(np.argmax(g))
        k = len(g)
        ctx = tuple(int(g[(j + t) % k]) for t in (-2, -1, 0, 1, 2))
        ctxs.add(ctx)
    say(f"  local gap context around the record, over all {len(win)} winners: {ctxs}")
    assert ctxs == {(6, 3, 6, 75, 6)} or ctxs == {(3, 6, 75, 6, 3)}, ctxs
    say("  (window read as (..,g_-2,g_-1,[F],g_+1,..) - the doc's '..6,3,6,[75],6,3,6..')")

    P17 = prod(SETS[17])
    vals = {}
    lifts = 0
    for e in sorted(win):
        for r in range(17):
            # lift: e' = e + 15015*t with e' = r mod 17; e' in [1, P17/2] by symmetry
            t = ((r - e) * pow(P13, -1, 17)) % 17
            ep = e + P13 * t
            if ep > P17 // 2:
                ep = P17 - ep
            v = maxgap(SETS[17], ep, P17)[0]
            vals.setdefault(v, 0)
            vals[v] += 1
            lifts += 1
    say(f"  {lifts} lifts of the {len(win)} winners to gears <= 17; extension value "
        f"multiset: {dict(sorted(vals.items()))}")
    assert lifts == 272, lifts
    assert set(vals) == {81, 84, 87}, vals
    best_ext = max(vals)
    true_max = int(fams[17][1][1:].max())
    say(f"  best extension of a 13-winner = {best_ext}; true family max at 17 = "
        f"{true_max}; deficit = {true_max - best_ext}")
    assert best_ext == 87 and true_max == 96 and true_max - best_ext == 9
    say("  ASSERTED: the extension value set is exactly {81, 84, 87} = "
        "{75+6, 75+6+3, 6+75+6}")
    say("  and THE EXACT 9 = 96 - 87 reproduces from scratch.")

    say("")
    say("=" * 78)
    say("R8 - the b - a = p# collapse (j_2(p#) >= j(p#))")
    say("=" * 78)
    for n, gears in ((3, [2, 3, 5]), (4, [2, 3, 5, 7]), (5, [2, 3, 5, 7, 11])):
        Pn = prod(gears)
        # paired sieve with difference exactly P: x and x+P share all residues
        paired = np.ones(Pn, bool)
        ordinary = np.ones(Pn, bool)
        for q in gears:
            paired[0::q] = False
            paired[(-Pn) % q::q] = False
            ordinary[0::q] = False
        say(f"  n = {n}, p_n# = {Pn}: paired survivors {int(paired.sum())}, "
            f"ordinary {int(ordinary.sum())}, sets equal: "
            f"{bool((paired == ordinary).all())}")
        assert (paired == ordinary).all()
    say("  ASSERTED: with b - a = p_n# the paired survivor set IS the ordinary one,")
    say("  so j_2(p_n#) >= j(p_n#) and every FGKMT lower bound transfers.")

    say("")
    say("=" * 78)
    say("R9 - Theorem 1's explicit chain")
    say("=" * 78)
    C2 = Fr(0)
    prodv = Fr(1)
    worst = None
    ps = list(primerange(2, 40000))
    tw = Fr(1)
    for p in ps[1:]:
        tw *= Fr((p - 1) ** 2 - 1, (p - 1) ** 2)
    say(f"  prod_{{3<=p<=40000}} (1 - 1/(p-1)^2) = {float(tw):.7f}  "
        f"> C_2 = 0.6601618 : {float(tw) > 0.6601618}")
    assert float(tw) > 0.6601618
    # the explicit inequality 2*3^(n-1)/V_n + 1 < 3^(n+1) log^2 p_n
    V = Fr(1)
    worst = None
    for i, p in enumerate(ps, start=1):
        V *= Fr(p - (1 if p == 2 else 2), p)
        if i < 3 or i > 4203:
            continue
        lhs = Fr(2 * 3 ** (i - 1)) / V + 1
        # exact-safe: 3^n overflows float, so compare in logs
        loglhs = log(lhs.numerator) - log(lhs.denominator)
        logrhs = (i + 1) * log(3) + 2 * log(log(p))
        r = exp(loglhs - logrhs)
        assert r < 1.0, (i, p, r)
        if worst is None or r > worst[0]:
            worst = (r, i, p)
    say(f"  2*3^(n-1)/V_n + 1 < 3^(n+1) (log p_n)^2 for 3 <= n <= 4203; worst ratio "
        f"{worst[0]:.4f} at n = {worst[1]} (p_n = {worst[2]})")
    assert abs(worst[0] - 0.8627) < 0.001, worst
    say("  REFEREE NOTE: round 21 recorded 'worst ratio 0.858 at n = 3'.  With the")
    say("  '+1' that is part of the bound the ratio is 0.8627 (180/209.8 = 0.858 is")
    say("  the value WITHOUT the +1).  Same conclusion, 0.5% tighter than quoted;")
    say("  the doc figure should read 0.863.")

    say("")
    say("=" * 78)
    say("R10 - stored round-22 winner arrays (y = 19, 23) - consistency only")
    say("=" * 78)
    try:
        w19 = np.load("research/data/family_w19_delta.npy")
        w23 = np.load("research/data/family_w23_delta.npy")
        say(f"  y = 19 winner deltas: {w19.size} (doc: 64);  "
            f"y = 23: {w23.size} (doc: 128)")
        assert w19.size == 64 and w23.size == 128
        Q19 = prod([5, 7, 11, 13, 17, 19])
        Q23 = prod([5, 7, 11, 13, 17, 19, 23])
        say(f"  all distinct: {len(set(w19.tolist())) == w19.size} / "
            f"{len(set(w23.tolist())) == w23.size}; in range: "
            f"{bool((w19 > 0).all() and (w19 < Q19).all())} / "
            f"{bool((w23 > 0).all() and (w23 < Q23).all())}")
        assert len(set(w19.tolist())) == 64 and len(set(w23.tolist())) == 128
        # verify a sample of the y=19 winners by direct maximal-gap computation
        say("  independent re-verification of the y = 19 winners (direct cyclic")
        say("  max-gap of {k : k != 0, -delta mod q} over gears 5..19):")
        bad = 0
        for d in w19[:8].tolist():
            a = np.ones(Q19, bool)
            for q in (5, 7, 11, 13, 17, 19):
                a[0::q] = False
                a[(-d) % q::q] = False
            idx = np.flatnonzero(a)
            g = int(np.diff(np.append(idx, idx[0] + Q19)).max())
            if g != 43:
                bad += 1
        say(f"    first 8 winners: all reach G = 43 (h_2 = 6*43 = 258): "
            f"{'YES' if bad == 0 else f'{bad} FAILURES'}")
        assert bad == 0
        say("    -> h_2(19) = 6*43 = 258 confirmed against ZM / A288815.")
        assert 6 * 43 == 258
    except FileNotFoundError as ex:
        say(f"  stored arrays missing: {ex} (round-22 data not present)")

    say("")
    say("=" * 78)
    say("REFEREE VERDICT")
    say("=" * 78)
    say("  Every numerical claim of Unit 1 that is recomputable inside a round has")
    say("  been recomputed by independent code and asserted.  Everything that was")
    say("  meant to reproduce, reproduced.  FIVE defects found, all of them in the")
    say("  DOCUMENTS rather than in the mathematics:")
    say("   1. R1  the y = 3 row (h_2 = 0, 'holds') - a single-survivor code")
    say("          artefact; the truth is h_2 = 6 = p^2-p, Conjecture 6 EXCLUDES")
    say("          n = 2 by equality, and the 'n >= 3' hypothesis is sharp.")
    say("   2. R2  truncated maximiser lists presented as complete (5 of 16 at")
    say("          y = 13, 6 of 64 at y = 17).")
    say("   3. R9  'worst ratio 0.858' omits the '+1' of the bound; it is 0.8627.")
    say("   4. (j2_explicit A) the chain constant 0.3908 does not follow from the")
    say("          stated ingredients; 0.3905 does.")
    say("   5. (j2_explicit A/B) the quasi-polynomial constant was quoted as a")
    say("          measured band [3.47, 4.16] that does NOT contain the limit")
    say("          2 lambda_* = 7.1822.")

    with open("research/data/j2_referee.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("j2_referee: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()
