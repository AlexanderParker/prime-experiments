"""twogap_table.py - Constructor round 24, item 1.

THE TWO-GAP STATEMENT, exactly, at every machine with a full-period lag-1
gap-pair census: is F_2(M) <= F(M) + q' ?  And what can bound it?

Three candidate suppliers, all computed from EXACT full-period data:

  (H)  the HISTOGRAM bound   F_2 <= F + G_2,  G_2 = the largest gap value that
       can accompany a gap of size F in the multiset (= F if W_1(F) >= 2, else
       the next value down).  This is the TIGHT bound over all cyclic
       rearrangements of the gap multiset, hence the best any function of the
       gap histogram can give - and by Lateral's Jordan=histogram theorem
       (docs/novel/nilpotent-invariants.md) the best any UNITARY INVARIANT of
       the blocked-walk operator N = BS can give.

  (A)  the ADJACENCY bound   F_2 <= F + A(M),  A(M) = max over adjacent gap
       pairs of min(g1,g2)  ("how large can the smaller of two adjacent gaps
       be").  A(M) <= q' would give the two-gap statement outright.

  (S)  the plain slack       F + q' - F_2.

Data: research/data/gap_pair_joint.csv (Mechanic, full period, lag 1) and
gap_pair_hist.csv (ghist rows).  Machines 11..31 are full period
(coverage 1.000000); machine 37 rows are partial and are reported separately
as LOWER bounds only, never as values.

Every claim below is asserted.
"""
import csv, collections, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

NEXTP = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41}


def load_ghist():
    h = collections.defaultdict(dict)  # (y,cov) -> {gap: count}
    for r in csv.DictReader(open(os.path.join(DATA, "gap_pair_hist.csv"))):
        if r["kind"] != "ghist":
            continue
        h[(int(r["y"]), r["coverage"])][int(r["value"])] = int(r["count"])
    return h


def load_pairs(lag=1):
    p = collections.defaultdict(dict)  # (y,cov) -> {(gu,gv): count}
    for r in csv.DictReader(open(os.path.join(DATA, "gap_pair_joint.csv"))):
        if int(r["lag"]) != lag:
            continue
        p[(int(r["y"]), r["coverage"])][(int(r["gu"]), int(r["gv"]))] = int(r["count"])
    return p


def g2_of(hist):
    """Largest value pairable with F in the multiset (tight histogram bound)."""
    F = max(hist)
    if hist[F] >= 2:
        return F
    rest = [g for g in hist if g < F]
    return max(rest) if rest else 0


def main():
    gh = load_ghist()
    pj = load_pairs(1)
    full = sorted(k for k in gh if k[1] == "1.000000")
    part = sorted(k for k in gh if k[1] != "1.000000")

    print("FULL-PERIOD MACHINES  (exact)")
    print(" y  q'    F   F_2  F+q'  slack | pair@F_2   A(M)  A<=q'? | G_2  F+G_2  H-margin")
    rows = []
    for key in full:
        y, cov = key
        hist = gh[key]
        pairs = pj[key]
        assert pairs, key
        F = max(hist)
        qp = NEXTP[y]
        G2 = g2_of(hist)
        # DATA TRAP: the joint census is a LINEAR scan - it has P-1 pairs, not
        # P, so the CYCLIC SEAM pair (last gap, first gap) is missing.  It is
        # recoverable exactly: the left-marginal is short by one at the LAST
        # gap's value, the right-marginal short by one at the FIRST gap's.
        mu = collections.Counter()
        mv = collections.Counter()
        for (gu, gv), c in pairs.items():
            mu[gu] += c
            mv[gv] += c
        du = [g for g in hist if hist[g] - mu[g] != 0]
        dv = [g for g in hist if hist[g] - mv[g] != 0]
        assert len(du) == 1 and len(dv) == 1, (y, du, dv)
        assert hist[du[0]] - mu[du[0]] == 1 and hist[dv[0]] - mv[dv[0]] == 1, (y, du, dv)
        seam = (du[0], dv[0])
        pairs = dict(pairs)
        pairs[seam] = pairs.get(seam, 0) + 1
        mu[seam[0]] += 1
        mv[seam[1]] += 1
        assert dict(mu) == hist and dict(mv) == hist, (y, "marginal != ghist")
        # F_2 and its attaining pair - computed AFTER the seam is stitched
        F2, arg = max(((gu + gv, (gu, gv)) for (gu, gv) in pairs), key=lambda t: t[0])
        A = max(min(gu, gv) for (gu, gv) in pairs)
        # sanity: F is a realised gap and appears in some adjacent pair
        assert any(gu == F or gv == F for (gu, gv) in pairs)
        # soundness of the two candidate bounds
        assert F2 <= F + G2, (y, F2, F, G2)
        assert F2 <= F + A, (y, F2, F, A)
        rows.append(dict(y=y, qp=qp, seam=seam, F=F, F2=F2, budget=F + qp, slack=F + qp - F2,
                         arg=arg, A=A, G2=G2, wF=hist[F], ngaps=sum(hist.values())))
        print(f"{y:3d} {qp:3d} {F:5d} {F2:5d} {F+qp:5d} {F+qp-F2:6d} | "
              f"{str(arg):>9s} {A:5d}  {'YES' if A <= qp else 'NO ':>3s}  | "
              f"{G2:4d} {F+G2:5d}  {F + qp - (F + G2):+6d}")

    print()
    print("VERDICTS")
    # (1) the two-gap statement itself
    bad = [r for r in rows if r["F2"] > r["budget"]]
    print(f"  two-gap statement F_2 <= F + q'      : holds {len(rows)-len(bad)}/{len(rows)}"
          + (f"  FAILS at {[r['y'] for r in bad]}" if bad else ""))
    assert not bad
    # (2) histogram bound
    hok = [r for r in rows if r["F"] + r["G2"] <= r["budget"]]
    print(f"  histogram bound F+G_2 <= F + q'      : holds {len(hok)}/{len(rows)}"
          f"  (fails at {[r['y'] for r in rows if r not in hok]})")
    # (3) adjacency bound
    aok = [r for r in rows if r["A"] <= r["qp"]]
    print(f"  adjacency law   A(M) <= q'           : holds {len(aok)}/{len(rows)}"
          f"  (fails at {[r['y'] for r in rows if r not in aok]})")
    # (4) the mirror reading of G_2: maximal gaps come in mirror pairs, so
    #     W_1(F) >= 2 and the histogram bound is EXACTLY 2F.
    print()
    print("  W_1(F) (multiplicity of the maximal gap) and the 2F identity:")
    for r in rows:
        print(f"    y={r['y']:3d}  W_1(F) = {r['wF']:3d}  (of {r['ngaps']} gaps)"
              f"   F+G_2 = {r['F']+r['G2']:4d}   2F = {2*r['F']:4d}"
              f"   {'EQUAL' if r['F']+r['G2'] == 2*r['F'] else 'DIFFER'}")
    assert all(r["wF"] >= 2 for r in rows), "mirror pairing of maximal gaps"
    assert all(r["F"] + r["G2"] == 2 * r["F"] for r in rows)
    # (5) cross-check against R52's machine-free corridor layer-0 column
    R52_LAYER0 = {11: 14, 13: 21, 17: 36, 19: 50, 23: 67, 29: 86, 31: 116}
    print()
    print("  CROSS-CHECK: R52's machine-free CORRIDOR layer-0 bound vs the"
          " histogram bound 2F")
    for r in rows:
        c = R52_LAYER0[r["y"]]
        print(f"    y={r['y']:3d}  corridor layer 0 = {c:4d}   2F = {2*r['F']:4d}"
              f"   diff {c-2*r['F']:+d}")
    print()
    print("  ratios  F_2/F  vs  required 1 + q'/F:")
    for r in rows:
        print(f"    y={r['y']:3d}  F_2/F = {r['F2']/r['F']:.4f}   1+q'/F = "
              f"{1+r['qp']/r['F']:.4f}   (F+G_2)/F = {(r['F']+r['G2'])/r['F']:.4f}"
              f"   A/q' = {r['A']/r['qp']:.4f}")

    # (6) HOW MUCH OF THE TWO-GAP STATEMENT IS COUNTING, AND HOW MUCH IS
    #     STRUCTURE?  Take the SAME gap multiset and arrange it uniformly at
    #     random on the cycle.  The expected number of adjacent ordered pairs
    #     with sum > B is  E(B) = (1/(n-1)) * #{(i,j), i != j : g_i+g_j > B},
    #     so the typical max adjacent pair sum is  R_2 = min{B : E(B) < 1}.
    #     This is a HISTOGRAM-ONLY quantity, but a typical-case one rather than
    #     the worst-case F + G_2 - and it is the honest question the worst-case
    #     bound hides: does the machine need any anti-correlation at all, or is
    #     the two-gap statement a pure counting fact about the tail?
    print()
    print("  RANDOM-ARRANGEMENT CONTROL (same histogram, uniform cyclic order)")
    print("     y   F_2  R_2(typ)  F+q'   R_2 <= F+q'?   F_2 - R_2")
    for key in full:
        y, cov = key
        hist = gh[key]
        r = [z for z in rows if z["y"] == y][0]
        vals = sorted(hist)
        cnt = [hist[v] for v in vals]
        n = sum(cnt)
        # count ordered pairs (i != j) with g_i + g_j > B, exactly
        import bisect
        suf = [0] * (len(vals) + 1)
        for i in range(len(vals) - 1, -1, -1):
            suf[i] = suf[i + 1] + cnt[i]
        R2 = None
        for B in range(2 * r["F"], 1, -1):
            tot = 0
            for i, v in enumerate(vals):
                j = bisect.bisect_right(vals, B - v)
                tot += cnt[i] * suf[j]
                if B - v == v:              # remove i == j self-pairs
                    pass
            # subtract ordered pairs using the same element twice
            same = sum(c for v, c in zip(vals, cnt) if 2 * v > B)
            tot -= same
            E = tot / (n - 1)
            if E >= 1:
                R2 = B
                break
        r["R2"] = R2
        print(f"    {y:3d} {r['F2']:5d} {R2:9d} {r['budget']:5d}"
              f"      {'YES' if R2 <= r['budget'] else 'NO ':>3s}"
              f"        {r['F2'] - R2:+5d}")
    print("     (R_2 is what the histogram alone predicts for a TYPICAL"
          " arrangement;\n      F + G_2 = 2F is what it forces in the WORST"
          " arrangement.)")

    print()
    print("PARTIAL-COVERAGE MACHINE 37 (lower bounds only - never a value)")
    for key in part:
        y, cov = key
        hist = gh[key]
        pairs = pj[key]
        F = max(hist)
        qp = NEXTP[y]
        F2 = max(gu + gv for (gu, gv) in pairs)
        A = max(min(gu, gv) for (gu, gv) in pairs)
        G2 = g2_of(hist)
        print(f"  y=37 cov={cov}  F>={F}  F_2>={F2}  A>={A}  G_2>={G2}  "
              f"(F+q'={F+qp} using the observed F as if exact - NOT a bound)")

    # corpus-known exact values beyond the scan, for the trend only
    print()
    print("CORPUS-KNOWN EXACT VALUES BEYOND THE PAIR CENSUS (see mechanic.md):")
    known = [(37, 88, 90, 41), (41, 97, 103, 43)]
    for y, F, F2, qp in known:
        print(f"  y={y:3d}  F={F}  F_2={F2}  F+q'={F+qp}  slack={F+qp-F2}"
              f"   F_2/F={F2/F:.4f}  1+q'/F={1+qp/F:.4f}")
    print()
    print("OK - all assertions passed")


if __name__ == "__main__":
    main()
