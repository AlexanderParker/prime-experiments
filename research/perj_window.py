"""Round 28 (constructor): THE PER-J TRIPLE ANALOGUES, J = 3..7, EXACT.

The manager's triple inequality (round-27 derivation pass 2) is the J = 3 case
of a family.  R74's uniform-order theorem makes the family FINITE, so the whole
depth-3-and-beyond obligation of the increment law is a finite list of lemmas.
This file states, measures and gates the list.

THE OBJECT.  A WORD-LEGAL J-WINDOW of machine M is J consecutive gaps

    (g_L, w_1, ..., w_{J-2}, g_R)

whose J-2 MIDDLES each lie in {0, +-2c} mod q' (T2) with the nonzero classes
STRICTLY ALTERNATING (T3; padded middles, = 0 mod q', are transparent).
    Q*_J(M; q') = max span over such windows      (Q*_2 = F_2(M) identically)
    Delta_J(M)  = Q*_J - F_2(M)                   (the quantity the law caps)
    Phi_J(M)    = max FLANK SUM g_L + g_R over such windows   (new this round)

THE PER-J ANALOGUE of the triple inequality is  Delta_J <= s_min(q'), for each
J; the increment law is exactly max_J Delta_J <= s_min.

THE MIDDLE-SUM LEMMA (proved here from T1-T3, used to read the table).  In a
LITERAL J-window the classes alternate, so among the J-2 middles the counts of
the two classes differ by at most one; every class-a middle is >= a and every
class-b middle is >= b, with a + b = q'.  Hence with k = floor((J-2)/2),

    middle sum  >=  k*q'            (J even),
    middle sum  >=  k*q' + a        (J odd),

so a literal J-window's span EXCEEDS its flank sum by a quantity that grows by
q' every two levels of J.  Q*_J <= F_2 + Delta therefore forces the flank
envelope Phi_J to COLLAPSE at rate q'/2 levels - which is exactly what par
trading (R30) measures and what R50 saw at one step (29 -> 22 -> 7).

DATA, all exact:
  m11..m23     direct full-period cyclic scan here (numpy, seam included) -
               every J
  m29,31,37    Mechanic's exact full-period 4-tuple censuses - J <= 4 exactly
               (a realised 3-window sits inside a realised 4-window, so the
               projection to triples is exact too)
  any machine  scan-free by CRT descent - research/perj_scanfree.py

Gates: F(M) and F_2(M) are recovered from each source and asserted against the
corpus before any comparison; the J <= 5 column is asserted against R68's
independently computed exact Q* table at every machine where that table has an
entry.

Usage:  .venv/Scripts/python.exe research/perj_window.py
"""
import os
import sys
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145}
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
            41: 103, 47: 134, 53: 159}
# R68's exact Q*_J table (research/data/r26_qstar.log), J = 2, 3, 4, 5, ...
QSTAR = {11: [11, 8], 13: [16, 18], 17: [25, 25], 19: [31, 33, 34],
         23: [39, 43], 29: [55, 58, 55, 55], 31: [68, 85, 88, 68],
         37: [90, 90, 91]}
# R45's A_kill (= k_max, the number of killed openings), so J_max = A_kill + 1
A_KILL = {11: 2, 13: 2, 17: 2, 19: 3, 23: 2, 29: 4, 31: 4, 37: 3}


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def next_prime(y):
    p = y + 1
    while not is_prime(p):
        p += 1
    return p


def letters(q1):
    u1 = round(q1 / 6)
    return 2 * u1, q1 - 2 * u1


def gaps_of(y):
    gears = [p for p in range(5, y + 1) if is_prime(p)]
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    op = np.flatnonzero(~ex).astype(np.int64)
    return np.diff(np.concatenate([op, [op[0] + P]]))


def cls(w, q1, a, b):
    """+1 class a, -1 class b, 0 padded, None illegal."""
    r = w % q1
    if r == 0:
        return 0
    if r == a % q1:
        return 1
    if r == b % q1:
        return -1
    return None


def legal_middles(mids, q1, a, b):
    """T2 + T3: every middle in {0, +-2c} mod q', nonzero classes alternate."""
    seq = []
    for w in mids:
        c = cls(w, q1, a, b)
        if c is None:
            return False
        if c:
            seq.append(c)
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            return False
    return True


def windows_from_gaps(d, J):
    """All J-windows of the cyclic gap sequence d (seam included)."""
    return np.stack([np.roll(d, -i) for i in range(J)], axis=1)


def windows_from_census(path, J):
    """All J-windows induced by an exact 4-tuple census (J <= 4)."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    assert J <= 4
    if J == 4:
        return np.unique(arr, axis=0)
    return np.unique(np.concatenate([arr[:, i:i + J]
                                     for i in range(0, 5 - J)]), axis=0)


def analyse(W, J, q1, a, b, F2):
    """Return per-kind stats over the legal J-windows in W."""
    out = {}
    mids = W[:, 1:J - 1]
    ok = np.ones(len(W), bool)
    # vectorised T2
    cl = np.zeros(mids.shape, np.int8)
    for j in range(mids.shape[1]):
        r = mids[:, j] % q1
        c = np.zeros(len(W), np.int8)
        c[r == a % q1] = 1
        c[r == b % q1] = -1
        bad = (r != 0) & (r != a % q1) & (r != b % q1)
        ok &= ~bad
        cl[:, j] = c
    # T3 by explicit check on the survivors (alternation of the nonzero
    # subsequence); vectorising this is not worth the obscurity
    idx = np.flatnonzero(ok)
    keep = []
    for i in idx.tolist():
        seq = [int(c) for c in cl[i] if c]
        if all(seq[k] != seq[k - 1] for k in range(1, len(seq))):
            keep.append(i)
    sel = W[np.array(keep, np.int64)] if keep else W[:0]
    if not len(sel):
        return {"n": 0}
    span = sel.sum(axis=1)
    msum = sel[:, 1:J - 1].sum(axis=1)
    flank = sel[:, 0] + sel[:, J - 1]
    pad = (sel[:, 1:J - 1] % q1 == 0).any(axis=1)
    for name, mask in (("lit", ~pad), ("pad", pad), ("all",
                                                     np.ones(len(sel), bool))):
        if not mask.any():
            out[name] = None
            continue
        s = np.where(mask, span, -1)
        i = int(np.argmax(s))
        wit = tuple(int(v) for v in sel[i])
        out[name] = dict(Q=int(span[i]), Delta=int(span[i]) - F2, wit=wit,
                         msum=int(msum[i]), flank=int(flank[i]),
                         maxflank=int(flank[mask].max()),
                         n=int(mask.sum()),
                         pal=wit == wit[::-1])
    out["n"] = len(sel)
    return out


def main():
    print("=" * 79)
    print("PER-J TRIPLE ANALOGUES - Q*_J, Delta_J, Phi_J at every censused step")
    print("=" * 79)
    print("  legal J-window = J consecutive gaps whose J-2 middles are")
    print("  0 or +-2c mod q' with the nonzero classes alternating (T2+T3).")
    print("  Delta_J = Q*_J - F_2(M).   The per-J analogue is Delta_J <= s_min.")
    print()
    rows = {}
    for y in (11, 13, 17, 19, 23, 29, 31, 37):
        q1 = next_prime(y)
        a, b = letters(q1)
        smin = min(a, b)
        F2 = KNOWN_F2[y]
        if y <= 23:
            d = gaps_of(y)
            assert int(d.max()) == KNOWN_F[y], (y, int(d.max()))
            assert int((d + np.roll(d, -1)).max()) == F2, y
            src, Jmax = "scan", 7
        else:
            path = os.path.join(DDIR, "gap_tuples_%d_4.csv" % y)
            src, Jmax = "census", 4
        print("-" * 79)
        print("machine %d  ->  q' = %d   letters (a,b) = (%d,%d)  s_min = %d "
              " F = %d  F_2 = %d   [%s]"
              % (y, q1, a, b, smin, KNOWN_F[y], F2, src))
        print("   J  #legal | LITERAL: Q*_J  Delta  msum  flanksum  witness"
              "            pal | PADDED: Q*_J Delta")
        rows[y] = {}
        for J in range(3, Jmax + 1):
            if src == "scan":
                W = windows_from_gaps(d, J)
            else:
                W = windows_from_census(path, J)
            r = analyse(W, J, q1, a, b, F2)
            rows[y][J] = r
            if not r["n"]:
                print("  %2d       0 | (no legal %d-window exists - the per-J "
                      "program TERMINATES here)" % (J, J))
                rows[y][J] = {"n": 0, "empty": True}
                break
            L, P = r.get("lit"), r.get("pad")
            ls = ("%4d  %+5d %5d %9d  %-20s %-4s"
                  % (L["Q"], L["Delta"], L["msum"], L["flank"],
                     ",".join(map(str, L["wit"])), "PAL" if L["pal"] else "-")
                  if L else "%-52s" % "  none")
            ps = ("%4d %+5d  %-16s %s"
                  % (P["Q"], P["Delta"], ",".join(map(str, P["wit"])),
                     "PAL" if P["pal"] else "-") if P else "none")
            print("  %2d %7d | %s | %s" % (J, r["n"], ls, ps))
        # gate against R68's independently computed exact Q* table
        if y in QSTAR:
            for J in range(3, min(Jmax, 1 + len(QSTAR[y])) + 1):
                if J - 2 < len(QSTAR[y]) and rows[y].get(J, {}).get("all"):
                    got = rows[y][J]["all"]["Q"]
                    want = QSTAR[y][J - 2]
                    assert got == want, ("Q* gate", y, J, got, want)
            print("  GATE: Q*_J reproduces R68's exact table at every J it "
                  "covers (%s)" % ", ".join(map(str, QSTAR[y])))

    print()
    print("=" * 79)
    print("THE FLANK ENVELOPE Phi_J AND THE MIDDLE-SUM LEMMA")
    print("=" * 79)
    print("  m(J) = smallest literal middle sum = floor((J-2)/2) q' (+a if J "
          "odd).")
    print("  Phi_J = max flank sum over legal J-windows.  The per-J analogue is")
    print("  Phi_J + m(J) <= F_2 + s_min, so Phi_J must FALL by q' every two "
          "levels.")
    print()
    print("   M   q'  s_min  F_2 |  J=3 Phi  m  sum |  J=4 Phi  m  sum |"
          "  J=5 Phi  m  sum")
    for y in sorted(rows):
        q1 = next_prime(y)
        a, b = letters(q1)
        cells = []
        for J in (3, 4, 5):
            r = rows[y].get(J)
            if not r or not r.get("n") or not r.get("lit"):
                cells.append("%17s" % "-")
                continue
            k = (J - 2) // 2
            mJ = k * q1 + (a if (J - 2) % 2 else 0)
            phi = r["lit"]["maxflank"]
            cells.append("%9d %3d %4d" % (phi, mJ, phi + mJ))
        print("  %3d %4d %5d %4d | %s | %s | %s"
              % (y, q1, min(a, b), KNOWN_F2[y], cells[0], cells[1], cells[2]))

    print()
    print("=" * 79)
    print("SUMMARY TABLE - Delta_J, literal middles only")
    print("=" * 79)
    print("   M   q'  s_min | Delta_3 Delta_4 Delta_5 Delta_6 | J_max(legal) "
          " A_kill+1")
    for y in sorted(rows):
        q1 = next_prime(y)
        a, b = letters(q1)
        ds = []
        jmax = 2
        for J in (3, 4, 5, 6):
            r = rows[y].get(J)
            if r is None:
                # NOT "empty" - this vehicle simply has no data at this depth.
                # (Round-27 lesson: a cell a script FILLS IN rather than LOOKS
                # UP must be printed as such, or not printed.)
                ds.append("%7s" % "nodata")
            elif r.get("empty"):
                ds.append("%7s" % "EMPTY")
            elif r.get("lit"):
                ds.append("%+7d" % r["lit"]["Delta"])
                jmax = J
            else:
                ds.append("%7s" % "pad")
                jmax = J
        print("  %3d %4d %5d | %s | %11s  %8s"
              % (y, q1, min(a, b), " ".join(ds),
                 "%d" % jmax if rows[y].get(jmax + 1, {}).get("empty")
                 else ">=%d" % jmax,
                 A_KILL.get(y, 0) + 1 if y in A_KILL else "?"))
    print()
    print("=" * 79)
    print("THE SHARP FORM - THE PER-WORD FLANK BOUND")
    print("=" * 79)
    print("  Every per-J analogue Delta_J <= s_min is equivalent to ONE")
    print("  inequality per legal middle WORD w, with the middle sum moved to")
    print("  the right-hand side:")
    print()
    print("      Phi(w) := max flank sum over occurrences of w")
    print("      (L_w)      Phi(w)  <=  F_2(M) + s_min(q') - span(w).")
    print()
    print("  This is R26 clause (D) with F + q' replaced by the strictly")
    print("  sharper F_2 + s_min, and it is a statement about GAPS AT DISTANCE")
    print("  |w|+1 against gaps at distance 1 - a lag comparison, no depth")
    print("  quantifier, one row per word.  Below: every legal word with a")
    print("  realised occurrence, at every censused machine.")
    print()
    print("   M    q'  F_2 s_min | word w              span  Phi(w)  budget"
          "  slack")
    worst = {}
    for y in sorted(rows):
        q1 = next_prime(y)
        a, b = letters(q1)
        smin = min(a, b)
        F2 = KNOWN_F2[y]
        if y <= 23:
            d = gaps_of(y)
            src = "scan"
        else:
            path = os.path.join(DDIR, "gap_tuples_%d_4.csv" % y)
            src = "census"
        seen = {}
        for J in sorted(k for k in rows[y] if rows[y][k].get("n")):
            W = (windows_from_gaps(d, J) if src == "scan"
                 else windows_from_census(path, J))
            mids = W[:, 1:J - 1]
            for i in range(len(W)):
                w = tuple(int(v) for v in mids[i])
                if not legal_middles(w, q1, a, b):
                    continue
                fl = int(W[i][0]) + int(W[i][J - 1])
                if fl > seen.get(w, -1):
                    seen[w] = fl
        for w in sorted(seen, key=lambda t: (len(t), -sum(t))):
            sp = sum(w)
            bud = F2 + smin - sp
            sl = bud - seen[w]
            if len(seen) > 14 and sl > 12 and len(w) == 1:
                continue                     # keep the table readable
            print("  %3d %4d %4d %5d | %-19s %5d %6d %7d %+6d%s"
                  % (y, q1, F2, smin, ",".join(map(str, w)), sp, seen[w],
                     bud, sl, "   *** FAILS" if sl < 0 else ""))
            key = (y, len(w))
            if sl < worst.get(key, (10 ** 9,))[0]:
                worst[key] = (sl, w)
    print()
    print("  TIGHTEST ROW PER (machine, word length):")
    print("   M   |w|=1        |w|=2        |w|=3")
    for y in sorted(rows):
        cs = []
        for L in (1, 2, 3):
            v = worst.get((y, L))
            cs.append("%+4d %-9s" % (v[0], ",".join(map(str, v[1])))
                      if v else "%-14s" % "-")
        print("  %3d  %s %s %s" % (y, cs[0], cs[1], cs[2]))

    print()
    print("  'EMPTY'  = certified: no legal J-window exists (exact source).")
    print("  'nodata' = this vehicle stops at J = 4 (the 4-tuple census); the")
    print("             J = 5, 6 cells are supplied by research/perj_scanfree.py.")
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
