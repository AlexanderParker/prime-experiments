"""Round 29 (constructor): THE EVEN-J MECHANISM.

Odd J has the palindrome/antipode route (R87).  Even J does not: Theorem B
forbids a literal even-J window from being a palindrome at all, so the mirror
lever has nothing to bite on.  This script builds and tests the replacement.

WHAT IS PROVED HERE (stated in the output, verified numerically below):

  R89  THE WORD REDUCTION.  The J-2 middles of a word-legal J-window are J-2
       CONSECUTIVE gaps of M, each a legal letter, with T3-alternating nonzero
       classes - that is, a REALISED LEGAL WORD of length J-2.  Conversely any
       realised legal word of length J-2, together with its two flanking gaps,
       IS a word-legal J-window.  Hence, with L(M) the longest realised legal
       word,
            Q*_J > -inf  <=>  L(M) >= J-2,      J_max(M) = L(M) + 2,
       and since a J-window kills J-1 openings, A_kill(M -> q') = L(M) + 1.
       (The identity J_max = A_kill + 1, MEASURED 8/8 in R81, is therefore a
       definitional theorem, not a coincidence.)

  R90  THE SAME-TOOTH LEMMA.  Write t_i in {+,-} for the tooth of the i-th
       killed opening.  A middle of class 0 (padded) leaves the tooth fixed; a
       middle of class +-1 flips it.  So the middle span x_{J-1} - x_1 =
       (t_{J-1} - t_1) c = 0 mod q' EXACTLY WHEN the number of NON-PADDED
       middles is even.  For a LITERAL even-J window all J-2 middles are
       non-padded, so the span is = 0 mod q' and >= ((J-2)/2) q'.  This is
       Theorem A's even case with a tooth reason, and it is the even-J
       structure the palindrome route cannot supply.

WHAT IS MEASURED: the per-word flank table Phi(w) with its argmax flank pair,
at every machine with an exact source, and the five pre-registered even-J
predictions EJ1-EJ6 scored against it.

SOURCES (exact only; a superset is used for NOTHING here):
  m11..m23   full cyclic period scan, F and F_2 asserted against the corpus
  m29,m31,m37  Mechanic's exact full-period 4-tuple censuses
  m29        also the exact 5-tuple census (r28) for length-3 words

Usage:  uv run python research/evenj_r29.py
"""
import os
import sys
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
            41: 103, 43: 116}
# recorded for the gate: R81/R68
REC_QSTAR = {11: {2: 11, 3: 8}, 13: {2: 16, 3: 18}, 17: {2: 25, 3: 25},
             19: {2: 31, 3: 33, 4: 34}, 23: {2: 39, 3: 43},
             29: {2: 55, 3: 58, 4: 55, 5: 55}, 31: {2: 68, 3: 85, 4: 88, 5: 68},
             37: {2: 90, 3: 90, 4: 91}}
REC_JMAX = {11: 3, 13: 3, 17: 3, 19: 4, 23: 3, 29: 5, 31: 5, 37: 4}
REC_AKILL = {11: 2, 13: 2, 17: 2, 19: 3, 23: 2, 29: 4, 31: 4, 37: 3}
CENSUS4 = {29: "gap_tuples_29_4.csv", 31: "gap_tuples_31_4.csv",
           37: "gap_tuples_37_4.csv"}
CENSUS5 = {29: os.path.join("r28", "gap_tuples_29_5.csv")}
SCANNED = (11, 13, 17, 19, 23)


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def next_prime(y):
    p = y + 1
    while not is_prime(p):
        p += 1
    return p


def gears_of(y):
    return [p for p in range(5, y + 1) if is_prime(p)]


def cls_of(v, q1, a, b):
    """T3 class of a gap value: 0 padded, +1 class a, -1 class b, None illegal."""
    r = v % q1
    if r == 0:
        return 0
    if r == a % q1:
        return 1
    if r == b % q1:
        return -1
    return None


def t3_ok(word, q1, a, b):
    """all letters legal AND nonzero classes strictly alternate."""
    last = 0
    for v in word:
        c = cls_of(v, q1, a, b)
        if c is None:
            return False
        if c:
            if c == last:
                return False
            last = c
    return True


# ------------------------------------------------------------------ sources -
_GAPS = {}


def gaps_of(y):
    if y in _GAPS:
        return _GAPS[y]
    gears = gears_of(y)
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    op = np.flatnonzero(~ex).astype(np.int64)
    d = np.diff(np.concatenate([op, [op[0] + P]]))
    _GAPS[y] = d
    return d


def uniq_rows(cols, base):
    """Distinct rows of the column stack, via int64 packing (values < base)."""
    key = np.zeros(len(cols[0]), np.int64)
    for c in cols:
        key = key * base + c
    key = np.unique(key)
    m = len(cols)
    out = np.empty((len(key), m), np.int64)
    for i in range(m - 1, -1, -1):
        out[:, i] = key % base
        key //= base
    return out


def tuples_from_scan(y, m):
    """Distinct realised m-tuples of consecutive gaps, cyclically closed."""
    d = gaps_of(y)
    return uniq_rows([np.roll(d, -i) for i in range(m)], int(d.max()) + 1)


def census(y, m):
    path = None
    if m == 4 and y in CENSUS4:
        path = os.path.join(DDIR, CENSUS4[y])
    elif m == 5 and y in CENSUS5:
        path = os.path.join(DDIR, CENSUS5[y])
    if path is None or not os.path.exists(path):
        return None
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)


def max_arity(y):
    if y in SCANNED:
        return 6
    for M in (5, 4):
        if census(y, M) is not None:
            return M
    return 0


def tuple_dict(y, m):
    """Realised m-tuples (as a set of python tuples) or None if no exact source.

    For a census of arity M >= m the induced m-tuples are EXACT (every realised
    m-tuple sits inside a realised M-tuple)."""
    if y in SCANNED:
        if m > 6:
            return None
        arr = tuples_from_scan(y, m)
    else:
        for M in (5, 4):
            if M < m:
                continue
            arr = census(y, M)
            if arr is not None:
                sub = np.concatenate([arr[:, i:i + m]
                                      for i in range(0, M - m + 1)], axis=0)
                arr = uniq_rows([sub[:, i] for i in range(m)],
                                int(sub.max()) + 1)
                break
        else:
            return None
    return set(map(tuple, arr.tolist()))


# --------------------------------------------------------------- the tables -
def analyse(y):
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    s_min = min(a, b)
    F, F2 = KNOWN_F[y], KNOWN_F2[y]
    out = {"y": y, "q1": q1, "a": a, "b": b, "s_min": s_min, "F": F, "F2": F2}

    # gate: the source reproduces F and F_2
    d1 = tuple_dict(y, 1)
    d2 = tuple_dict(y, 2)
    assert d1 is not None and d2 is not None, y
    assert max(t[0] for t in d1) == F, ("F gate", y)
    assert max(sum(t) for t in d2) == F2, ("F2 gate", y)
    out["values"] = sorted(t[0] for t in d1)
    out["Lambda"] = [v for v in out["values"] if cls_of(v, q1, a, b) is not None]

    # L(M): longest realised legal word.  The dictionary is built to the
    # FULL arity the source supports, so no table cell is ever filled in.
    R = max_arity(y)
    out["max_arity"] = R
    dicts = {r: tuple_dict(y, r) for r in range(1, R + 1)}
    L, Lcert = 0, False
    for r in range(1, R + 1):
        if any(t3_ok(t, q1, a, b) for t in dicts[r]):
            L = r
        else:
            Lcert = True                       # no legal word of this length
            break
    out["L"], out["L_certified"] = L, Lcert
    out["Jmax_pred"] = L + 2
    out["Akill_pred"] = L + 1

    # per-word flank table: Phi(w) and the argmax flank pair
    words = {}
    for r in sorted(k for k in dicts if k + 2 in dicts):
        for t in dicts[r + 2]:
            w = t[1:-1]
            if not t3_ok(w, q1, a, b):
                continue
            fs = t[0] + t[-1]
            cur = words.get(w)
            if cur is None or fs > cur[0]:
                words[w] = (fs, t[0], t[-1])
    out["words"] = words

    # the UNCONSTRAINED spectrum F_j from the same source, for the F_j - Q*_j
    # comparison (this is the quantity the spectrum-plus-depth certificate
    # discards, i.e. the work word-legality does)
    F_j = {}
    for r in range(1, R + 1):
        F_j[r] = max(sum(t) for t in dicts[r])
    out["F_j"] = F_j

    # Q*_J per J from the word table
    qstar = {}
    for w, (fs, gL, gR) in words.items():
        J = len(w) + 2
        span = sum(w) + fs
        if span > qstar.get(J, (-1,))[0]:
            qstar[J] = (span, (gL,) + w + (gR,))
    qstar[2] = (F2, None)
    out["qstar"] = qstar
    return out


def main():
    print("=" * 78)
    print("THE EVEN-J MECHANISM  (constructor, round 29)")
    print("=" * 78)
    print(__doc__.split("WHAT IS PROVED HERE")[1].split("WHAT IS MEASURED")[0])

    res = {}
    for y in (11, 13, 17, 19, 23, 29, 31, 37):
        res[y] = analyse(y)
        r = res[y]
        print("m%-3d q'=%-3d a=%-3d b=%-3d s_min=%-3d F=%-4d F_2=%-4d "
              "Lambda=%s  L=%d %s  (source arity %d)"
              % (y, r["q1"], r["a"], r["b"], r["s_min"], r["F"], r["F2"],
                 r["Lambda"], r["L"],
                 "CERTIFIED" if r["L_certified"] else "LOWER BOUND ONLY",
                 r["max_arity"]))

    # ---- GATE: reproduce the recorded Q* table where the source reaches ----
    print("\n" + "-" * 78)
    print("GATE 1 - reproduce R68/R81's recorded Q*_J from these sources")
    bad = ok = nodata = 0
    for y in res:
        for J, v in REC_QSTAR[y].items():
            got = res[y]["qstar"].get(J, (None,))[0]
            if got is None:
                nodata += 1
                print("   m%-3d J=%d recorded %3d : NO DATA at this arity "
                      "(source arity %d)" % (y, J, v, res[y]["max_arity"]))
            elif got != v:
                bad += 1
                print("   m%-3d J=%d recorded %3d : GOT %s  *** MISMATCH"
                      % (y, J, v, got))
            else:
                ok += 1
    print("   cells REPRODUCED: %d ; mismatches: %d ; no-data: %d"
          % (ok, bad, nodata))
    assert bad == 0 and ok >= 13, ("Q* gate failed", ok, bad)

    print("\nGATE 2 - R89: J_max = L + 2 and A_kill = L + 1")
    hits = tot = 0
    for y in res:
        r = res[y]
        ok1 = r["Jmax_pred"] == REC_JMAX[y]
        ok2 = r["Akill_pred"] == REC_AKILL[y]
        tot += 2
        hits += ok1 + ok2
        print("   m%-3d L=%d -> J_max %d (recorded %d) %s ; A_kill %d "
              "(recorded %d) %s"
              % (y, r["L"], r["Jmax_pred"], REC_JMAX[y], "OK" if ok1 else "**",
                 r["Akill_pred"], REC_AKILL[y], "OK" if ok2 else "**"))
    print("   R89 score: %d / %d" % (hits, tot))

    # ---------------------------------------------------------------- EJ1 ---
    print("\n" + "-" * 78)
    print("EJ1 - SAME-TOOTH LEMMA (R90).  middle span = 0 mod q' iff the number")
    print("      of NON-PADDED middles is even.  Checked on EVERY realised legal")
    print("      word at every machine, and the even-J maximisers listed.")
    viol = 0
    nchk = 0
    for y in res:
        r = res[y]
        for w in r["words"]:
            npad = sum(1 for v in w if v % r["q1"])
            nchk += 1
            if (sum(w) % r["q1"] == 0) != (npad % 2 == 0):
                viol += 1
                print("   VIOLATION m%d word %s" % (y, w))
    print("   %d realised legal words checked, %d violations" % (nchk, viol))
    for y in res:
        r = res[y]
        for J in sorted(r["qstar"]):
            if J % 2 or J < 4:
                continue
            span, win = r["qstar"][J]
            w = win[1:-1]
            lit = all(v % r["q1"] for v in w)
            print("   m%-3d J=%d  maximiser %s  span %3d  middle sum %3d "
                  "(= %d mod q')  %s"
                  % (y, J, win, span, sum(w), sum(w) % r["q1"],
                     "LITERAL" if lit else "padded"))

    # ---------------------------------------------------------------- EJ2 ---
    print("\n" + "-" * 78)
    print("EJ2 - the even-J depth cap IS the word length")
    for J in (4, 6):
        line = []
        for y in res:
            nonempty = J in res[y]["qstar"]
            pred = res[y]["L"] >= J - 2
            line.append("m%d:%s%s" % (y, "NE" if nonempty else "E ",
                                      "" if nonempty == pred else "**"))
        print("   J=%d  %s" % (J, "  ".join(line)))

    # ---------------------------------------------------------------- EJ3 ---
    print("\n" + "-" * 78)
    print("EJ3 - PAR-TRADING EXACTNESS:  eps(v) = Phi(u) - Phi(v) - x  for")
    print("      v = u.x  (drop the LAST letter) and v = x.u (drop the FIRST)")
    eps_all = []
    for y in sorted(res):
        r = res[y]
        W = r["words"]
        for v in sorted(W):
            if len(v) < 2:
                continue
            for tag, u, x in (("suffix", v[:-1], v[-1]),
                              ("prefix", v[1:], v[0])):
                if u not in W:
                    continue
                eps = W[u][0] - W[v][0] - x
                eps_all.append((y, v, tag, eps, r["s_min"]))
                print("   m%-3d %-14s %-6s u=%-12s Phi(u)=%3d Phi(v)=%3d "
                      "x=%3d  eps=%+4d   s_min=%d %s"
                      % (y, str(v), tag, str(u), W[u][0], W[v][0], x, eps,
                         r["s_min"], "" if abs(eps) <= r["s_min"] else "*FAIL*"))
    if eps_all:
        ok = sum(1 for e in eps_all if abs(e[3]) <= e[4])
        mean = sum(abs(e[3]) for e in eps_all) / len(eps_all)
        meanb = sum(e[4] for e in eps_all) / len(eps_all)
        print("   EJ3 score: %d / %d cells with |eps| <= s_min ; "
              "mean |eps| = %.2f vs mean s_min/2 = %.2f"
              % (ok, len(eps_all), mean, meanb / 2))

    # ---------------------------------------------------------------- EJ4 ---
    print("\n" + "-" * 78)
    print("EJ4 - the half-split of an even-J maximiser (overall and LITERAL)")
    for y in sorted(res):
        r = res[y]
        for J in sorted(r["qstar"]):
            if J % 2 or J < 4:
                continue
            cells = [("overall", r["qstar"][J][1])]
            best = None
            for v, (fs, gL, gR) in r["words"].items():
                if len(v) != J - 2 or not all(x % r["q1"] for x in v):
                    continue
                if best is None or sum(v) + fs > best[0]:
                    best = (sum(v) + fs, (gL,) + v + (gR,))
            if best is not None and best[1] != cells[0][1]:
                cells.append(("literal", best[1]))
            for tag, win in cells:
                hL, hR = win[0] + win[1], win[-2] + win[-1]
                print("   m%-3d J=%d %-8s %-18s h_L=%3d h_R=%3d  "
                      "min/F_2 = %.3f  span/F_2 = %.3f  (2F_2 wall = %d)"
                      % (y, J, tag, str(win), hL, hR,
                         min(hL, hR) / r["F2"], sum(win) / r["F2"],
                         2 * r["F2"]))

    # ---------------------------------------------------------------- EJ5 ---
    print("\n" + "-" * 78)
    print("EJ5 - the even-J flank ceiling  Phi_J <= F_2 - b   (literal cells)")
    for y in sorted(res):
        r = res[y]
        for J in sorted(r["qstar"]):
            if J % 2 or J < 4:
                continue
            span, win = r["qstar"][J]
            w = win[1:-1]
            lit = all(v % r["q1"] for v in w)
            # the literal maximum at this J, separately
            best = None
            for v, (fs, gL, gR) in r["words"].items():
                if len(v) != J - 2 or not all(x % r["q1"] for x in v):
                    continue
                if best is None or sum(v) + fs > best[0]:
                    best = (sum(v) + fs, (gL,) + v + (gR,), fs)
            if best is None:
                print("   m%-3d J=%d  NO literal window" % (y, J))
                continue
            print("   m%-3d J=%d  literal max %3d %s  Phi=%3d  F_2-b=%3d  %s"
                  % (y, J, best[0], best[1], best[2], r["F2"] - r["b"],
                     "OK margin %+d" % (r["F2"] - r["b"] - best[2])
                     if best[2] <= r["F2"] - r["b"] else "FAIL"))
            if not lit:
                print("        (overall maximiser at this J is PADDED: %s "
                      "span %d, Phi=%d)" % (win, span, win[0] + win[-1]))

    # ---------------------------------------------------------------- EJ6 ---
    print("\n" + "-" * 78)
    print("EJ6 - Delta_J = Q*_J - F_2, all J, literal and overall")
    print("   %-5s %-6s %s" % ("M", "F_2", "  ".join("J=%d" % J
                                                     for J in range(3, 7))))
    for y in sorted(res):
        r = res[y]
        cells = []
        for J in range(3, 7):
            if J in r["qstar"]:
                cells.append("%+4d" % (r["qstar"][J][0] - r["F2"]))
            elif r["L"] >= J - 2:
                cells.append("  nd")
            else:
                cells.append("   E")
        print("   m%-4d %-6d %s" % (y, r["F2"], "  ".join(cells)))
    print("\n   (E = certified EMPTY by R89 from L(M); nd = source arity too")
    print("    small to see this layer)")

    print("\n   LITERAL-ONLY Delta_J, and the residual chain eps_J =")
    print("   Delta_{J-1} - Delta_J  (the amount the maximiser LOSES by going")
    print("   one letter deeper).  This is the quantity Delta_J = O(1) is about.")
    print("   %-5s %-6s %s" % ("M", "s_min", "  ".join("J=%d" % J
                                                       for J in range(3, 7))))
    chain = []
    for y in sorted(res):
        r = res[y]
        lit = {}
        for v, (fs, gL, gR) in r["words"].items():
            if not all(x % r["q1"] for x in v):
                continue
            J = len(v) + 2
            lit[J] = max(lit.get(J, -1), sum(v) + fs)
        lit[2] = r["F2"]
        cells, prev = [], 0
        for J in range(3, 7):
            if J in lit:
                dl = lit[J] - r["F2"]
                cells.append("%+4d" % dl)
                chain.append((y, J, prev - dl, r["s_min"]))
                prev = dl
            elif r["L"] >= J - 2 and J - 2 <= r["max_arity"] - 2:
                cells.append("   -")      # no LITERAL word of this length
            elif r["L"] >= J - 2:
                cells.append("  nd")
            else:
                cells.append("   E")
        print("   m%-4d %-6d %s" % (y, r["s_min"], "  ".join(cells)))
    print("\n" + "-" * 78)
    print("EJ7 - THE WORK WORD-LEGALITY DOES: F_J - Q*_J, by depth")
    print("   (F_J from the SAME source, asserted against the corpus at J=1,2;")
    print("    this is exactly what the spectrum-plus-depth certificate throws")
    print("    away, and the reason it fails at 29 -> 31)")
    print("   %-5s %-26s %s" % ("M", "F_J  (J=1..)", "F_J - Q*_J  (J=3,4,5)"))
    par = {3: [], 4: [], 5: []}
    for y in sorted(res):
        r = res[y]
        assert r["F_j"][1] == r["F"] and r["F_j"][2] == r["F2"], ("F_j gate", y)
        cells = []
        for J in (3, 4, 5):
            if J in r["qstar"] and J in r["F_j"]:
                d = r["F_j"][J] - r["qstar"][J][0]
                cells.append("%4d" % d)
                par[J].append(d)
            else:
                cells.append("   .")
        print("   m%-4d %-26s %s"
              % (y, str([r["F_j"][j] for j in sorted(r["F_j"])]),
                 "  ".join(cells)))
    for J in (3, 4, 5):
        if par[J]:
            print("   J=%d : %d cells, range %d..%d, mean %.1f"
                  % (J, len(par[J]), min(par[J]), max(par[J]),
                     sum(par[J]) / len(par[J])))
    print("   READING: legality's work GROWS with depth and shows no parity")
    print("   effect - the even/odd split is in the STRUCTURE (palindromes,")
    print("   same tooth), not in this number.")

    print("\n   residual chain along the LITERAL maximisers:")
    for y, J, e, s in chain:
        print("      m%-3d  J=%d  eps = %+d   (s_min = %d)" % (y, J, e, s))
    if chain:
        print("      max |eps| along a maximising chain: %d, against s_min "
              "%d..%d" % (max(abs(e) for _, _, e, _ in chain),
                          min(s for *_, s in chain), max(s for *_, s in chain)))
    print("=" * 78)


if __name__ == "__main__":
    main()
