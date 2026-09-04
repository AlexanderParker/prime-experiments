"""LATERAL round 31 - THE SPECTRUM BOUND ON L, AND (B) RE-POSED.

(1) THE THEOREM.  Let M be a machine, q' = nextprime(y), u = 6^{-1} mod q'
taken as the SMALL representative, d' = 2u, a = d', b = q' - a,
a_min = min(a, b), G = F(M + q').  Then

  (i)   the smallest positive gap value in class +d' is a, in class -d' is b,
        and in the padded class 0 mod q' it is q';
  (ii)  T3 makes the nonzero-class letters strictly alternate, so any two
        CONSECUTIVE nonzero letters of a legal word sum to at least a + b = q';
  (iii) a realised legal word of m letters is the middle of a window
        x_0 < ... < x_{m+2} of consecutive openings of M whose m middle gaps
        are legal, so R68's ATTAINMENT THEOREM (proved) gives
        span(word) + before + after <= G with before, after >= 1.

  With p padded letters and n = m - p nonzero ones,
        span >= p q' + floor(n/2) q' + [n odd] a_min,
  hence with T = floor((G - 2) / q'):

        (SIMPLE)  L(M) <= 2T + 1                       (and <= 2T + 1 - p)
        (PARITY)  L(M) <= max( 2T, 2*floor((G - 2 - a_min)/q') + 1 ).

  UNCONDITIONAL given R68 and T3.  L <= 2 G / q' + 1: L is O(F/q'), not O(1).

(2) RE-POSING (B).  R99: F(M+q') <= F_2(M) + c_A L.  Substituting the bound
removes (B): for q' > 2 c_A,
        G <= ( q'(F_2 + c_A) - 4 c_A ) / ( q' - 2 c_A ),
and (D) F(M+q') <= F(M) + q' follows whenever, with eps = F_2 - F,
        8 F <= q'^2 - (eps + 12) q' + 16      (c_A = 4).

(3) THE FAMILY.  On the tooth-counterfactual family a_min = min(2v, q'-2v)
sweeps 1..(q'-1)/2 while the real machine is pinned near q'/3.  Is L governed
by a_min alone?

usage:
  uv run python research/lateral_r31.py corpus
  uv run python research/lateral_r31.py family
  uv run python research/lateral_r31.py all
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R30 = os.path.join(HERE, "data", "r30")
OUT = os.path.join(HERE, "data", "r31")

NGATE = 0

# ---- corpus INPUTS, all on record (cited in the round-31 append) ----------
# F(M) for M = {5..y}; F(M+q') is the entry at q'.
KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
# F_2(M), from Constructor R99's table (S_2 = F + q' - F_2)
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
            41: 103, 43: 116, 47: 134, 53: 159, 59: 173}
KNOWN_S2 = {11: 9, 13: 12, 17: 12, 19: 17, 23: 24, 29: 19, 31: 27, 37: 39,
            41: 31, 43: 34, 47: 37, 53: 45, 59: 49}
KNOWN_L = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2,
           43: 2, 47: 4, 53: 3}
EXPCAP = {11: 1, 13: 1, 17: 1, 19: 4, 23: 2, 29: 3, 31: 5, 37: 18, 41: 13,
          43: 10, 47: 5, 53: 21}
# realised max-length words on record (Mechanic V4 at m19..m37,
# Constructor's exhaustive round-29 decision at m47)
REALISED = {19: [(8, 15), (15, 8)],
            23: [(10,), (19,), (29,)],
            29: [(10, 21, 10)],
            31: [(12, 25, 12), (25, 12, 25)],
            37: [(14, 41), (27, 41), (41, 14), (41, 27)],
            47: [(18, 35, 18, 35), (35, 18, 35, 18)]}


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def next_prime(y):
    p = y + 1
    while not is_prime(p):
        p += 1
    return p


def letters(qp):
    """(a, b, a_min) for the next gear q'."""
    u = pow(6, -1, qp)
    u = min(u % qp, (-u) % qp)
    a = (2 * u) % qp
    return a, qp - a, min(a, qp - a)


def bound_simple(G, qp):
    return 2 * ((G - 2) // qp) + 1


def bound_parity(G, qp, amin):
    even = 2 * ((G - 2) // qp)
    odd = 2 * max(-1, (G - 2 - amin) // qp) + 1
    return max(even, odd)


def cls_of(v, qp, a):
    r = v % qp
    return 0 if r == 0 else 1 if r == a else -1 if r == (qp - a) % qp else None


# ------------------------------------------------------------------ corpus

def cmd_corpus(args):
    print("== (0) THE ARITHMETIC OF THE LETTERS: a = 2u, 3a = q' -+ 1 ==")
    print("   q'   u   a=d'   b   a_min  3a  q'-+1  a_min/q'")
    for y in sorted(KNOWN_L):
        qp = next_prime(y)
        u = pow(6, -1, qp)
        u = min(u % qp, (-u) % qp)
        a, b, am = letters(qp)
        gate(3 * a in (qp - 1, qp + 1),
             "q' = %d: 3a = %d = q' %s 1" % (qp, 3 * a, "-" if 3 * a < qp
                                             else "+"))
        gate(a + b == qp, "q' = %d: a + b = q'" % qp)
        gate(am == a, "q' = %d: a_min is the SMALLER letter a = d' (never b)" % qp)
        # the smallest positive value in each legal class
        gate(min(v for v in range(1, 3 * qp) if v % qp == a % qp) == a
             and min(v for v in range(1, 3 * qp) if v % qp == b % qp) == b
             and min(v for v in range(1, 3 * qp) if v % qp == 0) == qp,
             "q' = %d: class minima are a = %d, b = %d, padded = %d"
             % (qp, a, b, qp))
        print("   %-4d %-3d %-6d %-4d %-6d %-4d %-6d %.4f"
              % (qp, u, a, b, am, 3 * a, qp - 1 if 3 * a < qp else qp + 1,
                 am / qp))

    print("\n== (1) THE SPECTRUM BOUND ON L, CORPUS TABLE ==")
    print("   R68 (proved): consecutive openings x_0..x_J of M with a legal")
    print("   middle-gap word satisfy x_J - x_0 <= F(M+q').  Every gap >= 1.")
    hdr = ("   M     q'   G=F(M+q')  a_min  T   SIMPLE  PARITY  L   EXPCAP  "
           "G/a_min  min(SIMPLE,EXPCAP)")
    print(hdr)
    rows = []
    for y in sorted(KNOWN_L):
        qp = next_prime(y)
        G = KNOWN_F[qp]
        a, b, am = letters(qp)
        T = (G - 2) // qp
        bs = bound_simple(G, qp)
        bp = bound_parity(G, qp, am)
        L = KNOWN_L[y]
        ec = EXPCAP[y]
        mgr = G // am
        rows.append((y, qp, G, am, T, bs, bp, L, ec, mgr))
        print("   m%-4d %-4d %-10d %-6d %-3d %-7d %-7d %-3d %-7d %-8d %d"
              % (y, qp, G, am, T, bs, bp, L, ec, mgr, min(bs, ec)))
        gate(L <= bs, "m%d: L = %d <= SIMPLE bound %d" % (y, L, bs))
        gate(L <= bp, "m%d: L = %d <= PARITY bound %d" % (y, L, bp))
        gate(bp <= bs, "m%d: PARITY (%d) <= SIMPLE (%d)" % (y, bp, bs))
        gate(bs <= mgr, "m%d: SIMPLE (%d) <= the a_min form G/a_min (%d)"
             % (y, bs, mgr))
    tight_s = [r[0] for r in rows if r[5] == r[7]]
    tight_p = [r[0] for r in rows if r[6] == r[7]]
    print("   SIMPLE row : %s" % [r[5] for r in rows])
    print("   PARITY row : %s" % [r[6] for r in rows])
    print("   L      row : %s" % [r[7] for r in rows])
    print("   EXPCAP row : %s" % [r[8] for r in rows])
    print("   G/a_min    : %s" % [r[9] for r in rows])
    print("   TIGHT (bound = L): SIMPLE at m%s ; PARITY at m%s"
          % (tight_s, tight_p))
    better = [r[0] for r in rows if r[6] < r[5]]
    print("   PARITY strictly better than SIMPLE at: m%s" % better)
    wins = [r[0] for r in rows if r[5] < r[8]]
    ties = [r[0] for r in rows if r[5] == r[8]]
    loss = [r[0] for r in rows if r[5] > r[8]]
    print("   SIMPLE vs EXPCAP: beats at m%s, ties at m%s, loses at m%s"
          % (wins, ties, loss))

    # DIRECT CHECK of the theorem's span accounting on the realised words
    print("\n   direct span accounting on the realised words on record:")
    for y in sorted(REALISED):
        qp = next_prime(y)
        G = KNOWN_F[qp]
        a, b, am = letters(qp)
        for w in REALISED[y]:
            if len(w) != KNOWN_L[y]:
                continue
            p = sum(1 for v in w if v % qp == 0)
            n = len(w) - p
            lo = p * qp + (n // 2) * qp + (am if n % 2 else 0)
            print("      m%-3d %-22s span %3d  >= lower bound %3d ; "
                  "span + 2 = %3d <= G = %d   (p=%d, n=%d)"
                  % (y, str(w), sum(w), lo, sum(w) + 2, G, p, n))
            gate(sum(w) >= lo,
                 "m%d %s: span %d >= p q' + floor(n/2) q' + [n odd] a_min = %d"
                 % (y, w, sum(w), lo))
            gate(sum(w) + 2 <= G,
                 "m%d %s: span + 2 = %d <= F(M+q') = %d (R68)"
                 % (y, w, sum(w) + 2, G))

    print("\n== (2) RE-POSING (B): THE PRODUCT AGAINST THE DEPTH-2 SLACK ==")
    print("   c_A = 4 (literal, R99); c_B = the spectrum bound.")
    print("   M     q'   F    F_2  eps  S_2  c_B(S) 4c_B  c_B(P) 4c_B  "
          "verdict(S) verdict(P)")
    for y in sorted(KNOWN_L):
        qp = next_prime(y)
        G = KNOWN_F[qp]
        F, F2 = KNOWN_F[y], KNOWN_F2[y]
        eps = F2 - F
        S2 = F + qp - F2
        gate(S2 == KNOWN_S2[y], "m%d: S_2 = F + q' - F_2 = %d matches R99's "
                                "table" % (y, S2))
        a, b, am = letters(qp)
        bs, bp = bound_simple(G, qp), bound_parity(G, qp, am)
        print("   m%-4d %-4d %-4d %-4d %-4d %-4d %-6d %-5d %-6d %-5d %-10s %s"
              % (y, qp, F, F2, eps, S2, bs, 4 * bs, bp, 4 * bp,
                 "OK" if 4 * bs <= S2 else "FAILS",
                 "OK" if 4 * bp <= S2 else "FAILS"))
        gate(4 * bp <= S2, "m%d: c_A c_B = 4*%d = %d <= S_2 = %d (PARITY)"
             % (y, bp, 4 * bp, S2))

    print("\n   THE SELF-REFERENTIAL CLOSURE (c_A = 4):")
    print("     G <= (q'(F_2 + 4) - 16)/(q' - 8),  and (D) follows whenever")
    print("     8 F <= q'^2 - (eps + 12) q' + 16.")
    print("   M     q'   F    eps  closure-G  F+q'  (D)?   RHS/8      F     "
          "F/RHS   F/q'^2   F/q'")
    for y in sorted(KNOWN_F):
        if y == 59 and 61 not in KNOWN_F:
            Gknown = None
        qp = next_prime(y)
        F, F2 = KNOWN_F[y], KNOWN_F2[y]
        eps = F2 - F
        clo = (qp * (F2 + 4) - 16) / (qp - 8)
        rhs = (qp * qp - (eps + 12) * qp + 16) / 8.0
        G = KNOWN_F.get(qp)
        print("   m%-4d %-4d %-4d %-4d %-10.1f %-5d %-6s %-10.1f %-5d "
              "%-7s %-8.4f %.3f"
              % (y, qp, F, eps, clo, F + qp,
                 "OK" if clo <= F + qp else "no", rhs, F,
                 ("%.2f" % (F / rhs)) if rhs > 0 else "n/a",
                 F / (qp * qp), F / qp))
        if G is not None:
            gate(G <= clo + 1e-9,
                 "m%d: the closure bound G <= %.1f is TRUE (G = %d)"
                 % (y, clo, G))
        gate((clo <= F + qp) == (8 * F <= qp * qp - (eps + 12) * qp + 16),
             "m%d: the closure condition and 8F <= q'^2-(eps+12)q'+16 agree"
             % y)
    json.dump({"rows": rows}, open(os.path.join(OUT, "corpus_bound.json"), "w"))


# ------------------------------------------------------------------ family

STEPS = [([5, 7], 11), ([5, 7, 11], 13), ([5, 7, 11, 13], 17),
         ([5, 7, 11, 13, 17], 19), ([5, 7, 11, 13, 17, 19], 23),
         ([5, 7, 11, 13, 17, 19, 23], 29)]
COLS = ("oi vqp F F2 F3 L na nb n0 Qmax Q3 Fn capped "
        "w1 w2 w3 w4 w5 w6 w7").split()
CI = {c: i for i, c in enumerate(COLS)}
WMAX = 7
TWIN = {5: 1, 7: 1, 11: 2, 13: 2, 17: 3, 19: 3, 23: 4, 29: 5}


def space_of(ogears):
    return [list(range(1, (q - 1) // 2 + 1)) for q in ogears]


def load_step(ogears, qp):
    y = ogears[-1]
    p = os.path.join(R30, "L_%d_%d.npy" % (y, qp))
    if os.path.exists(p):
        return np.load(p), "full-direct"
    fs = sorted(f for f in os.listdir(R30)
                if f.startswith("L_%d_%d_" % (y, qp)) and f.endswith(".npy"))
    if not fs:
        return None, None
    arr = np.concatenate([np.load(os.path.join(R30, f)) for f in fs])
    return arr, ("full" if y != 23 else "SAMPLE")


def exposed(g, v):
    return frozenset(r for r in range(g) if r % g != v % g and r % g != (-v) % g)


def fits(X, g, v):
    E = exposed(g, v)
    xs = {x % g for x in X}
    return any(all((t + x) % g in E for x in xs) for t in range(g))


def alt_points(a, qp, m):
    X, cur, lets = [0], 0, [a, qp - a]
    for i in range(m):
        cur += lets[i % 2]
        X.append(cur)
    return X


def rank_avg(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rank_avg(x), rank_avg(y)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def eta2(vals, levels):
    vals = np.asarray(vals, dtype=np.float64)
    tot = ((vals - vals.mean()) ** 2).sum()
    if tot == 0:
        return 0.0
    b = 0.0
    for lv in np.unique(levels):
        m = levels == lv
        b += m.sum() * (vals[m].mean() - vals.mean()) ** 2
    return float(b / tot)


def cmd_family(args):
    print("== (3) THE FAMILY: IS L JUST THE SIZE OF THE LETTERS? ==")
    print("   a = 2 v_q' mod q', b = q' - a, a_min = min(a, b) sweeps")
    print("   1..(q'-1)/2 on the family and is pinned near q'/3 on the real")
    print("   machine.  G = F(M+q') direct where round 30 has it, else the")
    print("   record law max(F_2, max_J Q*_J) (gated in round 30).")
    for ogears, qp in STEPS:
        arr, kind = load_step(ogears, qp)
        if arr is None:
            print("\n  [%d->%d] no data on disk - skipped" % (ogears[-1], qp))
            continue
        y = ogears[-1]
        ovecs = list(itertools.product(*space_of(ogears)))
        oi = arr[:, CI["oi"]]
        vqp = arr[:, CI["vqp"]].astype(np.int64)
        L = arr[:, CI["L"]].astype(np.int64)
        F2 = arr[:, CI["F2"]].astype(np.int64)
        Qm = arr[:, CI["Qmax"]].astype(np.int64)
        Fn = arr[:, CI["Fn"]].astype(np.int64)
        A = (2 * vqp) % qp
        B = qp - A
        amin = np.minimum(A, B)
        law = np.maximum(F2, Qm)
        direct = Fn >= 0
        G = np.where(direct, Fn, law)
        if direct.any():
            gate(bool(np.all(law[direct] == Fn[direct])),
                 "%d->%d: record law == direct F(M+q') at all %d direct rows"
                 % (y, qp, int(direct.sum())))
        T = (G - 2) // qp
        bs = 2 * T + 1
        odd = 2 * np.maximum(-1, (G - 2 - amin) // qp) + 1
        bp = np.maximum(2 * T, odd)
        # padded-aware refinement: L <= 2T + 1 - p
        W = arr[:, [CI["w%d" % i] for i in range(1, WMAX + 1)]]
        pad = np.zeros(len(arr), dtype=np.int64)
        for j in range(WMAX):
            w = W[:, j]
            pad += ((w >= 0) & (w % qp == 0)).astype(np.int64)
        print("\n  [%d->%d] %s, %d rows, %d old members; a_min range %d..%d"
              % (y, qp, kind, len(arr), len(np.unique(oi)), amin.min(),
                 amin.max()))
        gate(bool(np.all(L <= bs)),
             "%d->%d: SIMPLE bound holds at all %d rows (%d violations)"
             % (y, qp, len(arr), int((L > bs).sum())))
        gate(bool(np.all(L <= bp)),
             "%d->%d: PARITY bound holds at all %d rows (%d violations)"
             % (y, qp, len(arr), int((L > bp).sum())))
        gate(bool(np.all(L <= bs - pad)),
             "%d->%d: the padded-aware bound L <= 2T+1-p holds at all rows"
             % (y, qp))
        tight_s = float(np.mean(L == bs))
        tight_p = float(np.mean(L == bp))
        print("      TIGHT: SIMPLE at %.1f%% of rows, PARITY at %.1f%%; "
              "PARITY strictly better than SIMPLE at %.1f%%"
              % (100 * tight_s, 100 * tight_p, 100 * np.mean(bp < bs)))
        print("      mean slack bound - L: SIMPLE %.2f  PARITY %.2f"
              % (float((bs - L).mean()), float((bp - L).mean())))
        # is L governed by a_min?
        print("      spearman(L, a_min) = %+.3f ;  spearman(L, G) = %+.3f ;  "
              "spearman(L, G/a_min) = %+.3f"
              % (spearman(L, amin), spearman(L, G),
                 spearman(L, G / np.maximum(amin, 1))))
        e_amin = eta2(L, amin)
        e_v = eta2(L, vqp)
        ov = np.array([ovecs[int(o)] for o in oi], dtype=np.int64)
        e_old = max(eta2(L, ov[:, i]) for i in range(ov.shape[1]))
        print("      eta^2(L | a_min) = %.3f   eta^2(L | v_q') = %.3f   "
              "(ratio %.3f)   best old-gear eta^2 = %.3f"
              % (e_amin, e_v, e_amin / e_v if e_v > 0 else float("nan"), e_old))
        print("      max L and mean L by a_min:")
        for v in sorted(set(int(x) for x in amin)):
            m = amin == v
            print("         a_min=%2d (a=%2d,b=%2d, %6d rows): max L %d  "
                  "mean L %.3f  P(L>=3) %.4f  bound(P) max %d"
                  % (v, int(A[m][0]), int(B[m][0]), int(m.sum()),
                     int(L[m].max()), float(L[m].mean()),
                     float(np.mean(L[m] >= 3)), int(bp[m].max())))
        mx = int(L.max())
        sel = L == mx
        am_at_max = sorted(set(int(x) for x in amin[sel]))
        print("      family max L = %d, attained at a_min in %s (a_min range "
              "%d..%d)" % (mx, am_at_max, int(amin.min()), int(amin.max())))
        # spread of L at the a_min that attains the maximum
        v0 = am_at_max[0]
        m0 = amin == v0
        print("      at a_min = %d: L ranges %d..%d over %d rows (spread %d) "
              "-> a_min is %s"
              % (v0, int(L[m0].min()), int(L[m0].max()), int(m0.sum()),
                 int(L[m0].max() - L[m0].min()),
                 "NECESSARY BUT NOT SUFFICIENT" if
                 int(L[m0].max() - L[m0].min()) >= 2 else "nearly sufficient"))
        # the real machine's own row
        ti = ovecs.index(tuple(TWIN[q] for q in ogears))
        real = (oi == ti) & (vqp == TWIN[qp])
        if real.any():
            r = int(np.flatnonzero(real)[0])
            print("      REAL machine: a_min = %d (%.3f q'), L = %d, "
                  "G = %d, SIMPLE %d, PARITY %d; family rows with a_min < "
                  "real: %.1f%%, their mean L %.3f vs real %d"
                  % (int(amin[r]), amin[r] / qp, int(L[r]), int(G[r]),
                     int(bs[r]), int(bp[r]),
                     100 * float(np.mean(amin < amin[r])),
                     float(L[amin < amin[r]].mean())
                     if (amin < amin[r]).any() else float("nan"), int(L[r])))
        # SIZE OF THE LETTERS vs ARITHMETIC OF THE TEETH: the {5,7}
        # admissibility of the member's own bare alternation (a,b,a),
        # against a_min, as competing explanations of L
        v5 = np.array([ovecs[int(o)][0] for o in oi])
        v7 = np.array([ovecs[int(o)][1] for o in oi])
        memo = {}
        adm = np.zeros(len(arr), bool)
        for i in range(len(arr)):
            k = (int(v5[i]), int(v7[i]), int(A[i]))
            if k not in memo:
                X = alt_points(k[2], qp, 3)
                memo[k] = fits(X, 5, k[0]) and fits(X, 7, k[1])
            adm[i] = memo[k]
        print("      SIZE vs ARITHMETIC: eta^2(L | a_min) = %.3f   "
              "eta^2(L | {5,7}-admissible) = %.3f   eta^2(L | both) = %.3f"
              % (e_amin, eta2(L, adm.astype(np.int64)),
                 eta2(L, amin * 2 + adm.astype(np.int64))))
        print("         P(L>=3): admissible %.4f   not admissible %.4f ; "
              "smallest a_min (=%d) %.4f   largest a_min (=%d) %.4f"
              % (float(np.mean(L[adm] >= 3)) if adm.any() else float("nan"),
                 float(np.mean(L[~adm] >= 3)) if (~adm).any() else float("nan"),
                 int(amin.min()), float(np.mean(L[amin == amin.min()] >= 3)),
                 int(amin.max()), float(np.mean(L[amin == amin.max()] >= 3))))
        print("         max L: admissible %d, not admissible %d"
              % (int(L[adm].max()) if adm.any() else -1,
                 int(L[~adm].max()) if (~adm).any() else -1))
        np.save(os.path.join(OUT, "bound_%d_%d.npy" % (y, qp)),
                np.stack([oi, vqp, L, G, amin, bs, bp, pad,
                          adm.astype(np.int64)], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["corpus", "family", "all"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd in ("corpus", "all"):
        cmd_corpus(a)
    if a.cmd in ("family", "all"):
        cmd_family(a)
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
