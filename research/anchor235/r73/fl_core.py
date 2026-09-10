"""fl_core.py -- shared helpers for node 4.i.b.ii.c (the fusion lemma), research/proof/fusion_lemma.md.

Everything is exact integer arithmetic.  Objects, by construction:

  column k = (6k-1, 6k+1); gear g with tooth v strikes k iff k = +-v (mod g); the real tooth is
  u_g = 6^{-1} mod g.  An opening is a column no gear strikes; gaps are the distances between
  consecutive openings (cyclic over one period).

  Letters of the next gear q' with tooth v (d = 2v mod q'): a gap value is PAD (0), UP (d),
  DOWN (-d) mod q', else BAD.  A word is LEGAL iff no letter is BAD and no two consecutive
  nonzero letters are equal.  L = the longest realised legal word; a MAXIMAL word is a realised
  legal word of length L.

  For a maximal word m the NEIGHBOURHOOD is read off the realised (L+2)-windows (a, m, c):
      P(m) = max a,   S(m) = max c,   N(m) = max (a + c) over one window,
  and R = max over m of P(m) + |m| + S(m) is the relaxed J_max-term of B_{L+1}
  (fusion_lemma.md, Theorem A).
"""
import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]


def u_of(g):
    return pow(6, -1, g)


def real_teeth(gears):
    return {g: u_of(g) for g in gears}


def sieve(teeth):
    """teeth: {gear: v}.  Returns (P, gaps) with gaps the cyclic gap array of the machine."""
    P = 1
    for g in teeth:
        P *= g
    blocked = np.zeros(P, dtype=bool)
    for g, v in teeth.items():
        blocked[v % g::g] = True
        blocked[(-v) % g::g] = True
    O = np.flatnonzero(~blocked).astype(np.int64)
    gaps = np.empty(O.size, dtype=np.int64)
    gaps[:-1] = O[1:] - O[:-1]
    gaps[-1] = O[0] + P - O[-1]
    return P, gaps


def letter_codes(vals, q, d):
    """0 PAD, 1 UP, 2 DOWN, 3 BAD."""
    vals = np.asarray(vals, dtype=np.int64)
    r = vals % q
    lt = np.full(vals.shape, 3, dtype=np.int8)
    lt[r == 0] = 0
    lt[r == d] = 1
    lt[r == (-d) % q] = 2
    return lt


def is_legal(word, q, d):
    prev = 0
    for v in word:
        c = int(letter_codes(np.array([v]), q, d)[0])
        if c == 3:
            return False
        if c != 0:
            if c == prev:
                return False
            prev = c
    return True


def consistent_starts(word, q, d):
    """The start residues t in {0, d} (x_1 + z = t) from which every opening of the word is
    struck; returns the list of (t_start, t_end)."""
    out = []
    for t0 in (0, d % q):
        t = t0
        ok = True
        for v in word:
            t = (t + int(v)) % q
            if t not in (0, d % q):
                ok = False
                break
        if ok:
            out.append((t0, t))
    return out


def flank_struck(a, c, starts, q, d):
    """For each consistent start (t0, t1) of the middle word: is the left flank opening struck
    (t0 - a in {0, d}) or the right (t1 + c in {0, d})?  Returns the list of (left, right)."""
    S = {0, d % q}
    return [(((t0 - int(a)) % q) in S, ((t1 + int(c)) % q) in S) for (t0, t1) in starts]


def longest_legal_run(gaps, q, d):
    """L: the longest run of consecutive gaps (cyclic) forming a legal word."""
    lt = letter_codes(gaps, q, d)
    n = lt.size
    ext = np.concatenate([lt, lt])
    best = 0
    run = 0
    prev = 0
    for i in range(2 * n):
        c = int(ext[i])
        if c == 3 or (c != 0 and c == prev):
            run = 0 if c == 3 else 1
            prev = 0 if c == 3 else c
        else:
            run += 1
            if c != 0:
                prev = c
        if run > best:
            best = run
    return min(best, n)


def windows_from_gaps(gaps, K):
    """All distinct cyclic K-windows of the gap array, as an (n, K) int64 array with counts."""
    n = gaps.size
    ext = np.concatenate([gaps, gaps[:K]])
    win = np.stack([ext[i:i + n] for i in range(K)], axis=1)
    uniq, cnt = np.unique(win, axis=0, return_counts=True)
    return uniq, cnt


def windows_from_dict(win, K):
    """The K-windows of a row dictionary (rows are full-depth windows; a 0 terminates a row)."""
    good = np.all(win[:, :K] != 0, axis=1)
    uniq = np.unique(win[good, :K].astype(np.int64), axis=0)
    return uniq, int((~good).sum())


def maximal_words(windowsL, q, d):
    """The realised legal words of length L, from the realised L-windows."""
    out = []
    for row in windowsL:
        w = tuple(int(x) for x in row)
        if is_legal(w, q, d):
            out.append(w)
    return sorted(set(out))


def neighbourhood(windowsL2, m):
    """P, S, N of the maximal word m off the realised (L+2)-windows (a, m, c)."""
    L = len(m)
    if L == 0:
        rows = windowsL2
    else:
        mid = np.array(m, dtype=np.int64)
        rows = windowsL2[np.all(windowsL2[:, 1:L + 1] == mid, axis=1)]
    if rows.shape[0] == 0:
        return None
    a = rows[:, 0]
    c = rows[:, L + 1]
    P = int(a.max())
    S = int(c.max())
    N = int((a + c).max())
    iN = int(np.argmax(a + c))
    preds = sorted(set(int(x) for x in a))
    succs = sorted(set(int(x) for x in c))
    return {"P": P, "S": S, "N": N, "N_witness": (int(a[iN]), int(c[iN])),
            "joint_realised": bool(np.any((a == P) & (c == S))),
            "best_c_given_P": int(c[a == P].max()), "best_a_given_S": int(a[c == S].max()),
            "preds": preds, "succs": succs, "rows": int(rows.shape[0])}


def analyse(windowsL2, windowsL, q, d, L):
    """Everything the branch reads at one step: the maximal words, their neighbourhoods, R,
    Q*_{J_max} (no phase quantifier), the fuse check on every flank."""
    words = maximal_words(windowsL, q, d) if L > 0 else [()]
    table = []
    R = None
    Qstar = None
    fuse_fail = 0
    for m in words:
        nb = neighbourhood(windowsL2, m)
        if nb is None:
            continue
        span = int(sum(m))
        starts = consistent_starts(m, q, d)
        # every predecessor and successor of m, at every consistent phase, must be unstruck
        for a in nb["preds"]:
            for (l, r) in flank_struck(a, 0, starts, q, d):
                if l:
                    fuse_fail += 1
        for c in nb["succs"]:
            for (l, r) in flank_struck(0, c, starts, q, d):
                if r:
                    fuse_fail += 1
        rel = nb["P"] + span + nb["S"]
        ex = nb["N"] + span
        row = {"m": list(m), "span": span, "P": nb["P"], "S": nb["S"], "N": nb["N"],
               "N_witness": list(nb["N_witness"]), "relaxed": rel, "exact": ex,
               "joint_realised": nb["joint_realised"],
               "best_c_given_P": nb["best_c_given_P"], "best_a_given_S": nb["best_a_given_S"],
               "starts": starts, "rows": nb["rows"]}
        table.append(row)
        R = rel if R is None else max(R, rel)
        Qstar = ex if Qstar is None else max(Qstar, ex)
    return {"L": L, "J_max": L + 2, "words": words, "table": table, "R": R,
            "Qstar_Jmax": Qstar, "fuse_fail": fuse_fail}


def mirror_check(table):
    """P(m) = S(reverse m) for every maximal word (the column-0 mirror)."""
    byw = {tuple(r["m"]): r for r in table}
    bad = 0
    for w, r in byw.items():
        rv = tuple(reversed(w))
        if rv in byw and byw[rv]["S"] != r["P"]:
            bad += 1
    return bad
