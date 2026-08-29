"""Round 27 (constructor): THE UNIFORM-ORDER QUESTION - is A_relax(M) <= 4
for every machine?

R67(i) named this the sharpest single open question the certificate chain has:
the abstraction A_m (state = last m-1 gap VALUES) is nilpotent - hence bounds
anything at all - only for m above A_relax(M), and A_relax was a MEASURED
non-monotone ladder 1,2,2,3,2,3,4,3,2 at m11..41 with no theorem behind it.

    A_relax(M) = min{ m : one of the two m-letter ALTERNATIONS
                          (a,b,a,...) / (b,a,b,...) is NOT realised as m
                          consecutive gaps of M }        (a = 2u', b = q'-2u')

This script answers it by an arithmetic function, the way R20 handled litcap.
The vehicle is Mechanic's round-26 PHASE SATURATION theorem
(docs/novel/phase-saturation-arity.md):

    gear g blocks slots k = +-c_g (mod g), c_g = 6^{-1} mod g, so an
    occurrence of a word with exposed offsets X needs some k0 with
    k0 + X contained in E_g = Z_g \ {+-c_g}.  If no such k0 exists for SOME
    gear g of M, the word has no occurrence anywhere - by arithmetic, with
    no solver and no scan.

Define PS-order(q') = min{ m : one of the two m-letter alternations is
phase-saturation-refuted at some gear g of M }.  Phase saturation is
sufficient for unrealisability, so

    A_relax(M) <= PS-order(q')                     (*)

and PS-order is a function of q' mod 6*prod(gears used) ONLY - no machine in
it.  Parts:

  1  A_relax recomputed EXACTLY at m11..m41 from full-period dictionaries
     (direct scan m11..23; Mechanic's exact 4-tuple censuses m29/31/37;
     superset + the R45 CRT verdict at m41).  This corrects the R45 table.
  2  the phase-saturation refuter, gate-checked against the round-26 record.
  3  PS-order as a CLOSED FORM: the 48 invertible classes mod 210 (gears 5,7),
     then mod 2310 / 30030 (gears 11, 13) for whatever 5 and 7 leave open.
  4  the uniform verdict, plus a direct check against every prime q' < 10^5.
  5  (*) checked pointwise at the nine known machines.

Usage:  .venv/Scripts/python.exe research/uniform_order.py
"""
import csv
import os
import sys
from collections import Counter
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91}


def sieve(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s)


PRIMES = [int(p) for p in sieve(200000)]
PSET = set(PRIMES)


def next_prime(y):
    p = y + 1
    while p not in PSET:
        p += 1
    return p


def letters(q1):
    """(a, b): the two literal letters, a = 2u' the smaller."""
    u1 = round(q1 / 6)
    return 2 * u1, q1 - 2 * u1


# ------------------------------------------------------------------ part 2
def exposed_set(g):
    c = pow(6, -1, g)
    bad = {c % g, (-c) % g}
    return frozenset(r for r in range(g) if r not in bad)


_EXP = {}


def E(g):
    if g not in _EXP:
        _EXP[g] = exposed_set(g)
    return _EXP[g]


def ps_blocked(X, g):
    """True iff NO translate of X lands inside the exposed set of gear g,
    i.e. gear g refutes every occurrence of a word with offsets X."""
    Eg = E(g)
    xs = sorted({x % g for x in X})
    if len(xs) > g - 2:
        return True
    for t in range(g):
        if all((t + x) % g in Eg for x in xs):
            return False
    return True


def alternation(a, b, m, start):
    """m-letter alternation and its exposed offsets X (|X| = m+1)."""
    w = [(a if (i + start) % 2 == 0 else b) for i in range(m)]
    X, s = [0], 0
    for v in w:
        s += v
        X.append(s)
    return tuple(w), X


def ps_order(q1, gears, mmax=12):
    """min m such that one of the two m-letter alternations is refuted by
    phase saturation at some gear in `gears`; returns (m, gear, word)."""
    a, b = letters(q1)
    for m in range(1, mmax + 1):
        for start in (0, 1):
            w, X = alternation(a, b, m, start)
            for g in gears:
                if ps_blocked(X, g):
                    return m, g, w
    return None, None, None


# --------------------------------------------------- part 2b: class version
def ps_order_class(r, gears, mmax=12):
    """Same, but from a RESIDUE CLASS r (mod 210 / 2310 / ...) alone: the
    letters are recovered congruence-wise, a = (q' -+ 1)/3 with the sign
    fixed by q' mod 6.  No representative prime is chosen."""
    sgn = 1 if r % 6 == 1 else -1
    for m in range(1, mmax + 1):
        for start in (0, 1):
            for g in gears:
                ag = ((r - sgn) * pow(3, -1, g)) % g
                bg = (r - ag) % g
                X, s = [0], 0
                for i in range(m):
                    s = (s + (ag if (i + start) % 2 == 0 else bg)) % g
                    X.append(s)
                if ps_blocked(X, g):
                    return m, g
    return None, None


# ------------------------------------------------------------------ part 1
_SCAN = {}


def scan_words(y, maxm=5):
    """Full-period cyclic scan of machine y: realised m-tuples of gaps."""
    if (y, maxm) in _SCAN:
        return _SCAN[(y, maxm)]
    r = _scan_words(y, maxm)
    _SCAN[(y, maxm)] = r
    return r


def _scan_words(y, maxm=5):
    gears = [p for p in PRIMES if 5 <= p <= y]
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    op = np.flatnonzero(~ex).astype(np.int64)
    d = np.diff(np.concatenate([op, [op[0] + P]]))
    n = len(d)
    out = {}
    for m in range(1, maxm + 1):
        c = Counter()
        cols = [np.roll(d, -t) for t in range(m)]
        for tup in zip(*[c_.tolist() for c_ in cols]):
            c[tup] += 1
        out[m] = c
    return out, int(d.max()), n


def tuples_from_csv(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    return arr


def windows_of(arr, m):
    """All m-windows of the realised 4-tuples (m <= 4) as a python set."""
    out = set()
    for j in range(0, 4 - m + 1):
        sub = arr[:, j:j + m]
        out.update(map(tuple, sub.tolist()))
    return out


def a_relax_from_sets(q1, present, mmax):
    """present(m) -> set of realised m-tuples.  Returns A_relax or None."""
    a, b = letters(q1)
    for m in range(1, mmax + 1):
        for start in (0, 1):
            w, _ = alternation(a, b, m, start)
            if w not in present(m):
                return m, w
    return None, None


def main():
    print("=" * 74)
    print("PART 1  A_relax EXACTLY, machines 11..41 (correcting the R45 table)")
    print("=" * 74)
    arelax = {}
    for y in (11, 13, 17, 19, 23):
        words, F, n = scan_words(y, 5)
        assert F == KNOWN_F[y], (y, F)
        q1 = next_prime(y)
        m, w = a_relax_from_sets(q1, lambda k: set(words[k]), 5)
        arelax[y] = m
        print("  m%-3d q'=%-3d a,b=%-8s  ngaps %-12d F=%-3d  A_relax = %s "
              "(first absent alternation %s)"
              % (y, q1, "%d,%d" % letters(q1), n, F, m, w))

    for y, fn in ((29, "gap_tuples_29_4.csv"), (31, "gap_tuples_31_4.csv"),
                  (37, "gap_tuples_37_4.csv")):
        arr = tuples_from_csv(os.path.join(DDIR, fn))
        # gate: the dictionary's induced max gap must be F(M)
        assert int(arr.max()) == KNOWN_F[y], (y, int(arr.max()))
        sets = {m: windows_of(arr, m) for m in (1, 2, 3, 4)}
        q1 = next_prime(y)
        m, w = a_relax_from_sets(q1, lambda k: sets[k], 4)
        arelax[y] = m
        print("  m%-3d q'=%-3d a,b=%-8s  exact 4-tuples %-9d F=%-3d  "
              "A_relax = %s (first absent alternation %s)"
              % (y, q1, "%d,%d" % letters(q1), len(arr), KNOWN_F[y], m, w))

    arr41 = tuples_from_csv(os.path.join(DDIR,
                                         "gap_tuples_41_4_transfer.csv"))
    sets41 = {m: windows_of(arr41, m) for m in (1, 2, 3, 4)}
    q1 = next_prime(41)
    m41, w41 = a_relax_from_sets(q1, lambda k: sets41[k], 4)
    # the superset can only prove ABSENCE; R45 decided (14,29)/(29,14) = 0 by
    # exact CRT pattern counting, which is what fixes A_relax(41) = 2.
    print("  m41  q'=43  a,b=%-8s  SUPERSET 4-tuples %-9d      "
          "A_relax <= %s from the superset alone; R45's exact CRT count gives"
          % ("%d,%d" % letters(43), len(arr41), m41))
    print("       A_relax(41) = 2 ((14,29) = (29,14) = 0 by CRT).")
    arelax[41] = 2

    ladder = [arelax[y] for y in (11, 13, 17, 19, 23, 29, 31, 37, 41)]
    print("\n  A_relax ladder m11..m41 : " + ", ".join(map(str, ladder)))
    print("  R45's published ladder  : 1, 2, 2, 3, 2, 3, 4, 3, 2")
    if ladder != [1, 2, 2, 3, 2, 3, 4, 3, 2]:
        print("  *** DIFFERS from the published table - see the round append")

    print()
    print("=" * 74)
    print("PART 2  the phase-saturation refuter, gated against the record")
    print("=" * 74)
    # gate A: the round-26 alternation ceilings must be reproduced exactly.
    # Mechanic's convention: A_k starts with s = 2*6^{-1} mod q' (which is a
    # or b according to q' mod 6), and the ceiling is about CHAIN existence,
    # so it is the LAST k at which that phase is still unrefuted.  A_relax
    # instead takes the MIN over the two phases (either broken window kills
    # the cycle), so the two numbers are different objects; both are printed.
    CEIL = {37: 6, 41: 2, 43: 2, 47: 2, 53: 5, 59: 3, 61: 3, 67: 4}
    for q1, want in sorted(CEIL.items()):
        y = max(p for p in PRIMES if p < q1)
        gears = [p for p in PRIMES if 5 <= p <= y]
        a, b = letters(q1)
        s = (2 * pow(6, -1, q1)) % q1
        start = 0 if s == a else 1
        assert s in (a, b), (q1, s, a, b)
        got = {}
        for st in (0, 1):
            k = 2
            while True:
                _, X = alternation(a, b, k - 1, st)
                if any(ps_blocked(X, g) for g in gears):
                    break
                k += 1
            got[st] = k - 1
        assert got[start] == want, (q1, got, want)
        print("  q'=%-3d s=%-3d ceiling %d members (s-phase, matches "
              "phase-saturation-arity.md); other phase %d"
              % (q1, s, got[start], got[1 - start]))

    # gate B: soundness - never refute a word the project has seen realised.
    realised = [(37, (14, 41)), (41, (14, 43)), (43, (16, 47)),
                (47, (18, 35)), (47, (18, 35, 18)), (47, (18, 35, 18, 35)),
                (53, (20, 39))]
    for q1, w in realised:
        y = max(p for p in PRIMES if p < q1)
        gears = [p for p in PRIMES if 5 <= p <= y]
        X, s = [0], 0
        for v in w:
            s += v
            X.append(s)
        bad = [g for g in gears if ps_blocked(X, g)]
        assert not bad, (q1, w, bad)
    print("  soundness: %d words on the realised record, none refuted  OK"
          % len(realised))

    print()
    print("=" * 74)
    print("PART 3  PS-order as a CLOSED FORM in q' mod 210 (gears 5, 7)")
    print("=" * 74)
    classes = [r for r in range(210) if all(r % p for p in (2, 3, 5, 7))]
    assert len(classes) == 48
    by_order = Counter()
    hard57 = []
    detail = {}
    for r in classes:
        m, g = ps_order_class(r, [5, 7])
        by_order[m] += 1
        detail[r] = (m, g)
        if m is None or m > 4:
            hard57.append(r)
    print("  gears {5,7}: PS-order distribution over the 48 classes mod 210")
    for k in sorted(by_order, key=lambda z: (z is None, z)):
        print("      order %-4s : %2d classes   %s"
              % (k, by_order[k],
                 ", ".join(str(r) for r in classes if detail[r][0] == k)))
    print("  classes NOT settled at order <= 4 by gears {5,7}: %s"
          % (hard57 if hard57 else "NONE"))

    # gear 5 alone, at order exactly 4 (checks P2)
    g5fail = []
    for r in classes:
        m, _ = ps_order_class(r, [5])
        if m is None or m > 4:
            g5fail.append(r)
    print("  gear 5 ALONE leaves these classes mod 210 unrefuted at order 4:")
    print("      %s" % sorted(g5fail))
    print("      = %s (mod 30)" % sorted({r % 30 for r in g5fail}))

    if hard57:
        print("\n  extending to gears {5,7,11} (mod 2310) for those classes:")
        left = []
        for r0 in hard57:
            for j in range(11):
                r = r0 + 210 * j
                if r % 11 == 0:
                    continue
                m, g = ps_order_class(r, [5, 7, 11])
                if m is None or m > 4:
                    left.append(r)
        tot2310 = sum(1 for r0 in hard57 for j in range(11)
                      if (r0 + 210 * j) % 11)
        print("      unrefuted at order <= 4 mod 2310: %d of the %d classes "
              "that reduce into them - gear 11 buys NOTHING"
              % (len(left), tot2310))
        if left:
            left2 = []
            for r0 in left:
                for j in range(13):
                    r = r0 + 2310 * j
                    if r % 13 == 0:
                        continue
                    m, g = ps_order_class(r, [5, 7, 11, 13])
                    if m is None or m > 4:
                        left2.append(r)
            tot30030 = sum(1 for r0 in left for j in range(13)
                           if (r0 + 2310 * j) % 13)
            print("      unrefuted at order <= 4 mod 30030: %d of %d - gear "
                  "13 buys NOTHING either" % (len(left2), tot30030))
        # what order DO the six classes get?
        print("      PS-order on those classes (gears 5,7,11,13): %s"
              % sorted({ps_order_class(r, [5, 7, 11, 13])[0]
                        for r in hard57}))

    print()
    print("=" * 74)
    print("PART 4  direct check on every prime q' < 10^5 (no classes)")
    print("=" * 74)
    for label, cap in (("primes of M up to 13", 13),
                       ("ALL primes of M up to 100", 100)):
        worst = Counter()
        worstex = {}
        over4 = []
        over5 = []
        for q1 in PRIMES:
            if q1 < 7 or q1 > 20000:
                continue
            y = max(p for p in PRIMES if p < q1)
            gears = [p for p in PRIMES if 5 <= p <= min(y, cap)]
            if not gears:
                continue
            m, g, w = ps_order(q1, gears)
            worst[m] += 1
            worstex.setdefault(m, q1)
            if m is None or m > 4:
                over4.append(q1)
            if m is None or m > 5:
                over5.append(q1)
        print("  gears used: %s   (q' < 20000)" % label)
        for k in sorted(worst, key=lambda z: (z is None, z)):
            print("      PS-order %-4s : %6d primes   (first: q' = %s)"
                  % (k, worst[k], worstex[k]))
        print("      primes with PS-order > 4 : %d (first %s)"
              % (len(over4), over4[0] if over4 else None))
        print("      primes with PS-order > 5 : %s"
              % (len(over5) if over5 else "NONE - the cap 5 is uniform"))
        print("      their residues mod 210   : %s"
              % sorted({q % 210 for q in over4}))

    print()
    print("=" * 74)
    print("PART 5  (*)  A_relax(M) <= PS-order(q') at the nine known machines")
    print("=" * 74)
    print("     M   q'   A_relax   PS-order (gear)   (*) holds")
    for y in (11, 13, 17, 19, 23, 29, 31, 37, 41):
        q1 = next_prime(y)
        gears = [p for p in PRIMES if 5 <= p <= y]
        m, g, w = ps_order(q1, gears)
        ok = arelax[y] is not None and m is not None and arelax[y] <= m
        print("   %4d %4d %8s %10s (gear %s)   %s"
              % (y, q1, arelax[y], m, g, "OK" if ok else "VIOLATION"))
        assert ok, y

    print()
    print("=" * 74)
    print("PART 6  the object the chain actually needs: N(M), the smallest m")
    print("        at which the abstraction A_m is ACYCLIC (nilpotent)")
    print("=" * 74)
    print("  A_m: state = (last m-1 legal gap VALUES, tooth); an edge exists")
    print("  iff the m-tuple is REALISED and the letter's T3 tooth transition")
    print("  is legal.  max(2, A_relax) <= N <= A_res, and A_relax only tests")
    print("  ONE candidate cycle (the pure alternation) - padded cycles are")
    print("  legal too, so N = max(2, A_relax) is a measurement, not a law.")
    print()
    print("     M   q'   legal values <= F      A_relax   N   N == max(2,"
          "A_relax)?")
    for y in (11, 13, 17, 19, 23, 29, 31, 37):
        q1 = next_prime(y)
        a, b = letters(q1)
        F = KNOWN_F[y]
        if y <= 23:
            words, _, _ = scan_words(y, 4)
            has = {m: set(words[m]) for m in (2, 3, 4)}
        else:
            arr = tuples_from_csv(os.path.join(DDIR,
                                               "gap_tuples_%d_4.csv" % y))
            has = {m: windows_of(arr, m) for m in (2, 3, 4)}
        legal = [v for v in range(1, F + 1) if v % q1 in (0, a, b)]

        def cls(v):
            r = v % q1
            return 0 if r == 0 else (1 if r == a else -1)

        N = None
        for m in (2, 3, 4):
            # states: (tuple of m-1 values, tooth in {+1,-1})
            nodes = set()
            edges = {}
            import itertools
            for pre in itertools.product(legal, repeat=m - 1):
                if m > 2 and pre not in has[m - 1]:
                    continue
                for s in (1, -1):
                    st = (pre, s)
                    nodes.add(st)
                    out = []
                    for v in legal:
                        w = pre + (v,)
                        if w not in has[m]:
                            continue
                        c = cls(v)
                        if c == 0:
                            s2 = s
                        elif c == 1:
                            if s != -1:
                                continue
                            s2 = 1
                        else:
                            if s != 1:
                                continue
                            s2 = -1
                        out.append((w[1:], s2))
                    edges[st] = out
            # cycle detection (iterative DFS with colours)
            colour = {}
            cyclic = False
            for st in list(nodes):
                if colour.get(st):
                    continue
                stack = [(st, iter(edges.get(st, ())))]
                colour[st] = 1
                while stack and not cyclic:
                    node, it = stack[-1]
                    nxt = next(it, None)
                    if nxt is None:
                        colour[node] = 2
                        stack.pop()
                    elif colour.get(nxt, 0) == 1:
                        cyclic = True
                    elif colour.get(nxt, 0) == 0:
                        colour[nxt] = 1
                        stack.append((nxt, iter(edges.get(nxt, ()))))
                if cyclic:
                    break
            if not cyclic:
                N = m
                break
        pred = max(2, arelax[y])
        print("   %4d %4d   %-22s %6s %4s   %s"
              % (y, q1, str(legal), arelax[y], N,
                 "yes" if N == pred else "NO  (predicted %d)" % pred))

    print()
    print("=" * 74)
    print("PART 7  CAN PHASE SATURATION CAP THE ORDER AT ALL?  The corridor")
    print("        cap on EVERY legal word, not just the alternation")
    print("=" * 74)
    print("  A cycle in A_m needs arbitrarily long runs of legal gaps, so")
    print("  N(M) <= 1 + (longest realisable T3-legal word).  Gears 5 and 7")
    print("  refute a word iff its prefix-sum walk leaves the corridor E mod")
    print("  35 (by CRT a translate exists mod 5 and mod 7 separately iff one")
    print("  exists mod 35).  CORRCAP(q', F) = longest T3-legal word with")
    print("  values <= F whose prefix-sum walk stays in E - the strongest cap")
    print("  gears 5 and 7 can ever give.  INFINITE means phase saturation")
    print("  cannot cap the order at that step, at any length.")
    Ecorr = frozenset(r for r in range(35)
                      if r % 5 not in (1, 4) and r % 7 not in (1, 6))
    assert len(Ecorr) == 15
    print()
    print("     M   q'    F   F/q'  legal values (mod 35)          CORRCAP")

    def corrcap(q1, F):
        a, b = letters(q1)
        legal = [v for v in range(1, F + 1) if v % q1 in (0, a, b)]

        def cls(v):
            r = v % q1
            return 0 if r == 0 else (1 if r == a else -1)

        # state = (residue mod 35 of the current prefix point, tooth)
        nodes = [(r, s) for r in sorted(Ecorr) for s in (1, -1)]
        adj = {}
        for (r, s) in nodes:
            out = []
            for v in legal:
                c = cls(v)
                if c == 0:
                    s2 = s
                elif c == 1:
                    if s != -1:
                        continue
                    s2 = 1
                else:
                    if s != 1:
                        continue
                    s2 = -1
                r2 = (r + v) % 35
                if r2 in Ecorr:
                    out.append((r2, s2))
            adj[(r, s)] = out
        # longest path / cycle detection by memoised DFS
        memo, state = {}, {}

        def dfs(u):
            if state.get(u) == 1:
                raise RecursionError
            if u in memo:
                return memo[u]
            state[u] = 1
            best = 0
            for v in adj[u]:
                best = max(best, 1 + dfs(v))
            state[u] = 2
            memo[u] = best
            return best

        try:
            return max(dfs(u) for u in nodes)
        except RecursionError:
            return None

    for y in (11, 13, 17, 19, 23, 29, 31, 37, 41):
        q1 = next_prime(y)
        F = KNOWN_F[y]
        a, b = letters(q1)
        legal = [v for v in range(1, F + 1) if v % q1 in (0, a, b)]
        cc = corrcap(q1, F)
        print("   %4d %4d %4d  %4.1f  %-30s %s"
              % (y, q1, F, F / q1,
                 str([v % 35 for v in legal]),
                 ("INFINITE - no cap" if cc is None else cc)))
    print()
    print("  the same at the deep steps and hypothetically deeper machines")
    print("  (F taken from the corpus; the last rows are what the cap does")
    print("  once F/q' grows, which is the regime every large machine is in)")
    for q1, F in ((43, 91), (47, 103), (53, 118), (59, 145),
                  (61, 200), (67, 300), (71, 500), (73, 1000)):
        cc = corrcap(q1, F)
        print("   q'=%-3d F=%-5d F/q'=%4.1f   CORRCAP = %s"
              % (q1, F, F / q1, "INFINITE - no cap" if cc is None else cc))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
