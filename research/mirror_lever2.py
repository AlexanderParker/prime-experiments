"""mirror_lever2.py - LATERAL round 26.

THE PARITY LEVER, PART II: the involution census, the exceptional window
located in SLOT space (scan-free), and the spectral face of the same law.

Round 25 (item 46) proved: the opening set is closed under k -> -k, so each
depth-j window census is mirror-invariant, N = prod(q-2) is odd, and hence
EXACTLY ONE depth-j window is self-mirror - located by its INDEX
t_j = -j/2 (mod N).  An index is only useful if you have enumerated the
period.  This round replaces the index by an ADDRESS.

PART A  the FULL affine symmetry group of the opening set is (Z/2)^m
        (multiplication by c = +-1 mod every gear, no translation), and only
        c = +-1 mod P preserves adjacency.  So the mirror is the ONLY
        involution that acts on WINDOWS: the lever gives mod 2 and can never
        give mod 4.  Proof + brute-force gate.

PART B  the self-mirror depth-j window is the window CENTRED ON SLOT 0
        (j even) or ON THE ANTIPODE (j odd).  Hence g_j* = 2 o_{j/2}
        (j even) and g_j* = 2 b_{(j+1)/2} - P (j odd), where o_i / b_i are
        the openings just above 0 / just above (P-1)/2.  Both are computed
        by sieving ~F slots: NO PERIOD SCAN, ANY MACHINE.
        COROLLARY  g_j* = j (mod 2).

PART C  therefore W_j(g) is EVEN for every g of the wrong parity, with NO
        computation at all, and even for every g of the right parity except
        the single value g_j*.  THE LEVER: any argument capping the count of
        depth-j windows of sum g at ONE proves there are NONE, for every
        g != g_j*.

PART D  words and tuples: #occ(w) = #occ(reverse w) exactly, so every
        realisability census is reverse-closed and only half of each
        non-palindromic pair needs deciding.  Audited against the project's
        own A_kill logs, with the measured cost of not having known it.

PART E  THE FIXED-POINT CRITERION.  For a PALINDROMIC tuple w of span s the
        occurrence set is mirror-invariant with exactly one candidate fixed
        point, the address k_w = -s/2 (mod P).  So #occ(w) is ODD iff w
        occurs at k_w - an O(#gears) test replacing a parity question.

PART F  the spectral face (backlog U4).  In the path decomposition
        spec(A) = union over gaps g of {2cos(pi j/(g+1))}, the multiplicity
        of 2cos(pi a/b) is #{gaps = -1 mod b} - INDEPENDENT of a.  So the
        eigenvalue multiplicities of A are the gap histogram's residue-class
        counts, invertible by Mobius over multiples, and their parity is
        Part C's.  The published |Farey(F+1)|-2 level count is corrected:
        it assumes every gap value 1..F is realised, and HOLES BREAK IT.

PART G  every gear is parity-obstructed.  Because W_1(1) is the ONLY odd entry
        of the gap histogram, N_1 (mod p) is odd and every other residue class
        count is even, for EVERY modulus - so alpha_1(p) is odd and the pole
        phase of round 21 is unattainable at every gear, not only at gear 5.
        This CORRECTS round 25's prediction P3.

Usage:  python mirror_lever2.py [--parts ABCDEFG] [--maxy 29]
"""

import argparse
import csv
import os
from math import gcd

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

NASSERT = 0


def ok(cond, msg):
    global NASSERT
    assert cond, "ASSERT FAILED: " + msg
    NASSERT += 1
    print(f"    [assert ok] {msg}")


# ---------------------------------------------------------------- machine
def primes_upto(y):
    out = []
    for n in range(5, y + 1):
        if all(n % d for d in range(2, int(n ** 0.5) + 1)):
            out.append(n)
    return out


def tooth(q):
    """u' with 6 u' = +-1 (mod q); teeth are +-u'."""
    u = round(q / 6)
    assert (6 * u - 1) % q == 0 or (6 * u + 1) % q == 0, q
    return u


def machine(y):
    gs = primes_upto(y)
    P = 1
    for q in gs:
        P *= q
    return gs, [tooth(q) for q in gs], P


def is_open(k, gs, us):
    for q, u in zip(gs, us):
        r = k % q
        if r == u or r == q - u:
            return False
    return True


def openings_after(start, count, gs, us):
    """the first `count` openings strictly greater than `start`."""
    out = []
    k = start + 1
    while len(out) < count:
        if is_open(k, gs, us):
            out.append(k)
        k += 1
    return out


def blocked_array(y):
    gs, us, P = machine(y)
    b = np.zeros(P, dtype=bool)
    for q, u in zip(gs, us):
        b[u::q] = True
        b[(q - u) % q::q] = True
    return ~b, P            # returns OPEN mask


F_KNOWN = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145}
# ladder steps M -> M + q' and their (D) budgets F(M) + q'
STEPS = [(11, 13), (13, 17), (17, 19), (19, 23), (23, 29), (29, 31),
         (31, 37), (37, 41), (41, 43), (43, 47), (47, 53)]


# ---------------------------------------------------------------- part A
def partA():
    print("\n=== PART A: the FULL affine symmetry group of the opening set ===")
    print("""
  CLAIM (proof, then gate).  Let O = {k : k != +-u_q (mod q) for every gear}.
  An affine map k -> c k + b (c a unit mod P) maps O onto O iff for every
  gear q it maps the tooth pair {+-u_q} onto itself.  Summing the two
  requirements c u + b = -+ u gives 2b = 0 (mod q), and q is odd, so b = 0;
  then c u = +- u with u invertible gives c = +-1 (mod q).  Conversely every
  such c works.  So

      Aff(O) = { k -> c k : c = +-1 (mod q) for each gear q }  =  (Z/2)^m,

  of order 2^m (m = #gears), with the MIRROR c = -1 as its all-flip element.
  Fixed points of the flip on a gear subset S: k = -k (mod q) for q in S,
  i.e. q | k, so exactly P / prod_{q in S} q of them - ONE for S = all.""")
    for y in (11, 13):
        gs, us, P = machine(y)
        m = len(gs)
        op, _ = blocked_array(y)
        idx = np.arange(P)
        # predicted group
        pred = set()
        for mask in range(1 << m):
            c = 0
            for i, q in enumerate(gs):          # CRT: c = +-1 mod q
                tgt = q - 1 if (mask >> i) & 1 else 1
                M = P // q
                c += tgt * M * pow(M, -1, q)
            pred.add(c % P)
        found = []
        if y == 11:
            for c in range(1, P):
                if gcd(c, P) != 1:
                    continue
                cimg = (c * idx) % P
                for b in range(P):
                    if np.array_equal(op[(cimg + b) % P], op):
                        found.append((c, b))
            cs = sorted({c for c, b in found})
            bs = sorted({b for c, b in found})
            nunits = sum(1 for c in range(1, P) if gcd(c, P) == 1)
            ok(bs == [0], f"m{y}: every affine symmetry has b = 0 "
                          f"(brute force over ALL {nunits} units x {P} "
                          f"shifts = {nunits*P:,} maps)")
            ok(sorted(pred) == cs,
               f"m{y}: Aff(O) = {{c = +-1 mod each gear}}, |Aff| = "
               f"{len(cs)} = 2^{m}")
        else:
            for c in range(1, P):
                if gcd(c, P) != 1:
                    continue
                if np.array_equal(op[(c * idx) % P], op):
                    found.append((c, 0))
            cs = sorted({c for c, b in found})
            ok(sorted(pred) == cs,
               f"m{y}: the c-exhaustive symmetry set = 2^{m} = {len(cs)} "
               f"predicted units")
        # fixed-point counts
        for c in cs[:8]:
            fx = int((((c - 1) * idx) % P == 0).sum())
            S = [q for q in gs if (c + 1) % q == 0]
            d = 1
            for q in S:
                d *= q
            predfx = P // d
            assert fx == predfx, (y, c, fx, predfx)
        ok(True, f"m{y}: fixed-point count of every element = P / prod_{{S}} q")
        # adjacency: which symmetries act on WINDOWS?
        ops = np.flatnonzero(op)
        keep = []
        for c in cs:
            img = np.sort((c * ops) % P)
            # c preserves adjacency iff the induced map on openings sends
            # consecutive openings to consecutive openings
            pos = {int(v): i for i, v in enumerate(ops)}
            n = len(ops)
            good = True
            for i in range(n):
                a1 = int((c * ops[i]) % P)
                a2 = int((c * ops[(i + 1) % n]) % P)
                if abs(pos[a2] - pos[a1]) % n not in (1, n - 1):
                    good = False
                    break
            if good:
                keep.append(c)
        ok(sorted(keep) == sorted({1, P - 1}),
           f"m{y}: ONLY c = +-1 preserves adjacency of openings, so the "
           f"mirror is the ONLY symmetry acting on windows")
    print("""
  A2 - THE CEILING WITHOUT THE AFFINE ASSUMPTION.  A map that acts on WINDOWS
  at all must preserve the CIRCULAR ORDER of Z_P, and the order-preserving
  bijections of a cycle are exactly the rotations k -> k+b, the reversing ones
  exactly the reflections k -> b-k.  Rotations: O+b = O needs {+-u}+b = {+-u}
  per gear, and adding the two equations gives 2b = 0, so b = 0.  Reflections:
  b-u = -u with b+u = u gives b = 0, while b-u = u with b+u = -u gives
  4u = 0 (mod q), impossible.  Either way b = 0 (mod q) for every gear.

      THE FULL SYMMETRY GROUP OF THE OPENING SET INSIDE THE CIRCLE Z_P IS
      {identity, mirror} = Z/2 - EXACTLY.

  So the parity lever is not merely the best symmetry lever available; it is
  the ONLY one.  Any mod-4 (or finer) counting argument must come from
  something that is not a symmetry of the opening set.""")
    for y in (11, 13):
        op, P = blocked_array(y)
        idx = np.arange(P)
        rot = [b for b in range(P) if np.array_equal(op[(idx + b) % P], op)]
        ref = [b for b in range(P) if np.array_equal(op[(b - idx) % P], op)]
        ok(rot == [0] and ref == [0],
           f"m{y}: brute force over ALL {2*P:,} rotations and reflections of "
           f"the circle Z_P: the only symmetries are the identity and the "
           f"mirror")
    print("""
  CONSEQUENCE - AN EXACT CEILING ON THE LEVER.  The window census carries a
  Z/2 action and nothing larger: 2^m affine symmetries exist but 2^m - 2 of
  them destroy consecutiveness, and outside the affine world there is nothing
  at all.  So "cap at one gives zero" is worth EXACTLY ONE UNIT and there is
  no mod-4 version to hope for.""")


# ---------------------------------------------------------------- part B
def gstar_table(y, J):
    """g_j* for j = 1..J, computed scan-free from the openings around slot 0
    and around the antipode."""
    gs, us, P = machine(y)
    need = (J + 1) // 2 + 1
    o = openings_after(0, need, gs, us)              # o[0] = o_1 etc
    a = (P - 1) // 2
    b = openings_after(a, need, gs, us)              # b[0] = b_1 > antipode
    out = {}
    for j in range(1, J + 1):
        if j % 2 == 0:
            out[j] = 2 * o[j // 2 - 1]
        else:
            out[j] = 2 * b[(j + 1) // 2 - 1] - P
    return out, o, b, P


def partB(maxy, J=12):
    print("\n=== PART B: the exceptional window, located in SLOT space ===")
    print("""
  THEOREM.  0 is an opening at every machine (u_q != 0), P is odd, and the
  mirror k -> -k fixes only slot 0.  A depth-j window is self-mirror iff its
  endpoint pair is {x, -x}; the arc joining them is then either the one
  THROUGH 0 or the one THROUGH THE ANTIPODE.  Counting openings on the arc:
      j EVEN  ->  the arc through 0 (0 is itself an opening, j/2 on each
                  side), endpoints +- o_{j/2},   g_j* = 2 o_{j/2};
      j ODD   ->  the arc through the antipode ((j+1)/2 openings each side,
                  none at the centre since P is odd), endpoints
                  P - b_{(j+1)/2} and b_{(j+1)/2},  g_j* = 2 b_{(j+1)/2} - P.
  Both need only the openings within F of slot 0 and of (P-1)/2.

  COROLLARY (free half of the law).   g_j* = j  (mod 2).""")
    ys = [y for y in (11, 13, 17, 19, 23, 29) if y <= maxy]
    print(f"\n  g_j* for j = 1..{J}  (scan-free; k_1 = first gap = g_2*/2)")
    hdr = "   y  " + "".join(f"{j:>6}" for j in range(1, J + 1))
    print(hdr)
    tabs = {}
    for y in (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53):
        t, o, b, P = gstar_table(y, J)
        tabs[y] = t
        print(f"  {y:3d}  " + "".join(f"{t[j]:>6}" for j in range(1, J + 1)))
        for j in range(1, J + 1):
            assert t[j] % 2 == j % 2, (y, j, t[j])
    ok(True, f"g_j* = j (mod 2) at every machine 11..53 and every j <= {J}")

    print("""
  AND THE j = 1 COLUMN IS THE CONSTANT 1 - A THEOREM, and it is the T3 law
  (item 29b) wearing a different hat.  Since P = 0 (mod q), the antipodal
  slot s = (P+1)/2 reduces mod every gear to inverse(2) = (q+1)/2.  Multiply
  by 6: 6s = 3(q+1) = 3 (mod q), while 6*(+-u) = +-1 by the tooth law.  So s
  is a tooth iff 3 = +-1 (mod q), i.e. q | 2 or q | 4 - impossible for q >= 5.
  SO THE ANTIPODAL SLOTS (P+-1)/2 ARE OPENINGS AT EVERY MACHINE and
      g_1* = 1  ALWAYS  (the antipodal gap is the shortest gap there is).
  Hence W_1(g) is EVEN for EVERY g >= 2, at every machine, with no side
  condition whatever: only the number of gaps of size 1 is odd.""")
    bad = []
    for y in (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 101, 211, 307):
        gs, us, P = machine(y)
        for q, u in zip(gs, us):
            s = (q + 1) // 2
            assert (6 * s - 3) % q == 0 and (6 * u - 1) % q * \
                ((6 * u + 1) % q) == 0, (q, u)
            if s in (u, q - u):
                bad.append((y, q))
        assert is_open((P + 1) // 2, gs, us) and is_open((P - 1) // 2, gs, us)
    ok(not bad, "6*(antipodal slot) = 3 and 6*(tooth) = +-1, so the antipodal "
                "slot is open at every gear of every machine up to y = 307 "
                "(also verified directly): g_1* = 1 always")

    # verify against the exact W_j tables
    print("\n  VERIFICATION against the exact full-period W_j census "
          "(data/depth_identity_<y>.csv):")
    for y in ys:
        path = os.path.join(DATA, f"depth_identity_{y}.csv")
        rows = list(csv.DictReader(open(path)))
        gmax = max(int(r["g"]) for r in rows)
        W = {}
        for r in rows:
            g = int(r["g"])
            for k, v in r.items():
                if k.startswith("W"):
                    W[(int(k[1:]), g)] = int(v)
        jmax = max(j for j, g in W)
        bad = []
        for j in range(1, min(J, jmax) + 1):
            oddset = sorted(g for g in range(1, gmax + 1)
                            if W.get((j, g), 0) % 2 == 1)
            pred = tabs[y][j]
            if pred <= gmax:
                if oddset != [pred]:
                    bad.append((j, oddset, pred))
            else:
                if oddset:
                    bad.append((j, oddset, f">{gmax}"))
        ok(not bad, f"m{y}: for every depth j <= {min(J, jmax)} the ONLY g "
                    f"with W_j(g) odd is the predicted g_j*   "
                    f"({bad if bad else 'no exceptions'})")
    return tabs


# ---------------------------------------------------------------- part C
def partC(tabs, maxy):
    print("\n=== PART C: the parity corollaries and the lever ===")
    print("""
  C1  W_j(g) is EVEN for every g with g != j (mod 2).  No computation.
      In particular EVERY EVEN GAP LENGTH occurs an even number of times,
      and every depth-2 window of ODD total length does too.
  C2  for g = j (mod 2), W_j(g) is even unless g = g_j*.
  C3  THE LEVER.  If a counting/covering argument shows that at most ONE
      depth-j window has sum g, then for every g != g_j* there are NONE.
      g_j* is Part B's one-line number, so the side condition is free.
  C4  Since g_1* = 1 (Part B's theorem), W_1(g) is EVEN for EVERY g >= 2:
      only the count of gaps of size 1 is odd.  In particular the number of
      MAXIMAL gaps is even at every machine - unconditionally, no side
      condition, no computation - so the maximal gap never occurs exactly
      once and the top eigenvalue of A (Part F) is never simple.""")
    print("\n     y     F   F par  g_1* (antipodal gap)   F = g_1*?   "
          "W_1(F) forced even?")
    for y in sorted(F_KNOWN):
        F = F_KNOWN[y]
        g1 = tabs[y][1] if y in tabs else gstar_table(y, 1)[0][1]
        forced = (F % 2 == 0) or (F != g1)
        print(f"    {y:3d}  {F:4d}   {'even' if F%2==0 else 'odd '}   "
              f"{g1:>8}                {'YES' if F==g1 else 'no ':>5}"
              f"        {'YES' if forced else 'NO':>3}")
        assert forced, (y, F, g1)
    ok(True, "W_1(F) is forced even at every machine 11..53 "
             "(unconditionally, since g_1* = 1 < 2 <= F)")

    print("""
  C4b THE EXCEPTION IS NEVER A QUALIFYING WINDOW - so on the (D) family the
      lever has NO side condition at all.  The merge law only quantifies over
      windows whose MIDDLE gaps all clear the next gear's tooth floor
      a = 2u'(q').  The exception window sits against slot 0 or the antipode,
      where the gaps are the machine's SHORTEST (k_1 = 3..10), so it fails
      that test at every rung and every depth:""")
    print("\n     step      floor a   depths j = 2..7: is the exception "
          "window qualifying?")
    for M, qp in STEPS:
        a = 2 * tooth(qp)
        gs, us, P = machine(M)
        o = openings_after(0, 6, gs, us)
        b = openings_after((P - 1) // 2, 6, gs, us)
        marks = []
        for j in range(2, 8):
            if j % 2 == 0:
                pts = [P - x for x in reversed(o[:j // 2])] + [P] + \
                      [P + x for x in o[:j // 2]]
            else:
                h = (j + 1) // 2
                pts = [P - x for x in reversed(b[:h])] + list(b[:h])
            w = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
            assert len(w) == j, (M, j, w)
            mids = w[1:-1]
            marks.append("Y" if mids and min(mids) >= a else "n")
        print(f"    {M:3d}->{qp:<3d}    {a:>4}       " + "  ".join(
            f"j{j}:{m}" for j, m in zip(range(2, 8), marks)))
        assert all(m == "n" for m in marks), (M, qp, marks)
    ok(True, "the mirror-exceptional window is NOT qualifying at any rung "
             "11->13 .. 47->53 or any depth j <= 7 - the lever applies to the "
             "whole (D) family with no exception")
    print("""
  C5  WHAT THE LEVER IS WORTH, priced.  A first-moment argument normally has
      to reach "expected count < 1" to conclude zero.  With the lever an
      EXACT count bound of "< 2" suffices, for every g != g_j*.  Part A says
      that is the whole of it: one factor of two, never four.""")


# ---------------------------------------------------------------- part D
def read_tuples(path, arity=4):
    out = set()
    with open(path) as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            out.add(tuple(int(x) for x in row[:arity]))
    return out


def partD():
    print("\n=== PART D: word / tuple reversal, and what it costs not to "
          "know it ===")
    print("""
  THEOREM.  The mirror sends an occurrence of the gap word w at address k to
  an occurrence of REVERSE(w) at address -(k + span w).  It is a bijection,
  so

      #occ(w) = #occ(reverse w)   EXACTLY, at every machine, every arity,

  and realisability is reverse-invariant.  The same holds for merge KILL
  words: the old machine's openings and the new gear's teeth are both
  negation-symmetric, so a kill word is realisable iff its reverse is.
  CONSEQUENCE: every realisability census - dictionary build, SAT
  refutation, CRT decision - need only decide ONE WORD PER REVERSE CLASS.""")
    for name, path in (("m23", "gap_tuples_23_4.csv"),
                       ("m29", "gap_tuples_29_4.csv"),
                       ("m31", "gap_tuples_31_4.csv"),
                       ("m37 (exact)", "gap_tuples_37_4.csv")):
        p = os.path.join(DATA, path)
        if not os.path.exists(p):
            print(f"    (missing {path} - skipped)")
            continue
        S = read_tuples(p)
        R = {t[::-1] for t in S}
        pal = sum(1 for t in S if t == t[::-1])
        ok(S == R, f"{name}: the realised 4-tuple dictionary is EXACTLY "
                   f"reverse-closed ({len(S):,} tuples, {pal} palindromes, "
                   f"{(len(S)-pal)//2:,} reverse pairs)")

    # audit the A_kill logs
    print("\n  AUDIT of this project's own arity censuses (research/data/r24):")
    for step, fn in (("43->47", "akillp_43_47.log"),
                     ("47->53", "akillp_47_53.log")):
        p = os.path.join(DATA, "r24", fn)
        if not os.path.exists(p):
            print(f"    (missing {fn})")
            continue
        recs = []
        for line in open(p):
            if "RESULT" not in line or "word (" not in line:
                continue
            w = line.split("word (")[1].split(")")[0]
            w = tuple(int(x) for x in w.split(","))
            verdict = "ZERO" if "ZERO" in line else "REALISED"
            t = 0.0
            if "calls," in line:
                t = float(line.split("calls,")[1].split("s)")[0])
            recs.append((w, verdict, t))
        by = {}
        for w, v, t in recs:
            by.setdefault(w, (v, t))
        # 1. every reverse pair present must AGREE - a falsifiable gate
        disagree = [(w, by[w][0], by[w[::-1]][0]) for w in by
                    if w[::-1] in by and by[w][0] != by[w[::-1]][0]]
        ok(not disagree, f"{step}: every decided word whose reverse was also "
                         f"decided got the SAME verdict "
                         f"({len(recs)} decisions) {disagree}")
        # 2. the redundant cost
        seen, redundant, total = set(), 0.0, 0.0
        for w, v, t in recs:
            total += t
            key = min(w, w[::-1])
            if key in seen:
                redundant += t
            seen.add(key)
        npal = sum(1 for w in by if w == w[::-1])
        print(f"    {step}: {len(by)} words decided, {npal} palindromic, "
              f"{(len(by)-npal)//2} reverse pairs;")
        print(f"           decision time {total:,.0f} s of which "
              f"{redundant:,.0f} s ({100*redundant/max(total,1e-9):.1f}%) was "
              f"spent on the SECOND member of a reverse pair.")
    print("""
  This is a live saving, not a hypothetical: the four span-141 k=3 words at
  47->53 are TWO reverse pairs, and both members of each were refuted at full
  SAT cost.""")


# ---------------------------------------------------------------- part E
def partE(maxy):
    print("\n=== PART E: the fixed-point criterion (parity by one lookup) ===")
    print("""
  For a PALINDROMIC tuple w of span s the occurrence set is mirror-INVARIANT
  (reverse w = w), so its size has the parity of the number of self-mirror
  occurrences.  An occurrence at k is self-mirror iff -(k + s) = k, i.e.
  2k = -s (mod P): since P is odd there is EXACTLY ONE candidate address

      k_w = -s * inverse(2)  (mod P).

  THEOREM.  #occ(w) is ODD iff w occurs at k_w, and EVEN otherwise - an
  O(#gears) test.  Specialising to w = (g,g): k_w = -g, and the occurrence
  needs openings at -g, 0, g with nothing between, i.e. g = k_1 exactly.
  That rederives round 25's "the unique odd depth-2 palindrome is (k_1,k_1)"
  in one line, and generalises it to every arity.""")
    for y in (11, 13, 17, 19, 23):
        op, P = blocked_array(y)
        gs, us, Pp = machine(y)
        ops = np.flatnonzero(op)
        gaps = np.diff(np.concatenate([ops, [ops[0] + P]]))
        assert gaps.sum() == P
        # census of palindromic 2- and 3-tuples, exactly, over the period
        w2 = {}
        for i in range(len(ops)):
            a, b = int(gaps[i]), int(gaps[(i + 1) % len(ops)])
            if a == b:
                w2[(a, b)] = w2.get((a, b), 0) + 1
        w3 = {}
        for i in range(len(ops)):
            a = int(gaps[i]); b = int(gaps[(i + 1) % len(ops)])
            c = int(gaps[(i + 2) % len(ops)])
            if a == c:
                w3[(a, b, c)] = w3.get((a, b, c), 0) + 1
        inv2 = pow(2, -1, P)
        bad = []
        for w, cnt in list(w2.items()) + list(w3.items()):
            s = sum(w)
            k = (-s * inv2) % P
            # does w occur at k?  need openings at k, k+w1, k+w1+w2, ...
            pts = [k]
            for g in w:
                pts.append(pts[-1] + g)
            hit = all(op[p % P] for p in pts) and all(
                not op[x % P] for a, b in zip(pts, pts[1:])
                for x in range(a + 1, b))
            if hit != (cnt % 2 == 1):
                bad.append((w, cnt, hit))
        ok(not bad, f"m{y}: the fixed-point criterion predicts the parity of "
                    f"EVERY palindromic 2- and 3-tuple count "
                    f"({len(w2)} + {len(w3)} words) {bad[:3]}")
        k1 = int(gaps[np.argmax(ops == 0)]) if 0 in ops else int(gaps[0])
        oddw2 = [w for w, c in w2.items() if c % 2 == 1]
        ok(oddw2 == [(k1, k1)], f"m{y}: the unique odd (g,g) is "
                                f"(k_1,k_1) = ({k1},{k1})")


# ---------------------------------------------------------------- part F
def totient(n):
    r, m = n, n
    p = 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            r -= r // p
        p += 1
    if m > 1:
        r -= r // m
    return r


def support_from_tuples(path):
    S = set()
    with open(path) as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            for x in row:
                S.add(int(x))
    return S


def partF(maxy):
    print("\n=== PART F: the spectral face - backlog U4 ===")
    print("""
  A = BS + (BS)^T is the disjoint union over gaps of path graphs P_g
  (item 36), so spec(A) = union_g {2 cos(pi j/(g+1)) : j = 1..g} with
  multiplicity W_1(g).  Write 2cos(pi j/(g+1)) in lowest terms a/b:
  b | g+1, and for FIXED b every a coprime to b arises from exactly the
  gaps g = -1 (mod b).  Hence

      THEOREM.  mult(2 cos(pi a/b)) = Sigma(b) := sum_{g = -1 mod b} W_1(g),
      INDEPENDENT of a.

  So the eigenvalue multiplicities of A ARE the gap histogram's residue-class
  counts, one class (-1) per modulus, and the histogram is recovered by
  Mobius inversion over multiples:  W_1(b-1) = sum_{t>=1} mu(t) Sigma(t b).
  Two corollaries fall out:
    (i)  PARITY.  By Part C, Sigma(b) is EVEN unless b | g_1* + 1.  Every
         eigenvalue multiplicity of A is even except at the divisors of
         (antipodal gap + 1).
    (ii) THE LEVEL COUNT IS A DIVISOR-CLOSURE STATISTIC, not a Farey count:
         #distinct = sum over b >= 2 that divide some (realised g)+1 of
         phi(b).  The published |Farey(F+1)| - 2 assumes EVERY g in [1,F] is
         realised.  Machines with HOLES break it.""")
    # supports
    sup = {}
    for y in (11, 13, 17, 19, 23, 29):
        path = os.path.join(DATA, f"depth_identity_{y}.csv")
        rows = list(csv.DictReader(open(path)))
        sup[y] = {int(r["g"]) for r in rows if int(r["W1"]) > 0}
    for y, fn in ((23, "gap_tuples_23_4.csv"), (29, "gap_tuples_29_4.csv"),
                  (31, "gap_tuples_31_4.csv"), (37, "gap_tuples_37_4.csv"),
                  (41, "gap_tuples_41_4_transfer.csv")):
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            continue
        s = support_from_tuples(p)
        if y in sup:
            ok(s == sup[y], f"m{y}: the 4-tuple dictionary's depth-1 support "
                            f"equals the full-period histogram support "
                            f"({len(s)} values) - the tuple route is "
                            f"validated where both exist")
        else:
            sup[y] = s
    ok(sup[41] == set(range(1, 92)) - {84, 87, 89},
       "m41 support = {1..91} minus the COV-SAT hole list {84,87,89}")

    print("\n     y     F   holes            distinct levels   naive Farey"
          "   loss")
    for y in sorted(sup):
        F = F_KNOWN[y]
        assert max(sup[y]) == F, (y, max(sup[y]), F)
        holes = sorted(set(range(1, F + 1)) - sup[y])
        B = set()
        for g in sup[y]:
            n = g + 1
            for b in range(2, n + 1):
                if n % b == 0:
                    B.add(b)
        nd = sum(totient(b) for b in sorted(B))
        naive = sum(totient(b) for b in range(2, F + 2))
        # explicit cross-check by direct construction of the level set
        S = set()
        for g in sorted(sup[y]):
            for j in range(1, g + 1):
                d = gcd(j, g + 1)
                S.add((j // d, (g + 1) // d))
        assert len(S) == nd, (y, len(S), nd)
        hs = str(holes) if len(holes) <= 6 else f"{len(holes)} holes"
        print(f"    {y:3d}  {F:4d}   {hs:<16} {nd:>12,}  {naive:>12,}"
              f"   {naive-nd:>6,}")
        sup[y] = (sup[y], holes, nd, naive)
    ok(True, "distinct-level counts recomputed on the TRUE support "
             "(direct construction agrees with the phi-sum at every machine)")

    # the loss rule
    print("""
  THE LOSS RULE.  b is absent iff every multiple of b in [2, F+1] is
  (hole+1).  For b > (F+1)/2 that is just "b-1 is a hole", so the loss is
  dominated by the LARGEST holes:""")
    for y in sorted(sup):
        S, holes, nd, naive = sup[y]
        F = F_KNOWN[y]
        big = [h for h in holes if h + 1 > (F + 1) / 2]
        pred = sum(totient(h + 1) for h in big)
        print(f"    m{y}: holes above (F-1)/2 = {big} -> predicted loss "
              f"{pred:,}, actual {naive-nd:,}"
              f"   {'MATCH' if pred == naive-nd else 'EXTRA SMALL-b LOSS'}")

    # level statistics on the TRUE set (repairing r22 part C as well)
    print("""
  AND THE LEVEL STATISTICS MOVE WITH IT.  Round 22 measured the distinct
  spectrum's spacing statistics on the FULL Farey set too.  Recomputed on the
  true level set (unfolded by pulling 2cos(pi x) back to x, as before):""")
    print("     y    #levels   <r~> true   <r~> naive   s_min/s_mean true"
          "   P(s<0.1 mean)")
    for y in sorted(sup):
        S, holes, nd, naive = sup[y]
        F = F_KNOWN[y]
        xs_true = sorted({(j / (g + 1)) for g in sorted(S)
                          for j in range(1, g + 1)})
        xs_nv = sorted({(j / (g + 1)) for g in range(1, F + 1)
                        for j in range(1, g + 1)})
        def stats(xs):
            s = np.diff(np.array(xs))
            s = s[s > 0]
            r = s[1:] / s[:-1]
            return (float(np.minimum(r, 1 / r).mean()),
                    float(s.min() / s.mean()),
                    float((s < 0.1 * s.mean()).mean()))
        rt, smt, pst = stats(xs_true)
        rn, _, _ = stats(xs_nv)
        print(f"    {y:3d}   {len(xs_true):>7,}    {rt:8.4f}    {rn:8.4f}"
              f"        {smt:10.4f}      {pst:10.4f}")
        assert pst == 0.0 and rt > 0.6027, (y, pst, rt)
    ok(True, "on the TRUE level set the hard gap survives (P(s<0.1 mean) = 0 "
             "exactly, forced: a subset of a set with a hard gap keeps it) "
             "and <r~> stays above GUE - the r22 conclusion is unchanged, "
             "only its numbers move")

    # parity of the multiplicities
    print("\n  PARITY OF THE MULTIPLICITIES (corollary i), checked exactly:")
    for y in (11, 13, 17, 19, 23, 29):
        rows = list(csv.DictReader(open(os.path.join(
            DATA, f"depth_identity_{y}.csv"))))
        W1 = {int(r["g"]): int(r["W1"]) for r in rows}
        g1 = gstar_table(y, 1)[0][1]
        F = F_KNOWN[y]
        bad = []
        for b in range(2, F + 2):
            sig = sum(W1.get(g, 0) for g in range(b - 1, F + 1, b))
            if (sig % 2 == 1) != ((g1 + 1) % b == 0):
                bad.append((b, sig))
        ok(not bad, f"m{y}: Sigma(b) is odd EXACTLY when b | g_1*+1 = "
                    f"{g1+1} (all b <= F+1) {bad[:4]}")


# ---------------------------------------------------------------- part G
def partG():
    print("\n=== PART G: the pole phase is unattainable AT EVERY GEAR ===")
    print("""
  Round 25 (item 48) proved the gear-5 pole phase 126 deg unattainable via
  the cell decomposition, and its GF(2) test over cell mirror orbits reported
  that GEAR 5 IS THE ONLY PARITY-OBSTRUCTED GEAR for p <= 37.  Part B's
  theorem makes that scope claim collapse.

  Set N_r^(p) = #{gaps = r (mod p)}.  Since W_1(1) is the ONLY odd entry of
  the gap histogram (Part B: g_1* = 1), and 1 = 1 (mod p) for every p,

      N_1^(p) is ODD and N_r^(p) is EVEN for every other r,
      at EVERY machine and EVERY modulus p.

  The bracket B_p = sum_s beta_s omega^s (beta_r = N_{r+1} - N_r, omega =
  e(1/p)) is real iff alpha_s := beta_s - beta_{-s} vanishes for
  s = 1..(p-1)/2 - the elements omega^s - omega^{-s} being Q-linearly
  independent (they use disjoint pairs of the power basis).  But

      alpha_1 = beta_1 - beta_{p-1} = N_2 - N_1 - N_0 + N_{p-1}
              = even - ODD - even + even  =  ODD  != 0.

  THEOREM.  For EVERY gear p >= 5 and every machine, alpha_1(p) is odd, so
  B_p is never exactly real and THE POLE PHASE IS NEVER ATTAINED - at gear 7,
  gear 11, gear 37, everywhere, not only at gear 5.""")
    for y in (11, 13, 17, 19, 23, 29):
        rows = list(csv.DictReader(open(os.path.join(
            DATA, f"depth_identity_{y}.csv"))))
        W1 = {int(r["g"]): int(r["W1"]) for r in rows}
        F = F_KNOWN[y]
        assert all(W1.get(g, 0) == 0 for g in range(F + 1, 65)), y
        tot = sum(W1.values())
        gs, _, _ = machine(y)
        line = []
        for p in gs + [41, 43]:
            N = [sum(W1.get(g, 0) for g in range(1, F + 1) if g % p == r)
                 for r in range(p)]
            assert sum(N) == tot
            par = [n % 2 for n in N]
            assert par == [1 if r == 1 % p else 0 for r in range(p)], (y, p, par)
            beta = [N[(r + 1) % p] - N[r] for r in range(p)]
            alpha = [beta[s] - beta[(-s) % p] for s in range(1, (p - 1) // 2 + 1)]
            assert alpha[0] % 2 == 1, (y, p, alpha)
            line.append(f"p{p}:a1={alpha[0]}")
        ok(True, f"m{y}: N_1^(p) odd and all other N_r^(p) even, and "
                 f"alpha_1(p) ODD, at every p in {gs + [41, 43]}")
        print("        " + "  ".join(line))
    print("""
  SELF-CORRECTION TO THIS LANE'S ROUND-25 RECORD.  Round-25 prediction P3
  ("gear 5 is the ONLY parity-obstructed gear among p <= 37") was scored
  CONFIRMED and called the structural half of backlog U3.  IT IS WRONG AS
  WORDED.  What its GF(2) test actually decided is whether the CELL-MATRIX
  constraints alone (row sums odd, plus the pole equations) force a parity
  contradiction - a strictly narrower question, since those constraints know
  nothing about W_1(1).  The correct statement:

      * every gear is parity-obstructed (this Part), so the pole phase is
        exactly unattainable everywhere;
      * the cell-orbit obstruction is special to gear 5 (round 25, still
        true about that test);
      * and the REAL distinction between gears 5 and 7 is round 25's
        MEASURED half - three equations instead of one, and asymmetries an
        order of magnitude larger and slower to decay.

  Round 25's item-48 conclusion survives; its uniqueness claim does not.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCDEFG")
    ap.add_argument("--maxy", type=int, default=29)
    a = ap.parse_args()
    tabs = None
    if "A" in a.parts:
        partA()
    if "B" in a.parts or "C" in a.parts:
        tabs = partB(a.maxy)
    if "C" in a.parts:
        partC(tabs, a.maxy)
    if "D" in a.parts:
        partD()
    if "E" in a.parts:
        partE(a.maxy)
    if "F" in a.parts:
        partF(a.maxy)
    if "G" in a.parts:
        partG()
    print(f"\n=== {NASSERT} ASSERTION GATES PASSED ===")


if __name__ == "__main__":
    main()
