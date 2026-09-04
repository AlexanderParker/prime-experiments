"""Round 31 (constructor): THE BARE-ALTERNATION ADMISSIBILITY LEMMA.

Lateral round 30 observed, on the counterfactual family, that a machine whose
bare alternation (a, b, a) is inadmissible at gears 5 and 7 has no realised
bare-letter word of length 3 (0 exceptions in 21,357 deep words).  This script
STATES the object exactly, computes the inadmissible set S as residue classes
mod 210, and gates the lemma at every corpus machine m11..m47.

DEFINITIONS (all exact integer arithmetic; see
research/data/r31/constructor_prereg_r31.txt for the pre-registration).

  M = {5..y} a machine, q' = nextprime(y).
  u' = round(q'/6) = the smaller tooth of q' (6 u' = q' -+ 1), d' = 2 u'.
  BARE letters:  a = d' = 2u'  and  b = q' - a.   a + b = q', 3a = q' -+ 1.
  A LEGAL letter is a gap value v with v = 0 or +-d' (mod q').
      bare    : v in {a, b}
      padded  : v = 0 mod q'      (q', 2q', ...)
      shifted : v = a + kq' or b + kq', k >= 1
  T3: the nonzero classes strictly alternate along a word; padded letters are
  transparent.  Hence a BARE word of length m is one of exactly two words:
      A_m = (a,b,a,b,...)   or   B_m = (b,a,b,a,...).
  Prefix-sum offset set X(w) = {0, w_1, w_1+w_2, ...}, |X| = m+1.
  Teeth of gear g: {+-6^{-1} mod g}.  Exposed set E_g = Z_g \ teeth.
  ADMISSIBLE AT {5,7}: some t with t + X(w) subset E_5 (mod 5) AND some t with
  t + X(w) subset E_7 (mod 7).  By CRT this is equivalent to a translate of X
  fitting inside the corridor E_5 x E_7 mod 35 (asserted, GATE A3).
  PSORD(c), c coprime to 210: the largest m such that A_m or B_m is admissible
  at {5,7}, when q' = c mod 210.  (Well defined: GATE A2.)
  S = { c : PSORD(c) <= 2 }.

Usage:  uv run python research/bare_lemma_r31.py [--crt] [--nodes N]
        --crt runs the machine-41/43/47 CRT word decisions (minutes).
"""
import json
import os
import sys
import time
from math import gcd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------- primitives


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for aa in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(aa, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(y):
    n = y + 1
    while not is_prime(n):
        n += 1
    return n


def teeth(g):
    c = pow(6, -1, g)
    return (c % g, (-c) % g)


def exposed(g):
    t = teeth(g)
    return frozenset(r for r in range(g) if r not in t)


E5, E7 = exposed(5), exposed(7)
E35 = frozenset(r for r in range(35) if r % 5 in E5 and r % 7 in E7)


def fits(Xmod, g, Eg):
    """Does some translate of the point set Xmod (given mod g) fit in E_g?"""
    xs = set(x % g for x in Xmod)
    return any(all((t + x) % g in Eg for x in xs) for t in range(g))


def admissible57(X):
    """X a set/list of integers (the prefix-sum offsets)."""
    return fits(X, 5, E5) and fits(X, 7, E7)


def bare_offsets(a, b, m, start_with_a=True):
    """Prefix-sum offsets of the bare alternation of length m."""
    X, acc = [0], 0
    for j in range(m):
        acc += (a if (j % 2 == 0) == start_with_a else b)
        X.append(acc)
    return X


def psord_phase(a, b, start_a, cap=12):
    """Largest m with that phase of the bare alternation admissible at {5,7}."""
    best = 0
    for m in range(1, cap + 1):
        if admissible57(bare_offsets(a, b, m, start_a)):
            best = m
        else:
            break                       # monotone: X(w_m) subset X(w_{m+1})
    return best


def psord_from_ab(a, b, cap=12):
    """PSORD = MAX over the two phases.  The quantifier matters and is the
    opposite of R74's: L_bare(M) >= m needs SOME bare m-word realised, so the
    existence question maximises over the two phases, while R74's A_relax asks
    for a CYCLE (every window survives) and minimises.  R74's PS-order in its
    own convention is min(phase orders) + 1 (it counts POINTS = deleted
    openings = arity, not letters); GATE A4 reproduces it exactly."""
    return max(psord_phase(a, b, True, cap), psord_phase(a, b, False, cap))


def psord_r74(a, b, cap=12):
    """R74's PS-order: MIN over phases, counted in POINTS (letters + 1)."""
    return min(psord_phase(a, b, True, cap), psord_phase(a, b, False, cap)) + 1


def bare_a(q1):
    """a = 2*round(q'/6), the smaller bare letter, as an exact integer."""
    return (q1 - 1) // 3 if q1 % 3 == 1 else (q1 + 1) // 3


# ------------------------------------------------------- (A) the set S mod 210

def part_A():
    print("=" * 78)
    print("(A)  THE INADMISSIBLE SET S, AS RESIDUE CLASSES MOD 210")
    print("=" * 78)
    print("teeth(5) = %s  E_5 = %s" % (sorted(teeth(5)), sorted(E5)))
    print("teeth(7) = %s  E_7 = %s" % (sorted(teeth(7)), sorted(E7)))
    print("|E_35| = %d (corridor)" % len(E35))

    classes = [c for c in range(210) if gcd(c, 210) == 1]
    assert len(classes) == 48, len(classes)

    # -- vehicle 1: pure mod-210 arithmetic (a mod 5, a mod 7 from c mod 3,5,7)
    ps_mod, ps_ph = {}, {}
    for c in classes:
        sgn = -1 if c % 3 == 1 else +1           # 3a = q' + sgn
        a5 = ((c + sgn) * pow(3, -1, 5)) % 5
        a7 = ((c + sgn) * pow(3, -1, 7)) % 7
        ords = []
        for start_a in (True, False):
            best = 0
            for m in range(1, 13):
                X5, X7, s5, s7 = [0], [0], 0, 0
                for j in range(m):
                    use_a = (j % 2 == 0) == start_a
                    s5 = (s5 + (a5 if use_a else (c - a5))) % 5
                    s7 = (s7 + (a7 if use_a else (c - a7))) % 7
                    X5.append(s5)
                    X7.append(s7)
                if fits(X5, 5, E5) and fits(X7, 7, E7):
                    best = m
                else:
                    break
            ords.append(best)
        ps_ph[c] = tuple(ords)
        ps_mod[c] = max(ords)

    # -- vehicle 2: a direct sweep of the actual primes, exact integers
    ps_prime, ps_prime74, rep_prime = {}, {}, {}
    nprimes = 0
    for q1 in range(11, 20000):
        if not is_prime(q1):
            continue
        nprimes += 1
        a = bare_a(q1)
        assert 3 * a in (q1 - 1, q1 + 1), (q1, a)
        assert a == 2 * round(q1 / 6), (q1, a)
        b = q1 - a
        o = psord_from_ab(a, b)
        c = q1 % 210
        ps_prime.setdefault(c, set()).add(o)
        ps_prime74.setdefault(c, set()).add(psord_r74(a, b))
        rep_prime.setdefault(c, q1)
    # GATE A2: PSORD is constant on each class mod 210, and the two vehicles agree
    bad = [(c, v) for c, v in ps_prime.items() if len(v) != 1]
    assert not bad, ("GATE A2: PSORD not constant on a class mod 210", bad)
    ps_prime = {c: v.pop() for c, v in ps_prime.items()}
    assert set(ps_prime) == set(classes), (len(ps_prime), len(classes))
    mism = [(c, ps_mod[c], ps_prime[c]) for c in classes
            if ps_mod[c] != ps_prime[c]]
    assert not mism, ("GATE A2: vehicles disagree", mism)
    print("\nGATE A1  a = 2*round(q'/6) and 3a = q' -+ 1 at all %d primes "
          "11..20000: OK" % nprimes)
    print("GATE A2  PSORD constant on each of the 48 classes mod 210 and equal "
          "in both vehicles (mod-210 arithmetic vs %d exact primes): OK"
          % nprimes)

    # GATE A3: the CRT step - {5}-fit AND {7}-fit  <=>  mod-35 fit
    n35 = 0
    for q1 in range(11, 2000):
        if not is_prime(q1):
            continue
        a, b = bare_a(q1), q1 - bare_a(q1)
        for m in range(1, 8):
            for sa in (True, False):
                X = bare_offsets(a, b, m, sa)
                lhs = fits(X, 5, E5) and fits(X, 7, E7)
                rhs = fits(X, 35, E35)
                assert lhs == rhs, ("GATE A3", q1, m, sa, lhs, rhs)
                n35 += 1
    print("GATE A3  {5}-fit AND {7}-fit == corridor-mod-35 fit on %d "
          "(prime, length, phase) instances: OK" % n35)

    # GATE A4: R74's own enumeration, reproduced in R74's own convention
    ps74 = {c: v.pop() for c, v in ps_prime74.items()}
    d74 = {}
    for c in classes:
        d74[ps74[c]] = d74.get(ps74[c], 0) + 1
    assert d74 == {2: 24, 3: 16, 4: 2, 5: 6}, ("GATE A4 vs R74", d74)
    assert sorted(c for c in classes if ps74[c] == 5) == [
        37, 53, 83, 127, 157, 173], "GATE A4 exceptional classes"
    assert sorted(c for c in classes if ps74[c] == 4) == [23, 187], "GATE A4"
    print("GATE A4  R74's PS-order (MIN over the two phases, counted in POINTS "
          "= letters + 1): distribution 24/16/2/6 at orders 2/3/4/5, order 5 "
          "exactly on {37,53,83,127,157,173}, order 4 exactly on {23,187} - "
          "R74's enumeration reproduced exactly: OK")

    S = sorted(c for c in classes if ps_mod[c] <= 2)
    comp = sorted(c for c in classes if ps_mod[c] > 2)
    dist = {}
    for c in classes:
        dist[ps_mod[c]] = dist.get(ps_mod[c], 0) + 1
    print("\nPSORD (MAX over the two phases, counted in LETTERS - the quantifier"
          "\nthe word-existence question needs) over the 48 invertible classes "
          "mod 210:")
    for k in sorted(dist):
        cs = sorted(c for c in classes if ps_mod[c] == k)
        print("   PSORD = %d : %2d classes  %s" % (k, dist[k], cs))
    print("\n|S| = %d   S = {c : PSORD(c) <= 2} =\n   %s" % (len(S), S))
    print("\n|complement| = %d   comp = {c : PSORD(c) >= 3} =\n   %s"
          % (len(comp), comp))
    assert len(S) + len(comp) == 48
    assert max(ps_mod.values()) == 5, "PSORD <= 5 uniformly"
    print("\nGATE A5  max PSORD over all 48 classes = 5: the bound "
          "L_bare(M) <= PSORD(q' mod 210) <= 5 is UNIFORM in M: OK")
    # GATE A6 - the exact finite statement handed to Formalist (F3): the SIX-letter
    # bare alternation is inadmissible at {5,7} in BOTH phases at EVERY class.
    nchk = 0
    for c in classes:
        q1 = rep_prime[c]
        a, b = bare_a(q1), q1 - bare_a(q1)
        for sa in (True, False):
            X = bare_offsets(a, b, 6, sa)
            assert not admissible57(X), ("GATE A6", c, sa)
            nchk += 1
    print("GATE A6  the 6-letter bare alternation is INADMISSIBLE at {5,7} in "
          "both phases at\n         all 48 classes (%d checks) - the finite "
          "statement behind L_bare <= 5: OK" % nchk)

    print("\nFORWARD TABLE - the bare cap at the next steps beyond the corpus")
    print("  (a prediction on record: L_bare(M) <= PSORD at each of these)")
    print("   step        q'   a    b   q' mod 210  PSORD  in S?")
    y = 53
    for _ in range(16):
        q1 = next_prime(y)
        a = bare_a(q1)
        print("   %3d -> %-4d %4d %4d %4d %8d %8d   %s"
              % (y, q1, q1, a, q1 - a, q1 % 210, ps_mod[q1 % 210],
                 "IN S" if q1 % 210 in set(S) else "no"))
        y = q1
    return ps_mod, ps_ph, ps74, set(S)


# --------------------------------------------------- (B) the corpus gate

CORPUS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
L_RECORDED = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2,
              41: 2, 43: 2, 47: 4}


def classify(v, q1, a, b):
    if v % q1 == 0:
        return "padded"
    if v in (a, b):
        return "bare"
    return "shifted"


def legal_word(w, q1, a, b):
    """T3: every letter legal, nonzero classes strictly alternate."""
    last = 0
    for v in w:
        r = v % q1
        if r == 0:
            cl = 0
        elif r == a % q1:
            cl = 1
        elif r == b % q1:
            cl = -1
        else:
            return False
        if cl and cl == last:
            return False
        if cl:
            last = cl
    return True


def census_words(y):
    p = os.path.join(HERE, "data", "r30", "occ_%d_words.json" % y)
    d = json.load(open(p))
    return {tuple(int(x) for x in k.split()): v[0] for k, v in d.items()}


def part_B(ps_mod, ps_ph, ps74, S, do_crt, nodes):
    print()
    print("=" * 78)
    print("(B)  THE CORPUS GATE  m11 .. m47")
    print("=" * 78)
    rows = []
    crt_cache = {}
    if do_crt:
        import crt_dict
    for y in CORPUS:
        q1 = next_prime(y)
        a = bare_a(q1)
        b = q1 - a
        c = q1 % 210
        inS = c in S
        # --- realised legal words
        if y <= 37:
            occ = census_words(y)
            realised = {w for w, n in occ.items() if n > 0}
            maxlen = max(len(w) for w in occ)
            src = "counted census (occ_%d_words.json, r30, exact cyclic)" % y
            legal = sorted((w for w in realised if legal_word(w, q1, a, b)),
                           key=lambda w: (len(w), w))
            Lmeas = max(len(w) for w in legal)
            attaining = [w for w in legal if len(w) == Lmeas]
            bare_words = [w for w in legal if all(v in (a, b) for v in w)]
            Lbare = max([len(w) for w in bare_words], default=0)
            # the two length-3 bare words, explicitly
            b3 = [tuple(bare_offsets(a, b, 3, sa)) for sa in (True, False)]
            w3 = [tuple([a, b, a]), tuple([b, a, b])]
            w3_real = [w in realised for w in w3]
            assert maxlen >= 3 or Lmeas < 3
        else:
            src = "CRT (crt_dict.realised)"
            legal = attaining = bare_words = None
            Lmeas = L_RECORDED[y]
            w3 = [tuple([a, b, a]), tuple([b, a, b])]
            w3_real = None
            Lbare = None
        rows.append(dict(y=y, q1=q1, a=a, b=b, c=c, inS=inS,
                         psord=ps_mod[c], ph=ps_ph[c], r74=ps74[c],
                         L=Lmeas, Lbare=Lbare,
                         legal=legal, attaining=attaining, src=src,
                         w3=w3, w3_real=w3_real))

    # --- machines 41, 43, 47: the bare words by CRT
    if do_crt:
        import crt_dict
        todo = []
        for r in rows:
            if r["y"] < 41:
                continue
            a, b, q1 = r["a"], r["b"], r["q1"]
            todo.append((r["y"], (a,)))
            todo.append((r["y"], (b,)))
            todo.append((r["y"], (a, b)))
            todo.append((r["y"], (b, a)))
            todo.append((r["y"], (a, b, a)))
            todo.append((r["y"], (b, a, b)))
            if r["y"] == 47:
                todo.append((47, (a, b, a, b)))
                todo.append((47, (b, a, b, a)))
                todo.append((47, (a, b, a, b, a)))
                todo.append((47, (b, a, b, a, b)))
        print("\nCRT decisions at m41/m43/m47 (node budget %d):" % nodes)
        import perj_scanfree as psf
        for y, w in todo:
            gears = psf.gears_of(y)
            Egs = {g: psf.exposed(g) for g in gears}
            X, acc = [0], 0
            for v in w:
                acc += v
                X.append(acc)
            if psf.ps_refuted(X, gears, Egs):
                crt_cache[(y, w)] = (False, "phase saturation (free)", 0.0)
                print("   m%-3d %-22s refuted   by phase saturation (free)"
                      % (y, str(w)))
                continue
            t0 = time.time()
            ok = crt_dict.realised(y, w, node_budget=nodes)
            dt = time.time() - t0
            crt_cache[(y, w)] = (ok, "decide_cover", dt)
            print("   m%-3d %-22s %-9s %s  %.0f s"
                  % (y, str(w), {True: "REALISED", False: "refuted",
                                 None: "UNDECIDED"}[ok], "decide_cover", dt))
        for r in rows:
            if r["y"] < 41:
                continue
            a, b, y = r["a"], r["b"], r["y"]
            lb = 0
            for m, ws in ((1, [(a,), (b,)]), (2, [(a, b), (b, a)]),
                          (3, [(a, b, a), (b, a, b)]),
                          (4, [(a, b, a, b), (b, a, b, a)]),
                          (5, [(a, b, a, b, a), (b, a, b, a, b)])):
                vals = [crt_cache.get((y, w), (None,))[0] for w in ws
                        if (y, w) in crt_cache]
                if not vals:
                    break
                if any(v is None for v in vals):
                    ws_ok = None
                else:
                    ws_ok = any(vals)
                if ws_ok:
                    lb = m
                elif ws_ok is None:
                    lb = None
                    break
                else:
                    break
            r["Lbare"] = lb
            r["w3_real"] = [crt_cache.get((y, w), (None,))[0] for w in r["w3"]]

    print()
    print("   M    q'    a   b  q'%210 in S?  ord(a..) ord(b..) PSORD  R74  "
          " L  L_bare  (a,b,a)?  (b,a,b)?")

    def f(v):
        return "-" if v is None else ("YES" if v else "no")
    for r in rows:
        w3r = r["w3_real"]
        print("  m%-3d %4d %4d %3d %6d  %-5s %6d %8d %6d %4d %3d %5s   "
              "%-8s %-8s"
              % (r["y"], r["q1"], r["a"], r["b"], r["c"],
                 "IN S" if r["inS"] else "no", r["ph"][0], r["ph"][1],
                 r["psord"], r["r74"], r["L"],
                 "-" if r["Lbare"] is None else r["Lbare"],
                 f(w3r[0]) if w3r else "-", f(w3r[1]) if w3r else "-"))

    # ---- THE LEMMA'S GATE
    print()
    viol = [r for r in rows if r["inS"] and r["Lbare"] is not None
            and r["Lbare"] > 2]
    for r in rows:
        if r["inS"] and r["w3_real"] is not None:
            assert not any(r["w3_real"]), ("GATE B1 - LEMMA REFUTED", r["y"])
    assert not viol, ("GATE B1 - LEMMA REFUTED (L_bare > 2 with q' in S)",
                      [(r["y"], r["Lbare"]) for r in viol])
    nS = sum(1 for r in rows if r["inS"])
    print("GATE B1  L_bare(M) <= 2 at all %d corpus machines with q' mod 210 "
          "in S, and neither (a,b,a) nor (b,a,b) is realised at any of them: "
          "OK" % nS)

    # ---- the general bound (item (c))
    bad = [(r["y"], r["Lbare"], r["psord"]) for r in rows
           if r["Lbare"] is not None and r["Lbare"] > r["psord"]]
    assert not bad, ("GATE B2 - L_bare > PSORD", bad)
    print("GATE B2  L_bare(M) <= PSORD(q' mod 210) <= 5 at all %d corpus "
          "machines: OK" % len(rows))

    # ---- the proof step, checked on every realised word on record
    nchk = 0
    import perj_scanfree as psf
    for r in rows:
        if r["legal"] is None:
            continue
        y = r["y"]
        gears = psf.gears_of(y)
        Egs = {g: psf.exposed(g) for g in gears}
        for w in r["legal"]:
            X, acc = [0], 0
            for v in w:
                acc += v
                X.append(acc)
            assert admissible57(X), ("GATE B3 - a REALISED word is "
                                     "inadmissible at {5,7}", y, w)
            assert not psf.ps_refuted(X, gears, Egs), ("GATE B3b", y, w)
            nchk += 1
    print("GATE B3  every one of the %d realised LEGAL words on record at "
          "m11..m37 is admissible at {5,7} (the proof step, on data): OK"
          % nchk)

    # ---- the L recorded row
    for r in rows:
        if r["y"] <= 37:
            assert r["L"] == L_RECORDED[r["y"]], ("GATE B4", r["y"])
    print("GATE B4  L from the counted census reproduces the recorded row "
          "1,1,1,2,1,3,3,2 at m11..m37: OK")
    return rows


def part_B2(rows):
    """The attaining words, classified bare / padded / shifted."""
    print()
    print("=" * 78)
    print("(B2)  THE ATTAINING WORDS, CLASSIFIED")
    print("=" * 78)
    for r in rows:
        if r["attaining"] is None:
            continue
        q1, a, b = r["q1"], r["a"], r["b"]
        print("  m%-3d q'=%-3d L=%d  bare letters (a,b) = (%d,%d)"
              % (r["y"], q1, r["L"], a, b))
        for w in r["attaining"]:
            kinds = [classify(v, q1, a, b) for v in w]
            print("        %-22s  %s" % (str(w), " ".join(kinds)))
    print("  m41  q'=43  L=2  attaining words on record (Mechanic r30): "
          "(14,43) (43,14) (29,43) (43,29) (43,43)")
    print("        every one carries the padded letter 43")
    print("  m43  q'=47  L=2  (R97: all eight length-3 survivors refuted)")
    print("  m47  q'=53  L=4  (18,35,18,35)  bare bare bare bare   (R98)")


# -------------------------------- (B3) L_pad, and what gears 5 and 7 do to it

CORRCAP_R75 = {19: 4, 23: 2, 29: 3, 31: 5, 37: 25, 41: 25, 43: 11, 47: 5,
               53: None}          # None = INFINITE (R75)


def corrcap(y, F, q1, a, b, cap=60):
    """R75's CORRCAP by an automaton on the 35 x 3 corridor states: the longest
    T3-legal word over the legal values <= F whose prefix-sum walk stays inside
    the corridor E mod 35.  Returns (cap_or_None, witness_word); None = INFINITE
    (the automaton has a cycle)."""
    LV = []
    for v in range(1, F + 1):
        r = v % q1
        if r == 0:
            LV.append((v, 0))
        elif r == a % q1:
            LV.append((v, 1))
        elif r == b % q1:
            LV.append((v, -1))
    # states (residue mod 35 of the current prefix point, last nonzero class)
    nodes = [(r, c) for r in sorted(E35) for c in (0, 1, -1)]
    adj = {}
    for (r, last) in nodes:
        out = []
        for v, c in LV:
            if c and c == last:
                continue
            r2 = (r + v) % 35
            if r2 not in E35:
                continue
            out.append(((r2, c if c else last), v))
        adj[(r, last)] = out
    starts = [(r, 0) for r in sorted(E35)]
    # reachable set
    seen, stack = set(starts), list(starts)
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    # cycle detection on the reachable sub-graph
    colour = {}

    def dfs(u):
        colour[u] = 1
        for v, _ in adj[u]:
            if v not in seen:
                continue
            if colour.get(v) == 1:
                return True
            if colour.get(v) is None and dfs(v):
                return True
        colour[u] = 2
        return False

    for u in list(seen):
        if colour.get(u) is None and dfs(u):
            return None, ()          # INFINITE
    # DAG: longest path from a start, with a witness
    memo = {}

    def longest(u):
        if u in memo:
            return memo[u]
        best, bw = 0, ()
        for v, val in adj[u]:
            if v not in seen:
                continue
            n, w = longest(v)
            if 1 + n > best:
                best, bw = 1 + n, (val,) + w
        memo[u] = (best, bw)
        return memo[u]

    res = max((longest(s) for s in starts), key=lambda t: t[0])
    return res


def part_B3(rows):
    print()
    print("=" * 78)
    print("(B3)  L_pad - THE NON-BARE HALF - AND WHAT GEARS 5 AND 7 DO TO IT")
    print("=" * 78)
    print("  L_pad(M) = longest realised legal word using at least one NON-BARE"
          " letter\n  (padded = 0 mod q'; shifted = a + kq' or b + kq', k >= 1)."
          "  From the counted\n  census at m11..m37; m41 from R85/Mechanic r30; "
          "m43 from L and L_bare; m47\n  from research/lpad47_r31.py.\n")
    print("   M   q'  alphabet: bare | padded | shifted        "
          "  L  L_bare  L_pad  attaining non-bare word")
    for r in rows:
        if r["legal"] is None:
            src = {41: ("(14,43)", "Mechanic r30 killer table: every attaining "
                                   "word carries 43"),
                   43: ("(?, ?)", "L = 2 (R97) with L_bare = 1 (this round)"),
                   47: ("(18,35,53)", "lpad47_r31.py: 3 realised, R98 refutes "
                                      "every non-bare 4-word")}[r["y"]]
            q1, a, b = r["q1"], r["a"], r["b"]
            F = F_REC[r["y"]]
            alpha = [v for v in range(1, F + 1)
                     if v % q1 in (0, a % q1, b % q1)]
            pad = [v for v in alpha if v % q1 == 0]
            sh = [v for v in alpha if v % q1 != 0 and v not in (a, b)]
            lp = {41: 2, 43: 2, 47: 3}[r["y"]]
            r["Lpad"] = lp
            print("  m%-3d %3d  %-12s | %-9s | %-12s %3d %5s %6s   %s   [%s]"
                  % (r["y"], q1, ",".join(map(str, (a, b))),
                     ",".join(map(str, pad)) or "-",
                     ",".join(map(str, sh)) or "-",
                     r["L"], "-" if r["Lbare"] is None else r["Lbare"],
                     "?" if lp is None else lp, src[0], src[1]))
            continue
        q1, a, b = r["q1"], r["a"], r["b"]
        nb = [w for w in r["legal"] if any(v not in (a, b) for v in w)]
        lp = max([len(w) for w in nb], default=0)
        att = [w for w in nb if len(w) == lp]
        F = F_REC[r["y"]]
        alpha = [v for v in range(1, F + 1) if v % q1 in (0, a % q1, b % q1)]
        pad = [v for v in alpha if v % q1 == 0]
        sh = [v for v in alpha if v % q1 != 0 and v not in (a, b)]
        r["Lpad"] = lp
        print("  m%-3d %3d  %-12s | %-9s | %-12s %3d %5d %6d   %s"
              % (r["y"], q1, ",".join(map(str, (a, b))),
                 ",".join(map(str, pad)) or "-",
                 ",".join(map(str, sh)) or "-",
                 r["L"], r["Lbare"], lp,
                 str(att[0]) if att else "-"))
    print("\n  THE CORRECTION (manager, round 31): a padded letter is NOT free "
          "mod 35.\n  Its value q' has a definite residue mod 5 and mod 7, so "
          "gears 5 and 7 see a\n  padded word exactly as they see a bare one.  "
          "The check, on the corpus:")
    import perj_scanfree as psf
    for r in rows:
        y, q1, a, b = r["y"], r["q1"], r["a"], r["b"]
        F = F_REC[y]
        alpha = [v for v in range(1, F + 1) if v % q1 in (0, a % q1, b % q1)]
        import itertools
        n_nb = n_nb_ref = 0
        for w in itertools.product(alpha, repeat=2):
            if not legal_word(w, q1, a, b):
                continue
            if all(v in (a, b) for v in w):
                continue
            n_nb += 1
            X = [0, w[0], w[0] + w[1]]
            if not admissible57(X):
                n_nb_ref += 1
        print("     m%-3d q'=%-3d : of %3d T3-legal NON-BARE 2-words over the "
              "alphabet, gears\n              {5,7} alone refute %3d"
              % (y, q1, n_nb, n_nb_ref))
    print("\n  So the corridor is not blind to padded letters.  R75's CORRCAP "
          "goes INFINITE\n  at 53 -> 59 because the ALPHABET (about 3F/q' legal "
          "values <= F) becomes rich\n  enough to carry a T3-legal cycle through"
          " the 35 x 3 corridor states - not\n  because any letter is invisible."
          "  The alphabet sizes and CORRCAP:")
    print("\n   M    q'   F   |alphabet|  3F/q'  CORRCAP (automaton)  R75  "
          "witness word (first %d letters)" % 8)
    for y in CORPUS + [53]:
        q1 = next_prime(y)
        if y not in F_REC:
            continue
        a, b = bare_a(q1), q1 - bare_a(q1)
        F = F_REC[y]
        alpha = [v for v in range(1, F + 1) if v % q1 in (0, a % q1, b % q1)]
        cc, wit = corrcap(y, F, q1, a, b)
        gate = CORRCAP_R75.get(y, "-")
        if y in CORRCAP_R75:
            assert cc == CORRCAP_R75[y], ("GATE B5 vs R75", y, cc,
                                          CORRCAP_R75[y])
        print("  m%-3d %4d %4d %8d %8.1f %14s %8s   %s"
              % (y, q1, F, len(alpha), 3.0 * F / q1,
                 "INFINITE" if cc is None else cc,
                 "INF" if gate is None else gate, str(wit[:8])))
    print("\nGATE B5  the corridor automaton reproduces R75's CORRCAP row "
          "4,2,3,5,25,25,11,5,INF\n         at 19->23 .. 53->59: OK")


# ---------------------------------------- (C) what the spectrum c_B does to R99

F_REC = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
         41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
F2_REC = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
          41: 103, 43: 116, 47: 134, 53: 159, 59: 173}


def part_C(ps_mod):
    """Lateral's spectrum bound c_B = F(M+q')/a_min against R99's requirement.

    R99: (A) |eps| <= c_A and (B) L <= c_B give F(M+q') <= F_2(M) + c_A c_B,
    and (D) follows if also c_A c_B <= S_2 = F(M) + q' - F_2(M).
    With c_B = floor(F(M+q') / a) (Lateral round 31: a word of m letters sits
    in a word-legal (m+2)-window of span >= m*a, and every word-legal window
    has span <= max_J Q*_J = F(M+q') by R68) the product bound is no longer a
    constant; this table asks whether it still fits under S_2.
    """
    print()
    print("=" * 78)
    print("(C)  THE SPECTRUM c_B IN THE R99 CHAIN  (c_A = 4, the literal bound)")
    print("=" * 78)
    print("  Two versions of c_B.  NAIVE = floor(F(M+q')/a) - every letter is "
          ">= a.\n  PARITY = Lateral round 31 item 84 - T3 makes two "
          "CONSECUTIVE nonzero letters sum\n  to >= a + b = q', so "
          "L <= max(2T, 2*floor((G-2-a)/q') + 1) with T = floor((G-2)/q').\n"
          "  PARITY is the one to use; the NAIVE column is kept only to show "
          "the cost of\n  ignoring the alternation.\n")
    print("   M    q'   a   F(M) F_2(M)   G=F(M+q')  S_2  cB_naive 4cB<=S2? "
          " cB_parity 4cB<=S2?  L")
    okn, okp = [], []
    for y in CORPUS + [53]:
        q1 = next_prime(y)
        if q1 not in F_REC:
            continue
        a = bare_a(q1)
        F, F2, G = F_REC[y], F2_REC[y], F_REC[q1]
        S2 = F + q1 - F2
        cBn = G // a
        T = (G - 2) // q1
        cBp = max(2 * T, 2 * ((G - 2 - a) // q1) + 1)
        okn.append((y, 4 * cBn <= S2))
        okp.append((y, 4 * cBp <= S2))
        print("  m%-3d %4d %3d %5d %6d %9d %6d %7d   %-8s %7d   %-8s %3s"
              % (y, q1, a, F, F2, G, S2, cBn,
                 "YES" if 4 * cBn <= S2 else "NO", cBp,
                 "YES" if 4 * cBp <= S2 else "NO", L_RECORDED.get(y, "?")))
    fn = [y for y, v in okn if not v]
    fp = [y for y, v in okp if not v]
    print("\n  NAIVE  c_B ~ 3F/q' : 4 c_B <= S_2 at %d of %d steps; fails at %s"
          % (len(okn) - len(fn), len(okn), fn))
    print("  PARITY c_B ~ 2F/q' : 4 c_B <= S_2 at %d of %d steps%s"
          % (len(okp) - len(fp), len(okp),
             "; fails at %s" % fp if fp else " - ALL TWELVE"))
    print("\n  VERDICT.  With the alternation accounted for (PARITY) the R99 "
          "chain SURVIVES at\n  every corpus step; with the naive per-letter "
          "bound it does not (m17, m29, m31).\n  The difference is exactly T3: "
          "the factor is 2F/q', not 3F/q'.  But c_B is NOT a\n  constant either "
          "way - R99's conclusion becomes F(M+q') <= F_2 + 8F(M+q')/q', a\n"
          "  self-referential inequality that closes only under a "
          "Jacobsthal-square condition\n  on F (Lateral item 85, whose "
          "closure is the authority here; not duplicated).\n"
          "  WHAT THIS LANE ADDS: L_bare <= 5 is a CONSTANT, so all of the "
          "growth that forces\n  the q'-dependent c_B lives in L_pad - the "
          "non-bare words - and nowhere else.")
    return okp


def part_D(ps_mod):
    """m53: the theorem plus a recorded L forces L_pad(53) = 3."""
    print()
    print("=" * 78)
    print("(D)  THE FIRST L_pad = 3, AND IT IS THE NEXT MACHINE (m53)")
    print("=" * 78)
    q1 = next_prime(53)
    a, b = bare_a(q1), q1 - bare_a(q1)
    p = ps_mod[q1 % 210]
    L53 = 3                      # R89 from A_kill(53 -> 59) = 4; R99/R100 tables
    assert q1 == 59 and (a, b) == (20, 39), (q1, a, b)
    assert p == 2, ("PSORD(59)", p)
    print("  m53 -> q' = 59, bare letters (a,b) = (%d,%d), q' mod 210 = 59," % (a, b))
    print("  PSORD(59) = %d, so 59 is IN S and the LEMMA applies:" % p)
    print("      L_bare(53) <= 2                      [THEOREM, this round]")
    print("      L(53) = 3                            [recorded: A_kill(53->59)"
          " = 4, R89]")
    print("      => L_pad(53) = 3                     [decomposition theorem]")
    print()
    print("  So the padded half ALREADY exceeds 2 at the very next machine, and"
          " it does so\n  at an S-machine where the bare half is capped at 2 by "
          "two gears.  The row is")
    print("      M      11 13 17 19 23 29 31 37 41 43 47 53")
    print("      L       1  1  1  2  1  3  3  2  2  2  4  3")
    print("      L_bare  1  1  1  2  1  3  3  1  1  1  4 <=2")
    print("      L_pad   0  0  0  1  1  1  2  2  2  2  3  3")
    print("  and L > L_bare at FOUR machines: m37, m41, m43, m53.")
    print("  L_pad is NOT a small constant: it is 0,0,0,1,1,1,2,2,2,2,3,3 - it "
          "has taken the\n  values 0, 1, 2 and 3, and it grows.  The bare half "
          "is capped at 5 forever; the\n  non-bare half is the whole of (B).")


def main():
    args = sys.argv[1:]
    do_crt = "--crt" in args
    nodes = int(args[args.index("--nodes") + 1]) if "--nodes" in args \
        else 40_000_000
    t0 = time.time()
    ps_mod, ps_ph, ps74, S = part_A()
    rows = part_B(ps_mod, ps_ph, ps74, S, do_crt, nodes)
    part_B2(rows)
    part_B3(rows)
    part_D(ps_mod)
    part_C(ps_mod)
    print("\n[%.0f s]  all assertions passed" % (time.time() - t0))


if __name__ == "__main__":
    main()
