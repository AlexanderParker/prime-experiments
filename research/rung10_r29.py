"""Round 29 (constructor): RUNG TEN, 43 -> 47, BY THE SPECTRUM-PLUS-DEPTH
CERTIFICATE - ratified from a clean process, with an INDEPENDENT gate that does
not consume any of Mechanic's F_J(43) values.

THE THEOREM (R84, restated here with every hypothesis named).

  Let M = {5..y} be a machine, q' = nextprime(y), c = 6^{-1} mod q',
  a = 2c mod q' written at its least positive representative, b = q' - a,
  s_min = min(a,b).

  (H1) MERGE LAW + T2/T3 (R40, proved).  Every gap of M + q' is a span of J
       consecutive gaps of M whose J-2 middles are each 0 or +-2c mod q' and
       whose nonzero classes strictly alternate ("word-legal J-window").
  (H2) ATTAINMENT THEOREM (R68, proved both ways).  Conversely every realised
       word-legal J-window's span is <= F(M+q').  Hence
           F(M + q') = max_J Q*_J(M; q').
  (H3) Q*_J <= F_J(M) by definition (a word-legal J-window IS a window of J
       consecutive gaps).
  (H4) EMPTINESS IS UPWARD CLOSED (proved).  Deleting a flank of a word-legal
       J-window leaves a word-legal (J-1)-window, so Q*_{J} = -inf for every
       J > J_max(M), J_max = A_kill(M -> q') + 1.

  THEOREM.  F(M + q')  <=  max_{2 <= J <= J_max(M)}  F_J(M).
  (D) at alpha = 3 holds at the step whenever that maximum is <= F(M) + q'.

PART A ratifies the rung with the recorded inputs (Mechanic r28: F_2(43) = 116,
F_3(43) = 125, F_4(43) = 132; A_kill(43 -> 47) = 3).

PART B is this lane's OWN certification of the same rung, computed here, which
uses none of those three numbers.  It rests only on

  (C1) the DELETION-LADDER CAP (proved, three lines - see `deletion_cap_note`):
       F_j(M) <= F(M + the next j-1 primes).  With F(47) = 118, F(53) = 145,
       F(59) = 161 on the corpus record,
           F_2(43) <= 118,  F_3(43) <= 145,  F_4(43) <= 161,
       so J = 2 and J = 3 are ALREADY under the budget 150 with nothing to
       compute, and the only live band in the whole rung is J = 4, spans
       151..161.
  (C2) that band, refuted here by exact CRT decisions (no period anywhere).
  (C3) the depth cap, re-derived here from scratch by the WORD ROUTE
       (theorem R89 of this round: J_max(M) = L(M) + 2, L = longest realised
       word-legal letter word), so the J = 5 emptiness is not quoted either.

HONEST DEPENDENCY, stated in the output: (C1) uses F(47), and F(47) <= 150 is
what the rung asserts, so PART B is not a logically independent proof of
F(47) <= 150 - no rung below machine 59 can be, because the corpus knows F
there outright.  What PART B establishes is that the CERTIFICATE'S OWN
obligation at this step is a bounded, finite, machine-43-only computation, and
it discharges it.  PART D prices the version that drops (C1) entirely.

Usage:
  uv run python research/rung10_r29.py            # A + C + D  (seconds)
  uv run python research/rung10_r29.py --sweep    # + PART B's J=4 band
  uv run python research/rung10_r29.py --sweep --floor 131   # descend further
"""
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                              # noqa: E402

# ---------------------------------------------------------------- corpus ----
# F(M) for M = {5..y}: the project's exact ladder.  Used as ASSERTED INPUT and
# re-derived below for y <= 19 from the period so the table is not just quoted.
KNOWN_F = {5: 4, 7: 4, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58,
           37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
            41: 103, 43: 116, 53: 159}
# Mechanic round-28 recorded spectrum of machine 43 (PART A only).
MECH_FJ_43 = {2: 116, 3: 125, 4: 132}
MECH_AKILL_43 = 3


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def next_prime(y):
    p = y + 1
    while not is_prime(p):
        p += 1
    return p


def gears_of(y):
    return [p for p in range(5, y + 1) if is_prime(p)]


def exposed(g):
    c = pow(6, -1, g)
    return frozenset(r for r in range(g) if r % g != c % g and r % g != (-c) % g)


def letters(y):
    """(q', a, b, s_min, legal letter values <= F(y) tagged with T3 class)."""
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    F = KNOWN_F[y]
    out = []
    for v in range(1, F + 1):
        r = v % q1
        if r == 0:
            out.append((v, 0))
        elif r == a % q1:
            out.append((v, 1))
        elif r == b % q1:
            out.append((v, -1))
    return q1, a, b, min(a, b), out


def legal_words(y, n, spec):
    """All T3-legal letter words of length n, pruned by the spectrum caps."""
    _, _, _, _, LV = letters(y)
    words = [[]]
    for _ in range(n):
        nxt = []
        for w in words:
            last = next((c for _, c in reversed(w) if c), 0)
            for v, c in LV:
                if c and c == last:
                    continue                                # T3 alternation
                cand = w + [(v, c)]
                if spec_ok([x for x, _ in cand], spec):
                    nxt.append(cand)
        words = nxt
    return [tuple(v for v, _ in w) for w in words]


def spec_ok(vs, spec):
    for j in range(1, min(len(vs), max(spec)) + 1):
        lim = spec.get(j)
        if lim is None:
            continue
        for i in range(0, len(vs) - j + 1):
            if sum(vs[i:i + j]) > lim:
                return False
    return True


def ps_refuted(X, gears, E):
    """Phase saturation (Mechanic r26): sound, machine-free, free."""
    for g in gears:
        Eg = E[g]
        xs = {x % g for x in X}
        if len(xs) > g - 2:
            return True
        if not any(all((t + x) % g in Eg for x in xs) for t in range(g)):
            return True
    return False


def prefix(t):
    X, acc = [0], 0
    for v in t:
        acc += v
        X.append(acc)
    return X


def deletion_cap_note():
    return (
        "DELETION-LADDER CAP (proved).  Let x_0 < ... < x_j be j+1 consecutive\n"
        "  openings of M and q'_1..q'_{j-1} the next j-1 primes.  Gear q'_i\n"
        "  blocks slot k iff k = +-c_i mod q'_i; P(M) is invertible mod each\n"
        "  q'_i, so by CRT some translate x + t.P(M) has x_i + t.P(M) = c_i\n"
        "  mod q'_i for every i = 1..j-1 simultaneously.  In that translate\n"
        "  every interior opening is killed, and the openings of the enlarged\n"
        "  machine are a subset of M's, so no opening lies strictly between\n"
        "  the images of x_0 and x_j.  Hence the enlarged machine has a gap of\n"
        "  length >= x_j - x_0, i.e.\n"
        "      F_j(M)  <=  F(M + {the next j-1 primes}).\n")


# --------------------------------------------------------------- workers ----
def job(args):
    y, tup, nb = args
    t0 = time.time()
    try:
        return tup, crt_dict.realised(y, tup, node_budget=nb), time.time() - t0
    except Exception:
        return tup, None, time.time() - t0


def decide_all(y, tuples, workers, nb, label):
    """Returns (realised list, undecided list); prints a per-tuple line."""
    if not tuples:
        print("      %s: nothing to decide" % label)
        return [], []
    t0 = time.time()
    yes, und = [], []
    with Pool(workers) as pool:
        for tup, ok, dt in pool.imap_unordered(
                job, [(y, t, nb) for t in tuples], chunksize=1):
            if ok is True:
                yes.append(tup)
            elif ok is None:
                und.append(tup)
    print("      %s: %d decided, %d REALISED, %d undecided  [%.0f s]"
          % (label, len(tuples), len(yes), len(und), time.time() - t0),
          flush=True)
    return yes, und


# ------------------------------------------------------------------ gates ---
def gate_small_machines():
    """Re-derive F and F_2 at m11..m19 from the period; assert the table."""
    import numpy as np
    from math import prod
    ok = []
    for y in (11, 13, 17, 19):
        gears = gears_of(y)
        P = prod(gears)
        ex = np.zeros(P, bool)
        for g in gears:
            u = pow(6, -1, g)
            ex[u % g::g] = True
            ex[(-u) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64)
        d = np.diff(np.concatenate([op, [op[0] + P]]))
        F = int(d.max())
        F2 = int((d + np.roll(d, -1)).max())
        assert F == KNOWN_F[y], ("F gate", y, F, KNOWN_F[y])
        assert F2 == KNOWN_F2[y], ("F2 gate", y, F2, KNOWN_F2[y])
        ok.append((y, F, F2))
    print("  GATE small machines (period scan, F and F_2 re-derived): "
          + ", ".join("m%d F=%d F_2=%d" % t for t in ok))
    print("  ASSERTED against the corpus table: PASS")


def main():
    args = sys.argv[1:]

    def opt(nm, d):
        return type(d)(args[args.index(nm) + 1]) if nm in args else d

    y = 43
    workers = opt("--workers", 6)
    nb = opt("--nodes", 2_000_000)
    floor = opt("--floor", 150)
    do_sweep = "--sweep" in args
    q1, a, b, s_min, LV = letters(y)
    F = KNOWN_F[y]
    budget = F + q1
    gears = gears_of(y)
    E = {g: exposed(g) for g in gears}

    print("=" * 74)
    print("RUNG TEN:  machine 43 -> q' = 47")
    print("  q' = %d, c = 6^-1 = %d, a = %d, b = %d, s_min = %d"
          % (q1, pow(6, -1, q1), a, b, s_min))
    print("  F(43) = %d  (corpus, exact) ;  BUDGET F(43) + q' = %d"
          % (F, budget))
    print("  legal letters <= F(43): %s"
          % ", ".join("%d(%+d)" % (v, c) for v, c in LV))
    assert budget == 150
    gate_small_machines()

    # ------------------------------------------------------------- PART A ---
    print("\n" + "-" * 74)
    print("PART A - RATIFICATION WITH THE RECORDED INPUTS (Mechanic r28)")
    Jmax = MECH_AKILL_43 + 1
    vals = [MECH_FJ_43[J] for J in range(2, Jmax + 1)]
    bound = max(vals)
    print("  J_max(43) = A_kill(43->47) + 1 = %d" % Jmax)
    for J in range(2, Jmax + 1):
        print("     F_%d(43) = %3d   (<= deletion-ladder cap %d)"
              % (J, MECH_FJ_43[J], KNOWN_F[next_prime(y) if J == 2 else
                                          (53 if J == 3 else 59)]))
    caps = {2: KNOWN_F[47], 3: KNOWN_F[53], 4: KNOWN_F[59]}
    for J in range(2, Jmax + 1):
        assert MECH_FJ_43[J] <= caps[J], ("recorded F_J above its own cap", J)
    print("  ASSERTION: every recorded F_J(43) respects its deletion cap: PASS")
    print("  CERTIFICATE:  F(47) <= max_J F_J(43) = %d  <=  %d = F(43)+47"
          % (bound, budget))
    assert bound <= budget
    print("  ==> (D) AT 43 -> 47 CERTIFIED, MARGIN %+d" % (budget - bound))
    print("  (corollary from machine 43 alone: F(47) <= %d ; true value %d)"
          % (bound, KNOWN_F[47]))
    print("  HYPOTHESES CONSUMED: H1 merge law, H2 attainment theorem,")
    print("    H3 Q*_J <= F_J, H4 upward-closed emptiness, plus the three")
    print("    recorded F_J(43) and A_kill(43->47) = 3.")

    # ------------------------------------------------------------- PART C ---
    # depth cap re-derived by the WORD route, not quoted.
    print("\n" + "-" * 74)
    print("PART C - THE DEPTH CAP, RE-DERIVED HERE (word route, R89)")
    print("  R89 (this round, proved): a word-legal J-window's J-2 middles are")
    print("  J-2 CONSECUTIVE gaps of M, each a legal letter, T3-alternating -")
    print("  i.e. a realised legal WORD of length J-2; and conversely any such")
    print("  word extends to a legal J-window by its two flanking gaps.  Hence")
    print("      Q*_J > -inf  <=>  L(M) >= J-2,   so  J_max(M) = L(M) + 2,")
    print("  L(M) = longest realised legal word.  (A_kill = L + 1.)")
    spec = {1: F, 2: KNOWN_F[47], 3: KNOWN_F[53], 4: KNOWN_F[59]}
    print("  spectrum caps used (deletion ladder): F_1..F_4 <= %s"
          % [spec[j] for j in (1, 2, 3, 4)])
    w1 = legal_words(y, 1, spec)
    w2 = legal_words(y, 2, spec)
    w3 = legal_words(y, 3, spec)
    print("  legal candidate words: length 1: %d, length 2: %d, length 3: %d"
          % (len(w1), len(w2), len(w3)))
    print("     len 2: %s" % sorted(w2))
    print("     len 3: %s" % sorted(w3))
    ps3 = [w for w in w3 if not ps_refuted(prefix(w), gears, E)]
    print("  phase saturation refutes %d of %d length-3 words for free"
          % (len(w3) - len(ps3), len(w3)))
    if do_sweep:
        yes3, und3 = decide_all(y, ps3, workers, nb, "length-3 words")
        if not yes3 and not und3:
            print("  ==> L(43) <= 2, so J_max(43) <= 4 and Q*_5(43) = -inf")
            print("      A_kill(43 -> 47) <= 3, re-derived with no census.")
        else:
            print("  ==> NOT closed: realised %s, undecided %s"
                  % (sorted(yes3), sorted(und3)))
    else:
        print("  (run with --sweep to decide the length-3 words)")

    # ------------------------------------------------------------- PART B ---
    print("\n" + "-" * 74)
    print("PART B - THIS LANE'S OWN CERTIFICATION (no F_J(43) consumed)")
    print(deletion_cap_note())
    print("  F_2(43) <= F(47) = %d  <= %d = budget   -> J = 2 CLEAR, no work"
          % (KNOWN_F[47], budget))
    print("  F_3(43) <= F(53) = %d  <= %d = budget   -> J = 3 CLEAR, no work"
          % (KNOWN_F[53], budget))
    print("  F_4(43) <= F(59) = %d  >  %d = budget   -> ONE LIVE BAND"
          % (KNOWN_F[59], budget))
    print("  Live obligation: no word-legal 4-window of machine 43 has span in"
          " [%d, %d]." % (floor + 1, KNOWN_F[59]))
    # candidates
    cand = {}
    for w in w2:
        m = sum(w)
        for gL in range(1, F + 1):
            for gR in range(1, F + 1):
                t = (gL,) + w + (gR,)
                if not spec_ok(list(t), spec):
                    continue
                s = gL + m + gR
                if s <= floor or s > KNOWN_F[59]:
                    continue
                cand[min(t, t[::-1])] = s
    print("  candidates after spectrum + T3 + mirror filters: %d" % len(cand))
    live = [(s, t) for t, s in cand.items()
            if not ps_refuted(prefix(t), gears, E)]
    print("  phase saturation refutes %d for free; %d go to CRT"
          % (len(cand) - len(live), len(live)))
    if do_sweep:
        blocks = {}
        for s, t in live:
            blocks.setdefault(s, []).append(t)
        allyes, allund = [], []
        for s in sorted(blocks, reverse=True):
            yes, und = decide_all(y, blocks[s], workers, nb, "span %3d" % s)
            allyes += yes
            allund += und
        if allyes:
            print("  ==> A WINDOW SURVIVES: %s - the rung is NOT closed by "
                  "this route" % sorted(allyes))
        elif allund:
            print("  ==> %d UNDECIDED at node budget %d - band not closed"
                  % (len(allund), nb))
        else:
            print("  ==> EVERY word-legal 4-window of span > %d REFUTED." % floor)
            print("      With J=2, J=3 clear by the deletion cap and J>=5 empty")
            print("      by PART C, max_J Q*_J(43; 47) <= %d <= %d = F(43)+47."
                  % (floor, budget))
            print("      (D) AT 43 -> 47 CERTIFIED BY THIS LANE'S OWN GATE.")
    else:
        print("  (run with --sweep to decide the band)")

    # ------------------------------------------------------------- PART D ---
    print("\n" + "-" * 74)
    print("PART D - PRICING THE VERSION THAT DROPS THE DELETION CAP")
    print("  Without (C1) the only machine-43-internal caps are F_j <= j.F(43)")
    print("  = %d / %d / %d at j = 2/3/4.  The obligation becomes 'no word-"
          "legal" % (2 * F, 3 * F, 4 * F))
    print("  J-window of span in [%d, j.F]' for J = 2,3,4:" % (floor + 1))
    spec_free = {1: F, 2: 2 * F, 3: 3 * F, 4: 4 * F}
    tot = 0
    for J in (2, 3, 4):
        ws = legal_words(y, J - 2, spec_free) if J > 2 else [()]
        c = {}
        for w in ws:
            m = sum(w)
            for gL in range(1, F + 1):
                for gR in range(1, F + 1):
                    t = (gL,) + w + (gR,)
                    if not spec_ok(list(t), spec_free):
                        continue
                    s = gL + m + gR
                    if s <= floor or s > J * F:
                        continue
                    c[min(t, t[::-1])] = s
        liv = [t for t in c if not ps_refuted(prefix(t), gears, E)]
        tot += len(liv)
        print("     J = %d : %6d candidates, %6d survive phase saturation"
              % (J, len(c), len(liv)))
    print("  TOTAL CRT decisions to make the rung deletion-cap-free: %d" % tot)
    print("  At the measured m43 cost profile that is the price of the fully")
    print("  self-contained rung; it is NOT paid here.")
    print("=" * 74)


if __name__ == "__main__":
    main()
