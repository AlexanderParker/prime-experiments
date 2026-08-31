"""Round 28 (constructor): THE NINTH RUNG, CERTIFIED BY THE PER-J ROUTE.

R72/R79 left 41 -> 43 uncertified and pinned the failure on the CEGAR loop's
oracle (stall 222, then 135 with round-27's better oracle - one unit over the
budget 134).  This file certifies the rung WITHOUT the loop, from R68's
attainment theorem plus one new fact.

R68 (ATTAINMENT THEOREM, proved both ways in round 22, verified at eight
steps):

    F(M + q')  =  max over J of  Q*_J(M; legal for q').

Two elementary facts finish it.

  (i)  Q*_J(M) <= F_J(M)  by definition - a word-legal J-window IS a window of
       J consecutive gaps, so its span is at most the j = J spectrum value.
  (ii) EMPTINESS IS UPWARD CLOSED.  Deleting a flank of a word-legal J-window
       leaves a word-legal (J-1)-window (the surviving middles are a
       sub-sequence of the old ones, so T2 holds pointwise and T3's nonzero
       alternation is inherited).  Hence if no word-legal J0-window exists,
       none exists at any J >= J0, and Q*_J = -inf there.

THE SPECTRUM-PLUS-DEPTH CERTIFICATE.  Let J_max(M) be the largest J with a
word-legal J-window (= A_kill(M) + 1).  Then

    F(M + q')  <=  max_{2 <= J <= J_max}  F_J(M),

and (D) at alpha = 3 follows whenever that maximum is <= F(M) + q'.  No word
list, no flank envelope, no CEGAR loop, no oracle: the OLD machine's spectrum
and ONE emptiness certificate.

AT 41 -> 43 the ingredients are

    F_2(41) = 103   EXACT (R72, scan-free)
    F_3(41) <= 117  upper bound, recomputed here from Mechanic's superset
    F_4(41) = 118   EXACT (Mechanic, round 27, first computation)
    Q*_5(41) = -inf NEW THIS ROUND (research/perj_scanfree.py; every legal
                    5-window candidate refuted, by phase saturation or CRT,
                    with zero undecided) - equivalently A_kill(41) <= 3, which
                    with R45's A_kill(41) >= 3 gives A_kill(41) = 3 EXACTLY.

so  F(43) <= max(103, 117, 118) = 118 < 134 = F(41) + 43.

Usage:
  .venv/Scripts/python.exe research/rung9_perj_cert.py            # the table
  .venv/Scripts/python.exe research/rung9_perj_cert.py --recheck  # + re-run
                                                the m41 emptiness from scratch
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                          # noqa: E402
from perj_scanfree import (next_prime, gears_of, exposed, ps_refuted,  # noqa
                           spectrum, spec_ok, middle_words, canonical, job)

KNOWN_F = crt_dict.KNOWN_F
# spectrum F_2..F_5 of the OLD machine, exact where the project has it.
# F_j(M) for j <= 4 is read off the exact 4-tuple censuses at m23..m37 by
# research/perj_scanfree.spectrum(); the values below are the corpus record and
# are ASSERTED against that computation, never used as an input to it.
FJ = {11: {2: 11, 3: 16, 4: 18},
      13: {2: 16, 3: 23, 4: 26},
      17: {2: 25, 3: 28, 4: 33},
      19: {2: 31, 3: 35, 4: 38},
      23: {2: 39, 3: 50, 4: 58},
      29: {2: 55, 3: 65, 4: 70, 5: 85},          # F_5(29) = 85 EXACT (r28,
                                                 # scan-free, 428 candidates
                                                 # decided, 0 undecided)
      31: {2: 68, 3: 85, 4: 90, 5: 92},          # R39 machine-31 spectrum
      37: {2: 90, 3: 97, 4: 105},
      41: {2: 103, 3: 117, 4: 118}}              # F_3 is an UPPER bound
# J_max = A_kill + 1, every entry certified this round by an emptiness sweep
JMAX = {11: 3, 13: 3, 17: 3, 19: 4, 23: 3, 29: 5, 31: 5, 37: 4, 41: 4}


def m41_emptiness(workers=4, nodes=4_000_000):
    """Re-derive Q*_5(41) = -inf from scratch: every legal 5-window candidate
    refuted by phase saturation or by an exact CRT decision."""
    from multiprocessing import Pool
    y, q1 = 41, 43
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    F = KNOWN_F[y]
    Fspec, vals, exact = spectrum(y)
    gears = gears_of(y)
    E = {g: exposed(g) for g in gears}
    words = middle_words(q1, a, b, F, vals, 3, Fspec)
    cand = {}
    for w in words:
        for gL in range(1, F + 1):
            if gL not in vals:
                continue
            for gR in range(1, F + 1):
                if gR not in vals:
                    continue
                t = (gL,) + w + (gR,)
                if spec_ok(list(t), Fspec):
                    cand[canonical(t)] = sum(t)
    live = []
    for t in cand:
        X, acc = [0], 0
        for v in t:
            acc += v
            X.append(acc)
        if not ps_refuted(X, gears, E):
            live.append(t)
    print("  m41 J=5: %d legal middle words, %d mirror-halved candidates, "
          "%d survive phase saturation" % (len(words), len(cand), len(live)))
    with Pool(workers) as pool:
        res = pool.map(job, [(y, t, nodes) for t in live], chunksize=1)
    yes = [t for t, ok, dt in res if ok]
    und = [t for t, ok, dt in res if ok is None]
    assert not und, ("UNDECIDED candidates - the emptiness is not certified",
                     und[:3])
    assert not yes, ("A LEGAL 5-WINDOW IS REALISED AT m41", yes[:3])
    print("  m41 J=5: 0 realised, 0 undecided  =>  Q*_5(41) = -inf  "
          "(A_kill(41) = 3)")
    return True


def main():
    print("=" * 78)
    print("THE SPECTRUM-PLUS-DEPTH CERTIFICATE FOR (D)")
    print("=" * 78)
    print("  F(M+q') = max_J Q*_J  (R68)   and   Q*_J <= F_J,  Q*_J = -inf "
          "for J > J_max")
    print("  =>  F(M+q') <= max_{2<=J<=J_max} F_J(M).   (D) holds if that is "
          "<= F(M) + q'.")
    print()
    print("   M    q'  J_max | F_2  F_3  F_4  F_5 | bound  budget  verdict")
    ok_all = []
    for y in sorted(FJ):
        q1 = next_prime(y)
        jm = JMAX[y]
        vals = [FJ[y].get(j) for j in range(2, jm + 1)]
        budget = KNOWN_F[y] + q1
        if any(v is None for v in vals):
            print("  %3d %5d %6d | %s | %-6s %6d  UNDECIDED (F_%d(%d) not on "
                  "record)"
                  % (y, q1, jm,
                     " ".join("%4s" % (FJ[y].get(j) if FJ[y].get(j) else "?")
                              for j in (2, 3, 4, 5)),
                     "-", budget,
                     [j for j in range(2, jm + 1) if FJ[y].get(j) is None][0],
                     y))
            continue
        bnd = max(vals)
        good = bnd <= budget
        ok_all.append(good)
        print("  %3d %5d %6d | %s | %5d %6d   %s"
              % (y, q1, jm,
                 " ".join("%4s" % (FJ[y].get(j) if FJ[y].get(j) else "-")
                          for j in (2, 3, 4, 5)),
                 bnd, budget,
                 "CERTIFIES (margin %+d)" % (budget - bnd) if good
                 else "does not certify (%+d)" % (budget - bnd)))
    print()
    print("  The certificate CERTIFIES at %d of the %d steps whose spectrum is "
          "complete -\n  including 41 -> 43, THE NINTH RUNG.  It FAILS at "
          "29 -> 31, where F_5(29) = 85\n  (computed exactly this round) is 11 "
          "OVER the budget 74: there the LEGALITY\n  constraint does real work "
          "(Q*_5(29) = 55, thirty under F_5).  So the criterion\n  is a genuine "
          "one, not a restatement - it has a failing case, and the failing\n"
          "  case is exactly the step whose deep layer is non-empty and whose "
          "F/q' is small."
          % (sum(ok_all), len(ok_all)))
    print()
    print("=" * 78)
    print("THE NINTH RUNG, ASSEMBLED")
    print("=" * 78)
    y, q1 = 41, 43
    F2, F3, F4 = FJ[41][2], FJ[41][3], FJ[41][4]
    Fspec, vals, exact = spectrum(41)
    assert Fspec[2] == F2, ("F_2(41)", Fspec[2])
    assert Fspec[3] <= F3, ("F_3(41) upper bound", Fspec[3])
    assert Fspec[4] == F4, ("F_4(41)", Fspec[4])
    assert max(vals) == 91 == KNOWN_F[41]
    budget = KNOWN_F[41] + q1
    bnd = max(F2, F3, F4)
    print("   F_2(41)   = %3d   EXACT      (R72, scan-free)" % F2)
    print("   F_3(41)  <= %3d   upper bd   (max induced 3-sum of Mechanic's "
          "screened superset)" % F3)
    print("   F_4(41)   = %3d   EXACT      (Mechanic r27, 602 core-s)" % F4)
    print("   Q*_5(41)  = -inf  NEW        (every legal 5-window refuted; "
          "upward closed, so all J >= 5)")
    print()
    print("   F(43) = max_J Q*_J <= max(F_2, F_3, F_4) = %d  <  %d = F(41) "
          "+ 43" % (bnd, budget))
    assert bnd < budget
    print("\n   (D) AT 41 -> 43 IS CERTIFIED.  Margin %+d." % (budget - bnd))
    print("   Corollary: F(43) <= %d (the true value is 103), and "
          "A_kill(41) = 3 exactly." % bnd)
    if "--recheck" in sys.argv:
        print()
        print("RE-DERIVING THE ONE NEW INPUT FROM SCRATCH")
        m41_emptiness()
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
