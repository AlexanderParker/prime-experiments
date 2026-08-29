"""Round 27 (mechanic): record the SCAN-DERIVED zeros of the A_kill(53->59)
k=3 level into the campaign log, so the solver is never asked for a verdict a
scan already owns (rule 29: a refutation may be free; rule 20: when SAT
stalls, buy the bound elsewhere).

THE ARGUMENT, and it is exact.  A realised 3-chain word (a, b) at 53 -> 59 is
THREE CONSECUTIVE OPENINGS of machine 53 with those gaps (every other slot of
the span blocked - that is what `check_occurrence` verifies).  Its span a + b
is therefore the span of a 2-window of machine 53.  Two completed scans over
machine 23's period, each with range workers that TILE it exactly, say which
2-window spans exist:

  * round 26, F_2(53): seed 145, cap 200, maximum 159.
    => NO 2-window of machine 53 has span in (159, 200].
    NOTE THE SCOPE.  "F_2(53) <= 159" is conditional on that run's span cap
    200 (the deletion-ladder cap F_2(53) <= F(59) was unavailable).  But "no
    2-window has span in (159, 200]" is NOT conditional on anything - the cap
    only conditions claims about spans ABOVE it.  Only the second statement is
    used here.
  * round 27, F(59) stage A: seed 203, cap 260, maximum 203 (= the seed).
    Q*_2 has no middle gaps so its legality condition is vacuous and
    Q*_2 = F_2 over the band.  => NO 2-window has span in (203, 260].

Every legal k=3 word whose span lands in either band is ZERO, with no solver.

Usage: python research/akill_scan_verdicts_r27.py [--write]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from akill_verify_r27 import legal_k3, dead_gear, parse_ranges  # noqa: E402
from akill_verify_r27 import (F2_LOGS, FA_LOGS, FT_LOGS, WJ3_LOGS,  # noqa
                              NOPEN23, F2_VALUE, F2_CAP, FA_SEED, FA_CAP,
                              FT_CAP, FT_VALUE, WJ3_VALUE, WJ3_CAP,
                              WJ3_JMAX, check_tiling)

LOG = os.path.join(HERE, "data", "r27", "akill_53_59.log")
QP_ = 59


def main():
    r2, m2 = parse_ranges(F2_LOGS)
    check_tiling(r2, NOPEN23, "F_2(53)")
    assert max(m2) == F2_VALUE, m2
    ra, ma = parse_ranges(FA_LOGS)
    check_tiling(ra, NOPEN23, "F(59) stage A")
    assert max(ma) == FA_SEED, ("stage A found something above its seed", ma)
    print("scan 1 (r26 F_2(53)):  tiles [0,%d), maxima %s -> no 2-window span "
          "in (%d, %d]" % (NOPEN23, m2, F2_VALUE, F2_CAP))
    print("scan 2 (r27 stage A):  tiles [0,%d), maxima %s -> no 2-window span "
          "in (%d, %d]" % (NOPEN23, ma, FA_SEED, FA_CAP))
    rt, mt = parse_ranges(FT_LOGS)
    check_tiling(rt, NOPEN23, "m53 top band")
    assert max(mt) == FT_VALUE, mt
    print("scan 3 (r27 top band): tiles [0,%d), maxima %s -> no 2-window span "
          "in (%d, %d]" % (NOPEN23, mt, FT_VALUE, FT_CAP))

    _, words, _ = legal_k3()
    # DEEPER LEVELS.  A realised k-chain is k consecutive openings of machine
    # 53 whose k-1 gaps are ALL legal letters, so it IS a word-legal window of
    # J = k-1 gaps (the J-2 middles are legal a fortiori).  The stage-A run
    # covered J = 2..7 at cap 260 with seed 203 and found nothing, so ANY word
    # of span in (203, 260] with at most 8 members is zero.  The two J = 2
    # scans (r26 F_2 and the r27 top band) refute only 3-CHAIN words, since
    # their windows have exactly two gaps.
    deeper = []
    from itertools import product
    from akill_verify_r27 import s_of, letters, window_valid   # noqa: E402
    vals = [v for v in range(1, 146)
            if v % QP_ in {0, s_of(QP_), (-s_of(QP_)) % QP_}]
    for nlet in (3, 4, 5):
        for w in product(vals, repeat=nlet):
            L = letters(w, QP_)
            if L is None or not window_valid(L):
                continue
            if dead_gear(w):
                continue
            if FA_SEED < sum(w) <= FA_CAP:
                deeper.append((w, "r27 F(59) stage-A band, span in (%d,%d], "
                                  "depth J = %d <= 7"
                                  % (FA_SEED, FA_CAP, nlet)))
            elif nlet <= WJ3_JMAX and WJ3_VALUE < sum(w) <= WJ3_CAP:
                deeper.append((w, "r27 depth-3 word-legal band, span in "
                                  "(%d,%d], depth J = %d <= %d"
                                  % (WJ3_VALUE, WJ3_CAP, nlet, WJ3_JMAX)))

    lines = []
    for w in words:                       # BOTH orientations, so the driver's
        sp = sum(w)                       # mirror step never re-queues one
        if dead_gear(w):
            continue                      # the screen already answers these
        if FT_VALUE < sp <= FT_CAP:
            why = "r27 m53 top-band scan, span in (%d,%d]" % (FT_VALUE,
                                                              FT_CAP)
        elif F2_VALUE < sp <= F2_CAP:
            why = "r26 F_2(53) scan, span in (%d,%d]" % (F2_VALUE, F2_CAP)
        elif FA_SEED < sp <= FA_CAP:
            why = "r27 F(59) stage-A band, span in (%d,%d]" % (FA_SEED, FA_CAP)
        else:
            continue
        lines.append("  RESULT m53 word (%d, %d) span %d: ZERO (0 SAT calls "
                     "- %s)" % (w[0], w[1], sp, why))
    rj, mj = parse_ranges(WJ3_LOGS)
    check_tiling(rj, NOPEN23, "depth-3 band")
    assert max(mj) == WJ3_VALUE, mj
    print("scan 4 (r27 depth-3):  tiles [0,%d), maxima %s -> no word-legal "
          "window of depth J <= %d with span in (%d, %d]"
          % (NOPEN23, mj, WJ3_JMAX, WJ3_VALUE, WJ3_CAP))
    n3 = len(lines)
    for w, why in deeper:
        lines.append("  RESULT m53 word (%s) span %d: ZERO (0 SAT calls - %s)"
                     % (", ".join(str(x) for x in w), sum(w), why))
    for ln in lines:
        print(ln)
    print("\n%d of the 36 legal k=3 words refuted by scan alone; "
          "%d deeper words (k = 4,5,6) as well" % (n3, len(deeper)))
    if "--write" in sys.argv:
        with open(LOG, "a", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln + "\n")
        print("appended to %s" % LOG)


if __name__ == "__main__":
    main()
