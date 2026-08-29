"""Round 27 (mechanic): the A_kill(53 -> 59) campaign, ENUMERATED AND PRICED.

C30 priced this at 27 solver calls using F_2(53) <= 159 as the 2-block span
cap.  That bound's `<=` direction is CONDITIONAL on the round-26 span cap 200
(the deletion-ladder cap F_2(53) <= F(59) is unavailable - the corpus F ladder
stops at 53).  A verdict resting on it would inherit that condition.

So this script enumerates the level BOTH WAYS:
  * CONDITIONAL caps  [145, 159, 304]  - F_1 exact, F_2 from C30, F_3 trivial
  * UNCONDITIONAL caps [145, 290, 435] - F_2 <= 2 F_1, F_3 <= 3 F_1
and reports exactly how many extra words unconditionality costs, after the
phase-saturation screen (K9) and the mirror halving (rule 27).  If the extra
words are few, the campaign is run unconditionally and the F_2 cap is not used
as an input at all.

Usage: python research/akill_53_59_plan_r27.py
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_kill import enumerate_words, legal_values, sub_ok      # noqa: E402
# alt_obstruct_r26 prints its own round-26 report at import time; swallow it.
with contextlib.redirect_stdout(io.StringIO()):
    from alt_obstruct_r26 import first_dead_gear              # noqa: E402

Y, QP = 53, 59


def screen(words):
    """Split by the phase-saturation obstruction (K9): zero, or needs SAT."""
    dead, live = [], []
    for w in words:
        X = [0]
        for g in w:
            X.append(X[-1] + g)
        (dead if first_dead_gear(X, Y) else live).append(w)
    return dead, live


def classes(words):
    """One representative per reverse class (rule 27)."""
    seen, reps = set(), []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        seen.add(w[::-1])
        reps.append(w)
    return reps


def report(caps, label):
    s, V, vals = legal_values(Y, QP)
    print("\n=== %s   caps %s" % (label, caps))
    print("    s = %d, V = %s, legal gap values %s" % (s, V, vals))
    prev = None
    plan = {}
    for k in (3, 4, 5, 6):
        _, _, _, words = enumerate_words(Y, QP, k - 1, caps)
        n_legal = len(words)
        if prev is not None:
            words = [w for w in words if sub_ok(w, prev)]
        n_ov = len(words)
        dead, live = screen(words)
        reps = classes(live)
        print("  k=%d: legal %5d -> after overlap %5d -> after screen %4d "
              "-> reverse classes %4d   (screen killed %d)"
              % (k, n_legal, n_ov, len(live), len(reps), len(dead)))
        plan[k] = reps
        # for the next level's overlap prune we must assume the LIVE words
        # could be realised (sound: the prune only uses realised sub-words,
        # and a screened-dead word is certainly not realised)
        prev = {w: 1 for w in live}
        if not live:
            print("       level is EMPTY - every deeper level is empty too")
            break
    return plan


def main():
    cond = report([145, 159, 304], "CONDITIONAL (uses F_2(53) <= 159, C30)")
    unc = report([145, 290, 435], "UNCONDITIONAL (F_2 <= 2F, F_3 <= 3F)")
    print("\nEXTRA COST OF UNCONDITIONALITY (reverse classes needing SAT):")
    tot_c = tot_u = 0
    for k in sorted(set(cond) | set(unc)):
        a, b = len(cond.get(k, [])), len(unc.get(k, []))
        tot_c += a
        tot_u += b
        print("  k=%d: %4d conditional  vs %4d unconditional  (+%d)"
              % (k, a, b, b - a))
    print("  TOTAL: %d vs %d solver calls (+%d)" % (tot_c, tot_u,
                                                    tot_u - tot_c))
    with open(os.path.join(HERE, "data", "r27",
                           "akill_53_59_plan.txt"), "w") as fh:
        for k in sorted(unc):
            for w in unc[k]:
                fh.write("%d %s\n" % (k, ",".join(str(x) for x in w)))
    print("\nword plan written to research/data/r27/akill_53_59_plan.txt")


if __name__ == "__main__":
    main()
