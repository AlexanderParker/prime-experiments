"""Round 28 (mechanic): the onset ladder's bottom rung, 11 -> 13 - and the step
where the SIMPLE form of the onset law FAILS.

Machine 11's period is 385 slots (135 openings), so this rung costs nothing and
extends the ladder one step below where the round's other scripts start.  It is
a real test, not padding: nothing about the law was fitted to it.

RESULT.  onset(11 -> 13) = 13, while min span of D_4(17) \\ D_4(13) = 10 - so
the SIMPLE law (7/8 elsewhere) MISSES here.  The reason is exactly the
refinement the law needs: the right-hand side must be intersected with what the
transfer can actually EMIT, and machine 11's dictionary (73 4-tuples) has no
walk that emits the span-10 witness (2,2,1,5) at all.  Intersected, the minimum
is 13 = the onset.  The CAUSAL form (the tuples refuted at the onset span are
realised at the next machine) holds here as everywhere.

usage: <venv>/python research/onset_m11_r28.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from onset_ladder_r28 import gaps_cyclic, ktuples      # noqa: E402
from dict_transfer import transfer                     # noqa: E402
from onset_r28 import screen                           # noqa: E402


def main():
    D = {y: ktuples(gaps_cyclic(y), 4) for y in (11, 13, 17)}
    for y in (11, 13, 17):
        g = gaps_cyclic(y)
        print("  m%-2d: %5d gaps, F = %2d, F_4 = %2d, %4d exact 4-tuples"
              % (y, len(g), int(g.max()), max(sum(t) for t in D[y]),
                 len(D[y])))
    assert not (D[11] - D[13]), "DEPTH-0 LEMMA VIOLATED at 11 -> 13"
    print("  depth-0 lemma D_4(11) subset D_4(13): OK")

    sup, _, _ = transfer(sorted(D[11]), 13, 26, 11, verbose=False)
    assert not (D[13] - sup), "superset violated"
    scr, _ = screen(sup, 13)
    assert not (D[13] - set(scr)), "screen unsound"
    onset = min(sum(t) for t in scr if t not in D[13])
    simple = min(sum(t) for t in D[17] - D[13])
    ref = [t for t in scr if sum(t) == onset and t not in D[13]]
    refined = min(sum(t) for t in scr if t in D[17] and t not in D[13])
    wit10 = min(D[17] - D[13], key=sum)

    print("\n  11 -> 13: superset %d (inflation %.3fx), screened %d"
          % (len(sup), len(sup) / len(D[13]), len(scr)))
    print("  ONSET                                    = %d" % onset)
    print("  SIMPLE law   min span D_4(17)\\D_4(13)     = %d   -> %s"
          % (simple, "HIT" if simple == onset else "MISS"))
    print("  REFINED law  min span of that set INTERSECTED with the "
          "transfer's emissions = %d   -> %s"
          % (refined, "HIT" if refined == onset else "MISS"))
    print("  the simple law's witness %s (span %d) is emitted by the "
          "11 -> 13 transfer: %s" % (wit10, simple, wit10 in set(scr)))
    print("  causal form: %d refuted at the onset span, all realised at m17: "
          "%s  (witness %s)"
          % (len(ref), all(t in D[17] for t in ref), sorted(ref)[0]))

    assert onset == 13 and simple == 10 and refined == 13
    assert wit10 not in set(scr), "the simple law's witness IS emitted"
    assert ref and all(t in D[17] for t in ref), "causal form fails"
    print("\nALL ASSERTIONS PASSED  (the SIMPLE form is refuted here; the "
          "REFINED and CAUSAL forms hold)")


if __name__ == "__main__":
    main()
