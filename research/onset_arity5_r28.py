"""Round 28 (mechanic): THE ONSET LAW AT OUTPUT ARITY 5.

Arities 2 and 3 came free by INDUCTION from the exact 4-tuple dictionaries.
Arity 5 does not - it needs exact 5-tuple dictionaries - but the machines whose
period is small enough (11, 13, 17, 19, 23) supply them in seconds, and that is
enough for three complete steps.

THE CONSTRUCT IS UNCHANGED: the SOURCE is still machine M's exact 4-tuple
dictionary, so the closure is still order 4; only the OUTPUT arity moves to 5.
That is the right variation - it asks whether the law is about the transfer, or
about the particular output size the chain happens to consume.

    step        truth needed     law's right-hand side needs
    11 -> 13      D_5(13)              D_5(17)
    13 -> 17      D_5(17)              D_5(19)
    17 -> 19      D_5(19)              D_5(23)

Usage:  <venv>/python research/onset_arity5_r28.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dict_transfer import transfer                     # noqa: E402
from onset_r28 import screen                           # noqa: E402
from onset_ladder_r28 import gaps_cyclic, ktuples      # noqa: E402

F1 = {13: 11, 17: 18, 19: 25, 23: 34}
F5 = {13: 28, 17: 35, 19: 47, 23: 65}          # C11, exact (arity 5 only)
# ROUND-28: arities 6 and 7 as well.  The span cap for output arity m must be
# >= F_m(target); it is taken from the target's OWN exact m-tuple dictionary,
# which is computed here, so it is exact rather than quoted - and the superset
# assertion below would catch it if it were not.
ARITY = int(sys.argv[1]) if len(sys.argv) > 1 else 5
STEPS = [(11, 13), (13, 17), (17, 19)]
NEXT = {13: 17, 17: 19, 19: 23}
ONSET4 = {(11, 13): 13, (13, 17): 15, (17, 19): 17}
ONSET3 = {(11, 13): 17, (13, 17): 14, (17, 19): 20}


def main():
    D4, D5 = {}, {}
    for y in (11, 13, 17, 19, 23):
        g = gaps_cyclic(y)
        D4[y] = ktuples(g, 4)
        D5[y] = ktuples(g, ARITY)
        if ARITY == 5 and y in F5:
            assert max(sum(t) for t in D5[y]) == F5[y], ("F_5 mismatch", y)
    print("ARITY 5.  source = the exact 4-tuple dictionary (order-4 closure "
          "unchanged); output arity 5\n")
    for y in sorted(D5):
        print("  m%-2d: %6d exact 4-tuples, %7d exact m-tuples, F_m = %s"
              % (y, len(D4[y]), len(D5[y]),
                 max(sum(t) for t in D5[y])))
    print("\n  step     onset_m   law: refined / simple    verdict   "
          "onset_4  onset_3")
    hits = shits = tested = skipped = 0
    for M, qp in STEPS:
        cap = max(sum(t) for t in D5[qp])
        sup, _, _ = transfer(sorted(D4[M]), qp, cap, F1[qp],
                             out_m=ARITY, verbose=False)
        assert not (D5[qp] - sup), ("SUPERSET VIOLATED", M, qp, ARITY)
        scr, _ = screen(sorted(sup), qp)
        assert not (D5[qp] - set(scr)), ("screen unsound", M, qp, ARITY)
        if D5[M] - D5[qp]:
            skipped += 1
            print("  %2d->%2d     the DEPTH-0 LEMMA FAILS at arity %d here "
                  "(q' = %d is not > 2(m+1) = %d) - step skipped, and this is "
                  "the sharpness table firing as an assertion"
                  % (M, qp, ARITY, qp, 2 * (ARITY + 1)))
            continue
        tested += 1
        o = min((sum(t) for t in scr if t not in D5[qp]), default=None)
        nxt = NEXT[qp]
        refined = min((sum(t) for t in scr
                       if t in D5[nxt] and t not in D5[qp]), default=None)
        simple = min((sum(t) for t in D5[nxt] - D5[qp]), default=None)
        hits += (refined == o)
        shits += (simple == o)
        print("  %2d->%2d     %4s        %4s / %-4s      %-9s%4d    %4d"
              % (M, qp, o, refined, simple,
                 "HIT" if refined == o else "MISS", ONSET4[(M, qp)],
                 ONSET3[(M, qp)]))
    print("\n  ARITY %d: refined law %d of %d testable; simple law %d of %d; "
          "%d step(s) skipped because the depth-0 lemma genuinely FAILS there"
          % (ARITY, hits, tested, shits, tested, skipped))
    print("  (the lemma needs q' > 2(m+1) = %d here; where it fails, that is "
          "C40's sharpness table firing as an assertion)" % (2 * (ARITY + 1)))


if __name__ == "__main__":
    main()
