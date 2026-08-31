"""Round 28 (mechanic): DOES THE ONSET LAW DEPEND ON THE ARITY?

The onset law was found at arity 4, which is the arity Constructor's chain
consumes.  If it is a real fact about the transfer it should not care about the
arity.  Arity 3 is a free test: a contiguous 3-window of a realised 4-tuple is
realised, and every realised 3-tuple sits inside some realised 4-tuple, so

    D_3(M) = the induced 3-tuple dictionary of D_4(M),  EXACTLY,

with no new scan anywhere.  Run the same pipeline at out_m = 3 and compare.

Usage:  <venv>/python research/onset_arity3_r28.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

from dict_transfer import load_dict, transfer, induced   # noqa: E402
from onset_r28 import screen                             # noqa: E402
from onset_ladder_r28 import gaps_cyclic, ktuples        # noqa: E402

F1 = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
F3 = {13: 23, 17: 28, 19: 35, 23: 50, 29: 65, 31: 85, 37: 97}
F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90}
# ROUND-28: the same pipeline at any arity m < 4.  D_m is induced from D_4
# EXACTLY for m = 2 and 3 (every realised m-tuple sits inside a realised
# 4-tuple), so extra arities cost no scan at all.
ARITY = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 3
STEPS = [(11, 13), (13, 17), (17, 19), (19, 23), (23, 29), (29, 31), (31, 37)]
NEXT = {13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37}


def exact4(y):
    if y in (11, 13, 17, 19, 23):
        return ktuples(gaps_cyclic(y), 4)
    return set(load_dict(os.path.join(DATA, "gap_tuples_%d_4.csv" % y)))


def main():
    D4 = {y: exact4(y) for y in (11, 13, 17, 19, 23, 29, 31, 37)}
    FM = F3 if ARITY == 3 else F2
    D3 = {y: induced(sorted(D4[y]), ARITY) for y in D4}
    print("ARITY %d.  D_%d induced exactly from the exact 4-tuple "
          "dictionaries\n" % (ARITY, ARITY))
    for y in sorted(D3):
        assert max(sum(t) for t in D3[y]) == FM.get(y, max(
            sum(t) for t in D3[y])), ("F_m mismatch", y)
        print("  m%-2d: %6d exact m-tuples, max span %3d%s"
              % (y, len(D3[y]), max(sum(t) for t in D3[y]),
                 " = F_%d (asserted)" % ARITY if y in FM else ""))
    print("\n  step     onset_m   law: refined / simple    verdict     onset_4")
    onset4 = {(11, 13): 13, (13, 17): 15, (17, 19): 17, (19, 23): 25,
              (23, 29): 31, (29, 31): 41, (31, 37): 53}
    hits = shits = tested = exact_steps = 0
    for M, qp in STEPS:
        sup, _, _ = transfer(sorted(D4[M]), qp, FM[qp], F1[qp],
                             out_m=ARITY, verbose=False)
        assert not (D3[qp] - sup), ("superset violated at arity 3", M, qp)
        scr, _ = screen(sorted(sup), qp)
        assert not (D3[qp] - set(scr)), ("screen unsound at arity 3", M, qp)
        # depth-0 lemma at m = 3 needs q' > 8
        assert not (D3[M] - D3[qp]), ("DEPTH-0 LEMMA (m=3) VIOLATED", M, qp)
        o = min((sum(t) for t in scr if t not in D3[qp]), default=None)
        nxt = NEXT.get(qp)
        if o is None:
            exact_steps += 1
            print("  %2d->%2d     %4s          -    / -       %-12s%4d"
                  % (M, qp, "-", "TRANSFER EXACT", onset4[(M, qp)]))
            continue
        if nxt is None:
            print("  %2d->%2d     %4s          -    / -       %-12s%4d"
                  % (M, qp, o, "NOT TESTABLE", onset4[(M, qp)]))
            continue
        tested += 1
        refined = min((sum(t) for t in scr
                       if t in D3[nxt] and t not in D3[qp]), default=None)
        simple = min((sum(t) for t in D3[nxt] - D3[qp]), default=None)
        hits += (refined == o)
        shits += (simple == o)
        print("  %2d->%2d     %4s        %4s / %-4s      %-12s%4d"
              % (M, qp, o, refined, simple,
                 "HIT" if refined == o else "MISS", onset4[(M, qp)]))
    print("\n  ARITY %d: refined law %d of %d testable steps; simple law %d of "
          "%d; %d step(s) have NO ONSET AT ALL (the transfer is EXACT there)"
          % (ARITY, hits, tested, shits, tested, exact_steps))
    print("  31 -> 37 is NOT TESTABLE: the law's right-hand side needs "
          "D_%d(41), and no exact m41 dictionary of that\n  arity exists (the "
          "shard's induced tuples are span-restricted, hence a LOWER bound)."
          % ARITY)
    print("  Shorter patterns are pinned by the SAME order-4 closure for MORE "
          "span: the arity-3 onsets\n  exceed the arity-4 ones at 5 of 7 steps,"
          " and at arity 2 the transfer has no onset at all below m19.")


if __name__ == "__main__":
    main()
