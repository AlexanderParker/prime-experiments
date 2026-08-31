"""Round 28 (mechanic): the onset law at output arity 5, on a BIG step.

The arity-5 tests in research/onset_arity5_r28.py run at 11->13, 13->17,
17->19 - the only steps whose exact 5-tuple dictionaries were cheap.  This
round's streamed passes add D_5(29) and D_5(31), which unlocks the step

    23 -> 29    truth = D_5(29),   law's right-hand side = D_5(31)

- a machine two rungs above anything the arity-5 test had reached, and the
first arity-5 test whose dictionaries came from full-period scans rather than
from a 37,000-slot period.

The construct is unchanged: SOURCE is machine 23's exact 4-tuple dictionary, so
the closure is still order 4; only the output arity is 5.

usage: <venv>/python research/onset_arity5_big_r28.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

from dict_transfer import load_dict, transfer          # noqa: E402
from onset_r28 import screen                           # noqa: E402
from onset_ladder_r28 import gaps_cyclic, ktuples      # noqa: E402


def load5(y):
    p = os.path.join(DATA, "r28", "gap_tuples_%d_5.csv" % y)
    rows = open(p).read().strip().split("\n")[1:]
    return {tuple(int(x) for x in r.split(",")) for r in rows}


def main():
    D4_23 = ktuples(gaps_cyclic(23), 4)
    D5_23 = ktuples(gaps_cyclic(23), 5)
    D5_29 = load5(29)
    print("  m23: %d exact 4-tuples, %d exact 5-tuples" % (len(D4_23),
                                                           len(D5_23)))
    print("  m29: %d exact 5-tuples (streamed full-period pass, this round)"
          % len(D5_29))
    assert not (D5_23 - D5_29), "DEPTH-0 LEMMA (m=5) VIOLATED at 23 -> 29"
    print("  depth-0 lemma at arity 5, 23 -> 29: HOLDS (q' = 29 > 2(m+1) = 12)")

    F5_29 = max(sum(t) for t in D5_29)
    print("  F_5(29) = %d (from the dictionary; C11 says 85)" % F5_29)
    sup, _, _ = transfer(sorted(D4_23), 29, F5_29, 43, out_m=5, verbose=False)
    assert not (D5_29 - sup), "SUPERSET VIOLATED at arity 5, 23 -> 29"
    scr, _ = screen(sorted(sup), 29)
    assert not (D5_29 - set(scr)), "screen unsound"
    onset = min(sum(t) for t in scr if t not in D5_29)
    print("\n  superset %d, screened %d,  ONSET(23 -> 29, arity 5) = %d"
          % (len(sup), len(scr), onset))

    p31 = os.path.join(DATA, "r28", "gap_tuples_31_5.csv")
    if not os.path.exists(p31):
        print("\n  D_5(31) not on disk yet - the law's right-hand side needs "
              "it; rerun when the\n  machine-31 streamed pass has written it.")
        return
    D5_31 = load5(31)
    assert not (D5_29 - D5_31), "DEPTH-0 LEMMA (m=5) VIOLATED at 29 -> 31"
    refined = min((sum(t) for t in scr
                   if t in D5_31 and t not in D5_29), default=None)
    simple = min((sum(t) for t in D5_31 - D5_29), default=None)
    ref = [t for t in scr if sum(t) == onset and t not in D5_29]
    print("  m31: %d exact 5-tuples" % len(D5_31))
    print("  REFINED law = %s -> %s;   SIMPLE law = %s -> %s"
          % (refined, "HIT" if refined == onset else "MISS",
             simple, "HIT" if simple == onset else "MISS"))
    print("  causal form: %d tuples refuted at the onset span, %d of them "
          "realised at m31 -> %s"
          % (len(ref), sum(1 for t in ref if t in D5_31),
             "ALL" if all(t in D5_31 for t in ref) else "PARTIAL"))
    print("  witness %s" % (sorted(ref)[0],))


if __name__ == "__main__":
    main()
