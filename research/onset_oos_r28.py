"""Round 28 (mechanic): the OUT-OF-SAMPLE test of the onset law.

The law (research/onset_law_r28.py, 6/6 in sample):

    onset(M -> q')  =  min span of  D_4(q'') \\ D_4(q'),   q'' the next prime.

Round 27 MEASURED onset(37 -> 41) = 68 from an exact machine-41 shard.  The law
says that number equals nu(41 -> 43) - computable from the m41 shard ALONE, with
no m43 dictionary, no scan and no solver.  A walk of span s has every 4-window
of span <= s, so a source dictionary that is exact at every span <= 77 emits
correctly at every span <= 77; capping the transfer at 75 keeps the whole
computation inside the shard's exact region AND inside memory (the first
attempt, capped at 90, exhausted RAM).

Usage:  <venv>/python research/onset_oos_r28.py [SPAN_CAP]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

from dict_transfer import load_dict, transfer          # noqa: E402
from onset_r28 import screen                           # noqa: E402

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 75


def main():
    shard = set(load_dict(os.path.join(DATA, "r27",
                                       "gap_tuples_41_4_exact_le77.csv")))
    src = sorted(t for t in shard if sum(t) <= CAP)
    print("source: round-27 exact m41 shard, %d 4-tuples of span <= %d "
          "(shard is exact to 77)" % (len(src), CAP))
    sup, _, _ = transfer(src, 43, CAP, CAP, verbose=False)
    print("  41 -> 43 transfer, span cap %d: %d candidates" % (CAP, len(sup)))
    new = [t for t in sup if t not in shard]
    print("  candidates NOT in D_4(41): %d" % len(new))
    scr, _ = screen(new, 43)
    v = min((sum(t) for t in scr), default=None)
    print("  nu(41 -> 43) = min span of a screened NEW candidate = %s" % v)
    print("\n  round-27 MEASURED onset(37 -> 41) = 68  ->  %s"
          % ("PREDICTED CORRECTLY, OUT OF SAMPLE" if v == 68 else
             "LAW FAILS OUT OF SAMPLE (law predicts onset = %s)" % v))
    if scr:
        w = min(scr, key=sum)
        print("  witness %s (span %d)" % (w, sum(w)))


if __name__ == "__main__":
    main()
