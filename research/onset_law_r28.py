"""Round 28 (mechanic): THE ONSET RECURSION - is the span-68 onset arithmetic?

WHAT research/onset_anatomy_r28.py TURNED UP.  Two ladders came out of the same
table and they are ONE ladder, shifted by one step:

    step        13->17 17->19 19->23 23->29 29->31 31->37 37->41
    ONSET          15     17     25     31     41     53     68
    min span of a  10     15     17     25     31     41     53
      NEW candidate (one not already in D_4(M))

i.e.  ONSET(M -> q')  =  nu(q' -> q''),   nu(A -> B) := min span of a
transfer-superset candidate at step A -> B that is not already in D_4(A).

Below the onset the superset is EXACT, so a candidate of span < onset that is
not in D_4(A) is a REALISED new tuple; hence, whenever nu < onset,

    nu(A -> B)  =  min span of  D_4(B) \\ D_4(A)
                =  the smallest span at which adding gear B creates a 4-tuple
                   the machine A did not already have.

So the CONJECTURED LAW is

    ONSET(M -> q')  =  min span of  D_4(q'') \\ D_4(q'),   q'' the next prime,

read: "the transfer M -> q' first over-generates exactly where the NEXT
machine's new repertoire begins - the transfer emits, ahead of schedule, the
tuples that only appear one gear later."

THIS SCRIPT DOES THREE THINGS.
 (1) THE MECHANISM TEST.  If the law is causal, the tuples refuted AT the onset
     span must be realised at the NEXT machine.  Six steps have both exact
     dictionaries; check every one.
 (2) THE LAW, CHECKED DIRECTLY as min span of D_4(q'') \\ D_4(q').
 (3) THE OUT-OF-SAMPLE TEST.  onset(37->41) = 68 was measured in round 27 from
     an exact m41 shard.  The law says it equals nu(41 -> 43) - which is
     computable from the m41 SHARD alone (exact at every span <= 77, and every
     4-window of a walk of span <= 68 has span <= 68 < 77), with NO m43
     dictionary and NO solver.  That is a genuine prediction of a measured
     number by a route that never saw it.

Usage:  <venv>/python research/onset_law_r28.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

from dict_transfer import load_dict, transfer          # noqa: E402
from onset_r28 import screen                           # noqa: E402
from onset_ladder_r28 import gaps_cyclic, ktuples      # noqa: E402

F1 = {17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91, 43: 103}
F4 = {17: 33, 19: 38, 23: 58, 29: 70, 31: 90, 37: 105, 41: 118, 43: 145}
ONSET = {(13, 17): 15, (17, 19): 17, (19, 23): 25, (23, 29): 31,
         (29, 31): 41, (31, 37): 53, (37, 41): 68}
NEXT = {17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41, 41: 43}


def exact(y):
    if y in (13, 17, 19, 23):
        return ktuples(gaps_cyclic(y), 4)
    return set(load_dict(os.path.join(DATA, "gap_tuples_%d_4.csv" % y)))


def main():
    D = {y: exact(y) for y in (13, 17, 19, 23, 29, 31, 37)}
    shard = set(load_dict(os.path.join(DATA, "r27",
                                       "gap_tuples_41_4_exact_le77.csv")))

    print("(1) MECHANISM TEST - are the tuples refuted AT the onset span "
          "realised at the NEXT machine?\n")
    print("    step     onset  refuted there   of them realised at the next "
          "machine   verdict")
    for (M, qp), o in sorted(ONSET.items()):
        if qp == 41:
            continue                       # needs D_4(43): not available
        sup, _, _ = transfer(sorted(D[M]), qp, F4[qp], F1[qp], verbose=False)
        cand, _ = screen(sup, qp)
        ref = [t for t in cand if sum(t) == o and t not in D[qp]]
        nxt = NEXT[qp]
        Dn = shard if nxt == 41 else D[nxt]
        hit = [t for t in ref if t in Dn]
        print("    %2d->%2d    %4d        %4d              %4d  (machine %d)"
              "                %s" % (M, qp, o, len(ref), len(hit), nxt,
                                      "ALL" if len(hit) == len(ref) else
                                      "PARTIAL"))
        if ref:
            print("            witness %s" % (sorted(ref)[0],))

    print("\n(2) THE LAW, CHECKED DIRECTLY:  onset(M->q') =?= min span of "
          "D_4(q'') \\ D_4(q')\n")
    print("    step     onset   min span of D_4(next) \\ D_4(q')   verdict")
    for (M, qp), o in sorted(ONSET.items()):
        nxt = NEXT[qp]
        Dn = shard if nxt == 41 else D.get(nxt)
        if Dn is None:
            print("    %2d->%2d    %4d   (D_4(%d) not available)" % (M, qp, o,
                                                                    nxt))
            continue
        v = min((sum(t) for t in Dn - D[qp]), default=None)
        print("    %2d->%2d    %4d                        %4s        %s"
              % (M, qp, o, v, "HIT" if v == o else "MISS"))

    print("\n(3) OUT-OF-SAMPLE: predict onset(37->41) = 68 from the m41 SHARD "
          "alone\n")
    src = sorted(t for t in shard if sum(t) <= 77)
    print("    source: the round-27 exact m41 shard, %d 4-tuples of span <= 77"
          % len(src))
    sup, _, _ = transfer(src, 43, 90, F1[43], verbose=False)
    cand, _ = screen(sup, 43)
    new = [t for t in cand if t not in shard]
    v = min((sum(t) for t in new), default=None)
    print("    nu(41 -> 43) = min span of a screened 41->43 candidate not in "
          "D_4(41) = %s" % v)
    print("    (valid for spans <= 77: every 4-window of a walk of span s has "
          "span <= s)")
    print("    round-27 MEASURED onset(37 -> 41) = 68   ->  %s"
          % ("PREDICTED CORRECTLY" if v == 68 else
             "LAW FAILS OUT OF SAMPLE (predicted %s)" % v))
    if new:
        print("    witness %s (span %d)" % (min(new, key=sum), v))


if __name__ == "__main__":
    main()
