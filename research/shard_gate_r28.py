"""Round 28 (mechanic): gate on the extended m41 exact 4-tuple shard.

The shard is the round's one artefact produced by a SOLVER rather than by a
theorem or a scan, so it gets its own two-sided check:
  * reverse-closed (the mirror halving is only sound if the emitted set is);
  * max span exactly 80 (the claimed frontier, not one more or less);
  * agrees with the round-27 span<=77 shard CELL FOR CELL below 77;
  * sits inside the certified walk-screened superset (a decision can only ever
    REMOVE from a superset);
  * contains every m37 4-tuple of span <= 80 - the depth-0 lemma, checked
    against the artefact rather than assumed by it.

usage: <venv>/python research/shard_gate_r28.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from dict_transfer import load_dict                    # noqa: E402

new = set(load_dict(os.path.join(DATA, "r28",
                                 "gap_tuples_41_4_exact_le80.csv")))
old = set(load_dict(os.path.join(DATA, "r27",
                                 "gap_tuples_41_4_exact_le77.csv")))
sup = set(load_dict(os.path.join(DATA, "r28",
                                 "gap_tuples_41_4_walkscreened.csv")))
m37 = set(load_dict(os.path.join(DATA, "gap_tuples_37_4.csv")))

assert all(t[::-1] in new for t in new), "NOT REVERSE-CLOSED"
assert max(sum(t) for t in new) == 80, "max span is not 80"
assert {t for t in new if sum(t) <= 77} == old, "disagrees with the r27 shard"
assert new <= sup, "not inside the certified walk-screened superset"
assert {t for t in m37 if sum(t) <= 80} <= new, "DEPTH-0 LEMMA VIOLATED"

print("m41 EXACT 4-TUPLE SHARD, span <= 80: %d tuples" % len(new))
print("  reverse-closed; max span exactly 80; agrees with the round-27")
print("  span<=77 shard cell for cell (%d tuples); inside the walk-screened" % len(old))
print("  superset (%d); contains every m37 4-tuple of span <= 80 (%d)"
      % (len(sup), len({t for t in m37 if sum(t) <= 80})))
print("ALL ASSERTIONS PASSED")
