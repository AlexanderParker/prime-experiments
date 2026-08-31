"""Round 28 (mechanic): the 37 -> 41 refuted-by-span table, extended to the new
frontier (80) and re-measured under the WALK screen.

Round 27 could only report this table to span 77, because that was the exact
shard's frontier.  The frontier is now 80 and the superset is tighter, so the
table gets three more rows and a second column - and the onset itself is a
control: it must still be 68.
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from dict_transfer import load_dict                    # noqa: E402

real = set(load_dict(os.path.join(DATA, "r28",
                                  "gap_tuples_41_4_exact_le80.csv")))
old = [t for t in load_dict(os.path.join(
    DATA, "r27", "gap_tuples_41_4_screened_spancap.csv")) if sum(t) <= 80]
new = [t for t in load_dict(os.path.join(
    DATA, "r28", "gap_tuples_41_4_walkscreened.csv")) if sum(t) <= 80]
assert set(new) <= set(old), "walk screen is not a subset of the r27 superset"
assert real <= set(new), "walk screen removed a realised tuple"


def table(c, label):
    tot, ref = Counter(), Counter()
    for t in c:
        tot[sum(t)] += 1
        if t not in real:
            ref[sum(t)] += 1
    onset = min(ref) if ref else None
    print("  %-26s candidates %7d  refuted %5d  ONSET %s"
          % (label, len(c), sum(ref.values()), onset))
    return tot, ref


t_old, r_old = table(old, "r27 emission-screened")
t_new, r_new = table(new, "r28 walk-screened")
print("\n  span  candidates(emis)  refuted(emis)  candidates(walk)  "
      "refuted(walk)")
for s in range(66, 81):
    print("  %4d %14d %14d %17d %14d"
          % (s, t_old.get(s, 0), r_old.get(s, 0),
             t_new.get(s, 0), r_new.get(s, 0)))
print("\n  realised tuples of span <= 80: %d" % len(real))
print("  inflation over the region: emission %.4fx, walk %.4fx"
      % (len(old) / len(real), len(new) / len(real)))
