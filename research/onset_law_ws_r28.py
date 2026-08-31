"""Round 28 (mechanic): the onset law under the WALK screen.

The walk screen changes one onset (13 -> 17: 15 -> 17), so the law has to be
checked against the screen actually in use.  The law's right-hand side is
intersected with the transfer's EMISSIONS, and the walk screen changes what
counts as an emission - so this is not a re-run, it is the law's own variable
moving.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from dict_transfer import load_dict
from onset_walkscreen_r28 import ws_transfer
from onset_ladder_r28 import gaps_cyclic, ktuples

F1 = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
F4 = {13: 26, 17: 33, 19: 38, 23: 58, 29: 70, 31: 90, 37: 105}
STEPS = [(11, 13), (13, 17), (17, 19), (19, 23), (23, 29), (29, 31)]
NEXT = {13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37}
D = {}
for y in (11, 13, 17, 19, 23):
    D[y] = ktuples(gaps_cyclic(y), 4)
for y in (29, 31, 37):
    D[y] = set(load_dict(os.path.join(DATA, "gap_tuples_%d_4.csv" % y)))
print("  step    onset(walk screen)   refined law   verdict   onset(emission)")
EMIS = {(11, 13): 13, (13, 17): 15, (17, 19): 17, (19, 23): 25,
        (23, 29): 31, (29, 31): 41}
hits = 0
for M, qp in STEPS:
    ws, _ = ws_transfer(sorted(D[M]), M, qp, F4[qp], F1[qp])
    assert not (D[qp] - ws), ("walk screen unsound", M, qp)
    o = min(sum(t) for t in ws if t not in D[qp])
    nxt = NEXT[qp]
    law = min((sum(t) for t in ws if t in D[nxt] and t not in D[qp]),
              default=None)
    hits += (law == o)
    print("  %2d->%2d        %4d              %4s        %-8s     %4d"
          % (M, qp, o, law, "HIT" if law == o else "MISS", EMIS[(M, qp)]))
print("\n  refined law under the WALK screen: %d of %d" % (hits, len(STEPS)))
