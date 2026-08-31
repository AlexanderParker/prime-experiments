"""Round 28: the depth-0 lemma at OTHER arities (m = 2, 3, 5), where the
threshold q' > 2(m+1) bites differently.  Exact 5-tuple dictionaries exist
in-round for machines 13, 17, 19, 23 (period scans)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from onset_ladder_r28 import gaps_cyclic, ktuples
D = {}
for y in (13, 17, 19, 23):
    g = gaps_cyclic(y)
    D[y] = {m: ktuples(g, m) for m in (2, 3, 4, 5, 6)}
print("  m  threshold q'>2(m+1)   13->17   17->19   19->23")
for m in (2, 3, 4, 5, 6):
    row = []
    for (a, b) in ((13, 17), (17, 19), (19, 23)):
        miss = D[a][m] - D[b][m]
        row.append("OK(%d)" % len(D[a][m]) if not miss
                   else "FAIL %d %s" % (len(miss), sorted(miss)[:2]))
    print("  %d        %2d            %s" % (m, 2 * (m + 1),
                                             "   ".join("%-12s" % r for r in row)))
