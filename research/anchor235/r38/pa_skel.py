"""Branch W.a part 4: the six {5,7} skeletons of the path, and the 515 no-lengthening cells."""
import os
from math import isqrt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "pa_skel.txt"), "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")

say("=== the six {5,7} skeletons of the path (offsets, mod 35) ===")
say("gear 5 strikes offset i iff q^2 = -6i or 2-6i (mod 5); same for 7.")
say("class(q^2 mod 35) | 5's offsets mod 5 | 7's offsets mod 7 | open offsets of {5,7} in 1..35"
    " | first open | largest gap")
rows = []
for c in range(1, 35):
    from math import gcd
    if gcd(c, 35) != 1: continue
    if not any((r * r) % 35 == c for r in range(1, 35)): continue
    o5 = sorted({i % 5 for i in range(5) if (-6 * i) % 5 == c % 5 or (2 - 6 * i) % 5 == c % 5})
    o7 = sorted({i % 7 for i in range(7) if (-6 * i) % 7 == c % 7 or (2 - 6 * i) % 7 == c % 7})
    op = [i for i in range(1, 36) if i % 5 not in o5 and i % 7 not in o7]
    gaps = [op[t + 1] - op[t] for t in range(len(op) - 1)]
    say("      %2d          %-10s          %-12s       %-2d               %2d           %2d"
        % (c, o5, o7, len(op), op[0], max(gaps)))
    rows.append((c, op))
say("")
say("every class leaves 3/5 x 5/7 = 3/7 of the offsets open at {5,7}: 15 in every 35.")
say("what differs between classes is WHERE they sit:")
for c, op in rows:
    say("  q^2 = %2d (35): open offsets 1..35 = %s" % (c, op))
LOG.close()
