"""Branch 2f.i - detail on the compatible violators found by cp_compat.py viol."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_compat import (PROOF, admissible, best_core, coherent_member, gears_of,  # noqa: E402
                       incompat, next_prime, real_tooth, seps)

B = 30
for y in (11, 13, 17, 19):
    gears = gears_of(y)
    q1 = next_prime(y)
    gs = gears + [q1]
    rats = admissible(gs, B)
    with open(os.path.join(PROOF, "chain_teeth_r33_fam_m%d.json" % y)) as f:
        rows = json.load(f)
    print("=== m%d + q'=%d : gears %s, %d admissible rationals ===" % (y, q1, gs, len(rats)))
    hits = 0
    for row in rows:
        if not (row["viol"] or not row["pair_ok"]):
            continue
        teeth = list(row["teeth"]) + [row["v1"]]
        k, rc, core = best_core(gs, teeth, rats)
        if k < len(gs):
            continue
        hits += 1
        # every rational realising the full core
        allrc = []
        s = seps(gs, teeth)
        for (r, c) in rats:
            if all((r * si) % q in (c % q, (-c) % q) for q, si in zip(gs, s)):
                allrc.append((r, c))
        print(" FULLY COMPATIBLE VIOLATOR  teeth(old)=%s v'=%s a=%s  seps=%s" %
              (row["teeth"], row["v1"], row["a"], s))
        print("   rationals c/r: %s ; F=%d F2=%d chain=%d budget=%d viol=%s pair_ok=%s L=%d"
              % (["%d/%d" % (c, r) for (r, c) in allrc], row["F"], row["F2"], row["chain"],
                 row["F"] + q1, row["viol"], row["pair_ok"], row["L"]))
        print("   argmax %s" % row["argmax"])
        print("   real teeth %s" % [real_tooth(q) for q in gs])
        # is it exactly the coherent member of that rational?
        for (r, c) in allrc:
            cm = coherent_member(gs, r, c)
            print("   coherent member of %d/%d = %s  (equal: %s)" % (c, r, cm, cm == teeth))
    print(" fully compatible violators at m%d: %d" % (y, hits))
    del rows
