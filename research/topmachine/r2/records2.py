"""Section 1 (continued): the waste in a record cover, and the corrected sub-threshold
reduction (F_top depends only on m and on the gears that are <= F + 1).

usage: uv run python research/topmachine/r2/records2.py results/records2.json
"""

import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r1"))
from cover import F_cover, minpieces  # noqa: E402

P = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
     101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
     193, 197, 199, 211, 223, 227, 229, 233, 239, 241]


def waste_large_gear(m):
    """All gears large: F = 2m - (m mod 2), the cover is r = m distance-2 dominoes."""
    L = 2 * m - (m % 2)
    r = minpieces((1 << L) - 1, L)
    return {"m": m, "L": L, "pieces_needed": r, "coverage": 2 * r, "waste": 2 * r - L}


def main():
    out = {}

    # waste in the record cover
    ws = [waste_large_gear(m) for m in range(2, 13)]
    out["record_cover_waste"] = ws
    for x in ws:
        print("m=%2d  L=%2d  pieces=%2d  coverage=%2d  waste=%d" %
              (x["m"], x["L"], x["pieces_needed"], x["coverage"], x["waste"]))
    print("waste = 0 for even m, 1 for odd m:",
          all((x["waste"] == 0) == (x["m"] % 2 == 0) for x in ws))
    out["waste_rule_holds"] = all((x["waste"] == 0) == (x["m"] % 2 == 0) for x in ws)

    # corrected sub-threshold reduction:
    #   F(G) should depend only on m and on S = {g in G : g <= F(G) + 1}.
    # Test: for each G, compute F, read off S; rebuild with the same S and DIFFERENT large
    # gears (all > F + 1), recompute; require the same F and the same S.
    tests = []
    cases = []
    for m in range(3, 9):
        for r in range(0, min(5, m) + 1):
            for S in combinations(P[:6], r):  # candidate small parts drawn from 7..23
                bigs_pools = [P[6:], P[10:], P[20:], P[30:]]
                fs = []
                Ss = []
                for pool in bigs_pools:
                    G = sorted(list(S) + [p for p in pool if p not in S][: m - r])
                    if len(G) != m:
                        fs.append(None)
                        Ss.append(None)
                        continue
                    f, st = F_cover(G)
                    if st != "exact":
                        fs.append(None)
                        Ss.append(None)
                        continue
                    fs.append(f)
                    Ss.append(tuple(g for g in G if g <= f + 1))
                good = [i for i in range(len(fs)) if fs[i] is not None]
                if len(good) < 2:
                    continue
                # only compare the variants whose small part really is S
                keep = [i for i in good if Ss[i] == tuple(S)]
                if len(keep) < 2:
                    cases.append({"m": m, "S": list(S), "F": [fs[i] for i in good],
                                  "small_parts": [list(Ss[i]) for i in good],
                                  "comparable": False})
                    continue
                ok = len(set(fs[i] for i in keep)) == 1
                tests.append(ok)
                cases.append({"m": m, "S": list(S), "F": [fs[i] for i in keep],
                              "small_parts": [list(Ss[i]) for i in keep],
                              "comparable": True, "ok": ok})
                if not ok:
                    print("  EXCEPTION m=%d S=%s F=%s" % (m, S, [fs[i] for i in keep]))
    out["reduction_cases"] = len(tests)
    out["reduction_exceptions"] = sum(1 for t in tests if not t)
    out["reduction_detail"] = cases
    print("corrected sub-threshold reduction: %d comparable cases, %d exceptions"
          % (len(tests), out["reduction_exceptions"]))

    # the rule as a table: F(m, small part)
    rule = {}
    for c in cases:
        if c.get("comparable") and c.get("ok"):
            rule["m=%d small=%s" % (c["m"], c["S"])] = c["F"][0]
    out["rule"] = rule
    for k in sorted(rule, key=lambda s: (int(s.split("=")[1].split()[0]), s)):
        print("  %-30s F_top = %d  (parity law %s)"
              % (k, rule[k], 2 * int(k.split("=")[1].split()[0]) - (int(k.split("=")[1].split()[0]) % 2)))

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
