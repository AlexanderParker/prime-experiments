"""rr_mech.py -- the mechanism at the record's two ends, from the enumerated configurations.

Reads results/rec_<top>_<F>.json written by rr_record.py and reports, per machine:
  * m(F), n1(F), N(F), the multiset of neighbour pairs (L, R);
  * whether every gear is a sole striker inside the record at every occurrence;
  * the sole-striker counts per gear (the tiling of the record);
  * the strikers of the first column outside each end (who buys the end), and how often gear 5
    is among them;
  * the mirror structure of the occurrences.
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LADDER = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}


def main():
    print("| M | F | m(F) | n1(F) | N(F) | neighbour pairs (L,R) | all gears busy | "
          "sole strikers per gear | interior cols | sole share |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    tot_end = 0
    end5 = 0
    endcounts = Counter()
    for top, F in LADDER.items():
        p = os.path.join(OUT, f"rec_{top}_{F}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        gears = [int(g) for g in d["gears"]]
        rows = d["rows"]
        pairs = Counter((r["L"], r["R"]) for r in rows)
        sole = {g: len(rows[0]["sole_strikers"][str(g)]) for g in gears}
        nsole = sum(sole.values())
        print(f"| {{5..{top}}} | {F} | {d['m_v']} | {d['n1']} | {d['N']} | {dict(pairs)} | "
              f"{d['all_gears_busy']} | {sole} | {F - 1} | {nsole}/{F - 1} |")
        for r in rows:
            for side in ("strikers_left", "strikers_right"):
                key = "L" if side == "strikers_left" else "R"
                if r[key] > 1:
                    tot_end += 1
                    endcounts[len(r[side])] += 1
                    if 5 in [int(x) for x in r[side]]:
                        end5 += 1
    print(f"\nfirst-outside-column census over every record occurrence and both ends: "
          f"{tot_end} columns; striker-count histogram {dict(sorted(endcounts.items()))}; "
          f"gear 5 among the strikers at {end5} of {tot_end}")


if __name__ == "__main__":
    main()
