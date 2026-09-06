"""R7: which gear sets can the wrong boundary hurt?

Under the wrong boundary (core = {g <= L}) the ONLY gear whose classification changes at length
L is g = L + 1, so a difference in the record requires F + 1 to be a gear.  This counts how many
of the 6,659 scannable sets have F + 1 in the gear set, to compare with the 605 that actually
differ.

usage: uv run python research/topmachine/r7/check.py
"""

import itertools
import json
import os
from math import prod

from rule import SCAN_CAP, F_rule, pairwise_coprime

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
POOL = [5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49]


def main():
    sets = []
    for m in range(2, 7):
        for s in itertools.combinations(POOL, m):
            if prod(s) <= SCAN_CAP and pairwise_coprime(s):
                sets.append(s)
    has = 0
    fin = 0
    for s in sets:
        f = F_rule(list(s))
        if f + 1 in s:
            has += 1
        if f in s:
            fin += 1
    out = {"sets": len(sets), "F_plus_1_is_a_gear": has, "F_is_a_gear": fin}
    print(out, flush=True)
    json.dump(out, open(os.path.join(RES, "check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
