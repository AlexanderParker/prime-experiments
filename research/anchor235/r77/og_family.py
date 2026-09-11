"""og_family.py -- the same census on the sections the counter-machines DO kill (P4).

(a) The tooth family's exhibited killers (first_realisation.md 3.5) at p = 17, 29, 37, 41, 43, 47, 53:
    per column, the family strikers; each strike is REAL (the gear divides a member) or PHANTOM (it does
    not); a column is phantom-only if all its family strikes are phantom. Report: phantom strikes, phantom-
    only columns, the twin columns of the section and which family gears kill them, and whether the
    killing gears lie above the section's length (tail gears).
(b) V17's machine at the same cuts: gears {5..p} plus the twin members of the section; per column the
    strikers; the columns struck only by added gears (quotient 1).
(c) For the real engine: the census invariant "every struck member has an integer quotient >= 5".
Output: results/family.json, results/family.log
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from og_common import (family_strikes_column, finer_section, is_prime, real_teeth, strikers_of_column)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

KILLERS = {
    17: [1, 1, 2, 6, 1],
    29: [1, 1, 1, 2, 1, 2, 4, 2],
    37: [2, 1, 2, 2, 4, 3, 3, 9, 13, 5],
    41: [1, 2, 1, 4, 6, 3, 6, 7, 8, 1, 1],
    43: [1, 2, 4, 3, 7, 5, 3, 5, 9, 12, 12, 16],
    47: [1, 3, 4, 5, 6, 3, 2, 7, 2, 17, 8, 20, 3],
    53: [2, 1, 4, 5, 7, 4, 2, 8, 2, 7, 4, 10, 22, 14],
}


def main():
    out = []
    log = []
    for p, teeth in KILLERS.items():
        a, b, cols, gears = finer_section(p)
        L = len(cols)
        rt = real_teeth(gears)
        moved = [g for g, v, w in zip(gears, teeth, rt) if v != w]
        twins = [j for j in cols if not strikers_of_column(j, gears)]
        phantom_strikes = 0
        real_strikes = 0
        phantom_only = []
        family_killed = True
        twin_killers = {}
        struck_primes = 0
        for j in cols:
            fam = family_strikes_column(j, gears, teeth)
            if not fam:
                family_killed = False
            real = {g for g, s in strikers_of_column(j, gears)}
            ph = [g for g in fam if g not in real]
            phantom_strikes += len(ph)
            real_strikes += len([g for g in fam if g in real])
            if fam and not (set(fam) & real):
                phantom_only.append(j)
            if j in twins:
                twin_killers[j] = fam
            # a phantom strike on a column whose members are both prime = a struck prime
            if ph and is_prime(6 * j - 1) and is_prime(6 * j + 1):
                struck_primes += 1
        # V17: added gears = the twin members
        added = []
        for j in twins:
            added += [6 * j - 1, 6 * j + 1]
        v17_only_added = [j for j in cols if not strikers_of_column(j, gears) and
                          any((6 * j - 1) % g == 0 or (6 * j + 1) % g == 0 for g in added)]
        # real census invariant
        bad = 0
        for j in cols:
            for g, s in strikers_of_column(j, gears):
                n = 6 * j + s
                if n % g != 0 or n // g < 5:
                    bad += 1
        r = dict(p=p, section=(cols[0], cols[-1]), length=L, gears=gears, real_teeth=rt, killer=teeth,
                 moved_gears=moved, n_moved=len(moved), moved_above_length=[g for g in moved if g > L],
                 family_kills=family_killed, twins=twins, twin_killers={str(k): v for k, v in twin_killers.items()},
                 twin_killers_above_length={str(k): [g for g in v if g > L] for k, v in twin_killers.items()},
                 phantom_strikes=phantom_strikes, real_strikes=real_strikes, phantom_only_cols=phantom_only,
                 n_phantom_only=len(phantom_only), n_twins=len(twins), struck_prime_columns=struck_primes,
                 v17_added_gears=added, v17_cols_struck_only_by_added=v17_only_added,
                 real_invariant_violations=bad)
        out.append(r)
        line = (f"p={p:<3} cols {cols[0]}..{cols[-1]} (L={L}) killer moves {len(moved)} gears {moved} "
                f"(above L: {r['moved_above_length']}); kills={family_killed}; twins={twins}; "
                f"twin killers={twin_killers}; phantom strikes={phantom_strikes} real={real_strikes}; "
                f"phantom-only cols={phantom_only} ({len(phantom_only)} vs {len(twins)} twins); "
                f"V17 added={added} only-added cols={v17_only_added}; real invariant violations={bad}")
        print(line)
        log.append(line)
    with open(os.path.join(RES, "family.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RES, "family.log"), "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
