"""Per-gear fields on a section, each starting at the gear's square.

Gear field of g = the members n of S in the section whose LEAST prime factor is g
(n = g * m with m >= g a survivor of the gears below g): the gear's own strikes from its
square on, with echoes (multiples struck earlier by a smaller gear) excluded. The square
field = {g^2} is the first member of every gear field. The gear fields partition the
composites of S (each composite has one least factor), so they are the fields of the owner's
second kind: one per gear, the square first.

Per gear: size, the square present?, class split, longest run of S free of the gear field,
the number of columns where this gear field is the only striker (exclusive kills), and the
composition of its members by second factor (m prime / m composite).
Usage: uv run python gear_fields.py lo hi [maxgears]
"""
import sys
from sympy import factorint, primerange


def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); maxg = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    S = [n for n in range(lo, hi) if n % 6 in (1, 5)]
    lpf, om = {}, {}
    for n in S:
        f = factorint(n); lpf[n] = min(f); om[n] = sum(f.values())
    cols = [k for k in range((lo + 1) // 6 + 1, (hi - 2) // 6 + 1) if 6 * k - 1 >= lo and 6 * k + 1 < hi]
    gears = [g for g in primerange(5, int(hi ** 0.5) + 1)]
    squares = [g * g for g in gears if lo <= g * g < hi]
    print(f"section [{lo}, {hi}): |S| = {len(S)}, columns {len(cols)}, gears {gears[0]}..{gears[-1]} ({len(gears)}); square field: {len(squares)} members {squares[:6]}{'...' if len(squares) > 6 else ''}, all right members, exclusive kills = "
          + str(sum(1 for q in squares if q - 2 >= lo and om[q - 2] == 1)))
    print("gear | field size | share of composites | square in section | left/right | m prime | longest free run of S | exclusive kills")
    comps = sum(1 for n in S if om[n] >= 2)
    for g in gears[:maxg]:
        F = [n for n in S if lpf[n] == g]
        if not F: print(f"{g} | 0"); continue
        Fs = set(F)
        left = sum(1 for n in F if n % 6 == 5)
        mprime = sum(1 for n in F if om[n] == 2)
        best = cnt = 0
        for n in S:
            if n in Fs: cnt = 0
            else: cnt += 1; best = max(best, cnt)
        excl = 0
        for k in cols:
            a, b = 6 * k - 1, 6 * k + 1
            here = lpf[a] == g or lpf[b] == g
            other = (om[a] >= 2 and lpf[a] != g) or (om[b] >= 2 and lpf[b] != g)
            if here and not other: excl += 1
        print(f"{g} | {len(F)} | {100*len(F)/comps:.1f}% | {'yes' if lo <= g*g < hi else 'no'} | {left}/{len(F)-left} | {mprime} | {best} | {excl}")
    # the rest, summed
    rest = [n for n in S if om[n] >= 2 and lpf[n] > gears[maxg - 1]]
    print(f"gears above {gears[maxg-1]}: {len(rest)} members ({100*len(rest)/comps:.1f}% of composites)")


if __name__ == "__main__":
    main()
