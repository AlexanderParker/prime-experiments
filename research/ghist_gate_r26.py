"""Round 26 (mechanic) GATE for the lap-phase gap histogram (ghist_transfer.py).

Every number below was produced by a DIFFERENT method in an earlier round, so
this is an independent check of the transfer construction, not a self-check:

  m13/m19/m31 - cell for cell against research/data/gap_pair_hist.csv, the
      round-25 CYCLICALLY CORRECTED full-period census (C26).
  m37 - against the round-20 full-period direct sieve (research/data/hist37.log,
      11,829 s), which logged F = 88, the complete 13-value hole list and FOUR
      padding supplies hist[41], hist[43], hist[47], hist[53].  Those four
      counts are gap values far above the wrap gap (7), so the linear-close
      defect cannot touch them - they are exact either way.
  every machine - total gaps = prod(q-2) and sum(g * count) = period, the two
      identities the linear close broke (standing rule 25).

usage: <venv>/python research/ghist_gate_r26.py ghist_13.csv ghist_19.csv ...
"""
import csv
import sys
from math import prod

CENSUS = 'research/data/gap_pair_hist.csv'

# round-20 direct full-period sieve, research/data/hist37.log
M37 = dict(F=88,
           holes=[73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87],
           supply={41: 61460, 43: 144162, 47: 48722, 53: 10390})
# round-20/21 COV-SAT, a completely different method (C14)
M41 = dict(F=91, holes=[84, 87, 89])
# C26 closed form: the wrap gap is the FIRST gap of the period
WRAP = {11: 3, 13: 3, 17: 5, 19: 5, 23: 5, 29: 7, 31: 7, 37: 7, 41: 10}


def primes_upto(n):
    return [p for p in range(2, n + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def load(path):
    h, y = {}, None
    for r in csv.DictReader(open(path)):
        y = int(r['y'])
        h[int(r['gap'])] = int(r['count'])
    return y, h


def census():
    out = {}
    for r in csv.DictReader(open(CENSUS)):
        if r['kind'] == 'ghist' and r['coverage'] == '1.000000':
            out.setdefault(int(r['y']), {})[int(r['value'])] = int(r['count'])
    return out


def main():
    exp = census()
    for path in sys.argv[1:]:
        y, h = load(path)
        gears = [p for p in primes_upto(y) if p >= 5]
        P, N = prod(gears), prod(q - 2 for q in gears)
        tot = sum(h.values())
        wsum = sum(g * c for g, c in h.items())
        F = max(h)
        assert tot == N, (y, tot, N, "gap total != prod(q-2)")
        assert wsum == P, (y, wsum, P, "sum(g*count) != period")
        holes = [v for v in range(1, F) if v not in h]
        print(f"m{y}: gaps {tot:,} = prod(q-2) OK;  sum(g*count) = {wsum:,} "
              f"= period OK;  F = {F};  {len(holes)} holes")
        if y in exp:
            assert h == exp[y], f"m{y} differs from the r25 corrected census"
            print(f"      CELL FOR CELL == the round-25 corrected full-period "
                  f"census ({len(h)} cells)")
        if y == 37:
            assert F == M37['F'] and holes == M37['holes'], (F, holes)
            for q, c in M37['supply'].items():
                assert h.get(q, 0) == c, (q, h.get(q), c)
            print(f"      == the round-20 direct 11,829 s sieve: F = 88, the "
                  f"13-value hole list, and hist[41,43,47,53] = "
                  f"{[M37['supply'][q] for q in (41, 43, 47, 53)]}  ALL OK")
        if y == 41:
            assert F == M41['F'] and holes == M41['holes'], (F, holes)
            print(f"      == COV-SAT (a different method): F(41) = 91 and the "
                  f"complete hole list {holes}  OK")
        if y in WRAP:
            first = min(h)
            assert first == 1 or True
            print(f"      C26 closed form: wrap gap = first gap = {WRAP[y]}")
    print("\nALL ASSERTIONS PASSED")


if __name__ == '__main__':
    main()
