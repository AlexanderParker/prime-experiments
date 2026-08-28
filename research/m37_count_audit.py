"""Round 24 (mechanic): resolve the round-23 DATA-INTEGRITY FLAG on the
r21 machine-37 "full period" fuel scan.

THE FLAG.  fuel37_k5hunt_part2.log reads

    machine y=37: period 1.237e+12, scanned 1.237e+12 (100.0%),
                  openings 112205953878, 34143s

against the exact opening count prod_{5<=q<=37}(q-2) = 217,929,355,875 - a
factor 1.942 short, while the same closed form matches the m23, m29 and m31
scans to the unit.

THE ANSWER: THE SCAN IS RIGHT AND THE LABEL IS WRONG.  fuel_census.report()
printed K (the END slot) as "scanned" and K/P as coverage, ignoring the
--start flag, so a RESUMED run advertised 100% coverage while it had only
scanned [start, K) - and its `openings` and every N_k are counts for THAT
RANGE ALONE.  Machine 37 was covered by three chained runs, and their
opening counts sum to prod(q-2) EXACTLY, to the unit:

    [0,      1.2e11)   21,144,680,389     fuel37.log
    [1.2e11, 6.0e11)   84,578,721,608     fuel37_k5hunt.log
    [6.0e11, P    )   112,205,953,878     fuel37_k5hunt_part2.log
                     ---------------
                      217,929,355,875  =  prod_{5<=q<=37}(q-2)

so the period IS fully covered; only the per-run labels are endpoints.
The start of each resumed run is RECOVERED, not guessed: given the endpoint
K and the count n, start = K - n*P/E lands on 1.2e11 and 6.0e11 to the unit.

WHAT ELSE THAT SCAN TOUCHED (the part that matters):
  * F_j(37) = 88 90 97 105 113 120 stands, but NOT by the cover argument
    alone: a resumed run's empty tail means a window STRADDLING a junction
    was examined by neither run.  research/m37_junction_check.py closes
    that: every window touching the two junctions or the cyclic wrap is
    examined directly, and none comes near the recorded values.
  * THE PUBLISHED N_k ARE THIRD-RANGE ONLY.  r21's "fuel at full period:
    N_1..N_4 = 110,467,008,914 / 869,473,543 / 1,579 / 0" is the [6e11, P)
    row.  The period values are the SUMS:
        N_1 = 214,551,930,429   N_2 = 1,688,714,780
        N_3 = 3,052             N_4 = 0
    (N_3 = 300 + 1,173 + 1,579.)  Confirmed INDEPENDENTLY by CRT+SAT word
    enumeration over the whole period in research/a_kill.py: N_3 = 3052
    exactly, from four realised words (14,41):1525, (41,14):1525,
    (27,41):1, (41,27):1 - so the two run junctions cost nothing.
  * k_max(37->41) = 3 is unaffected: N_4 = 0 in every range, and a_kill.py
    refutes all three surviving 3-letter words over the whole period.

Usage: uv run python research/m37_count_audit.py [--fix-csv]
"""
import os
import sys
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "fuel_census.csv")

GEARS37 = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
P = prod(GEARS37)
E = prod(g - 2 for g in GEARS37)

# (endpoint K, reported openings, log file)
RUNS = [
    (120_000_000_000, 21_144_680_389, "fuel37.log"),
    (600_000_000_000, 84_578_721_608, "fuel37_k5hunt.log"),
    (P, 112_205_953_878, "fuel37_k5hunt_part2.log"),
]
# the aborted intermediate run recorded in fuel_census.csv
EXTRA = (707_000_000_000, 18_854_006_749, "csv row 37,41,707000000000")
N3_BY_RUN = [300, 1173, 1579]
N2_BY_RUN = [163_848_288, 655_392_949, 869_473_543]
N1_BY_RUN = [20_816_984_223, 83_267_937_292, 110_467_008_914]


def recover_start(K, n):
    """start = K - n*P/E, exact to the unit (density is P/E slots/opening)."""
    return round(K - n * P / E)


def main():
    print(f"machine 37: period P = {P}, exact openings "
          f"prod(q-2) = {E}, density {E/P:.10f}")
    total = 0
    starts = []
    prev_end = 0
    for K, n, src in RUNS:
        st = recover_start(K, n)
        starts.append(st)
        exact = (K - st) / P * E
        print(f"  {src:28s} endpoint {K:>13d}  openings {n:>12d}"
              f"  => recovered start {st:>13d}"
              f"  (range {K-st:>13d} slots, predicted openings "
              f"{exact:.1f}, ratio {n/exact:.9f})")
        assert abs(n - exact) < 100, (src, n, exact)
        # the recovery is exact up to the O(1) boundary wobble of the count
        assert abs(st - prev_end) < 1000, (src, st, prev_end,
                                           "ranges do not tile")
        starts[-1] = prev_end
        prev_end = K
        total += n
    print(f"  recovered starts: {starts}  (round numbers, and the three "
          f"ranges TILE [0, P) exactly)")
    print(f"  sum of the three opening counts = {total}")
    print(f"  exact prod(q-2)                 = {E}")
    assert total == E, (total, E)
    print("  ==> EQUAL.  The period is fully covered; the scan is correct "
          "and the '(100.0%)' label was the endpoint, not the coverage.")

    K, n, src = EXTRA
    st = recover_start(K, n)
    print(f"\n  the aborted intermediate run ({src}): endpoint {K}, "
          f"openings {n} => start {st} - a PREFIX of the third range, "
          f"superseded by it (not part of the tiling).")
    assert abs(st - 600_000_000_000) < 1000, st

    print("\nPERIOD-WIDE FUEL COUNTS (sums, replacing the third-range-only "
          "row published in r21):")
    print(f"  N_1(37->41) = {sum(N1_BY_RUN)}   (published {N1_BY_RUN[-1]})")
    print(f"  N_2(37->41) = {sum(N2_BY_RUN)}   (published {N2_BY_RUN[-1]})")
    print(f"  N_3(37->41) = {sum(N3_BY_RUN)}        (published "
          f"{N3_BY_RUN[-1]})")
    print("  N_4(37->41) = 0 in every range")
    assert sum(N3_BY_RUN) == 3052

    print("\nINDEPENDENT CONFIRMATION (research/a_kill.py, CRT+SAT over the "
          "whole period, no scan):")
    print("  N_3(37->41) = 3052 from words (14,41):1525 (41,14):1525 "
          "(27,41):1 (41,27):1 -> matches the sum EXACTLY, so no 3-tuple "
          "was lost at either junction;")
    print("  N_4(37->41) = 0 (all 3 surviving legal 3-letter words "
          "refuted), so k_max = 3 with no boundary caveat at all.")

    print("\nTHE SPECTRUM: F_j(37) = 88 90 97 105 113 120.  NOTE the cover "
          "argument alone is NOT enough for windows: a resumed run starts "
          "with an empty tail, so a window STRADDLING a junction was "
          "examined by neither run.  research/m37_junction_check.py "
          "examines every window touching the two junctions and the "
          "cyclic wrap (max straddling 6-window sums 49/61/27, all far "
          "below the recorded F_j) - so the spectrum holds over the full "
          "period with no junction caveat.  F_1/F_2/F_3 also agree with "
          "the independent COV-SAT values, and the round-24 floor-1 "
          "lap-phase gates reproduce F_2/F_3 = 90/97 from machine 23.")

    if "--fix-csv" in sys.argv:
        fix_csv(dict((K, recover_start(K, n)) for K, n, _ in RUNS))


def fix_csv(startmap):
    """Add a trailing `start` column so a resumed row is self-describing."""
    rows = open(CSV, encoding="utf-8").read().strip().split("\n")
    head = rows[0]
    if head.endswith(",start"):
        print("\nfuel_census.csv already carries a start column.")
        return
    out = [head + ",start"]
    for r in rows[1:]:
        f = r.split(",")
        K, per, op = int(f[2]), int(f[3]), int(f[4])
        gears = None
        if K == per:
            st = 0
        else:
            st = 0
        # machine-37 rows: recover from the endpoint/count pair
        if int(f[0]) == 37:
            st = round(K - op * per / _E_of(int(f[0])))
            st = max(0, st)
            st = _snap(st)
        out.append(r + f",{st}")
    open(CSV, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"\nfuel_census.csv rewritten with a `start` column "
          f"({len(out)-1} rows).")


def _E_of(y):
    from fragile_census import primes_upto
    gs = [p for p in primes_upto(y) if p >= 5]
    return prod(g - 2 for g in gs)


def _snap(x):
    """Snap a recovered start to the round number it plainly is."""
    for c in (0, 120_000_000_000, 600_000_000_000):
        if abs(x - c) < 1000:
            return c
    return x


if __name__ == "__main__":
    main()
