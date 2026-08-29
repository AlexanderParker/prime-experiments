"""Round 27 (mechanic) - HONEST PRICING of the exact m41 4-tuple census.

Brief item (a) says: price it first.  Two candidate vehicles:

  ROUTE A (CRT decision per candidate).  Take the phase-saturation-screened
  superset (2,814,574 tuples, C31) and decide every one with Constructor's
  scan-free set-cover CSP (crt_dict.decide_cover).  Cost = N_reverse_classes x
  mean decision time (rule 27: a tuple and its reverse have equal occurrence
  counts, so only one per reverse class needs deciding).

  ROUTE B (lap-phase transfer scan).  ghist_transfer's construction extended
  to emit 4-tuples: T = 29*31*37*41 = 1,363,783 laps of machine 23's period.
  Round 26 MEASURED 0.062 s/lap for the histogram alone => ~85,000 core-s, and
  tuple emission is strictly more work.  That route enumerates all
  prod(q-2) = 8.499e12 openings of machine 41's period; the cost is irreducible
  for a period vehicle.

This script measures ROUTE A's per-tuple cost STRATIFIED BY SPAN, because the
cover CSP's cost is driven by |Y| = span - 5 (the points that must be blocked),
and the superset's spans run from 6 to 145.  A span-restricted shard is the
natural self-contained deliverable if the full price exceeds the round.

Every claim here is a measurement on a stated sample, not an extrapolation of
a share (rule 1) - the projection is labelled as a projection.

Usage:  python research/price_m41_census_r27.py [wall_budget_s] [per_stratum]
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crt_dict import decide_cover, gears_of, Budget          # noqa: E402

SCREENED = os.path.join(HERE, "data", "r26", "gap_tuples_41_4_screened.csv")

# span strata: (lo, hi) inclusive
STRATA = [(1, 60), (61, 80), (81, 100), (101, 110), (111, 120),
          (121, 130), (131, 140), (141, 145)]


def load(path):
    out = []
    with open(path) as fh:
        head = fh.readline()
        assert head.strip() == "g1,g2,g3,g4", head
        for line in fh:
            out.append(tuple(int(x) for x in line.split(",")))
    return out


def make_XY(gaps):
    X = [0]
    for g in gaps:
        X.append(X[-1] + g)
    xs = set(X)
    Y = [t for t in range(1, X[-1]) if t not in xs]
    return X, Y


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    t0 = time.time()
    tuples = load(SCREENED)
    print("screened superset: %d tuples  (loaded in %.1f s)"
          % (len(tuples), time.time() - t0), flush=True)

    # -------- reverse classes, exactly (mirror law, rule 27) --------
    S = set(tuples)
    reps = []
    n_pal = 0
    for t in S:
        r = t[::-1]
        assert r in S, ("superset is not reverse-closed at", t)
        if t == r:
            n_pal += 1
        if t <= r:
            reps.append(t)
    print("reverse classes: %d  (palindromes %d) - decisions needed"
          % (len(reps), n_pal), flush=True)

    # -------- population per stratum, exact --------
    pop = {k: 0 for k in STRATA}
    bucket = {k: [] for k in STRATA}
    for t in reps:
        s = sum(t)
        for k in STRATA:
            if k[0] <= s <= k[1]:
                pop[k] += 1
                bucket[k].append(t)
                break
    print("\nreverse classes by span stratum (exact population):")
    for k in STRATA:
        print("   span %3d-%3d : %9d" % (k[0], k[1], pop[k]))
    print("   TOTAL        : %9d" % sum(pop.values()), flush=True)

    # -------- timed sample per stratum --------
    qs = gears_of(41)
    rng = random.Random(20260829)
    print("\nTIMED SAMPLE, machine 41 (%d gears), <= %d tuples per stratum, "
          "wall budget %.0f s\n" % (len(qs), per, budget), flush=True)
    print("  span-range      n   yes    no  unk   mean_s   med_s    max_s"
          "     projected core-s for the stratum")
    total_proj = 0.0
    done_all = True
    for k in STRATA:
        if not bucket[k]:
            continue
        samp = rng.sample(bucket[k], min(per, len(bucket[k])))
        times, yes, no, unk = [], 0, 0, 0
        for t in samp:
            if time.time() - t0 > budget:
                done_all = False
                break
            X, Y = make_XY(t)
            ta = time.perf_counter()
            try:
                ok, _, _ = decide_cover(qs, X, Y, node_budget=2_000_000)
                if ok:
                    yes += 1
                else:
                    no += 1
            except Budget:
                unk += 1
            times.append(time.perf_counter() - ta)
        if not times:
            print("  %3d-%3d      (no time left)" % k, flush=True)
            continue
        times.sort()
        m = len(times)
        mean = sum(times) / m
        proj = mean * pop[k]
        total_proj += proj
        print("  %3d-%3d %8d %5d %5d %4d %8.3f %7.3f %8.3f   %14.0f"
              % (k[0], k[1], m, yes, no, unk, mean, times[m // 2], times[-1],
                 proj), flush=True)

    print("\nPROJECTION, ROUTE A (per-stratum mean x exact stratum population)")
    print("  %.0f core-seconds = %.1f core-hours" % (total_proj,
                                                     total_proj / 3600.0))
    for w in (3, 4, 6):
        print("    at %d workers: %.1f wall hours" % (w, total_proj / w / 3600))
    if not done_all:
        print("  (some strata truncated by the wall budget - the projection "
              "uses only the strata that were sampled)")
    print("\nROUTE B (round-26 measurement, histogram only): ~85,000 core-s;")
    print("  4-tuple emission is strictly more work than the histogram.")


if __name__ == "__main__":
    main()
