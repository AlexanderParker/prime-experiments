"""Round 28 (mechanic): IS THE INFLATION ONSET ARITHMETIC?

ROUND-27 FINDING (C33).  At the 37 -> 41 arity-4 dictionary transfer, every
reverse class of span <= 67 is REALISED and the first refutation anywhere is at
span 68; the refuted count then climbs sharply.  So the order-4 closure of the
dictionary transfer is EXACT below 68 and decays over the next ~30 units.

THE QUESTION.  Is 68 predictable from the machine's constants?  If it is, a
future superset can be pre-trimmed by THEOREM: "every candidate of span < onset
is realised" turns a decision problem into a lookup below the onset, and
"the transfer's exactness ends at X" is a statement about the closure, not
about a particular scan.

THE TEST, and it costs no solver at all.  At the three steps 23->29, 29->31,
31->37 BOTH exact 4-tuple dictionaries exist on disk (full-period scans, C21).
So the whole onset curve can be computed exactly:

    superset  = dict_transfer(exact M-dictionary) with the target's F_1/F_4 caps
    screened  = phase-saturation screen at the target machine  (C31/K9)
    refuted   = screened \\ exact target dictionary
    onset     = min span over `refuted`

and, as a soundness gate, exact-target MINUS superset must be EMPTY at every
step (the transfer is a superset or the tool is wrong).

Usage:  <venv>/python research/onset_r28.py
"""
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

from dict_transfer import load_dict, transfer          # noqa: E402

# exact values (C11 / C15).  F_4 exact at 29, 31, 37; F_1 exact everywhere.
F1 = {29: 43, 31: 58, 37: 88, 41: 91}
F4 = {29: 70, 31: 90, 37: 105, 41: 118}                # F_4(41) = 118 (C32)
STEPS = [(23, 29), (29, 31), (31, 37)]


def primes_upto(n):
    return [p for p in range(2, n + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def screen(tuples, Y):
    """Phase-saturation screen (K9) at machine Y; returns the survivors."""
    gears = [p for p in primes_upto(Y) if p >= 5]
    out = []
    killed = Counter()
    for t in tuples:
        X = [0]
        for g in t:
            X.append(X[-1] + g)
        dead_by = None
        for q in gears:
            if q >= 2 * len(X):
                break
            s = (-2 * pow(6, -1, q)) % q
            bad = set()
            for x in X:
                bad.add(x % q)
                bad.add((x - s) % q)
            if len(bad) == q:
                dead_by = q
                break
        if dead_by is None:
            out.append(t)
        else:
            killed[dead_by] += 1
    return out, killed


def onset_of(cands, truth, label):
    """cands: iterable of tuples (a superset).  truth: set of realised tuples."""
    by_span_tot = Counter()
    by_span_ref = Counter()
    for t in cands:
        sp = sum(t)
        by_span_tot[sp] += 1
        if t not in truth:
            by_span_ref[sp] += 1
    onset = min(by_span_ref) if by_span_ref else None
    print("    %-22s candidates %8d  refuted %7d  ONSET %s"
          % (label, sum(by_span_tot.values()), sum(by_span_ref.values()),
             onset if onset is not None else "NONE (exact everywhere)"))
    return onset, by_span_tot, by_span_ref


def main():
    print("THE INFLATION ONSET, EXACTLY, AT EVERY STEP WHERE BOTH EXACT "
          "4-TUPLE DICTIONARIES EXIST\n")
    results = {}
    for M, qp in STEPS:
        src = load_dict(os.path.join(DATA, "gap_tuples_%d_4.csv" % M))
        truth = set(load_dict(os.path.join(DATA, "gap_tuples_%d_4.csv" % qp)))
        print("  %d -> %d : source %d, exact target %d" %
              (M, qp, len(src), len(truth)))
        sup, _, _ = transfer(src, qp, F4[qp], F1[qp], verbose=False)
        missing = truth - sup
        assert not missing, ("SUPERSET VIOLATED", M, qp, sorted(missing)[:5])
        print("    soundness: exact target \\ superset = EMPTY  (%d candidates,"
              " inflation %.4fx)" % (len(sup), len(sup) / len(truth)))
        scr, killed = screen(sup, qp)
        assert not (truth - set(scr)), "SCREEN REMOVED A REALISED TUPLE"
        print("    screen killed %d (%s)"
              % (len(sup) - len(scr),
                 ", ".join("gear %d: %d" % kv for kv in sorted(killed.items()))
                 or "none"))
        o_raw, _, _ = onset_of(sorted(sup), truth, "unscreened superset")
        o_scr, tot, ref = onset_of(scr, truth, "screened superset")
        results[(M, qp)] = (o_raw, o_scr, tot, ref, len(truth))
        lo = (o_scr or 0)
        print("      span:  " + " ".join("%4d" % s for s in range(lo, lo + 12)))
        print("      refut: " + " ".join("%4d" % ref.get(s, 0)
                                         for s in range(lo, lo + 12)))
        print("      cands: " + " ".join("%4d" % tot.get(s, 0)
                                         for s in range(lo, lo + 12)))
        print()

    # --- the 37 -> 41 row, from the round-27 exact shard (span <= 77 only) ---
    shard = os.path.join(DATA, "r27", "gap_tuples_41_4_exact_le77.csv")
    scap = os.path.join(DATA, "r27", "gap_tuples_41_4_screened_spancap.csv")
    if os.path.exists(shard) and os.path.exists(scap):
        real = set(load_dict(shard))
        cand = [t for t in load_dict(scap) if sum(t) <= 77]
        print("  37 -> 41 : from the round-27 exact shard (complete to span 77)")
        o, tot, ref = onset_of(cand, real, "screened superset")
        results[(37, 41)] = (None, o, tot, ref, len(real))
        print("      span:  " + " ".join("%4d" % s for s in range(o, o + 10)))
        print("      refut: " + " ".join("%4d" % ref.get(s, 0)
                                         for s in range(o, o + 10)))
        print("      cands: " + " ".join("%4d" % tot.get(s, 0)
                                         for s in range(o, o + 10)))
        print()

    print("\n  THE ONSET LADDER")
    print("    step        onset  F(M)  F_2(M)  F(q')  F_2(q')  onset/F(M)  "
          "onset/F(q')")
    F2 = {13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90, 41: 103}
    F1a = dict(F1); F1a.update({13: 11, 17: 18, 19: 25, 23: 34})
    for (M, qp), (o_raw, o_scr, _, _, _) in sorted(results.items()):
        o = o_scr
        print("    %2d -> %2d      %4s  %4d  %6d  %5d  %7d      %.3f       %.3f"
              % (M, qp, o, F1a[M], F2[M], F1a[qp], F2[qp],
                 o / F1a[M], o / F1a[qp]))

    print("\n  PRE-REGISTERED CANDIDATE FORMULAS (prereg_mechanic_r28.md D1-D4)")
    prevp = {29: 23, 31: 29, 37: 31, 41: 37}
    prevprev = {29: 19, 31: 23, 37: 29, 41: 31}
    for (M, qp), (_, o, _, _, _) in sorted(results.items()):
        d1 = F2[prevp[qp]]                      # F_2 of the machine below M
        d2 = 2 * F1a[prevprev[qp]]              # 2 F of two machines below
        d3 = round(0.773 * F1a[M])              # constant ratio to F(M)
        print("    %2d -> %2d  measured %3d | D1 %3d %-4s D2 %3d %-4s "
              "D3 %3d %-4s" % (M, qp, o,
                               d1, "HIT" if d1 == o else "miss",
                               d2, "HIT" if d2 == o else "miss",
                               d3, "HIT" if d3 == o else "miss"))


if __name__ == "__main__":
    main()
