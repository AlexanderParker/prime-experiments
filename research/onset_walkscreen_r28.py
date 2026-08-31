"""Round 28 (mechanic): THE WALK-SCREENED DICTIONARY TRANSFER, and what it does
to the inflation onset.

WHAT ROUND 27 SAW.  At 37 -> 41 the screened arity-4 superset is EXACT below
span 68 and starts refuting at 68.  The question the brief asks is whether 68
is arithmetic.

WHAT THIS ROUND FOUND FIRST (research/onset_ladder_r28.py).  The UNSCREENED
onset is 9 at every one of seven steps, and it is explained exactly:

    X_5(M) := min span of a 5-walk whose two 4-windows are both realised at M
              but which is not itself realised at M
            = 9 AT EVERY MACHINE 13, 17, 19, 23, with the SAME witness
              (1, 2, 3, 2, 1).

and that witness is not an accident of the machine - it is PHASE-SATURATED:
its exposed set X = {0,1,3,6,8,9} has X mod 5 = {0,1,3,4} and
(X - s_5) mod 5 = {0,1,2,3} with s_5 = -2*6^{-1} = 3 mod 5, so gear 5 has NO
admissible phase and (1,2,3,2,1) is ZERO BY THEOREM at every machine (K9).

THE CONSEQUENCE, AND IT IS A TOOL IMPROVEMENT, NOT A REMARK.  The round-26
screen (C31) is applied to the EMITTED 4-tuple.  But the transfer's emission
comes from a WALK of M-openings, and every point of that walk - the deleted
interiors included - is an M-opening, so the WHOLE WALK must have an admissible
phase at every gear q <= M.  Screening the walk is
  * SOUND: a realised walk has an actual phase at every gear, so it survives;
  * STRICTLY STRONGER: it sees obstructions in the deleted interiors, which the
    emitted tuple has forgotten;
  * a PREFIX PRUNE: the bad-phase set only grows as points are added, so a
    saturated prefix can be cut in the DFS.

This script builds the walk-screened transfer, re-measures the onset at every
step where the exact target dictionary exists, and re-emits the m41 arity-4
superset for Constructor.

Usage:
  <venv>/python research/onset_walkscreen_r28.py ladder
  <venv>/python research/onset_walkscreen_r28.py m41
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "r28")
sys.path.insert(0, HERE)

from dict_transfer import load_dict, transfer, build_next   # noqa: E402
from onset_r28 import screen, onset_of                      # noqa: E402
from onset_ladder_r28 import gaps_cyclic, ktuples, primes_upto   # noqa: E402

F1 = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91}
F4 = {13: 26, 17: 33, 19: 38, 23: 58, 29: 70, 31: 90, 37: 105, 41: 118}
STEPS = [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31), (31, 37)]


def gear_data(M, cap=None):
    """[(q, s_q, full_mask)] for every gear q <= M, small gears first.

    `cap` drops gears above it.  THIS IS ONLY SOUND WITH AN ARGUMENT, and the
    argument is: a gear q can saturate only if the walk has >= q/2 exposed
    points (|FREE_q| >= q - 2|X|).  A walk emitting a 4-tuple has 5 surviving
    points plus its kills, and two kills differ by 0 or +-s mod q', so they are
    at least min(s, q'-s) apart; with the span cap that bounds the kill count,
    hence the point count, hence the largest gear that can ever fire.  Callers
    must pass a cap they have computed, not one they have guessed.
    """
    out = []
    for q in primes_upto(M):
        if q < 5 or (cap is not None and q > cap):
            continue
        s = (-2 * pow(6, -1, q)) % q
        out.append((q, s, (1 << q) - 1))
    return out


def ws_transfer(tuples, M, qp, f4cap, f1cap, out_m=4, gear_cap=None):
    """dict_transfer with the WALK-SCREEN prefix prune (K9 on the walk)."""
    Mo = len(tuples[0])
    nxt = build_next(tuples)
    up = pow(6, -1, qp)
    s = (2 * up) % qp
    gears = gear_data(M, gear_cap)
    res = set()
    stats = {"nodes": 0, "pruned": 0}

    def bad_masks(masks, d):
        """add exposed point d to every gear's bad-phase mask; None if any gear
        saturates."""
        new = []
        for i, (q, sq, full) in enumerate(gears):
            r = d % q
            m = masks[i] | (1 << r) | (1 << ((r - sq) % q))
            if m == full:
                return None
            new.append(m)
        return new

    def step(ctx, d, nsurv, cur, out, masks):
        for g in nxt.get(ctx, ()):
            d2 = d + g
            cur2 = cur + g
            if cur2 > f1cap or d2 > f4cap:
                continue
            stats["nodes"] += 1
            m2 = bad_masks(masks, d2)
            if m2 is None:
                stats["pruned"] += 1
                continue                     # walk is phase-saturated: ZERO
            c2 = (ctx + (g,))[-(Mo - 1):]
            r = d2 % qp
            if r == A or r == B:
                step(c2, d2, nsurv, cur2, out, m2)
            else:
                o2 = out + (cur2,)
                if nsurv + 1 == out_m:
                    res.add(o2)
                else:
                    step(c2, d2, nsurv + 1, 0, o2, m2)

    for A in range(qp):
        B = (A - s) % qp
        if A == 0 or B == 0:
            continue
        step((), 0, 0, 0, (), bad_masks([0] * len(gears), 0))
    return res, stats


def ladder():
    print("THE WALK-SCREENED TRANSFER vs THE ROUND-26 EMISSION SCREEN\n")
    d4 = {}
    for y in (13, 17, 19, 23):
        d4[y] = ktuples(gaps_cyclic(y), 4)
    for y in (29, 31, 37):
        d4[y] = set(load_dict(os.path.join(DATA, "gap_tuples_%d_4.csv" % y)))
    rows = []
    for M, qp in STEPS:
        src = sorted(d4[M])
        truth = d4[qp]
        sup, _, _ = transfer(src, qp, F4[qp], F1[qp], verbose=False)
        scr, _ = screen(sup, qp)
        ws, st = ws_transfer(src, M, qp, F4[qp], F1[qp])
        assert not (truth - ws), ("WALK-SCREEN REMOVED A REALISED TUPLE",
                                  M, qp, sorted(truth - ws)[:3])
        wss, _ = screen(sorted(ws), qp)          # both screens together
        assert not (truth - set(wss)), "COMBINED SCREEN UNSOUND"
        o_raw, _, _ = onset_of(sorted(sup), truth, "raw superset")
        o_scr, _, _ = onset_of(scr, truth, "emission-screened (r26/r27)")
        o_ws, _, _ = onset_of(sorted(ws), truth, "WALK-screened")
        o_b, tot, ref = onset_of(wss, truth, "walk + emission screened")
        print("  %d -> %d  truth %d | raw %d  emis %d  walk %d  both %d"
              "   (DFS pruned %d of %d nodes)"
              % (M, qp, len(truth), len(sup), len(scr), len(ws), len(wss),
                 st["pruned"], st["nodes"]))
        if o_b:
            print("      span:  " + " ".join("%4d" % x
                                             for x in range(o_b, o_b + 8)))
            print("      refut: " + " ".join("%4d" % ref.get(x, 0)
                                             for x in range(o_b, o_b + 8)))
        rows.append((M, qp, len(truth), len(sup), len(scr), len(ws), len(wss),
                     o_raw, o_scr, o_ws, o_b))
        print()

    print("\n  SUMMARY - SUPERSET SIZES AND ONSETS")
    print("    step     truth      raw    emis-scr  walk-scr  both  |  "
          "onset: raw  emis  walk  both")
    for (M, qp, t, r, e, w, b, o1, o2, o3, o4) in rows:
        print("    %2d->%2d %8d %8d %9d %9d %7d |        %3s %5s %5s %5s"
              % (M, qp, t, r, e, w, b, o1, o2, o3, o4))
    print("\n    inflation (superset / truth)")
    for (M, qp, t, r, e, w, b, *_) in rows:
        print("    %2d->%2d   raw %.3fx   emis %.3fx   walk %.3fx   both %.3fx"
              % (M, qp, r / t, e / t, w / t, b / t))


def m41():
    """Re-emit Constructor's arity-4 m41 superset with the walk screen."""
    src = load_dict(os.path.join(DATA, "gap_tuples_37_4.csv"))
    print("m41 arity-4 superset, walk-screened: source m37 exact %d 4-tuples"
          % len(src))
    # gear cap, computed not guessed: s = 2*6^-1 mod 41 = 14, so kills are >= 14
    # apart; span <= F_4(41) = 118 allows at most 9 kills, so a walk has at most
    # 5 + 9 = 14 exposed points and no gear above 2*14 = 28 can ever saturate.
    s41 = (2 * pow(6, -1, 41)) % 41
    mink = min(s41, 41 - s41)
    maxpts = 5 + 118 // mink
    print("  kill spacing >= %d, span <= 118 -> <= %d points -> gears <= %d "
          "can fire" % (mink, maxpts, 2 * maxpts))
    ws, st = ws_transfer(src, 37, 41, 118, 91, gear_cap=2 * maxpts)
    print("  walk-screened transfer: %d tuples (DFS pruned %d of %d nodes)"
          % (len(ws), st["pruned"], st["nodes"]))
    both, _ = screen(sorted(ws), 41)
    print("  + emission screen (C31): %d tuples" % len(both))
    prev = os.path.join(DATA, "r27", "gap_tuples_41_4_screened_spancap.csv")
    if os.path.exists(prev):
        old = set(load_dict(prev))
        print("  round-27 screened+spancap superset: %d" % len(old))
        assert set(both) <= old, "NOT A SUBSET OF THE ROUND-27 SUPERSET"
        print("  new superset is a SUBSET of it (asserted); removed %d more"
              % (len(old) - len(both)))
    shard = os.path.join(DATA, "r27", "gap_tuples_41_4_exact_le77.csv")
    if os.path.exists(shard):
        real = set(load_dict(shard))
        assert real <= set(both), "WALK SCREEN REMOVED A REALISED m41 TUPLE"
        print("  soundness: all %d exact m41 tuples of the r27 shard survive"
              % len(real))
        cand = [t for t in both if sum(t) <= 77]
        onset_of(cand, real, "m41 walk+emission screened, span<=77")
    p = os.path.join(OUT, "gap_tuples_41_4_walkscreened.csv")
    with open(p, "w") as f:
        f.write("g1,g2,g3,g4\n")
        for t in sorted(both):
            f.write(",".join(map(str, t)) + "\n")
    print("  wrote %s" % p)


if __name__ == "__main__":
    (ladder if sys.argv[1] == "ladder" else m41)()
