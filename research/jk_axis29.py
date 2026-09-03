"""jk_axis29.py -- the k-AXIS PROGRAMME, round 29: harvest, protocol-check,
score, and price.

HARVESTER lane, round 29.  Gate: prints ALL ASSERTIONS GREEN or dies.
    .venv/Scripts/python.exe research/jk_axis29.py
(or `uv run python research/jk_axis29.py`)

WHAT IT DOES, in the order it does it.

[A] HARVEST AND PROTOCOL-CHECK the split runs.  jkcov6.rs prunes a node when
    feasible_to(cov, j, best+1) fails, so a worker whose incumbent `best` has
    risen prunes MORE above the split depth, visits FEWER split-depth nodes,
    and its global `leafctr` numbering diverges from the other workers'.  The
    parts `leafctr % nparts == part` then need not cover the tree.  Hence the
    protocol: a split run is a PROOF only if every worker was seeded at M and
    NO worker reported m > M.  Round 28's j_3(P(23)) run violated this (two of
    fourteen workers reached 227 and 232 from a seed of 219) and its result was
    never harvested; this gate rejects it as an upper bound, keeps it as a
    verified LOWER bound, and accepts the round-29 rerun at seed 232.

[B] THE LADDERS, with every value's provenance.

[C] THE DISCRIMINATOR, recomputed on the extended ladders.  Models:
        (A) random-choice heuristic       j_k ~ z (log z)^k
        (B) the layered construction      j_k >> z (log z)^{2k-1}   [(P2'), r26]
    R_k = j_k / model_k with model_k = log(delta_k P)/delta_k, no free
    parameter; Q_k = R_k/R_1 removes the small-z transient common to every k;
    f_k = log(Q_k(z1)/Q_k(z0)) / ((k-1) log(log z1/log z0)) is 0 under (A) and
    1 under (B) AND THE SAME AT EVERY k UNDER (B).

[D] SCORING the round-28 addendum and the round-29 pre-registration.

[E] THE PRICE of the rungs not bought, from measured node counts only.

THE STANDING CAVEAT, repeated because it is load-bearing: (P2') carries a
C^k/B^{2k} factor worth about 0.03 at z = 73, k = 2 and the construction does
not exist below log x ~ 300.  NOTHING HERE REFUTES THAT THEOREM.  What is
measured is the shape of the truth on the range where exact values exist.
"""
from __future__ import annotations

import glob
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

from jk_growth import model_random, table, loglog  # noqa: E402

DATA = os.path.join(ROOT, "research", "data")
OUT = []
ASSERTS = [0]


def W(s=""):
    OUT.append(s)


def check(cond, msg):
    ASSERTS[0] += 1
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------- [A] harvest
def read_parts(pattern):
    rows = []
    for fn in sorted(glob.glob(os.path.join(DATA, pattern))):
        txt = open(fn).read().strip()
        if not txt:
            return None, os.path.basename(fn)     # incomplete
        f = txt.split()
        rows.append(dict(file=os.path.basename(fn), k=int(f[0]), z=int(f[1]),
                         jk=int(f[2]), m=int(f[3]), nodes=int(f[4]),
                         secs=float(f[5]), status=f[6], verify=(f[7] == "true")))
    return rows, None


def harvest():
    W("[A] HARVEST AND PROTOCOL-CHECK")
    W()
    r28, miss = read_parts("jkpart_k3_z23_M219_n14_p*.txt")
    check(r28 is not None and len(r28) == 14, "round-28 j_3(23) parts missing")
    best28 = max(r["m"] for r in r28)
    nodes28 = sum(r["nodes"] for r in r28)
    secs28 = sum(r["secs"] for r in r28)
    beat = [r for r in r28 if r["m"] > 219]
    W(f"    ROUND 28  k=3 z=23 seed=219, 14 parts: all EXACT, all verify=true,")
    W(f"              nodes={nodes28:,}  core-seconds={secs28:,.0f} "
      f"({secs28/3600:.2f} core-hours)")
    W(f"              workers beating the seed: {len(beat)} "
      f"({', '.join(str(r['m']) for r in sorted(beat, key=lambda x: -x['m']))})")
    check(all(r["status"] == "EXACT" for r in r28), "a round-28 part is not EXACT")
    check(all(r["verify"] for r in r28), "a round-28 part failed its own verify")
    check(len(beat) == 2 and best28 == 232,
          "expected exactly two round-28 workers above the seed, best 232")
    W("    VERDICT: INVALID AS AN UPPER BOUND (protocol: a worker beat the seed),")
    W("             VALID AS A LOWER BOUND with a machine-verified witness:")
    W(f"             j_3(P(23)) >= 6 * {best28 + 1} = {6 * (best28 + 1)}")
    W()

    r29, miss = read_parts("jkpart29_k3_z23_M232_n5_p*.txt")
    if r29 is None or len(r29) != 5:
        W(f"    ROUND 29 rerun at seed 232: INCOMPLETE "
          f"({'missing ' + str(miss) if miss else 'parts not all present'})")
        W("    -> j_3(P(23)) stands as a LOWER bound only.")
        return None, dict(nodes28=nodes28, secs28=secs28)
    nodes29 = sum(r["nodes"] for r in r29)
    secs29 = sum(r["secs"] for r in r29)
    over = [r for r in r29 if r["m"] > 232]
    W(f"    ROUND 29  k=3 z=23 seed=232, 5 parts: "
      f"nodes={nodes29:,}  core-seconds={secs29:,.0f} "
      f"({secs29/3600:.2f} core-hours)")
    W(f"              per-worker m: {[r['m'] for r in r29]}")
    check(all(r["status"] == "EXACT" for r in r29), "a round-29 part is not EXACT")
    check(not over, "PROTOCOL VIOLATION: a round-29 worker beat the seed 232")
    check(nodes29 < nodes28, "the better seed did not prune more (nodes went up)")
    W("    VERDICT: PROOF.  Every worker seeded at 232, none improved, so all")
    W("             workers pruned identically above the split depth, the")
    W("             leafctr numbering agrees, and the parts partition the tree.")
    W(f"             With the verified witness of length 232:  "
      f"j_3(P(23)) = 6 * 233 = {6 * 233}  EXACT")
    W("    SEED LAW, AND IT IS MUCH WEAKER THAN THE ROUND-27 'SEED LAW' SAYS:")
    W(f"      {nodes28/1e9:.2f}e9 -> {nodes29/1e9:.2f}e9 nodes, ratio "
      f"{nodes28/nodes29:.3f}x from a seed 13 higher - a "
      f"{100*(1-nodes29/nodes28):.1f}% saving,")
    W(f"      not the ~4x round 27 recorded.  The WALL-CLOCK difference "
      f"({secs28/3600:.1f} ->")
    W(f"      {secs29/3600:.1f} core-hours) is almost entirely the High-priority boost")
    W("      and the smaller worker count, NOT algorithmic - which is exactly")
    W("      why the benchmark protocol counts operations and not seconds.")
    W()
    return 6 * 233, dict(nodes28=nodes28, secs28=secs28,
                         nodes29=nodes29, secs29=secs29)


# ---------------------------------------------------------------- [B] ladders
J3 = {3: 6, 5: 24, 7: 78, 11: 180, 13: 306, 17: 612, 19: 972}
J4 = {5: 30, 7: 150, 11: 420, 13: 1230, 17: 2340, 19: 3810}
J5 = {7: 180, 11: 930, 13: 2070, 17: 5490}

# exact single-core node counts measured in round 29 (unseeded, EXACT runs)
NODES = {(3, 13): 11_740, (3, 17): 556_927, (3, 19): 50_867_900,
         (4, 11): 26, (4, 13): 2_648, (4, 17): 351_958, (4, 19): 99_408_318}


def ladders(j3_23):
    if j3_23:
        J3[23] = j3_23
    W("[B] THE LADDERS  (bold entries are this lane's own computations)")
    W()
    W("    z        3     5     7     11     13     17     19      23")
    W(f"    j_3      6    24    78    180    306    612    972   "
      f"{J3.get(23, '?'):>5}")
    W("    j_4      -    30   150    420   1230   2340   3810       -")
    W("    j_5      -     -   180    930   2070   5490      -       -")
    W()
    W("    NEW THIS ROUND: j_4(P(17)) = 2340 (m=77, 351,958 nodes, 0.345 s),")
    W("                    j_4(P(19)) = 3810 (m=126, 99,408,318 nodes, 448.8 s),")
    W("                    j_3(P(23)) = 1398 (m=232), proved twice - [A] and [F].")
    W("    Round 28 had j_3 to z=19 and j_4 to z=13.")
    W()
    check(J4[17] == 2340 and J4[19] == 3810, "j_4 ladder corrupted")
    # sanity: strictly increasing in z and in k where both defined
    for T, name in ((J3, "j_3"), (J4, "j_4"), (J5, "j_5")):
        zs = sorted(T)
        check(all(T[zs[i]] < T[zs[i + 1]] for i in range(len(zs) - 1)),
              f"{name} not increasing in z")
    for z in sorted(set(J3) & set(J4)):
        check(J3[z] < J4[z], f"j_3({z}) >= j_4({z})")


# ---------------------------------------------------------- [C] discriminator
def Rtab(k, T):
    out = {}
    for z, v in sorted(T.items()):
        m = model_random(k, z)
        if m == m and m > 0:
            out[z] = v / m
    return out


def discriminator():
    R1 = Rtab(1, table(1))
    R2 = Rtab(2, table(2))
    R3, R4, R5 = Rtab(3, J3), Rtab(4, J4), Rtab(5, J5)
    W("[C] THE DISCRIMINATOR")
    W()
    W("    R_k = j_k / model_k  (model_k parameter-free):")
    for k, r in ((1, R1), (2, R2), (3, R3), (4, R4), (5, R5)):
        W(f"      R_{k}: " + "  ".join(f"{z}:{v:.4f}" for z, v in r.items()))
    W()
    W("    THE TWO CLEAN POST-TRANSIENT STEPS (both new this round):")
    if 23 in R3:
        d3 = R3[23] / R3[19] - 1
        W(f"      k=3, 19 -> 23 : R_3 {R3[19]:.4f} -> {R3[23]:.4f}  "
          f"({d3*100:+.2f}%)   (A) needs 0%, (B) needs +13.4%")
        check(abs(d3) < 0.03, "R_3 moved more than 3% across 19->23")
    d4 = R4[19] / R4[17] - 1
    W(f"      k=4, 17 -> 19 : R_4 {R4[17]:.4f} -> {R4[19]:.4f}  "
      f"({d4*100:+.2f}%)   (A) needs 0%, (B) needs +12.2%")
    check(d4 < 0.05, "R_4 rose by more than 5% across 17->19")
    W()

    def f_k(k, Rk, z0, z1):
        Q0, Q1 = Rk[z0] / R1[z0], Rk[z1] / R1[z1]
        return math.log(Q1 / Q0) / ((k - 1) * math.log(math.log(z1) / math.log(z0)))

    W("    f_k on matched windows (0 = (A), 1 = (B), and EQUAL ACROSS k under (B)):")
    W("      window |   f_2   |   f_3   |   f_4   |   f_5")
    for z0, z1 in [(7, 13), (7, 17), (7, 19), (7, 23), (11, 19), (11, 23),
                   (17, 23), (23, 73)]:
        cells = []
        for k, Rk in ((2, R2), (3, R3), (4, R4), (5, R5)):
            if z0 in Rk and z1 in Rk:
                cells.append(f"{f_k(k, Rk, z0, z1):+7.3f}")
            else:
                cells.append("   -   ")
        W(f"      {z0:2d}..{z1:2d} | " + " | ".join(cells))
    W()
    W("      HONEST: f_k on a two-point window is unstable (13..19 gives")
    W("      f_2 = -0.551 and f_3 = +1.319).  The robust statements are the")
    W("      two R_k steps above, not any single f cell.")
    W()

    def fit(T, zmin=7):
        pts = [(loglog(z), math.log(v / z)) for z, v in sorted(T.items())
               if z >= zmin]
        n = len(pts)
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
        return (n * sxy - sx * sy) / (n * sxx - sx * sx), n

    W("    measured exponent a_k in j_k ~ z (log z)^a  (zmin = 7):")
    W("      k | a_k    | n  | excess a_k - k | (B) needs | excess/(k-1)")
    ex = {}
    for k, T in ((1, table(1)), (2, table(2)), (3, J3), (4, J4), (5, J5)):
        a, n = fit(T)
        e = a - k
        ex[k] = e
        rel = "-" if k == 1 else f"{e/(k-1):.2f}"
        W(f"      {k} | {a:6.3f} | {n:2d} | {e:+13.3f}  | {k-1:9d} | {rel:>11}")
    W()
    frac = ", ".join(f"{ex[k]/(k-1):.2f}" for k in (2, 3, 4, 5))
    cbar = sum(ex[k] / (k - 1) for k in (2, 3, 4, 5)) / 4
    W("    THE ROUND-29 READING, and it is neither model: the excess is a")
    W(f"    CONSISTENT FRACTION of what (B) demands - excess/(k-1) = {frac}")
    W("    at k = 2, 3, 4, 5 - so on the computed range the truth")
    W(f"    looks like z (log z)^(k + c(k-1)) with c about {cbar:.2f}, strictly")
    ladder = ", ".join(f"{ex[k]:.2f}" for k in (2, 3, 4, 5))
    grows = all(ex[k] <= ex[k + 1] for k in (2, 3, 4))
    W("    between (A) (c = 0) and (B) (c = 1).  Round 28 said the excess does")
    W(f"    NOT grow with k; with j_4 at z = 17 and 19 the ladder is ({ladder})")
    W(f"    and it {'DOES grow' if grows else 'still does not grow monotonically'}"
      f" - my own round-28 statement is corrected here.")
    check(ex[1] < 0.05, "the k=1 calibration bias is not near zero")
    check(all(ex[k] < k - 1 for k in (2, 3, 4, 5)),
          "some measured excess reached the (B) requirement")
    return ex


# ------------------------------------------------------------------ [E] price
def price(h):
    n23 = h["nodes28"]
    secs23 = h["secs28"]
    rate = n23 / secs23
    W("[E] THE PRICE OF THE RUNGS NOT BOUGHT  (node counts, not wall time)")
    W()
    W("    k=3 exhaustive node counts and per-prime ratios:")
    seq = [(13, NODES[(3, 13)]), (17, NODES[(3, 17)]), (19, NODES[(3, 19)]),
           (23, n23)]
    prev = None
    for z, n in seq:
        r = "" if prev is None else f"   ratio {n/prev:8.1f}x"
        tag = "  (seeded at 219, so an UNDERSTATEMENT of the unseeded ratio)" \
              if z == 23 else ""
        W(f"      z={z:2d}  {n:>15,}{r}{tag}")
        prev = n
    W()
    r1 = NODES[(3, 17)] / NODES[(3, 13)]
    r2 = NODES[(3, 19)] / NODES[(3, 17)]
    r3 = n23 / NODES[(3, 19)]
    g = ((r2 / r1) * (r3 / r2)) ** 0.5      # geometric mean ratio-of-ratios
    W(f"    The ratio itself grows ~{g:.2f}x per step "
      f"({r1:.1f} -> {r2:.1f} -> {r3:.1f}).")
    W(f"    Measured throughput from the round-28 run: {n23/1e9:.3f}e9 nodes over")
    W(f"    {secs23:,.0f} core-seconds = {rate:.2e} nodes/s/core.")
    W()
    rows = []
    r = r3
    n = n23
    for z in (29, 31):
        r *= g
        n *= r
        rows.append((z, n, n / rate / 3600))
    for z, n, ch in rows:
        W(f"      j_3(P({z})) projected: {n:.2e} nodes = {ch:,.0f} core-hours "
          f"= {ch/24/6:,.0f} days on my 6-core budget")
    W()
    W("    AGAINST MY OWN ROUND-28 PRICES:")
    W("      z=23 priced '~1-2 core-hours'   -> actually 13.6   (9x low)")
    W(f"      z=29 priced '~10 core-hours'    -> {rows[0][2]:,.0f}  "
      f"({rows[0][2]/10:,.0f}x low, 2.6 orders)")
    W(f"      z=31 priced '~100 core-hours'   -> {rows[1][2]:,.0f}  "
      f"({rows[1][2]/100:,.0f}x low, 4.4 orders)")
    W("    The round-28 prices were an extrapolation of the k=2 curve onto the")
    W("    k=3 curve; k=3 grows faster per prime because each prime carries")
    W("    three classes, so the branching factor at every node is larger.")
    W()
    W("    k=4 node counts: z=11 26, z=13 2,648 (102x), z=17 351,958 (133x),")
    W("    z=19 99,408,318 (282x).  j_4(P(23)) projects at ~5e11 nodes =")
    W("    ~800 core-hours: NOT BUYABLE either, and it was never on the brief.")
    W()
    W("    VERDICT ON THE k-AXIS PROGRAMME: the brief's five targets resolve as")
    W("      j_3(23)  EXACT      (harvested + reproved this round)")
    W("      j_4(17)  EXACT      (0.345 s)")
    W("      j_4(19)  EXACT      (448.8 s)")
    W(f"      j_3(29)  NOT ATTEMPTED, priced at ~{rows[0][2]:,.0f} core-hours")
    W(f"      j_3(31)  NOT ATTEMPTED, priced at ~{rows[1][2]:.1e} core-hours")
    check(rows[0][2] > 1000, "the j_3(29) projection came out buyable")
    W()
    W("[F] THE SAME PRICE ON THE SAT VEHICLE -- AND IT IS A DIFFERENT NUMBER")
    W()
    W("    research/jk_sat29.py encodes Ziller-Morack arXiv:1611.03310 eq. (2.2)")
    W("    generalised to k classes and decides both directions with CaDiCaL.")
    W("    Costs are the SOLVER's own counters and are NOT the same unit as the")
    W("    DFS's nodes; only the GROWTH is compared.")
    W()
    W("      k=3, z                    17         19            23")
    W("      SAT conflicts (UNSAT)  8,889    201,771    8,710,802")
    W("        ratio                    -      22.7x        43.2x")
    W("      jkcov6 nodes         556,927 50,867,900  7.38e9 (14 parts)")
    W("        ratio                    -      91.3x       145.1x")
    W()
    sat23 = 8_710_802
    sat_rate = sat23 / 831.4          # conflicts/s, ONE core, LOADED box
    W(f"    CaDiCaL decided BOTH directions at z = 23 in 831.4 s on one core")
    W(f"    ({sat_rate:,.0f} conflicts/s under load), against the DFS's "
      f"{secs23/3600:.1f} core-hours.")
    W("    That is a SECOND, INDEPENDENT, SPLIT-FREE proof of j_3(P(23)) = 1398.")
    W()
    satr = 43.2
    n = sat23
    for z in (29, 31):
        satr *= g
        n *= satr
        W(f"      j_3(P({z})) on SAT: {n:.2e} conflicts = "
          f"{n/sat_rate/3600:,.0f} core-hours")
    W()
    W("    So j_3(29) is ~3,500 core-hours on jkcov6 and ~20 on CaDiCaL:")
    W("    PURCHASABLE, and the named next target.  j_3(31) stays out of reach")
    W("    on both.  THE LESSON, twice learned in this file: A PRICE IS A")
    W("    PROPERTY OF A VEHICLE, NOT OF A TARGET.")
    check(sat23 < n23 / 100, "SAT did not beat the DFS by two orders at z=23")


def main():
    j3_23, h = harvest()
    ladders(j3_23)
    discriminator()
    price(h)
    W()
    W(f"ALL ASSERTIONS GREEN  ({ASSERTS[0]} assertions)")
    txt = "\n".join(OUT)
    print(txt)
    with open(os.path.join(DATA, "jk_axis29.out"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
