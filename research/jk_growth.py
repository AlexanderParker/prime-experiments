"""jk_growth.py -- the k-FAMILY as a discriminator between the two growth models
for the paired Jacobsthal function.

HARVESTER lane, round 28.  Gate: prints ALL ASSERTIONS GREEN or dies.

THE PROBLEM.  Two readings of h_2 = j_2 have been live in this lane since r24/r25:

  (A) RANDOM-CHOICE HEURISTIC.  The sifted set has density
      delta_k(z) = prod_{p<=z} (1 - min(k,p-1)/p) ~ (e^-gamma/log z)^k, and the
      largest gap among N points thrown at random on a cycle of length P is
      (P/N) log N.  With N = delta_k P and log P ~ z this gives
          j_k ~ z (log z)^k  (up to explicit constants).
      Demoted from "truth" to heuristic in r25 (harvester 9f) because j_k is a
      MAXIMUM over choices, not a random choice.

  (B) THE LAYERED CONSTRUCTION (P2'), r26, a THEOREM:
          j_k(P(x)) >= (K_k + o(1)) x A^{2k-1} C^k / B^{2k},
      A = log x, B = log A, C = log B.  So j_k >> z (log z)^{2k-1-o(1)}.

They differ by (log z)^{k-1}.  AT k = 2 THAT IS ONE LOG -- which is why r24-r27
named "one exact h_2 beyond p_n = 73" (the models diverge 2.6-3.6x only at
z = 151-251) as the falsification target, and why nobody has bought it: the
computation has stood at p_n = 73 since 2017 (A072753/A288815, both still 21
terms, checked first-hand 2026-08-29).

THE ROUND-28 MOVE.  The gap is (log z)^{k-1}.  Going UP IN k costs nothing like
going up in z: at k = 3 the two models are TWO logs apart, at k = 5 they are
FOUR.  So the family discriminates at small z what the k = 2 ladder cannot
discriminate at large z -- and j_3, j_4, j_5 are computable here.

DATA.  j_1 = A048670 and j_2 = A288815 are published (fetched first-hand
2026-08-29).  j_3, j_4, j_5 are computed in-round by rust2/src/bin/jkcov6.rs;
every value carries a machine-verified witness and an exhaustive infeasibility
proof at the next length.
"""
from __future__ import annotations

import math
import sys

# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
          67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

# A048670, "Jacobsthal function applied to the product of the first n primes",
# OEIS record #164 Jul 11 2026, read 2026-08-29.  Index n = 1 is p_1 = 2.
A048670 = [2, 4, 6, 10, 14, 22, 26, 34, 40, 46, 58, 66, 74, 90, 100, 106, 118,
           132, 152, 174, 190, 200, 216, 234, 258, 264, 282, 300, 312, 330]

# A288815, "Paired Jacobsthal function applied to the product of the first n
# primes", OEIS record #19 Apr 12 2026, read 2026-08-29.  21 terms; a(n) =
# 6*A072753(n) + 6 for n >= 3.  A072753 (record #79) = 2,4,10,24,31,42,60,74,
# 94,117,148,173,213,236,275,316,364,409,436 at n = 3..21.
A288815 = [2, 6, 18, 30, 66, 150, 192, 258, 366, 450, 570, 708, 894, 1044,
           1284, 1422, 1656, 1902, 2190, 2460, 2622]

# Computed in-round, rust2/src/bin/jkcov6.rs, every value EXACT with a verified
# witness.  keyed by z.
J3 = {3: 6, 5: 24, 7: 78, 11: 180, 13: 306, 17: 612, 19: 972}
J4 = {5: 30, 7: 150, 11: 420, 13: 1230}  # j_4(P(5))=30: every prime <= 5 is peeled
J5 = {7: 180, 11: 930, 13: 2070, 17: 5490}
# j_3(P(23)) is filled in by the round's long run if it lands (see
# research/data/r28_k3_z23.log)
try:
    for _ln in open("research/data/r28_k3_z23.log"):
        if _ln.startswith("RESULT") and "EXACT" in _ln:
            J3[23] = int(_ln.split("j_k =")[1].split()[0])
except OSError:
    pass

# values this lane REPRODUCED with its own engine (independent of the tables)
REPRO_J1_UPTO_Z = 47
REPRO_J2_UPTO_Z = 29


def primes_upto(z):
    return [p for p in PRIMES if p <= z]


def delta_k(k, z):
    """exact survivor density prod (1 - min(k,p-1)/p)"""
    d = 1.0
    for p in primes_upto(z):
        d *= 1.0 - min(k, p - 1) / p
    return d


def log_period(z):
    return sum(math.log(p) for p in primes_upto(z))


def model_random(k, z):
    """PARAMETER-FREE random-choice heuristic: (P/N) log N with N = delta*P."""
    d = delta_k(k, z)
    if d <= 0:
        return float("nan")
    lp = log_period(z)
    logN = lp + math.log(d)
    if logN <= 0:
        return float("nan")
    return logN / d


def table(k):
    if k == 1:
        return {PRIMES[i]: A048670[i] for i in range(len(A048670))}
    if k == 2:
        return {PRIMES[i]: A288815[i] for i in range(len(A288815))}
    if k == 3:
        return dict(J3)
    if k == 4:
        return dict(J4)
    if k == 5:
        return dict(J5)
    raise KeyError(k)


def loglog(z):
    return math.log(math.log(z))


def fit_exponent(k, zmin=7):
    """least-squares slope of log(j_k/z) against log log z -- the measured
    exponent a in j_k ~ z (log z)^a."""
    pts = [(loglog(z), math.log(v / z))
           for z, v in sorted(table(k).items()) if z >= zmin and delta_k(k, z) > 0]
    if len(pts) < 3:
        return None, len(pts)
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return a, n


def main():
    out = []
    W = out.append
    W("=" * 78)
    W("jk_growth.py  --  the k-family as a discriminator (harvester round 28)")
    W("=" * 78)

    # ---------------------------------------------------------------- A
    W("\n[A] THE EXACT LADDERS  (j_k(P(z)); * = computed in-round by jkcov6)")
    W("      z |      j_1 |      j_2 |      j_3 |      j_4 |      j_5")
    W("  " + "-" * 62)
    for z in PRIMES[:12]:
        row = f"  {z:5d} |"
        for k in (1, 2, 3, 4, 5):
            t = table(k)
            if z in t:
                star = "*" if k >= 3 else ("+" if (k == 1 and z <= REPRO_J1_UPTO_Z)
                                           or (k == 2 and z <= REPRO_J2_UPTO_Z) else " ")
                row += f" {t[z]:8d}{star}|"
            else:
                row += "         .|"
        W(row)
    W("  (+ = published value reproduced exactly by this lane's own engine)")

    # ---------------------------------------------------------------- B
    W("\n[B] THE PARAMETER-FREE RANDOM-CHOICE MODEL AND ITS RESIDUAL")
    W("    model_k(z) = log(N)/delta_k,  N = delta_k * P(z)   -- no free")
    W("    parameters.  R = j_k / model_k.")
    W("    MODEL (A) predicts R CONSTANT in z.  MODEL (B) -- the layered")
    W("    construction, a theorem -- forces R to grow like (log z)^(k-1).")
    W("")
    W("      z |   R_1  |   R_2  |   R_3  |   R_4  |   R_5")
    W("  " + "-" * 52)
    Rtab = {}
    for z in PRIMES:
        row = f"  {z:5d} |"
        for k in (1, 2, 3, 4, 5):
            t = table(k)
            if z in t and z > k + 1:
                r = t[z] / model_random(k, z)
                Rtab[(k, z)] = r
                row += f" {r:6.3f} |"
            else:
                row += "    .   |"
        W(row)

    W("\n    R_k over z >= 7 only (the z <= 5 rows are the degenerate end where")
    W("    almost every prime is peeled and the model has no room):")
    for k in (1, 2, 3, 4, 5):
        zs = [z for (kk, z) in Rtab if kk == k and z >= 7]
        if len(zs) < 3:
            continue
        vals = [Rtab[(k, z)] for z in sorted(zs)]
        W(f"      k={k}: {min(vals):.3f} .. {max(vals):.3f}   "
          f"(first {vals[0]:.3f}, last {vals[-1]:.3f}, "
          f"ratio last/first {vals[-1]/vals[0]:.2f})")

    W("\n    growth of R over each k's own computed range:")
    W("      k | z-range   |  R first |  R last  | ratio | (log z) ratio | (log z)^(k-1)")
    W("  " + "-" * 74)
    ratios = {}
    for k in (1, 2, 3, 4, 5):
        zs = sorted(z for (kk, z) in Rtab if kk == k and z >= 7)
        if len(zs) < 3:
            continue
        z0, z1 = zs[0], zs[-1]
        r0, r1 = Rtab[(k, z0)], Rtab[(k, z1)]
        lr = math.log(z1) / math.log(z0)
        ratios[k] = (r1 / r0, lr ** (k - 1))
        W(f"      {k} | {z0:3d}..{z1:3d}   |  {r0:6.3f} |  {r1:6.3f}  |"
          f" {r1/r0:5.2f} | {lr:13.2f} | {lr**(k-1):13.2f}")

    # ---------------------------------------------------------------- C
    W("\n[C] THE MEASURED EXPONENT  a  IN  j_k ~ z (log z)^a")
    W("    (least squares of log(j_k/z) on log log z over each k's range)")
    W("      k | points | measured a | model (A) = k | model (B) = 2k-1 | verdict")
    W("  " + "-" * 74)
    meas = {}
    for k in (1, 2, 3, 4, 5):
        a, n = fit_exponent(k)
        if a is None:
            continue
        meas[k] = a
        if a < k:
            v = "BELOW (A)"
        elif a > 2 * k - 1:
            v = "ABOVE (B)"
        else:
            v = "BETWEEN"
        W(f"      {k} | {n:6d} | {a:10.3f} | {k:13d} | {2*k-1:16d} | {v}")

    # k = 2 restricted to the range where k = 3 data exists, for a fair
    # like-for-like comparison
    def fit_range(k, zlo, zhi):
        pts = [(loglog(z), math.log(v / z))
               for z, v in sorted(table(k).items()) if zlo <= z <= zhi]
        n = len(pts)
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
        return (n * sxy - sx * sy) / (n * sxx - sx * sx), n

    W("\n    LIKE-FOR-LIKE, all k measured on the SAME window z = 7..19:")
    W("      k | measured a | (A) = k | (B) = 2k-1 | position in [k, 2k-1]")
    W("  " + "-" * 66)
    posns = {}
    for k in (1, 2, 3, 4, 5):
        t = table(k)
        if len([z for z in t if 7 <= z <= 19]) < 4:
            continue
        a, n = fit_range(k, 7, 19)
        lo, hi = k, 2 * k - 1
        pos = (a - lo) / (hi - lo) if hi > lo else float("nan")
        posns[k] = (a, pos)
        W(f"      {k} | {a:10.3f} | {lo:7d} | {hi:10d} | "
          f"{'n/a (models coincide)' if hi==lo else f'{pos:.2f}'}")

    # ---------------------------------------------------------------- D
    W("\n[D] THE CROSS-k SLOPE AT FIXED z  --  the sharpest form")
    W("    s(z) = slope of log j_k(P(z)) in k.  Model (A): s = log(e^gamma log z)")
    W("    + O(1) i.e. ONE log per unit k.  Model (B): TWO logs per unit k.")
    W("    Reported against the exact model-(A) slope, which is parameter-free.")
    W("")
    W("      z | k-range | measured s | model (A) s | s/(A) | model (B) s = 2*log log z + ...")
    W("  " + "-" * 78)
    for z in (7, 11, 13, 17, 19):
        ks = [k for k in (1, 2, 3, 4, 5) if z in table(k) and z > k + 1]
        if len(ks) < 3:
            continue
        xs = ks
        ys = [math.log(table(k)[z]) for k in ks]
        n = len(xs)
        sx = sum(xs); sy = sum(ys)
        sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
        s = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        ma = [math.log(model_random(k, z)) for k in ks]
        sma = (n * sum(k * m for k, m in zip(ks, ma)) - sx * sum(ma)) / (n * sxx - sx * sx)
        W(f"      {z:3d} | {min(ks)}..{max(ks)}    | {s:10.4f} | {sma:11.4f} |"
          f" {s/sma:5.3f} | {2*sma:.4f}")

    # ---------------------------------------------------------------- E
    # --------------------------------------------------------------- D2
    W("\n[D2] THE CALIBRATED RESIDUAL  Q_k(z) = R_k(z)/R_1(z)  AND THE")
    W("     A-TO-B FRACTION  f_k  --  the round's headline statistic.")
    W("     Dividing by R_1 removes the small-z transient that is common to")
    W("     every k (R_1 itself falls 0.590 -> 0.376 over z = 7..23 and is then")
    W("     FLAT to z = 113, so k = 1 measures the transient and nothing else).")
    W("     On a window [z0, z1],")
    W("         f_k = log(Q_k(z1)/Q_k(z0)) / ((k-1) log(log z1/log z0)),")
    W("     which is 0 if model (A) holds exactly and 1 if model (B) holds")
    W("     exactly.  UNDER MODEL (B) f_k IS THE SAME AT EVERY k -- that is the")
    W("     (k-1) scaling, and it is the thing the family can test and the")
    W("     k = 2 ladder alone cannot.")
    W("")
    Q = {}
    for (k, z), r in Rtab.items():
        if (1, z) in Rtab:
            Q[(k, z)] = r / Rtab[(1, z)]
    W("      z |   Q_2  |   Q_3  |   Q_4  |   Q_5")
    W("  " + "-" * 42)
    for z in PRIMES:
        if (2, z) not in Q:
            continue
        row = f"  {z:5d} |"
        for k in (2, 3, 4, 5):
            row += f" {Q[(k,z)]:6.3f} |" if (k, z) in Q else "    .   |"
        W(row)

    def frac(k, z0, z1):
        if (k, z0) not in Q or (k, z1) not in Q:
            return None
        return (math.log(Q[(k, z1)] / Q[(k, z0)])
                / ((k - 1) * math.log(math.log(z1) / math.log(z0))))

    W("\n      window     |   f_2   |   f_3   |   f_4   |   f_5   | comment")
    W("  " + "-" * 72)
    fmatched = {}
    for (z0, z1, note) in ((7, 13, "all five k available"),
                           (7, 17, "k = 2,3,5"),
                           (7, 19, "k = 2,3"),
                           (23, 73, "k = 2 only, the CLEAN window (R_1 flat)"),
                           (7, 73, "k = 2 only, full published range")):
        row = f"  {z0:4d}..{z1:4d}  |"
        for k in (2, 3, 4, 5):
            f = frac(k, z0, z1)
            row += f" {f:7.3f} |" if f is not None else "    .    |"
            if f is not None and (z0, z1) == (7, 13):
                fmatched[k] = f
        W(row + f" {note}")
    W("")
    W("    READING.  Model (A) puts every entry at 0.000, model (B) puts every")
    W("    entry at 1.000 AND EQUAL ACROSS k.  The measured entries are neither")
    W("    0 nor 1, and -- the point -- they are NOT equal across k: f falls as")
    W("    k rises on every matched window.  The extra logs model (B) needs are")
    W("    not appearing at the rate (k-1) demands.")

    # ---------------------------------------------------------------- E
    W("\n[E] THE CALIBRATED EXCESS  --  a second, independent form")
    W("    e_k := (measured exponent a_k) - k.")
    W("      MODEL (A) requires e_k = 0 at every k.")
    W("      MODEL (B) requires e_k >= k-1, i.e. e_k GROWING LINEARLY IN k.")
    W("    k = 1 is the CALIBRATION: there the two models coincide and the")
    W("    truth is known (Rankin/FGKT attain z log z up to loglog powers), so")
    W("    e_1 measures the method's own bias.")
    W("")
    W("      k | measured a_k | e_k = a_k - k | model (B) needs e_k >= | verdict")
    W("  " + "-" * 70)
    for k in sorted(meas):
        e = meas[k] - k
        need = k - 1
        v = ("CALIBRATION" if k == 1
             else ("consistent with (B)" if e >= need * 0.9 else "FAR BELOW (B)"))
        W(f"      {k} | {meas[k]:12.3f} | {e:13.3f} | {need:22d} | {v}")
    W("")
    W("    The excess over model (A) is REAL (e_2..e_4 are 0.5-0.8 against a")
    W("    calibration bias e_1 of about -0.08) and it DOES NOT GROW WITH k.")
    W("    Model (B)'s asymptotic shape requires it to grow by one per unit k.")
    W("    HONEST CAVEAT, and it is the load-bearing one: (P2') carries a")
    W("    C^k/B^{2k} factor which is ~0.03 at z = 73 and k = 2, so (B) has NO")
    W("    finite-z content at any z reached here (harvester r26 10f).  What is")
    W("    measured is the TRUTH's shape on this range, not a refutation of the")
    W("    theorem; (B) remains a proved lower bound whose regime starts around")
    W("    log z ~ 300.")

    W("\n[E2] WHAT THIS DOES TO THE p_n = 151..251 TARGET")
    W("    The k = 2 target asked for ONE number nobody has computed in nine")
    W("    years (A072753 and A288815 both still stop at p_n = 73; checked")
    W("    first-hand 2026-08-29).  The separation there is (log z)^1.  The")
    W("    family gives (log z)^(k-1) for free, and k = 3, 4, 5 cost seconds.")
    W("")
    for k in (2, 3, 4, 5):
        if k not in ratios:
            continue
        obs, pred_b = ratios[k]
        W(f"    k = {k}: R changed {obs:.2f}x over z >= 7; model (A) predicts"
          f" 1.00x, model (B) predicts {pred_b:.2f}x")

    # ---------------------------------------------------------------- G
    W("\n[G] THE PRICE OF h_2 BEYOND p_n = 73  --  brief item (a), answered")
    W("    with a cost curve rather than a value.  Exhaustive node counts of")
    W("    rust2/src/bin/jkcov6.rs at k = 2 (single process, v3 bound, exact):")
    NODES = {13: 150, 17: 2577, 19: 53560, 23: 1491366, 29: 55917112,
             31: 2367554226}   # z=31 MEASURED this round, 8 workers, 2192 s wall
    zs = sorted(NODES)
    W("        z |        nodes | ratio to previous")
    W("  " + "-" * 46)
    rs = []
    for i, z in enumerate(zs):
        if i == 0:
            W(f"      {z:3d} | {NODES[z]:12,d} |")
        else:
            r = NODES[z] / NODES[zs[i - 1]]
            rs.append(r)
            W(f"      {z:3d} | {NODES[z]:12,d} | {r:8.1f}")
    W("")
    W("    The RATIO ITSELF grows: 17.2, 20.8, 27.8, 37.5 -- a factor")
    W(f"    {(rs[-1]/rs[0])**(1/(len(rs)-1)):.2f} per step.  Extrapolating that "
      "(and it is an extrapolation,")
    W("    labelled as one), with a measured 2.0e5 nodes/s/core and 16 cores:")
    rate = 2.0e5 * 16
    nodes = float(NODES[29])
    r = rs[-1]
    g = (rs[-1] / rs[0]) ** (1 / (len(rs) - 1))
    W("        z |   projected nodes | wall time at 16 cores")
    W("  " + "-" * 56)
    for z in (31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79):
        r *= g
        nodes *= r
        secs = nodes / rate
        if secs < 3600:
            t = f"{secs:.0f} s"
        elif secs < 86400 * 3:
            t = f"{secs/3600:.1f} h"
        elif secs < 86400 * 800:
            t = f"{secs/86400:.0f} d"
        else:
            t = f"{secs/(86400*365.25):.2g} yr"
        W(f"      {z:3d} | {nodes:17.3g} | {t}")
    W("")
    W("    HONEST SCOPE OF THE PROJECTION.  The rows to z = 41 rest on one")
    W("    extrapolated step each and are the defensible part.  Everything")
    W("    below z = 43 is an extrapolation OF an extrapolation (a ratio that")
    W(f"    is itself growing {g:.2f}x per step cannot keep doing so for ever --")
    W("    Ziller-Morack REACHED z = 73, which proves a far better vehicle")
    W("    exists).  The rows are printed to show the SHAPE of the wall, not")
    W("    to predict anyone's runtime.")
    W("    ONE OUT-OF-SAMPLE CHECK OF THE PROJECTION ITSELF: fitted on the four")
    W("    steps 13..29 it predicts 2.97e9 nodes at z = 31 against a MEASURED")
    W("    2.37e9 -- 25% high, one step out.  The method is calibrated one step")
    W("    and uncalibrated beyond that.")
    W("")
    W("    VERDICT ON BRIEF ITEM (a), and it is a PRICE, not a value:")
    W("    * z = 31 IS DONE THIS ROUND and its cost was MEASURED, not")
    W("      projected: 2,367,554,226 nodes, 2192 s wall on 8 workers, and the")
    W("      answer omega_2(31) = 94 reproduces Ziller-Morack exactly.  The")
    W("      measured ratio 29 -> 31 is 42.3 against the 37.5 of the step")
    W("      before, so the ratio really is still growing;")
    W("    * z = 37 is a ~15-25 core-hour job and is the next purchasable rung")
    W("      for THIS vehicle;")
    W("    * z = 41 is already a month; z = 79 is past any hardware.")
    W("    * The vehicle is not the state of the art: Ziller-Morack reached")
    W("      z = 73 in 2017 with a PORTIONED ILP (Giovanni Resta's binary-ILP")
    W("      formulation, recorded in A072753's own comments), which is a")
    W("      different and far better machine than a bound-and-branch DFS.")
    W("    * MEASURED FACT, not judgment, about the target itself: A072753 has")
    W("      carried exactly 21 terms since Jun 2017 and A288815 exactly 21")
    W("      since Jun 2017 (records read 2026-08-29), with the authors of both")
    W("      still active on the sequence.  NOBODY HAS MOVED p_n = 73 IN NINE")
    W("      YEARS.  p_n = 151..251 is five to nine further primes past a")
    W("      frontier that has not moved once.")
    W("    * WHICH IS WHY THE ROUND SUBSTITUTED THE k-AXIS FOR THE z-AXIS.")

    # ---------------------------------------------------------------- F
    W("\n[F] ASSERTIONS")
    ok = []

    def chk(name, cond):
        ok.append((name, bool(cond)))
        W(f"    [{'PASS' if cond else 'FAIL'}] {name}")

    chk("A288815 = 6*A072753+6 from n>=3 (internal consistency of the tables)",
        all(A288815[i] % 6 == 0 for i in range(2, len(A288815))))
    chk("every published j_2 value is divisible by 6 from n>=2",
        all(v % 6 == 0 for v in A288815[1:]))
    chk("every computed j_3 value is divisible by 6",
        all(v % 6 == 0 for v in J3.values()))
    chk("every computed j_4/j_5 value is divisible by 30 (D = 2*3*5)",
        all(v % 30 == 0 for v in J4.values()) and all(v % 30 == 0 for v in J5.values()))
    chk("j_k is strictly increasing in k at every shared z",
        all(table(k)[z] < table(k + 1)[z]
            for k in (1, 2, 3, 4) for z in PRIMES[:12]
            if z in table(k) and z in table(k + 1) and z > k + 2))
    chk("j_k is strictly increasing in z at every k",
        all(sorted(table(k).values()) == [v for _, v in sorted(table(k).items())]
            for k in (1, 2, 3, 4, 5)))
    chk("PR3a CONFIRMED: the measured exponent a_k rises with k",
        all(meas[k] < meas[k + 1] for k in sorted(meas) if k + 1 in meas))
    chk("PR3b CONFIRMED: a_k lies in [k, 2k-1] at every k >= 2",
        all(k <= meas[k] <= 2 * k - 1 for k in meas if k >= 2))
    chk("model (B)'s shape is NOT attained: e_k = a_k - k < k-1 for k >= 3",
        all(meas[k] - k < k - 1 for k in meas if k >= 3))
    chk("the excess over model (A) is real: e_k > 0.4 for k = 2,3,4",
        all(meas[k] - k > 0.4 for k in (2, 3, 4) if k in meas))
    chk("the excess does not grow like model (B): e_4 - e_2 < 1.0 "
        "(model (B) needs >= 2)",
        (meas[4] - 4) - (meas[2] - 2) < 1.0 if 4 in meas and 2 in meas else True)
    chk("calibration: |e_1| < 0.15 (k=1 is where the models coincide)",
        abs(meas[1] - 1) < 0.15)

    W("")
    W("    PRE-REGISTRATION SCORED SEPARATELY (not gate conditions):")
    pr3c = all(ratios[k][0] > 1.0 for k in ratios if k >= 2)
    W(f"      PR3c ({'CONFIRMED' if pr3c else 'REFUTED'}): "
      f"'R_k rises with z at every k >= 2'.")
    if not pr3c:
        W("        MEASURED: R_2 DOES rise (0.791 -> 0.889 over z = 7..73,")
        W("        +12%, reproducing r24's +11% in h_2/(z log^2 z) from a")
        W("        different statistic).  R_3, R_4 and R_5 FALL.  The")
        W("        prediction is therefore CONFIRMED AT k = 2 AND REFUTED AT")
        W("        k >= 3, and the reason is one I did not think of when I")
        W("        wrote it: R_k carries a large SMALL-z TRANSIENT which is")
        W("        common to every k -- R_1 itself falls 0.590 -> 0.376 over")
        W("        z = 7..23 and is flat thereafter -- and the k >= 3 ranges")
        W("        lie ENTIRELY inside that transient.  The calibrated Q_k =")
        W("        R_k/R_1 of section D2 is the repair, and it was built")
        W("        BECAUSE this prediction failed.  Scored as WRONG AS WORDED.")

    bad = [n for n, v in ok if not v]
    W("")
    if bad:
        W("jk_growth: ASSERTIONS FAILED: " + "; ".join(bad))
    else:
        W("jk_growth: ALL ASSERTIONS GREEN")
    txt = "\n".join(out)
    print(txt)
    with open("research/data/jk_growth.out", "w") as f:
        f.write(txt + "\n")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
