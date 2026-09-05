"""R3.h.i - the flank brick: the two-sided walk from a junction, full periods m11..m23.

A junction x is an opening of M = {5..q} that is also a tooth of q' = nextprime(q).
L^+(x) = next opening above x, minus x;  L^-(x) = x minus previous opening.
S = L^- + L^+ is the span (the length of the merged gap when q' kills exactly x).

Computed here: the bucket identity (F1), the both-sides bar (F2), two-sided kill-spacing (F3),
the missing-gear (umbrella) bar (F4/F13), the pair (L^-, L^+) as an object (F6-F8), the
junction-versus-opening comparison (F9), column 0 (F10), the naive bucket bound (F12).

Writes results/fw_period.txt.
"""
import os
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "fw_period.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
MACHINES = [11, 13, 17, 19, 23]
F_KNOWN = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}


def build(q):
    gs = [p for p in PRIMES if p <= q]
    P = 1
    for g in gs:
        P *= g
    ba = bytearray(P)
    one = b"\x01"
    for g in gs:
        u = pow(6, -1, g)
        for t in (u, g - u):
            n = len(range(t, P, g))
            ba[t::g] = one * n
    return gs, P, ba


def openings_of(ba, P):
    op = array("l")
    pos = ba.find(0)
    while pos != -1:
        op.append(pos)
        pos = ba.find(0, pos + 1)
    return op


def pearson(xs, ys):
    n = len(xs)
    sx = sy = sxx = syy = sxy = 0
    for a, b in zip(xs, ys):
        sx += a
        sy += b
        sxx += a * a
        syy += b * b
        sxy += a * b
    num = n * sxy - sx * sy
    den = ((n * sxx - sx * sx) * (n * syy - sy * sy)) ** 0.5
    return num / den if den else 0.0


def main():
    for q in MACHINES:
        gs, P, ba = build(q)
        qp = next(p for p in PRIMES if p > q)
        up = pow(6, -1, qp)
        teeth_qp = (up % qp, (-up) % qp)
        U = {g: pow(6, -1, g) for g in gs}
        DG = {g: (2 * U[g]) % g for g in gs}
        ARC = {g: min(DG[g], g - DG[g]) for g in gs}      # short arc a_g
        LONG = {g: g - ARC[g] for g in gs}

        op = openings_of(ba, P)
        n = len(op)
        F = 0
        gaps = array("l", bytes(4 * n))
        for i in range(n - 1):
            d = op[i + 1] - op[i]
            gaps[i] = d
            if d > F:
                F = d
        gaps[n - 1] = P - op[n - 1] + op[0]
        if gaps[n - 1] > F:
            F = gaps[n - 1]
        d0 = op[1] - op[0]                                  # op[0] = 0 (shield)
        F2 = max(gaps[i] + gaps[(i + 1) % n] for i in range(n))

        say("")
        say("=" * 78)
        say("MACHINE m%d  gears %s  q'=%d  u'=%d  teeth of q' = %s" % (q, gs, qp, up, teeth_qp))
        say("period P = %d   openings = %d   F = %d (record %s)   F_2 = %d   d_0 = %d"
            % (P, n, F, F_KNOWN[q], F2, d0))
        say("column 0 open: %s   0 a tooth of q': %s   (shield: q' | 6*0)"
            % (op[0] == 0, 0 in teeth_qp))

        # ---- junctions ------------------------------------------------------
        jidx = [i for i in range(n) if op[i] % qp in teeth_qp]
        say("junctions (openings that are teeth of q'): %d = %.4f of openings (2/q' = %.4f)"
            % (len(jidx), len(jidx) / n, 2 / qp))

        Lm = [gaps[(i - 1) % n] for i in jidx]
        Lp = [gaps[i] for i in jidx]
        S = [a + b for a, b in zip(Lm, Lp)]
        smax = max(S)
        say("max span S over junctions = %d   budget F + q' = %d   slack %d   (F_2 = %d, %s)"
            % (smax, F + qp, F + qp - smax, F2,
               "max S = F_2" if smax == F2 else "max S < F_2 by %d" % (F2 - smax)))

        # every argmax junction
        arg = [k for k in range(len(jidx)) if S[k] == smax]
        say("argmax junctions: %d of them; columns %s"
            % (len(arg), [op[jidx[k]] for k in arg[:6]]))

        # ---- F1 the arc identity -------------------------------------------
        exc1 = 0
        checked1 = 0
        step = 1 if q <= 19 else 7
        for i in range(0, n, step):
            x = op[i]
            for g in gs:
                u = U[g]
                bp = min((u - x) % g, (-u - x) % g)
                bm = min((x - u) % g, (x + u) % g)
                checked1 += 1
                if bp + bm not in (ARC[g], LONG[g]):
                    exc1 += 1
        say("F1 arc identity  b+ + b- in {a_g, g-a_g}: %d (opening,gear) pairs checked, %d exceptions"
            % (checked1, exc1))

        # ---- per-junction gear analysis -------------------------------------
        both = [0] * len(jidx)          # gears striking both flanks
        miss = [0] * len(jidx)          # gears missing the whole stretch
        one = [0] * len(jidx)
        exc2 = exc4 = exc13 = 0
        minmiss_slack = []              # (g_miss - a_g) - S  for the smallest missing gear
        smallest_missing = []
        arc_short = arc_long = 0
        b1b2_fail = 0
        depth_tot = 0
        depth_cols = 0
        for k, i in enumerate(jidx):
            x = op[i]
            lm, lp, s = Lm[k], Lp[k], S[k]
            bs = []
            for g in gs:
                u = U[g]
                bp = min((u - x) % g, (-u - x) % g)
                bm = min((x - u) % g, (x + u) % g)
                bs.append((bp, bm, g))
                hitf = bp <= lp - 1
                hitb = bm <= lm - 1
                if hitf and hitb:
                    both[k] += 1
                    if ARC[g] > s:
                        exc2 += 1
                    if bp + bm == ARC[g]:
                        arc_short += 1
                    else:
                        arc_long += 1
                elif hitf or hitb:
                    one[k] += 1
                else:
                    miss[k] += 1
                    if LONG[g] < s + 2:
                        exc4 += 1
            mg = [g for (bp, bm, g) in bs if bp > lp - 1 and bm > lm - 1]
            if mg:
                gm = min(mg)
                smallest_missing.append(gm)
                minmiss_slack.append(LONG[gm] - s)
                if s > LONG[gm]:
                    exc13 += 1
            else:
                smallest_missing.append(0)
            fb = sorted(bp for (bp, bm, g) in bs)
            if lp > fb[0] + fb[1]:
                b1b2_fail += 1
            depth_cols += (s - 1)
            for off in range(1, lp):
                c = x + off
                depth_tot += sum(1 for g in gs if (c - U[g]) % g == 0 or (c + U[g]) % g == 0)
            for off in range(1, lm):
                c = x - off
                depth_tot += sum(1 for g in gs if (c - U[g]) % g == 0 or (c + U[g]) % g == 0)

        say("F2 both-sides bar (shared gear => a_g <= S): %d exceptions" % exc2)
        say("F4 missing gear => long arc >= S+2: %d exceptions" % exc4)
        say("F13 S <= long arc of smallest missing gear: %d exceptions" % exc13)
        say("gears per junction: both flanks mean %.3f max %d | one flank mean %.3f | miss mean %.3f"
            % (sum(both) / len(both), max(both), sum(one) / len(one), sum(miss) / len(miss)))
        say("shared-gear nearest pair: short arc %d, long arc %d (nearest strikes are always on "
            "opposite teeth: b+ + b- = an arc)" % (arc_short, arc_long))
        say("junctions with NO shared gear: %d of %d" % (sum(1 for b in both if b == 0), len(both)))
        if minmiss_slack:
            say("smallest missing gear: min %d max %d | umbrella slack (long arc - S): min %d "
                "median %d max %d" % (min(g for g in smallest_missing if g), max(smallest_missing),
                                      min(minmiss_slack), sorted(minmiss_slack)[len(minmiss_slack) // 2],
                                      max(minmiss_slack)))
        say("junctions where every gear strikes the stretch: %d"
            % sum(1 for g in smallest_missing if g == 0))
        say("F12 naive bucket bound L+ <= b(1)+b(2): %d failures of %d junctions"
            % (b1b2_fail, len(jidx)))
        say("mean depth of blocked columns in junction stretches: %.4f (machine sum 2/g = %.4f)"
            % (depth_tot / depth_cols, sum(2 / g for g in gs)))

        # ---- F6/F8 the pair as an object ------------------------------------
        rj = pearson(Lm, Lp)
        allL = [gaps[(i - 1) % n] for i in range(n)]
        allR = [gaps[i] for i in range(n)]
        ra = pearson(allL, allR)
        say("correlation of (L^-, L^+): junctions r = %+.5f (n=%d, s.e. %.5f) | all openings "
            "r = %+.5f (n=%d) | independent null 0" % (rj, len(jidx), 1 / len(jidx) ** 0.5, ra, n))
        thr = 0.7 * F
        big = [Lm[k] for k in range(len(jidx)) if Lp[k] >= thr]
        allbig = [allL[i] for i in range(n) if allR[i] >= thr]
        mg_all = sum(allR) / n
        say("E[L^- | L^+ >= 0.7F] junctions %.4f (n=%d) | all openings %.4f (n=%d) | mean gap %.4f"
            % ((sum(big) / len(big)) if big else float("nan"), len(big),
               (sum(allbig) / len(allbig)) if allbig else float("nan"), len(allbig), mg_all))

        # ---- F7 exchangeability ---------------------------------------------
        opset = {}
        for i in range(n):
            opset[op[i]] = i
        exc7 = 0
        for k, i in enumerate(jidx[:200000]):
            x = op[i]
            j = opset[(-x) % P]
            if gaps[j] != Lm[k] or gaps[(j - 1) % n] != Lp[k]:
                exc7 += 1
        say("F7 exchangeability (mirror x -> -x swaps the flanks): %d exceptions in %d junctions"
            % (exc7, min(len(jidx), 200000)))

        # ---- F9 junction versus all openings --------------------------------
        mj = sum(S) / len(S)
        ma = sum(allL[i] + allR[i] for i in range(n)) / n
        varj = sum((s - mj) ** 2 for s in S) / len(S)
        se = (varj / len(S)) ** 0.5
        say("F9 mean span: junctions %.5f | all openings %.5f | difference %+.5f = %+.2f s.e."
            % (mj, ma, mj - ma, (mj - ma) / se))
        say("F9 max span: junctions %d | all openings (F_2) %d" % (smax, F2))

        # ---- F10 the wrap pair at junctions ---------------------------------
        wrap = [k for k in range(len(jidx)) if Lm[k] == d0 and Lp[k] == d0]
        say("F10 wrap flanks (d_0, d_0) = (%d, %d) at junctions: %d occurrences (column 0 itself "
            "is never a junction)" % (d0, d0, len(wrap)))

        # ---- distribution of (L^-, L^+) --------------------------------------
        from collections import Counter
        cs = Counter(S)
        say("span distribution (top 12 by size): %s"
            % sorted(cs.items(), key=lambda t: -t[0])[:12])
        cp = Counter(zip(Lm, Lp))
        say("largest (L^-, L^+) pairs: %s"
            % sorted(cp.items(), key=lambda t: -(t[0][0] + t[0][1]))[:8])

        # ---- F5 conditional support ------------------------------------------
        byLp = {}
        for k in range(len(jidx)):
            byLp.setdefault(Lp[k], set()).add(Lm[k])
        realised = set(allR)
        pinned = [(v, sorted(s)) for v, s in sorted(byLp.items())
                  if sum(1 for k in range(len(jidx)) if Lp[k] == v) >= 30 and s != realised]
        say("F5 realised gap values: %s" % sorted(realised))
        say("F5 L^+ values (>=30 junctions) whose L^- support is NOT the full spectrum: %d"
            % len(pinned))
        for v, s in pinned[:6]:
            say("    L^+ = %2d -> L^- support %s (missing %s)" % (v, s, sorted(realised - set(s))))

        # ---- top-10 spans: mechanism ------------------------------------------
        order = sorted(range(len(jidx)), key=lambda k: -S[k])
        seen = set()
        top = []
        for k in order:
            key = (Lm[k], Lp[k])
            if len(top) >= 10:
                break
            top.append(k)
        say("TOP SPANS (junction column, L^-, L^+, S, tooth of q', stoppers, shared gears):")
        for k in top:
            i = jidx[k]
            x = op[i]
            lm, lp = Lm[k], Lp[k]
            tooth = "+u'" if x % qp == teeth_qp[0] else "-u'"
            endf = [g for g in gs if (x + lp - 1 - U[g]) % g == 0 or (x + lp - 1 + U[g]) % g == 0]
            endb = [g for g in gs if (x - lm + 1 - U[g]) % g == 0 or (x - lm + 1 + U[g]) % g == 0]
            sh = []
            for g in gs:
                u = U[g]
                bp = min((u - x) % g, (-u - x) % g)
                bm = min((x - u) % g, (x + u) % g)
                if bp <= lp - 1 and bm <= lm - 1:
                    sh.append((g, bp, bm, "short" if bp + bm == ARC[g] else "long"))
            wordf = []
            for off in range(1, lp):
                c = x + off
                wordf.append(min(g for g in gs if (c - U[g]) % g == 0 or (c + U[g]) % g == 0))
            wordb = []
            for off in range(1, lm):
                c = x - off
                wordb.append(min(g for g in gs if (c - U[g]) % g == 0 or (c + U[g]) % g == 0))
            say("  x=%d  (%d, %d) S=%d  tooth %s" % (x, lm, lp, lm + lp, tooth))
            say("     backward word (outward from x): %s   last-column strikers %s" % (wordb, endb))
            say("     forward  word (outward from x): %s   last-column strikers %s" % (wordf, endf))
            say("     shared gears (g, b+, b-, arc): %s" % sh)
        del ba, op, gaps


main()
LOG.close()
