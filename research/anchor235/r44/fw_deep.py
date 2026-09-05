"""R3.h.i part 2 - the anchor coupling of the two flanks, the correlation decomposition,
the gear bands, the L4 certificate at junctions, and the layer nest of the longest flanks.

Writes results/fw_deep.txt.
"""
import os
from array import array
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "fw_deep.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
MACHINES = [11, 13, 17, 19, 23]


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
            ba[t::g] = one * len(range(t, P, g))
    return gs, P, ba


def openings_of(ba):
    op = array("l")
    pos = ba.find(0)
    while pos != -1:
        op.append(pos)
        pos = ba.find(0, pos + 1)
    return op


def pearson(pairs):
    n = len(pairs)
    if n < 3:
        return None
    sx = sy = sxx = syy = sxy = 0
    for a, b in pairs:
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b
    num = n * sxy - sx * sy
    den = ((n * sxx - sx * sx) * (n * syy - sy * sy)) ** 0.5
    return num / den if den else None


# --- the anchor coupling, derived from the open classes alone ------------------
A5 = {0, 2, 3}                      # openings mod 5 (teeth +-1)


def allowed_minus_mod5(lp):
    """the L^- classes mod 5 compatible with L^+ = lp (mod 5) through gear 5 alone"""
    rs = [r for r in A5 if (r + lp) % 5 in A5]
    out = set()
    for r in rs:
        for s in A5:
            out.add((r - s) % 5)
    return out, rs


def main():
    say("ANCHOR COUPLING of the two flanks, from gear 5's open classes {0,2,3} mod 5:")
    for lp in range(5):
        out, rs = allowed_minus_mod5(lp)
        say("  L^+ = %d (mod 5)  ->  x = %s (mod 5)  ->  L^- in %s (mod 5), forbidden %s"
            % (lp, sorted(rs), sorted(out), sorted(set(range(5)) - out)))

    for q in MACHINES:
        gs, P, ba = build(q)
        qp = next(p for p in PRIMES if p > q)
        up = pow(6, -1, qp)
        teeth_qp = (up % qp, (-up) % qp)
        U = {g: pow(6, -1, g) for g in gs}
        DG = {g: (2 * U[g]) % g for g in gs}
        ARC = {g: min(DG[g], g - DG[g]) for g in gs}
        LONG = {g: g - ARC[g] for g in gs}
        op = openings_of(ba)
        n = len(op)
        gaps = array("l", bytes(4 * n))
        for i in range(n - 1):
            gaps[i] = op[i + 1] - op[i]
        gaps[n - 1] = P - op[n - 1] + op[0]
        F = max(gaps)

        say("")
        say("=" * 78)
        say("MACHINE m%d  P=%d  openings=%d  F=%d  q'=%d" % (q, P, n, F, qp))

        # --- C1 the mod-5 coupling, every opening --------------------------------
        exc5 = 0
        for i in range(n):
            lm = gaps[(i - 1) % n]
            lp = gaps[i]
            out, _ = allowed_minus_mod5(lp % 5)
            if lm % 5 not in out:
                exc5 += 1
        say("C1 anchor coupling (L^- mod 5 in the set forced by L^+ mod 5): %d exceptions in %d "
            "openings" % (exc5, n))

        # --- C2 the mod-35 pair table (gears 5 and 7) ---------------------------
        A35 = set(r for r in range(35) if r % 5 in A5 and r % 7 not in
                  (pow(6, -1, 7) % 7, (-pow(6, -1, 7)) % 7))
        pair_ok = set()
        for lm in range(35):
            for lp in range(35):
                if any(((r + lp) % 35 in A35 and (r - lm) % 35 in A35) for r in A35):
                    pair_ok.add((lm, lp))
        exc35 = 0
        seen35 = set()
        for i in range(n):
            lm = gaps[(i - 1) % n] % 35
            lp = gaps[i] % 35
            seen35.add((lm, lp))
            if (lm, lp) not in pair_ok:
                exc35 += 1
        say("C2 {5,7} coupling: %d of 1225 (L^- , L^+) classes mod 35 admissible; %d exceptions "
            "in %d openings; %d classes realised" % (len(pair_ok), exc35, n, len(seen35)))

        # --- C3 correlation decomposition ---------------------------------------
        allp = [(gaps[(i - 1) % n], gaps[i]) for i in range(n)]
        r_all = pearson(allp)
        by5 = {}
        by35 = {}
        for i in range(n):
            x = op[i]
            by5.setdefault(x % 5, []).append((gaps[(i - 1) % n], gaps[i]))
            by35.setdefault(x % 35, []).append((gaps[(i - 1) % n], gaps[i]))
        w5 = sum(len(v) * pearson(v) for v in by5.values() if pearson(v) is not None) / n
        w35 = sum(len(v) * pearson(v) for v in by35.values() if pearson(v) is not None) / n
        say("C3 correlation of (L^-, L^+): pooled %+.5f | within x mod 5 classes %+.5f | "
            "within x mod 35 classes %+.5f" % (r_all, w5, w35))
        say("C3 per-class (x mod 5): %s"
            % {k: round(pearson(v), 4) for k, v in sorted(by5.items())})

        # --- C4 gear bands -------------------------------------------------------
        jidx = [i for i in range(n) if op[i] % qp in teeth_qp]
        bandI = bandII = bandIII = 0
        hitI = hitII = hitIII = 0
        for i in jidx:
            x = op[i]
            lm = gaps[(i - 1) % n]
            lp = gaps[i]
            S = lm + lp
            for g in gs:
                u = U[g]
                bp = min((u - x) % g, (-u - x) % g)
                bm = min((x - u) % g, (x + u) % g)
                hf = bp <= lp - 1
                hb = bm <= lm - 1
                if LONG[g] < S + 2:
                    bandI += 1
                    if hf or hb:
                        hitI += 1
                elif ARC[g] <= S:
                    bandII += 1
                    if hf or hb:
                        hitII += 1
                else:
                    bandIII += 1
                    if hf or hb:
                        hitIII += 1
        say("C4 gear bands over %d junctions (cells = junction x gear):" % len(jidx))
        say("   band I  long arc < S+2  (must strike):      %8d cells, %8d strike (%.4f)"
            % (bandI, hitI, hitI / bandI if bandI else 0))
        say("   band II short arc <= S <= long arc-2:       %8d cells, %8d strike (%.4f)"
            % (bandII, hitII, hitII / bandII if bandII else 0))
        say("   band III short arc > S (at most one strike):%8d cells, %8d strike (%.4f)"
            % (bandIII, hitIII, hitIII / bandIII if bandIII else 0))

        # --- C5 the L4 certificate at the junctions with content -----------------
        opset = set(op)
        thr = F - 4
        cand = []
        for i in jidx:
            lm = gaps[(i - 1) % n]
            lp = gaps[i]
            if lm + lp >= thr:
                cand.append((i, lm, lp))
        say("C5 L4 re-phasing certificate at the %d junctions with S >= F-4 = %d:"
            % (len(cand), thr))
        worst = None
        losses = []
        for (i, lm, lp) in cand:
            x = op[i]
            S = lm + lp
            best = -1
            bestg = None
            for g0 in gs:
                u = U[g0]
                for t in (x - u, x + u):        # translate g0 so a tooth lands on x
                    def blocked(c):
                        if (c - t - u) % g0 == 0 or (c - t + u) % g0 == 0:
                            return True
                        for g in gs:
                            if g == g0:
                                continue
                            if (c - U[g]) % g == 0 or (c + U[g]) % g == 0:
                                return True
                        return False
                    lo = x
                    while blocked(lo):
                        lo -= 1
                    hi = x
                    while blocked(hi):
                        hi += 1
                    cert = hi - lo
                    if cert > best:
                        best = cert
                        bestg = (g0, "low" if t == x - u else "high")
            losses.append(S - best)
            if worst is None or S - best > worst[0]:
                worst = (S - best, x, lm, lp, best, bestg)
        if losses:
            say("   loss = S - cert:  min %d  median %d  max %d  against q' = %d ; exceptions "
                "(loss > q'): %d" % (min(losses), sorted(losses)[len(losses) // 2], max(losses),
                                     qp, sum(1 for l in losses if l > qp)))
            say("   worst cell: loss %d at x=%d (%d,%d), cert %d by gear %s"
                % (worst[0], worst[1], worst[2], worst[3], worst[4], worst[5]))

        # --- C6 layer nest of the two flanks at the ten longest junctions --------
        order = sorted(range(len(jidx)), key=lambda k: -(gaps[(jidx[k] - 1) % n] + gaps[jidx[k]]))
        say("C6 layer nest of each flank at the ten longest junctions "
            "(k_g = pieces of the flank at layer g; g* = largest gear removing an interior "
            "survivor of the flank):")
        for k in order[:10]:
            i = jidx[k]
            x = op[i]
            lm = gaps[(i - 1) % n]
            lp = gaps[i]
            line = []
            for side, (lo, hi) in (("back", (x - lm, x)), ("fwd", (x, x + lp))):
                ks = []
                gstar = None
                prev = None
                for g in gs:
                    surv = [c for c in range(lo, hi + 1)
                            if all((c - U[h]) % h != 0 and (c + U[h]) % h != 0
                                   for h in gs if h <= g)]
                    ks.append((g, len(surv) - 1))
                    if prev is not None and len(surv) - 1 < prev:
                        gstar = g
                    prev = len(surv) - 1
                line.append((side, hi - lo, ks, gstar))
            say("  x=%d (%d,%d) S=%d" % (x, lm, lp, lm + lp))
            for side, ln, ks, gstar in line:
                if gstar:
                    below = [kk for (gg, kk) in ks if gg < gstar][-1]
                    fus = "closed by %d, joining %d pieces" % (gstar, below)
                else:
                    fus = "no gear removes an interior survivor"
                say("     %-4s len %2d  k_g %s  %s" % (side, ln, ks, fus))
        del ba, op, gaps


main()
LOG.close()
