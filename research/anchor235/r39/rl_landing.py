"""R2.a.i.a - part 2: the landing on the landscape.

For every prime q = 5..20000: the walk length L, whether the landing offset L is an ISLAND for
B in {5, 7, 11, 13, 17, 19}, its admissible-gear count |G(L)| (gears <= q), lambda(L), and its
rank among the offsets 1..d by lambda.  Also the landing histogram over offsets and its
concentration on the first islands.

The null this is measured against (order one, per gear, no interaction): the landing is missed
by every gear, and for one gear the conditional chance that the offset is BARRED rather than
merely missed is |Bar(g)| / (g - 2).  So the order-one prediction is

    P(landing is a B-island)  =  prod_{5<=g<=B} |Bar(g)| / (g - 2),

against the unconditional island density  rho_B = prod |Bar(g)| / g.  Any excess over the first
number is structure the landscape carries beyond order one.

Writes results/rl_landing.txt.
Usage: uv run python research/anchor235/r39/rl_landing.py
"""
import os
from collections import Counter
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "rl_landing.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


QMAX = 20000


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


FL = sieve(QMAX + 10)
GEARS = [p for p in range(5, QMAX + 1) if FL[p]]
NG = len(GEARS)
UU = [pow(6, -1, g) for g in GEARS]
say("gears 5..%d: %d" % (QMAX, NG))

# ---- flat chi table: chi_g(x) = #{x, x+2} that are nonzero QRs mod g, x = -6i mod g
START = [0] * NG
tot = 0
for j, g in enumerate(GEARS):
    START[j] = tot
    tot += g
CHI = np.zeros(tot, dtype=np.int8)
for j, g in enumerate(GEARS):
    qr = np.zeros(g, dtype=bool)
    for t in range(1, (g + 1) // 2):
        qr[(t * t) % g] = True
    CHI[START[j]:START[j] + g] = qr.astype(np.int8) + np.roll(qr, -2).astype(np.int8)
BARSZ = [int((CHI[START[j]:START[j] + GEARS[j]] == 0).sum()) for j in range(NG)]
say("flat chi table built: %d entries (%.1f MB)" % (tot, tot / 1e6))

lam_glob = np.load(os.path.join(OUT, "rl_lambda.npy"))
ISL = {}
for B in (5, 7, 11, 13, 17, 19, 23):
    res = np.load(os.path.join(OUT, "rl_isl_%d.npy" % B))
    mod = int(open(os.path.join(OUT, "rl_isl_%d_mod.txt" % B)).read())
    ISL[B] = (set(int(v) for v in res), mod)


def is_island(i, B):
    res, mod = ISL[B]
    return (i % mod) in res


# ---------------------------------------------------------------- the sweep
rows = []
land_hist = Counter()
for qi, q in enumerate(GEARS):
    nq = qi + 1
    qq = q * q
    POS = 768
    while True:
        struck = bytearray(POS)
        for j in range(nq):
            g = GEARS[j]
            u = UU[j]
            r = qq % g
            a = ((2 - r) * u) % g
            b = ((-r) * u) % g
            for off in (a, b):
                for t in range(off, POS, g):
                    struck[t] = 1
        L = None
        for i in range(1, POS):
            if not struck[i]:
                L = i
                break
        if L is not None:
            break
        POS *= 2
    # d, the forward tooth arc of the top gear
    c = pow(6, -1, q)
    d = (2 * c) % q
    # |G(L)| and lambda(L) restricted to the machine {5..q}
    x6 = -6 * L
    nadm = 0
    lam_q = 0.0
    for j in range(nq):
        g = GEARS[j]
        cc = int(CHI[START[j] + (x6 % g)])
        if cc:
            nadm += 1
            lam_q += 2.0 * cc / (g - 1)
    rows.append((q, L, d, nadm, lam_q))
    land_hist[L] += 1

say("walks: %d;  L min %d, median %d, max %d"
    % (len(rows), min(r[1] for r in rows), int(np.median([r[1] for r in rows])),
       max(r[1] for r in rows)))

# ---------------------------------------------------------------- island status of the landing
say("")
say("=== 3. is the landing an island? measured against the order-one null ===")
say(" B    measured        null prod |Bar|/(g-2)   island density rho_B   enrich(meas/rho)  ratio meas/null")
for B in (5, 7, 11, 13, 17, 19):
    cnt = sum(1 for r in rows if is_island(r[1], B))
    frac = cnt / len(rows)
    null = 1.0
    rho = 1.0
    for j, g in enumerate(GEARS):
        if g > B:
            break
        null *= BARSZ[j] / (g - 2)
        rho *= BARSZ[j] / g
    say("%3d   %5d / %d = %.4f      %.4f                 %.6f              %6.2fx          %.3f"
        % (B, cnt, len(rows), frac, null, rho, frac / rho, frac / null))

say("")
say("=== 3b. the landing offset histogram, top 20 ===")
say("offset  landings   is B=7 island   is B=13 island   lambda(i)")
for i, n in land_hist.most_common(20):
    say("%6d  %8d   %-13s   %-14s   %.4f"
        % (i, n, is_island(i, 7), is_island(i, 13),
           lam_glob[i] if i < len(lam_glob) else float("nan")))
tot_land = sum(land_hist.values())
first4 = sum(land_hist[i] for i in (5, 10, 12, 17))
say("the four smallest B=7 islands 5, 10, 12, 17 take %d of %d landings = %.4f"
    % (first4, tot_land, first4 / tot_land))
mod5_1 = sum(n for i, n in land_hist.items() if i % 5 == 1)
say("landings at offsets = 1 (mod 5): %d  (gear 5 strikes them at every q)" % mod5_1)
say("distinct landing offsets: %d;  largest landing offset %d"
    % (len(land_hist), max(land_hist)))

# ---------------------------------------------------------------- lambda rank of the landing
say("")
say("=== 3c. rank of the landing by lambda among the offsets 1..d ===")
ranks = []
nadm_rel = []
for (q, L, d, nadm, lam_q) in rows:
    hi = min(d, len(lam_glob))
    if hi <= 2:
        continue
    seg = lam_glob[1:hi]
    r = float((seg < lam_glob[L]).sum()) / len(seg)
    ranks.append(r)
    nadm_rel.append(nadm / (sum(1 for g in GEARS if g <= q)))
ranks = np.array(ranks)
say("percentile of lambda(L) among lambda(1..d-1): mean %.4f, median %.4f, "
    "below 0.25 at %d of %d, above 0.75 at %d"
    % (ranks.mean(), np.median(ranks), int((ranks < 0.25).sum()), len(ranks),
       int((ranks > 0.75).sum())))
na = np.array(nadm_rel)
say("|G(L)| as a fraction of the machine: mean %.4f, min %.4f, max %.4f (generic 0.75)"
    % (na.mean(), na.min(), na.max()))

lamL = np.array([lam_glob[r[1]] for r in rows])
say("lambda(L) over the sweep: mean %.4f (mean over all offsets 3.4433), min %.4f, max %.4f"
    % (lamL.mean(), lamL.min(), lamL.max()))

# ---------------------------------------------------------------- by walk length band
say("")
say("=== 3d. island status of the landing by q band ===")
say("  q range      walks   B=5     B=7     B=11    B=13")
bands = [(5, 100), (100, 1000), (1000, 5000), (5000, 20000)]
for lo, hi in bands:
    sub = [r for r in rows if lo <= r[0] < hi]
    if not sub:
        continue
    line = "%6d-%-6d %6d" % (lo, hi, len(sub))
    for B in (5, 7, 11, 13):
        line += "  %.4f" % (sum(1 for r in sub if is_island(r[1], B)) / len(sub))
    say(line)

np.save(os.path.join(OUT, "rl_rows.npy"), np.array([(r[0], r[1], r[2]) for r in rows],
                                                   dtype=np.int64))
say("")
say("saved (q, L, d) rows")
LOG.close()
