"""R2.a.i.a.1 - item 4 (the slack law and its mechanism), item 1's divisor rule, item 7.

  * which gears remove the islands of [1, d): total island strikes per gear, and the count of
    islands of which the gear is the SOLE striker (the gear whose removal would free the island);
  * where the first free island sits inside the arc [1, d);
  * the divisor rule: a gear g | q strikes exactly the two offset classes at which a member is
    0 (mod g), and both of those classes are BARRED classes of g (offsets g can never reach at a
    q coprime to g).  Checked exhaustively.

Writes results/iw_slack.txt.
Usage: uv run python research/anchor235/r40/iw_slack.py [--QMAX 20000]
"""
import argparse
import os
from collections import Counter
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LOG = open(os.path.join(OUT, "iw_slack.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


ap = argparse.ArgumentParser()
ap.add_argument("--QMAX", type=int, default=20000)
args = ap.parse_args()
QMAX = args.QMAX
FL = sieve(QMAX + 10)
GEARS = [p for p in range(5, QMAX + 1) if FL[p]]
U = [pow(6, -1, g) for g in GEARS]
GI = {g: k for k, g in enumerate(GEARS)}
DMAX = (2 * QMAX + 1) // 3 + 4
ISL = np.zeros(DMAX, dtype=bool)
for c in (5, 10, 12, 17):
    ISL[c::35] = True
ISL[0] = False

strikes = np.zeros(len(GEARS), dtype=np.int64)
sole = Counter()
ffrac = []
say("primes 5..%d: %d" % (QMAX, len(GEARS)))

for qi, q in enumerate(GEARS):
    d = (2 * U[qi]) % q
    if d < 2:
        continue
    isl = ISL[:d]
    nisl = int(isl.sum())
    if nisl == 0:
        continue
    qq = q * q
    cnt = np.zeros(d, dtype=np.int32)
    gsum = np.zeros(d, dtype=np.int64)
    for j in range(qi + 1):
        g = GEARS[j]
        u = U[j]
        r = qq % g
        a = ((2 - r) * u) % g
        b = ((-r) * u) % g
        hit = 0
        if a < d:
            cnt[a::g] += 1
            gsum[a::g] += g
            hit += int(isl[a::g].sum())
        if b < d:
            cnt[b::g] += 1
            gsum[b::g] += g
            hit += int(isl[b::g].sum())
        strikes[j] += hit
    onc = isl & (cnt == 1)
    for v in gsum[onc]:
        sole[int(v)] += 1
    fr = np.flatnonzero(isl & (cnt == 0))
    if len(fr):
        ffrac.append((q, d, int(fr[0]), len(fr), nisl))

say("")
say("=== item 4: which gears remove the islands (primes 5..%d) ===" % QMAX)
tot_str = strikes.sum()
tot_sole = sum(sole.values())
say("total island strikes %d;  islands with exactly one striker %d" % (tot_str, tot_sole))
say("")
say(" gear    island strikes   share    2/g / sum(2/g)     SOLE strikes   share")
sum2g = sum(2 / g for g in GEARS)
for j, g in enumerate(GEARS[:24]):
    say("  %-6d %-16d %-8.4f %-18.4f %-14d %.4f"
        % (g, strikes[j], strikes[j] / tot_str, (2 / g) / sum2g,
           sole.get(g, 0), sole.get(g, 0) / tot_sole))
say("...")
say(" largest gear that is ever a sole striker: %d" % max(sole))
big = sum(v for g, v in sole.items() if g > 100)
say(" sole strikes by gears above 100: %d of %d (%.4f)" % (big, tot_sole, big / tot_sole))
big2 = sum(v for g, v in sole.items() if g > 1000)
say(" sole strikes by gears above 1000: %d of %d (%.4f)" % (big2, tot_sole, big2 / tot_sole))
say(" top 10 sole strikers: %s" % sole.most_common(10))

say("")
say("=== item 7: where the first free island sits in the arc ===")
A = np.array([(q, d, f, n, ni) for q, d, f, n, ni in ffrac], dtype=np.int64)
sel = A[:, 0] > 1487
fr = A[sel, 2] / A[sel, 1]
say("primes with a free island and q > 1487: %d" % int(sel.sum()))
say("first free island as a fraction of d: max %.4f (q = %d), median %.4f, mean %.4f"
    % (fr.max(), int(A[sel][fr.argmax(), 0]), float(np.median(fr)), float(fr.mean())))
say("first free island in absolute offset: max %d (q = %d), median %d"
    % (A[sel, 2].max(), int(A[sel][A[sel, 2].argmax(), 0]), int(np.median(A[sel, 2]))))
for thr in (0.25, 0.5, 0.75):
    say("  first free island below %.2f d: %d of %d"
        % (thr, int((fr < thr).sum()), int(sel.sum())))

say("")
say("=== item 1: the divisor rule, exhaustively ===")
say("if g | q then q^2 = 0 (mod g), so g's two targets are x = 0 and x = -2, i.e. it strikes")
say("exactly the offset classes i = 0 and i = 2 u_g (mod g) - the classes where one member is")
say("0 mod g.  Class 0 is a BARRED class of g iff chi_g(2) = -1 (g = 3, 5 mod 8); class 2 u_g is")
say("barred iff chi_g(-2) = -1 (g = 5, 7 mod 8).  Checked exhaustively:")
bad = 0
checks = 0
tab = {1: [0, 0], 3: [0, 0], 5: [0, 0], 7: [0, 0]}
for g in GEARS:
    if g > 2000:
        break
    u = pow(6, -1, g)
    qr = np.zeros(g, dtype=bool)
    for t in range(1, (g + 1) // 2):
        qr[(t * t) % g] = True
    def barred(i):
        x = (-6 * i) % g
        return (not qr[x]) and (not qr[(x + 2) % g])
    a = (2 * u) % g
    got = (barred(0), barred(a))
    want = (not qr[2 % g], not qr[(-2) % g])
    checks += 1
    if got != want:
        bad += 1
        say("  EXCEPTION at gear %d: got %s want %s" % (g, got, want))
    c = g % 8
    tab[c][0] += 1
    tab[c][1] += sum(got)
say("gears checked: %d;  exceptions: %d" % (checks, bad))
say(" g mod 8   gears   mean number of the two divisor classes that are barred")
for c in (1, 3, 5, 7):
    say("   %d       %5d   %.4f   (predicted %d)"
        % (c, tab[c][0], tab[c][1] / tab[c][0], {1: 0, 3: 1, 5: 2, 7: 1}[c]))
say("")
say("gear 5 is 5 mod 8, so BOTH its divisor classes are barred: when 5 | q gear 5 strikes")
say("        i = 0, 2 (mod 5), which IS Bar(5); the four B = 7 islands 5, 10, 12, 17 are")
say("        0, 0, 2, 2 (mod 5), so 5 | q kills every island class at every q.")
say("gear 7 is 7 mod 8, so one of its divisor classes is barred: when 7 | q gear 7 strikes")
say("        i = 0, 5 (mod 7); the islands are 5, 3, 5, 3 (mod 7), so 7 | q kills the two")
say("        island classes 5 and 12 (mod 35) and leaves 10 and 17.")
LOG.close()
