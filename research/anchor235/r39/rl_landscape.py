"""R2.a.i.a - THE REACHABILITY LANDSCAPE, part 1: the landscape and its islands.

Column k_0 + i of the walk from q^2 carries q^2 + 6i - 2 and q^2 + 6i, so gear g strikes
offset i iff q^2 = 2 - 6i or -6i (mod g).  For a gear g != q the value q^2 mod g is a NONZERO
square, so with x = -6i mod g:

    g is ADMISSIBLE at offset i   iff   x  or  x + 2  is a nonzero quadratic residue mod g.

That is q-free: which gears can EVER reach an offset is a property of the offset alone.
Bar(g) = the offset classes mod g at which g is barred.  An offset barred by every gear
5 <= g <= B is an ISLAND for bound B.

Computed here:
  A. |Bar(g)| against the closed form (g + 1 - chi(2) - chi(-2))/4 for every gear 5..QMAX.
  B. Bar(5), Bar(7), Bar(11), Bar(13) written out; no gear reaches every offset.
  C. |G(i)| and lambda(i) for i = 0..IMAX over all gears <= QMAX; the extremes.
  D. the offsets admissible for EVERY gear (the -6t^2 family) forward and backward.
  E. the mirror i -> d_g - i (mod g) on Bar(g), by g mod 4.
  F. islands: exact residue sets mod P_B by CRT for B in {5,7,11,13,17,19,23}; counts, density,
     first island past 0, maximal gap, gap spectrum.

Writes results/rl_landscape.txt.
Usage: uv run python research/anchor235/r39/rl_landscape.py
"""
import os
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "rl_landscape.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


QMAX = 20000
IMAX = 20000


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


FL = sieve(QMAX + 10)
GEARS = [p for p in range(5, QMAX + 1) if FL[p]]
say("gears 5..%d: %d;  offsets 0..%d" % (QMAX, len(GEARS), IMAX))


def qr_flags(g):
    """qr[v] = True iff v is a NONZERO quadratic residue mod g."""
    qr = np.zeros(g, dtype=bool)
    for t in range(1, (g + 1) // 2):
        qr[(t * t) % g] = True
    return qr


def bar_mask(g):
    """barred[x] = True iff neither x nor x+2 is a nonzero QR mod g  (x = -6i mod g)."""
    qr = qr_flags(g)
    return ~qr & ~np.roll(qr, -2)


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


# ---------------------------------------------------------------- A. the bar-size closed form
say("")
say("=== A. |Bar(g)| against the closed form (g + 1 - chi(2) - chi(-2))/4 ===")
bad = 0
by8 = {1: [], 3: [], 5: [], 7: []}
barsz = {}
for g in GEARS:
    bm = bar_mask(g)
    n = int(bm.sum())
    barsz[g] = n
    pred = (g + 1 - legendre(2, g) - legendre(-2, g)) // 4
    if n != pred:
        bad += 1
        if bad <= 5:
            say("   MISMATCH g=%d measured %d predicted %d" % (g, n, pred))
    by8[g % 8].append(n / g)
say("gears checked: %d;  mismatches with the closed form: %d" % (len(GEARS), bad))
say("closed form by class: g = 1 mod 8 -> (g-1)/4;  3 mod 8 -> (g+1)/4;"
    "  5 mod 8 -> (g+3)/4;  7 mod 8 -> (g+1)/4")
for c in (1, 3, 5, 7):
    v = np.array(by8[c])
    say("   g = %d mod 8: %5d gears, |Bar|/g mean %.6f (limit 0.25)" % (c, len(v), v.mean()))
say("min |Bar(g)| over all gears: %d (at g = %d);  every gear is barred somewhere: %s"
    % (min(barsz.values()), min(barsz, key=barsz.get), all(v >= 1 for v in barsz.values())))

# ---------------------------------------------------------------- B. the small gears written out
say("")
say("=== B. the bottom of the landscape, gear by gear ===")
for g in (5, 7, 11, 13, 17, 19, 23):
    bm = bar_mask(g)
    u = pow(6, -1, g)
    barred_i = sorted(int(i) for i in range(g) if bm[(-6 * i) % g])
    reach_i = sorted(set(range(g)) - set(barred_i))
    say("gear %2d: barred at i = %s (mod %d)   |Bar| = %d;  reaches i = %s"
        % (g, barred_i, g, len(barred_i), reach_i))

# ---------------------------------------------------------------- C. |G(i)| and lambda(i)
say("")
say("=== C. |G(i)| (gears <= %d admissible at offset i) and lambda(i) ===" % QMAX)
I = np.arange(0, IMAX + 1, dtype=np.int64)
admcount = np.zeros(IMAX + 1, dtype=np.int32)
lam = np.zeros(IMAX + 1, dtype=np.float64)
lam13 = np.zeros(IMAX + 1, dtype=np.float64)      # gears <= 13 only
for g in GEARS:
    qr = qr_flags(g)
    chi = qr.astype(np.int8) + np.roll(qr, -2).astype(np.int8)   # chi_g at x = -6i
    x = (-6 * I) % g
    c = chi[x]
    admcount += (c > 0)
    w = 2.0 * c / (g - 1)
    lam += w
    if g <= 13:
        lam13 += w
say("|G(i)| over i = 1..%d: min %d (i = %d), max %d (i = %d), mean %.2f  (3/4 of %d = %.1f)"
    % (IMAX, admcount[1:].min(), int(np.argmin(admcount[1:])) + 1,
       admcount[1:].max(), int(np.argmax(admcount[1:])) + 1,
       admcount[1:].mean(), len(GEARS), 0.75 * len(GEARS)))
say("lambda(i) over i = 1..%d: min %.4f (i = %d), max %.4f (i = %d), mean %.4f"
    % (IMAX, lam[1:].min(), int(np.argmin(lam[1:])) + 1,
       lam[1:].max(), int(np.argmax(lam[1:])) + 1, lam[1:].mean()))
say("lambda(0) = %.4f;  |G(0)| = %d (predicted: exactly the gears = +-1 mod 8)"
    % (lam[0], admcount[0]))
m8 = sum(1 for g in GEARS if g % 8 in (1, 7))
say("   gears = +-1 mod 8 among the %d gears: %d;  agrees: %s"
    % (len(GEARS), m8, m8 == int(admcount[0])))
lo = np.argsort(lam[1:81])[:8] + 1
hi = np.argsort(lam[1:81])[-8:] + 1
say("eight lowest-lambda offsets in 1..80: %s" % sorted(int(v) for v in lo))
say("eight highest-lambda offsets in 1..80: %s" % sorted(int(v) for v in hi))
say("variance of lambda over i = 1..%d: total %.5f, from gears <= 13 only %.5f (share %.4f)"
    % (IMAX, lam[1:].var(), lam13[1:].var(), lam13[1:].var() / lam[1:].var()))
say("correlation of lambda with lambda restricted to gears <= 13: %.5f"
    % float(np.corrcoef(lam[1:], lam13[1:])[0, 1]))

# ---------------------------------------------------------------- D. offsets every gear reaches
say("")
say("=== D. offsets admissible for EVERY gear <= %d ===" % QMAX)
full = np.flatnonzero(admcount == len(GEARS))
say("forward offsets i in 1..%d admissible for every gear: %d %s"
    % (IMAX, len(full[full >= 1]), list(full[full >= 1][:10])))
# backward: i = -6 t^2
say("backward, the family i = -6 t^2 (member q^2 - 36t^2 = (q-6t)(q+6t)):")
for t in range(1, 9):
    i = -6 * t * t
    barred = []
    for g in GEARS[:200]:
        qr = qr_flags(g)
        x = (-6 * i) % g
        if not (qr[x] or qr[(x + 2) % g]):
            barred.append(g)
    say("   t = %d, i = %6d:  gears <= %d barred: %s   (predicted: g | 6t and g != +-1 mod 8)"
        % (t, i, GEARS[199], barred if barred else "none"))
say("   2 - 6i is never a perfect square (m^2 = 2 mod 3 unsolvable), so no other full family")

# ---------------------------------------------------------------- E. the mirror
say("")
say("=== E. the mirror i -> d_g - i (mod g), d_g = 2 * 6^{-1} ===")
sym1 = sym3 = anti1 = anti3 = 0
for g in GEARS:
    bm = bar_mask(g)
    d = (2 * pow(6, -1, g)) % g
    barred_i = np.array([i for i in range(g) if bm[(-6 * i) % g]])
    img = set(int((d - i) % g) for i in barred_i)
    src = set(int(i) for i in barred_i)
    if g % 4 == 1:
        if img == src:
            sym1 += 1
        else:
            anti1 += 1
    else:
        if img & src:
            anti3 += 1
        else:
            sym3 += 1
say("g = 1 mod 4: Bar(g) preserved by the mirror at %d of %d gears (failures %d)"
    % (sym1, sym1 + anti1, anti1))
say("g = 3 mod 4: Bar(g) mapped entirely OFF itself (into the admissible set) at %d of %d"
    % (sym3, sym3 + anti3))

# ---------------------------------------------------------------- F. islands
say("")
say("=== F. islands: offsets no gear <= B can ever reach ===")


def crt_islands(B):
    """explicit residue list of the island set mod P_B, by iterated CRT."""
    res = [0]
    mod = 1
    for g in GEARS:
        if g > B:
            break
        bm = bar_mask(g)
        bg = [i for i in range(g) if bm[(-6 * i) % g]]
        new = []
        inv = pow(mod, -1, g)
        for r in res:
            for b in bg:
                t = (r + mod * (((b - r) * inv) % g)) % (mod * g)
                new.append(t)
        res = sorted(new)
        mod *= g
    return np.array(res, dtype=np.int64), mod


say(" B     P_B          islands   density      1/density   first island   max gap   min gap")
ISL = {}
for B in (5, 7, 11, 13, 17, 19, 23):
    res, mod = crt_islands(B)
    ISL[B] = (res, mod)
    gaps = np.diff(np.concatenate([res, [res[0] + mod]]))
    first = int(res[res >= 1][0]) if (res >= 1).any() else -1
    say("%3d  %-12d %7d   %.8f  %10.1f  %8d      %8d  %8d"
        % (B, mod, len(res), len(res) / mod, mod / len(res), first,
           int(gaps.max()), int(gaps.min())))
say("")
for B in (5, 7, 11, 13):
    res, mod = ISL[B]
    say("B = %2d: islands mod %d = %s" % (B, mod, list(int(v) for v in res[:60])))
say("")
say("island count = prod |Bar(g)| over 5 <= g <= B, exactly:")
run = 1
for g in (5, 7, 11, 13, 17, 19, 23):
    run *= barsz[g]
    say("   B = %2d: |Bar| = %d, cumulative product %d" % (g, barsz[g], run))

say("")
say("=== F2. gap spectrum between consecutive islands ===")
for B in (7, 11, 13, 17, 19):
    res, mod = ISL[B]
    gaps = np.diff(np.concatenate([res, [res[0] + mod]]))
    say("B = %2d: gaps min %d, median %d, mean %.1f, max %d; distinct gap values %d"
        % (B, gaps.min(), int(np.median(gaps)), gaps.mean(), gaps.max(), len(set(gaps.tolist()))))

say("")
say("=== F3. the first twenty islands past offset 0 ===")
for B in (5, 7, 11, 13, 17, 19):
    res, mod = ISL[B]
    r = res[res >= 1]
    say("B = %2d: %s" % (B, list(int(v) for v in r[:20])))

np.save(os.path.join(OUT, "rl_lambda.npy"), lam)
np.save(os.path.join(OUT, "rl_admcount.npy"), admcount)
for B in (5, 7, 11, 13, 17, 19, 23):
    res, mod = ISL[B]
    np.save(os.path.join(OUT, "rl_isl_%d.npy" % B), res)
    with open(os.path.join(OUT, "rl_isl_%d_mod.txt" % B), "w") as f:
        f.write(str(mod))
say("")
say("saved lambda, |G(i)| and the island residue sets to results/")
LOG.close()
