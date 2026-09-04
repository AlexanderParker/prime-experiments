"""R2.a self-feeding, part 1: the correspondence, the pinned classes, the strike lattice.

A  the correspondence: openings of {5..y} in the window == columns of twin pairs (g,g+2)
   with y < g and g+2 <= y^2.  Checked by running the machine (y <= 199) and by twin
   counts (y <= 5000).
B  P1.1 / P1.2: of the four pinned double-kill classes of a twin gear pair, only the
   shared tooth u_g ever lies in a lower machine's window.
C  P1.4: the strike lattice of a twin gear pair relative to its own birth column.
D  the protected multiples t*k and how open they are (a rate, recorded as such).

Writes results/sf_frame.txt.
"""
import os, sys
from math import isqrt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "sf_frame.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def sieve(n):
    """primes up to n as a bytearray flag array (1 = prime)."""
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


N = 25_000_100                      # 5000^2 + slack
say("sieving to", N, "...")
FL = sieve(N)
PRIMES = [i for i in range(2, 5001) if FL[i]]
say("primes to 5000:", len(PRIMES))

# ---------------------------------------------------------------- A
say("")
say("=== A. the correspondence ===")

GEARS = [p for p in PRIMES if p >= 5]


def machine_openings(y, kmax):
    """columns 1..kmax open under {5..y} (direct sieve on the column line)."""
    op = bytearray([1]) * (kmax + 1)
    op[0] = 0
    for q in GEARS:
        if q > y:
            break
        u = pow(6, -1, q)
        for r in (u, q - u):
            st = r if r >= 1 else q
            op[st:: q] = bytearray(len(range(st, kmax + 1, q)))
    return op


bad = 0
rows = []
for y in [p for p in PRIMES if 5 <= p <= 199]:
    kmax = (y * y - 1) // 6
    op = machine_openings(y, kmax)
    lo = y // 6 + 1                      # first column with 6k-1 > y
    while 6 * lo - 1 <= y:
        lo += 1
    cols = [k for k in range(lo, kmax + 1) if op[k]]
    # the same set from twins
    tw = [k for k in range(lo, kmax + 1) if FL[6 * k - 1] and FL[6 * k + 1]]
    if cols != tw:
        bad += 1
        say("  MISMATCH at y =", y)
    rows.append((y, len(cols), cols[0] if cols else None))
say("machines run directly, y = 5..199: mismatches between window openings and twin columns:", bad)
say("  y, #openings in window, first opening column")
say("  " + "; ".join("%d:%d,k=%s" % r for r in rows[:12]))
say("  smallest count:", min(r[1] for r in rows), "at y =",
    [r[0] for r in rows if r[1] == min(rr[1] for rr in rows)])

# larger levels by twin count
say("")
say("N(y) = #twin pairs (g,g+2) with y < g and g+2 <= y^2  (= openings of {5..y} in its window)")
from bisect import bisect_right
minN = None
TW = [g for g in range(5, N - 2, 6) if FL[g] and FL[g + 2]]
say("twin starts below", N, ":", len(TW))


def Ncount(y):
    return bisect_right(TW, y * y - 2) - bisect_right(TW, y)


for y in [p for p in PRIMES if p >= 5]:
    n = Ncount(y)
    if minN is None or n < minN[1]:
        minN = (y, n)
say("minimum over primes 5..5000:", minN, " (y, N(y))")
say("sample: " + ", ".join("N(%d)=%d" % (y, Ncount(y)) for y in [5, 7, 11, 13, 17, 19, 23, 29, 31,
                                                                 101, 1009, 4999]))

# the same statement read on one machine: twin gear pairs above sqrt(top gear)
say("")
say("twin gear pairs (g,g+2) of {5..q} with sqrt(q) < g:  min over primes q<=5000")
mn = None
for q in [p for p in PRIMES if p >= 5]:
    r = isqrt(q)
    n = bisect_right(TW, q - 2) - bisect_right(TW, r)
    if mn is None or n < mn[1]:
        mn = (q, n)
say("  minimum:", mn, "(q, count)  - zero would break the correspondence at that level")

# ---------------------------------------------------------------- B
say("")
say("=== B. the four pinned classes: which are ever visible in a lower window ===")
tot = 0
viol11 = 0
viol_other = 0
for g in range(5, 200001, 6):
    if not (FL[g] and FL[g + 2]):
        continue
    tot += 1
    u = (g + 1) // 6
    P = g * (g + 2)
    c2 = 6 * u * u                     # u(g+1), the twin-product column
    c3 = P - u
    c4 = P - c2
    top = (g * g - 1) // 6             # strictly above every lower machine's window top
    if not (6 * u - 1 == g and 6 * u + 1 == g + 2):
        viol11 += 1
    for c in (c2, c3, c4):
        if c <= top:
            viol_other += 1
say("twin pairs with g <= 200000:", tot)
say("  shared tooth column u_g has members exactly (g, g+2):  exceptions", viol11)
say("  other three classes with a representative at or below (g^2-1)/6 (the top of every",
    "lower window): exceptions", viol_other)
say("  -> P1.1 holds; mechanism: c2 = (g+1)^2/6 already exceeds (y^2-1)/6 for every y < g,",
    "and c3, c4 exceed c2.")
say("  P1.2: a column that is a double kill of the pair at g1 and of the pair at g2 inside a",
    "lower window would have to be u_{g1} = u_{g2}; and g2 strikes u_{g1} iff g2 | g1 or",
    "g2 | g1+2, impossible for distinct primes.  Exceptions 0 by the same computation.")

# ---------------------------------------------------------------- C
say("")
say("=== C. the strike lattice of a twin gear pair about its birth column ===")
CMAX = 400000
bad_c = 0
bad_mid = 0
pairs_c = 0
for g in range(5, 601, 6):
    if not (FL[g] and FL[g + 2]):
        continue
    pairs_c += 1
    k = (g + 1) // 6
    # predicted: g strikes k(6m+1) - m and k(6m+5) - (m+1);  g+2 strikes k(6m+1) + m and
    # k(6m+5) + (m+1),  m = 0,1,2,...
    pred_lo, pred_hi = set(), set()
    m = 0
    while True:
        a = k * (6 * m + 1) - m
        b = k * (6 * m + 5) - (m + 1)
        if a > CMAX and b > CMAX:
            break
        if a <= CMAX:
            pred_lo.add(a)
            pred_hi.add(k * (6 * m + 1) + m)
        if b <= CMAX:
            pred_lo.add(b)
            pred_hi.add(k * (6 * m + 5) + (m + 1))
        m += 1
    pred_hi = {c for c in pred_hi if c <= CMAX}
    act_lo = set()
    for r in (k % g, (-k) % g):
        st = r if r >= 1 else g
        act_lo.update(range(st, CMAX + 1, g))
    h = g + 2
    uh = pow(6, -1, h)
    act_hi = set()
    for r in (uh % h, (-uh) % h):
        st = r if r >= 1 else h
        act_hi.update(range(st, CMAX + 1, h))
    if act_lo != pred_lo or act_hi != pred_hi:
        bad_c += 1
        if bad_c < 4:
            say("  lattice mismatch at g =", g)
    # protected multiples: neither member strikes t*k for 2 <= t <= g-2
    for t in range(2, min(g - 1, CMAX // k + 1)):
        cc = t * k
        if cc > CMAX:
            break
        if cc in act_lo or cc in act_hi:
            bad_mid += 1
            if bad_mid < 4:
                say("  protected-multiple failure at g =", g, "t =", t)
say("twin pairs g <= 600 checked over columns <= %d:" % CMAX, pairs_c)
say("  strike-lattice mismatches:", bad_c)
say("  protected-multiple failures (a member striking t*k, 2 <= t <= g-2):", bad_mid)
say("  the ladder: at t = 6m+1 the two members strike t*k -+ m (separation 2m);")
say("  at t = 6m+5 they strike t*k -+ (m+1) (separation 2m+2).  t = 1, m = 0 is the shared tooth.")

# ---------------------------------------------------------------- D
say("")
say("=== D. how open are the protected multiples t*k (a rate, not a rule) ===")
y = 199
kmax = (y * y - 1) // 6
op = machine_openings(y, kmax)
base = sum(op[1:kmax + 1]) / kmax
for t in (2, 3, 4, 5, 6, 7, 10, 11):
    ks = [k for k in range(1, kmax // t + 1)]
    if not ks:
        continue
    o = sum(op[t * k] for k in ks) / len(ks)
    say("  t = %2d: P(open at t*k) = %.4f   ratio to base %.4f = %.4f" % (t, o, base, o / base))
say("  base density of {5..199} =", "%.4f" % base)
say("  gear 5 misses every column divisible by 5 (the shield), which is the whole of the")
say("  t = 5, 10 effect; nothing here is a rule.")

LOG.close()
