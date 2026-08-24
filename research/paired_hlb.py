"""Harvester round 21 (c): N5 - PAIRED-HOLT EIGEN-ANALYSIS and the
HARDY-LITTLEWOOD-B ANALOGUE IN PAIRED CYCLES.

Holt (arXiv:1510.00743 Thm 5.5) proves that in the ONE-residue cycle of gaps,
fixed-gap populations converge to Hardy-Littlewood Conjecture-B ratios.  This
script establishes the PAIRED (two-residue / twin-slot) analogue in four exact
pieces:

1. LOCAL FACTOR IDENTITY (proved, two lines; asserted here at scale):
   for prime q >= 5 with twin teeth T = {u, -u}, u = 6^{-1} mod q,
       c_q(g) := #{r in Z_q : r, r+g not in T}  =  q - nu_q(H_g),
   H_g = {0, 2, 6g, 6g+2},  nu_q = # distinct residues of H_g mod q.
   THE MACHINE'S TRANSFER DIAGONAL IS THE HARDY-LITTLEWOOD PRIME-QUADRUPLET
   LOCAL FACTOR: c_q = q-2 iff q | 6g (pair collision), q-3 iff q | 6g+-2,
   q-4 generic.  Proof: forbidden r-values {u,-u,u-g,-u-g}; multiply by the
   unit -6 and shift by +1: {0, 2, 6g, 6g+2}; affine maps preserve
   distinctness.

2. THE PINCH THEOREM (paired HL-B in cycles - proved via the depth-sum
   identity, verified exactly here):  with N2(g) = prod_q c_q(g) (pairs of
   openings at lag g, exact by CRT) and N3(0,t,g) = prod_q c_q({0,t,g}),
       N2(g) - sum_{t=1}^{g-1} N3(0,t,g)  <=  n_g(M)  <=  N2(g),
   because sum_j W_j(g) = N2(g) (depth-sum identity, Lateral r20, proved) and
   every j>=2 window has an interior opening at some t (union bound).  Since
   N3/N2 ~ prod (q-6)/(q-4) -> 0 like 1/log^2 y, the CONSECUTIVE-gap
   populations converge to the HL quadruplet ratios:
       n_g / n_g'  ->  S(g)/S(g'),   S(g) = prod_{5<=q<=6g+2} (q-nu_q(H_g))/(q-4)
   (a FINITE product - factors are 1 beyond q = 6g+2), with explicit rate.

3. CONVERGENCE TABLE (pure CRT products, no sieve): the pinch interval around
   S(g)/S(g') at y = 10^2..10^6.

4. EIGEN-ANALYSIS (the round's dynamical result): aggregated by (sum s,
   length j), the paired transfer is generically
       m_{s,j}(M+q') = (q'-2j-2) m_{s,j}(M) + 2j m_{s,j+1}(M) + sporadic,
   the exact two-residue analogue of Holt's (p-j-1, superdiag j).  The matrix
   diag(q-2j-2) + superdiag(2j) has eigenvector matrix
       v^(k)_j = (-1)^(k-j) C(k-1, j-1)   - PASCAL, q-INDEPENDENT, and
   IDENTICAL to the eigenvectors of Holt's one-residue matrix: the paired
   system is Holt's system with doubled level spacing.  Verified in exact
   rational arithmetic.  Also verified: the EXACT word-level transfer
   n_w(M+q') = sum_W n_W(M) #{r : image(W,r) = w}  (image = deterministic
   fusion of W in copy r) - every word count, two rungs, exact.
"""
import numpy as np
from math import prod, comb, log
from fractions import Fraction
from collections import Counter
from sympy import primerange

# ---------------------------------------------------------------- 1. identity
print("=" * 78)
print("1. LOCAL FACTOR IDENTITY  c_q(g) = q - nu_q({0,2,6g,6g+2})")
print("=" * 78)
for q in primerange(5, 2000):
    u = pow(6, -1, q)
    T = {u, q - u}
    for g in list(range(1, 60)) + [q - 1, q, q + 1, 5 * q, 3 * q + 7]:
        forb = {u % q, (-u) % q, (u - g) % q, (-u - g) % q}
        c = q - len(forb)
        nu = len({0 % q, 2 % q, (6 * g) % q, (6 * g + 2) % q})
        assert c == q - nu, (q, g, c, nu)
        expect = q - 2 if g % q == 0 else (q - 3 if (6 * g % q in (2, q - 2)) else q - 4)
        assert c == expect, (q, g)
print("  exact for all primes 5 <= q < 2000, g in 1..59 and boundary cases:")
print("  c_q = q-2 iff q | 6g,  q-3 iff q | 6g -+ 2,  q-4 otherwise")
print("  = the Hardy-Littlewood local factor of the quadruple (p, p+2, p+6g, p+6g+2)")

# ------------------------------------------------- 2. pinch, verified by sieve
print()
print("=" * 78)
print("2. PINCH  N2 - sum N3 <= n_g <= N2  (exact sieves, slot machines 13/17/19)")
print("=" * 78)


def slot_openings(gears, P):
    a = np.ones(P, bool)
    for q in gears:
        u = pow(6, -1, q)
        a[u % q::q] = False
        a[(-u) % q::q] = False
    return np.flatnonzero(a)


def cq_set(q, X):
    """c_q(X) = #{r : r+x not in T for all x in X}"""
    u = pow(6, -1, q)
    T = {u % q, (-u) % q}
    forb = {(t - x) % q for t in T for x in X}
    return q - len(forb)


GCAP = 26
for gears in ([5, 7, 11, 13], [5, 7, 11, 13, 17], [5, 7, 11, 13, 17, 19]):
    P = prod(gears)
    idx = slot_openings(gears, P)
    gaps = np.diff(np.append(idx, idx[0] + P))
    hist = Counter(int(g) for g in gaps)
    worst = 0.0
    for g in range(1, GCAP + 1):
        N2 = prod(cq_set(q, (0, g)) for q in gears)
        N3sum = sum(prod(cq_set(q, (0, t, g)) for q in gears) for t in range(1, g))
        n = hist.get(g, 0)
        assert n <= N2, (gears, g)
        assert n >= N2 - N3sum, (gears, g, n, N2, N3sum)
        if N2:
            worst = max(worst, N3sum / N2)
    print(f"  machine {gears[-1]:>2} (P={P:>9,}): pinch holds for every g <= {GCAP}; "
          f"max correction sum N3/N2 = {worst:.3f}")

# ------------------------------------------------- 3. convergence table
print()
print("=" * 78)
print("3. CONVERGENCE OF n_g / n_g' TO THE HL QUADRUPLET RATIO (CRT products)")
print("=" * 78)


def S_fin(g):
    """finite singular product prod_{5<=q<=6g+2} c_q(g)/(q-4)"""
    r = 1.0
    for q in primerange(5, 6 * g + 3):
        r *= cq_set(q, (0, g)) / (q - 4)
    return r


PAIRS = [(5, 4), (7, 6), (10, 9), (12, 11)]
YS = [100, 1000, 10 ** 4, 10 ** 5, 10 ** 6]
GAPS = sorted({g for p in PAIRS for g in p})
# one incremental pass over primes; snapshot the pinch data at each checkpoint
state = {}
for g in GAPS:
    state[g] = {"logR": 0.0,                        # log prod c_q(0,g)/(q-4)
                "logT": {t: 0.0 for t in range(1, g)}}   # log prod c(0,t,g)/c(0,g)
snap = {y: {} for y in YS}
yi = 0
for q in primerange(5, YS[-1] + 1):
    while yi < len(YS) and q > YS[yi]:
        for g in GAPS:
            corr = sum(np.exp(v) for v in state[g]["logT"].values()
                       if v != -np.inf)
            snap[YS[yi]][g] = (state[g]["logR"], corr)
        yi += 1
    u = pow(6, -1, q)
    T = (u, q - u)
    for g in GAPS:
        c2 = q - len({(t - x) % q for t in T for x in (0, g)})
        state[g]["logR"] += log(c2) - log(q - 4)
        for t in state[g]["logT"]:
            if state[g]["logT"][t] == -np.inf:
                continue
            c3 = q - len({(tt - x) % q for tt in T for x in (0, t, g)})
            state[g]["logT"][t] = (-np.inf if c3 == 0
                                   else state[g]["logT"][t] + log(c3) - log(c2))
while yi < len(YS):
    for g in GAPS:
        corr = sum(np.exp(v) for v in state[g]["logT"].values() if v != -np.inf)
        snap[YS[yi]][g] = (state[g]["logR"], corr)
    yi += 1

for g, g2 in PAIRS:
    target = S_fin(g) / S_fin(g2)
    row = []
    for y in YS:
        lR, cg = snap[y][g]
        lR2, cg2 = snap[y][g2]
        rat = np.exp(lR - lR2)
        lo = rat * (1 - cg)
        hi = rat / (1 - cg2) if cg2 < 1 else float("inf")
        row.append((lo, hi, cg))
        assert lo <= target * (1 + 1e-9) and hi >= target * (1 - 1e-9), (g, g2, y)
    cells = "  ".join(f"[{max(lo,0):5.2f},{min(hi,99):5.2f}]" for lo, hi, _ in row)
    print(f"  n_{g}/n_{g2} -> {target:6.3f}:  y=1e2..1e6: {cells}")
print(f"  correction sums 1 - n_g/N2 <= sum_t N3/N2 at y = 1e2..1e6, per gap:")
for g in GAPS:
    print(f"    g={g:>2}: " + "  ".join(f"{snap[y][g][1]:7.3f}" for y in YS))
print("  the interval closes onto the HL quadruplet ratio like 1/log^2 y;")
print("  N2 ratios EQUAL the finite HL product exactly once y > 6g+2 (cancellation)")

# ------------------------------------------------- 4a. exact word-level transfer
print()
print("=" * 78)
print("4a. EXACT WORD-LEVEL TRANSFER (image enumeration), rungs +17 and +19")
print("=" * 78)
SCAP = 24


def word_counts(gears, P, scap):
    idx = slot_openings(gears, P)
    n = len(idx)
    ext = np.append(idx, idx[:64] + P)
    cnt = Counter()
    for i in range(n):
        s = 0
        for j in range(i + 1, i + 64):
            s = int(ext[j] - ext[i])
            if s > scap:
                break
            cnt[tuple(int(ext[k + 1] - ext[k]) for k in range(i, j))] += 1
    return cnt


def transfer(cnt_old, q):
    u = pow(6, -1, q)
    T = {u % q, (-u) % q}
    out = Counter()
    for W, n in cnt_old.items():
        sig = [0]
        for g in W:
            sig.append(sig[-1] + g)
        for r in range(q):
            if (r + sig[0]) % q in T or (r + sig[-1]) % q in T:
                continue                      # an end dies: absorbed elsewhere
            alive = [p for p in sig if (r + p) % q not in T]
            w = tuple(alive[i + 1] - alive[i] for i in range(len(alive) - 1))
            out[w] += n
    return out


old = word_counts([5, 7, 11, 13], 5005, SCAP)
newc = word_counts([5, 7, 11, 13, 17], 85085, SCAP)
pred = transfer(old, 17)
keys = {w for w in set(newc) | set(pred) if sum(w) <= SCAP}
assert all(pred[w] == newc[w] for w in keys)
print(f"  [5,7,11,13] -> +17: {len(keys)} distinct words (sum <= {SCAP}), ALL EXACT")
old2, new2 = newc, word_counts([5, 7, 11, 13, 17, 19], 1616615, SCAP)
pred2 = transfer(old2, 19)
keys2 = {w for w in set(new2) | set(pred2) if sum(w) <= SCAP}
assert all(pred2[w] == new2[w] for w in keys2)
print(f"  [..17]      -> +19: {len(keys2)} distinct words (sum <= {SCAP}), ALL EXACT")

# ------------------------------------------------- 4b. aggregated law + sporadic
print()
print("=" * 78)
print("4b. AGGREGATED LAW  m_(s,j) -> (q-2j-2) m_(s,j) + 2j m_(s,j+1) + sporadic")
print("=" * 78)


def aggregate(cnt, scap):
    m = Counter()
    for w, n in cnt.items():
        if sum(w) <= scap:
            m[(sum(w), len(w))] += n
    return m


mo, mn = aggregate(old, SCAP), aggregate(newc, SCAP)
q = 17
tot_dev, tot = 0, 0
for (s, j), v in sorted(mn.items()):
    gen = (q - 2 * j - 2) * mo.get((s, j), 0) + 2 * j * mo.get((s, j + 1), 0)
    tot_dev += abs(v - gen)
    tot += v
print(f"  rung +17: sum_|sporadic| / sum m = {tot_dev}/{tot} = {tot_dev/tot:.4f}")
print(f"  (sporadic = the finitely many residue coincidences mod q'; the exact")
print(f"   transfer in 4a carries them; the generic part is Holt-shaped)")

# ------------------------------------------------- 4c. Pascal eigenvectors
print()
print("=" * 78)
print("4c. PASCAL EIGENVECTORS, q-INDEPENDENT, SHARED WITH HOLT'S MATRIX")
print("=" * 78)
K = 12
for q in (17, 19, 101, 997):
    for paired in (True, False):
        A = [[Fraction(0)] * K for _ in range(K)]
        for j in range(1, K + 1):
            A[j - 1][j - 1] = Fraction(q - 2 * j - 2 if paired else q - j - 1)
            if j < K:
                A[j - 1][j] = Fraction(2 * j if paired else j)
        for k in range(1, K + 1):
            v = [Fraction((-1) ** (k - j) * comb(k - 1, j - 1)) for j in range(1, K + 1)]
            lam = Fraction(q - 2 * k - 2 if paired else q - k - 1)
            Av = [sum(A[i][j] * v[j] for j in range(K)) for i in range(K)]
            assert Av == [lam * x for x in v], (q, k, paired)
print(f"  v^(k)_j = (-1)^(k-j) C(k-1,j-1) is an exact eigenvector (eigenvalue")
print(f"  q-2k-2 paired / q-k-1 one-residue) for q in {{17,19,101,997}}, k <= {K}:")
print(f"  SAME eigenvector matrix (inverse Pascal), paired = doubled spacing.")
print(f"  Normalised spectra: (q-2j-2)/(q-2) vs Holt's (q-j-1)/(q-2).")

print("\nALL ASSERTIONS PASSED.")
