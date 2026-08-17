"""Round 3 lateral: the gap-graded Bezout split law, and the complete
overcount formula for the real machine.

THE LAW. For a gear pair q < q' = q + g (both prime >= 5, g even), the split
double-kill class "q kills left, q' kills right" (q | 6k-1, q' | 6k+1) solves
q'b - qa = 2 with member 6k+1 = q'b. Substituting q' = q + g and t = a - b:
gb - qt = 2, so b = 2 g^{-1} (mod q). Closed form for the least representative:

    m0 = (-2 * q^{-1}) mod g          # depends only on q mod g
    b0 = (2 + m0*q) / g               # least positive b, exactly integral
    i  = (q' - b0) * q^{-1} mod 6     # mod-6 alignment step, i in [0,6)
    b* = b0 + i*q
    x  = (q' * b* - 1) / 6            # least k with q|6k-1, q'|6k+1

and the mirror class (q kills right, q' kills left) sits at P - x, P = qq'.
Depth gradation: m0 = 0 iff g | 2, so g = 2 is the UNIQUE gap with b0 = 1
identically (x = u', the twin pin, at depth ~P/(6q)); every other gap has
b0 >= (2+q)/g, i.e. lowest possible split depth ~ P/(6g), reached only when
the alignment i = 0 lands. In-window split count is then pure floor arithmetic.

THE FORMULA. For the real machine (gears 5..y, window [1,K], V = 6K+1):

    overcount = SAME + PAIRSPLIT - CORR
    SAME      = sum_{j>=2} (-1)^j sum_{|S|=j, prod S <= V} mult(prod S)
                (inclusion-exclusion over squarefree gear products; mult = #
                member multiples <= V, pure floor counting)
    PAIRSPLIT = sum over unordered gear pairs of in-window split-class hits,
                each class located by THE LAW above (no CRT, no sieve)
    CORR      = sum_k (omega(l)omega(r) - 1) over slots with both members
                gearful (multi-gear-side overlap; = # extra ordered split
                incidences on cnt>=3 slots; computed by census, reported)

Verifications in this file:
  1. law vs brute CRT, exhaustive, all prime pairs 5 <= q < q' <= 400;
  2. formula vs window array at three real scales y = 53, 101, 211 (exact);
  3. gap table: in-window split supply by gap class - the doubles ledger as a
     functional of the prime gaps below y.

Run: uv run python research/split_gap_law.py
"""
from collections import defaultdict

from tooth_sharing import isprime, uprime, window_metrics, crt2

def primes(a, b):
    return [p for p in range(a, b + 1) if isprime(p)]

# ---------- the law ----------

def split_rep(q, qp):
    """Least k >= 1 with q | 6k-1 and qp | 6k+1, by the gap law (no CRT)."""
    g = qp - q
    m0 = (-2 * pow(q, -1, g)) % g
    b0 = (2 + m0 * q) // g
    i = ((qp - b0) * pow(q, -1, 6)) % 6
    return (qp * (b0 + i * q) - 1) // 6

def splits_in_window(q, qp, K):
    """# slots in [1,K] where {q,qp} split-kill, via the law (both classes)."""
    P = q * qp
    x = split_rep(q, qp)
    total = 0
    for z in (x, P - x):
        if z <= K:
            total += (K - z) // P + 1
    return total

def verify_law(bound=400):
    print("=" * 72)
    print(f"PART 1: law vs brute CRT, all prime pairs 5 <= q < q' <= {bound}")
    ps = primes(5, bound)
    checked = fails = 0
    for a, q in enumerate(ps):
        uq = pow(6, -1, q)
        for qp in ps[a + 1:]:
            uqp = pow(6, -1, qp)
            # brute: (L,R) class = CRT(left tooth of q, right tooth of q')
            x_brute = crt2(uq % q, q, (qp - uqp) % qp, qp)
            x_law = split_rep(q, qp)
            mirror = crt2((q - uq) % q, q, uqp % qp, qp)
            checked += 1
            if x_law != x_brute or mirror != (q * qp - x_brute) % (q * qp):
                fails += 1
                if fails < 5:
                    print(f"  FAIL ({q},{qp}): law {x_law} brute {x_brute}")
    print(f"  {checked} pairs checked: law == CRT and mirror == P - x. fails {fails}")
    # depth gradation exhibits
    print("  depth exhibits (x/P, law-predicted floor ~ (m0/g + i)/6):")
    for q, qp in [(101, 103), (97, 101), (101, 107), (89, 101), (101, 113)]:
        g = qp - q
        m0 = (-2 * pow(q, -1, g)) % g
        x = split_rep(q, qp)
        print(f"    ({q},{qp}) g={g:>2}: m0={m0:>2}  x={x:>5}  x/P={x/(q*qp):.4f}")

# ---------- the formula ----------

def count_res6(N, r):
    """#{1 <= m <= N : m = r mod 6}, r in [0,5]."""
    if N <= 0:
        return 0
    if r == 0:
        return N // 6
    return (N - r) // 6 + 1 if r <= N else 0

def mult_members(s, K):
    """# member multiples of s in the window: left v=6k-1<=6K-1, right v=6k+1<=6K+1."""
    sm = s % 6                                   # s = +-1 mod 6, self-inverse
    left = count_res6((6 * K - 1) // s, (5 * sm) % 6)
    right = count_res6((6 * K + 1) // s, sm)
    return left + right

def same_formula(gears, K):
    """Inclusion-exclusion over squarefree products of >= 2 gears (DFS, cap V)."""
    V = 6 * K + 1
    total = 0
    n = len(gears)

    def dfs(idx, prod, j):
        nonlocal total
        for t in range(idx, n):
            np_ = prod * gears[t]
            if np_ > V:
                break                            # gears sorted ascending
            if j + 1 >= 2:
                total += (-1) ** (j + 1) * mult_members(np_, K)
            dfs(t + 1, np_, j + 1)
    dfs(0, 1, 0)
    return total

def census(gears, K):
    """Ground truth by divisor enumeration: SAME, B, CORR, overcount, omega data."""
    V = 6 * K + 1
    oml = defaultdict(int)                       # slot -> omega(left member)
    omr = defaultdict(int)
    for q in gears:
        for m in range(q, V + 1, q):
            r = m % 6
            if r == 5 and m <= 6 * K - 1:
                oml[(m + 1) // 6] += 1
            elif r == 1 and m >= 7:
                omr[(m - 1) // 6] += 1
    marks = sum(oml.values()) + sum(omr.values())
    killed = set(oml) | set(omr)
    same = sum(v - 1 for v in oml.values()) + sum(v - 1 for v in omr.values())
    both = [k for k in oml if k in omr]
    pairinc = sum(oml[k] * omr[k] for k in both)  # ordered split incidences
    corr = sum(oml[k] * omr[k] - 1 for k in both)
    lone = sum(1 for k in killed if oml.get(k, 0) + omr.get(k, 0) == 1)
    return dict(marks=marks, overcount=marks - len(killed), same=same,
                B=len(both), pairinc=pairinc, corr=corr, lone=lone)

def formula_test(y):
    gears = primes(5, y)
    K = (y * y - 1) // 6
    cen = census(gears, K)
    arr = window_metrics([(q, (pow(6, -1, q), q - pow(6, -1, q))) for q in gears], K)
    same_f = same_formula(gears, K)
    pairsplit = sum(splits_in_window(gears[i], gears[j], K)
                    for i in range(len(gears)) for j in range(i + 1, len(gears)))
    total = same_f + pairsplit - cen['corr']
    ok1 = "OK" if same_f == cen['same'] else "MISMATCH"
    ok2 = "OK" if pairsplit == cen['pairinc'] else "MISMATCH"
    ok3 = "OK" if total == arr['overcount'] == cen['overcount'] else "MISMATCH"
    print(f"  y={y:>4} K={K:>6} gears={len(gears):>3}: "
          f"SAME formula {same_f} vs census {cen['same']} [{ok1}]; "
          f"PAIRSPLIT law {pairsplit} vs census {cen['pairinc']} [{ok2}]")
    print(f"          overcount = {same_f} + {pairsplit} - CORR {cen['corr']} "
          f"= {total} vs window array {arr['overcount']} [{ok3}]")
    return gears, K

def gap_table(y):
    gears = primes(5, y)
    K = (y * y - 1) // 6
    by_gap = defaultdict(lambda: [0, 0, 0])      # gap -> [pairs, splits, pairs_with_hit]
    big = defaultdict(lambda: [0, 0])            # same, restricted to P > K
    for i in range(len(gears)):
        for j in range(i + 1, len(gears)):
            q, qp = gears[i], gears[j]
            g = qp - q
            s = splits_in_window(q, qp, K)
            key = g if g <= 12 else 99
            by_gap[key][0] += 1
            by_gap[key][1] += s
            by_gap[key][2] += (s > 0)
            if q * qp > K:
                big[key][0] += 1
                big[key][1] += (s > 0)
    print(f"  y={y}, K={K}: in-window split supply by gap class "
          f"(99 = gaps > 12; 'big' = pairs with period P > K):")
    print(f"    {'gap':>4} {'pairs':>6} {'splits':>7} {'mean':>6} {'hit%':>6} "
          f"{'big pairs':>9} {'big hit%':>8}")
    for g in sorted(by_gap):
        n, s, h = by_gap[g]
        bn, bh = big[g]
        print(f"    {g:>4} {n:>6} {s:>7} {s/n:>6.2f} {100*h/n:>5.1f}% "
              f"{bn:>9} {(100*bh/bn if bn else 0):>7.1f}%")

if __name__ == "__main__":
    verify_law()
    print("=" * 72)
    print("PART 2: complete overcount formula, three real scales (exact checks)")
    for y in (53, 101, 211):
        formula_test(y)
    print("=" * 72)
    print("PART 3: the doubles ledger as a functional of the prime gaps below y")
    for y in (101, 211):
        gap_table(y)
