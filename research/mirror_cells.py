"""LATERAL round 25: THE MIRROR LAW AS A STRUCTURAL CONSTRAINT, and the
gear-p CELL DECOMPOSITION of the gap histogram's Fourier transform.

THE MIRROR.  Slot k is blocked iff some gear q has 6k = -+1 (mod q), a condition
invariant under k -> -k, so the OPENING SET IS EXACTLY CLOSED UNDER k -> -k.
k = 0 is always an opening and P = prod q is odd, so 0 is the ONLY fixed slot.
Mirror therefore acts on every derived object; part A works out what it forces.

THE CELLS.  Fix a gear p.  Openings avoid the two teeth of p, so they live on
the exposed set A_p (|A_p| = p-2), and p-2 consecutive exposed phases span
exactly p slots.  Hence for zeta = e(1/p),

    zeta^{gap} depends ONLY on (start phase i, n mod (p-2)),

n = the number of exposed p-phases the gap crosses (the EXPOSED-STEP count of
item 38).  The (p-2) x (p-2) integer CELL MATRIX M[i][s] therefore determines
the whole frequency-1/p transform of the gap histogram.  Mirror pairs the
cells and CRT fixes the row sums; at p = 5 that leaves THREE free integers.

Parts:
  A  mirror involution, fixed points, and the WINDOW PARITY LAW
  B  adjacent gaps and gap words under mirror (the round-25 brief's question)
  C  the gear-5 cell matrix; the exact relation 2(N_1-N_4) = N_2-N_3
  D  H_5(1) from three integers; the two integer conditions for the pole phase
     126 deg; THE PARITY THEOREM (126 deg is exactly unattainable)
  E  gear 7 (backlog U3): cells, mirror orbits, GF(2) parity test, measured
     bracket asymmetries
  F  backlog U2: the amplitude near-law |H_5(1)|/H0 * meangap = 1.015 - its
     exact cell form, the phase-blind step model, a CORRIDOR-RENEWAL LADDER
     (moduli 5, 35, 385, 5005, ...) and the asymptotic prediction

Exact integers throughout; every float is labelled.  Assertion-gated.
Inputs: data/gap_pair_hist.csv (Mechanic census) and data/spiral_<y>.json
(written by research/spiral29.py this round).

Usage: python mirror_cells.py [--parts ABCDEF] [--maxy 19]
"""
import os, sys, cmath, math, json
from math import prod, pi, sqrt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PHI = (1 + sqrt(5)) / 2
OMEGA = cmath.exp(2j * pi / 5)
IMCO = 2 * math.sin(math.radians(36)) + math.sin(math.radians(72))


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def openings(y):
    gears = primes(5, y)
    P = prod(gears)
    killed = np.zeros(P, dtype=bool)
    for q in gears:
        u = pow(6, -1, q)
        for t in (u, q - u):
            killed[t % q::q] = True
    o = np.flatnonzero(~killed).astype(np.int64)
    assert o[0] == 0 and o.size == prod(q - 2 for q in gears)
    return o, P


def gaps_of(o, P):
    g = (np.roll(o, -1) - o) % P
    return np.where(g == 0, P, g).astype(np.int64)


def deg(z):
    return math.degrees(cmath.phase(z))


def exposed(p):
    u = pow(6, -1, p)
    t = {u % p, (-u) % p}
    return [r for r in range(p) if r not in t]


def delta_table(p):
    """Delta[i][s] = slot advance from the i-th exposed phase over s exposed
    steps, mod p (s = 0..p-3).  p-2 exposed steps == p slots exactly."""
    A = exposed(p)
    n = len(A)
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        cur, adv = A[i], 0
        for s in range(n):
            D[i][s] = adv % p
            step = 1
            while (cur + step) % p not in A:
                step += 1
            adv += step
            cur = (cur + step) % p
        assert adv == p, (p, i, adv)
    return A, D


def load_ghist(close_cycle=True):
    """machine -> (coverage, {gap: count}) from the shared census file.

    REPAIR (found by this round's parity law, 2026-08-29): every FULL-PERIOD
    ghist row in data/gap_pair_hist.csv carries N-1 gaps, not N = prod(q-2) -
    the census closed the period linearly and dropped the WRAP-AROUND gap.  Its
    size is forced: P - sum(g*count), and it equals k_1, the first opening
    (3,3,5,5,5,7,7 at m11..31).  Relative error 1e-9, harmless for densities,
    but it breaks every exact integer identity; restored here.
    """
    best = {}
    with open(os.path.join(DATA, "gap_pair_hist.csv")) as f:
        next(f)
        for line in f:
            yy, cov, kind, idx, v, c = line.strip().split(",")
            if kind != "ghist":
                continue
            y, cov = int(yy), float(cov)
            if y not in best or cov > best[y][0]:
                best[y] = (cov, {})
            if abs(cov - best[y][0]) < 1e-12:
                d = best[y][1]
                d[int(v)] = d.get(int(v), 0) + int(c)
    if close_cycle:
        for y, (cov, h) in best.items():
            if abs(cov - 1.0) > 1e-12:
                continue
            gears = primes(5, y)
            P, N = prod(gears), prod(q - 2 for q in gears)
            miss = P - sum(v * c for v, c in h.items())
            if sum(h.values()) == N - 1 and 1 <= miss <= max(h):
                h[miss] = h.get(miss, 0) + 1
            assert sum(h.values()) == N and \
                sum(v * c for v, c in h.items()) == P, y
    return best


def classes5(h):
    N = [0] * 5
    for v, c in h.items():
        N[v % 5] += c
    return N


# ------------------------------------------------------------------ part A
def partA(ys, Jmax=12):
    print("=" * 78)
    print("PART A: the mirror involution and the WINDOW PARITY LAW")
    print("  claim 1: the opening set is exactly closed under k -> -k, and 0 is")
    print("           its ONLY fixed point (P odd).")
    print("  claim 2: mirror sends the depth-j window starting at opening index")
    print("           t to the one starting at N-t-j, so there is EXACTLY ONE")
    print("           fixed window per depth (2t = -j mod N, N odd).  Hence")
    print("           W_j(g) is EVEN for every g except the single length g_j*")
    print("           of that window, where it is ODD.")
    for y in ys:
        o, P = openings(y)
        N = o.size
        S = set(int(x) for x in o)
        assert all((-int(x)) % P in S for x in o)
        assert N % 2 == 1
        inv2 = pow(2, -1, N)
        rows = []
        for j in range(1, Jmax + 1):
            g = (np.roll(o, -j) - o) % P
            g = np.where(g == 0, P, g)
            hc = np.bincount(g)
            odd = np.flatnonzero(hc % 2 == 1)
            t = (-j * inv2) % N
            assert (-int(o[(t + j) % N])) % P == int(o[t]), (y, j)
            gstar = int((int(o[(t + j) % N]) - int(o[t])) % P) or P
            assert odd.size == 1 and int(odd[0]) == gstar, (y, j, odd, gstar)
            rows.append((j, gstar, int(hc[gstar])))
        g1 = gaps_of(o, P)
        F = int(g1.max())
        h1 = np.bincount(g1)
        print(f"  machine {y}: N = {N}, P = {P}  -  unique odd class per depth "
              f"CONFIRMED for j = 1..{Jmax}")
        print(f"    (j, g_j*, W_j(g_j*)): "
              + ", ".join(f"({j},{g},{c})" for j, g, c in rows[:6]) + " ...")
        print(f"    F = {F}, W_1(F) = {int(h1[F])} "
              f"({'ODD' if h1[F] % 2 else 'EVEN'}); antipodal gap g_1* = "
              f"{rows[0][1]}, first opening k_1 = {int(o[1])}")
    print("  ASSERT ok: mirror closure; exactly one odd class per depth; the")
    print("  odd class always sits at the length of the window predicted by")
    print("  t = -j/2 (mod N).")


# ------------------------------------------------------------------ part B
def partB(ys, jmax=4):
    print("=" * 78)
    print("PART B: ADJACENT GAPS UNDER MIRROR (the round-25 brief's question)")
    print("  Mirror sends the gap word (g_1..g_j) at openings (k_0..k_j) to the")
    print("  REVERSED word at (-k_j..-k_0).  Its fixed points are the words")
    print("  centred on slot 0 (j even) or on the antipode (j odd) - exactly one")
    print("  per depth.  Consequences: the j-gap word census is EXACTLY")
    print("  reverse-symmetric, and every PALINDROMIC word count is EVEN except")
    print("  that one word.")
    for y in ys:
        o, P = openings(y)
        g = gaps_of(o, P)
        B = int(g.max()) + 1
        for j in range(2, jmax + 1):
            code = np.zeros(g.size, dtype=np.int64)
            rcode = np.zeros(g.size, dtype=np.int64)
            for t in range(j):
                code = code * B + np.roll(g, -t)
                rcode = rcode * B + np.roll(g, -(j - 1 - t))
            u, cnt = np.unique(code, return_counts=True)
            ru, rcnt = np.unique(rcode, return_counts=True)
            assert np.array_equal(u, ru) and np.array_equal(cnt, rcnt), (y, j)
            # palindromes: code == its own reverse-code
            pal = code[code == rcode]
            pu, pc = np.unique(pal, return_counts=True)
            oddp = pu[pc % 2 == 1]
            assert oddp.size == 1, (y, j, oddp)
            w = []
            v = int(oddp[0])
            for _ in range(j):
                w.append(v % B)
                v //= B
            w = tuple(reversed(w))
            if j == 2:
                k1 = int(o[1])
                assert w == (k1, k1), (y, w, k1)
                print(f"  machine {y}: W_2 census exactly symmetric "
                      f"({u.size} words); unique odd palindrome ({k1},{k1}) = "
                      f"the pair flanking slot 0")
            else:
                print(f"    depth {j}: {u.size} words, exactly "
                      f"reverse-symmetric; unique odd palindrome {w}")
        F = int(g.max())
        f2 = int((g + np.roll(g, -1)).max())
        n_ff = int(((g == F) & (np.roll(g, -1) == F)).sum())
        print(f"    F = {F}, F_2 = {f2}, #adjacent (F,F) pairs = {n_ff} "
              f"({'EVEN' if n_ff % 2 == 0 else 'ODD'})")
    print("  ASSERT ok: exact reverse symmetry of the gap-word census and the")
    print("  unique-odd-palindrome law at every tested depth and machine.")
    print("  COROLLARY FOR THE TWO-GAP LAW: any adjacent configuration with")
    print("  g_1 = g_2 - in particular an (F,F) pair realising F_2 = 2F - occurs")
    print("  an EVEN number of times, because the one self-mirror adjacent pair")
    print("  is (k_1,k_1) and k_1 < F at every machine.  So a counting argument")
    print("  that caps such configurations at ONE proves there are NONE.")


# ------------------------------------------------------------------ part C
def cells5(o, P):
    g = gaps_of(o, P)
    sr = (o % 5).astype(np.int64)
    gr = (g % 5).astype(np.int64)
    return np.bincount(sr * 5 + gr, minlength=25).reshape(5, 5), g


def partC(ys):
    print("=" * 78)
    print("PART C: the gear-5 CELL MATRIX and the exact mirror relation")
    A, D = delta_table(5)
    assert A == [0, 2, 3] and D == [[0, 2, 3], [0, 1, 3], [0, 2, 4]], (A, D)
    print(f"  A_5 = {A}, Delta[i][s] = {D}: 3 exposed steps = 5 slots, so the")
    print("  gap residue mod 5 is a function of (start phase, n mod 3).")
    print("  Mirror maps cell (i,s) to (-(phase_i+Delta), s), giving")
    print("      T[0][2]=T[3][2],  T[0][3]=T[2][3],  T[2][0]=T[3][0];")
    print("  CRT gives row sums N/3.  Six orbits minus three row constraints =")
    print("  THREE FREE INTEGERS (e,b,c) = (T[2][0], T[0][2], T[0][3]).")
    for y in ys:
        o, P = openings(y)
        N = o.size
        T, g = cells5(o, P)
        assert (T[1] == 0).all() and (T[4] == 0).all()
        for r in (0, 2, 3):
            assert int(T[r].sum()) == N // 3, (y, r)
        assert T[0][2] == T[3][2] and T[0][3] == T[2][3] and T[2][0] == T[3][0]
        Nc = [int(T[:, r].sum()) for r in range(5)]
        e, b, c, a = int(T[2][0]), int(T[0][2]), int(T[0][3]), int(T[0][0])
        assert a == N // 3 - b - c
        assert Nc[0] == a + 2 * e and Nc[2] == 2 * b and Nc[3] == 2 * c
        assert Nc[1] == N // 3 - e - c and Nc[4] == N // 3 - e - b
        assert 2 * (Nc[1] - Nc[4]) == Nc[2] - Nc[3]
        print(f"  machine {y}: N={N}  (e,b,c)=({e},{b},{c})  a={a}  "
              f"N0..N4 = {Nc}")
    print("  ASSERT ok: mirror equalities, row sums, the derived forms of")
    print("  N_0..N_4, and the EXACT RELATION  2(N_1 - N_4) = N_2 - N_3.")
    print("  the same relation on the census histograms (full-period rows must")
    print("  satisfy it; the partial-coverage m37 row is the CONTROL):")
    best = load_ghist()
    print(f"    {'y':>4} {'cov':>8} {'2(N1-N4)-(N2-N3)':>18} {'N2 even':>8} "
          f"{'N3 even':>8}")
    for y in sorted(best):
        cov, h = best[y]
        Nc = classes5(h)
        d = 2 * (Nc[1] - Nc[4]) - (Nc[2] - Nc[3])
        print(f"    {y:>4} {cov:>8.4f} {d:>18} {str(Nc[2]%2==0):>8} "
              f"{str(Nc[3]%2==0):>8}")
        if abs(cov - 1.0) < 1e-12:
            assert d == 0 and Nc[2] % 2 == 0 and Nc[3] % 2 == 0, (y, Nc)
    print("  ASSERT ok: exact at every full-period machine (11..31); it fails")
    print("  on the partial m37 window, as a period-wide law must.")


# ------------------------------------------------------------------ part D
def partD(ys):
    print("=" * 78)
    print("PART D: H_5(1) from three integers, and THE PARITY THEOREM")
    print("  Substituting the cell forms into H = sum_r N_r omega^r and using")
    print("  1 + omega + omega^4 = phi:")
    print("      Re H = phi*N/3 + (3-phi)*e - ((3phi+1)/2)*(b+c)")
    print("      Im H = (2 sin36 + sin72)*(b-c) = (2 sin36 + sin72)*(N_2-N_3)/2")
    print("  The whole frequency-1/5 transform is carried by three integers and")
    print("  its IMAGINARY part by ONE.")
    print("  arg H = 126 deg (the pole phase of item 32) <=> two integer")
    print("  conditions   (a) N_0+N_1 = 2 N_3   and   (b) N_0+N_1 = N_2+N_4,")
    print("  and in cell variables (b) reads  2(b+c-e) = N/3.  N = prod (q-2)")
    print("  is ODD, so N/3 is odd and (b) is UNSATISFIABLE:")
    print("      THEOREM.  D := (N_0+N_1) - (N_2+N_4) is ODD at every machine,")
    print("      equivalently (N_2+N_3) - 2N_0 = 2 (mod 4).  THE POLE PHASE 126")
    print("      IS NEVER ATTAINED EXACTLY, AT ANY MACHINE.")
    for y in ys:
        o, P = openings(y)
        N = o.size
        T, g = cells5(o, P)
        e, b, c = int(T[2][0]), int(T[0][2]), int(T[0][3])
        Nc = [int(T[:, r].sum()) for r in range(5)]
        H = sum(Nc[r] * OMEGA ** r for r in range(5))
        re = PHI * N / 3 + (3 - PHI) * e - ((3 * PHI + 1) / 2) * (b + c)
        im = IMCO * (b - c)
        assert abs(H.real - re) < 1e-6 * max(1, abs(H)), (y, H.real, re)
        assert abs(H.imag - im) < 1e-6 * max(1, abs(H)), (y, H.imag, im)
    print(f"  ASSERT ok: the three-integer closed forms reproduce Re H and Im H "
          f"exactly at m{'/'.join(map(str,ys))}.")
    best = load_ghist()
    print(f"    {'y':>4} {'cov':>7} {'D=(N0+N1)-(N2+N4)':>19} {'D odd':>6} "
          f"{'(N2+N3)-2N0 mod4':>18} {'argH':>8} {'dev126':>8} "
          f"{'floor dev(deg)':>14}")
    for y in sorted(best):
        cov, h = best[y]
        Nc = classes5(h)
        H = sum(Nc[r] * OMEGA ** r for r in range(5))
        D = (Nc[0] + Nc[1]) - (Nc[2] + Nc[4])
        m4 = ((Nc[2] + Nc[3]) - 2 * Nc[0]) % 4
        mindev = math.degrees(1.0 / abs(H)) if abs(H) else float("nan")
        print(f"    {y:>4} {cov:>7.4f} {D:>19} {str(D%2==1):>6} {m4:>18} "
              f"{deg(H):>8.2f} {deg(H)-126:>+8.2f} {mindev:>14.2e}")
        if abs(cov - 1.0) < 1e-12:
            assert D % 2 == 1 and m4 == 2, (y, D, m4)
    print("  ASSERT ok: D odd and (N_2+N_3)-2N_0 = 2 (mod 4) at every")
    print("  full-period machine 11..31.")
    print("  HONEST SCOPE: the parity floor forces |dev| > 0 but only by ~1e-6")
    print("  deg; it kills the pin as an EXACT statement, not as an")
    print("  approximate one.  The measured +-4 deg is a different quantity.")


# ------------------------------------------------------------------ part E
def gf2_pole(p, Nmod2=1):
    """Is the gear-p pole-phase condition satisfiable mod 2?

    Unknowns: cell matrix M[i][s], reduced to MIRROR ORBITS.  Constraints over
    GF(2): the p-2 row sums (= N/(p-2), odd at every machine) and the (p-1)/2
    pole conditions beta_r = beta_{-r}, beta_r = N_{r+1} - N_r.
    """
    A, D = delta_table(p)
    n = len(A)
    par = list(range(n * n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def uni(x, y):
        x, y = find(x), find(y)
        if x != y:
            par[max(x, y)] = min(x, y)
    for i in range(n):
        for s in range(n):
            j = A.index((-(A[i] + D[i][s])) % p)
            uni(i * n + s, j * n + s)
    roots = sorted({find(x) for x in range(n * n)})
    col = {r: k for k, r in enumerate(roots)}
    nv = len(roots)
    rows, rhs = [], []
    for i in range(n):
        v = [0] * nv
        for s in range(n):
            v[col[find(i * n + s)]] ^= 1
        rows.append(v)
        rhs.append(Nmod2)
    Nr = [[0] * nv for _ in range(p)]
    for i in range(n):
        for s in range(n):
            Nr[D[i][s]][col[find(i * n + s)]] ^= 1
    beta = [[Nr[(r + 1) % p][k] ^ Nr[r][k] for k in range(nv)]
            for r in range(p)]
    for r in range(1, (p - 1) // 2 + 1):
        rows.append([beta[r][k] ^ beta[(-r) % p][k] for k in range(nv)])
        rhs.append(0)
    m, piv = len(rows), 0
    for cidx in range(nv):
        pr = next((r for r in range(piv, m) if rows[r][cidx]), None)
        if pr is None:
            continue
        rows[piv], rows[pr] = rows[pr], rows[piv]
        rhs[piv], rhs[pr] = rhs[pr], rhs[piv]
        for r in range(m):
            if r != piv and rows[r][cidx]:
                rows[r] = [a ^ b for a, b in zip(rows[r], rows[piv])]
                rhs[r] ^= rhs[piv]
        piv += 1
    ok = all(any(rows[r]) or rhs[r] == 0 for r in range(m))
    return nv, ok


def partE():
    print("=" * 78)
    print("PART E: gear 7 (backlog U3) - cells, orbits, parity")
    print("  General count: (p-2)^2 cells, p-2 of them mirror-fixed, hence")
    print("  (p-2)(p-1)/2 orbits and (p-2)(p-3)/2 free integers after the row")
    print("  sums.  The pole-phase condition is (p-1)/2 integer equations.")
    print(f"  {'p':>3} {'cells':>6} {'orbits':>7} {'free':>5} {'poleEqs':>8} "
          f"{'parity-obstructed':>18}")
    for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        A, D = delta_table(p)
        nv, ok = gf2_pole(p)
        assert nv == (p - 2) * (p - 1) // 2, (p, nv)
        print(f"  {p:>3} {len(A)**2:>6} {nv:>7} {(p-2)*(p-3)//2:>5} "
              f"{(p-1)//2:>8} {str(not ok):>18}")
    assert not gf2_pole(5)[1], "gear 5 must be parity-obstructed"
    print("  ASSERT ok: the GF(2) test independently reproduces part D's")
    print("  theorem at p = 5.")
    best = load_ghist()
    print("  measured bracket asymmetries (B_p real <=> beta_r = beta_{-r} for")
    print("  all r; at p=5 that is 2 equations, at p=7 it is 3):")
    print("  Im B_p = sum_r alpha_r sin(2 pi r/p), alpha_r = b_r - b_{-r}; the")
    print("  sines are Q-independent, so for INTEGER alpha the bracket is real")
    print("  only if every alpha_r vanishes.  At p = 5 part D proves alpha_1 =")
    print("  -D is ODD, so it never does: instead the RATIO alpha_1/alpha_2")
    print("  must approach the irrational -sin36/sin72 = -1/phi = %.6f."
          % (-1 / PHI))
    print(f"    {'y':>4} {'cov':>7} {'argB(5,1)':>10} {'argB(7,1)':>10} "
          f"{'a1/N @5':>10} {'a2/N @5':>10} {'a1/a2':>9} {'-1/phi dev':>11} "
          f"{'max|a|/N @7':>12}")
    for y in sorted(best):
        cov, h = best[y]
        N = sum(h.values())
        out = []
        for p in (5, 7):
            Np = [0] * p
            for v, c in h.items():
                Np[v % p] += c
            be = [Np[(r + 1) % p] - Np[r] for r in range(p)]
            asym = [be[r] - be[(-r) % p] for r in range(1, (p - 1) // 2 + 1)]
            w = cmath.exp(2j * pi / p)
            H = sum(c * w ** v for v, c in h.items())
            B = H * (1 - w) / w
            a = deg(B)
            a = a - 180 if a > 90 else (a + 180 if a <= -90 else a)
            out.append((a, asym))
        a1, a2 = out[0][1]
        ratio = a1 / a2 if a2 else float("nan")
        print(f"    {y:>4} {cov:>7.4f} {out[0][0]:>10.2f} {out[1][0]:>10.2f} "
              f"{a1/N:>+10.5f} {a2/N:>+10.5f} {ratio:>9.5f} "
              f"{ratio+1/PHI:>+11.5f} "
              f"{max(abs(x) for x in out[1][1])/N:>12.5f}")
    print("  READING: at p = 5 the pole phase is ONE ratio of two integers")
    print("  converging on the golden direction -1/phi (and provably never")
    print("  reaching it, part D); at p = 7 THREE independent asymmetries must")
    print("  vanish at once, and they are an order of magnitude larger and")
    print("  decaying far more slowly.  That is the U3 asymmetry, measured.")


# ------------------------------------------------------------------ part F
def renewal_transform(m, gl, lam, z):
    """Exact first-passage transform of the corridor-renewal model at modulus
    m: openings are an independent thinning (rate a) of the slots exposed for
    every gear in gl, with a fixed by the true mean gap lam.  MODEL, floats."""
    teeth = {q: {pow(6, -1, q) % q, (-pow(6, -1, q)) % q} for q in gl}
    inE = [all((r % q) not in teeth[q] for q in gl) for r in range(m)]
    E = [r for r in range(m) if inE[r]]
    a = m / (len(E) * lam)
    if a > 1 + 1e-9:
        return None, a
    a = min(a, 1.0)
    al = [0j] * (m + 1)
    be = [0j] * (m + 1)
    al[m], be[m] = 0j, 1 + 0j
    for r in range(m - 1, -1, -1):
        h = a if inE[r] else 0.0
        al[r] = z * (h + (1 - h) * al[r + 1])
        be[r] = z * (1 - h) * be[r + 1]
    v0 = al[0] / (1 - be[0]) if abs(1 - be[0]) > 1e-15 else al[0]
    v = [al[r] + be[r] * v0 for r in range(m)]
    return sum(v[(r + 1) % m] for r in E) / len(E), a


def partF(ys):
    print("=" * 78)
    print("PART F: backlog U2 - THE AMPLITUDE NEAR-LAW |H_5(1)|/H0 * meangap")
    A1 = (OMEGA + 2 * OMEGA ** 2) / 3
    print("  Column s of the cell matrix contributes N_1 w + N_2 w^2 (s=1) and")
    print("  N_3 w^3 + N_4 w^4 (s=2).  The PHASE-BLIND step model (M[i][s]")
    print("  independent of i) would force N_2 = 2 N_1 and N_3 = 2 N_4 and give")
    print("      H/N = qt_0 + qt_1 A_1 + qt_2 conj(A_1),  A_1 = (w+2w^2)/3,")
    print("  qt_s = fraction of gaps whose exposed-step count n = s (mod 3):")
    print("      qt_0 = N_0/N,  qt_1 = (N_1+N_2)/N,  qt_2 = (N_3+N_4)/N.")
    print(f"  A_1 = {A1:.6f} (float); sum_s A_s = (2-phi)/3 = {(2-PHI)/3:.6f}")
    rows = []
    for y in ys:
        fn = os.path.join(DATA, f"spiral_{y}.json")
        if not os.path.exists(fn):
            continue
        d = json.load(open(fn))
        N, P = d["N"], d["P"]
        T = np.array(d["cell"], dtype=np.int64)
        Nc = [int(T[:, r].sum()) for r in range(5)]
        assert sum(Nc) == N
        H = sum(Nc[r] * OMEGA ** r for r in range(5))
        lam = P / N
        qt = [Nc[0] / N, (Nc[1] + Nc[2]) / N, (Nc[3] + Nc[4]) / N]
        assert abs(sum(qt) - 1.0) < 1e-12
        Hpb = N * (qt[0] + qt[1] * A1 + qt[2] * A1.conjugate())
        rows.append((y, N, P, lam, H, qt, Hpb, Nc))
    print(f"  {'y':>4} {'lam=P/N':>9} {'|H|/N':>8} {'argH':>8} "
          f"{'|H|/N*lam':>10} {'N2/2N1':>8} {'N3/2N4':>8} "
          f"{'|Hpb|/|H|':>10} {'argHpb':>8}")
    for y, N, P, lam, H, qt, Hpb, Nc in rows:
        print(f"  {y:>4} {lam:>9.4f} {abs(H)/N:>8.4f} {deg(H):>8.2f} "
              f"{abs(H)/N*lam:>10.4f} {Nc[2]/(2*Nc[1]):>8.4f} "
              f"{Nc[3]/(2*Nc[4]):>8.4f} {abs(Hpb)/abs(H):>10.4f} "
              f"{deg(Hpb):>8.2f}")
    print("  (N2/2N1 and N3/2N4 are 1 exactly iff the cells are phase-blind;")
    print("  their departure from 1 is the whole phase-grading content.)")
    print("  THE FLOOR.  If consecutive openings decorrelate, the endpoint")
    print("  residues become independent uniform on A_5 and")
    print("      |H|/N -> |1+w^2+w^3|^2/9 = (2-phi)/9 = %.6f (REAL, arg 0)."
          % ((2 - PHI) / 9))
    print("  So |H|/N * lam is bounded BELOW by 0.042440*lam, which passes")
    print("  1.015 at lam = %.2f: the near-law CANNOT be an invariant."
          % (1.015 * 9 / (2 - PHI)))
    print("  CORRIDOR-RENEWAL LADDER (model, floats): openings = independent")
    print("  thinning of the slots exposed mod m, rate fixed by the true lam.")
    mods = [(5, [5]), (35, [5, 7]), (385, [5, 7, 11]), (5005, [5, 7, 11, 13]),
            (85085, [5, 7, 11, 13, 17])]
    print(f"  {'y':>4} {'lam':>8} {'meas':>8} " +
          " ".join(f"{'m='+str(m):>9}" for m, _ in mods))
    for y, N, P, lam, H, qt, Hpb, Nc in rows:
        out = []
        for m, gl in mods:
            if P % m:
                out.append("      -   ")
                continue
            v, a = renewal_transform(m, gl, lam, OMEGA)
            if v is None:
                out.append("      -   ")
                continue
            if m == P:      # a = 1: the model IS the machine - a hard gate
                assert abs(abs(v) * lam - abs(H) / N * lam) < 1e-9, (y, m)
            out.append(f"{abs(v)*lam:>9.4f}")
        print(f"  {y:>4} {lam:>8.4f} {abs(H)/N*lam:>8.4f} " + " ".join(out))
    print("  ASSERT ok: at m = P the model reproduces the machine's own value")
    print("  to 1e-9 (a = 1, no thinning) - the ladder's correctness gate.")
    print("  READING: NO FIXED corridor depth reproduces the flat 1.015.  The")
    print("  m=5 column DECAYS (1.09 -> 0.87) and every deeper column RISES")
    print("  (m=385: 1.126 -> 1.115; m=5005: 1.036 -> 1.055; m=85085: 1.015 ->")
    print("  1.043).  The measured flatness is the cancellation of those two")
    print("  drifts as the machine's own corridor depth grows with it.")
    print("  MODEL TURN-UP (fixed corridor depth, lam pushed past the data):")
    print(f"    {'lam':>6} " + " ".join(f"{'m='+str(m):>9}" for m, _ in mods[:3]))
    for lam in (4, 6, 8, 12, 16, 24, 32, 48, 64):
        cells = []
        for m, gl in mods[:3]:
            v, a = renewal_transform(m, gl, lam, OMEGA)
            cells.append(f"{abs(v)*lam:>9.4f}" if v is not None else "      -  ")
        print(f"    {lam:>6} " + " ".join(cells))
    print("  Every column has a MINIMUM and then grows without bound, tending")
    print("  to (2-phi)*lam/9.  (The approach is slow and from BELOW: the O(a)")
    print("  correction partly cancels the floor.)")
    # ---- the exact average-arm anchor
    print("  THE EXACT ANCHOR (round-21 closure, restated as an average).  The")
    print("  depth-sum identity gives sum_{j=1..N-1} What_j(omega) = (2-phi)")
    print("  n_side^2 - N, so the MEAN ARM over the N-1 proper depths is")
    print("      ((2-phi)n_side^2 - N)/(N-1) -> (2-phi)N/9,  REAL POSITIVE,")
    print("  i.e. exactly the decorrelation floor.  So the near-law says")
    print("      |What_1| / mean arm = (1.015/lam)/((2-phi)/9) = 23.92/lam,")
    print("  and lam = 23.92 is the machine size at which DEPTH 1 BECOMES A")
    print("  TYPICAL ARM.  The 'constant' 1.015 is that crossing scale, not an")
    print("  invariant.  Full-depth verification of the closure:")
    for y in (11, 13):
        o, P = openings(y)
        N = o.size
        n_side = N // 3
        r = (o % 5).astype(np.int64)
        tot = 0j
        for j in range(1, N):        # proper windows only; j = N is the period
            d = (np.roll(r, -j) - r) % 5
            cls = np.bincount(d, minlength=5)
            tot += sum(int(cls[t]) * OMEGA ** t for t in range(5))
        pred = (2 - PHI) * n_side ** 2 - N
        assert abs(tot.imag) < 1e-6 * abs(tot), (y, tot)
        assert abs(tot.real - pred) < 1e-6 * abs(pred), (y, tot, pred)
        W1 = sum(int(x) * OMEGA ** t for t, x in
                 enumerate(np.bincount((np.roll(r, -1) - r) % 5, minlength=5)))
        print(f"    m{y}: sum_j = {tot.real:.4f} (+{tot.imag:.2e}i) vs "
              f"(2-phi)n_side^2-N = {pred:.4f};  |W_1|/mean arm = "
              f"{abs(W1)/(abs(tot)/(N-1)):.4f}, 23.92/lam = "
              f"{23.92/(P/N):.4f}")
    print("  ASSERT ok: the closure holds over ALL N depths, exactly and real.")


if __name__ == "__main__":
    parts = sys.argv[sys.argv.index("--parts") + 1] if "--parts" in sys.argv \
        else "ABCDEF"
    maxy = int(sys.argv[sys.argv.index("--maxy") + 1]) if "--maxy" in sys.argv \
        else 19
    small = [y for y in (11, 13, 17, 19, 23) if y <= maxy]
    if "A" in parts:
        partA(small)
    if "B" in parts:
        partB(small)
    if "C" in parts:
        partC(small)
    if "D" in parts:
        partD(small)
    if "E" in parts:
        partE()
    if "F" in parts:
        partF([11, 13, 17, 19, 23, 29])
    print("DONE")
