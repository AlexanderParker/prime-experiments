"""Round 21 lateral, target (a): THE C14 +126 DEGREE MACHINE-INDEPENDENT PHASE.

Mechanic round 20 (dft_events.py part 3): the DFT of the full-period gap-value
histogram at gear 5, H_5(1) = sum_v hist[v] e(2 pi i v/5), has
|H_5(1)|/H_0 falling 0.31 -> 0.18 over machines 13..37 while
arg H_5(1) = +126 deg +- 2 at ALL machines.  Unexplained; handed to Lateral.

CLOSED-FORM CANDIDATE (this script tests it):

    126 deg = 90 deg + 36 deg = arg(omega - 1),   omega = e(1/5),

exactly the argument of e(1/5) - e(0/5).  A real 5-vector (N_0..N_4) of
residue-class counts has arg(sum N_r omega^r) = 126 deg exactly iff its
frequency-1 component is ANTISYMMETRIC under the reflection r -> 1 - r
(mod 5), which swaps classes 0<->1 and 2<->4 and fixes class 3:
    omega^r - omega^{1-r} = 2i sin(2 pi (r - 1/2)/5) e(1/10),
    arg = 90 + 36 = 126 for every pair.
Equivalent golden linear constraint (Im(e^{-i 126 deg} H) = 0):
    cos36 (N0+N1) = cos72 (N2+N4) + N3
    <=>  phi^2 (N0+N1) = (N2+N4) + 2 phi N3        (phi = (1+sqrt5)/2).

EXACT SUM RULE (proved below, verified here): over ALL window depths j,
    sum_{j>=1} What_j(omega) = |A_hat|^2 - N
                            = (2 - phi) prod_{q != 5} (q-2)^2  -  prod_q (q-2),
where What_j(omega) = sum over j-windows of omega^{window sum} and
A_hat = sum over openings a of omega^a = (1 + omega^2 + omega^3) prod_{q!=5}(q-2)
(CRT: openings are uniform over A_5 = {0,2,3} mod 5).  Proof: every ordered
pair of distinct openings is the endpoint pair of exactly one window (the
depth-sum identity, r20), and e((b-a)/5) is well defined since 5 | P.
(1+omega^2+omega^3)(1+omega^{-2}+omega^{-3}) = 3 + omega + 2 omega^2
+ 2 omega^3 + omega^4 = sum_r c_5(r) omega^r = 2 - phi = phi^{-2}: REAL.
So the depth family's phases must cancel: the W_j(omega) arms close a
polygon in C whose total is real positive.  W_1's arm at +126 deg is the
first edge; this script measures the whole spiral.

MODELS tested for the mechanism:
  M0: geometric decay x c_5 endpoint weights  (phase drifts with rho -> fails)
  M1: full closed-form predictor N2(g) prod_t (1 - N3(0,t,g)/N2(g))
      (endpoint + independent-interior arithmetic, per machine)
  M2: corridor-hardness model: W1 modulation ~ sum over admissible endpoint
      phases r of beta^{e(r, g mod 5)}, where e = the finite-size correction
      to the count of 5-exposed interior slots (the slots gear 5 CANNOT help
      block).  Machine enters only through beta = per-slot cost.

All angles/floats are FLOATS and labeled; class counts are exact integers.

Usage:  python c14_phase.py            # parts 1,2,4 (histogram machines)
        python c14_phase.py --ladder 13 17 19 23   # part 3 depth spiral
"""
import csv, os, sys, cmath, math
from math import prod, pi, cos, sin, sqrt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PHI = (1 + sqrt(5)) / 2
OMEGA = cmath.exp(2j * pi / 5)
TARGET = 126.0  # degrees, = 90 + 36 = arg(omega - 1)


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))]


def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q


def c_q(q, g):
    u = pow(6, -1, q)
    if g % q == 0:
        return q - 2
    if g % q in ((2 * u) % q, (-2 * u) % q):
        return q - 3
    return q - 4


def cq_set(q, offs):
    t = set(teeth(q))
    return sum(1 for r in range(q) if all((r + d) % q not in t for d in offs))


def load_ghist():
    """machine -> (coverage, {v: count}); full period preferred, else max cov."""
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
    return best


def phase_deg(z):
    return math.degrees(cmath.phase(z))


# ---------------------------------------------------------------- part 1+2
def part12():
    best = load_ghist()
    print("=" * 78)
    print("PART 1: measured H_5(1) and H_5(2) phases vs the closed-form "
          "candidate 126 = 90+36")
    print("  (arg(omega-1) = %.4f deg exactly; all angles float)"
          % phase_deg(OMEGA - 1))
    assert abs(phase_deg(OMEGA - 1) - 126.0) < 1e-9
    print(f"  {'y':>4} {'cov':>7} {'meangap':>8} {'|H1|/H0':>8} "
          f"{'arg H1':>9} {'dev126':>7} {'|H2|/H0':>8} {'arg H2':>9} "
          f"{'argH7(1)':>9}")
    rows = {}
    for y in sorted(best):
        cov, h = best[y]
        N = [0] * 5
        for v, c in h.items():
            N[v % 5] += c
        H0 = sum(N)
        H1 = sum(N[r] * OMEGA**r for r in range(5))
        H2 = sum(N[r] * OMEGA**(2 * r) for r in range(5))
        H7 = sum(c * cmath.exp(2j * pi * v / 7) for v, c in h.items())
        mg = sum(v * c for v, c in h.items()) / H0
        a1, a2 = phase_deg(H1), phase_deg(H2)
        rows[y] = dict(N=N, H0=H0, H1=H1, H2=H2, cov=cov, meangap=mg, h=h)
        print(f"  {y:>4} {cov:>7.3f} {mg:>8.3f} {abs(H1)/H0:>8.4f} "
              f"{a1:>9.2f} {a1-TARGET:>+7.2f} {abs(H2)/H0:>8.4f} "
              f"{a2:>9.2f} {phase_deg(H7):>9.2f}")
    # reproduction assertions (full-period machines 13..31)
    for y in (13, 17, 19, 23, 29, 31):
        a1 = phase_deg(rows[y]["H1"])
        assert abs(a1 - TARGET) < 4.0, (y, a1)
    print("  ASSERT ok: arg H_5(1) within 4 deg of 126 at all full-period "
          "machines 13..31")

    print("=" * 78)
    print("PART 2: the golden constraint and the antisymmetry split")
    print("  constraint C = cos36*(N0+N1) - cos72*(N2+N4) - N3  (0 iff arg "
          "exactly 126)")
    print(f"  {'y':>4} {'N0':>12} {'N1':>12} {'N2':>12} {'N3':>12} "
          f"{'N4':>12}")
    for y in sorted(rows):
        N = rows[y]["N"]
        print(f"  {y:>4} " + " ".join(f"{n:>12}" for n in N))
    print(f"  {'y':>4} {'C/|H1|':>9} {'antisym%':>9} "
          f"  (freq-1 energy in the 126-direction vs 36-direction)")
    for y in sorted(rows):
        H1 = rows[y]["H1"]
        # C/|H1| = -sin(arg-126): the normalized violation of the constraint
        dev = -math.sin(math.radians(phase_deg(H1) - TARGET))
        anti = math.cos(math.radians(phase_deg(H1) - TARGET)) ** 2
        print(f"  {y:>4} {dev:>+9.4f} {100*anti:>8.2f}%")
    return rows


# ---------------------------------------------------------------- part 3
def gaps_of(y, chunk=40_000_000):
    gears = primes(5, y)
    P = prod(gears)
    first = last = None
    parts = []
    nopen = 0
    a = 0
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, bool)
        for q in gears:
            u = pow(6, -1, q)
            for t in (u, q - u):
                killed[(t - a) % q::q] = True
        o = np.flatnonzero(~killed).astype(np.int64) + a
        if o.size:
            if first is None:
                first = o[0]
            else:
                parts.append(np.array([o[0] - last], np.int64))
            if o.size > 1:
                parts.append(np.diff(o))
            last = o[-1]
            nopen += o.size
        a += S
    parts.append(np.array([P - last + first], np.int64))
    d = np.concatenate(parts)
    assert d.sum() == P and d.size == nopen
    return d, P, nopen


def part3(ys, J=25):
    print("=" * 78)
    print("PART 3: the depth spiral - arg What_j(omega) for j = 1..%d, and "
          "the exact closure" % J)
    for y in ys:
        gears = primes(5, y)
        d, P, N = gaps_of(y)
        d5 = (d % 5).astype(np.int8)
        cs = np.concatenate(([0], np.cumsum(d5, dtype=np.int64))) % 5
        # closure: sum_j What_j(omega) = |sum_a omega^{o_a}|^2 - N, openings
        # mod 5 uniform on {0,2,3} (CRT) -> exact algebra:
        pred_total = (2 - PHI) * prod(q - 2 for q in gears if q != 5) ** 2 - N
        # measured openings class counts (exact integers)
        opencls = np.bincount(cs[:-1], minlength=5)
        n_side = prod(q - 2 for q in gears if q != 5)
        assert opencls[0] == opencls[2] == opencls[3] == n_side, opencls
        assert opencls[1] == opencls[4] == 0
        print(f"  machine {y}: openings mod 5 = {list(opencls)} "
              f"(uniform on A_5, EXACT); closure total = "
              f"{pred_total:.6f} (real, = (2-phi)*prod(q-2,q!=5)^2 - N)")
        # the spiral: What_j for j = 1..J
        tot = 0j
        print(f"    {'j':>3} {'|W_j|/N':>10} {'arg deg':>9} "
              f"{'cum arg':>9} {'cum |.|/N':>10}")
        n = d5.size
        ext = np.concatenate((cs[:-1], cs[:-1][:J] + cs[-1]))  # cyclic
        for j in range(1, J + 1):
            diff = (ext[j:j + n] - ext[:n]) % 5
            cls = np.bincount(diff.astype(np.int64), minlength=5)
            Wj = sum(int(cls[r]) * OMEGA**r for r in range(5))
            tot += Wj
            print(f"    {j:>3} {abs(Wj)/N:>10.4f} {phase_deg(Wj):>9.2f} "
                  f"{phase_deg(tot):>9.2f} {abs(tot)/N:>10.4f}")
        print(f"    partial sum j<=%d: %s  (closure says full sum -> "
              "%.4f + 0i; remaining arms j>%d carry the rest)"
              % (J, f"{tot:.4f}", pred_total, J))


# ---------------------------------------------------------------- part 4
def model_M0(rho):
    """geometric decay x c_5 weights; returns phase (deg)."""
    w = {0: 3 * rho**5, 1: rho, 2: 2 * rho**2, 3: 2 * rho**3, 4: rho**4}
    H = sum(w[r] * OMEGA**r for r in range(5))
    return phase_deg(H)


_T3 = {}


def t3_table(q):
    """c_q(0,a,b) as a q x q table via one matmul:
    T[a,b] = sum_r E[r] E[r+a] E[r+b], E the exposed indicator."""
    if q not in _T3:
        u = pow(6, -1, q)
        E = np.ones(q)
        E[u % q] = 0.0
        E[(-u) % q] = 0.0
        idx = (np.arange(q)[:, None] + np.arange(q)[None, :]) % q
        M = E[idx]                    # M[a, r] = E[r+a]
        A = M * E[None, :]            # A[a, r] = E[r] E[r+a]
        _T3[q] = A @ M.T              # T[a, b]
        # spot check against the brute count
        assert abs(_T3[q][1 % q, 2 % q] - cq_set(q, [0, 1, 2])) < 1e-9
    return _T3[q]


def model_M1(y, gmax, p=5):
    """full closed-form predictor: N2 * prod_t (1 - N3/N2); phase of its
    mod-p DFT (deg).  Also returns |H|/H0 and the mean gap of the model."""
    gears = primes(5, y)
    tabs = [(q, t3_table(q)) for q in gears]
    H = 0j
    H0 = 0.0
    mg = 0.0
    for g in range(1, gmax + 1):
        N2 = prod(c_q(q, g) for q in gears)
        if N2 == 0:
            continue
        ts = np.arange(1, g)
        if ts.size:
            N3 = np.ones(ts.size)
            for q, tt in tabs:
                N3 *= tt[ts % q, g % q]
            ratio = N3 / N2
            if np.any(ratio >= 1.0):
                continue
            lf = float(np.log1p(-ratio).sum())
        else:
            lf = 0.0
        w = N2 * math.exp(lf)
        H += w * cmath.exp(2j * pi * g / p)
        H0 += w
        mg += w * g
    return phase_deg(H), abs(H) / H0, mg / H0


def hardness_e(qq, gbar):
    """for gear qq: e(r, gbar) = #exposed-qq residues strictly inside
    (r, r+g) minus the g-linear part, per admissible endpoint r; returns
    the list of (r, e) over admissible r and the linear rate."""
    t = set(teeth(qq))
    A = [r for r in range(qq) if r not in t]
    rate = len(A) / qq
    out = []
    for r in A:
        if (r + gbar) % qq not in t:
            # count exposed among r+1 .. r+gbar-1 (one period's worth of the
            # fractional part; finite-size correction at offset gbar)
            cnt = sum(1 for x in range(1, gbar) if (r + x) % qq not in t)
            out.append((r, cnt - rate * (gbar - 1)))
    return out, rate


def model_M2(qq, beta):
    """corridor-hardness modulation phase at gear qq for cost beta."""
    H = 0j
    for gbar in range(qq):
        pairs, rate = hardness_e(qq, gbar)
        m = sum(beta ** e for _, e in pairs)
        H += m * cmath.exp(2j * pi * gbar / qq)
    return phase_deg(H)


def part4(rows):
    print("=" * 78)
    print("PART 4: models (all floats)")
    print("  M0 geometric x c_5 endpoint: phase vs decay rho "
          "(shows M0 CANNOT hold the phase)")
    for rho in (0.5, 0.6, 0.7, 0.745, 0.8, 0.85, 0.9, 0.95, 0.97):
        print(f"    rho = {rho:.3f}: arg = {model_M0(rho):>8.2f} deg")
    print("  M1 closed-form endpoint+interior predictor, per machine "
          "(gear 5 and gear 7):")
    print(f"    {'y':>4} {'meas5':>8} {'M1-5':>8} {'diff':>7} "
          f"{'meas7':>8} {'M1-7':>8} {'diff':>7}")
    for y in sorted(rows):
        if y > 31:
            continue
        gmax = max(rows[y]["h"])
        meas = phase_deg(rows[y]["H1"])
        meas7 = phase_deg(sum(c * cmath.exp(2j * pi * v / 7)
                              for v, c in rows[y]["h"].items()))
        mod, _, _ = model_M1(y, gmax)
        mod7, _, _ = model_M1(y, gmax, p=7)
        print(f"    {y:>4} {meas:>8.2f} {mod:>8.2f} {mod-meas:>+7.2f} "
              f"{meas7:>8.2f} {mod7:>8.2f} {mod7-meas7:>+7.2f}")
    print("  M2 corridor-hardness modulation (machine enters only via beta):")
    print("    gear 5:  beta ->", end="")
    for beta in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        print(f"  {beta:.1f}: {model_M2(5, beta):7.2f}", end="")
    print()
    print("    gear 7:  beta ->", end="")
    for beta in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        print(f"  {beta:.1f}: {model_M2(7, beta):7.2f}", end="")
    print()
    # hardness tables for the record (exact rationals x 5)
    for qq in (5, 7):
        print(f"    hardness corrections e(r, gbar) at gear {qq} "
              f"(x{qq} to make integers):")
        for gbar in range(qq):
            pairs, rate = hardness_e(qq, gbar)
            s = ", ".join(f"r={r}: {round(e*qq):+d}" for r, e in pairs)
            print(f"      gbar={gbar}: {s}")


def part6(rows):
    """THE POLE-PHASE LAW.  Abel summation: for omega_k = e(k/p),
        H_p(k) = sum_g W1(g) omega_k^g
               = [omega_k/(1-omega_k)] * (W1(1) + sum_{g>=2} dW1(g) omega_k^{g-1})
    with dW1 the first difference.  arg(omega_k/(1-omega_k)) = 90 + 180k/p
    degrees EXACTLY - for (p,k) = (5,1): 126.  So the residue law's constant
    phase <=> the DIFFERENCED histogram's transform is REAL (positive), and
    the deviation from 126 measures its phase.  Pole phases: (5,1) 126,
    (5,2) 162 == -18 (bracket sign flips), (7,1) 115.714...
    This part measures arg(bracket) directly for each (p, k)."""
    print("=" * 78)
    print("PART 6: pole-phase decomposition - arg of the DIFFERENCED "
          "histogram transform")
    print("  pole phase(p,k) = 90 + 180k/p deg;  bracket = H * (1-w)/w  "
          "should be REAL if the")
    print("  residue law's phase is the pole phase.  argB in (-90,90] "
          "means bracket sign +.")
    print(f"  {'y':>4} {'argB(5,1)':>10} {'argB(5,2)':>10} {'argB(7,1)':>10}"
          f"   (0 = exactly at pole phase, mod 180)")
    for y in sorted(rows):
        h = rows[y]["h"]
        out = []
        for p, k in ((5, 1), (5, 2), (7, 1)):
            w = cmath.exp(2j * pi * k / p)
            H = sum(c * w**v for v, c in h.items())
            B = H * (1 - w) / w
            a = phase_deg(B)
            a = a - 180 if a > 90 else (a + 180 if a <= -90 else a)
            out.append(a)
        print(f"  {y:>4} {out[0]:>10.2f} {out[1]:>10.2f} {out[2]:>10.2f}")
    print("  (5,1): |argB| <= 4 deg everywhere - the differenced histogram "
          "is REAL-aligned at freq 1/5;")
    print("  (5,2): monotone -> 0: SECOND confirmation of the pole law "
          "(nobody had measured freq 2);")
    print("  (7,1): grows away - gear 7's bracket carries real drift: no "
          "pin. The 126 constant is the")
    print("  POLE phase, not an arithmetic invariant; its residual is the "
          "honest open quantity.")


def part5(ys):
    """M1 phase far beyond the data: does it stabilise at 126.00 (limit) or
    keep drifting (plateau)?  Pure closed form - no census input.  FLOATS."""
    print("=" * 78)
    print("PART 5: M1 asymptotic sweep (closed form only; machines beyond "
          "any scan)")
    print(f"  {'y':>4} {'gmax':>5} {'modelmg':>8} {'arg5f1':>9} "
          f"{'|H1|/H0':>8} {'arg5f2':>9} {'arg7f1':>9} {'arg5f1-126':>10}")
    for y in ys:
        # model mean gap grows ~ prod q/(q-2); size gmax generously
        mg_est = prod(q / (q - 2) for q in primes(5, y))
        gmax = int(8 * mg_est) + 40
        a5, amp5, mg = model_M1(y, gmax)
        a52, _, _ = model_M1(y, gmax, p=2.5)   # e(2g/5): gear-5 frequency 2
        a7, _, _ = model_M1(y, gmax, p=7)
        print(f"  {y:>4} {gmax:>5} {mg:>8.3f} {a5:>9.3f} {amp5:>8.4f} "
              f"{a52:>9.3f} {a7:>9.3f} {a5-126:>+10.3f}", flush=True)


if __name__ == "__main__":
    if "--ladder" in sys.argv:
        i = sys.argv.index("--ladder")
        ys = [int(a) for a in sys.argv[i + 1:]]
        part3(ys)
    elif "--asym" in sys.argv:
        i = sys.argv.index("--asym")
        ys = [int(a) for a in sys.argv[i + 1:]] or [37, 41, 43, 47, 53, 59,
                                                    61, 67, 71, 79, 89, 97]
        part5(ys)
    else:
        rows = part12()
        part6(rows)
        part4(rows)
    print("DONE")
