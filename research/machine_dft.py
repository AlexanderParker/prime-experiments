"""Round 20 lateral: THE MACHINE IN FREQUENCY SPACE (human directive, frame 2).

The blocking indicator of gear q is a 2-point set {u, -u} mod q (u = 6^{-1}),
so every census object has an exponential-sum form. Exact facts established
and verified here:

A. PER-GEAR SPECTRUM (closed form).  For the exposed set A_q = Z_q - {u,-u}:
       hat_q(0) = q - 2,     hat_q(j) = -2 cos(2 pi j u / q)   (j != 0).
   The whole machine's DFT FACTORISES over gears (CRT):
       hat(j) = prod_q hat_q( j * (P/q)^{-1} mod q ).

B. THE T3 LAW (tooth phase).  6u = 1 (mod q)  ==>  3u = (q+1)/2 (mod q).
   The TRIPLED teeth {3u, -3u} = {(q+1)/2, (q-1)/2} are ADJACENT residues at
   the antipode - for every prime gear q >= 5.  Hence
       hat_q(3) = -2 cos(pi / q)  ->  -2   (near-extremal for a 2-point set):
   at local frequency 3 every gear is, in phase terms, almost a SINGLE tooth.
   This is the Fourier avatar of the tooth law u' ~ q/6 (teeth at +-60 deg).

C. THE GOLDEN MODE (the machine's spectral gap).  Gear 5's local frequency 2:
       hat_5(2) = -2 cos(4 pi / 5) = 2 cos(pi/5) = phi = (1+sqrt5)/2.
   Normalized: phi/3 = 0.5393.  Every other gear has max non-DC ratio
   2 cos(pi/q)/(q-2) < phi/3, and products only shrink; so for EVERY machine
   containing gear 5 the largest non-DC spectral line is gear 5's golden mode,
       max_{chi != 1} |hat(chi)| / hat(1) = phi/3,
   machine-independent.  This is the spectral-gap form of gear-5's corridor
   dominance (AP lemma, exclusion law, pinning all live at gear 5).

D. GAP-HISTOGRAM SPECTRUM.  The pair correlation is the inverse DFT of the
   power spectrum; the gap histogram's period-5 wiggle is the golden line:
   DFT of the gear-5 autocorrelation c_5(g) has weights |hat_5(1)|^2 = 1/phi^2
   = 0.382 and |hat_5(2)|^2 = phi^2 = 2.618 - the phi^2 line dominates by
   phi^4 = 6.85.  Dividing the measured histogram by the closed-form N2(g)
   should collapse the gear-frequency lines; measured collapse reported.
"""
import csv, os, cmath, math
from math import prod, pi, cos, sqrt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))]

print("=" * 76)
print("PART A: per-gear spectrum closed form vs direct DFT (gears 5..43)")
worst = 0.0
for q in primes(5, 43):
    u = pow(6, -1, q)
    a = np.ones(q); a[u % q] = 0; a[(-u) % q] = 0
    F = np.fft.fft(a)
    for j in range(q):
        pred = q - 2 if j == 0 else -2 * cos(2 * pi * j * u / q)
        worst = max(worst, abs(F[j].real - pred) + abs(F[j].imag))
assert worst < 1e-9
print(f"  exact (max deviation {worst:.1e}); coefficients are REAL: "
      f"hat_q(j) = -2cos(2 pi j u/q)")

print("=" * 76)
print("PART B: the T3 law - tripled teeth are adjacent at the antipode")
for q in primes(5, 100000):
    u = pow(6, -1, q)
    assert (3 * u) % q == (q + 1) // 2, q
print("  3u = (q+1)/2 mod q for ALL primes 5 <= q <= 100000 (proof: 6u=1 =>")
print("  2*(3u) = 1 = q+1 mod q; 2 invertible). hat_q(3) = -2cos(pi/q) -> -2.")

print("=" * 76)
print("PART C: the golden mode is the spectral gap of every machine")
phi = (1 + sqrt(5)) / 2
assert abs(-2 * cos(4 * pi / 5) - phi) < 1e-12
for y in (13, 17):
    gears = primes(5, y)
    P = prod(gears)
    # enumerate ALL characters via the product formula; find max non-DC ratio
    best = 0.0; arg = None
    import itertools
    for js in itertools.product(*[range(q) for q in gears]):
        if all(j == 0 for j in js):
            continue
        v = 1.0
        for q, j in zip(gears, js):
            u = pow(6, -1, q)
            v *= (q - 2) if j == 0 else abs(2 * cos(2 * pi * j * u / q))
        if v > best:
            best = v; arg = js
    dc = prod(q - 2 for q in gears)
    print(f"  machine {y}: max non-DC |hat|/DC = {best/dc:.6f} at local "
          f"frequencies {arg} vs phi/3 = {phi/3:.6f}")
    assert abs(best / dc - phi / 3) < 1e-12
    assert arg[0] in (2, 3) and all(j == 0 for j in arg[1:])

print("=" * 76)
print("PART A2: global factorisation vs direct FFT (machine 17, P = 85085)")
gears = primes(5, 17)
P = prod(gears)
killed = np.zeros(P, bool)
for q in gears:
    u = pow(6, -1, q)
    killed[u % q::q] = True
    killed[(q - u) % q::q] = True
ind = (~killed).astype(float)
F = np.fft.fft(ind)
Nq = {q: pow(P // q, -1, q) for q in gears}
pred = np.ones(P)
for q in gears:
    u = pow(6, -1, q)
    loc = np.array([(j * Nq[q]) % q for j in range(P)])
    hq = np.where(loc == 0, q - 2.0, -2 * np.cos(2 * pi * loc * u / q))
    pred *= hq
dev = np.max(np.abs(F.real - pred)) + np.max(np.abs(F.imag))
print(f"  max |FFT - product formula| = {dev:.2e} over all {P} frequencies")
assert dev < 1e-4
print("  (the machine's full 85085-point spectrum is CLOSED FORM, and real)")

print("=" * 76)
print("PART D: gear-frequency lines of the gap histogram, before/after N2")

def cq2(q, g):
    u = pow(6, -1, q)
    if g % q == 0: return q - 2
    if g % q in ((2 * u) % q, (-2 * u) % q): return q - 3
    return q - 4

def line_power(x, f):
    """|sum x(g) e(-2 pi i f g)|^2 / len^2 of a detrended series."""
    g = np.arange(len(x))
    A = np.vstack([np.ones_like(g), g]).T.astype(float)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ coef
    z = np.sum(r * np.exp(-2j * pi * f * g))
    return abs(z) ** 2 / len(x) ** 2

W1 = {}
for y in (23, 29):
    p = os.path.join(DATA, f"depth_identity_{y}.csv")
    W1[y] = {int(r["g"]): int(r["W1"]) for r in csv.DictReader(open(p))
             if int(r["W1"])}
from collections import defaultdict
h = defaultdict(int)
for r in csv.DictReader(open(os.path.join(DATA, "gap_pair_joint.csv"))):
    if int(r["y"]) == 31 and int(r["lag"]) == 1:
        h[int(r["gu"])] += int(r["count"])
W1[31] = dict(h)

def cq_set(q, offs):
    u = pow(6, -1, q)
    t = {u % q, (-u) % q}
    return sum(1 for r in range(q) if all((r + d) % q not in t for d in offs))

for y in (23, 29, 31):
    gears = primes(5, y)
    gs = sorted(g for g in W1[y] if W1[y][g] > 0)
    lw = np.array([math.log(W1[y][g]) for g in gs])
    ln2 = np.array([math.log(prod(cq2(q, g) for q in gears)) for g in gs])
    lpred = []
    for g in gs:
        N2 = prod(cq2(q, g) for q in gears)
        f = 0.0
        for t in range(1, g):
            f += math.log(1.0 - prod(cq_set(q, [0, t, g]) for q in gears) / N2)
        lpred.append(math.log(N2) + f)
    lpred = np.array(lpred)
    print(f"  machine {y} ({len(gs)} gap values): line power raw -> "
          f"minus N2 -> minus FULL closed form")
    for name, f in (("1/5", 0.2), ("2/5 (golden)", 0.4), ("1/7", 1 / 7),
                    ("2/7", 2 / 7), ("3/7", 3 / 7)):
        b = line_power(lw, f)
        a = line_power(lw - ln2, f)
        c = line_power(lw - lpred, f)
        print(f"    line {name:>12}: {b:.4f} -> {a:.4f} -> {c:.4f}  "
              f"(full form removes {100*(1-c/b) if b else 0:+.0f}%)")
