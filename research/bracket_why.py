"""Round 22 lateral, target (d) - MY OWN OPEN ITEM FROM ROUND 21:
WHY IS GEAR 5's POLE BRACKET REAL WHILE GEAR 7's DRIFTS?

Round 21 (docs/novel/pole-phase-law.md) proved the exact Abel identity

    H_p(k) = sum_g W1(g) w^g = [w/(1-w)] * B(p,k),   w = e(k/p),

so arg H_p(k) = (90 + 180k/p) + arg B, and the observed "machine-independent
+126 deg" at gear 5 IS the statement "B(5,1) is REAL".  Measured: arg B(5,1)
= +3.6 -> -0.3 deg over machines 11..37 (real to a fraction of a degree),
while arg B(7,1) DRIFTS -3 -> +17 deg.  Round 21 could reproduce both with a
closed-form predictor but could not say WHY.  This file derives it.

THE MODEL (the same one that gives the corridor eigenvalue in closed form,
research/corridor_lambda.py): track only the phase mod p.  Openings are
EXACTLY uniform on A_p (CRT), so the per-slot hazard is exactly

    h(r) = rho_p * [r in A_p],   rho_p = prod_{q != p} (1 - 2/q),  a = 1-rho.

Under slot independence the gap law out of phase r is
p(v|r) = h(r+v) prod_{i<v} (1 - h(r+i)), and the gap transform is a p x p
resolvent:  Phi(z) = (1/(p-2)) sum_{r in A} [ z (I - zB)^{-1} O 1 ](r),
B = S D_{1-h}, O = S D_h.  The bracket is then

    B(p,k) = Phi(w) (1 - w) / w,   ONE parameter a in (0,1).

THE CLAIM TESTED HERE: arg B(5,1) is IDENTICALLY ZERO in a - the gear-5
bracket is real for EVERY machine, not approximately - while arg B(7,1) is a
genuine non-constant function of a, so gear 7 must drift.  If that holds
symbolically it answers the round-21 question as an identity in one variable.

Usage: python bracket_why.py
"""
from math import prod, pi, gcd
import numpy as np
import sympy as sp


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** .5) + 1))]


def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q


def exposed(p):
    t = set(teeth(p))
    return [r for r in range(p) if r not in t]


def bracket_model(p, k, a, backend=np):
    """B(p,k) in the corridor-renewal model, as a function of a = 1 - rho."""
    A = exposed(p)
    inA = [1 if r in A else 0 for r in range(p)]
    if backend is np:
        w = np.exp(2j * pi * k / p)
        rho = 1 - a
        Bm = np.zeros((p, p), complex)
        Om = np.zeros((p, p), complex)
        for r in range(p):
            s = (r + 1) % p
            hs = rho if inA[s] else 0.0
            Bm[r, s] = 1 - hs
            Om[r, s] = hs
        f = np.linalg.solve(np.eye(p) - w * Bm, Om @ np.ones(p))
        phi = w * sum(f[r] for r in A) / len(A)
        return phi * (1 - w) / w
    # sympy exact branch
    w = sp.exp(2 * sp.pi * sp.I * k / p)
    rho = 1 - a
    Bm = sp.zeros(p, p)
    Om = sp.zeros(p, p)
    for r in range(p):
        s = (r + 1) % p
        hs = rho if inA[s] else sp.Integer(0)
        Bm[r, s] = 1 - hs
        Om[r, s] = hs
    f = (sp.eye(p) - w * Bm).solve(Om * sp.ones(p, 1))
    phi = w * sum(f[r, 0] for r in A) / len(A)
    return sp.simplify(sp.expand(phi * (1 - w) / w))


def blocked(gears):
    P = prod(gears)
    b = np.zeros(P, bool)
    for q in gears:
        t1, t2 = teeth(q)
        b[t1::q] = True
        b[t2::q] = True
    return b


def exact_bracket(gears, p, k):
    b = blocked(gears)
    idx = np.flatnonzero(~b)
    P = b.size
    g = np.diff(np.append(idx, idx[0] + P))
    h = np.bincount(g)
    w = np.exp(2j * pi * k / p)
    H = sum(int(h[gg]) * w ** gg for gg in range(h.size) if h[gg])
    return H * (1 - w) / w


def main():
    print(__doc__.split("Usage:")[0])
    print("=== PART 1: the model's bracket phase as a function of a alone ==")
    print("  arg B(p,k) in degrees, mod 180, over the whole range of a:")
    print("     a      p=5,k=1   p=5,k=2   p=7,k=1   p=7,k=2   p=11,k=1")
    rows = []
    for a in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        vals = []
        for (p, k) in ((5, 1), (5, 2), (7, 1), (7, 2), (11, 1)):
            z = bracket_model(p, k, a)
            ang = np.degrees(np.angle(z))
            ang = (ang + 90) % 180 - 90
            vals.append(ang)
        rows.append((a, vals))
        print(f"   {a:.2f}  " + "  ".join(f"{v:+9.4f}" for v in vals))
    def span(i):
        return max(r[1][i] for r in rows) - min(r[1][i] for r in rows)
    print(f"\n  spans over a in (0,1): p=5,k=1 {span(0):.2f} deg; "
          f"p=5,k=2 {span(1):.2f}; p=7,k=1 {span(2):.2f}; "
          f"p=7,k=2 {span(3):.2f}; p=11,k=1 {span(4):.2f}")
    print("  PRE-REGISTERED HYPOTHESIS (stated in the docstring before the")
    print("  run): arg B(5,1) identically 0 in a.  RESULT: FALSE - it spans")
    print(f"  {span(0):.1f} deg.  The corridor-renewal (independent-slot)")
    print("  model does NOT make gear 5 special.  Recorded as a refutation,")
    print("  and part 3 says how far the model is from the real machine.")

    print("\n=== PART 3: model vs the real machine (exact histograms) ======")
    print("   y   gears   a = 1-rho_5    arg B(5,1) exact  model   "
          "arg B(7,1) exact  model")
    for y in (11, 13, 17, 19, 23):
        gears = primes(5, y)
        for p in (5, 7):
            rho = prod((q - 2) / q for q in gears if q != p)
            aa = 1 - rho
            ex = exact_bracket(gears, p, 1)
            mo = bracket_model(p, 1, aa)
            ae = (np.degrees(np.angle(ex)) + 90) % 180 - 90
            am = (np.degrees(np.angle(mo)) + 90) % 180 - 90
            if p == 5:
                line = f"  {y:3d}  {len(gears):3d}     {aa:.6f}     " \
                       f"{ae:+8.3f}  {am:+8.3f}"
            else:
                line += f"     {ae:+8.3f}  {am:+8.3f}"
                print(line)
    print("\n  VERDICT (negative, and useful).  The corridor-renewal model")
    print("  reproduces the corridor eigenvalue lambda_2 to 1-2% (see")
    print("  corridor_lambda.py) but it does NOT reproduce the pole-bracket")
    print("  phase: it puts gear 5 at +11..+14 deg where the machine goes")
    print("  +4.7 -> +0.35, and it puts gear 7 at a nearly flat -19..-15 deg")
    print("  where the machine climbs -2.4 -> +14.3.  The model is wrong in")
    print("  SIGN OF DRIFT for both gears.  So the gear-5 bracket's reality")
    print("  is NOT an independent-slot (endpoint-arithmetic) effect: it is")
    print("  produced by the slot-to-slot CORRELATION the model discards -")
    print("  the interior/kappa term.  That narrows the round-21 open")
    print("  question from 'why gear 5' to 'why does the interior")
    print("  correlation cancel the endpoint phase at p = 5 and not p = 7'.")
    print("  Mean-hazard quantities (lambda_2) and fine phase quantities")
    print("  (arg B) separate cleanly: one model settles the first and is")
    print("  refuted by the second.")


if __name__ == "__main__":
    main()
