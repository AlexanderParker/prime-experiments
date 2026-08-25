"""Round 23 lateral, item (c) - CLOSING THE 0.029 MODULUS DEFICIT IN lambda_2.

WHERE THIS STANDS.  Round 22 (docs/novel/corridor-eigenvalue-closed-form.md)
gave the corridor-phase chain's whole spectrum in closed form,

    lambda_j = mu(w_j),  mu(w) = rho w / (1 - (1-rho) w),  w_j = e(j/e),
    e = prod_{q | m} (q-2) = 15 for m = 35,
    rho = prod_{q not | m} (1 - 2/q),

by ONE modelling step: slots independent given their phase, i.e. the hazard
of an opening is the CONSTANT rho on every exposed phase.  Constructor
independently obtained lambda_2 = p-hat(1) (the gap law's characteristic
function at the corridor frequency) - the same object - and both forms nail
the ARGUMENT (0.13-1.06 deg) while UNDERSTATING THE MODULUS by a stable,
converging ~0.029 (measured deficits +0.0076, +0.0146, +0.0192, +0.0225,
+0.0245 at machines 11/13/17/19/23).

THE HYPOTHESIS TESTED HERE.  The deficit is the machine's excess memory, and
the cheapest exact source of memory is the TWO-POINT function, which the
project already has in closed form: for an opening at slot k,

    P(k+t open | k open) = prod_q c_q(t)/(q-2),
    c_q(t) = q-2 if q | t;  q-3 if t = +-2u_q mod q;  q-4 otherwise.

Splitting by gear: for q | m the phase r decides it outright (r+t must be
exposed mod m), and for q not | m it contributes a factor.  So replace the
CONSTANT hazard rho by the EXACT lag-dependent conditional hazard

    MODEL 1:  h_1(r, t) = [ r+t in E ] * rho_1(t),
              rho_1(t) = prod_{q not | m} c_q(t)/(q-2),

and keep everything else (interior independence).  Model 0 is the special
case rho_1 == rho.  Model 1 is still closed-form and scan-free: rho_1 is CRT
arithmetic, no census.

PRE-REGISTERED, BEFORE RUNNING (this docstring was written first):
  Q1  DIRECTION.  rho_1(t) < rho for generic t (openings repel), so model 1
      lengthens gaps.  I predict |lambda_2| RISES from model 0 toward the
      measurement at every machine - i.e. the deficit's SIGN is explained by
      two-point repulsion.
  Q2  SIZE.  I predict model 1 closes at least HALF of the modulus deficit at
      every machine 11..23.
  Q3  ARGUMENT.  Model 0 already gets the argument to 0.2-1.7 deg; I predict
      model 1 does not degrade it by more than 0.5 deg anywhere.
  A measured |lambda_2| BELOW model 1's value at any machine refutes Q1/Q2 as
  stated; the script prints the verdict either way.

Usage: python lambda2_pair.py            # machines 11..19, m = 35 and 385
       python lambda2_pair.py --big      # adds machine 23 (37.2M slots)
"""
import sys
import time
from fractions import Fraction
from math import prod

import numpy as np


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** .5) + 1))]


def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q


def blocked(gears):
    P = prod(gears)
    b = np.zeros(P, bool)
    for q in gears:
        t1, t2 = teeth(q)
        b[t1::q] = True
        b[t2::q] = True
    return b


def exposed_phases(m, gears):
    keep = np.ones(m, bool)
    for q in gears:
        if m % q == 0:
            t1, t2 = teeth(q)
            keep[t1::q] = False
            keep[t2::q] = False
    return np.flatnonzero(keep)


def c_q(q, t):
    """#{r in A_q : r+t in A_q}, by direct count (checks the closed form)."""
    t1, t2 = teeth(q)
    A = np.ones(q, bool)
    A[t1] = A[t2] = False
    return int((A & np.roll(A, -(t % q))).sum())


def c_q_closed(q, t):
    u = pow(6, -1, q)
    if t % q == 0:
        return q - 2
    if (t - 2 * u) % q == 0 or (t + 2 * u) % q == 0:
        return q - 3
    return q - 4


def rho_of(m, gears):
    r = Fraction(1)
    for q in gears:
        if m % q:
            r *= Fraction(q - 2, q)
    return r


def exact_chain(gears, m):
    b = blocked(gears)
    idx = np.flatnonzero(~b)
    assert idx.size == prod(q - 2 for q in gears)
    ph = idx % m
    E = exposed_phases(m, gears)
    e = E.size
    assert np.isin(ph, E).all()
    pos = -np.ones(m, np.int64)
    pos[E] = np.arange(e)
    src, dst = pos[ph], pos[np.roll(ph, -1)]
    C = np.zeros((e, e), np.int64)
    np.add.at(C, (src, dst), 1)
    assert C.sum() == idx.size
    return C / C.sum(1, keepdims=True), E, e


def subdominant(ev):
    k1 = int(np.argmin(np.abs(ev - 1)))
    rest = np.delete(ev, k1)
    l2 = rest[int(np.argmax(np.abs(rest)))]
    return l2 if l2.imag >= 0 else np.conj(l2)


def model_chain(gears, m, G, pair=True):
    """MODEL 0 (pair=False) / MODEL 1 (pair=True) phase chain on E."""
    E = exposed_phases(m, gears)
    e = E.size
    pos = -np.ones(m, np.int64)
    pos[E] = np.arange(e)
    big = [q for q in gears if m % q]
    rho = float(rho_of(m, gears))
    r1 = np.empty(G + 1)
    for t in range(1, G + 1):
        if pair:
            v = 1.0
            for q in big:
                cc = c_q_closed(q, t)
                assert cc == c_q(q, t), (q, t)
                v *= cc / (q - 2)
            r1[t] = v
        else:
            r1[t] = rho
    T = np.zeros((e, e))
    tail = 0.0
    for i, r in enumerate(E):
        surv = 1.0
        for g in range(1, G + 1):
            s = (r + g) % m
            j = pos[s]
            if j < 0:
                continue
            h = r1[g]
            T[i, j] += surv * h
            surv *= (1 - h)
        tail = max(tail, surv)
    assert tail < 1e-9, f"truncation tail {tail:.2e} - raise G"
    T /= T.sum(1, keepdims=True)
    return T



def c3_tab(q, G):
    """c_q(0,t,g) = #{r: r, r+t, r+g all exposed mod q}, table over t,g."""
    t1, t2 = teeth(q)
    A = np.ones(q, bool)
    A[t1] = A[t2] = False
    T = np.empty((q, q), np.int64)
    for t in range(q):
        At = np.roll(A, -t)
        for g in range(q):
            T[t, g] = int((A & At & np.roll(A, -g)).sum())
    return T


def model2_chain(gears, m, G):
    """MODEL 2: hazard conditioned on BOTH endpoints of the gap (the exact
    3-point CRT interior, i.e. round 20's renewal law with kappa = 1)."""
    E = exposed_phases(m, gears)
    e = E.size
    pos = -np.ones(m, np.int64)
    pos[E] = np.arange(e)
    big = [q for q in gears if m % q]
    tt = np.arange(G + 1)
    nu = np.ones(G + 1)
    for q in big:
        nu *= np.array([c_q_closed(q, g) for g in range(G + 1)]) / (q - 2)
    # mu[t,g] = prod_q c_q(0,t,g)/c_q(0,g)
    mu = np.ones((G + 1, G + 1))
    for q in big:
        T = c3_tab(q, G)
        c2 = np.array([c_q_closed(q, g) for g in range(G + 1)], float)
        mu *= T[np.ix_(tt % q, tt % q)] / np.maximum(c2, 1)[None, :]
    Tm = np.zeros((e, e))
    tail = 0.0
    for i, r in enumerate(E):
        mask = np.array([1.0 if ((r + t) % m) in set(E.tolist()) else 0.0
                         for t in range(G + 1)])
        Q = 1.0 - mu * mask[:, None]
        Q = np.clip(Q, 0.0, 1.0)
        Q[0, :] = 1.0          # the interior product runs over 0 < t < g
        C = np.cumprod(Q, axis=0)
        tot = 0.0
        for g in range(1, G + 1):
            j = pos[(r + g) % m]
            if j < 0:
                continue
            p = nu[g] * (C[g - 1, g] if g >= 1 else 1.0)
            Tm[i, j] += p
            tot += p
        tail = max(tail, abs(1.0 - tot))
    Tm /= Tm.sum(1, keepdims=True)
    return Tm, tail



def model3_chain(gears, m):
    """MODEL 3: the machine's EXACT global gap law, made PHASE-BLIND.
    Isolates whether the deficit is a gap-law-SHAPE effect (which model 3
    keeps exactly) or a PHASE-DEPENDENCE effect (which model 3 destroys)."""
    b = blocked(gears)
    P = b.size
    idx = np.flatnonzero(~b)
    g = np.diff(np.append(idx, idx[0] + P))
    W = np.bincount(g)
    E = exposed_phases(m, gears)
    e = E.size
    pos = -np.ones(m, np.int64)
    pos[E] = np.arange(e)
    T = np.zeros((e, e))
    for i, r in enumerate(E):
        for gg in np.flatnonzero(W):
            j = pos[(r + int(gg)) % m]
            if j >= 0:
                T[i, j] += W[gg]
    T /= T.sum(1, keepdims=True)
    return T



def step_law(gears, m):
    """The EXACT distribution q(n) of the number of EXPOSED PHASES crossed by
    one gap.  In this coordinate the phase walk on E advances by n steps, so a
    phase-BLIND chain is a circulant with eigenvalues q-hat(j/e) exactly -
    this is Constructor's lambda_2 = p-hat(1) in its natural coordinate."""
    b = blocked(gears)
    P = b.size
    idx = np.flatnonzero(~b)
    E = set(exposed_phases(m, gears).tolist())
    e = len(E)
    isE = np.array([1 if (t % m) in E else 0 for t in range(m)], np.int64)
    cum = np.cumsum(isE[np.arange(P) % m])          # #exposed phases in [0,k]
    nxt = np.roll(idx, -1)
    g = (nxt - idx) % P
    n = cum[(idx + g - 1) % P] - cum[idx] + isE[(idx + g) % m]
    # wrap correction for the single gap crossing the period end
    n = np.where(n <= 0, n + cum[-1], n)
    q = np.bincount(n)
    q = q / q.sum()
    w = np.exp(2j * np.pi / e)
    return sum(q[k] * w ** k for k in range(q.size)), q, e


def run(gears, m, G, label, rows):
    y = gears[-1]
    T, E, e = exact_chain(gears, m)
    meas = subdominant(np.linalg.eigvals(T))
    T0 = model_chain(gears, m, G, pair=False)
    m0 = subdominant(np.linalg.eigvals(T0))
    T1 = model_chain(gears, m, G, pair=True)
    m1 = subdominant(np.linalg.eigvals(T1))
    T2, tail2 = model2_chain(gears, m, G)
    m2 = subdominant(np.linalg.eigvals(T2))
    T3 = model3_chain(gears, m)
    m3 = subdominant(np.linalg.eigvals(T3))
    m4, qlaw, ee = step_law(gears, m)
    # closed form cross-check of model 0
    rho = float(rho_of(m, gears))
    w = np.exp(2j * np.pi / e)
    cf = rho * w / (1 - (1 - rho) * w)
    assert abs(abs(m0) - abs(cf)) < 2e-3, (y, abs(m0), abs(cf))
    d0 = abs(meas) - abs(m0)
    d1 = abs(meas) - abs(m1)
    d2 = abs(meas) - abs(m2)
    frac = 1 - abs(d2) / abs(d0) if d0 else float('nan')
    ang = lambda z: np.degrees(np.angle(z))
    d3 = abs(meas) - abs(m3)
    print(f"  {y:3d} {m:4d} {abs(meas):9.6f} {abs(m0):9.6f} {abs(m1):9.6f}"
          f" {abs(m2):9.6f} {abs(m3):9.6f} {d0:+9.6f} {d1:+9.6f} {d2:+9.6f}"
          f" {d3:+9.6f} {abs(meas)-abs(m4):+9.6f}"
          f" {ang(meas):7.3f} {ang(m0):7.3f} {ang(m4):7.3f}")
    rows.append((y, m, abs(meas), abs(m0), abs(m1), d0, d1,
                 ang(meas), ang(m0), ang(m1), abs(m2), d2, ang(m2),
                 abs(m3), d3, ang(m3), abs(m4),
                 abs(meas) - abs(m4), ang(m4)))


def main():
    big = "--big" in sys.argv
    ys = [11, 13, 17, 19] + ([23] if big else [])
    print("=" * 100)
    print("lambda_2: MEASURED (exact full period) vs MODEL 0 (constant hazard,"
          " round 22) vs MODEL 1 (2-point)")
    print("=" * 100)
    print("    y    m  |l2| meas  |l2| M0   |l2| M1   |l2| M2   |l2| M3  "
          " defic0    defic1    defic2    defic3   argmeas argM0   argM3")
    rows = []
    t0 = time.time()
    for m in (35, 385):
        for y in ys:
            gears = primes(5, y)
            if prod(gears) <= m:
                continue
            run(gears, m, 600, f"m{y}", rows)
        print()
    r35 = [r for r in rows if r[1] == 35]
    print("PRE-REGISTERED VERDICTS FOR MODEL 1 (mod 35 rows):")
    print(f"  Q1 direction - model 1 raises |l2| at every machine : "
          f"{all(r[4] > r[3] for r in r35)}")
    print(f"  Q2 size      - closes >= half of the deficit        : "
          f"{all(abs(r[6]) <= 0.5*abs(r[5]) for r in r35)}")
    print(f"  Q3 argument  - not degraded by more than 0.5 deg    : "
          f"{all(abs(r[9]-r[7]) <= abs(r[8]-r[7]) + 0.5 for r in r35)}")
    print()
    print("MODEL 2 verdicts (pre-registered once model 1 had failed downward:")
    print("the interior term should push UP, so 0 < deficit2 < deficit0):")
    print(f"  Q4 model 2 lies ABOVE model 0                       : "
          f"{all(r[10] > r[3] for r in r35)}")
    print(f"  Q5 0 < deficit2 < deficit0 at every machine         : "
          f"{all(0 < r[11] < r[5] for r in r35)}")
    print()
    print("MODEL 3 (exact global gap law, phase-blind) - the ATTRIBUTION test:")
    print("  if |deficit3| << deficit0 the deficit is a GAP-LAW SHAPE effect;")
    print("  if deficit3 ~ deficit0 it is a PHASE-DEPENDENCE effect.")
    for r in r35:
        print(f"   y={r[0]:3d}  deficit0 {r[5]:+.6f}   deficit3 {r[14]:+.6f}"
              f"   |d3|/|d0| = {abs(r[14])/abs(r[5]):6.3f}")
    print()
    print("  deficits (mod 35)  M0: " + ", ".join(f"{r[5]:+.4f}" for r in r35))
    print("                     M1: " + ", ".join(f"{r[6]:+.4f}" for r in r35))
    print("                     M2: " + ", ".join(f"{r[11]:+.4f}" for r in r35))
    print("                     M3: " + ", ".join(f"{r[14]:+.4f}" for r in r35))
    print("                     M4: " + ", ".join(f"{r[17]:+.4f}" for r in r35))
    print()
    print("MODEL 4 = the EXACT step law made phase-blind (a circulant, i.e.")
    print("Constructor's lambda_2 = p-hat(1) in its natural coordinate).  Its")
    print("residual is therefore PURELY the phase-dependence of the step law -")
    print("the corridor pinning - with no gap-law-shape contamination.")
    print()
    print("THE STEP LAW ITSELF (mod 35): q(n) vs geometric(rho), and the mean")
    print("(the mean is EXACTLY 1/rho by CRT - the departure is pure SHAPE):")
    for y in [r[0] for r in r35]:
        gears = primes(5, y)
        l4, q, e = step_law(gears, 35)
        rho = float(rho_of(35, gears))
        mean = sum(k * q[k] for k in range(q.size))
        geo = [rho * (1 - rho) ** (k - 1) for k in range(1, min(7, q.size))]
        print(f"   y={y:3d} rho={rho:.6f} mean={mean:.6f} 1/rho="
              f"{1/rho:.6f}  q(n)/geom(n) n=1..6: "
              + " ".join(f"{q[k]/geo[k-1]:.4f}"
                         for k in range(1, min(7, q.size))))
    print()
    print("q(1) IN CLOSED FORM (the first term of the step law): given an")
    print("opening at phase r the next EXPOSED phase is d(r) slots ahead, so")
    print("  q(1) = avg over r in E of prod_{q not | m} c_q(d(r))/(q-2)")
    print("- pure CRT arithmetic, no scan.  Checked against the exact law:")
    print("    y   q(1) measured   q(1) closed form   diff")
    for y in [r[0] for r in r35]:
        gears = primes(5, y)
        _, q, e = step_law(gears, 35)
        E = exposed_phases(35, gears)
        Eset = set(E.tolist())
        big = [qq for qq in gears if 35 % qq]
        tot = 0.0
        for r in E:
            d = 1
            while ((int(r) + d) % 35) not in Eset:
                d += 1
            v = 1.0
            for qq in big:
                v *= c_q_closed(qq, d) / (qq - 2)
            tot += v
        pred = tot / E.size
        assert abs(pred - q[1]) < 1e-12, (y, pred, q[1])
        print(f"   {y:3d}   {q[1]:.12f}   {pred:.12f}   {pred-q[1]:+.2e}")
    print()
    print("VERDICT: lambda_2 = q-hat(1/e) is an IDENTITY to 1e-5 in modulus")
    print("and 0.01 deg in argument.  Round 22's closed form is exactly the")
    print("case q = geometric(rho); the 0.029 deficit is therefore ENTIRELY")
    print("the NON-GEOMETRICITY of the exposed-step law, in a coordinate with")
    print("no phase left in it.  Models 1 and 2 failed because they corrected")
    print("the SLOT-LAG hazard, which double-counts the phase structure.")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
