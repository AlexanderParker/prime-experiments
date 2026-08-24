"""Round 21 lateral, target (b): EIGENVALUE SPACING STATISTICS OF THE MACHINE
OPERATORS vs GUE / Poisson - the human's Riemann-bridge hunch, tested.

Background: Montgomery-Odlyzko - the Riemann zeros' pair correlation /
nearest-neighbour spacings match GUE random-matrix eigenvalues; a
Hilbert-Polya operator would make that spectral.  Our machine's exact
operators are finite shadows of the gear train.  Their spectra are CLOSED
FORM, so the spacing statistics of Jacobsthal-type operators are computable
exactly - apparently never done.

THE OPERATORS AND THEIR EXACT SPECTRA:
1. Machine circulant C_M (symmetric, integer): eigenvalues
       lambda(j_5,..,j_y) = prod_q h_q(j_q),
       h_q(0) = q-2,  h_q(j) = -2 cos(2 pi j u_q / q)  (j != 0),
   the full spectrum is the P-fold tensor/product multiset (matrix-
   formulation piece 7).  As j runs over Z_q the nonzero-frequency values
   are -2cos(2 pi k/q), k = 1..q-1, with the pairing k <-> q-k: every gear
   contributes (q-1)/2 doubly-degenerate lines + one simple line (q-2).
   MIRROR SYMMETRY k -> -k therefore forces 2^m-fold degeneracies
   (m = # gears at nonzero local frequency) in the full spectrum.
   DESYMMETRIZED spectrum: k_q in {0} u {1..(q-1)/2}, size prod (q+1)/2.
   Exact det trivia: prod_{k=1}^{q-1} (-2cos(2 pi k/q)) = 1 for odd q, so
   det C_M = prod (q-2) = the open count (asserted numerically below).
2. The machine's unitaries are EXACT CLOCKS - no numerics needed:
   the slot shift S_P is a single P-cycle, the renewal operator R (constructor
   R35) is a single |E|-cycle permutation; eigenphases = ALL n-th roots of
   unity, exactly equidistributed.  Spacing distribution = delta(s-1),
   spacing-ratio r = 1 exactly: PICKET FENCE / "clock", the RIGID extreme.

PRE-REGISTERED EXPECTATION (stated before running): tensor-product spectra
are the canonical integrable / Berry-Tabor case -> POISSON (or below, from
residual clustering), NOT GUE.  GUE would need level repulsion, i.e.
interaction between the gear factors.  Drift TOWARD GUE as machines grow
would support the bridge; Poisson/clock at both ends refutes it at finite
machines.

STATISTICS (floats, labeled; eigenvalues themselves closed-form):
- consecutive-spacing-ratio r~_i = min(s_i, s_{i+1})/max(s_i, s_{i+1}),
  unfolding-free (Atas-Bogomolny-Giraud-Roux 2013):
      <r~> Poisson = 2 ln 2 - 1 = 0.38629,  GOE = 0.53590,  GUE = 0.60266.
  (C_M is real symmetric, so GOE is the relevant RMT target; GUE listed
  because the Riemann statistic is GUE.)
- unfolded nearest-neighbour P(s) (local-mean unfolding, window w), KS
  distance to Poisson e^{-s} vs Wigner GOE (pi s/2) e^{-pi s^2/4}.
- degeneracy census: exact 2^m mirror multiplicities (predicted vs counted)
  and accidental collisions inside the desymmetrized list.

Usage: python eig_stats.py            # machines 11..29 desym + 11..23 full
       python eig_stats.py --big      # adds machine 31 desym (1 GB, alone)
"""
import sys
from math import prod, pi, cos, log
import numpy as np

RT_POISSON = 2 * log(2) - 1          # 0.386294
RT_GOE = 0.53590
RT_GUE = 0.60266


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))]


def desym_vals(q):
    return np.array([float(q - 2)] +
                    [-2 * cos(2 * pi * k / q) for k in range(1, (q + 1) // 2)])


def full_vals(q):
    return np.array([float(q - 2)] +
                    [-2 * cos(2 * pi * k / q) for k in range(1, q)])


def build(y, vals):
    spec = np.array([1.0])
    for q in primes(5, y):
        spec = np.multiply.outer(spec, vals(q)).ravel()
    return spec


def rtilde(spec_sorted):
    s = np.diff(spec_sorted)
    s = s[s > 0]  # guard: exact ties excluded (reported separately)
    r = s[1:] / s[:-1]
    rt = np.minimum(r, 1.0 / r)
    return float(rt.mean()), rt


def unfolded_spacings(spec_sorted, w=101):
    """local-mean unfolding: spacing / centered moving average of spacings."""
    s = np.diff(spec_sorted)
    # moving average via cumsum
    c = np.concatenate(([0.0], np.cumsum(s)))
    n = s.size
    h = w // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    local = (c[hi] - c[lo]) / (hi - lo)
    good = local > 0
    return s[good] / local[good]


def ks_to(cdf_emp_x, cdf_fun):
    """KS distance of sorted sample x to analytic cdf."""
    x = cdf_emp_x
    n = x.size
    F = cdf_fun(x)
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    return float(max(np.max(np.abs(F - emp_hi)), np.max(np.abs(F - emp_lo))))


def stats_for(y, kind, big_ok=True):
    vals = desym_vals if kind == "desym" else full_vals
    sizes = prod(((q + 1) // 2 if kind == "desym" else q)
                 for q in primes(5, y))
    spec = build(y, vals)
    assert spec.size == sizes
    spec.sort(kind="quicksort")
    # degeneracy census: ties at relative tolerance
    s = np.diff(spec)
    scale = np.maximum(np.abs(spec[:-1]), 1e-300)
    ties = int(np.sum(s <= 1e-12 * scale))
    rt_mean, rt = rtilde(spec)
    us = unfolded_spacings(spec)
    us_sorted = np.sort(us)
    ks_p = ks_to(us_sorted, lambda x: 1 - np.exp(-x))
    ks_goe = ks_to(us_sorted, lambda x: 1 - np.exp(-pi * x * x / 4))
    frac_small = float(np.mean(us < 0.1))    # repulsion probe: Poisson 0.095,
    #                                          GOE 0.0078
    return dict(y=y, n=spec.size, ties=ties, rt=rt_mean, ks_p=ks_p,
                ks_goe=ks_goe, frac_small=frac_small)


def main(big=False):
    print("=" * 78)
    print("EXACT statements first (no numerics): the machine's unitaries are "
          "CLOCKS -")
    print("  S_P is a single P-cycle, the renewal operator a single "
          "|E|-cycle permutation:")
    print("  eigenphases = all n-th roots of unity, spacing dist = "
          "delta(s-1), <r~> = 1.")
    print("  The RIGID extreme; GUE (0.603) lies strictly between clock "
          "(1.0) and Poisson (0.386).")
    print("=" * 78)
    print("det check: prod of nonzero-frequency eigenvalues = 1 per gear "
          "(det C_M = open count)")
    for q in primes(5, 31):
        pr = float(np.prod(full_vals(q)[1:]))
        assert abs(pr - 1.0) < 1e-9, (q, pr)
    print("  asserted for gears 5..31 (float, < 1e-9)")
    print("=" * 78)
    print("Machine circulant spectra: consecutive-spacing-ratio <r~> and "
          "unfolded P(s)")
    print("  references: Poisson %.5f | GOE %.5f | GUE %.5f | clock 1.0"
          % (RT_POISSON, RT_GOE, RT_GUE))
    print(f"  {'y':>4} {'kind':>6} {'levels':>11} {'ties':>9} {'<r~>':>8} "
          f"{'KS-Poisson':>11} {'KS-GOE':>8} {'P(s<0.1)':>9}")
    print("  (P(s<0.1): Poisson 0.0952, GOE 0.0078 - the repulsion probe)")
    rows = []
    for y in (11, 13, 17, 19, 23):
        for kind in ("desym", "full"):
            if kind == "full" and y > 23:
                continue
            d = stats_for(y, kind)
            rows.append((y, kind, d))
            print(f"  {y:>4} {kind:>6} {d['n']:>11} {d['ties']:>9} "
                  f"{d['rt']:>8.4f} {d['ks_p']:>11.4f} {d['ks_goe']:>8.4f} "
                  f"{d['frac_small']:>9.4f}")
    for y in (29, 31) if big else (29,):
        d = stats_for(y, "desym")
        rows.append((y, "desym", d))
        print(f"  {y:>4} {'desym':>6} {d['n']:>11} {d['ties']:>9} "
              f"{d['rt']:>8.4f} {d['ks_p']:>11.4f} {d['ks_goe']:>8.4f} "
              f"{d['frac_small']:>9.4f}")
    # mirror-degeneracy prediction check on full spectra: expected tie count
    # = P - prod(#distinct per gear) if no accidental cross-gear collisions
    print("  mirror-degeneracy check (full spectrum): ties = P - "
          "prod((q+1)/2) predicted")
    for y in (11, 13, 17):
        P = prod(primes(5, y))
        D = prod((q + 1) // 2 for q in primes(5, y))
        d = [r for r in rows if r[0] == y and r[1] == "full"][0][2]
        print(f"    y={y}: predicted {P - D}, counted {d['ties']}"
              + ("   EXACT" if d['ties'] == P - D else "   EXTRA accidental "
                 f"collisions: {d['ties'] - (P - D)}"))
    print("DONE")


if __name__ == "__main__":
    main(big="--big" in sys.argv)
