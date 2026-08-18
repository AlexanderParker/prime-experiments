"""Record-growth law for saturated runs (round 8 analysis).

Model: per-slot rate of a maximal load-1 run of length L starting near
member m is rho_L(m) ~ A_L * s(m)^L, with s(m) = 2p(1-p), p = 3/ln m (the
per-slot probability of exactly one prime member, independence baseline).
A_L absorbs constellation/boundary structure. This script:
  - merges round-7 and round-8 renewal tables (decade counts),
  - fits A_L per decade (stability check = model quality),
  - checks the L=13 instance list against C * integral of s^13,
  - predicts the first L=14 arrival member by (a) A_L extrapolation and
    (b) per-decade N(L+1)/N(L) ratio extrapolation.
All fits labeled fits. Usage: uv run python research/satruns_model.py
"""
import csv
import os
from math import log

DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def s_of_m(m):
    p = 3.0 / log(m)
    return 2.0 * p * (1.0 - p)


def integral_sL(k_lo, k_hi, L, pts=2000):
    """integral over slots k in [k_lo, k_hi] of s(6k)^L (log grid)."""
    if k_hi <= k_lo:
        return 0.0
    tot = 0.0
    import numpy as np
    ks = np.geomspace(max(k_lo, 2), k_hi, pts)
    vals = np.array([s_of_m(6.0 * k) ** L for k in ks])
    return float(np.trapezoid(vals, ks))


def load_renewal():
    """combined decade -> (slots, counts dict L->n). Round 7 (satruns_renewal,
    to member 7.2e10) + round 8 (…_r8, member 7.2e10..1.002e12) + the running
    chunked scan (…_renewal, new schema k_from,k_to - slots computed as the
    chunk's overlap with the decade's k-range)."""
    dec = {}
    for fn in ("satruns_renewal.csv", "satruns_deep_renewal_r8.csv",
               "satruns_deep_renewal.csv"):
        path = os.path.join(DDIR, fn)
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            d = int(r["decade"])
            if "k_from" in r:  # new chunked schema
                ka, kb = int(r["k_from"]), int(r["k_to"])
                lo = max(ka, (10 ** d + 1) // 6)
                hi = min(kb, (10 ** (d + 1) + 1) // 6)
                slots = max(0, hi - lo + 1)
            else:
                slots = int(r.get("slots_in_decade")
                            or r.get("slots_scanned_in_decade"))
            row = dec.setdefault(d, {"slots": 0, "n": {}})
            row["slots"] += slots
            for Lname, L in (("L8", 8), ("L9", 9), ("L10", 10),
                             ("L11", 11), ("L12", 12), ("L13plus", 13)):
                row["n"][L] = row["n"].get(L, 0) + int(r[Lname])
    return dec


def main():
    dec = load_renewal()
    K7 = 12_000_000_000
    K8_path = os.path.join(DDIR, "satruns_deep_renewal.csv")
    have8 = os.path.exists(K8_path)
    print("combined per-decade counts (L13plus = L >= 13):")
    print(f"{'d':>3} {'slots':>14} {'L8':>7} {'L9':>6} {'L10':>6} "
          f"{'L11':>5} {'L12':>5} {'L13+':>5} {'r(9/8)':>7} {'r(13/12)':>8}")
    for d in sorted(dec):
        n = dec[d]["n"]
        r98 = n.get(9, 0) / n[8] if n.get(8) else float("nan")
        r1312 = (n.get(13, 0) / n[12]) if n.get(12) else float("nan")
        print(f"{d:>3} {dec[d]['slots']:>14} {n.get(8,0):>7} {n.get(9,0):>6} "
              f"{n.get(10,0):>6} {n.get(11,0):>5} {n.get(12,0):>5} "
              f"{n.get(13,0):>5} {r98:>7.3f} {r1312:>8.3f}")
    # A_L fits per decade (use decade k-ranges actually scanned)
    print("\nA_L = N_d(L) / integral s^L, per decade (stability check):")
    print(f"{'d':>3}" + "".join(f"{f'A_{L}':>10}" for L in range(8, 14)))
    A_global = {}
    for L in range(8, 14):
        num = den = 0.0
        for d in sorted(dec):
            k_lo = max(1, (10 ** d + 1) // 6)
            k_hi = min((10 ** (d + 1) + 1) // 6,
                       167_000_000_000 if have8 else K7)
            n = dec[d]["n"].get(L, 0)
            I = integral_sL(k_lo, k_hi, L)
            num += n
            den += I
        A_global[L] = num / den if den else float("nan")
    for d in sorted(dec):
        k_lo = max(1, (10 ** d + 1) // 6)
        k_hi = min((10 ** (d + 1) + 1) // 6,
                   167_000_000_000 if have8 else K7)
        row = f"{d:>3}"
        for L in range(8, 14):
            I = integral_sL(k_lo, k_hi, L)
            n = dec[d]["n"].get(L, 0)
            row += f"{(n / I if I else float('nan')):>10.3f}"
        print(row)
    print("global A_L: " + "  ".join(f"A_{L}={A_global[L]:.3f}"
                                     for L in range(8, 14)))
    # prediction: first L=14
    # (a) extrapolate A_14 from log-linear trend of A_L
    import numpy as np
    Ls = np.array([L for L in range(8, 14) if A_global[L] > 0])
    lgA = np.array([log(A_global[L]) for L in Ls])
    sl, ic = np.polyfit(Ls, lgA, 1)
    A14 = float(np.exp(sl * 14 + ic))
    # cumulative expected count of L=14 up to member M: A14 * int s^14
    for M in (1e11, 1e12, 1e13, 1e14):
        I = integral_sL(2, M / 6, 14)
        print(f"expected # L=14 up to member {M:.0e}: {A14 * I:.2f} "
              f"(A_14 extrapolated = {A14:.3f}, slope {sl:.3f}/L)")
    # find M with expected count = 1
    lo, hi = 1e10, 1e18
    while hi / lo > 1.05:
        mid = (lo * hi) ** 0.5
        if A14 * integral_sL(2, mid / 6, 14) < 1:
            lo = mid
        else:
            hi = mid
    print(f"first L=14 expected near member ~{(lo*hi)**0.5:.2e} "
          f"[A-extrapolation; fit, not law]")

    # ---- record ladder vs the CRT cap [13, 32] (Lateral round 8) ----
    print("\nrecord ladder: predicted first-arrival member M(L) with "
          "A_L = exp(%.3f*L%+.3f) [extrapolated fit]:" % (sl, ic))
    print(f"{'L':>3} {'A_L':>9} {'M(L) first arrival':>19}")
    for L in range(14, 33):
        AL = float(np.exp(sl * L + ic))
        lo, hi = 1e6, 1e120
        while hi / lo > 1.1:
            mid = (lo * hi) ** 0.5
            if AL * integral_sL(2, mid / 6, L, pts=4000) < 1:
                lo = mid
            else:
                hi = mid
        M = (lo * hi) ** 0.5
        note = "  <- CRT cap (absolute ceiling)" if L == 32 else ""
        print(f"{L:>3} {AL:>9.4f} {M:>19.2e}{note}")

    # ---- renewal-rate exponent refit: rate(L>=8) ~ C/(ln m)^beta ----
    print("\nrenewal refit: per-slot rate of L>=8 runs vs 1/(ln m)^beta:")
    xs, ys_ = [], []
    for d in sorted(dec):
        if d < 5:
            continue  # small-count decades
        n8 = sum(dec[d]["n"].get(L, 0) for L in range(8, 14))
        if n8 < 10 or dec[d]["slots"] <= 0:
            continue
        m_mid = 10 ** (d + 0.5)
        xs.append(log(log(m_mid)))
        ys_.append(log(n8 / dec[d]["slots"]))
    if len(xs) >= 3:
        b_, c_ = np.polyfit(xs, ys_, 1)
        pred = np.polyval([b_, c_], xs)
        resid = float(np.abs(np.array(ys_) - pred).max())
        print(f"  beta = {-b_:.2f} (naive independence predicts ~8 for "
              f"L>=8 mix), C = e^{c_:.2f}, max ln-resid {resid:.3f} over "
              f"{len(xs)} decades [fit]")


if __name__ == "__main__":
    main()
