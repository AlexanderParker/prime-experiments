"""
LATERAL round 27 - U6 and U9, unblocked by Mechanic's EXACT full-period m37 gap
histogram (research/data/r26/ghist_37.csv, cyclically closed by construction).

U6: does alpha_1/alpha_2 at gear 5 keep rising past -1/phi (permanent overshoot)
    or turn back (oscillation about the golden direction)?
U9: the amplitude plateau |H_5(1)|/N * lam - does it break UP (round-25's
    corridor-renewal ladder) or DOWN (the round-21 closed-form M1 model)?

Both were "a five-minute computation the moment the array exists" (round-26 note).
Everything here is exact integer arithmetic on the histogram plus one complex sum.

Usage: python ghist37_u69.py
"""
import cmath
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def gears_upto(y):
    return [q for q in GEARS if q <= y]


def hist_from_scan(y):
    """Direct full-period sieve - only used for m11 (P = 385)."""
    gs = gears_upto(y)
    P = 1
    for q in gs:
        P *= q
    t = [set(((pow(6, -1, q)) % q, (-pow(6, -1, q)) % q)) for q in gs]
    op = [k for k in range(P) if all(k % q not in tt for q, tt in zip(gs, t))]
    h = {}
    for i in range(len(op)):
        g = (op[(i + 1) % len(op)] - op[i]) % P
        h[g] = h.get(g, 0) + 1
    return h, P


def hist_from_csv(path):
    h = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            h[int(row["gap"])] = int(row["count"])
    return h


def stats(y, h):
    gs = gears_upto(y)
    P = 1
    for q in gs:
        P *= q
    Ntrue = 1
    for q in gs:
        Ntrue *= q - 2
    N = sum(h.values())
    tot = sum(g * c for g, c in h.items())
    # gear-5 residue classes
    Nr = [0] * 5
    for g, c in h.items():
        Nr[g % 5] += c
    beta = [Nr[(r + 1) % 5] - Nr[r] for r in range(5)]
    a1 = beta[1] - beta[4]
    a2 = beta[2] - beta[3]
    om = cmath.exp(2j * cmath.pi / 5)
    H = sum(c * om ** (g % 5) for g, c in h.items())
    lam = P / N
    return dict(y=y, P=P, N=N, Ntrue=Ntrue, tot=tot, Nr=Nr, a1=a1, a2=a2,
                ratio=a1 / a2, H=H, argH=cmath.phase(H) * 180 / cmath.pi,
                absH=abs(H), lam=lam, amp=abs(H) / N * lam, F=max(h))


def main():
    phi = (1 + 5 ** 0.5) / 2
    rows = []

    h11, _ = hist_from_scan(11)
    rows.append(stats(11, h11))
    for y in (13, 17, 19, 23, 29, 31, 37):
        p = os.path.join(DATA, "r26", "ghist_%d.csv" % y)
        rows.append(stats(y, hist_from_csv(p)))

    print("=== GATES on the exact cyclic histograms ===")
    for r in rows:
        gate(r["N"] == r["Ntrue"],
             "m%-2d: total gap count = prod(q-2) = %d" % (r["y"], r["Ntrue"]))
        gate(r["tot"] == r["P"], "m%-2d: sum of gaps = P = %d" % (r["y"], r["P"]))
    # item 53: gap 1 is the ONLY odd entry
    for y, h in [(11, h11)] + [(y, hist_from_csv(os.path.join(DATA, "r26", "ghist_%d.csv" % y)))
                               for y in (13, 17, 19, 23, 29, 31, 37)]:
        odd = sorted(g for g, c in h.items() if c % 2)
        gate(odd == [1], "m%-2d: the only ODD histogram entry is gap 1 (item 53)" % y)

    print("\n=== U6: the gear-5 asymmetry ratio alpha_1/alpha_2 vs -1/phi ===")
    print("  -1/phi = %.6f" % (-1 / phi))
    print("   y    N_0..N_4 (mod 5 classes of the gap)                  alpha_1     alpha_2     ratio      dev from -1/phi")
    for r in rows:
        print("  %-4d %-52s %-11d %-11d %+.6f  %+.6f"
              % (r["y"], r["Nr"], r["a1"], r["a2"], r["ratio"], r["ratio"] + 1 / phi))
    for r in rows:
        gate(r["a1"] % 2 == 1, "m%-2d: alpha_1 is ODD (item 56)" % r["y"])

    print("\n=== U9: the amplitude plateau |H_5(1)|/N * lam ===")
    print("   y    lam        |H|/N      arg H (deg)   amplitude |H|/N*lam")
    for r in rows:
        print("  %-4d %-10.5f %-10.6f %-13.5f %.6f"
              % (r["y"], r["lam"], r["absH"] / r["N"], r["argH"], r["amp"]))

    # cross-gate against the published round-26 arg ladder
    published = {13: 129.776, 17: 127.808, 19: 126.334, 23: 126.352, 29: 126.059,
                 31: 125.768, 37: 125.659}
    for r in rows:
        if r["y"] in published:
            gate(abs(r["argH"] - published[r["y"]]) < 5e-3,
                 "m%-2d: arg H_5(1) = %.3f matches Mechanic's exact ladder %.3f"
                 % (r["y"], r["argH"], published[r["y"]]))
    # and against this lane's own round-25 amplitude table
    amp25 = {11: 1.1260, 13: 1.0362, 17: 1.0150, 19: 1.0139, 23: 1.0193, 29: 1.0161}
    for r in rows:
        if r["y"] in amp25:
            gate(abs(r["amp"] - amp25[r["y"]]) < 6e-4,
                 "m%-2d: amplitude %.4f matches round-25 table %.4f"
                 % (r["y"], r["amp"], amp25[r["y"]]))

    print("\n=== VERDICTS ===")
    r31 = [r for r in rows if r["y"] == 31][0]
    r37 = [r for r in rows if r["y"] == 37][0]
    r29 = [r for r in rows if r["y"] == 29][0]
    print("  U6  ratio m29 %+.6f  m31 %+.6f  m37 %+.6f   (-1/phi = %.6f)"
          % (r29["ratio"], r31["ratio"], r37["ratio"], -1 / phi))
    print("      increments: m29->m31 %+.6f, m31->m37 %+.6f"
          % (r31["ratio"] - r29["ratio"], r37["ratio"] - r31["ratio"]))
    print("  U9  amplitude m29 %.6f  m31 %.6f  m37 %.6f"
          % (r29["amp"], r31["amp"], r37["amp"]))
    print("      break direction m31 -> m37: %s"
          % ("DOWN" if r37["amp"] < r31["amp"] else "UP"))
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
