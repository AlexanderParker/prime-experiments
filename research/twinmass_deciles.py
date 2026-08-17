"""Depth-resolved twin mass vs product baseline (mechanic round 6, part 3).

Round 5 found the real joint zero-mass (twin share n0/W) at 0.77-0.85 of
its product-model baseline P(omega_L=0)*P(omega_R=0) (independent per-gear
classes), globally. Here the ratio is resolved by depth decile (10 equal
slot bands of the full window). The product baseline is depth-UNIFORM by
construction (class densities do not depend on t), so any depth structure
in the ratio is real machine structure. For shape comparison the table
also gives the HL-model column ~ mean over the decile of 1/ln^2(member),
normalised to match the global real mass.

Output (append): research/data/twinmass_deciles.csv. Usage:
uv run python research/twinmass_deciles.py [y...]   (default 10007 50021)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime


def run(y, seg=16_000_000):
    gears = [q for q in primes_upto(y) if q >= 5]
    uvals = [pow(6, -1, q) for q in gears]
    k_lo = -((-(y - 1)) // 6)
    k_hi = (y * y + 1) // 6
    W = k_hi - k_lo + 1
    edges = [k_lo + (W * i) // 10 for i in range(11)]
    n0 = np.zeros(10, dtype=np.int64)
    slots = np.zeros(10, dtype=np.int64)
    inv_ln2 = np.zeros(10)
    for a in range(k_lo, k_hi + 1, seg):
        b = min(k_hi + 1, a + seg)
        n = b - a
        cntL = np.zeros(n, np.int16)
        cntR = np.zeros(n, np.int16)
        for q, u in zip(gears, uvals):
            cntL[(u - a) % q::q] += 1
            cntR[(-u - a) % q::q] += 1
        if a == k_lo:
            for arr, m in ((cntL, 6 * k_lo - 1), (cntR, 6 * k_lo + 1)):
                if m <= y and is_prime(m):
                    arr[0] = 0
        z = (cntL == 0) & (cntR == 0)
        kk = np.arange(a, b, dtype=np.int64)
        dec = np.minimum(np.searchsorted(edges, kk, side="right") - 1, 9)
        n0 += np.bincount(dec, weights=z, minlength=10).astype(np.int64)
        slots += np.bincount(dec, minlength=10)
        inv_ln2 += np.bincount(dec, weights=1.0 / np.log(6.0 * kk) ** 2,
                               minlength=10)
    # product baseline: depth-uniform
    both0 = 1.0
    for sgn in (1, -1):
        pr = 1.0
        for q, u in zip(gears, uvals):
            rr = (sgn * u) % q
            c = (k_hi - rr) // q - (k_lo - 1 - rr) // q
            pr *= 1.0 - c / W
        both0 *= pr
    return dict(y=y, W=W, k_lo=k_lo, n0=n0, slots=slots,
                inv_ln2=inv_ln2, both0=both0)


def main():
    ys = [int(a) for a in sys.argv[1:]] or [10007, 50021]
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, "twinmass_deciles.csv")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a")
    if new:
        f.write("y,decile,slots,n0,share,ratio_vs_product_baseline,"
                "hl_shape_norm\n")
    for y in ys:
        t0 = time.time()
        r = run(y)
        both0 = r["both0"]
        tot_real = r["n0"].sum()
        hl = r["inv_ln2"] / r["inv_ln2"].sum() * tot_real  # HL-shaped alloc
        print(f"y={y} both0_baseline={both0:.6f} global_ratio="
              f"{tot_real / r['W'] / both0:.4f}  ({time.time()-t0:.0f}s)")
        print(f"{'dec':>4} {'slots':>10} {'n0':>9} {'share':>8} "
              f"{'ratio_base':>10} {'hl_norm':>9} {'real/hl':>8}")
        for d in range(10):
            share = r["n0"][d] / r["slots"][d]
            ratio = share / both0
            f.write(f"{y},{d},{r['slots'][d]},{r['n0'][d]},{share:.6f},"
                    f"{ratio:.4f},{hl[d]:.0f}\n")
            print(f"{d:>4} {r['slots'][d]:>10} {r['n0'][d]:>9} {share:>8.4f} "
                  f"{ratio:>10.4f} {hl[d]:>9.0f} {r['n0'][d]/hl[d]:>8.4f}")
        sys.stdout.flush()
    f.close()
    print("wrote twinmass_deciles.csv")


if __name__ == "__main__":
    main()
