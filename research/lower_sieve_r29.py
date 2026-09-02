"""The lower sieve and the new gear - manager, round 29 (pre-registration:
data/r29/lower_sieve_prereg.md).

Human's construction: a machine = the lower sieve (gears found so far, period P) + the new
gear q. The lower sieve repeats up to q^2; the new gear then joins it for the next machine,
whose window runs to q_next^2. Read, on the new section (q^2, q_next^2):
  - the lower sieve's twin spots there (neither number has a factor below q),
  - which of them gear q bites (q divides one number) and which survive as new twins,
and, in the period, how gear q's lift-and-delete acts on the old openings.

Usage: python lower_sieve_r29.py [--qmax 5000]
"""
import argparse
from math import prod

import numpy as np

from word_tree_r29 import spf_sieve

NGATE = 0
NFAIL = 0


def gate(cond, msg):
    global NGATE, NFAIL
    NGATE += 1
    NFAIL += (not cond)
    print(("  GATE ok:   " if cond else "  GATE FAIL: ") + msg)


def teeth(g):
    u = pow(6, -1, g)
    return {u % g, (-u) % g}


def section_spots(q, q_next, spf):
    """lower-sieve twin spots of gears < q inside (q^2, q_next^2): both numbers have spf >= q."""
    k_lo, k_hi = q * q // 6 + 1, (q_next * q_next - 2) // 6
    ks = np.arange(k_lo, k_hi + 1)
    lo, hi = 6 * ks - 1, 6 * ks + 1
    f_lo, f_hi = spf[lo], spf[hi]
    spot = (f_lo >= q) & (f_hi >= q)
    twin = (f_lo == lo) & (f_hi == hi)
    bitten = spot & ~twin  # then the death rung is q (every composite below q_next^2 has a factor <= q)
    bites = ((lo % q == 0) | (hi % q == 0))  # number-kills of gear q, bitten spot or not
    return ks, lo, hi, spot, twin, bitten, bites


def period_lift(gears, q):
    """old openings of period P, their q lifts, how many lifts gear q deletes."""
    P = prod(gears)
    old_open = [a for a in range(P) if all(a % g not in teeth(g) for g in gears)]
    t = teeth(q)
    deleted = {}
    for a in old_open:
        lifts = [a + j * P for j in range(q)]
        deleted[a] = [k for k in lifts if k % q in t]
    return P, old_open, deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmax", type=int, default=5000)
    a = ap.parse_args()
    qmax = a.qmax
    primes = [int(x) for x in np.flatnonzero(spf_sieve(qmax + 200) == np.arange(qmax + 201)) if x >= 5]
    spf = spf_sieve(primes[primes.index(max(p for p in primes if p <= qmax)) + 1] ** 2 + 10).astype(np.int64)

    # ---- Q2 in the period: lift and delete, small q ----
    print("=== Q2, the period: gear q joins the lower sieve (lift to q copies, delete the class +-u_q) ===")
    l3 = True
    for q in (7, 11, 13):
        gears = [g for g in primes if g < q]
        P, old_open, deleted = period_lift(gears, q)
        counts = {a: len(deleted[a]) for a in old_open}
        l3 &= all(c == 2 for c in counts.values())
        print(f"  lower sieve {gears} (period {P}), openings {old_open}; gear {q} teeth {sorted(teeth(q))}:")
        for a in old_open:
            print(f"    opening {a:>2} mod {P}: deleted lifts k = {deleted[a]} ({len(deleted[a])} of {q}); survives in {q - len(deleted[a])} copies")
    gate(l3, "L3 each old opening is deleted in exactly 2 of its q lifts (q = 7, 11, 13)")

    # ---- Q1 in the section: listings ----
    print("\n=== Q1, the section: lower-sieve twin spots of gears < q inside (q^2, q_next^2), and what gear q does to them ===")
    for q in (5, 7, 11, 13, 47, 997):
        i = primes.index(q)
        q_next = primes[i + 1]
        ks, lo, hi, spot, twin, bitten, bites = section_spots(q, q_next, spf)
        print(f"  rung {q} -> {q_next}: section slots {ks[0]}..{ks[-1]} ({len(ks)} slots); lower sieve = gears < {q}")
        print(f"    lower-sieve twin spots: {int(spot.sum())}; gear {q} bites {int(bitten.sum())} of them; new twins {int(twin.sum())}")
        if q <= 47:
            print("    spots: " + ", ".join(
                f"k={int(k)} " + ("TWIN" if t else f"BITTEN ({int(l)}={q}*{int(l) // q}|{int(h)})" if l % q == 0 else f"BITTEN ({int(l)}|{int(h)}={q}*{int(h) // q})")
                for k, l, h, t in zip(ks[spot], lo[spot], hi[spot], twin[spot])))
        else:
            b = ks[bitten]
            print("    bitten spots: " + ", ".join(
                f"k={int(k)} ({int(l)}={q}*{int(l) // q}|{int(h)})" if l % q == 0 else f"k={int(k)} ({int(l)}|{int(h)}={q}*{int(h) // q})"
                for k, l, h in zip(b, lo[bitten], hi[bitten])) + (" (none)" if len(b) == 0 else ""))
            print(f"    new twins at k = {[int(k) for k in ks[twin]]}")
        print("    all number-kills of gear q in the section (bitten or masked by a smaller gear on the other side): " + ", ".join(
            f"k={int(k)} {'BITE' if bt else 'masked'} ({int(l)}={q}*{int(l) // q}|{int(h)})" if l % q == 0 else f"k={int(k)} {'BITE' if bt else 'masked'} ({int(l)}|{int(h)}={q}*{int(h) // q})"
            for k, l, h, bt in zip(ks[bites], lo[bites], hi[bites], bitten[bites])))

    # ---- gates over all rungs ----
    print(f"\n=== gates over the rungs q -> q_next with q_next <= {qmax} ===")
    l1 = l2 = True
    max_bitten = 0
    worst_frac = (0.0, None)
    l5_fail = []
    bitten_hist = {}
    for i in range(len(primes) - 1):
        q, q_next = primes[i], primes[i + 1]
        if q_next > qmax:
            break
        ks, lo, hi, spot, twin, bitten, bites = section_spots(q, q_next, spf)
        # L1: spots = twins + (death rung q) disjoint; death rung q means spf of both >= q and not twin - by construction; check via spf
        dr = np.minimum(np.where(lo == spf[lo], np.iinfo(np.int64).max, spf[lo]), np.where(hi == spf[hi], np.iinfo(np.int64).max, spf[hi]))
        l1 &= bool(np.array_equal(spot, twin | (dr == q))) and not bool((twin & (dr == q)).any())
        # L2: bites = q x primes in (q, q_next^2/q), spacing >= q/3 slots
        ms = sorted(int(n // q) for n in np.concatenate([lo[lo % q == 0], hi[hi % q == 0]]))
        # pre-registered as "m prime" - wrong: q divides q*m for every m = +-1 mod 6 in the band;
        # m composite means a smaller gear masks the kill on the same side (47*49, 997*1001)
        # (numbers of the section are 6k_lo-1 .. 6k_hi+1; q_next^2 - 2 belongs to the next section's first slot)
        n_lo, n_hi = int(lo[0]), int(hi[-1])
        exp = [m for m in range(q + 1, n_hi // q + 1) if m % 6 in (1, 5) and n_lo <= m * q <= n_hi]
        l2 &= ms == exp
        bk = ks[bites]
        if len(bk) > 1:
            l2 &= bool((np.diff(bk) >= (q - 1) // 3).all())  # m, m+2 give numbers 2q apart = (q-1)/3 slots at least
        nb = int(bitten.sum())
        bitten_hist[nb] = bitten_hist.get(nb, 0) + 1
        max_bitten = max(max_bitten, nb)
        if q >= 300:
            frac = nb / int(spot.sum())
            if frac > worst_frac[0]:
                worst_frac = (frac, q)
            # L5: a new twin between consecutive bites
            tk = ks[twin]
            for x, y in zip(bk, bk[1:]):
                if not ((tk > x) & (tk < y)).any():
                    l5_fail.append((q, int(x), int(y)))
    gate(l1, "L1 lower-sieve twin spots in the section = new twins + death-rung-q slots (disjoint), every rung")
    gate(l2, "L2 (corrected) gear q's number-kills in its section = q x {m = +-1 mod 6 in (q, q_next^2/q)}, consecutive kills >= (q-1)/3 slots apart")
    gate(max_bitten <= 3 and worst_frac[0] < 0.25,
         f"L4 bitten spots <= 3 per section (max {max_bitten}; histogram {dict(sorted(bitten_hist.items()))}) and bitten fraction < 0.25 at q >= 300 (worst {worst_frac[0]:.3f} at q = {worst_frac[1]})")
    gate(not l5_fail, f"L5 a new twin between consecutive number-kills of gear q at q >= 300 (failures: {l5_fail[:10]}{' ...' if len(l5_fail) > 10 else ''})")
    print(f"\n{NGATE - NFAIL}/{NGATE} gates passed")


if __name__ == "__main__":
    main()
