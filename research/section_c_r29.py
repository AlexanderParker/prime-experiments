"""Section rerun C - the tooth counterfactual on the new section (pre-registration:
data/r29/section_c_prereg.md).

Move one gear's teeth from {+-u} (u = 6^{-1} mod s*) to {+-v} and read the new section
(p^2, q^2) of the rung p -> q with every other gear real: which new twins the counterfactual
machine loses (real twin primes in the class +-v), which slots it gains (slots blocked only by
gear s*, outside +-v), and what the gained slots are as numbers.

Usage: python section_c_r29.py [--qmax 5000]
"""
import argparse

import numpy as np

from word_tree_r29 import spf_sieve

NGATE = 0
NFAIL = 0


def gate(cond, msg):
    global NGATE, NFAIL
    NGATE += 1
    NFAIL += (not cond)
    print(("  GATE ok:   " if cond else "  GATE FAIL: ") + msg)


def section(p, q):
    return p * p // 6 + 1, (q * q - 2) // 6


def strip(n, s):
    """n with every factor s removed (vectorised)."""
    n = n.copy()
    for _ in range(40):
        d = n % s == 0
        if not d.any():
            break
        n[d] //= s
    return n


def real_word(k_lo, k_hi, spf, p):
    """per slot: (twin?, sole killer or 0). Sole killer s: s is the smallest gear dividing a side
    and, after removing s from both numbers, neither has a prime factor <= p (so no other gear
    touches the slot)."""
    ks = np.arange(k_lo, k_hi + 1)
    lo, hi = 6 * ks - 1, 6 * ks + 1
    f_lo, f_hi = spf[lo], spf[hi]
    twin = (f_lo == lo) & (f_hi == hi)
    big = np.iinfo(np.int64).max
    s = np.minimum(np.where(f_lo == lo, big, f_lo), np.where(f_hi == hi, big, f_hi))
    s = np.where(twin, 1, s)
    r_lo, r_hi = np.ones_like(lo), np.ones_like(hi)
    for g in np.unique(s[~twin]):
        m = s == g
        r_lo[m] = strip(lo[m], int(g))
        r_hi[m] = strip(hi[m], int(g))
    clean = lambda r: (r == 1) | (spf[r] > p)
    sole = (~twin) & clean(r_lo) & clean(r_hi)
    return ks, twin, np.where(sole, s, 0), lo, hi


def counterfactual(ks, twin, sole, s_star, v):
    r = ks % s_star
    in_v = (r == v % s_star) | (r == (-v) % s_star)
    lost = twin & in_v
    gained = (sole == s_star) & ~in_v
    return lost, gained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmax", type=int, default=5000)
    a = ap.parse_args()
    qmax = a.qmax
    primes = [int(x) for x in np.flatnonzero(spf_sieve(qmax + 100) == np.arange(qmax + 101)) if x >= 5]
    spf = spf_sieve(qmax * qmax + 10).astype(np.int64)

    # ---- listings ----
    for q, stars in ((53, (13, 47)), (997, (991,))):
        i = primes.index(q)
        p = primes[i - 1]
        k_lo, k_hi = section(p, q)
        ks, twin, sole, lo, hi = real_word(k_lo, k_hi, spf, p)
        print(f"\n=== section {p} -> {q}: slots {k_lo}..{k_hi}, {int(twin.sum())} new twins ===")
        for s_star in stars:
            u = pow(6, -1, s_star)
            teeth = sorted({u % s_star, (-u) % s_star})
            print(f"  gear {s_star}: real teeth {teeth}; slots blocked only by gear {s_star}: "
                  + ", ".join(f"k={int(k)} ({int(l)}={s_star}*{int(l) // s_star}|{int(h)})" if l % s_star == 0
                              else f"k={int(k)} ({int(l)}|{int(h)}={s_star}*{int(h) // s_star})"
                              for k, l, h in zip(ks[sole == s_star], lo[sole == s_star], hi[sole == s_star])))
            vs = [v for v in range(1, (s_star + 1) // 2) if v not in teeth and (s_star - v) not in teeth]
            if q > 100:
                vs = vs[:1]
            for v in vs:
                lost, gained = counterfactual(ks, twin, sole, s_star, v)
                print(f"    teeth -> {{+-{v}}}: lost twins k = {[int(k) for k in ks[lost]]}; "
                      f"gained slots k = {[int(k) for k in ks[gained]]}; survivors {int(twin.sum() - lost.sum() + gained.sum())} vs real {int(twin.sum())}")

    # ---- gate C1 over sections q >= 1000 ----
    print(f"\n=== C1 over sections 1000 <= q <= {qmax}: counterfactual survivors / real new twins ===")
    for s_star, vs in ((13, (1, 3, 4, 5, 6)), (7, (2, 3))):
        tot_real = 0
        tot_cf = {v: 0 for v in vs}
        for i in range(1, len(primes)):
            p, q = primes[i - 1], primes[i]
            if q > qmax:
                break
            if q < 1000:
                continue
            k_lo, k_hi = section(p, q)
            ks, twin, sole, lo, hi = real_word(k_lo, k_hi, spf, p)
            tot_real += int(twin.sum())
            for v in vs:
                lost, gained = counterfactual(ks, twin, sole, s_star, v)
                tot_cf[v] += int(twin.sum() - lost.sum() + gained.sum())
        ratios = {v: tot_cf[v] / tot_real for v in vs}
        print(f"  s* = {s_star}: real {tot_real}; ratios " + ", ".join(f"v={v}: {r:.4f}" for v, r in ratios.items()))
        gate(all(abs(r - 1) < 0.10 for r in ratios.values()),
             f"C1 s* = {s_star}: every moved-teeth survivor count within 10% of the real count")

    # ---- post hoc (not pre-registered): the relaxed survivors R (no gear but s* touches the
    # slot) by residue class mod s*; the tooth class is where s* divides a side ----
    print(f"\n=== post hoc: relaxed survivors by class mod s*, sections 1000 <= q <= {qmax} ===")
    for s_star in (7, 13, 31):
        u = pow(6, -1, s_star)
        cls = np.zeros(s_star, dtype=np.int64)
        logsum, nn = 0.0, 0
        for i in range(1, len(primes)):
            p, q = primes[i - 1], primes[i]
            if q > qmax:
                break
            if q < 1000:
                continue
            k_lo, k_hi = section(p, q)
            ks, twin, sole, lo, hi = real_word(k_lo, k_hi, spf, p)
            R = twin | (sole == s_star)
            cls += np.bincount(ks[R] % s_star, minlength=s_star)
            logsum += np.log(6.0 * ks[R]).sum()
            nn += int(R.sum())
        tooth = (cls[u % s_star] + cls[(-u) % s_star]) / 2
        others = np.delete(cls, [u % s_star, (-u) % s_star])
        L = logsum / nn
        pred = L / (L - np.log(s_star))
        print(f"  s* = {s_star}: tooth classes {cls[u % s_star]}, {cls[(-u) % s_star]}; other classes "
              f"min {others.min()} max {others.max()} mean {others.mean():.0f}; tooth/other = {tooth / others.mean():.4f}; "
              f"cofactor model ln n / ln(n/s*) = {pred:.4f}")
    print(f"\n{NGATE - NFAIL}/{NGATE} gates passed")


if __name__ == "__main__":
    main()
