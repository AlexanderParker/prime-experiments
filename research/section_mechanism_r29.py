"""Exact mechanism display of one section - manager, round 29.

For the section p -> q (numbers in (p^2, q^2), slots p^2 < 6k+1 < q^2) every blocked slot is
a composite 6k+-1 = s * m with s its smallest prime factor (a gear <= p) and m in (p^2/s, q^2/s)
with no factor below s. So the section's word is decided by the lattice of prime pairs on the
hyperbola band s * m in (p^2, q^2), and the new twins are the slots that no pair reaches.

The display lists, for one section, every slot as: twin, or the factorisation s * m that kills
it, with m marked prime or rough; then the slots grouped by killing gear s with their m's, so
the pattern of which products fall in the band is visible without any count.

Usage: python section_mechanism_r29.py --q 53 [--q 199 ...]
"""
import argparse

import numpy as np

from word_tree_r29 import spf_sieve


def is_prime(n, spf):
    return n >= 2 and spf[n] == n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, action="append", default=None)
    a = ap.parse_args()
    qs = a.q or [31, 53]
    qmax = max(qs)
    spf = spf_sieve(qmax * qmax + 10)
    primes = [int(x) for x in np.flatnonzero(spf[: qmax + 2] == np.arange(qmax + 2)) if x >= 5]
    for q in qs:
        i = primes.index(q)
        p = primes[i - 1]
        k_lo = p * p // 6 + 1
        k_hi = (q * q - 2) // 6
        print(f"=== section {p} -> {q}: numbers ({p * p}, {q * q}), slots {k_lo}..{k_hi} ===")
        by_gear = {}
        twins = []
        for k in range(k_lo, k_hi + 1):
            lo, hi = 6 * k - 1, 6 * k + 1
            f_lo, f_hi = int(spf[lo]), int(spf[hi])
            if f_lo == lo and f_hi == hi:
                twins.append(k)
                print(f"  k={k:>6}  ({lo}, {hi})  TWIN  residues " +
                      " ".join(f"{g}:{k % g}" for g in primes[: i] if g <= 23))
                continue
            parts = []
            for n, f in ((lo, f_lo), (hi, f_hi)):
                if f != n:
                    m = n // f
                    tag = "prime" if is_prime(m, spf) else f"rough(spf {int(spf[m])})"
                    parts.append(f"{n} = {f} * {m} ({tag})")
                    by_gear.setdefault(f, []).append((k, n, m))
            s = min([f_lo] * (f_lo != lo) + [f_hi] * (f_hi != hi))
            print(f"  k={k:>6}  ({lo}, {hi})  killed by {s:>3}: " + "; ".join(parts))
        print(f"  --- by gear: the products s * m landing in ({p * p}, {q * q}) ---")
        for s in sorted(by_gear):
            ms = by_gear[s]
            band = f"m in ({p * p / s:.1f}, {q * q / s:.1f})"
            print(f"  gear {s:>4} {band:>26}: " +
                  ", ".join(f"{m}{'' if is_prime(m, spf) else '*'}" for _, _, m in ms) +
                  ("   (* = rough, not prime)" if any(not is_prime(m, spf) for _, _, m in ms) else ""))
        print(f"  new twins: {len(twins)} at k = {twins}\n")


if __name__ == "__main__":
    main()
