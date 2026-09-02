"""Section reruns A and B - manager, round 29 (pre-registration: data/r29/section_ab_prereg.md).

A. Odometer order of the new twins of one section: the human's sort step on the new part only.
   Each new twin k has digit vector (k mod 5, k mod 7, ..., k mod p); list the section's twins
   in lex order (mod 5 most significant) and in reverse lex (mod p most significant) with the
   carry position (first digit at which consecutive sorted twins differ) and the k-difference.

B. Kill lists as words with cross-section provenance: gear s kills in the section exactly
   s * m for m in (p^2/s, q^2/s) with spf(m) >= s. List per gear the m's it consumes, each
   tagged with the section that produced m, and the reverse table (which sections feed this
   one, by gear).

Usage: python section_ab_r29.py [--q 31 --q 53 ...] [--qmax 5000]
"""
import argparse
import bisect
from collections import Counter
from itertools import product
from math import isqrt

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


def section_of(m, primes):
    """rung index i with primes[i-1]^2 < m < primes[i]^2, or 'base' for m < 25."""
    if m < 25:
        return "base"
    r = isqrt(m)
    i = bisect.bisect_right(primes, r)  # primes[i] > sqrt(m) >= primes[i-1]
    return f"{primes[i - 1]}->{primes[i]}"


def new_twins(p, q, spf):
    k_lo = p * p // 6 + 1
    k_hi = (q * q - 2) // 6
    ks = np.arange(k_lo, k_hi + 1)
    lo, hi = 6 * ks - 1, 6 * ks + 1
    tw = ks[(spf[lo] == lo) & (spf[hi] == hi)]
    return k_lo, k_hi, [int(k) for k in tw]


def carry_pos(v, w):
    for i, (a, b) in enumerate(zip(v, w)):
        if a != b:
            return i
    return len(v)


def part_a(p, q, gears, twins, full):
    print(f"  --- A. odometer order of the {len(twins)} new twins, digits mod {gears} ---")
    vecs = {k: tuple(k % g for g in gears) for k in twins}
    for name, key in (("lex (mod 5 first)", lambda k: vecs[k]),
                      ("reverse lex (mod p first)", lambda k: vecs[k][::-1])):
        order = sorted(twins, key=key)
        carries = []
        if full:
            print(f"  {name}:")
        prev = None
        for k in order:
            v = vecs[k] if name.startswith("lex") else vecs[k][::-1]
            c = carry_pos(prev[1], v) if prev else None
            if prev:
                carries.append(c)
            if full:
                digits = " ".join(f"{d:>3}" for d in v)
                tail = "" if prev is None else f"  carry at digit {c} (gear {(gears if name.startswith('lex') else gears[::-1])[c]}), dk = {k - prev[0]:+d}"
                print(f"    k={k:>7}  [{digits}]{tail}")
            prev = (k, v)
        print(f"  {name}: carry positions in order: {carries}")
    return vecs


def part_b(p, q, gears, spf, k_lo, k_hi, primes, full):
    print(f"  --- B. kill lists K_s = s x {{m in (p^2/s, q^2/s): spf(m) >= s}} ---")
    by_gear = {}
    for k in range(k_lo, k_hi + 1):
        for n in (6 * k - 1, 6 * k + 1):
            f = int(spf[n])
            if f != n:
                by_gear.setdefault(f, []).append((k, n // f))
    feeders = {}
    for s in sorted(by_gear):
        ms = by_gear[s]
        lo_b, hi_b = p * p / s, q * q / s
        # forced identity: the m's are exactly the s-rough numbers in the band
        # the section is a slot range, so the numbers are 6k_lo-1 .. 6k_hi+1 (the slot whose
        # 6k+1 = q^2 belongs to the next section although its 6k-1 < q^2)
        n_lo, n_hi = 6 * k_lo - 1, 6 * k_hi + 1
        band = [m for m in range(int(lo_b) + 1, int(np.ceil(hi_b))) if n_lo <= m * s <= n_hi and spf[m] >= s]
        gate_ok = [m for _, m in ms] == band
        if not gate_ok:
            print(f"  IDENTITY FAIL gear {s}: {[m for _, m in ms]} vs {band}")
        srcs = [(m, section_of(m, primes)) for _, m in ms]
        for m, src in srcs:
            feeders.setdefault(src, Counter())[s] += 1
        allprime = all(spf[m] == m for _, m in ms)
        if full or s >= p // 4 or len(ms) <= 12:
            body = ", ".join(f"{m}{'' if spf[m] == m else '*'}[{src}]" for m, src in srcs)
        else:
            body = f"{ms[0][1]}[{srcs[0][1]}] ... {ms[-1][1]}[{srcs[-1][1]}]  ({len(ms)} m's, {sum(spf[m] == m for _, m in ms)} prime)"
        print(f"  gear {s:>4}  m in ({lo_b:8.1f}, {hi_b:8.1f})  {'all prime' if allprime else '* = rough  '}: {body}")
    print(f"  fed by section (m's origin) -> gears consuming it:")
    def src_key(x):
        return -1 if x == "base" else int(x.split("->")[0])
    for src in sorted(feeders, key=src_key):
        c = feeders[src]
        print(f"    {src:>10}: " + ", ".join(f"{s}x{n}" if n > 1 else f"{s}" for s, n in sorted(c.items())))
    return by_gear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, action="append", default=None)
    ap.add_argument("--qmax", type=int, default=5000)
    a = ap.parse_args()
    show = a.q or [31, 53, 199, 997]
    qmax = max(a.qmax, max(show))
    primes = [int(x) for x in np.flatnonzero(spf_sieve(qmax + 100) == np.arange(qmax + 101)) if x >= 5]
    spf = spf_sieve(qmax * qmax + 10)

    for q in show:
        i = primes.index(q)
        p = primes[i - 1]
        gears = primes[:i]  # 5..p
        k_lo, k_hi, twins = new_twins(p, q, spf)
        full = q <= 200
        print(f"\n=== section {p} -> {q}: numbers ({p * p}, {q * q}), slots {k_lo}..{k_hi}, {len(twins)} new twins ===")
        part_a(p, q, gears, twins, full)
        part_b(p, q, gears, spf, k_lo, k_hi, primes, full)

    # ---- gates over all sections q <= qmax ----
    print(f"\n=== gates over all sections with q <= {qmax} ===")
    carry_obs = Counter()
    carry_model = Counter()
    top4 = [5, 7, 11, 13]
    open_vecs = [v for v in product(*[range(g) for g in top4]) if all(v[j] not in teeth(g) for j, g in enumerate(top4))]
    rng = np.random.default_rng(29)
    b1_ok = True
    b2_ok = True
    b2_counts = Counter()
    n_sections = 0
    prev_hi_band = {}
    for i in range(1, len(primes)):
        p, q = primes[i - 1], primes[i]
        if q > qmax:
            break
        n_sections += 1
        k_lo, k_hi, twins = new_twins(p, q, spf)
        if q >= 1000 and len(twins) >= 2:
            vecs = sorted(tuple(k % g for g in top4) for k in twins)
            for v, w in zip(vecs, vecs[1:]):
                carry_obs[carry_pos(v, w)] += 1
            samp = sorted(open_vecs[j] for j in rng.integers(0, len(open_vecs), len(twins)))
            for v, w in zip(samp, samp[1:]):
                carry_model[carry_pos(v, w)] += 1
        # B1: for s > q^(2/3) the m's are exactly the primes in the band; bands contiguous
        ns = np.arange(p * p + 1, q * q)
        ns = ns[(ns % 6 == 1) | (ns % 6 == 5)]
        fs = spf[ns]
        for s in [g for g in primes[:i] if g ** 3 > q * q]:
            lo_b, hi_b = p * p // s, -(-q * q // s)
            band = np.arange(lo_b + 1, hi_b)
            band = band[(band * s > p * p) & (band * s < q * q)]
            band_primes = band[spf[band] == band].tolist()
            kills = (ns[fs == s] // s).tolist()
            if kills != band_primes:
                b1_ok = False
            if s in prev_hi_band and prev_hi_band[s] != p * p:
                b1_ok = False
            prev_hi_band[s] = q * q
        # B2: gear p's own kills
        own = (ns[fs == p] // p).tolist()
        exp = [m for m in primes[i:] if m * p < q * q]
        if own != exp:
            b2_ok = False
        b2_counts[len(own)] += 1
    tot_o, tot_m = sum(carry_obs.values()), sum(carry_model.values())
    tv = 0.5 * sum(abs(carry_obs[c] / tot_o - carry_model[c] / tot_m) for c in range(5))
    print(f"  carry position (lex, digits mod 5,7,11,13) observed {dict(sorted(carry_obs.items()))}")
    print(f"  carry position iid-uniform model            {dict(sorted(carry_model.items()))}   TV = {tv:.4f}")
    gate(tv < 0.05, f"A1 carry positions within TV 0.05 of iid-uniform open vectors (TV {tv:.4f})")
    gate(b1_ok, f"B1 for s > q^(2/3) kills = s x all primes in the band, bands contiguous across sections ({n_sections} sections)")
    gate(b2_ok, f"B2 gear p's own kills = p x primes q, q_2, ... below q^2/p (as numbers)")
    gate(max(b2_counts) <= 3, f"B2' gear p's own kills number 1 to 3 (pre-registered); distribution {dict(sorted(b2_counts.items()))}")
    print(f"\n{NGATE - NFAIL}/{NGATE} gates passed")


if __name__ == "__main__":
    main()
