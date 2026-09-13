"""The sub-machine's wheel at phase -q (owner: go; 2026-09-13).

Below 2q every painted member of the landing family (12k - 1, 12k + 1) is a composite below 2q,
hence has a prime factor at most sqrt(2q); so below 2q "painted by the machine" = "painted by
the sub-machine B(q) = the gears up to sqrt(2q)", exactly. The run at the zone start, L(q), is
therefore the painted run of the B(q)-wheel on the m-line at the phase k_0 (the first k with
12k - 1 > q), as long as it stays below 2q. Consequence, exact: L(q) <= R_B, the m-line record
of the sub-machine (its longest painted run over a full period), whenever R_B < q/12.
This script: (1) the m-line record R_y of the wheel of the gears 5..y for y up to 29, computed
exactly over a full period (the opens are the CRT survivors, prod (h - 2) per period; the record
is the largest gap between consecutive opens, cyclically); (2) for every machine q with
sqrt(2q) <= 29, L(q) against R_{B(q)}; (3) the sufficient condition R_{B(q)} < q/12 checked.
Usage: uv run python submachine_phase.py
"""
import numpy as np
from sympy import primerange, isprime


def opens_mline(gears):
    """all k in [0, P) with 12k -+ 1 not divisible by any gear, by CRT lifting; P = prod gears"""
    ks = np.array([0], dtype=np.int64); P = 1
    for h in gears:
        inv = pow(12, -1, h); teeth = {inv % h, (-inv) % h}
        good = np.array([r for r in range(h) if r not in teeth], dtype=np.int64)
        # combine: k = ks + P * t, need k mod h in good: for each ks, the t in [0, h) with (ks + P t) % h in good
        new = []
        Pinv = pow(P % h, -1, h)
        for r in good:
            t = ((r - ks) % h) * Pinv % h
            new.append(ks + P * t)
        ks = np.concatenate(new); P *= h
    return np.sort(ks), P


def record(gears):
    ks, P = opens_mline(gears)
    gaps = np.diff(np.concatenate([ks, [ks[0] + P]])) - 1
    return int(gaps.max()), P, len(ks)


def main():
    primes = list(primerange(5, 30))
    R = {}
    print("y | gears 5..y | period P on the m-line | opens per period | record R_y (longest painted run) | R_y against y^2/24")
    for y in primes:
        gears = [h for h in primes if h <= y]
        r, P, n = record(gears); R[y] = r
        print(f"{y} | {gears} | {P} | {n} | {r} | {r / (y * y / 24):.2f}")
    print("\nmachines q with sqrt(2q) <= 29: q | sub-machine top y | L(q) | R_y | L <= R | R_y < q/12")
    qs = [q for q in primerange(11, 421)]
    ok = 0; suff = 0
    for q in qs:
        y = max(h for h in primes if h * h <= 2 * q) if 2 * q >= 25 else 5
        k0 = (q + 1) // 12 + 1; k = k0
        while not (isprime(12 * k - 1) and isprime(12 * k + 1)): k += 1
        L = k - k0
        ok += L <= R[y]; suff += R[y] < q / 12
        if q in (11, 31, 101, 199, 307, 401, 419): print(f"{q} | {y} | {L} | {R[y]} | {L <= R[y]} | {R[y] < q / 12}")
    print(f"L(q) <= R_(B(q)) at {ok} of {len(qs)} machines; the sufficient condition R_B < q/12 holds at {suff} of {len(qs)}")


if __name__ == "__main__":
    main()
