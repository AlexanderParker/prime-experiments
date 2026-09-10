"""The nth prime from the machine: the construction evaluated and checked.

Two checks.

(A) The count on a stretch by the core / tail split (the leftover identity read for single
    primes).  For a stretch (x, x + 6L] with core t = 6L + 1 and x + 6L < t^3:
        pi(x + 6L) - pi(x)  =  #{m in the stretch : no prime <= t divides m}
                              - #{m in the stretch : m = P1 P2, P1, P2 primes > t}.
    The first term is the core's open count (inclusion-exclusion over the core, or a sieve by
    the core alone); the second is the tail's strikes on core-open numbers, each a product of
    exactly two tail primes (kernel primeOrSemiprime_of_rough_lt_cube).

(B) p_n by sections of the stack by squares from base 2: cuts c_1 = 2, c_{k+1} = p_k^2 with p_k
    the first prime >= c_k; on section k the primes are exactly the members with no prime factor
    <= p_k that are > 1 (square-root rule); so p_n = the (n - N_k)-th such member of the section
    that holds it, N_k = pi(c_k) accumulated from the sections below.  Checked against sympy.

Usage: uv run python nth_prime.py
"""
import math, random, time
import numpy as np
from sympy import prime, primepi, isprime, primerange


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def core_open_count(x, L, core):
    """#{m in (x, x+6L] : no prime in core divides m}, by a sieve of the stretch by the core."""
    a, b = x + 1, x + 6 * L
    s = np.ones(b - a + 1, dtype=bool)
    for p in core:
        start = ((a + p - 1) // p) * p
        s[start - a::p] = False
    return int(s.sum())


def tail_semiprimes(x, L, t):
    """#{m in (x, x+6L] : m = P1*P2, primes P1 <= P2 both > t}."""
    a, b = x + 1, x + 6 * L
    cnt = 0
    for P1 in primerange(t + 1, math.isqrt(b) + 1):
        m0 = ((a + P1 - 1) // P1) * P1
        for m in range(m0, b + 1, P1):
            P2 = m // P1
            if P2 >= P1 and P2 > t and isprime(P2):
                cnt += 1
    return cnt


def check_A(trials=12, seed=1):
    rng = random.Random(seed)
    print("(A) the stretch count: x, L, t, pi-difference, core-open, tail semiprimes, core-open - semiprimes, agree")
    ok = 0
    for _ in range(trials):
        L = rng.choice([50, 100, 200, 400])
        t = 6 * L + 1
        x = rng.randrange(10 ** 6, min(10 ** 9, t ** 3 - 6 * L - 1))
        core = [int(p) for p in primes_upto(t)]
        pd = int(primepi(x + 6 * L) - primepi(x))
        co = core_open_count(x, L, core)
        sp = tail_semiprimes(x, L, t)
        agree = (co - sp == pd); ok += agree
        print(f"  {x} {L} {t} {pd} {co} {sp} {co - sp} {agree}")
    print(f"  agree at {ok} of {trials}")


def nth_prime_by_sections(ns):
    """Return {n: p_n} for the requested n, by walking the sections of the base-2 stack."""
    ns = sorted(ns); want = set(ns); out = {}
    c = 2; N = 0  # N = pi(c) so far (primes below c)
    while ns and len(out) < len(want):
        p = int(c) if isprime(c) else int(prime(primepi(c) + 1))  # first prime >= c
        c_next = p * p
        machine = primes_upto(p)  # machine k: the primes <= p_k
        # members of [c, c_next) with no prime factor <= p, and > 1: exactly the primes there
        a, b = c, c_next - 1
        if b - a + 1 > 3 * 10 ** 8:
            break
        s = np.ones(b - a + 1, dtype=bool)
        for q in machine:
            start = max(q * q, ((a + q - 1) // q) * q)
            s[start - a::q] = False
        # primes <= p inside the section are the machine's own gears sitting in the section
        for q in machine:
            if a <= q <= b:
                s[q - a] = True
        if a <= 1:
            s[:2 - a] = False
        idx = np.nonzero(s)[0]
        cnt = len(idx)
        for n in ns:
            if N < n <= N + cnt:
                out[n] = (int(a + idx[n - N - 1]), (c, c_next, p, N, cnt))
        N += cnt; c = c_next
        ns = [n for n in ns if n not in out]
    return out


def check_B():
    ns = [1, 2, 3, 4, 5, 9, 10, 25, 100, 146, 147, 1000, 10000, 58174, 58175, 100000, 1000000, 5000000, 10000000]
    t0 = time.time()
    got = nth_prime_by_sections(ns)
    print(f"(B) p_n by sections ({time.time() - t0:.1f}s): n, section [c_k, c_k+1), machine's top gear p_k, N_k, primes in section, p_n, sympy, agree")
    ok = 0
    for n in ns:
        if n not in got:
            print(f"  {n}: beyond the last section computed"); continue
        pn, (c, cn, p, N, cnt) = got[n]
        ref = int(prime(n)); ok += (pn == ref)
        print(f"  {n} [{c}, {cn}) {p} {N} {cnt} {pn} {ref} {pn == ref}")
    print(f"  agree at {ok} of {len([n for n in ns if n in got])}")


if __name__ == "__main__":
    check_A()
    check_B()
