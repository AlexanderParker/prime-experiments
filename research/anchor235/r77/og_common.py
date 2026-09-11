"""og_common.py -- shared constructions for the origin-mechanic lane (research/proof/origin_mechanic.md).

Objects, by construction:
  the fold {2, 3}; its survivors S = {6j - 1, 6j + 1 : j >= 1} (and 1);
  column j = (6j - 1, 6j + 1); left member 6j - 1 (side -1), right member 6j + 1 (side +1);
  gear g (a prime >= 5) strikes the survivor n iff g | n; the survivors it strikes are the dilate g.S;
  engine {5..q}; least striker of a struck n = least gear dividing n = lpf(n) when lpf(n) <= q;
  quotient = n / least striker; depth = Omega(n).
  section at the cut p: columns a+1 .. b-1 with 6a + 1 = p^2, 6b + 1 = p'^2 (the finer statement);
  construction sections: [p_k^2, p_{k+1}^2), p_{k+1} = nextprime(p_k^2), engine {5..prevprime(p_{k+1})}.
"""
import numpy as np

PRIMES_SMALL = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s)


def nextprime(n):
    from sympy import nextprime as np_
    return int(np_(n))


def prevprime(n):
    from sympy import prevprime as pp_
    return int(pp_(n))


def is_prime(n):
    from sympy import isprime
    return bool(isprime(int(n)))


def gears_of(q):
    """the engine {5..q}: the primes from 5 to q"""
    return [int(g) for g in primes_upto(q) if g >= 5]


def k_of(g):
    """the column holding g: g = 6 k +- 1"""
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def side_of(n):
    """-1 if n = 6j - 1, +1 if n = 6j + 1"""
    r = n % 6
    if r == 5:
        return -1
    if r == 1:
        return 1
    raise ValueError(n)


def column_of(n):
    return (n + 1) // 6 if n % 6 == 5 else (n - 1) // 6


def spf_table(N):
    """smallest prime factor for 0..N (int32); spf[1] = 1, spf[0] = 0"""
    spf = np.zeros(N + 1, dtype=np.int32)
    spf[1] = 1
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == 0:
            spf[i] = i
            blk = spf[i * i::i]
            mask = blk == 0
            blk[mask] = i
            spf[i * i::i] = blk
    rest = np.flatnonzero(spf == 0)
    spf[rest] = rest
    return spf


def omega_from_spf(n, spf):
    """Omega(n) by the spf table (n <= len(spf) - 1)"""
    c = 0
    while n > 1:
        n //= int(spf[n])
        c += 1
    return c


def finer_section(p):
    """the section at the cut p: (a, b, columns a+1..b-1, engine gears)"""
    pn = nextprime(p)
    a = (p * p - 1) // 6
    b = (pn * pn - 1) // 6
    return a, b, list(range(a + 1, b)), gears_of(p)


def construction_sections(base, links):
    """the chain from base: (k, p_k, p_{k+1}, a, b, engine top q) for the first `links` links"""
    out = []
    c = base
    for k in range(links):
        pk = nextprime(c - 1)
        cn = pk * pk
        pk1 = nextprime(cn - 1)
        q = prevprime(pk1)
        a = (cn - 1) // 6
        b = (pk1 * pk1 - 1) // 6
        out.append(dict(k=k + 1, p_k=pk, p_k1=pk1, lo=cn, hi=pk1 * pk1, a=a, b=b, q=q))
        c = cn
    return out


def strikers_of_column(j, gears):
    """list of (g, side) for the gears of `gears` dividing a member of column j"""
    out = []
    lo, hi = 6 * j - 1, 6 * j + 1
    for g in gears:
        if lo % g == 0:
            out.append((g, -1))
        elif hi % g == 0:
            out.append((g, 1))
    return out


def family_strikes_column(j, gears, teeth):
    """the tooth family: gear g with teeth +-v strikes column j iff j = +-v (mod g); returns list of g"""
    return [g for g, v in zip(gears, teeth) if (j % g) in (v % g, (-v) % g)]


def real_teeth(gears):
    return [k_of(g) for g in gears]


def struck_segment(q, start, n):
    """bool array over columns start .. start+n-1: struck by some gear of {5..q} (real teeth)"""
    arr = np.zeros(n, dtype=bool)
    for g in gears_of(q):
        k = k_of(g)
        for t in (k % g, (-k) % g):
            i0 = (t - start) % g
            arr[i0::g] = True
    return arr


def record_F(q, limit=None):
    """F(q) = the largest distance between consecutive open columns over a full period (small q)"""
    P = 1
    for g in gears_of(q):
        P *= g
    n = P if limit is None else min(P, limit)
    s = struck_segment(q, 0, n + 1)
    op = np.flatnonzero(~s)
    return int((op[1:] - op[:-1]).max())
