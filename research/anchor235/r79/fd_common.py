"""fd_common.py -- shared constructions for the fields lane (research/proof/fields.md).

Objects, by construction:
  the fold {2, 3}; survivors S = {6k - 1, 6k + 1 : k >= 1} (and 1);
  column k = (6k - 1, 6k + 1): left member 6k - 1 (class -1), right member 6k + 1 (class +1);
  Omega(n) = number of prime factors with multiplicity; Omega_-(n) = those = 5 (mod 6);
  field j = F_j = {n in S : Omega(n) = j}; F_1 = the primes >= 5; the square field Q = {p^2};
  the engine {5..q} with sight [1, q'^2), q' = nextprime(q); on the sight struck = composite = overlay of F_j, j >= 2;
  the finer section at p: columns a+1 .. b-1 with 6a + 1 = p^2, 6b + 1 = p'^2;
  the construction sections [p_k^2, p_{k+1}^2), p_{k+1} = nextprime(p_k^2), engine {5..prevprime(p_{k+1})}.
"""
import numpy as np
from sympy import nextprime as _np, prevprime as _pp, isprime as _ip


def nextprime(n):
    return int(_np(n))


def prevprime(n):
    return int(_pp(n))


def is_prime(n):
    return bool(_ip(int(n)))


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s)


def gears_of(q):
    """the engine {5..q}"""
    return [int(g) for g in primes_upto(q) if g >= 5]


def k_of(g):
    """the column holding the survivor g"""
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def eps_of(g):
    return -1 if g % 6 == 5 else 1


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


def omega_arrays(n, spf):
    """vectorised: for an int64 array n of survivors (each <= len(spf) - 1) return
    (Omega, Omega_-, lpf) as int8, int8, int64.  Omega(1) = 0, lpf(1) = 1."""
    n = n.astype(np.int64).copy()
    om = np.zeros(len(n), dtype=np.int8)
    omm = np.zeros(len(n), dtype=np.int8)
    lpf = np.ones(len(n), dtype=np.int64)
    first = True
    while True:
        act = np.flatnonzero(n > 1)
        if len(act) == 0:
            break
        p = spf[n[act]].astype(np.int64)
        if first:
            lpf[act] = p
            first = False
        om[act] += 1
        omm[act] += (p % 6 == 5).astype(np.int8)
        n[act] //= p
    return om, omm, lpf


def column_members(k):
    """k: int64 array of columns -> (left, right)"""
    k = np.asarray(k, dtype=np.int64)
    return 6 * k - 1, 6 * k + 1


def finer_section(p):
    """the finer section at p: (a, b, columns a+1..b-1 as int64 array, engine, p')"""
    pn = nextprime(p)
    a = (p * p - 1) // 6
    b = (pn * pn - 1) // 6
    return dict(p=p, pn=pn, a=a, b=b, cols=np.arange(a + 1, b, dtype=np.int64), gears=gears_of(p), q=p,
                name=f"finer p={p}")


def construction_sections(base, links):
    """the chain from `base`: the first `links` sections as dicts (like finer_section)"""
    out = []
    c = base
    for k in range(links):
        pk = nextprime(c - 1)
        cn = pk * pk
        pk1 = nextprime(cn - 1)
        q = prevprime(pk1)
        a = (cn - 1) // 6
        b = (pk1 * pk1 - 1) // 6
        out.append(dict(p=pk, pn=pk1, a=a, b=b, cols=np.arange(a + 1, b, dtype=np.int64), gears=gears_of(q), q=q,
                        name=f"base {base} link {k + 1} [{cn}, {pk1 * pk1})", k=k + 1, lo=cn, hi=pk1 * pk1))
        c = cn
    return out


def section_census(sec, spf):
    """attach to the section dict the Omega data of both members of every column"""
    L, R = column_members(sec["cols"])
    sec["L"], sec["R"] = L, R
    sec["omL"], sec["ommL"], sec["lpfL"] = omega_arrays(L, spf)
    sec["omR"], sec["ommR"], sec["lpfR"] = omega_arrays(R, spf)
    return sec


def struck_columns(cols, gears):
    """bool: column struck by some gear (real teeth +-k_g mod g)"""
    out = np.zeros(len(cols), dtype=bool)
    for g in gears:
        k = k_of(g)
        r = cols % g
        out |= (r == k % g) | (r == (-k) % g)
    return out


def factor_omega(n):
    """Omega, Omega_-, lpf by sympy.factorint (for numbers beyond any spf table)"""
    from sympy import factorint
    f = factorint(int(n))
    om = sum(f.values())
    omm = sum(e for p, e in f.items() if p % 6 == 5)
    lpf = min(f) if f else 1
    return om, omm, lpf
