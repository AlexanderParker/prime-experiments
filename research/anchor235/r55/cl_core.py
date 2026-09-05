"""r55 cl_core - collision laws for gear pairs: the exact machinery.

Frame.  Column k is the pair (6k-1, 6k+1).  A gear is a prime g >= 5 with a separation s_g;
at phase c it strikes the columns c and c + s_g (mod g).  The real machine has
s_g = 3^{-1} (mod g).  Short arc a_g = min(s_g, g - s_g).

max_g(L)          the most columns g can strike on a run of L, over its phases
joint_max(S; L)   the most columns the set S can strike together (union), over all phases
c_S(L)            = sum_g max_g(L) - joint_max(S; L)          the block deficit
c_max(g,h;L)      = min overlap over the phase pairs where BOTH gears are maximal
c_dis(g,h;L)      = min (max_g - n_g) + (max_h - n_h) over the phase pairs with NO overlap

THE ONE-ORBIT REDUCTION (cited: the_wall.md 5a).  For a fixed gear set with pairwise coprime
moduli, CRT makes every phase vector a diagonal translate (c_g, c_h, ...) = (t, t, ...), so
"over all phases" is "slide a window of length L over one period of the fixed pattern".  Every
number here is therefore an exact maximum over a period, not a search.

Standard library plus numpy.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def primes_upto(n):
    b = bytearray([1]) * (n + 1)
    b[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if b[i]:
            b[i * i::i] = bytearray(len(b[i * i::i]))
    return [i for i in range(2, n + 1) if b[i]]


PRIMES = primes_upto(20000)
_PSET = set(PRIMES)


def is_prime(n):
    return n in _PSET


def real_sep(g):
    """s_g = 3^{-1} (mod g) - the machine's own separation."""
    return pow(3, -1, g)


def coherent_sep(g, c, r):
    """s_g = c * r^{-1} (mod g) - the coherent family c/r (separation_drives_K.md N-S2)."""
    return (c * pow(r, -1, g)) % g


def arc(g, s):
    return min(s % g, g - (s % g))


def maxstrike(g, s, L):
    """file 20 Lemma 2, valid for any separation (the proof uses only the two cyclic arcs)."""
    a = arc(g, s)
    q, r = divmod(L, g)
    return 2 * q + (2 if r > a else (1 if r >= 1 else 0))


def gear_indicator(g, s, n):
    """0/1 array of length n: x is struck by gear g at phase 0."""
    v = np.zeros(n, dtype=np.int8)
    v[0::g] = 1
    v[(s % g)::g] = 1
    return v


def pair_profile(g, sg, h, sh, Lmax, want_split=False):
    """Exact c(g,h;L) for L = 1..Lmax (and optionally c_max, c_dis).

    Returns dict with arrays indexed 0..Lmax (index L).  inf is encoded as -1.
    """
    P = g * h
    n = P + Lmax + 2
    Ag = gear_indicator(g, sg, n).astype(np.int32)
    Ah = gear_indicator(h, sh, n).astype(np.int32)
    Ai = Ag * Ah
    # running window counts over t = 0..P-1 for the current L
    ng = np.zeros(P, dtype=np.int32)
    nh = np.zeros(P, dtype=np.int32)
    ni = np.zeros(P, dtype=np.int32)
    t = np.arange(P)
    c = np.zeros(Lmax + 1, dtype=np.int64)
    cmax = np.full(Lmax + 1, -1, dtype=np.int64)
    cdis = np.full(Lmax + 1, -1, dtype=np.int64)
    mg_arr = np.zeros(Lmax + 1, dtype=np.int64)
    mh_arr = np.zeros(Lmax + 1, dtype=np.int64)
    jm = np.zeros(Lmax + 1, dtype=np.int64)
    for L in range(1, Lmax + 1):
        idx = t + (L - 1)
        ng += Ag[idx]
        nh += Ah[idx]
        ni += Ai[idx]
        mg = maxstrike(g, sg, L)
        mh = maxstrike(h, sh, L)
        union = ng + nh - ni
        u = int(union.max())
        mg_arr[L], mh_arr[L], jm[L] = mg, mh, u
        c[L] = mg + mh - u
        if want_split:
            sel = (ng == mg) & (nh == mh)
            cmax[L] = int(ni[sel].min()) if sel.any() else -1
            sel2 = (ni == 0)
            if sel2.any():
                cdis[L] = int(((mg - ng) + (mh - nh))[sel2].min())
            else:
                cdis[L] = -1
    return {"c": c, "cmax": cmax, "cdis": cdis, "maxg": mg_arr, "maxh": mh_arr, "joint": jm}


def pair_c(g, sg, h, sh, Lmax):
    """Just c(g,h;L), L = 1..Lmax (faster: no split)."""
    return pair_profile(g, sg, h, sh, Lmax, want_split=False)["c"]


def block_joint_max(gears, seps, Lmax):
    """joint_max(B; L) for L = 1..Lmax, exact over one period of the block."""
    P = 1
    for g in gears:
        P *= g
    n = P + Lmax + 2
    U = np.zeros(n, dtype=np.int8)
    for g, s in zip(gears, seps):
        U[0::g] = 1
        U[(s % g)::g] = 1
    U = U.astype(np.int32)
    cnt = np.zeros(P, dtype=np.int32)
    t = np.arange(P)
    out = np.zeros(Lmax + 1, dtype=np.int64)
    for L in range(1, Lmax + 1):
        cnt += U[t + (L - 1)]
        out[L] = int(cnt.max())
    return out


def block_deficit(gears, seps, Lmax):
    jm = block_joint_max(gears, seps, Lmax)
    out = np.zeros(Lmax + 1, dtype=np.int64)
    for L in range(1, Lmax + 1):
        out[L] = sum(maxstrike(g, s, L) for g, s in zip(gears, seps)) - jm[L]
    return out, jm


def block_jm_at(gears, seps, L):
    """joint_max(B; L) at a SINGLE L, cost O(prod g + L).  Exact."""
    P = 1
    for g in gears:
        P *= g
    n = P + L + 1
    U = np.zeros(n, dtype=np.int8)
    for g, s in zip(gears, seps):
        U[0::g] = 1
        U[(s % g)::g] = 1
    pre = np.empty(n + 1, dtype=np.int32)
    pre[0] = 0
    np.cumsum(U, dtype=np.int32, out=pre[1:])
    return int((pre[L:L + P] - pre[0:P]).max())


def block_c_at(gears, seps, L):
    return sum(maxstrike(g, s, L) for g, s in zip(gears, seps)) - block_jm_at(gears, seps, L)


def pair_c_at(g, sg, h, sh, L):
    return block_c_at([g, h], [sg, sh], L)


def say_factory(lines):
    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    return say


def dump(lines, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
