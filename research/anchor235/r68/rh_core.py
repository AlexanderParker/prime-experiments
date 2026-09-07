"""rh_core.py -- shared machinery for node R4.c.ii (THE RICH HALF OF THE RECORD).

Everything is exact integer arithmetic on the anchored column coordinate: column k is the pair
(6k - 1, 6k + 1); gear g strikes k iff k = +-u_g (mod g), u_g = 6^{-1} mod g; the two teeth are
at column separation sep(g) = 2 u_g mod g.

The PULLBACK of M along q' (research/proof/rich_half.md 0.1) is the machine read along the slots
x_0 + s_j + m q'.  A set O of slot offsets is FEASIBLE iff every gear has a phase x with
x + o != +-u_g (mod g) for all o in O, i.e. iff the residues of O mod g leave two free residues at
separation 2u_g.  By the one-orbit reduction (docs/proofs/21) the phase vectors are exactly the
translates x_0 mod P(M), so "feasible" = "some translate opens all of O".

    omega(slots, gears)          the largest feasible subset of the slots (the rich function)
    count_distribution(...)      the exact number of translates x mod P(M) with each open count
                                 on the slots (a union DP over the gears' phases)
    pair deficits                c(g, h; L), joint_max, min_g, joint_min, k(g, h; n) with any
                                 separations, by sliding a window over one period gh
"""
import os
from math import gcd

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

CORPUS_F = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
            43: 103, 47: 118, 53: 145, 59: 161}
CORPUS_L = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2, 43: 2, 47: 4, 53: 3}

# record fusions (flank, middles..., flank) of M + q' as realised windows of M, from
# pad_cap.md 2.6 / r61 detail_K14_base23.json / mechanic.md (37 -> 41).  Mirror images omitted.
RECORD_FUSIONS = {
    13: [(5, 11, 2)],
    17: [(18, 7), (7, 13, 5)],
    19: [(7, 15, 8, 4)],
    23: [(23, 10, 10)],
    29: [(18, 10, 30), (23, 10, 25)],
    31: [(11, 12, 37, 28)],
    37: [(15, 41, 14, 21)],
}


def u_of(g):
    return pow(6, -1, g)


def sep(g):
    """Column separation of the two teeth of gear g."""
    return (2 * u_of(g)) % g


def arc(s, g):
    return min(s % g, (-s) % g)


def twisted_sep(g, q):
    """Separation of the pullback along q': 2 u_g q'^{-1} mod g."""
    return (2 * u_of(g) * pow(q, -1, g)) % g


def gears_of(y):
    return [p for p in PRIMES if p <= y]


def next_gear(y):
    return PRIMES[PRIMES.index(y) + 1]


def letter_data(q):
    u = u_of(q)
    d = (2 * u) % q
    a = min(d, q - d)
    return {"q": q, "u": u, "d": d, "a": a, "b": q - a}


# ------------------------------------------------------------------ feasibility over subsets

def _popcounts(n):
    pc = np.zeros(1 << n, dtype=np.int8)
    for i in range(n):
        pc[1 << i: 1 << (i + 1)] = pc[0: 1 << i] + 1
    return pc


def residue_masks(slots, g):
    """For every subset of the slots (index bitmask), the residues mod g it occupies, as a
    g-bit integer.  Doubling construction, O(2^n)."""
    n = len(slots)
    R = np.zeros(1 << n, dtype=np.uint64)
    for i, o in enumerate(slots):
        b = np.uint64(1) << np.uint64(o % g)
        R[1 << i: 1 << (i + 1)] = R[0: 1 << i] | b
    return R


def gear_feasible(slots, g, s=None):
    """Boolean over subsets: gear g (teeth at column separation s, default the real 2u_g) has a
    phase avoiding the subset, i.e. the free residues contain a pair y, y + s."""
    assert g < 64
    if s is None:
        s = sep(g)
    R = residue_masks(slots, g)
    full = np.uint64((1 << g) - 1)
    free = (~R) & full
    s = s % g
    rot = ((free << np.uint64(s)) | (free >> np.uint64(g - s))) & full   # bit y of rot = free[y - s]
    return (free & rot) != 0


def omega(slots, gears, seps=None, want_args=True):
    """The largest feasible subset of `slots` under `gears` (real teeth unless seps[g] given).
    Returns (best, argmax masks, feasibility array, popcount array)."""
    n = len(slots)
    feas = np.ones(1 << n, dtype=bool)
    for g in gears:
        feas &= gear_feasible(slots, g, None if seps is None else seps.get(g))
    pc = _popcounts(n)
    best = int(pc[feas].max())
    args = np.flatnonzero(feas & (pc == best)) if want_args else None
    return best, args, feas, pc


def omega_prefix(slots, gears):
    """Omega for every prefix {gears[0..i]}; per-gear feasibility kept.  Returns list of
    (gear, omega after adding it) and the final (best, args)."""
    n = len(slots)
    pc = _popcounts(n)
    feas = np.ones(1 << n, dtype=bool)
    chain = []
    for g in gears:
        feas &= gear_feasible(slots, g)
        chain.append((g, int(pc[feas].max())))
    best = chain[-1][1]
    args = np.flatnonzero(feas & (pc == best))
    return chain, best, args, feas, pc


def mask_to_set(mask, slots):
    return [slots[i] for i in range(len(slots)) if (mask >> i) & 1]


def phases_for(slots_open, g):
    """Phases x mod g (the residue of the reference column x_0) under which gear g strikes none
    of the given open slots."""
    u = u_of(g)
    t = {u % g, (-u) % g}
    return [x for x in range(g) if all(((x + o) % g) not in t for o in slots_open)]


# ------------------------------------------------------------------ exact distribution over translates

def count_distribution(slots, gears):
    """Number of translates x mod P(M) (P = prod gears) by the struck subset of the slots.
    Returns dict: struck-mask -> count (Python ints, exact)."""
    dist = {0: 1}
    for g in gears:
        u = u_of(g)
        t1, t2 = u % g, (-u) % g
        cnt = {}
        for x in range(g):
            B = 0
            for i, o in enumerate(slots):
                r = (x + o) % g
                if r == t1 or r == t2:
                    B |= 1 << i
            cnt[B] = cnt.get(B, 0) + 1
        new = {}
        for U, c in dist.items():
            for B, cb in cnt.items():
                new[U | B] = new.get(U | B, 0) + c * cb
        dist = new
    return dist


def open_count_distribution(slots, gears):
    """dict open-count -> number of translates, and the total P."""
    dist = count_distribution(slots, gears)
    n = len(slots)
    out = {}
    for U, c in dist.items():
        k = n - bin(U).count("1")
        out[k] = out.get(k, 0) + c
    P = 1
    for g in gears:
        P *= g
    assert sum(out.values()) == P
    return out, P


# ------------------------------------------------------------------ pair deficits (docs/proofs/21) with any separation

def strike_pattern(g, s):
    """Boolean period pattern of gear g with teeth at 0 and s."""
    p = np.zeros(g, dtype=bool)
    p[0] = True
    p[s % g] = True
    return p


def window_counts(pattern, L):
    """Number of marks in every window of length L of the cyclic pattern."""
    P = pattern.size
    reps = L // P + 2
    ext = np.tile(pattern, reps)[:P + L].astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(ext)])
    return cs[L:L + P] - cs[:P]


def max_g(g, s, L):
    return int(window_counts(strike_pattern(g, s), L).max())


def min_g(g, s, L):
    return int(window_counts(strike_pattern(g, s), L).min())


def pair_pattern(g, sg, h, sh):
    """Union pattern of two gears over the period gh (one-orbit reduction: both at phase 0)."""
    P = g * h
    r = np.arange(P)
    a = (r % g == 0) | (r % g == sg % g)
    b = (r % h == 0) | (r % h == sh % h)
    return a | b, a, b


def pair_deficits(g, sg, h, sh, Lmax):
    """c(g,h;L) = max_g + max_h - joint_max and k(g,h;n) = min_g + min_h - joint_min, L = 1..Lmax."""
    U, A, B = pair_pattern(g, sg, h, sh)
    c, k = [], []
    for L in range(1, Lmax + 1):
        wu = window_counts(U, L)
        wa = window_counts(A, L)
        wb = window_counts(B, L)
        c.append(int(wa.max() + wb.max() - wu.max()))
        k.append(int(wa.min() + wb.min() - wu.min()))
    return c, k


# ------------------------------------------------------------------ machines and scans

def sieve_machine(gears):
    P = 1
    for g in gears:
        P *= g
    blocked = np.zeros(P, dtype=bool)
    for g in gears:
        v = u_of(g)
        blocked[v % g::g] = True
        blocked[(-v) % g::g] = True
    return P, np.flatnonzero(~blocked).astype(np.int64)


def strikes(g, X):
    u = u_of(g)
    r = np.asarray(X) % g
    return (r == u % g) | (r == (-u) % g)
