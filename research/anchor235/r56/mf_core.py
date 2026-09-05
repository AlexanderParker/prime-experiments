"""mf_core.py -- the merge forest of the gear ladder, built exactly.

The merge law (docs/proofs/05 (D)) says every gap of M + q' is a union of consecutive gaps of M.
So on the unrolled line the rung-n partition of the integers into gaps is COARSER than the
rung-(n-1) partition, and the whole ladder is a laminar family / rooted forest.

Representation.  Adding gear q' to M makes q' copies of M's period.  Index the old gaps of the
tiled period by t = j * N_old + i  (copy j, old gap i).  The openings of M' are exactly the tiled
old openings whose residue mod q' avoids the two teeth, so:

    newpos[.] = the tiled indices of the surviving openings, in order
    ORDER of new gap s          = newpos[s+1] - newpos[s]        (the number of parents)
    PARENTS of new gap s        = tiled old gaps newpos[s] .. newpos[s+1]-1
    SIZE of new gap s           = O_new[s+1] - O_new[s]
    BIRTH rung                  = n if order > 1 else birth of its unique parent

Ancestor counts at every layer are carried along the same way, using the exact tiled prefix sum
    S_tiled[t] = (t // N) * Total + S[t % N]
so nothing of size q' * N is ever materialised.

This module builds levels m5 .. m23 on full periods and exposes the descent helpers.
"""
import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31]


def u_of(g):
    return pow(6, -1, g)


def teeth(g):
    u = u_of(g)
    return u % g, (-u) % g


class Level:
    """One rung of the ladder, on its full period."""

    def __init__(self, n, gears, P, O, newpos, order, size, birth, anc):
        self.n = n                  # rung index (0 = {5})
        self.gears = gears          # list of gears
        self.q = gears[-1]
        self.P = P                  # period
        self.O = O                  # openings, sorted, in [0, P)
        self.N = O.size             # = number of gaps
        self.newpos = newpos        # tiled index in the level below (None at n = 0)
        self.order = order          # number of parents
        self.size = size            # gap sizes
        self.birth = birth          # birth rung
        self.anc = anc              # dict layer index k -> ancestor count array
        self.F = int(size.max())

    def prefix(self, arr):
        s = np.zeros(self.N + 1, dtype=np.int64)
        np.cumsum(arr, out=s[1:])
        return s


def tiled_sum(pref, total, N, t0, t1):
    """sum of the tiled array over tiled indices [t0, t1), exact, no materialisation."""
    return (t1 // N - t0 // N) * total + pref[t1 % N] - pref[t0 % N]


def build_levels(gears=PRIMES[:7], vs=None, verbose=False):
    """Full-period forest for {5}, {5,7}, ..., gears.  vs = tooth values (family), default real."""
    if vs is None:
        vs = [u_of(g) for g in gears]
    levels = []
    g0, v0 = gears[0], vs[0]
    P = g0
    blocked = np.zeros(P, dtype=bool)
    blocked[v0 % g0::g0] = True
    blocked[(-v0) % g0::g0] = True
    O = np.flatnonzero(~blocked).astype(np.int64)
    size = np.diff(np.concatenate([O, [O[0] + P]])).astype(np.int64)
    lv = Level(0, [g0], P, O, None, np.ones(O.size, dtype=np.int64), size,
               np.zeros(O.size, dtype=np.int8), {})
    levels.append(lv)

    for n in range(1, len(gears)):
        q, v = gears[n], vs[n]
        old = levels[-1]
        Nold, Pold = old.N, old.P
        t1_, t2_ = v % q, (-v) % q
        res = (old.O % q).astype(np.int64)
        parts = []
        for j in range(q):
            sh = (j * Pold) % q
            r = res + sh
            r -= q * (r >= q)
            m = (r != t1_) & (r != t2_)
            parts.append(np.flatnonzero(m).astype(np.int64) + j * Nold)
        newpos = np.concatenate(parts)
        del parts
        Onew = old.O[newpos % Nold] + (newpos // Nold) * Pold
        Pnew = Pold * q
        Nnew = newpos.size
        order = np.empty(Nnew, dtype=np.int64)
        order[:-1] = newpos[1:] - newpos[:-1]
        order[-1] = newpos[0] + q * Nold - newpos[-1]
        size = np.empty(Nnew, dtype=np.int64)
        size[:-1] = Onew[1:] - Onew[:-1]
        size[-1] = Onew[0] + Pnew - Onew[-1]
        birth = np.where(order > 1, n, old.birth[newpos % Nold]).astype(np.int8)
        # ancestor counts at every lower layer
        anc = {}
        t0 = newpos
        t1 = np.empty(Nnew, dtype=np.int64)
        t1[:-1] = newpos[1:]
        t1[-1] = newpos[0] + q * Nold
        for k in range(n):
            base = old.anc[k] if k < n - 1 else np.ones(Nold, dtype=np.int64)
            pref = old.prefix(base)
            tot = int(pref[-1])
            anc[k] = ((t1 // Nold - t0 // Nold) * tot
                      + pref[t1 % Nold] - pref[t0 % Nold])
        lv = Level(n, list(gears[:n + 1]), Pnew, Onew, newpos, order, size, birth, anc)
        levels.append(lv)
        if verbose:
            print(f"  level {n}: {{5..{q}}} P={Pnew} N={Nnew} F={lv.F} "
                  f"maxorder={int(order.max())}", flush=True)
    return levels


def parents(levels, n, i):
    """Indices of the parent gaps at level n-1 of gap i at level n (in level n-1's period)."""
    lv = levels[n]
    t0 = int(lv.newpos[i])
    t1 = int(lv.newpos[i + 1]) if i + 1 < lv.N else int(lv.newpos[0]) + lv.q * levels[n - 1].N
    Nold = levels[n - 1].N
    return [(t % Nold) for t in range(t0, t1)]


def layer_words(levels, n, i):
    """The gap word at every layer from n down to 0, for gap i of level n.  Returns dict
    layer -> (list of indices, list of sizes)."""
    out = {n: ([i], [int(levels[n].size[i])])}
    cur = [i]
    for k in range(n, 0, -1):
        nxt = []
        for a in cur:
            nxt.extend(parents(levels, k, a))
        out[k - 1] = (nxt, [int(levels[k - 1].size[a]) for a in nxt])
        cur = nxt
    return out
