"""r53 sk_core - self-contained core for the small-K adversarial theorems.

Frame.  Column k is the pair (6k-1, 6k+1).  A gear is a prime g >= 5; it strikes column k
iff k = +-u_g (mod g) with u_g = 6^{-1} (mod g).  The separation of the two teeth is
d_g = 2 u_g = 3^{-1} (mod g); the SHORT ARC is a_g = min(d_g, g - d_g) and the LONG ARC is
g - a_g.  From 6 u_g = g -+ 1 one gets 3 a_g = g -+ 1, so a_g is even and the long arc is
2 a_g -+ 1.

A set S of gears COVERS L if some choice of phases makes all of L consecutive columns struck.
F(S) = the least L that S cannot cover.  A(K) = max{F(S) : |S| = K}.

Two independent engines, both exhaustive:

  (1) cover_set(S, L)     - direct search over the phases of an EXPLICIT prime set.
  (2) Level(L, K)         - the type-reduced search over ALL primes >= 5 at once, using

      TYPE LEMMA.  If g - a_g >= L then g > L, each residue class of g meets a run of L
      consecutive columns at most once, and the two teeth can both lie in the run only at
      distance exactly a_g.  So the column-subsets of {0..L-1} the gear can realise are
      exactly  {} , {i, i+a} with i+a <= L-1 , {i} with i < a or i >= L-a,  a function of
      a_g alone.

      Hence at level L the infinite pool of primes collapses to a finite item list:
        * every prime p >= 5 with p - a_p <= L-1 (a finite list, p < 3L/2 + 1), multiplicity 1;
        * for each even a < L, the type domino(a) with multiplicity the number of primes among
          {3a-1, 3a+1} that are prime and BIG at L (0, 1 or 2);
        * one type "single" (arc >= L), multiplicity K (infinitely many primes qualify).

      A search over that item list quantifies over ALL primes, not a truncated pool.

No third-party imports; standard library only.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def primes_upto(n):
    b = bytearray([1]) * (n + 1)
    b[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if b[i]:
            b[i * i::i] = bytearray(len(b[i * i::i]))
    return [i for i in range(2, n + 1) if b[i]]


_PR = set(primes_upto(300000))


def is_prime(n):
    if n < 300000:
        return n in _PR
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def sep(g):
    """d_g = 2 u_g = 3^{-1} (mod g)."""
    return pow(3, -1, g)


def arc(g):
    d = sep(g)
    return min(d, g - d)


# ---------------------------------------------------------------- engine (1)

def strike_mask(g, phase, L):
    """Columns of 0..L-1 struck by gear g when its low tooth sits at column = phase (mod g)."""
    d = sep(g)
    m = 0
    for i in range(phase % g, L, g):
        m |= 1 << i
    for i in range((phase + d) % g, L, g):
        m |= 1 << i
    return m


def cover_set(S, L, want_witness=False):
    """Exhaustive: can the explicit gear set S cover L consecutive columns?

    Branch on the LEFTMOST uncovered column: it must be struck by some unused gear, and a
    gear strikes a given column in exactly two ways (that column is its low tooth or its
    high tooth).  Complete, and it terminates because each step spends one gear."""
    full = (1 << L) - 1
    S = sorted(S)
    n = len(S)
    cache = {}

    def rec(covered, used):
        if covered == full:
            return ()
        if used == (1 << n) - 1:
            return None
        key = (covered, used)
        if key in cache:
            return cache[key]
        un = ~covered & full
        pos = (un & -un).bit_length() - 1
        out = None
        for i in range(n):
            if used >> i & 1:
                continue
            g = S[i]
            d = sep(g)
            for ph in {pos % g, (pos - d) % g}:
                m = strike_mask(g, ph, L)
                r = rec(covered | m, used | (1 << i))
                if r is not None:
                    out = ((g, ph),) + r
                    break
            if out is not None:
                break
        cache[key] = out
        return out

    w = rec(0, 0)
    if want_witness:
        return (w is not None), w
    return w is not None


def F_of(S, lo=1):
    """The least L that S cannot cover (search upward from lo)."""
    L = max(1, lo)
    while cover_set(S, L):
        L += 1
    return L


# ---------------------------------------------------------------- engine (2)

def build_items(L, K):
    """The type-reduced item list at level L: (kind, key, multiplicity)."""
    items = []
    for p in primes_upto(3 * L + 4):
        if p >= 5 and p - arc(p) <= L - 1:
            items.append(('p', p, 1))
    a = 2
    while a < L:
        m = 0
        for g in (3 * a - 1, 3 * a + 1):
            if g >= 5 and is_prime(g) and arc(g) == a and g - a >= L:
                m += 1
        if m:
            items.append(('d', a, m))
        a += 2
    items.append(('s', None, K))
    return items


def masks_for(kind, key, L):
    """Every column-subset of {0..L-1} the item can realise, as bitmasks."""
    full = (1 << L) - 1
    if kind == 'p':
        p = key
        out = {strike_mask(p, ph, L) & full for ph in range(p)}
        return sorted(out)
    if kind == 'd':
        a = key
        out = {0}
        for i in range(L):
            if i + a <= L - 1:
                out.add((1 << i) | (1 << (i + a)))
            if i < a or i >= L - a:
                out.add(1 << i)
        return sorted(out)
    return [0] + [1 << i for i in range(L)]


class Level:
    """Everything needed to answer 'can K primes >= 5 block L columns?' at one L."""

    def __init__(self, L, K):
        self.L = L
        self.K = K
        self.full = (1 << L) - 1
        self.items = build_items(L, K)
        self.cap = []
        self.wins = [w for w in (8, 16, 24, 32) if w < L]
        self.wcap = {w: [] for w in self.wins}
        for kind, key, _m in self.items:
            ms = masks_for(kind, key, L)
            self.cap.append(max(bin(m).count("1") for m in ms))
            for w in self.wins:
                wm = (1 << w) - 1
                best = 0
                for m in ms:
                    for s in range(0, L):
                        c = bin((m >> s) & wm).count("1")
                        if c > best:
                            best = c
                self.wcap[w].append(best)

    def options(self, idx, pos):
        """Masks of item idx that cover column pos (the leftmost uncovered one).

        For a domino of arc a the option 'pos alone' is always realisable: as the legal
        singleton {pos} if pos < a, and otherwise as the pair {pos-a, pos} whose left column
        is already covered.  So the two options below are complete at the leftmost hole."""
        kind, key, _m = self.items[idx]
        L = self.L
        if kind == 'p':
            p = key
            d = sep(p)
            return [strike_mask(p, ph, L) & self.full
                    for ph in sorted({pos % p, (pos - d) % p})]
        if kind == 'd':
            a = key
            outs = [1 << pos]
            if pos + a <= L - 1:
                outs.append((1 << pos) | (1 << (pos + a)))
            return outs
        return [1 << pos]

    def coverable(self, node_cap=None, memo_cap=4_000_000):
        n = len(self.items)
        mult0 = bytes(min(m, 255) for _k, _key, m in self.items)
        fail = set()
        nodes = [0]
        witness = []
        cap = self.cap
        opt_cache = {}

        def opts(i, pos):
            k = (i, pos)
            r = opt_cache.get(k)
            if r is None:
                r = self.options(i, pos)
                opt_cache[k] = r
            return r

        def topsum(vals, mult, kleft):
            avail = []
            for i in range(n):
                m = mult[i]
                if m:
                    v = vals[i]
                    for _ in range(m if m < kleft else kleft):
                        avail.append(v)
            avail.sort(reverse=True)
            return sum(avail[:kleft])

        def rec(covered, mult, kleft):
            if covered == self.full:
                return True
            if kleft == 0:
                return False
            if node_cap is not None and nodes[0] > node_cap:
                raise RuntimeError("node cap")
            nodes[0] += 1
            key = (covered, mult, kleft)
            if key in fail:
                return False
            un = ~covered & self.full
            todo = bin(un).count("1")
            if topsum(cap, mult, kleft) < todo:
                if len(fail) < memo_cap:
                    fail.add(key)
                return False
            pos = (un & -un).bit_length() - 1
            for w in self.wins:
                need = bin((un >> pos) & ((1 << w) - 1)).count("1")
                if topsum(self.wcap[w], mult, kleft) < need:
                    if len(fail) < memo_cap:
                        fail.add(key)
                    return False
            for i in range(n):
                if not mult[i]:
                    continue
                nm = bytes(mult[:i]) + bytes([mult[i] - 1]) + bytes(mult[i + 1:])
                for m in opts(i, pos):
                    if rec(covered | m, nm, kleft - 1):
                        witness.append((self.items[i][0], self.items[i][1], m))
                        return True
            if len(fail) < memo_cap:
                fail.add(key)
            return False

        ok = rec(0, mult0, self.K)
        self.nodes = nodes[0]
        self.witness = list(reversed(witness)) if ok else None
        return ok


def coverable_any(L, K, node_cap=None):
    lv = Level(L, K)
    ok = lv.coverable(node_cap=node_cap)
    return ok, lv


def A_exact(K, L0=1, node_cap=None, verbose=False):
    """The least L that no K primes >= 5 can cover.  Exhaustive over ALL primes."""
    L = max(1, L0)
    last = None
    while True:
        ok, lv = coverable_any(L, K, node_cap=node_cap)
        if verbose:
            print(f"  K={K} L={L}: {'cover' if ok else 'NO COVER'} "
                  f"({lv.nodes} nodes, {len(lv.items)} item types)", flush=True)
        if not ok:
            return L, last
        last = lv.witness
        L += 1
