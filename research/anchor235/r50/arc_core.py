"""Branch 5d.ii.i.a THE ARC MULTISET (prover, round 50).  Core engine.

Column frame as in r48/cover_core.py: gear g strikes column k iff k = +-u_g (mod g)
with u_g = 6^{-1} mod g; separation d_g = 2 u_g = 3^{-1} mod g; short arc
a_g = min(d_g, g - d_g) (so 3 a_g = g -+ 1, and a_g is always EVEN); long arc g - a_g.

THE TYPE REDUCTION (this round's tool).  Fix a target run of L consecutive columns
0..L-1.  Split every gear by its long arc against the run:

  * SMALL at L:  g - a_g <= L - 1.  The long arc fits inside the run, so the gear can
    strike more than a domino; it is kept as a concrete prime with all g phases.
  * BIG at L:    g - a_g >= L.  Then g > L - 1, so each residue class of the gear meets
    the run at most once, and the two teeth can both be in the run only at distance
    exactly a_g.  The set of column-subsets the gear can realise inside the run is

        {} , {i, i+a}  for i + a <= L-1 , {i}  for i < a or i >= L - a

    which depends ONLY on a = a_g.  (A tooth's two neighbours sit at distance a on one
    side and g - a >= L on the other, so the far neighbour is always outside the run.)

Consequence: at level L the whole infinite pool of primes reduces to a FINITE list of
item types with multiplicities:

  * every prime p with p - a_p <= L-1 (a finite list, p <= (3L-2)/2), multiplicity 1;
  * for each even arc a < L, the type domino(a) with multiplicity = the number of primes
    in {3a-1, 3a+1} that are prime and BIG at L (0, 1 or 2);
  * one type single ("arc >= L"), multiplicity K (infinitely many primes qualify).

So the adversarial ladder A(K) computed with these items is exhaustive over ALL primes,
not over a truncated pool.  That is strictly stronger than r48's pool-149 statement.

Search: cover the LEFTMOST uncovered column with the next item (both phases for a
concrete gear; for a domino item only two useful options - the pair {pos, pos+a}, or
"pos alone", which is always realisable, either as a legal singleton or as the pair
{pos-a, pos} whose left column is already covered).  Failed states are memoised.
Exhaustive, so "not coverable by any K gears" is a proof.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r48"))
from cover_core import F_of, coverable, arcs  # noqa: E402,F401

RESULTS = os.path.join(HERE, "results")


def primes_upto(n):
    b = [True] * (n + 1)
    b[0] = b[1] = False
    for i in range(2, int(n ** .5) + 1):
        if b[i]:
            b[i * i::i] = [False] * len(b[i * i::i])
    return [i for i in range(2, n + 1) if b[i]]


_PR = set(primes_upto(200000))


def is_prime(n):
    if n < 200000:
        return n in _PR
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def arc(g):
    d = pow(3, -1, g)
    return min(d, g - d)


def build_items(L, K):
    """Item list at level L.  Each item is (kind, key, multiplicity).

    kind 'p' : concrete prime, key = p
    kind 'd' : domino of arc a, key = a
    kind 's' : single column, key = None
    """
    items = []
    conc = [p for p in primes_upto(3 * L) if p >= 5 and p - arc(p) <= L - 1]
    for p in conc:
        items.append(('p', p, 1))
    # big arcs
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
    """All realisable column-subsets, as bitmasks (used only for capacity)."""
    full = (1 << L) - 1
    if kind == 'p':
        p = key
        d = pow(3, -1, p)
        out = set()
        for o in range(p):
            m = 0
            for i in range(o, L, p):
                m |= 1 << i
            for i in range((o + d) % p, L, p):
                m |= 1 << i
            out.add(m & full)
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
    """Everything needed to answer 'can K gears block L columns?' at one L."""

    def __init__(self, L, K):
        self.L = L
        self.K = K
        self.full = (1 << L) - 1
        self.items = build_items(L, K)
        self.cap = []
        self.pmask = []          # for kind 'p': list of masks indexed by phase
        self.wins = [w for w in (10, 20, 30) if w < L]
        self.wcap = {w: [] for w in self.wins}
        for kind, key, _m in self.items:
            ms = masks_for(kind, key, L)
            self.cap.append(max(bin(m).count("1") for m in ms))
            self.pmask.append(ms if kind == 'p' else None)
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
        """The masks of item idx that cover column pos (leftmost uncovered)."""
        kind, key, _m = self.items[idx]
        L = self.L
        if kind == 'p':
            p = key
            d = pow(3, -1, p)
            outs = []
            for o in {pos % p, (pos - d) % p}:
                m = 0
                for i in range(o, L, p):
                    m |= 1 << i
                for i in range((o + d) % p, L, p):
                    m |= 1 << i
                outs.append(m & self.full)
            return outs
        if kind == 'd':
            a = key
            outs = [1 << pos]
            if pos + a <= L - 1:
                outs.append((1 << pos) | (1 << (pos + a)))
            return outs
        return [1 << pos]

    def coverable(self, node_cap=None, memo_cap=6_000_000):
        """Is there a multiset of at most K items covering 0..L-1?  Exhaustive.

        Pruning, all exact upper bounds on what the remaining items can do:
          * global capacity: the kleft largest per-item capacities must reach the
            number of uncovered columns;
          * window capacity: for each window length w in wins, the kleft largest
            per-item capacities INSIDE a window of length w must reach the number
            of uncovered columns in [pos, pos+w) - the sharp form for dominoes,
            which give at most 2 in any window however long it is.
        Dropping a memo entry costs speed, never correctness."""
        n = len(self.items)
        mult0 = bytes(min(m, 255) for _k, _key, m in self.items)
        fail = set()
        nodes = [0]
        witness = []
        cap = self.cap
        wins = self.wins
        wcap = self.wcap
        options_cache = {}

        def opts(i, pos):
            k = (i, pos)
            r = options_cache.get(k)
            if r is None:
                r = self.options(i, pos)
                options_cache[k] = r
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
            un = ~covered & self.full
            todo = bin(un).count("1")
            key = (covered, mult, kleft)
            if key in fail:
                return False
            if topsum(cap, mult, kleft) < todo:
                if len(fail) < memo_cap:
                    fail.add(key)
                return False
            pos = (un & -un).bit_length() - 1
            for w in wins:
                need = bin((un >> pos) & ((1 << w) - 1)).count("1")
                if topsum(wcap[w], mult, kleft) < need:
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


def A_exact(K, L0=1, verbose=True, node_cap=None):
    """The largest L that K gears (any primes >= 5) can block, exhaustive.

    Returns (A, witness) where A = L_max + 1 is the SPAN (the r48 convention:
    F(A) = the smallest L that cannot be covered)."""
    L = max(1, L0)
    last = None
    while True:
        lv = Level(L, K)
        ok = lv.coverable(node_cap=node_cap)
        if verbose:
            print(f"  K={K} L={L}: {'cover' if ok else 'NO COVER'} "
                  f"({lv.nodes} nodes, {len(lv.items)} item types)", flush=True)
        if not ok:
            return L, last
        last = lv.witness
        L += 1


if __name__ == "__main__":
    # Gate 1: the r48 F ladder through cover_core (unchanged tool).
    LADDER = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
    ps = [5, 7, 11, 13, 17, 19, 23, 29, 31]
    prev = 1
    for i in range(2, len(ps) + 1):
        f = F_of(ps[:i], lo=prev)
        prev = f
        q = ps[i - 1]
        print(f"gate F({{5..{q}}}) = {f}  {'OK' if LADDER[q] == f else 'MISMATCH'}",
              flush=True)
    # Gate 2: the type-reduced adversary reproduces A(K) for K <= 6.
    KNOWN = {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28}
    for K in range(1, 7):
        a, w = A_exact(K, L0=max(1, KNOWN[K] - 3), verbose=False)
        print(f"gate A({K}) = {a}  (r48: {KNOWN[K]})  "
              f"{'OK' if a == KNOWN[K] else 'MISMATCH'}", flush=True)
        print("      witness:", w, flush=True)
