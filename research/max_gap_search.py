"""F(2, y) by searching for the longest coverable run, not by scanning the period.

Scanning a full period costs `prod(q)` steps, which stops being feasible past
y = 31. But a gap of size G in the survivor pattern is exactly a run of G - 1
consecutive *blocked* positions, and a position is blocked when some odd prime
q <= y has it in that prime's pair of blocked residues `{o_q, o_q + 1}`. Each prime
gets one offset `o_q`, chosen once.

So `F(2, y) = 1 + (longest run of positions that can all be covered this way)`, and
that is a finite constraint search over the offsets rather than a walk over the
period. The search assigns primes to the leftmost uncovered position, tries both
ways that prime's pair can cover it, and prunes when the unused primes cannot
possibly cover what is left.

Exactness comes from the search being exhaustive: every prime assignment that could
cover the leftmost uncovered position is tried, so if no assignment covers the run,
none exists.
"""

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interval_avoidance import odd_primes_upto


def coverable(L, qs):
    """Can positions 0..L-1 all be blocked, one offset chosen per prime?"""

    def cover_mask(q, o):
        m = 0
        for i in range(L):
            if i % q == o % q or i % q == (o + 1) % q:
                m |= 1 << i
        return m

    full = (1 << L) - 1
    # cheap necessary condition: each prime q can block at most this many of the L
    capacity = {q: max(len(bin(cover_mask(q, o))) for o in range(q)) for q in qs}
    del capacity  # only the masks matter; kept explicit for clarity

    masks = {q: [cover_mask(q, o) for o in range(q)] for q in qs}
    best_per_prime = {q: max(bin(m).count("1") for m in masks[q]) for q in qs}

    def search(covered, remaining):
        if covered == full:
            return True
        # prune: can the remaining primes cover the remaining positions at all?
        todo = L - bin(covered).count("1")
        if sum(best_per_prime[q] for q in remaining) < todo:
            return False
        # leftmost uncovered position
        pos = (~covered & full).bit_length() - 1
        pos = min(i for i in range(L) if not (covered >> i) & 1)
        for idx, q in enumerate(remaining):
            rest = remaining[:idx] + remaining[idx + 1 :]
            for o in (pos % q, (pos - 1) % q):
                if search(covered | masks[q][o], rest):
                    return True
        return False

    return search(0, tuple(qs))


def max_gap_search(y, cap=400):
    """Smallest L that cannot be covered gives F(2, y) = L."""
    qs = odd_primes_upto(y)
    L = 1
    while L < cap:
        if not coverable(L, qs):
            return L
        L += 1
    raise RuntimeError("cap reached")


if __name__ == "__main__":
    known = {5: 6, 7: 15, 11: 21, 13: 33, 17: 54, 19: 75, 23: 102, 29: 129, 31: 174}
    ys = [int(a) for a in sys.argv[1:]] or [5, 7, 11, 13]
    for y in ys:
        f = max_gap_search(y)
        tag = ""
        if y in known:
            tag = "matches" if f == known[y] else f"MISMATCH (period scan: {known[y]})"
        print(f"F(2,{y}) = {f}  {tag}")
