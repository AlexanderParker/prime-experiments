"""How large a configuration must be before gear q can block one of it, and what that costs.

In halved coordinates each odd prime `q` blocks the adjacent pair `{o, o+1} mod q` for its offset
`o`. For a set `S` of positions, gear `q` is *forced* to block one of `S` exactly when

    W_q(S) = { s-1, s : s in S }  mod q   ==   Z_q

because then no offset of `q` avoids all of `S`. If `|W_q(S)| < q` there is an offset leaving every
position of `S` simultaneously exposed.

Two things follow, and this module verifies both.

**The minimal size law.** Each position contributes a domino `{s-1, s}` of size at most 2, so
covering `Z_q` needs at least `ceil(q/2) = (q+1)/2` positions for odd `q`, and that many suffice -
choose `s` at residues `0, 2, 4, ..., q-1`, whose dominoes are `{q-1,0}, {1,2}, ..., {q-2,q-1}`.
Since `3` is invertible mod `q`, integer positions with those residues and all `= 0 mod 3` exist by
CRT. So

    minimal forbidden configuration size for gear q  =  (q+1)/2, exactly.

Read the other way round - the exposure form - **any `(q-1)/2` positions can be simultaneously
exposed to gear `q`**, whatever their spacing.

**The factorisation law.** Let `w(S) = |{s-1, s : s in S}|` counted over the *integers*. Then
`|W_q(S)| = w(S)` whenever `q > span(S) + 1`, and in particular the gears above that threshold
contribute `prod (q - w(S))`, which depends on the size and adjacency structure of `S` but not on its
placement. All remaining shape dependence sits in the gears at or below the threshold.

The threshold is `span + 1`, not `span`: the two extreme values in `{s-1, s : s in S}` are
`min(S) - 1` and `max(S)`, which differ by `span(S) + 1`, so they collide mod `q` exactly when
`q = span(S) + 1`. Checking `q > span` instead admits that one wraparound collision - for `S = {0, 12}`
and `q = 13`, `W_13 = {0, 11, 12}` has size 3 while `2|S| = 4` - and the check below caught it.
"""

import itertools
import sys
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hazard import exposed_count, gap_count, odd_primes_upto


def covers(S, q):
    """Does gear `q` have to block one of `S`? True iff the dominoes cover `Z_q`."""
    seen = set()
    for s in S:
        seen.add((s - 1) % q)
        seen.add(s % q)
    return len(seen) == q


def min_size(q):
    """Smallest `|S|` with `W_q(S) = Z_q`, by exhaustive search over residue sets.

    Positions are free to be any integers `= 0 mod 3`: since `3` is invertible mod `q`, any residue
    is reachable, so the search is over residue *sets* and spacing is not a constraint here.
    Exponential in `q`, so only run for small gears - `size_construction` covers the rest.
    """
    for k in range(1, q + 1):
        for R in itertools.combinations(range(q), k):
            if covers(R, q):
                return k, R
    return None, None


def size_construction(q):
    """The explicit `(q+1)/2` cover: residues `0, 2, 4, ..., q-1`, dominoes tiling with overlap 1."""
    R = tuple(range(0, q, 2))
    return R, covers(R, q), len(R) == (q + 1) // 2


def domino_mask(s, q):
    return (1 << ((s - 1) % q)) | (1 << (s % q))


def min_span(q, cap):
    """Smallest span of a forbidden configuration whose positions are multiples of 3.

    Dynamic programme over covered-residue bitmasks: process candidate positions `3t` in ascending
    `t`, carrying the set of reachable covers. The first `t` at which some cover becomes full is the
    minimal span, since the position just added is then the largest in the configuration.
    Returns `(span, S)` or `(None, None)` if nothing fits inside `cap`.
    """
    full = (1 << q) - 1
    # mask -> a witness configuration reaching it
    states = {domino_mask(0, q): (0,)}
    if domino_mask(0, q) == full:
        return 0, (0,)
    for t in range(1, cap // 3 + 1):
        s = 3 * t
        dm = domino_mask(s, q)
        additions = {}
        for mask, witness in states.items():
            nm = mask | dm
            if nm == full:
                return s, witness + (s,)
            if nm not in states and nm not in additions:
                additions[nm] = witness + (s,)
        states.update(additions)
    return None, None


def minimal_forbidden_words(q, cap):
    """Minimal forbidden *gap words* for gear `q`, in units of 3, within span `cap`.

    A word is the gap sequence of a forbidden configuration; minimal means no forbidden
    configuration sits strictly inside it.
    """
    found = []
    for span_t in range(1, cap // 3 + 1):
        interior = [3 * t for t in range(1, span_t)]
        end = 3 * span_t
        for r in range(len(interior) + 1):
            for T in itertools.combinations(interior, r):
                S = (0,) + T + (end,)
                if not covers(S, q):
                    continue
                word = tuple((b - a) // 3 for a, b in zip(S, S[1:]))
                if any(is_factor(w, word) for w in found):
                    continue
                found.append(word)
    return found


def is_factor(small, big):
    n, m = len(small), len(big)
    if n > m:
        return False
    return any(big[i:i + n] == small for i in range(m - n + 1))


def span_of_shape(S):
    return max(S) - min(S)


def w_over_integers(S):
    """`w(S) = |{s-1, s : s in S}|`, counted over the integers rather than mod anything."""
    return len({x for s in S for x in (s - 1, s)})


def check_factorisation(primes, L):
    """Verify that gears above `span + 1` factor out of the gap-`L` inclusion-exclusion.

    For each position set `S` arising in `gap_count`, compare `|W_q(S)|` against `w(S)` for every
    gear, and confirm the two agree exactly once `q > span(S) + 1`.
    """
    bad = []
    interior = list(range(3, L, 3))
    for r in range(len(interior) + 1):
        for T in itertools.combinations(interior, r):
            S = (0,) + T + (L,)
            threshold = span_of_shape(S) + 1
            for q in primes:
                w_mod = len({x % q for s in S for x in (s - 1, s)})
                if q > threshold and w_mod != w_over_integers(S):
                    bad.append((S, q, w_mod, w_over_integers(S)))
    return bad


def split_count(primes, S, L):
    """`exposed_count` split into the shape-carrying gears and the shape-blind tail.

    `S` spans `L`, so the tail is the gears with `q > L + 1`.
    """
    head_gears = [q for q in primes if q <= L + 1]
    tail_gears = [q for q in primes if q > L + 1]
    head = exposed_count(head_gears, S) if head_gears else 1
    tail = prod(q - w_over_integers(S) for q in tail_gears)
    return head, tail


if __name__ == "__main__":
    print("minimal forbidden configuration size, gear by gear (exhaustive)")
    print(f"  {'q':>4} {'min |S|':>8} {'(q+1)/2':>8} {'match':>6}   residues")
    for q in [3, 5, 7, 11, 13, 17, 19]:
        k, R = min_size(q)
        print(f"  {q:>4} {k:>8} {(q + 1) // 2:>8} {str(k == (q + 1) // 2):>6}   {R}")

    print("\nthe explicit (q+1)/2 construction, checked for every gear to 200")
    fails = []
    for q in odd_primes_upto(200):
        R, ok, right_size = size_construction(q)
        if not (ok and right_size):
            fails.append(q)
    print(f"  gears checked: {len(odd_primes_upto(200))}, failures: {len(fails)}")
    print(f"  and the counting bound |S| >= ceil(q/2) = (q+1)/2 is immediate, so the size is exact")

    print("\nminimal span, positions restricted to multiples of 3")
    print(f"  {'q':>4} {'min span':>9} {'span/q':>7}   configuration")
    for q in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        span, S = min_span(q, 6 * q)
        if span is None:
            print(f"  {q:>4} {'none':>9} {'-':>7}   within span {6 * q}")
        else:
            print(f"  {q:>4} {span:>9} {span / q:>7.2f}   {S}")

    print("\nminimal forbidden gap words, in units of 3")
    for q in [3, 5, 7, 11]:
        words = minimal_forbidden_words(q, 3 * (q + 2))
        if not words:
            print(f"  q = {q:>3}: none - gear 3 is what restricts positions to one class mod 3, "
                  f"so inside that class it forces nothing further")
            continue
        shortest = min(len(w) for w in words)
        longest = max(len(w) for w in words)
        sample = sorted(words, key=len)[:4]
        print(f"  q = {q:>3}: {len(words):>3} words, lengths {shortest}..{longest}, "
              f"e.g. {[''.join(map(str, w)) for w in sample]}")

    print("\ndoes the forbidden-word length stay bounded as q grows?")
    print("  minimal size is exactly (q+1)/2, so a minimal word has (q-1)/2 letters:")
    for q in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        print(f"    q = {q:>3}: minimal word length = {(q - 1) // 2}")

    print("\nfactorisation law: gears with q > span+1 contribute q - w(S) regardless of placement")
    for primes, L in (([3, 5, 7, 11, 13], 9), ([3, 5, 7, 11, 13, 17, 19], 12),
                      (odd_primes_upto(23), 15), (odd_primes_upto(31), 18)):
        bad = check_factorisation(primes, L)
        print(f"  gears to {primes[-1]:>3}, L = {L:>3}: exceptions = {len(bad)}")
        if bad:
            print(f"    first {bad[0]}")

    print("\nthe split reproduces exposed_count exactly")
    for primes, L in (([3, 5, 7, 11, 13], 9), (odd_primes_upto(23), 9), (odd_primes_upto(31), 12),
                      (odd_primes_upto(37), 18)):
        interior = list(range(3, L, 3))
        ok = True
        for r in range(len(interior) + 1):
            for T in itertools.combinations(interior, r):
                S = (0,) + T + (L,)
                head, tail = split_count(primes, S, L)
                if head * tail != exposed_count(primes, S):
                    ok = False
        print(f"  gears to {primes[-1]:>3}, L = {L:>3}: split == direct: {ok}")

    print("\nso n_j for 3j = L needs shape only from gears q <= L")
    for y in (23, 31, 37):
        primes = odd_primes_upto(y)
        for L in (3, 6, 9, 12):
            small = [q for q in primes if q <= L]
            direct = gap_count(primes, L)
            print(f"  y = {y:>3}, L = {L:>3}: shape gears {small}, n = {direct}")
