"""Is the admissible gap word recognised by a finite automaton?

A *gap word* is the sequence of gaps of the exposed set, in units of 3 (all gaps are multiples of 3
by the gear-3 law). A word `w = a_1 ... a_n` corresponds to the position set
`S = {0, 3a_1, 3a_1 + 3a_2, ...}`, and gear `q` *forces* one of those positions to be blocked exactly
when `W_q(S) = {s-1, s : s in S} mod q` is all of `Z_q`. Call `w` **admissible** if no gear forces,
and **forbidden** otherwise.

The key structural fact making this searchable: admissibility is **factor-closed**. If `S' ⊆ S` then
`W_q(S') ⊆ W_q(S)`, so a set that fails to cover has no subset that covers. Hence every factor of an
admissible word is admissible, and a word is *minimally* forbidden exactly when it is forbidden while
both of its length-`(n-1)` factors are admissible.

That gives a level-by-level search: extend admissible words of length `n-1` by one letter, test the
result, and keep the admissible ones for the next level. The admissible words of each length are
precisely the states reachable in the candidate automaton, so their growth answers the question. If the
minimal forbidden set is finite, the language is recognised by a finite automaton and the gap counts
`n_j` are letter statistics of a transfer matrix - the route proposed as idea 1 in
`docs/ideas-from-the-session.md`.

Caveats the search cannot escape, both reported in the output: a maximum word length and a maximum
letter. Minimal forbidden words longer or wider than the box are invisible, and gear `q`'s *own*
minimal configuration needs `(q-1)/2` letters (see `minimal_forbidden.py`), so a box of length `L`
cannot see any gear beyond `q = 2L + 1` acting alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hazard import odd_primes_upto
from minimal_forbidden import covers


def positions(word):
    """Position set of a gap word, in units of 1 with gaps multiples of 3."""
    S = [0]
    for a in word:
        S.append(S[-1] + 3 * a)
    return S


def forced_by(word, primes):
    """The first gear that forces a block inside `word`, or None if the word is admissible."""
    S = positions(word)
    for q in primes:
        if covers(S, q):
            return q
    return None


def scan(primes, maxlen, maxletter, cap=4_000_000, verbose=True):
    """Level-by-level search. Returns `(minimal_forbidden, level_sizes, truncated)`."""
    letters = list(range(1, maxletter + 1))
    admissible = [()]
    minimal = []
    level_sizes = []
    truncated = False
    for n in range(1, maxlen + 1):
        nxt = []
        for w in admissible:
            for a in letters:
                cand = w + (a,)
                if forced_by(cand, primes) is None:
                    nxt.append(cand)
                elif len(cand) == 1 or forced_by(cand[1:], primes) is None:
                    # both length-(n-1) factors are admissible: cand[:-1] is, being in `admissible`
                    minimal.append(cand)
        level_sizes.append(len(nxt))
        if verbose:
            print(f"    length {n:>2}: {len(nxt):>9} admissible, "
                  f"{sum(1 for m in minimal if len(m) == n):>6} newly minimal forbidden")
        if len(nxt) > cap:
            truncated = True
            if verbose:
                print(f"    stopped: admissible count exceeded {cap}")
            break
        if not nxt:
            break
        admissible = nxt
    return minimal, level_sizes, truncated


def longest_minimal(minimal):
    return max((len(m) for m in minimal), default=0)


if __name__ == "__main__":
    MAXLEN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    MAXLETTER = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"search box: word length <= {MAXLEN}, letters <= {MAXLETTER}")
    print(f"a box of length {MAXLEN} can see a gear acting alone only up to q = {2 * MAXLEN + 1}\n")

    prev = None
    for y in (7, 11, 13, 17, 19, 23, 29, 31):
        primes = odd_primes_upto(y)
        print(f"  gears to {y}:")
        minimal, sizes, truncated = scan(primes, MAXLEN, MAXLETTER)
        longest = longest_minimal(minimal)
        added = None if prev is None else len(set(minimal) - prev)
        print(f"    total minimal forbidden: {len(minimal)}, longest: {longest}"
              + (f", new since previous gear set: {added}" if added is not None else "")
              + (" (TRUNCATED)" if truncated else ""))
        prev = set(minimal)
        print()

    print("reading: if 'longest' stops growing and 'new since previous' reaches 0 while the box is")
    print("still wider than the longest minimal word, the minimal forbidden set is finite within the")
    print("box and a finite automaton exists. If 'longest' tracks the box, the box is the limit and")
    print("the question is unresolved at that width.")
