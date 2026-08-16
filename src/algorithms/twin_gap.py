"""Twin prime gap by double modular blocking.

This is a direct extension of the next-prime algorithm described in the
project README. That algorithm finds the gap to the next prime after n by
computing, for every known prime q <= sqrt(n), the single residue

    r_q = -n mod q

and cycling it forward to mark every position g where n + g must be
composite. The first unmarked position of the correct parity is the prime
gap. There is no lookahead: only n's own residues are used.

The twin version changes exactly one thing. A twin pair needs *two*
simultaneous primes, n + g and n + g + 2, so every trial divisor q blocks
*two* residues instead of one:

    r_q = -n mod q          (blocks n + g)
    s_q = -(n + 2) mod q    (blocks n + g + 2)

For q = 2 the two residues coincide (both members of a twin pair are odd),
so the parity rule is unchanged. For every odd q <= sqrt(n) the two
residues are distinct, so each odd divisor removes 2 of its q positions
instead of 1.

The first position that survives all of the double blocking is the gap to
the next twin prime pair. As in the single-prime case this is exact, not a
heuristic: if g survives every divisor q <= sqrt(n + g + 2) then neither
n + g nor n + g + 2 has a prime factor <= its own square root, so both are
prime.

The one structural difference from the single-prime case is that the search
window can no longer be fixed at sqrt(n). Twin gaps are believed to grow
like log(n)^2, which is far below sqrt(n), but nothing forces a survivor to
appear inside a window chosen in advance. The window is therefore grown by
doubling until a survivor is found, which keeps the search forward-only
(no candidate ahead of n is ever tested for primality).
"""

import math

# Trial divisors are the "known primes" the algorithm assumes as input, the
# same role known_primes plays in notebooks/test11.ipynb. They are grown by the
# project's own single-residue blocking rule - no sieve and no outside
# primality test is used anywhere in this module.
_known_primes = [2, 3]


def _extend_known_primes():
    """Append the next prime using the single-residue blocked-slot rule.

    For the last known prime p, every divisor q blocks the slots g = -p mod q,
    cycled forward. The first unblocked even slot is the gap to the next prime.

    Note on the cycling bound. The README and notebooks/test11.ipynb cycle the
    slots only as far as sqrt(p), which silently assumes the prime gap after p
    never exceeds sqrt(p). That assumption fails: at p = 113, sqrt(p) is about
    10.6 while the real gap is 14, so slot 12 (blocked by 5, since 125 = 5^3) is
    never marked and the rule returns 125 as prime. The fix keeps the rule
    itself untouched and only grows the cycling reach until an unblocked slot is
    actually found inside it, with divisors taken up to sqrt(p + reach).
    """
    p = _known_primes[-1]
    reach = math.isqrt(p) + 2
    while True:
        bound = math.isqrt(p + reach) + 1
        blocked = set()
        for q in _known_primes:
            if q > bound:
                break
            g = (-p) % q
            while g <= reach:
                blocked.add(g)
                g += q
        gap = 2
        while gap <= reach and gap in blocked:
            gap += 2
        if gap <= reach:
            _known_primes.append(p + gap)
            return
        reach *= 2


def trial_divisors(limit):
    """Return all known primes <= limit, growing the cache with the algorithm."""
    while _known_primes[-1] <= limit:
        _extend_known_primes()
    lo, hi = 0, len(_known_primes)
    while lo < hi:
        mid = (lo + hi) // 2
        if _known_primes[mid] <= limit:
            lo = mid + 1
        else:
            hi = mid
    return _known_primes[:lo]


def blocked_twin_slots(n, window):
    """Mark every position g in [0, window] that cannot start a twin pair.

    Returns a bytearray of length window + 1 where a 1 means "blocked",
    together with the divisor bound used. A position is blocked when some
    trial divisor q divides n + g or n + g + 2.
    """
    bound = math.isqrt(n + window + 2) + 1
    blocked = bytearray(window + 1)
    for q in trial_divisors(bound):
        for r in ((-n) % q, (-(n + 2)) % q):
            if r <= window:
                span = (window - r) // q + 1
                blocked[r :: q] = b"\x01" * span
    return blocked, bound


def next_twin_gap(n, window=None):
    """Return the smallest g >= 1 such that n + g and n + g + 2 are twin primes.

    n must be at least 1000 so that the divisor bound stays well below n; for
    smaller n a trial divisor can coincide with a candidate and mask a real
    pair.
    """
    if n < 1000:
        raise ValueError("next_twin_gap requires n >= 1000")

    if window is None:
        window = max(64, 4 * int(math.log(n) ** 2))

    while True:
        blocked, _ = blocked_twin_slots(n, window)
        # Both members of a twin pair are odd, so g has the parity that makes
        # n + g odd. This is the same offset rule as the single-prime version.
        start = 1 if n % 2 == 0 else 2
        for g in range(start, window + 1, 2):
            if not blocked[g]:
                return g
        window *= 2


def next_twin_gap_cursor(n, extend_divisors=True):
    """Twin gap in the cursor form of rust2/src/main.rs get_next_prime_gap.

    That version keeps one running bucket per trial divisor holding its next
    blocked gap, and advances a bucket only when the gap under test passes it.
    There is no window and no cycling bound. The twin version needs two buckets
    per divisor:

        a_q  starts at -n mod q        (kills n + g)
        b_q  starts at -(n + 2) mod q  (kills n + g + 2)

    Divisor 2 is kept in the loop rather than special-cased, so the parity of
    the gap is enforced by the algorithm itself instead of by an offset rule.

    With extend_divisors the divisor bound follows the candidate, staying at
    primes <= sqrt(n + g + 2); without it the bound is fixed at sqrt(n) as in the
    Rust code.
    """
    def ceil_sqrt(m):
        r = math.isqrt(m)
        return r if r * r == m else r + 1

    divisors = []
    cursor_a = []
    cursor_b = []

    def ensure(bound):
        ds = trial_divisors(bound)
        while len(divisors) < len(ds):
            q = ds[len(divisors)]
            divisors.append(q)
            cursor_a.append((-n) % q)
            cursor_b.append((-(n + 2)) % q)

    ensure(ceil_sqrt(n))

    gap = 0
    while True:
        gap += 1
        if extend_divisors:
            ensure(ceil_sqrt(n + gap + 2))
        blocked = False
        for i, q in enumerate(divisors):
            while cursor_a[i] < gap:
                cursor_a[i] += q
            while cursor_b[i] < gap:
                cursor_b[i] += q
            if cursor_a[i] == gap or cursor_b[i] == gap:
                blocked = True
                break
        if not blocked:
            return gap


def next_twin_pair(n, window=None):
    """Return the first twin prime pair (p, p + 2) with p > n."""
    g = next_twin_gap(n, window)
    return n + g, n + g + 2


def survivor_classes(limit):
    """Count double-unblocked residue classes in one primorial period.

    For divisors up to `limit` the blocked pattern repeats with period
    prod(q <= limit). Exactly one class survives modulo 2 and q - 2 classes
    survive modulo each odd q, so the count is prod(q - 2) over odd q.
    This product is never zero, which is why the twin blocking pattern can
    never cover every position - the open question is how early in the period
    the first survivor occurs.
    """
    total = 1
    for q in trial_divisors(limit):
        if q > 2:
            total *= q - 2
    return total
