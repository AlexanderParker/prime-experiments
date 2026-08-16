"""A counting route to bounding F(2,y), by contradiction.

The shape of the argument, following the proof-by-contradiction form: the constructor finds a
twin in the window unless the whole window is threatened, that is unless a *covering* exists;
and a covering cannot exist because there are too few offset vectors for one to be among them.

Setting (halved coordinates, as in `rust2/src/bin/maxgap.rs`). Each odd prime `q <= y` blocks the
adjacent residue pair `{o_q, o_q + 1} mod q`, with one offset chosen per prime. Position `i` of a
run is *covered* when some prime blocks it. Write

    Q = {3, 5, ..., y}      P = prod_{q in Q} q      d = prod_{q in Q} (1 - 2/q)

so `d` is the chance a single position escapes every prime, and `N(L)` for the number of offset
vectors covering all of `[0, L)`. Then `F_h(y)` is the least `L` with `N(L) = 0`.

**Conjectured lemma.**  N(L) <= P (1 - d)^L.

Equivalently, with offsets uniform and independent, `Pr[every position covered] <= (1 - d)^L`:
the events "position `i` is covered" are no more likely to hold simultaneously than if they were
independent. Given the lemma, `N(L) < 1` forces `N(L) = 0`, so

    F_h(y) <= L_0(y) = ceil( log P / -log(1 - d) )

which is about `theta(y)/d(y)`, of order `y log^2 y` - comfortably below the `y^2/2` the window
needs, for every `y >= 23`.

**Why the lemma needs the prime 3, and why it is plausible.** For a single prime the chance that
two given positions both escape is `1 - k/q`, where `k` counts the offsets that would block
either: `k = 4` when the positions are at distance `>= 2`, and `k = 3` when they are adjacent,
since then the forbidden offsets `{i-1, i, i+1}` overlap. So

    distance >= 2 : factor (1 - 4/q) < (1 - 2/q)^2   -> negatively correlated
    adjacent      : factor (1 - 3/q) > (1 - 2/q)^2   -> positively correlated, for q >= 5

and the adjacent factor at `q = 3` is `1 - 3/3 = 0` exactly. **Gear 3 blocks `{o, o+1}` of three
residues, leaving only `o + 2`, so of any two adjacent positions at least one is always blocked
by gear 3 alone.** The single positively-correlated family is therefore annihilated, and this is
measurable: for `{5, 7, 11}` the adjacent pair probability is `0.1662` against `d^2 = 0.1230`
(positive), while for `{3, 5, 7, 11}` it is exactly `0`.

Consistently with that, the lemma is violated by gear sets omitting 3 - `{5,7}` at `L = 2`,
`{7,11,13}` with ratio `1.29` - and holds with zero violations for every set containing 3 that
has been checked. The machine always contains 3: it is half of the `2, 3` block that makes the
6-cycle.

**Status: conjecture.** Verified exhaustively for the sets below, not proved.
"""

import itertools
import sys
from math import ceil, log, prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# exact F_h values established by the covering search in rust2/src/bin/maxgap.rs
EXACT_FH = {5: 6, 7: 15, 11: 21, 13: 33, 17: 54, 19: 75, 23: 102,
            29: 129, 31: 174, 37: 264, 41: 273, 43: 309}


def odd_primes_upto(limit):
    return [p for p in range(3, limit + 1)
            if all(p % k for k in range(2, int(p**0.5) + 1))]


def density(primes):
    """`d`: chance a single position escapes every prime."""
    return prod(1 - 2 / q for q in primes)


def bound_L0(primes):
    """Least `L` with `P (1 - d)^L < 1`, the bound the lemma would give."""
    P = prod(primes)
    d = density(primes)
    return ceil(log(P) / -log(1 - d))


def masks_for(q, Lmax):
    """Blocked-position bitmask over `[0, Lmax)` for each offset of `q`."""
    out = []
    for o in range(q):
        m = 0
        for i in range(Lmax):
            if i % q in (o % q, (o + 1) % q):
                m |= 1 << i
        out.append(m)
    return out


def covering_counts(primes, Lmax, chunk_head=2):
    """`N(L)` for every `L <= Lmax`, by enumerating offset vectors in chunks.

    Each vector's coverage is summarised by the index of its lowest uncovered position, so one
    pass over the vectors yields every `N(L)` at once.
    """
    head, tail = primes[:chunk_head], primes[chunk_head:]
    hm = [masks_for(q, Lmax) for q in head]
    tm = [masks_for(q, Lmax) for q in tail]
    hist = [0] * (Lmax + 2)
    for hc in itertools.product(*hm):
        base = 0
        for m in hc:
            base |= m
        acc = [base]
        for ms in tm:
            acc = [a | m for a in acc for m in ms]
        for a in acc:
            reach = ((~a) & (a + 1)).bit_length() - 1  # lowest zero bit
            hist[min(reach, Lmax + 1)] += 1
    return [sum(hist[L:]) for L in range(1, Lmax + 1)]


def check_lemma(primes, Lmax):
    """Worst ratio of `N(L)` to the conjectured bound, and the true `F_h`."""
    P = prod(primes)
    d = density(primes)
    counts = covering_counts(primes, Lmax)
    worst, at = 0.0, None
    violations = 0
    for L in range(1, Lmax + 1):
        pred = P * (1 - d) ** L
        r = counts[L - 1] / pred
        if r > worst:
            worst, at = r, L
        if counts[L - 1] > pred + 1e-9:
            violations += 1
    fh = next((L for L in range(1, Lmax + 1) if counts[L - 1] == 0), None)
    return {"primes": tuple(primes), "P": P, "d": d, "worst_ratio": worst,
            "at_L": at, "violations": violations, "true_Fh": fh}


def pair_probabilities(primes, span=12):
    """Measured probability that two positions both escape, adjacent and non-adjacent."""
    adj = non = total = 0
    adj_pairs = non_pairs = 0
    for offs in itertools.product(*[range(q) for q in primes]):
        surv = {i for i in range(span)
                if not any(i % q in (o % q, (o + 1) % q) for q, o in zip(primes, offs))}
        adj += sum(1 for i in range(span - 1) if i in surv and i + 1 in surv)
        non += sum(1 for i in range(span) for j in range(i + 2, span)
                   if i in surv and j in surv)
        total += 1
    adj_pairs = (span - 1) * total
    non_pairs = sum(1 for i in range(span) for j in range(i + 2, span)) * total
    return adj / adj_pairs, non / non_pairs


if __name__ == "__main__":
    print("lemma check: N(L) <= P (1 - d)^L")
    print(f"  {'primes':>26} {'P':>9} {'d':>9} {'worst ratio':>12} {'violations':>11} "
          f"{'true F_h':>9}")
    for primes, Lmax in (([3, 5], 8), ([3, 5, 7], 18), ([3, 5, 7, 11], 24),
                         ([3, 5, 7, 11, 13], 36), ([3, 5, 7, 11, 13, 17], 58),
                         ([3, 5, 7, 11, 13, 17, 19], 80),
                         ([5, 7], 8), ([5, 7, 11], 10), ([7, 11, 13], 14)):
        r = check_lemma(primes, Lmax)
        print(f"  {str(r['primes']):>26} {r['P']:>9} {r['d']:>9.6f} "
              f"{r['worst_ratio']:>12.4f} {r['violations']:>11} {str(r['true_Fh']):>9}")

    print("\nthe adjacent-pair mechanism: gear 3 forces the positive correlation to zero")
    print(f"  {'primes':>18} {'d^2':>10} {'Pr[adjacent]':>13} {'Pr[distance>=2]':>16} "
          f"{'adjacent sign':>14}")
    for primes in ([3, 5, 7], [5, 7, 11], [3, 5, 7, 11]):
        a, n = pair_probabilities(primes)
        d = density(primes)
        sign = "POSITIVE" if a > d * d else "negative"
        print(f"  {str(primes):>18} {d * d:>10.6f} {a:>13.6f} {n:>16.6f} {sign:>14}")

    print("\nif the lemma holds, does the bound cover every y?")
    print("  need F_h(y) < y^2/2. exact values settle y <= 43; the bound must settle the rest.")
    print(f"  {'y':>5} {'exact F_h':>10} {'bound L0':>9} {'y^2/2':>9} {'exact ok':>9} "
          f"{'bound ok':>9}")
    for y in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 101, 1009):
        primes = odd_primes_upto(y)
        L0 = bound_L0(primes)
        ex = EXACT_FH.get(y)
        print(f"  {y:>5} {str(ex) if ex else '-':>10} {L0:>9} {y * y / 2:>9.1f} "
              f"{str(ex < y * y / 2) if ex else '-':>9} {str(L0 < y * y / 2):>9}")
    print("\n  the two ranges overlap from y = 23 to y = 43, so the union is complete.")
