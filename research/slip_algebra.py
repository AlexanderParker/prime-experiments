"""Slip algebra of the gear machine: periods, relative slip, and exact kill-turns.

This treats the machine the way it is actually built rather than running it. Everything
below is exact integer arithmetic; nothing here is a density, an average, or an estimate.

Coordinates. Work in pair-index space `k`, where `k` names the candidate pair
`(6k - 1, 6k + 1)`. Gear `q` (an odd prime, `q >= 5`) blocks exactly two classes of `k`,
given in closed form by

    u_q = (q + 1) / 6   if q = 5 mod 6,      u_q = (q - 1) / 6   if q = 1 mod 6
    teeth(q) = { u_q, q - u_q }  mod q

Sub-machines. A set `S` of gears is a sub-machine with period `P_S = prod(S)`, since the
gears are coprime. One *turn* of `S` is `P_S` steps of `k`.

Slip. Two distinct quantities are worth keeping apart:

  * the user-facing slip, `|P_S - P_T|`, the difference of two cycle lengths - how far
    one cycle falls behind the other in a single turn, realigning at `P_S * P_T`;
  * the machine-facing slip, `P_S mod q`, which is how far gear `q`'s tooth advances,
    measured inside `S`'s frame, from one turn of `S` to the next.

The second is the one that governs blocking, and because `gcd(P_S, q) = 1` the tooth
visits all `q` residues in `q` turns of `S`, each exactly once.

The turn law. Let `k0` be open under `S`. The positions `k0 + t * P_S` for `t` in
`0 .. q - 1` are all open under `S`, and gear `q` strikes exactly the two turns

    t = (+/- u_q - k0) * P_S^{-1}   (mod q)

so `q - 2` of every `q` turns survive. This is the whole of `prod (q - 2)`, derived from
slip arithmetic alone, and it is why a class can never be closed out: `q >= 5 > 2`.
"""

import itertools
from math import prod


def odd_primes_upto(limit):
    """Odd primes to `limit`, grown by trial division against the list so far."""
    primes = [2]
    n = 3
    while n <= limit:
        if all(n % p for p in itertools.takewhile(lambda p: p * p <= n, primes)):
            primes.append(n)
        n += 2
    return [p for p in primes if p > 2]


def gears_upto(limit):
    """The gears of the twin machine: odd primes from 5 up."""
    return [q for q in odd_primes_upto(limit) if q >= 5]


def tooth(q):
    """The lower tooth `u_q`, in closed form."""
    return (q + 1) // 6 if q % 6 == 5 else (q - 1) // 6


def teeth(q):
    """Both blocked classes of `k` modulo `q`."""
    u = tooth(q)
    return sorted({u % q, (q - u) % q})


def slip_table(gear_sets, targets):
    """Machine slip `P mod q` and cycle slip `|P - q|` for each sub-machine and gear."""
    rows = []
    for S in gear_sets:
        P = prod(S)
        for q in targets:
            if q in S:
                continue
            rows.append(
                {
                    "sub": S,
                    "period": P,
                    "gear": q,
                    "machine_slip": P % q,
                    "cycle_slip": abs(P - q),
                    "turns_to_realign": q,
                    "realign_at": P * q,
                }
            )
    return rows


def kill_turns(k0, S, q):
    """Which turns of `S` gear `q` strikes, from the closed-form formula."""
    P = prod(S) if S else 1
    inv = pow(P % q, -1, q)
    return sorted({((t - k0) * inv) % q for t in teeth(q)})


def kill_turns_bruteforce(k0, S, q):
    """The same turns, found by testing every turn. Used to check the formula."""
    P = prod(S) if S else 1
    return sorted(t for t in range(q) if (k0 + t * P) % q in teeth(q))


def open_classes(S):
    """Every `k` in one period of `S` that no gear of `S` blocks."""
    P = prod(S) if S else 1
    tset = {q: set(teeth(q)) for q in S}
    return [k for k in range(P) if all(k % q not in tset[q] for q in S)]


def surviving_turns(k0, S, q):
    """The `q - 2` turns of `S` that gear `q` leaves open, and where they land."""
    killed = set(kill_turns(k0, S, q))
    P = prod(S) if S else 1
    return [(t, k0 + t * P) for t in range(q) if t not in killed]


def smallest_open(gears, bound):
    """Smallest positive `k` open under every gear in `gears`, by nested turn search.

    Rather than sieving a period, this walks the sub-machine tree: refine the open
    classes of `S` one gear at a time using the turn law. Representatives above `bound`
    are dropped, which costs nothing in correctness for answers at or below `bound`: the
    level-`i` ancestor of a final `k` is `k mod P_i`, which is at most `k`, so pruning
    above `bound` never discards an ancestor of a surviving answer that is itself within
    `bound`. Without the prune the class count reaches `prod (q - 2)`, which is 214
    million by `q = 29`.

    Returns the surviving representatives and, per level, the period, the count kept,
    and the smallest positive survivor - the first candidate the machine cannot rule out
    with the gears it has so far.
    """
    reps = [0]
    trail = []
    for i, q in enumerate(gears):
        S = tuple(gears[:i])
        P = prod(S) if S else 1
        nxt = []
        for k0 in reps:
            killed = set(kill_turns(k0, S, q)) if S else set(teeth(q))
            for t in range(q):
                if t in killed:
                    continue
                k = k0 + t * P
                if k <= bound:
                    nxt.append(k)
        reps = sorted(nxt)
        positive = [k for k in reps if k > 0]
        trail.append((q, P * q, len(reps), positive[0] if positive else None))
    return reps, trail


if __name__ == "__main__":
    G = gears_upto(29)

    print("gear teeth, in closed form")
    for q in G:
        print(f"  q = {q:>2}   u_q = {tooth(q):>2}   teeth {teeth(q)}   "
              f"open classes {q - 2} of {q}")

    print("\nslip: cycle slip |P - q| against machine slip P mod q")
    sets = [(5,), (7,), (5, 7), (5, 11), (5, 7, 11)]
    print(f"  {'sub-machine':>14} {'period':>7} {'gear':>5} {'|P-q|':>7} "
          f"{'P mod q':>8} {'realign':>9}")
    for r in slip_table(sets, G):
        if r["gear"] > 13:
            continue
        print(f"  {str(r['sub']):>14} {r['period']:>7} {r['gear']:>5} "
              f"{r['cycle_slip']:>7} {r['machine_slip']:>8} {r['realign_at']:>9}")

    print("\nturn law: closed-form kill-turns against brute force")
    bad = 0
    for r in range(1, 4):
        for S in itertools.combinations(G, r):
            for q in G:
                if q in S:
                    continue
                for k0 in open_classes(S)[:6]:
                    a = kill_turns(k0, S, q)
                    b = kill_turns_bruteforce(k0, S, q)
                    if a != b:
                        bad += 1
                        print(f"  MISMATCH S={S} q={q} k0={k0}: {a} vs {b}")
    print(f"  mismatches: {bad}")

    print("\nsurvival count: every class of S keeps exactly q - 2 of its q turns")
    for S in [(5,), (5, 7), (5, 7, 11)]:
        for q in [11, 13, 17]:
            if q in S:
                continue
            counts = {len(surviving_turns(k0, S, q)) for k0 in open_classes(S)}
            print(f"  S = {str(S):>12}  adding q = {q:>2}:  survivors per class "
                  f"{sorted(counts)}  (q - 2 = {q - 2})")

    print("\nsmallest open k as gears are added, by nested turn search")
    bound = 2000
    reps, trail = smallest_open(G, bound)
    print(f"  representatives pruned above k = {bound}")
    print(f"  {'gear added':>11} {'period':>10} {'kept <= bound':>14} "
          f"{'smallest k > 0':>15} {'pair':>16}")
    for q, P, n, k in trail:
        pair = f"({6 * k - 1},{6 * k + 1})" if k else "-"
        print(f"  {q:>11} {P:>10} {n:>14} {str(k):>15} {pair:>16}")
