"""The walk backwards for the killed twin candidates: columns (n, n+2) in the window (q, q^2]
with a composite member. Same flips as backward_walk.py: to the gear pair (g, g+2) the axis is
(n + g + 2)/2 and the carried gears are those dividing n + g + 2.
For a killed column the killers are the gears striking it (h | n or h | n + 2). A carried gear
keeps openness, so a killer h can only be carried onto an anchor that h also strikes: the gear
pair containing h. Reported per machine: (a) the table for the small machine; (b) the rule
check: a killer h is carried exactly onto the gear pair with member h, on the matching side
(h | n + 2 -> the pair (h, h + 2), h | n -> the pair (h - 2, h)), and never onto any other
anchor; (c) how many killed columns can have every killer certified this way (every killer is
a member of a gear pair) and how many have a killer no gear pair contains.
Usage: uv run python backward_walk_killed.py q [q ...]
"""
import sys
from sympy import primerange, isprime


def run(q):
    gears = list(primerange(5, q + 1))
    pairs = [g for g in primerange(5, q - 1) if isprime(g + 2) and g + 2 <= q]
    members = set(pairs) | set(g + 2 for g in pairs)
    killed = [n for n in range(q + 1, q * q) if n % 6 == 5 and not (isprime(n) and isprime(n + 2))]
    print(f"\n== machine {q}: gear pairs {[(g, g + 2) for g in pairs]}; killed columns in the window {len(killed)}")
    def killers(n): return [h for h in gears if n % h == 0 or (n + 2) % h == 0]
    if q <= 13:
        print("(a) killed column: killers (side) | flips to each gear pair: axis sum, carried gears")
        for n in killed:
            ks = [f"{h}{'L' if n % h == 0 else 'R'}" for h in killers(n)]
            print(f"   ({n}, {n+2}) killers {ks} | " + "; ".join(f"-> ({g}, {g+2}) {n+g+2}: {[h for h in [2,3]+gears if (n+g+2) % h == 0]}" for g in pairs))
    # (b) rule check
    viol = 0; carried_onto = 0
    for n in killed:
        for h in killers(n):
            for g in pairs:
                carried = (n + g + 2) % h == 0
                expected = (g == h and (n + 2) % h == 0) or (g + 2 == h and n % h == 0)
                if carried != expected: viol += 1
                if carried: carried_onto += 1
    print(f"(b) killer carried exactly onto the gear pair containing it, matching side: violations {viol} over {sum(len(killers(n)) for n in killed)} killer instances; killer-carrying flips found {carried_onto}")
    # (c)
    allc = sum(1 for n in killed if all(h in members for h in killers(n)))
    onec = sum(1 for n in killed if any(h in members for h in killers(n)))
    none = len(killed) - onec
    print(f"(c) killed columns whose every killer sits in a gear pair: {allc}; with at least one such killer: {onec}; with no killer in any gear pair: {none}")
    # killers by gear, and whether that gear is a pair member
    from collections import Counter
    c = Counter(h for n in killed for h in killers(n))
    print("    strikes on killed columns by gear (pair member marked *): " + ", ".join(f"{h}{'*' if h in members else ''}:{c[h]}" for h in gears))


if __name__ == "__main__":
    for a in sys.argv[1:]: run(int(a))
