"""The machine {5..g} below the next prime's square, across all the fields.

For each column k with 6k + 1 < g'^2 (g' the next prime after g), each member 6k -+ 1 is
either prime (field 1) or composite; a composite below g'^2 has least factor <= g, so it lies
in the gear field of that least factor (part G), in the factor-count field of its Omega, and
in the square field if it is a square. Prints, per machine: the column map (one line per
column: left member's field / right member's field, twin marked), then the tally: twins,
columns killed by each gear field (as the least factor of a killing member), by each
factor-count field, by the square field; and the closed form that locates the twins: the
columns avoiding every gear field's residues (the wheel's open columns) inside the range.
Usage: uv run python machine_map.py gmax [full]   (full prints the column map for every machine; otherwise for g <= 7)
"""
import sys
from collections import Counter
from sympy import factorint, isprime, nextprime, primerange


def tag(n):
    if n < 2: return "1"
    if isprime(n): return "P"
    f = factorint(n); g = min(f); om = sum(f.values())
    sq = "sq" if len(f) == 1 and om == 2 else ""
    return f"g{g}/F{om}{('/' + sq) if sq else ''}"


def main():
    gmax = int(sys.argv[1]); full = len(sys.argv) > 2
    for g in primerange(5, gmax + 1):
        gp = nextprime(g); top = gp * gp
        cols = [k for k in range(1, top) if 6 * k + 1 < top]
        twins = []; gear_kill = Counter(); field_kill = Counter(); sq_kill = 0; lines = []
        for k in cols:
            a, b = 6 * k - 1, 6 * k + 1
            ta, tb = tag(a), tag(b)
            tw = (ta == "P" and tb == "P")
            if tw: twins.append(k)
            else:
                for n, t in ((a, ta), (b, tb)):
                    if t != "P":
                        f = factorint(n); gear_kill[min(f)] += 1; field_kill[sum(f.values())] += 1
                        if len(f) == 1 and sum(f.values()) == 2: sq_kill += 1
            lines.append(f"  k={k:3d} ({a},{b}): {ta:10s} | {tb:10s} {'TWIN' if tw else ''}")
        print(f"\nmachine {{5..{g}}}, columns below {gp}^2 = {top}: {len(cols)} columns, {len(twins)} twins at k = {twins}")
        if full or g <= 7:
            print("\n".join(lines))
        print(f"  member-kills by gear field (least factor): {dict(sorted(gear_kill.items()))}")
        print(f"  member-kills by factor-count field: {dict(sorted(field_kill.items()))}; by the square field: {sq_kill}")
        # closed form: columns avoiding every gear field residue (k = +-6^-1 mod h for h <= g)
        gears = list(primerange(5, g + 1))
        avoid = [k for k in cols if all((6 * k - 1) % h and (6 * k + 1) % h for h in gears)]
        print(f"  columns avoiding every gear residue (the wheel's open columns) in the range: {avoid} -> equal to the twins: {avoid == twins}")


if __name__ == "__main__":
    main()
