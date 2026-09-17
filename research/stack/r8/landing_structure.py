"""Round 73 / loop entry 78: does the landing's algebraic shape matter, or only its divisibility?

The machine has exactly one way to switch a gear off: the gear must divide the mirror, that is
the landing (`landing_gear_never_strikes`).  The question this settles is whether anything ELSE
about a landing helps - whether a landing with strong algebraic structure but few gears does
better than a shapeless landing that happens to carry more.

Group A: landings of the form 2^a 3^b - maximal structure, only the gears 2 and 3 carried.
Group B: the next twin centre above each, whatever its shape.
Measure: the smallest multiplier j with t j again a twin centre.
"""

import sys

from sympy import factorint, isprime


def smallest_j(t, cap=4000):
    j = 2
    while j <= cap:
        if isprime(t * j - 1) and isprime(t * j + 1):
            return j
        j += 1
    return None


def main():
    structured = []
    seen = set()
    for a in range(1, 40):
        for b in range(1, 25):
            t = 2 ** a * 3 ** b
            if 100 <= t <= 10 ** 7:
                seen.add(t)
    for t in sorted(seen):
        if isprime(t - 1) and isprime(t + 1):
            structured.append(t)

    control = []
    for t in structured:
        c = t
        while True:
            c += 6
            if isprime(c - 1) and isprime(c + 1):
                control.append(c)
                break

    print("landings of the form 2^a 3^b against the next twin centre above each")
    print("     t = 2^a 3^b   gears   smallest j       control t   gears   smallest j")
    ja, jb = [], []
    for t, c in zip(structured, control):
        x, y = smallest_j(t), smallest_j(c)
        ja.append(x)
        jb.append(y)
        print(
            "  %14d  %6d  %11s    %12d  %6d  %11s"
            % (t, len(factorint(t)), x, c, len(factorint(c)), y)
        )
    print()
    print(
        "  mean smallest multiplier: structured %.1f (%d landings, 2 gears each), control %.1f (%d landings)"
        % (sum(ja) / len(ja), len(ja), sum(jb) / len(jb), len(jb))
    )
    print(
        "  mean gears carried: structured %.2f, control %.2f"
        % (
            sum(len(factorint(t)) for t in structured) / len(structured),
            sum(len(factorint(c)) for c in control) / len(control),
        )
    )


if __name__ == "__main__":
    sys.exit(main())
