"""Which fields mirror. For machine q the mirror of n about q#/2 is q# - n; a column (n, n+2)
mirrors to the column (q# - n - 2, q# - n). For every field with kills in the window (q, q^2]
(cycle 1): the share of its kills n whose mirror image q# - n is again a kill of the SAME field,
the share whose mirror is a kill of any field, and the share whose mirror is a twin member.
The machine's own rows mirror exactly (g | n iff g | q# - n for every gear g <= q), so a field
whose kills are decided by the gears up to q alone must mirror exactly; the others are measured.
Usage: uv run python mirror_fields.py qmax
"""
import sys
from collections import defaultdict
from sympy import primerange, isprime, factorint
from window_fields import fields_of


def main():
    qmax = int(sys.argv[1])
    print("machine | field | kills in window | mirror is a kill of the same field | mirror is a kill of any field | mirror is a twin member")
    kinds = defaultdict(lambda: [0, 0])
    for q in primerange(5, qmax + 1):
        P = 1
        for r in primerange(2, q + 1): P *= r
        F = defaultdict(list)
        for n in range(q + 1, q * q + 1):
            if n % 6 not in (1, 5) or isprime(n): continue
            for f in fields_of(n, factorint(n)): F[f].append(n)
        for f in sorted(F, key=lambda s: (s.split(':')[0], int(s.split(':')[1]) if ':' in s else 0)):
            same = anyk = tw = 0
            for n in F[f]:
                m = P - n
                if isprime(m):
                    if isprime(m - 2) or isprime(m + 2): tw += 1
                    continue
                anyk += 1
                if f in fields_of(m, factorint(m)): same += 1
            k = len(F[f]); print(f"{q} | {f} | {k} | {same}/{k} | {anyk}/{k} | {tw}/{k}")
            kind = f.split(':')[0] + (':g<=q' if ':' in f and f.split(':')[0] != 'products' and int(f.split(':')[1]) <= q else (':g>q' if ':' in f and f.split(':')[0] != 'products' else ''))
            kinds[kind][0] += same; kinds[kind][1] += k
    print("\nby field kind: kills whose mirror is a kill of the same field")
    for kd, (s, k) in sorted(kinds.items()): print(f"  {kd}: {s}/{k} = {s/k:.3f}")


if __name__ == "__main__":
    main()
