"""The stack with one mirror per layer (owner, 2026-09-15).

Gear selection as the spiral: base = lowest gears with product P at most q/2; layers = the other
gears g.  Layer g: the sieve of g from the origin up to its axis a = P g, mirrored once about the
axis onto [a, 2a]; the layer ends at its landing 2 P g - 1.  No cycling past the landing.
Slot line: the twin slots (6j - 1, 6j + 1).  Stack = all layers over each other.  Holes = slots
no layer marks.  A hole is a twin or not; a hole that is not a twin is struck by a gear whose
layer has already ended (or by a gear above q... none inside the window by the square-root
rule).  Parts of the stack = the spans between consecutive landings.  Per machine: the picture
(small q), and per span: holes, true holes (twins), false holes.

usage: uv run python research/stack/r8/stack_one_mirror.py 31 37 101 499 997 1999
"""
import sys
from sympy import primerange, isprime
from pathlib import Path

def build(q):
    ps = list(primerange(2, q + 1)); base = []; P = 1
    for p in ps:
        if P * p <= q // 2: P *= p; base.append(p)
        else: break
    gears = [g for g in ps if g not in base]
    ends = {g: 2 * P * g - 1 for g in gears}
    jmax = max(ends.values()) // 6
    j0 = q // 6 + 1
    cols = list(range(j0, jmax + 1))
    marks = {g: [False] * len(cols) for g in gears}
    stack = [False] * len(cols)
    for g in [x for x in base if x >= 5] + gears:
        end = ends.get(g, 10 ** 12)
        for i, j in enumerate(cols):
            n = 6 * j - 1
            if n > end: break
            if n % g == 0 or (n + 2) % g == 0:
                stack[i] = True
                if g in marks: marks[g][i] = True
    twin = [isprime(6 * j - 1) and isprime(6 * j + 1) for j in cols]
    return P, base, gears, ends, cols, marks, stack, twin

def main():
    out = [__doc__.strip(), ""]
    for q in map(int, sys.argv[1:]):
        P, base, gears, ends, cols, marks, stack, twin = build(q)
        out.append(f"q = {q}: base {base} (P = {P}); layers {gears}; landings {[ends[g] for g in gears]}; slot line j = {cols[0]}..{cols[-1]} (to the last landing {max(ends.values())}, window top {q*q})")
        if len(cols) <= 130:
            out.append(f"{'slot j':>10} " + "".join(str(j % 10) for j in cols))
            for g in gears:
                row = "".join("#" if marks[g][i] else ("." if 6 * cols[i] - 1 <= ends[g] else " ") for i in range(len(cols)))
                lj = (2 * P * g) // 6 - cols[0]
                if 0 <= lj < len(cols): row = row[:lj] + "L" + row[lj + 1:]
                out.append(f"{'gear '+str(g):>10} {row}")
            out.append(f"{'stack':>10} " + "".join("#" if s else "." for s in stack))
            out.append(f"{'holes':>10} " + "".join(" " if s else ("T" if twin[i] else "f") for i, s in enumerate(stack)) + "   T = hole and twin, f = hole not a twin")
        # spans between consecutive landings
        bounds = [6 * cols[0] - 1] + [ends[g] for g in gears]
        rows = []
        for k in range(len(bounds) - 1):
            lo, hi = bounds[k], bounds[k + 1]
            idx = [i for i, j in enumerate(cols) if lo < 6 * j - 1 <= hi]
            holes = [i for i in idx if not stack[i]]
            t = sum(twin[i] for i in holes); f = len(holes) - t
            tw = sum(twin[i] for i in idx)
            rows.append((k, lo, hi, gears[k] if k < len(gears) else None, len(idx), tw, len(holes), t, f))
        out.append(f"   spans between landings: (span, from, to, layers ending at 'to', slots, twins in span, holes, true holes, false holes)")
        for r in rows: out.append(f"   {r}")
        out.append("")
    Path("research/stack/r8/results_stack_one_mirror.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
