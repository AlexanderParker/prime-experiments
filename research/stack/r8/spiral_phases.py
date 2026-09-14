"""The primorial spiral with every gear's phase tracked (owner, 2026-09-15).

The spiral carries the base residues (the landing is -1 mod the base product at every step).
Every other gear is out of phase with that open set: its phase at a column n is n mod gear,
and it strikes the slot (n, n + 2) when the phase is 0 or gear - 2 (its two teeth).  A step
about {base, g} moves the column by 2 P g up or down, so every gear's phase slips by
+-(2 P g mod gear); the gear g itself and the base slip by 0.

Table per machine: rows = gears outside the base (and the base gears, always -1); columns =
after each step; entry = phase, marked with * when it sits on a tooth.  The last column is the
landing.  Then the correction the landing needs: for each gear on a tooth at the landing, the
phase moves the final flip {3, h} would give (6h mod gear, either direction).

usage: uv run python research/stack/r8/spiral_phases.py 31 101
"""
import sys
from sympy import primerange, isprime
from pathlib import Path

def main():
    out = [__doc__.strip(), ""]
    for q in map(int, sys.argv[1:]):
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        steps = [p for p in ps if p not in base][::-1]
        gears = [p for p in ps if p >= 5]
        n = -1; sign = 1; cols = []
        for g in steps:
            n += sign * 2 * P * g; cols.append((g, sign, n)); sign = -sign
        E = n
        out.append(f"q = {q}: base {base} (P = {P}); steps about {{base, g}} for g = {steps} (up, down, ...); landing {E} {'TWIN' if isprime(E) and isprime(E + 2) else 'struck'}")
        hdr = f"{'gear':>6} {'slip per step (2Pg mod gear)':<0}"
        out.append(f"{'step':>6} " + " ".join(f"{g:>5}{'+' if s > 0 else '-'}" for g, s, n in cols) + "   phase after each step; * = on a tooth (0 or gear-2)")
        out.append(f"{'column':>6} " + " ".join(f"{n:>6}" for g, s, n in cols))
        for h in gears:
            row = []
            for g, s, nn in cols:
                ph = nn % h
                row.append(f"{ph:>5}{'*' if ph in (0, h - 2) else ' '}")
            tag = " base, carried" if h in base else ""
            out.append(f"{h:>6} " + " ".join(row) + tag)
        # slips per step for each gear
        out.append("   slips (2 P g mod gear, signed by the step's direction):")
        for h in gears:
            if h in base: continue
            sl = []
            for g, s, nn in cols:
                v = (2 * P * g) % h; v = v if v <= h // 2 else v - h
                sl.append(f"{s * v:>6}")
            out.append(f"{h:>6} " + " ".join(sl))
        strikers = [h for h in gears if E % h in (0, h - 2)]
        out.append(f"   at the landing the gears on a tooth: {strikers}")
        if strikers:
            out.append("   correction by one more flip about {3, h'}: the phase move 6h' mod gear for each striking gear, up (+) and down (-); a move that takes every striker off its teeth and puts no other gear on one is a passing h':")
            high = [p for p in ps if p * p > q]
            for hp in high[:12]:
                moves = ", ".join(f"{h}: +{(6 * hp) % h}/-{(-6 * hp) % h}" for h in strikers)
                ok = {d: all(((E + 6 * d * hp) % h) not in (0, h - 2) for h in gears if h != hp) and q < E + 6 * d * hp <= q * q - 2 for d in (1, -1)}
                out.append(f"      h' = {hp:>4}: {moves}   up {'PASS' if ok[1] else 'no'}, down {'PASS' if ok[-1] else 'no'}")
        out.append("")
    Path("research/stack/r8/results_spiral_phases.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
