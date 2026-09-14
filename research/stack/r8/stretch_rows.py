"""The stretch (q, 7q] against the columns E_1 + 6h: which rows paint, as the fields explorer
paints them (row g at every multiple of g).

From the level-1 landing E_1 <= q the final flip with a gear h lands at L = E_1 + 6h <= 7q.  A
member of the landing that is composite has a prime factor at most sqrt(7q), so the rows that can
paint a landing column are the gears up to sqrt(7q), about 2.6 sqrt q: the sub-machine's gears
and the next few.  For each machine: the largest painting row seen, the rows that paint at all,
and for each row g the two classes of h it paints (h = -E_1 6^{-1} and -(E_1 + 2) 6^{-1} mod g).
Then per column: rows painting it, or PASS.

usage: uv run python research/stack/r8/stretch_rows.py 499 1999
"""
import sys
from sympy import primerange, factorint
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
from levels_rule_and_final import levels_landing

def main():
    out = [__doc__.strip(), ""]
    for q in map(int, sys.argv[1:]):
        ps = list(primerange(2, q + 1)); r = int(q ** 0.5); high = [p for p in ps if p > r]
        bounds, lands = levels_landing(q, ps); E1 = [L for k, b, bk, gk, L in lands if k == 1][0]
        cols = [(h, E1 + 6 * h) for h in high if q < E1 + 6 * h and E1 + 6 * h + 2 <= q * q]
        rows = {}; maxrow = 0
        per = []
        for h, L in cols:
            painters = []
            for m in (L, L + 2):
                f = factorint(m)
                for p in f:
                    if p >= 5 and (len(f) > 1 or f[p] > 1):
                        painters.append((p, m)); rows.setdefault(p, []).append(h); maxrow = max(maxrow, p)
            per.append((h, L, sorted(set(painters))))
        out.append(f"q = {q}: E_1 = {E1}, columns E_1 + 6h for h in ({(q - E1) / 6:.1f}, {q}]: {len(cols)}; stretch top 7q = {7 * q}, sqrt(7q) = {(7 * q) ** 0.5:.1f}; largest painting row {maxrow}; rows that paint: {sorted(rows)}")
        for g in sorted(rows):
            inv = pow(6, -1, g); a, b = (-E1 * inv) % g, (-(E1 + 2) * inv) % g
            out.append(f"   row {g:>3}: classes of h {a}, {b} (mod {g}); paints the columns of h = {rows[g]}")
        out.append("   columns:")
        for h, L, painters in per:
            out.append(f"      h = {h:>4}  ({L}, {L + 2})  " + ("PASS" if not painters else "; ".join(f"row {p} at {m}" for p, m in painters)))
        out.append("")
    Path("research/stack/r8/results_stretch_rows.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(l for l in out[2:] if not l.startswith("      h") or 'PASS' in l))

if __name__ == "__main__":
    main()
