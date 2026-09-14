"""Mirror overlays (owner, 2026-09-14): no chaining.  Each gear g above the base gives a LAYER,
the landings of the flip about {base, g} from home with any number of periods: the columns
2 k P g - 1 (k >= 1) inside the window (q, q^2].  The base layer is the flip about {base} alone:
the columns 2 m P - 1.  A column 2 m P - 1 is in layer g iff g | m; its stack depth is the number
of gears above the base dividing m; it is open to the base and to every gear whose layer holds
it (residue -1 carried).

Per machine: for each layer, its columns and which are twins; the overlay: every twin of the
base layer with its depth and the layers holding it; the twins held by no gear layer.

usage: uv run python research/stack/r8/layer_stack.py 101 499
"""
import sys
import numpy as np
from sympy import primerange, factorint
from pathlib import Path

def main():
    out = [__doc__.strip(), ""]
    for q in map(int, sys.argv[1:]):
        N = q * q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
        for i in range(2, int(N ** 0.5) + 1):
            if sv[i]: sv[i * i::i] = False
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        gears = [g for g in ps if g not in base]
        mmax = (q * q + 1) // (2 * P)
        cols = {m: 2 * m * P - 1 for m in range(1, mmax + 1) if 2 * m * P - 1 > q}
        twins = {m: n for m, n in cols.items() if sv[n] and sv[n + 2]}
        out.append(f"q = {q}: base {base} (P = {P}); base layer = columns 2mP - 1, m = 1..{mmax}: {len(cols)} columns in the window, {len(twins)} twins")
        out.append("layers (gear: columns k -> 2kPg - 1, T marks a twin):")
        for g in gears:
            ks = [k for k in range(1, mmax // g + 1) if 2 * k * P * g - 1 > q]
            if not ks: continue
            marks = " ".join(f"{k}{'T' if (k * g) in twins else ''}" for k in ks)
            out.append(f"   gear {g:>3}: {len(ks):>3} columns, {sum((k * g) in twins for k in ks):>2} twins; k = {marks[:150]}{'...' if len(marks) > 150 else ''}")
        out.append("overlay, the twins of the base layer with their depth (gears above the base dividing m) and layers:")
        nolayer = []
        for m, n in twins.items():
            f = factorint(m); layers = sorted(p for p in f if p in gears)
            depth = len(layers)
            if depth == 0: nolayer.append((m, n))
            out.append(f"   m = {m:>4}: ({n}, {n + 2})  depth {depth}  layers {layers}  m = " + "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items())))
        out.append(f"twins held by no gear layer (m with no gear factor above the base): {[(m, n) for m, n in nolayer]}")
        out.append("")
    Path("research/stack/r8/results_layer_stack.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
