"""og_maps.py -- the strike maps for the document: gear per column with the struck member written as
g x m (the dilation form), for the real engine at p = 17 and p = 29 (the finer sections), the exhibited
killer at 17, and the one-gear killer at 29 (gear 29 moved to tooth 2, og_nearest.py).
Output: results/maps.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from og_common import family_strikes_column, finer_section, is_prime, real_teeth, strikers_of_column

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

lines = []


def real_map(p):
    a, b, cols, gears = finer_section(p)
    lines.append(f"real engine {{5..{p}}} on the section columns {cols[0]}..{cols[-1]} (members {6*cols[0]-1}..{6*cols[-1]+1}):")
    for j in cols:
        st = strikers_of_column(j, gears)
        if not st:
            lines.append(f"  col {j}: ({6*j-1}, {6*j+1}) OPEN  twin prime pair")
            continue
        parts = []
        for g, s in st:
            n = 6 * j + s
            m = n // g
            parts.append(f"{n} = {g} x {m}{' (m prime)' if is_prime(m) else ''}")
        lines.append(f"  col {j}: " + "; ".join(parts))


def family_map(p, teeth, title):
    a, b, cols, gears = finer_section(p)
    lines.append(f"{title} on columns {cols[0]}..{cols[-1]}; teeth {teeth} against real {real_teeth(gears)}:")
    for j in cols:
        fam = family_strikes_column(j, gears, teeth)
        real = {g for g, s in strikers_of_column(j, gears)}
        parts = []
        for g in fam:
            if g in real:
                parts.append(f"{g} (real)")
            else:
                parts.append(f"{g} PHANTOM on ({6*j-1}, {6*j+1})")
        lines.append(f"  col {j}: " + ("; ".join(parts) if parts else "NOT STRUCK"))


real_map(17)
family_map(17, [1, 1, 2, 6, 1], "killer (1,1,2,6,1)")
real_map(29)
family_map(29, [1, 1, 2, 2, 3, 3, 4, 2], "one-gear killer (29 moved to tooth 2)")
with open(os.path.join(RES, "maps.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
