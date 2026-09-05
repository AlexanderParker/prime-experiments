"""Which teeth make the glue work?  m19 family, hard attaining runs, grouped by the small
gears' teeth (v_5, v_7) and by the number of hard runs, with the real member marked."""
import random, sys
from gl_family import score
from gl_glue import gears_of, us_of

gears = gears_of(19)
realv = tuple(min(u, g - u) for g, u in zip(gears, us_of(gears)))
rng = random.Random(4242)
rows = []
seen = set()
while len(rows) < 400:
    vs = tuple(rng.randrange(1, (g + 1) // 2) for g in gears)
    if vs in seen:
        continue
    seen.add(vs)
    F, F2, tr, trok, hd, hdok, exc = score(gears, vs)
    rows.append((vs, F, F2, hd, hdok))
rr = score(gears, realv)
print(f"real {realv}: F={rr[0]} F_2={rr[1]} hard {rr[5]}/{rr[4]}")
big = [r for r in rows if r[3] >= 10]
print(f"members with >=10 hard attaining runs: {len(big)}/400; "
      f"pooled hard rate {sum(r[4] for r in big)}/{sum(r[3] for r in big)} = "
      f"{100*sum(r[4] for r in big)/max(sum(r[3] for r in big),1):.1f}%")
above = [r for r in big if r[4] / r[3] >= rr[5] / rr[4]]
print(f"  of those, {len(above)} have a hard rate >= the real machine's "
      f"{100*rr[5]/rr[4]:.1f}%  -> real percentile {100*(1-len(above)/max(len(big),1)):.1f}")
for key in (0, 1):
    d = {}
    for r in rows:
        k = r[0][key]
        a = d.setdefault(k, [0, 0])
        a[0] += r[3]; a[1] += r[4]
    print(f"  by v_{gears[key]}: " + "  ".join(
        f"{k}: {a[1]}/{a[0]}={100*a[1]/max(a[0],1):.0f}%" for k, a in sorted(d.items()))
        + f"   (real v_{gears[key]}={realv[key]})")
# by whether F equals the real F
d = {}
for r in rows:
    a = d.setdefault(r[1] >= rr[0], [0, 0])
    a[0] += r[3]; a[1] += r[4]
print("  by F >= real F:", {k: f"{a[1]}/{a[0]}" for k, a in d.items()})
