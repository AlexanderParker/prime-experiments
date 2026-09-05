"""W3 - collect every sep_*.jsonl into the tables the branch document carries."""
import collections
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

recs = []
for p in glob.glob(os.path.join(OUT, "sep_*.jsonl")):
    if "anat" in os.path.basename(p) or "g12" in os.path.basename(p) or "frontier" in os.path.basename(p):
        continue
    for line in open(p):
        recs.append(json.loads(line))

by = collections.defaultdict(list)
for r in recs:
    if r.get("K") is None:
        continue
    by[(r["d"], r["fam"])].append(r)

DS = sorted({d for d, f in by})
print("=" * 100)
print("K(d) BY SEPARATION FAMILY   (E = HiGHS-certified optimal)")
print("=" * 100)
hdr = "%-6s %-5s %-6s" % ("d", "m", "real")
fams = sorted({f for d, f in by if f.startswith("coh:")})
for f in fams:
    hdr += " %-9s" % f
hdr += "  %-28s %s" % ("random (n, min/med/max)", "percentile of real")
print(hdr)
for d in DS:
    if (d, "real") not in by:
        continue
    real = by[(d, "real")][0]
    row = "%-6d %-5d %-6s" % (d, real["m"], "%d%s" % (real["K"], "E" if real["exact"] else "u"))
    for f in fams:
        v = by.get((d, f))
        row += " %-9s" % (("%d%s" % (v[0]["K"], "E" if v[0]["exact"] else "u")) if v else "-")
    R = by.get((d, "rand"), [])
    if R:
        Ks = sorted(r["K"] for r in R)
        n = len(Ks)
        below = sum(1 for k in Ks if k < real["K"])
        eq = sum(1 for k in Ks if k == real["K"])
        cnt = dict(sorted(collections.Counter(Ks).items()))
        row += "  n=%-3d %-22s below=%.3f mid=%.3f" % (
            n, "%d/%.1f/%d %s" % (min(Ks), statistics.median(Ks), max(Ks), cnt),
            below / n, (below + eq / 2) / n)
    print(row)

print()
print("=" * 100)
print("BUDGET  sum|S_j| / m  AT THE OPTIMUM")
print("=" * 100)
print("%-6s %-8s %-10s %-10s %s" % ("d", "real", "rand med", "rand range", "coherent"))
for d in DS:
    if (d, "real") not in by:
        continue
    real = by[(d, "real")][0]
    R = [r["budget_m"] for r in by.get((d, "rand"), [])]
    C = ["%s %.3f" % (f.split(":")[1], by[(d, f)][0]["budget_m"]) for f in fams if (d, f) in by]
    print("%-6d %-8.3f %-10s %-10s %s"
          % (d, real["budget_m"],
             "%.3f" % statistics.median(R) if R else "-",
             "%.3f-%.3f" % (min(R), max(R)) if R else "-", "  ".join(C)))

print()
print("FREE separation rows:")
for d in DS:
    if (d, "free") in by:
        r = by[(d, "free")][0]
        print("  d=%-6d K_free=%-3d %s budget/m=%.3f"
              % (d, r["K"], "EXACT" if r["exact"] else "UB(lb %d)" % r["lb"], r["budget_m"]))

print()
print("REAL optimal gear sets:")
for d in DS:
    if (d, "real") in by:
        print("  d=%-6d %s" % (d, by[(d, "real")][0]["gears"]))
