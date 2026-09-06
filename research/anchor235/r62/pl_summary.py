"""pl_summary.py -- the cross-rung tables the branch document needs:

(1) the L4-analogue census: is every gear busy (a sole striker) inside the attaining 2-run?
(2) item 4(a): the availability gate's band once the pinned letter closes it at F + 3 - a_L,
    against the deep-chain cap floor((F + q')/J_max) recorded in availability_gate.md 2.5;
(3) item 4(b): the twin rungs;
(4) the excess table E(a_L) with the coupling status, and the family summary.

Reads results/pl_profile.json, pl_spare.json, pl_depth.json, pl_family.json.
Outputs results/pl_summary.txt.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

prof = json.load(open(os.path.join(OUT, "pl_profile.json")))
spare = json.load(open(os.path.join(OUT, "pl_spare.json")))
depth = json.load(open(os.path.join(OUT, "pl_depth.json")))
fam = json.load(open(os.path.join(OUT, "pl_family.json")))

# availability_gate.md 2.5, cited: J_max (measured) and the deep-chain cap per rung
JMAX = {7: 3, 11: 2, 13: 3, 17: 3, 19: 3, 23: 4, 29: 3, 31: 5}
DEEP = {7: 3, 11: 8, 13: 6, 17: 9, 19: 12, 23: 12, 29: 21, 31: 14}
AHASM = {7: 2, 11: 0, 13: 0, 17: 7, 19: 12, 23: 20, 29: 25, 31: 35}

lines = []
W = lines.append

W("=== (1) the L4-analogue: every gear is a sole striker inside the attaining 2-run ===")
W("machine | sizes | sizes where EVERY gear is busy | sizes with a free gear | gears")
tot = totall = 0
for k, e in spare.items():
    rows = e["rows"]
    allbusy = sum(1 for x in rows if len(x["busy"]) == len(e["gears"]))
    W(f"m{e['gears'][-1]} | {len(rows)} | {allbusy} | "
      f"{sum(1 for x in rows if x['free'] > 0)} | {e['gears']}")
    tot += allbusy
    totall += len(rows)
W(f"  TOTAL: every gear busy at {tot} of {totall} attaining 2-runs; a free gear at 0")

W("\n=== (2) item 4(a): the gate's top under the pinned letter, and the residual band ===")
W("rung q' | F | a_L | pinned top F+3-a_L | a_hasM (measured) | deep cap | "
  "band [deep+1, top] | realised sizes in the band | F - q'/3")
for k, e in prof.items():
    q = int(k)
    if q not in JMAX:
        continue
    F, aL = e["F"], e["aL"]
    top = F + 3 - aL
    lo = DEEP[q] + 1
    rs = [x["v"] for x in e["rows"] if lo <= x["v"] <= top]
    W(f"{q} | {F} | {aL} | {top} | {AHASM[q]} | {DEEP[q]} | [{lo}, {top}] | "
      f"{len(rs)} | {F - q/3:.1f}")

W("\n=== (3) item 4(b): the twin rungs (q' - 2 a gear of M) ===")
W("rung q' | twin? | a_L | Leg(a_L) n M | Pad(a_L) n M | E(a_L) | depth of the glue | "
  "gears moved")
for k, e in prof.items():
    q = int(k)
    aL = e["aL"]
    row = next((x for x in e["rows"] if x["v"] == aL), None)
    if row is None:
        W(f"{q} | - | {aL} | a_L unrealised")
        continue
    twin = (q - 2) in e["gears"]
    d = depth.get(k)
    dr = next((x for x in d["rows"] if x["v"] == aL), None) if d else None
    W(f"{q} | {'TWIN' if twin else 'no'} | {aL} | {row['leg'] or '-'} | {row['pad'] or '-'} | "
      f"{row['E']:+d} | {dr['depth'] if dr else '-'} | "
      f"{dr['target'] if dr else '-'}")

W("\n=== (4) the excess of the letter, with everything about it ===")
W("rung q' | F | F_2 | a_L | r(a_L) | E(a_L) | max E over sizes | #E>3 | #sizes | "
  "U_leg | U_full | U_pad")
for k, e in prof.items():
    aL = e["aL"]
    row = next((x for x in e["rows"] if x["v"] == aL), None)
    mx = max(x["E"] for x in e["rows"])
    if row is None:
        W(f"{k} | {e['F']} | {e['F2']} | {aL} | unrealised | - | {mx} | "
          f"{sum(1 for x in e['rows'] if x['E'] > 3)} | {len(e['rows'])} | - | - | -")
        continue
    W(f"{k} | {e['F']} | {e['F2']} | {aL} | {row['r']} | {row['E']:+d} | {mx} | "
      f"{sum(1 for x in e['rows'] if x['E'] > 3)} | {len(e['rows'])} | "
      f"{row['u_leg']} | {row['u_full']} | {row['u_pad']}")

W("\n=== (5) the family, summarised ===")
allE = [x["E"] for rs in fam.values() for x in rs if x["E"] is not None]
W(f"  members with a_L realised: {len(allE)}; 0 <= E <= 3 at "
  f"{sum(1 for e in allE if 0 <= e <= 3)}; E <= 3 at {sum(1 for e in allE if e <= 3)}; "
  f"E >= 0 at {sum(1 for e in allE if e >= 0)}; range {min(allE):+d} .. {max(allE):+d}")
for name, rs in fam.items():
    ok = [x for x in rs if x["tag"] != "REAL" and x["E"] is not None and 0 <= x["E"] <= 3]
    up = [x for x in rs if x["tag"] != "REAL" and x["E"] is not None and x["E"] <= 3]
    W(f"  rung {name}: 0<=E<=3 at {len(ok)} of 20; E<=3 at {len(up)} of 20; "
      f"REAL E = {rs[0]['E']:+d}")
v5 = [x for rs in fam.values() for x in rs if x["E"] is not None and x["vs"][0] == 1]
v5b = [x for rs in fam.values() for x in rs if x["E"] is not None and x["vs"][0] != 1]
W(f"  members with gear 5 at its real tooth (v_5 = 1): {len(v5)}, E <= 3 at "
  f"{sum(1 for x in v5 if x['E'] <= 3)} ({sum(1 for x in v5 if x['E'] <= 3)/len(v5):.2f}); "
  f"with v_5 = 2: {len(v5b)}, E <= 3 at {sum(1 for x in v5b if x['E'] <= 3)} "
  f"({sum(1 for x in v5b if x['E'] <= 3)/len(v5b):.2f})")

txt = "\n".join(lines)
open(os.path.join(OUT, "pl_summary.txt"), "w").write(txt)
print(txt)
