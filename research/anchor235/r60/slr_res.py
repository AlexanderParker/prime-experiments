"""slr_res.py -- how much of the short-letter row's deficit is residue-explained, and the band.

Consumes results/slr_row.json and results/slr_m31.json.

(1) The gear-5 pair filter as a BOUND on the row:
        r5(v) := max { a realised, a <= min(F, F_2 - v), (a mod 5, v mod 5) achievable mod 5 }
    r(v) <= r5(v) always (proved: the three openings must be open mod 5).  The deficit splits into
    the residue-explained part min(F, F_2-v) - r5(v) and the unexplained part r5(v) - r(v).

(2) The two-gap gear-5 cost C_5(a, v) = |T_5 u (T_5 - a) u (T_5 - a - v)| at the extremal cell,
    against its minimum over the admissible a.

(3) The law G(a_L) := a_L + r(a_L) <= F(M) + 3, across every rung, and G(v) for all v for contrast.

(4) The residual band at rung 29->31 under the proved cap, the measured row, and the new law.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

T5 = {1, 4}
O5 = [x for x in range(5) if x not in T5]
ACH = set(((o1 - o0) % 5, (o2 - o1) % 5) for o0 in O5 for o1 in O5 for o2 in O5)


def C5(a, v):
    return len(T5 | {(t - a) % 5 for t in T5} | {(t - a - v) % 5 for t in T5})


def main():
    row = json.load(open(os.path.join(OUT, "slr_row.json")))
    m31 = json.load(open(os.path.join(OUT, "slr_m31.json")))
    ent = []
    for k in row:
        e = row[k]
        ent.append((int(k), e["gears"][-1], e["F"], e["F2"], e["aL"], e["bL"],
                    {int(a): b for a, b in e["r"].items()}, set(e["realised"])))
    ent.append((37, 31, m31["F"], m31["F2"], 12, 25,
                {int(a): b for a, b in m31["r"].items()}, set(m31["realised"])))
    lines = []
    W = lines.append
    W("=== (1) THE GEAR-5 PAIR FILTER AS A BOUND ON THE SHORT-LETTER ROW ===")
    W("rung q' | M | F | F_2 | a_L | cap=min(F,F_2-a_L) | r5(a_L) | r(a_L) | residue-explained | "
      "unexplained | a_L mod 5 | forbidden a-classes")
    for qn, top, F, F2, aL, bL, r, realised in ent:
        if aL not in r:
            W(f"{qn} | {{5..{top}}} | {F} | {F2} | {aL} | - | - | 0 | - | - | {aL%5} | -")
            continue
        cap = min(F, F2 - aL)
        r5 = max([a for a in sorted(realised) if a <= cap and (a % 5, aL % 5) in ACH], default=0)
        bad = [c for c in range(5) if (c, aL % 5) not in ACH]
        W(f"{qn} | {{5..{top}}} | {F} | {F2} | {aL} | {cap} | {r5} | {r[aL]} | "
          f"{cap - r5} | {r5 - r[aL]} | {aL%5} | {bad}")
    W("\n=== (2) THE GEAR-5 COST OF THE EXTREMAL CELL ===")
    W("rung q' | a_L | r(a_L) | C_5(r(a_L), a_L) | min C_5 over admissible a | attains min? | "
      "r(a_L) mod 5")
    for qn, top, F, F2, aL, bL, r, realised in ent:
        if aL not in r or r[aL] == 0:
            continue
        cap = min(F, F2 - aL)
        adm = [a for a in sorted(realised) if a <= cap and (a % 5, aL % 5) in ACH]
        mn = min(C5(a, aL) for a in adm)
        W(f"{qn} | {aL} | {r[aL]} | {C5(r[aL], aL)} | {mn} | "
          f"{'YES' if C5(r[aL], aL) == mn else 'no'} | {r[aL] % 5}")
    W("\n=== (3) G(v) = v + r(v): the largest 2-run through a v-gap ===")
    W("rung q' | M | F | F_2 | a_L | G(a_L) | G(a_L) - F | G(b_L) - F | max_v G(v) | "
      "#v with G(v) > F+3")
    for qn, top, F, F2, aL, bL, r, realised in ent:
        if aL not in r:
            W(f"{qn} | {{5..{top}}} | {F} | {F2} | {aL} | - | - | - | - | -")
            continue
        G = {v: v + r[v] for v in r}
        over = [v for v in G if G[v] > F + 3]
        W(f"{qn} | {{5..{top}}} | {F} | {F2} | {aL} | {G[aL]} | {G[aL]-F} | "
          f"{(G[bL]-F) if bL in G else '-'} | {max(G.values())} | {len(over)} of {len(G)}")
    W("\n=== (3b) IS G(a_L) SPECIAL?  the two-sided pin G(a_L) in [F, F+3] and its base rate ===")
    W("rung q' | F | F_2 | G(a_L)-F | rank of G(a_L) among realised v (1 = smallest) | "
      "percentile | #v with G(v) in [F, F+3] | #realised | base rate | G(b_L)-F | G(q')-F")
    for qn, top, F, F2, aL, bL, r, realised in ent:
        if aL not in r:
            continue
        G = {v: v + r[v] for v in r}
        vals = sorted(G.values())
        rank = sum(1 for x in vals if x < G[aL]) + 1
        inwin = [v for v in G if F <= G[v] <= F + 3]
        W(f"{qn} | {F} | {F2} | {G[aL]-F} | {rank} of {len(vals)} | "
          f"{100.0*rank/len(vals):.0f}% | {len(inwin)} | {len(G)} | "
          f"{100.0*len(inwin)/len(G):.0f}% | {(G[bL]-F) if bL in G else '-'} | "
          f"{(G[qn]-F) if qn in G else '-'}")

    W("\n=== (4) THE RESIDUAL BAND AT RUNG 29->31 (M = {5..29}, F = 43, q' = 31, budget 74) ===")
    e = [x for x in ent if x[0] == 31][0]
    _, _, F, F2, aL, bL, r, realised = e
    for Jmax, tag in ((5, "measured J_max = 5 (merge_forest 2.2)"),
                      (6, "the universal bare-word cap J_max = 6")):
        lowcap = (F + 31) // Jmax
        for name, top_ in (("proved pair cap F_2 - a_L", F2 - aL),
                           ("F(M) (the cap is vacuous)", F),
                           ("the new law F - a_L + 3", F - aL + 3),
                           ("the measured row r(a_L)", r[aL])):
            hi = min(top_, F)
            n = len([v for v in realised if lowcap < v <= hi])
            W(f"  {tag}: deep-chain cap reaches a <= {lowcap}; gate top by {name} = {top_} "
              f"-> band [{lowcap+1}, {hi}], {n} realised sizes")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "slr_res.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
