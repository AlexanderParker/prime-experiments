"""ag_family.py -- the availability gate on the tooth-counterfactual family, on the recorded
budget violators, and the level-2 dictionary characterisation at the real rungs.

Family member = same gears, teeth at +-v_g with v_g uniform in 1..(g-1)/2 (alignment-rules
section 5), the SAME 20 members as r57/fc_family.py (same rng seed), so the two tables compare
member by member.

Part A: gate quantities (a_gate, a_hasM, a_gate2, holes) for 20 members + the real machine at
        13->17, 17->19, 19->23.
Part B: the five recorded budget violators -- is the violating `a` inside the band (gate open)?
Part C: the dictionary at the real rungs: which legal v occur adjacent to a_has, and whether the
        gate-open set is an initial segment of [a_L, a_hasM].

Writes results/ag_family.txt.
"""
import os, sys, gc, json, random, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
sys.path.insert(0, HERE)
from mf_core import build_levels, u_of            # noqa: E402
from ag_gate import gate_stats, letters           # noqa: E402

OUT = os.path.join(HERE, "results")


def member_gate(gears, vs):
    """gate quantities of M = gears[:-1] against the incoming q' = gears[-1] with tooth vs[-1]."""
    L = build_levels(gears[:-1], vs[:-1])
    old = L[-1]
    q = gears[-1]
    aL, bL = letters(q, vs[-1])
    st = gate_stats(old.size, q, aL, bL)
    m = st["m"]
    realised = np.flatnonzero(m).tolist()
    D = st["D"]
    F2 = max((a + int(np.flatnonzero(D[a]).max()) for a in realised if D[a].any()), default=0)
    legal_a = lambda a: (a % q in (0, aL, bL)) and a >= aL
    gopen = {a: (legal_a(a) or st["hasM"][a] > 0) for a in realised}
    a_gate = max([a for a in realised if gopen[a]], default=0)
    a_hasM = max([a for a in realised if st["hasM"][a] > 0], default=0)
    a_gate2 = max([a for a in realised if st["has2"][a] > 0], default=0)
    holes = [a for a in realised if aL <= a < a_hasM and st["hasM"][a] == 0]
    out = dict(q=q, Fold=int(old.F), F2=int(F2), aL=int(aL), bL=int(bL), a_gate=int(a_gate),
               a_hasM=int(a_hasM), a_gate2=int(a_gate2), holes=holes,
               gopen={a: bool(gopen[a]) for a in realised},
               hasM={a: int(st["hasM"][a]) for a in realised},
               has={a: int(st["has"][a]) for a in realised},
               legal={a: bool(legal_a(a)) for a in realised})
    del L, old, st
    gc.collect()
    return out


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== THE AVAILABILITY GATE ON THE FAMILY, THE VIOLATORS, AND THE DICTIONARY ===")
    rng = random.Random(20260906)          # the same 20 members as r57/fc_family.py
    setups = [([5, 7, 11, 13, 17], "13->17"), ([5, 7, 11, 13, 17, 19], "17->19"),
              ([5, 7, 11, 13, 17, 19, 23], "19->23")]
    fam = {}
    for gears, name in setups:
        real = [min(u_of(g), g - u_of(g)) for g in gears]
        W(f"\n--- rung {name} (gears {gears}); real teeth v = {real} ---")
        W("member | v | F_old | F_2 | a_L | a_gate | a_gate/F_old | a_hasM | a_gate2 | "
          "F_2 - a_L | a_hasM <= F_2 - a_L? | gate never closes (a_gate = F_old)? | holes")
        vsets = [("REAL", real)]
        seen = set()
        while len(vsets) < 21:
            vs = [rng.randrange(1, (g - 1) // 2 + 1) for g in gears]
            if tuple(vs) in seen or vs == real:
                continue
            seen.add(tuple(vs))
            vsets.append((f"m{len(vsets)}", vs))
        rows = []
        for tag, vs in vsets:
            o = member_gate(gears, vs)
            never = o["a_gate"] == o["Fold"]
            capok = o["a_hasM"] <= o["F2"] - o["aL"]
            W(f"{tag} | {vs} | {o['Fold']} | {o['F2']} | {o['aL']} | {o['a_gate']} | "
              f"{o['a_gate']/o['Fold']:.3f} | {o['a_hasM']} | {o['a_gate2']} | "
              f"{o['F2']-o['aL']} | {'yes' if capok else 'NO'} | "
              f"{'YES' if never else 'no'} | {o['holes']}")
            rows.append((tag, o, never, capok))
        fr = sorted(o["a_gate"] / o["Fold"] for tag, o, _, _ in rows if tag != "REAL")
        realfrac = [o["a_gate"] / o["Fold"] for tag, o, _, _ in rows if tag == "REAL"][0]
        med = fr[len(fr) // 2]
        nnever = sum(1 for tag, o, nv, _ in rows if tag != "REAL" and nv)
        ncap = sum(1 for tag, o, _, ck in rows if tag != "REAL" and ck)
        W(f"  family a_gate/F_old: min {fr[0]:.3f} median {med:.3f} max {fr[-1]:.3f}; "
          f"REAL {realfrac:.3f} -> {'at or below' if realfrac <= med else 'ABOVE'} the median; "
          f"members below the real value: {sum(1 for x in fr if x < realfrac)} of 20")
        W(f"  members whose gate never closes (a_gate = F_old): {nnever} of 20 "
          f"(real: {'YES' if rows[0][2] else 'no'})")
        W(f"  members obeying the closed form a_hasM <= F_2 - a_L: {ncap} of 20 "
          f"(real: {'yes' if rows[0][3] else 'NO'})")
        fam[name] = dict(real=realfrac, median=med, min=fr[0], max=fr[-1], nnever=nnever,
                         ncap=ncap, real_never=rows[0][2])
        W(f"  [{time.time()-t0:.1f}s]")

    W("\n--- the five recorded budget violators: is the violating `a` inside the band? ---")
    VIOL = [([5, 7, 11, 13, 17, 19], [1, 3, 4, 4, 4, 4], 19, "m17 (1,3,4,4,4) + v_19 = 4"),
            ([5, 7, 11, 13, 17, 19], [2, 3, 3, 3, 3, 3], 13, "m17 (2,3,3,3,3) + v_19 = 3"),
            ([5, 7, 11, 13], [1, 1, 5, 1], 11, "m11 (1,1,5) + v_13 = 1"),
            ([5, 7, 11, 13, 17, 19], [1, 3, 2, 1, 6, 3], 13, "m17 (1,3,2,1,6) + v_19 = 3"),
            ([5, 7, 11, 13, 17, 19], [1, 2, 1, 4, 4, 5], 10, "m17 (1,2,1,4,4) + v_19 = 5")]
    W("violator | F_old | q' | a_L | violating a | a/F_old | a legal? | hasM(a) | gate at a | "
      "a_gate | a <= a_gate?")
    nin = 0
    for gears, vs, av, tag in VIOL:
        o = member_gate(gears, vs)
        g = o["gopen"].get(av, False)
        nin += bool(g)
        W(f"{tag} | {o['Fold']} | {o['q']} | {o['aL']} | {av} | {av/o['Fold']:.3f} | "
          f"{'YES' if o['legal'].get(av) else 'no'} | {o['hasM'].get(av, 0)} | "
          f"{'OPEN' if g else 'CLOSED'} | {o['a_gate']} | {'yes' if av <= o['a_gate'] else 'NO'}")
    W(f"  violators breaking at an `a` with the gate OPEN: {nin} of 5")

    W("\n--- the dictionary at the real rungs: which legal v occurs at the top of the gate ---")
    G = json.load(open(os.path.join(OUT, "ag_gate.json")))
    W("q' | a_L | a_hasM | max legal nbr at a_hasM | min legal nbr at a_hasM | "
      "holes in [a_L, a_hasM] | initial segment?")
    for q in ["7", "11", "13", "17", "19", "23", "29", "31"]:
        g = G[q]
        aL = g["aL"]; ah = g["a_hasM"]
        hm = {int(k): v for k, v in g["hasM"].items()}
        mx = {int(k): v for k, v in g["maxlegalM"].items()}
        mn = {int(k): v for k, v in g["minlegal"].items()}
        holes = [a for a in sorted(hm) if aL <= a < ah and hm[a] == 0]
        W(f"{q} | {aL} | {ah} | {mx.get(ah, 0)} | {mn.get(ah, 0)} | {holes} | "
          f"{'yes' if not holes else 'NO'}")
    W(f"\n[total {time.time()-t0:.1f}s]")
    json.dump(fam, open(os.path.join(OUT, "ag_family.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "ag_family.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
