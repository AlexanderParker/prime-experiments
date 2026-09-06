"""ag_band.py -- the gate's consequences: the per-J gate ladder, the three regimes, the residual
band, and the slack profile on it.

Consumes  results/ag_gate.json      (this round: has, hasM, has2, legality, a_gate, F_2)
and       ../r57/results/fc_frontier.json, ../r57/results/fc_top31.json
          (branch 4.i.a's exact frontier Rest(a), Rest_2, Rest_3, Rest_{>=4}, slack s(a))
and       ../r57/results/N_29.npy, n1_29.npy   (m29 neighbour profile, for the rung-31 old machine)

The per-J gate ladder (proved from file 05 T2, verified here):
  J = 2      no condition
  J = 3      a is legal (a is the middle) OR hasM(a) > 0 (a is an end)
  J >= 4     hasM(a) > 0  (a always has an INTERIOR piece as a run-neighbour)
so above a_hasM only J = 3 with a legal survives, and above a_gate only J = 2 survives.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
R57 = os.path.join(HERE, "..", "r57", "results")

JMAX = {7: 3, 11: 2, 13: 3, 17: 3, 19: 3, 23: 4, 29: 3, 31: 5}   # merge_forest 2.2, = 1 + D_q'
BAREWORD = 6                                                      # the universal cap


def main():
    G = json.load(open(os.path.join(OUT, "ag_gate.json")))
    FR = json.load(open(os.path.join(R57, "fc_frontier.json")))
    T31 = json.load(open(os.path.join(R57, "fc_top31.json")))
    N29 = np.load(os.path.join(R57, "N_29.npy"))
    n129 = np.load(os.path.join(R57, "n1_29.npy"))
    lines = []
    W = lines.append
    W("=== THE GATE'S CONSEQUENCES: the per-J ladder, the three regimes, the residual band ===")
    summary = []
    viol_gate3 = viol_gate4 = cells = 0
    for q in ["7", "11", "13", "17", "19", "23", "29", "31"]:
        g = G[q]
        qq = g["q"]; Fold = g["Fold"]; F2 = g["F2"]; aL = g["aL"]; bL = g["bL"]
        if q == "31":
            Rest = {int(k): v for k, v in T31["Rest"].items() if v >= 0}
            R2 = {int(k): v for k, v in T31["R2"].items()}
            R3 = {int(k): v for k, v in T31["R3"].items()}
            R4 = {int(k): v for k, v in T31["R4"].items()}
            s = {int(k): v for k, v in T31["s"].items()}
            Nn = {a: int(N29[a]) for a in range(N29.size)}
            n1 = {a: int(n129[a]) for a in range(n129.size)}
            F = max(a + Rest[a] for a in Rest)
        else:
            f = FR[q]
            Rest = {int(k): v for k, v in f["Rest"].items() if v >= 0}
            R2 = {int(k): v for k, v in f["R2"].items()}
            R3 = {int(k): v for k, v in f["R3"].items()}
            R4 = {int(k): v for k, v in f["R4"].items()}
            s = {int(k): v for k, v in f["s"].items()}
            Nn = {int(k): v for k, v in f["N"].items()}
            n1 = {int(k): v for k, v in f["n1"].items()}
            F = f["F"]
        gate = {int(k): v for k, v in g["gate_open"].items()}
        legal = {int(k): v for k, v in g["legal_a"].items()}
        hasM = {int(k): v for k, v in g["hasM"].items()}
        a_gate, a_hasM, a_has = g["a_gate"], g["a_hasM"], g["a_has"]
        budget = Fold + qq

        # --- verification of the per-J gate ladder against the exact frontier
        v3 = [a for a in Rest if not gate.get(a, False) and R3.get(a, -1) >= 0]
        v3 += [a for a in Rest if not gate.get(a, False) and R4.get(a, -1) >= 0]
        v4 = [a for a in Rest if hasM.get(a, 0) == 0 and R4.get(a, -1) >= 0]
        viol_gate3 += len(v3); viol_gate4 += len(v4); cells += len(Rest)

        # --- the three regimes
        reg2 = sorted(a for a in Rest if a > a_gate)
        reg3 = sorted(a for a in Rest if a_hasM < a <= a_gate)
        band_all = sorted(a for a in Rest if a <= a_hasM and gate.get(a, False))
        # the deep-chain cap reaches a while J_max * a <= budget; the letter floor closes a < a_L
        cap_meas = budget // JMAX[qq]
        cap_bare = budget // BAREWORD
        resid_meas = [a for a in band_all if a > cap_meas]
        resid_bare = [a for a in band_all if a > cap_bare]
        amin = min(Rest, key=lambda a: (s[a], -a))
        W(f"\n=== rung -> +{qq}: F_old = {Fold}, F = {F}, budget = {budget}, F_2 = {F2}, "
          f"a_L = {aL}, J_max = {JMAX[qq]} ===")
        W(f"  a_gate = {a_gate}, a_hasM = {a_hasM}, a_has = {a_has}; "
          f"deep-chain cap reaches a <= {cap_meas} (measured J_max) / {cap_bare} (bare-word 6)")
        W(f"  gate-ladder violations: J>=3 above the gate: {v3}; J>=4 above a_hasM: {v4}")
        W(f"  regime J<=2 (a > a_gate):  {reg2}")
        W(f"  regime J<=3, a legal middle (a_hasM < a <= a_gate): {reg3}")
        if reg3:
            W("    a | Rest(a) | Rest_3 | N(a) | a+N(a) | budget | a+N(a) <= budget?")
            for a in reg3:
                W(f"    {a} | {Rest[a]} | {R3.get(a,-1)} | {Nn.get(a,-1)} | {a+Nn.get(a,0)} | "
                  f"{budget} | {'yes' if a + Nn.get(a, 0) <= budget else 'NO'}")
        W(f"  residual band (gate open, above the deep-chain cap): "
          f"measured J_max -> {resid_meas[:1]}..{resid_meas[-1:]} "
          f"({len(resid_meas)} realised sizes); bare-word 6 -> {resid_bare[:1]}..{resid_bare[-1:]} "
          f"({len(resid_bare)} realised sizes)")
        if resid_meas:
            W("    slack on the band: " + "  ".join(f"{a}:{s[a]}" for a in resid_meas))
            W(f"    min slack on the band = {min(s[a] for a in resid_meas)} "
              f"(global budget slack {min(s.values())} at a = {amin}); "
              f"a* in the band? {'YES' if amin in resid_meas else 'no'}")
        # what the two-piece regime needs
        top2 = max((a + R2[a] for a in Rest if R2.get(a, -1) >= 0), default=-1)
        W(f"  max_a (a + Rest_2(a)) = {top2} vs F_2 = {F2}; pair-statement slack "
          f"F_old + q' - F_2 = {budget - F2}")
        summary.append(dict(q=qq, Fold=Fold, F=F, F2=F2, aL=aL, a_gate=a_gate, a_hasM=a_hasM,
                            a_has=a_has, cap_meas=cap_meas, cap_bare=cap_bare,
                            band_meas=[resid_meas[0], resid_meas[-1]] if resid_meas else [],
                            nband_meas=len(resid_meas),
                            band_bare=[resid_bare[0], resid_bare[-1]] if resid_bare else [],
                            nband_bare=len(resid_bare),
                            minslack_band=(min(s[a] for a in resid_meas) if resid_meas else None),
                            amin=amin, mins=min(s.values()), astar_in=amin in resid_meas,
                            reg3=reg3, top2=top2))
    W("\n=== SUMMARY ===")
    W("q' | F_old | F | F_2 | a_L | a_gate | a_gate/F_old | a_hasM | a_has | F_2-a_L | "
      "deep cap (Jmax/6) | band (Jmax) | width | band (6) | width | min slack on band | a* in band")
    for r in summary:
        bm = f"[{r['band_meas'][0]},{r['band_meas'][1]}]" if r["band_meas"] else "empty"
        bb = f"[{r['band_bare'][0]},{r['band_bare'][1]}]" if r["band_bare"] else "empty"
        W(f"{r['q']} | {r['Fold']} | {r['F']} | {r['F2']} | {r['aL']} | {r['a_gate']} | "
          f"{r['a_gate']/r['Fold']:.3f} | {r['a_hasM']} | {r['a_has']} | {r['F2']-r['aL']} | "
          f"{r['cap_meas']}/{r['cap_bare']} | {bm} | {r['nband_meas']} | {bb} | {r['nband_bare']} | "
          f"{r['minslack_band']} | {r['astar_in']}")
    W(f"\ngate-ladder verification over {cells} (rung, a) cells: "
      f"{viol_gate3} violations of 'gate closed => no J >= 3', "
      f"{viol_gate4} violations of 'hasM = 0 => no J >= 4'")
    json.dump(summary, open(os.path.join(OUT, "ag_band.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "ag_band.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
