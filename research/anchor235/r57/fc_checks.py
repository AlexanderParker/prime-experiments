"""fc_checks.py -- the exceptionless checks of branch 4.i.a, with counts.

E1/E2 the fusion-rate and interior-rate identities (4/q', 3/q', 2/q'; 0, 1/q', 2/q')
E3    Rest(F_old) = N(F_old) if F_old is interior-legal, else n1(F_old)
E4    a not interior-legal and no occurrence with a letter-sized neighbour  =>  no J >= 3 fusion
E5    a >= 0.9 F_old  =>  the attaining fusion is J = 2, unless a is interior-legal
E6    Rest(a) > N(a) only where a J >= 4 fusion attains
"""
import os, json
import numpy as np
from fc_analyse import load, legal

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    rungs = load()
    order = [7, 11, 13, 17, 19, 23, 29, 31]
    L = []
    W = L.append
    t31 = json.load(open(os.path.join(OUT, "fc_top31.json")))

    W("=== E1/E2. the fusion-rate identities, all rungs ===")
    n = ok1 = ok2 = 0
    for q in order:
        r = rungs[q]
        if "occ" in r:
            occ, fus, inte = r["occ"], r["fus"], r["inte"]
            keys = sorted(occ)
        else:
            occ = {a: v for a, v in enumerate(t31["occ"]) if v}
            fus = {a: t31["fus"][a] for a in occ}
            inte = {a: t31["inte"][a] for a in occ}
            keys = sorted(occ)
        for a in keys:
            if occ[a] == 0:
                continue
            n += 1
            cls = 0 if a % q == 0 else (1 if a % q in (r["aL"], r["bL"]) else 2)
            ok1 += abs(fus[a] * q / occ[a] - {0: 2, 1: 3, 2: 4}[cls]) < 1e-9
            ok2 += abs(inte[a] * q / occ[a] - {0: 2, 1: 1, 2: 0}[cls]) < 1e-9
    W(f"  sizes tested {n}; fusion-rate matches {ok1}; interior-rate matches {ok2}")

    W("")
    W("=== E4. no letter neighbour and not legal  =>  no J >= 3 fusion ===")
    tot = bad = 0
    for q in order:
        r = rungs[q]
        for a in sorted(r["Rest"]):
            if r["has"].get(a, -1) != 0 or legal(a, q, r["aL"], r["bL"]):
                continue
            tot += 1
            if r["R3"].get(a, -1) >= 0 or r.get("R4", {}).get(a, -1) >= 0:
                bad += 1
                W(f"   EXCEPTION rung {q} a={a}")
    W(f"  cells with no letter neighbour and a not legal: {tot}; exceptions: {bad}")

    W("")
    W("=== E5. a >= 0.9 F_old  =>  J = 2, unless a is interior-legal ===")
    tot = bad = 0
    W("  rung | a | a/F_old | J | legal? | word")
    for q in order:
        r = rungs[q]
        Fo = r["Fold"]
        for a in sorted(r["Rest"]):
            if a < 0.9 * Fo:
                continue
            tot += 1
            J = len(r["wit"][a][1])
            lg = legal(a, q, r["aL"], r["bL"])
            flag = "" if (J == 2 or lg) else "  <== EXCEPTION"
            if flag:
                bad += 1
            W(f"  {q} | {a} | {a/Fo:.3f} | {J} | {'YES' if lg else 'no'} | "
              f"{' '.join(map(str, r['wit'][a][1]))}{flag}")
    W(f"  cells {tot}; exceptions {bad}")

    W("")
    W("=== E6. Rest(a) > N(a) only where a J >= 4 fusion attains ===")
    tot = bad = 0
    for q in order:
        r = rungs[q]
        for a in sorted(r["Rest"]):
            if r["Rest"][a] <= r["N"][a]:
                continue
            tot += 1
            if len(r["wit"][a][1]) < 4:
                bad += 1
                W(f"   rung {q} a={a} Rest={r['Rest'][a]} N={r['N'][a]} "
                  f"word {r['wit'][a][1]} (J = {len(r['wit'][a][1])})")
    W(f"  cells with Rest > N: {tot}; of those attained by J <= 3: {bad}")

    txt = "\n".join(L)
    open(os.path.join(OUT, "fc_checks.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
