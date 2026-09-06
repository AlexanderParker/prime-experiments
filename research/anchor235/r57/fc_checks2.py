"""fc_checks2.py -- the frontier split by order: the J = 2 part against F_2(M) (the pair
statement) and the J >= 3 part against max_{J>=3} Q*_J (the chain statement), per rung.
Also the corrected reach statement (why the neighbour law N(v) <= F_2 does NOT bound Rest).
"""
import os
from fc_analyse import load, legal, F2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# recorded values, research/proof/neighbour_profile.md 2.3 (cited, not re-derived)
QSTAR3 = {11: 8, 13: 18, 17: 25, 19: 33, 23: 43, 29: 58}     # by the OLD machine's F? keyed by q'
QSTAR_GE3 = {13: 18, 17: 25, 19: 34, 23: 43, 29: 58}          # max_{J>=3} Q*_J at rungs 11->13 ..


def main():
    rungs = load()
    order = [7, 11, 13, 17, 19, 23, 29, 31]
    L = []
    W = L.append
    W("=== the frontier split by fusion order ===")
    W("rung q' | F_old | F | max_a (a + Rest_2(a)) | F_2(M) | equal? | "
      "max_a (a + Rest_{J>=3}(a)) | at a | F = max of the two?")
    ok2 = 0
    for q in order:
        r = rungs[q]
        Fo, F = r["Fold"], r["F"]
        p2 = max([(a + r["R2"][a], a) for a in sorted(r["R2"]) if r["R2"].get(a, -1) >= 0],
                 default=(0, 0))
        deep = []
        for a in sorted(r["Rest"]):
            v = max(r["R3"].get(a, -1), r.get("R4", {}).get(a, -1))
            if v >= 0:
                deep.append((a + v, a))
        p3 = max(deep, default=(0, 0))
        f2 = F2[Fo]
        good = p2[0] == f2
        ok2 += good
        W(f"{q} | {Fo} | {F} | {p2[0]} (at a = {p2[1]}) | {f2} | {'yes' if good else 'NO'} | "
          f"{p3[0]} | {p3[1]} | {'yes' if max(p2[0], p3[0]) == F else 'NO'}")
    W(f"  max_a (a + Rest_2(a)) = F_2(M) at {ok2} of {len(order)} rungs")
    W("")
    W("The split is the attainment identity (docs/proofs/08) in the frontier's coordinates:")
    W("the J = 2 half of the frontier is the PAIR statement (node 1), the J >= 3 half is the")
    W("CHAIN statement (node 2).  Cited, not re-derived.")
    W("")
    W("=== where the record's obligation lives, per rung ===")
    W("rung | record F | attained by J = 2 (a + Rest_2 = F)? | by J >= 3? | a")
    for q in order:
        r = rungs[q]
        F = r["F"]
        by2 = [a for a in sorted(r["R2"]) if r["R2"].get(a, -1) >= 0 and a + r["R2"][a] == F]
        by3 = [a for a in sorted(r["Rest"])
               if max(r["R3"].get(a, -1), r.get("R4", {}).get(a, -1)) >= 0
               and a + max(r["R3"].get(a, -1), r.get("R4", {}).get(a, -1)) == F]
        W(f"{q} | {F} | {by2 if by2 else 'no'} | {by3 if by3 else 'no'}")
    txt = "\n".join(L)
    open(os.path.join(OUT, "fc_checks2.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
