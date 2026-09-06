"""fc_family.py -- the frontier and its slack profile on the tooth-counterfactual family.

Family member = same gears, teeth at +-v_g with v_g uniform in 1..(g-1)/2 (alignment-rules
section 5).  For each member the last gear is the incoming q': M = gears[:-1], q' = gears[-1].
Measured per member: F_old, F, the budget slack min s, the minimiser a*/F_old, the number of
strict local minima of s on a >= a_L, convexity, whether the budget is violated, and if so at
which a the slack goes negative (interior or top).

Part A: 20 members plus the real machine at 13->17, 17->19, 19->23 (shape and minimiser).
Part B: 300 members at 13->17 and 17->19 (violator hunt).

Writes results/fc_family.txt.
"""
import os, sys, gc, random, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r56"))
from mf_core import build_levels, u_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def frontier(gears, vs):
    L = build_levels(gears, vs)
    lv, old = L[-1], L[-2]
    q, Fold = lv.q, old.F
    v = vs[-1] % q
    d = (2 * v) % q
    aL = min(d, q - d) if d else 0
    bL = q - aL if d else 0
    Jm = int(lv.order.max())
    mx = np.zeros(lv.N, dtype=np.int64)
    for k in range(Jm):
        idx = np.flatnonzero(lv.order > k)
        mx[idx] = np.maximum(mx[idx], old.size[(lv.newpos[idx] + k) % old.N])
    rest = lv.size - mx
    A = int(mx.max())
    Rest = -np.ones(A + 1, dtype=np.int64)
    np.maximum.at(Rest, mx, rest)
    realised = [a for a in range(A + 1) if Rest[a] >= 0]
    s = {a: int(Fold + q - a - Rest[a]) for a in realised}
    F = int(lv.F)
    out = dict(q=q, Fold=Fold, F=F, aL=int(aL), bL=int(bL), realised=realised, s=s,
               Rest={a: int(Rest[a]) for a in realised},
               budget=int(Fold + q - F), viol=[a for a in realised if s[a] < 0])
    del L, lv, old, mx, rest
    gc.collect()
    return out


def shape(o):
    """minimiser, #strict local minima on a >= a_L, convexity, and the shape summary."""
    ra = [a for a in o["realised"] if a >= max(1, o["aL"])]
    if len(ra) < 3:
        return None
    sv = [o["s"][a] for a in ra]
    amin = ra[int(np.argmin(sv))]
    loc = 0
    for i in range(1, len(sv) - 1):
        if sv[i] < sv[i - 1] and sv[i] < sv[i + 1]:
            loc += 1
    d2 = [sv[i + 1] - 2 * sv[i] + sv[i - 1] for i in range(1, len(sv) - 1)]
    convex = all(x >= 0 for x in d2)
    return dict(amin=amin, frac=amin / o["Fold"], nloc=loc, convex=convex,
                negd2=sum(1 for x in d2 if x < 0), ncells=len(sv))


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    rng = random.Random(20260906)
    W("=== the frontier's slack profile on the tooth-counterfactual family ===")
    W("s(a) = F_old + q' - a - Rest(a); budget violated iff some s(a) < 0.")
    setups = [([5, 7, 11, 13, 17], "13->17"), ([5, 7, 11, 13, 17, 19], "17->19"),
              ([5, 7, 11, 13, 17, 19, 23], "19->23")]
    for gears, name in setups:
        real = [min(u_of(g), g - u_of(g)) for g in gears]
        W(f"\n--- rung {name} (gears {gears}); real teeth v = {real} ---")
        W("member | v | F_old | F | q' | budget slack | a* | a*/F_old | in [0.55,0.75]? | "
          "#strict local minima | convex? | #negative 2nd differences | violates?")
        rows = []
        seen = set()
        vsets = [("REAL", real)]
        while len(vsets) < 21:
            vs = [rng.randrange(1, (g - 1) // 2 + 1) for g in gears]
            if tuple(vs) in seen or vs == real:
                continue
            seen.add(tuple(vs))
            vsets.append((f"m{len(vsets)}", vs))
        inband = 0
        for tag, vs in vsets:
            o = frontier(gears, vs)
            sh = shape(o)
            band = 0.55 <= sh["frac"] <= 0.75
            if tag != "REAL" and band:
                inband += 1
            W(f"{tag} | {vs} | {o['Fold']} | {o['F']} | {o['q']} | {o['budget']} | {sh['amin']} | "
              f"{sh['frac']:.3f} | {'yes' if band else 'no'} | {sh['nloc']} | "
              f"{'yes' if sh['convex'] else 'no'} | {sh['negd2']} | "
              f"{'YES at a=' + str(o['viol']) if o['viol'] else 'no'}")
            rows.append((tag, o, sh))
        W(f"  members (excluding the real machine) with a*/F_old in [0.55, 0.75]: "
          f"{inband} of 20")
        W(f"  [{time.time()-t0:.1f}s]")

    W("\n--- the three recorded budget violators (theory_tree node 2f.i), slack profile ---")
    known = [([5, 7, 11, 13, 17, 19], [1, 3, 4, 4, 4, 3], "m17 (1,3,4,4,4) + real 19"),
             ([5, 7, 11, 13, 17, 19], [2, 3, 3, 3, 3, 3], "m17 (2,3,3,3,3) + real 19"),
             ([5, 7, 11, 13], [1, 1, 5, 2], "m11 (1,1,5) + real 13")]
    for gears, vs, tag in known:
        o = frontier(gears, vs)
        sh = shape(o)
        W(f"{tag}: F_old = {o['Fold']}, q' = {o['q']}, F = {o['F']}, budget = "
          f"{o['Fold'] + o['q']}, budget slack = {o['budget']}")
        W(f"   s < 0 at a = {o['viol']} (a/F_old = "
          f"{[round(a / o['Fold'], 3) for a in o['viol']]}), s = "
          f"{[o['s'][a] for a in o['viol']]}; global minimiser a* = {sh['amin']} "
          f"({sh['frac']:.3f} F_old); strict local minima {sh['nloc']}; convex {sh['convex']}")
        W("   full slack profile a: s(a) = " +
          " ".join(f"{a}:{o['s'][a]}" for a in o["realised"]))

    W("\n--- violator hunt: 400 random members, incoming gear at its REAL tooth ---")
    for gears, name in ([5, 7, 11, 13, 17], "13->17"), ([5, 7, 11, 13, 17, 19], "17->19"):
        nv = 0
        tot = 0
        details = []
        for _ in range(400):
            vs = [rng.randrange(1, (g - 1) // 2 + 1) for g in gears[:-1]]
            vs.append(min(u_of(gears[-1]), gears[-1] - u_of(gears[-1])))
            o = frontier(gears, vs)
            tot += 1
            if o["viol"]:
                nv += 1
                sh = shape(o)
                fr = [round(a / o["Fold"], 3) for a in o["viol"]]
                details.append((vs, o["Fold"], o["q"], o["F"], o["viol"], fr,
                                [o["s"][a] for a in o["viol"]], sh["amin"]))
        W(f"{name}: {nv} budget violators of {tot} ({100*nv/tot:.2f}%)")
        for dtl in details:
            vs, Fo, q, F, viol, fr, sval, am = dtl
            W(f"   v={vs} F_old={Fo} q'={q} F={F} (budget {Fo+q}); s < 0 at a = {viol} "
              f"(a/F_old = {fr}), s = {sval}; global minimiser a* = {am} "
              f"({am/Fo:.3f} F_old)")
    W(f"\n[total {time.time()-t0:.1f}s]")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "fc_family.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
