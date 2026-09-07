"""The engine with the first manifold gears, exactly over full periods.

Stage k: gears q' = g_1 < g_2 < ... < g_k (the first k primes above q), period P_k = q# x prod g_i.
Over [0, P_k) we count the openings of the k-gear manifold by length L, how many hold an engine
twin slot (a pair with both members coprime to q#), and compare with the CRT product exactly.
The arc table of q' alone: for each residue r mod q# of the arc's start (n = 1 mod q'), which arc
positions i in 0..q'-4 are slots.  Then the same counts with random phases for every gear.

usage: uv run python research/valves/r4/composite_exact.py q [Pmax]
"""
import sys, os, json, math
import numpy as np
from fractions import Fraction

q = int(sys.argv[1]); Pmax = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000_000
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


pr = primes_upto(200)
engine = [p for p in pr if p <= q]; qsharp = math.prod(engine)
gears_all = [p for p in pr if p > q]
res = np.arange(qsharp)
slot_pat = np.ones(qsharp, dtype=bool)
for p in engine:
    slot_pat &= (res % p != 0) & ((res + 2) % p != 0)
slots = [int(r) for r in np.flatnonzero(slot_pat)]
qprime = gears_all[0]


def S_L(L):
    return sum(1 for r in range(qsharp) if any(slot_pat[(r + i) % qsharp] for i in range(L)))


def runs_and_slots(opn, P, offset=0):
    """openings of the integer-open indicator opn over a full period (cyclic), with slot counts."""
    po = opn & np.roll(opn, -2)              # pair n open iff n and n+2 open (cyclic)
    # rotate so that position 0 is a struck pair (exists unless everything is open)
    z = int(np.flatnonzero(~po)[0])
    po_r = np.roll(po, -z)
    d = np.diff(np.concatenate([[0], po_r.astype(np.int8), [0]]))
    st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
    L = en - st
    eo = np.tile(slot_pat, P // qsharp)
    eo_r = np.roll(eo, -z)
    C = np.concatenate([[0], np.cumsum(eo_r)])
    sl = C[en] - C[st]
    starts = (st + z) % P
    return L, sl, starts


out = {"q": q, "qsharp": qsharp, "slots": slots, "qprime": qprime, "stages": []}
lines = []

# ---- the arc table of q' alone: arc start n = 1 (mod q'), positions i = 0..q'-4 (residues 1..q'-3 mod q')
arc_len = qprime - 3
per_pos = [0] * arc_len; with_k = {}
for r in range(qsharp):
    k = 0
    for i in range(arc_len):
        if slot_pat[(r + i) % qsharp]:
            per_pos[i] += 1; k += 1
    with_k[k] = with_k.get(k, 0) + 1
lines.append(f"q = {q}, q# = {qsharp}, slots mod q# = {len(slots)} {slots if len(slots) <= 20 else ''}; q' = {qprime}, arc length q'-3 = {arc_len}")
lines.append(f"arc table (q' alone, one arc per residue r mod q# of the arc start): slots at arc position i, over the {qsharp} starts: "
             f"{per_pos} (uniform = {len(slots)} each); arcs holding k slots: {dict(sorted(with_k.items()))}; "
             f"arcs with >= 1 slot = {qsharp - with_k.get(0, 0)} = |S_{arc_len}| = {S_L(arc_len)} of {qsharp}")
out["arc_table"] = {"per_position": per_pos, "arcs_with_k_slots": with_k}

# ---- stages, exact, phase zero, then random phases
rng = np.random.default_rng(7)
prev = None
P = qsharp; gears = []
for g in gears_all:
    if P * g > Pmax:
        break
    gears.append(g); P *= g
    W = math.prod(gears)
    for trial in ["zero", "rand1", "rand2"]:
        phases = {h: 0 for h in gears} if trial == "zero" else {h: int(rng.integers(h)) for h in gears}
        opn = np.ones(P, dtype=bool)
        for h in gears:
            opn[phases[h]::h] = False
        L, sl, starts = runs_and_slots(opn, P)
        Ls = sorted(set(L.tolist()))
        rec = {"gears": list(gears), "period": P, "trial": trial, "phases": phases, "L": {}}
        for Lv in Ls:
            m = L == Lv
            cnt = int(m.sum()); al = int((sl[m] > 0).sum())
            # CRT exact count over the period: q# x W x second difference of prod (g - k)/g
            def cW(k):
                return math.prod(max(h - k, 0) for h in gears)
            if Lv == 1:
                crt = qsharp * (cW(2) - 2 * cW(4) + cW(5))
            else:
                crt = qsharp * (cW(Lv + 2) - 2 * cW(Lv + 3) + cW(Lv + 4))
            al_crt = Fraction(crt * S_L(Lv), qsharp)
            # residues of starts mod q#: uniform?
            rh = np.bincount(starts[m] % qsharp, minlength=qsharp)
            rec["L"][Lv] = {"count": cnt, "crt": crt, "aligned": al, "aligned_crt": float(al_crt),
                            "res_min": int(rh.min()), "res_max": int(rh.max())}
        rec["max_L"] = max(Ls)
        out["stages"].append(rec)
        if trial == "zero":
            lines.append(f"stage gears {gears} period {P}: longest opening {max(Ls)} (ceiling q'-3 = {arc_len}); per L: "
                         + "; ".join(f"L={Lv}: {rec['L'][Lv]['count']} (CRT {rec['L'][Lv]['crt']}), aligned {rec['L'][Lv]['aligned']} "
                                     f"(CRT {rec['L'][Lv]['aligned_crt']:.1f}), starts mod q# in [{rec['L'][Lv]['res_min']}, {rec['L'][Lv]['res_max']}]"
                                     for Lv in Ls))
            if prev is not None:
                # thinning factor per L against the CRT factor (g - L - 2)/g for the count of blocks of length >= L
                thin = []
                for Lv in Ls:
                    if Lv in prev["L"] and Lv >= 2:
                        a1 = sum(prev["L"][x]["aligned"] for x in prev["L"] if x >= Lv)
                        a2 = sum(rec["L"][x]["aligned"] for x in rec["L"] if x >= Lv)
                        c1 = sum(prev["L"][x]["count"] for x in prev["L"] if x >= Lv)
                        c2 = sum(rec["L"][x]["count"] for x in rec["L"] if x >= Lv)
                        thin.append(f"L>={Lv}: openings x{g * c2 / c1 / 1.0:.4f}, aligned x{g * a2 / a1 if a1 else float('nan'):.4f} "
                                    f"(CRT {g - Lv - 2}) per period-of-{g}")
                lines.append(f"  adding gear {g}: " + "; ".join(thin))
            prev = rec
        else:
            same = all(rec["L"][Lv]["count"] == out["stages"][-1 - (1 if trial == "rand1" else 2)]["L"][Lv]["count"]
                       and rec["L"][Lv]["aligned"] == out["stages"][-1 - (1 if trial == "rand1" else 2)]["L"][Lv]["aligned"]
                       for Lv in Ls) and set(Ls) == set(out["stages"][-1 - (1 if trial == "rand1" else 2)]["L"].keys())
            lines.append(f"  {trial} phases {phases}: counts and aligned counts identical to phase zero for every L: {same}")

json.dump(out, open(os.path.join(outdir, f"composite_q{q}.json"), "w"), indent=1, default=str)
print("\n".join(lines))
