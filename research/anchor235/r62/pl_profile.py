"""pl_profile.py -- G(v) = v + r(v) and its excess E(v) = G(v) - F(M) for EVERY realised size,
classified by the three coupling predicates, at every machine {5} .. {5..29} (rebuilt here on full
periods) and {5..31} (read from the parent branch's cached exact run, r60/results/slr_m31.json).

    Pad(v)  = { p >= 5 : p | v }
    Leg(v)  = { p >= 5 : p | 3v-1 or p | 3v+1 }
    Coup(v) = Pad(v) u Leg(v)          (the chain law's "p can strike both ends of a v-gap")

    U_leg  : Leg(v)  n M = {}     the brief's literal predicate
    U_full : Coup(v) n M = {}     the chain law's predicate
    U_pad  : Pad(v)  n M = {}

Outputs results/pl_profile.txt and results/pl_profile.json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
sys.path.insert(0, os.path.join(HERE, "..", "r58"))
from mf_core import build_levels, u_of                    # noqa: E402
from ag_gate import build_m29_gaps                        # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def letters(q):
    a = (2 * u_of(q)) % q
    return (min(a, q - a), max(a, q - a))


def factors(n):
    out = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.add(d)
            n //= d
        d += 1
    if n > 1:
        out.add(n)
    return {p for p in out if p >= 5}


def pad(v):
    return factors(v)


def leg(v):
    return factors(3 * v - 1) | factors(3 * v + 1)


def dict_of(size, chunk=20_000_000):
    n = size.size
    mx = int(size.max())
    W = mx + 1
    D = np.zeros((W, W), dtype=np.int64)
    m = np.zeros(W, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        c = size[s:e].astype(np.int64)
        if e < n:
            r = size[s + 1:e + 1].astype(np.int64)
        else:
            r = np.empty(e - s, dtype=np.int64)
            r[:-1] = size[s + 1:e]
            r[-1] = size[0]
        m += np.bincount(c, minlength=W)
        D += np.bincount(c * W + r, minlength=W * W).reshape(W, W)
        del c, r
    return D, m, mx


def profile_from_D(D, m):
    realised = np.flatnonzero(m).tolist()
    F = max(realised)
    F2 = max(a + v for a in realised for v in np.flatnonzero(D[a]).tolist())
    r = {}
    for v in realised:
        s = set(np.flatnonzero(D[:, v]).tolist()) | set(np.flatnonzero(D[v]).tolist())
        r[v] = max(s) if s else 0
    return realised, F, F2, r


def report(gears, qn, realised, F, F2, r, W, res):
    aL, bL = letters(qn)
    M = set(gears)
    rows = []
    for v in realised:
        P, L = pad(v), leg(v)
        C = P | L
        rows.append(dict(v=v, r=r[v], G=v + r[v], E=v + r[v] - F,
                         pad=sorted(P & M), leg=sorted(L & M), coup=sorted(C & M),
                         u_leg=not (L & M), u_full=not (C & M), u_pad=not (P & M)))
    tag = f"M = {{5..{gears[-1]}}}" if len(gears) > 1 else "M = {5}"
    W(f"\n=== {tag}   q' = {qn}   F = {F}   F_2 = {F2}   a_L = {aL}   b_L = {bL} ===")
    W(f"  realised sizes {len(realised)};  E(v) = v + r(v) - F  ranges "
      f"{min(x['E'] for x in rows)} .. {max(x['E'] for x in rows)}  (F_2 - F = {F2 - F})")
    W("  v : r(v) G(v) E(v) | pad n M | leg n M   (* = E > 3)")
    for x in rows:
        star = "*" if x["E"] > 3 else " "
        W(f"   {star}{x['v']:>3}: {x['r']:>3} {x['G']:>3} {x['E']:>+4}  | "
          f"{','.join(map(str, x['pad'])) or '-':<10} | {','.join(map(str, x['leg'])) or '-'}")
    for name, key in (("U_leg  (brief)", "u_leg"), ("U_full (chain law)", "u_full"),
                      ("U_pad", "u_pad")):
        unc = [x for x in rows if x[key]]
        bad = [x for x in unc if x["E"] > 3]
        cou = [x for x in rows if not x[key]]
        badc = [x for x in cou if x["E"] > 3]
        ru = f"{len(bad)}/{len(unc)}" if unc else "-/0"
        rc = f"{len(badc)}/{len(cou)}" if cou else "-/0"
        W(f"  {name:<20}: uncoupled sizes {sorted(x['v'] for x in unc)}")
        W(f"  {'':<20}  violations E>3: uncoupled {ru}"
          f"{' (' + ','.join(f'{x[chr(118)]}:{x[chr(69)]:+d}' for x in bad) + ')' if bad else ''}"
          f" | coupled {rc}")
    aLrow = next((x for x in rows if x["v"] == aL), None)
    if aLrow:
        W(f"  the letter a_L = {aL}: r = {aLrow['r']}, G = {aLrow['G']}, E = {aLrow['E']:+d}; "
          f"pad n M = {aLrow['pad'] or '-'}, leg n M = {aLrow['leg'] or '-'}, "
          f"U_leg {aLrow['u_leg']}, U_full {aLrow['u_full']}, U_pad {aLrow['u_pad']}")
    else:
        W(f"  the letter a_L = {aL} is NOT REALISED as a gap size of M")
    res[str(qn)] = dict(gears=list(gears), qn=qn, F=F, F2=F2, aL=aL, bL=bL, rows=rows)


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== THE PINNED LETTER: G(v) = v + r(v), E(v) = G(v) - F, and the coupling predicates ===")
    res = {}
    L = build_levels()                       # {5} .. {5..23}
    for n in range(0, 7):
        D, m, mx = dict_of(L[n].size)
        realised, F, F2, r = profile_from_D(D, m)
        report(L[n].gears, PRIMES[n + 1], realised, F, F2, r, W, res)
        del D, m
    W(f"\n[full-period machines to m23 done {time.time()-t0:.1f}s]")
    g29 = build_m29_gaps(L[6])
    W(f"[m29 gap array: {g29.size:,} gaps, F = {int(g29.max())}]")
    D, m, mx = dict_of(g29)
    realised, F, F2, r = profile_from_D(D, m)
    report([5, 7, 11, 13, 17, 19, 23, 29], 31, realised, F, F2, r, W, res)
    del D, m, g29, L
    W(f"[m29 done {time.time()-t0:.1f}s]")

    # m31 from the parent branch's cached exact streamed run (r60/results/slr_m31.json)
    p = os.path.join(HERE, "..", "r60", "results", "slr_m31.json")
    if os.path.exists(p):
        j = json.load(open(p))
        realised = j["realised"]
        r = {int(k): int(v) for k, v in j["r"].items()}
        report([5, 7, 11, 13, 17, 19, 23, 29, 31], 37, realised, j["F"], j["F2"], r, W, res)
        W("  [m31 read from r60/results/slr_m31.json -- the parent branch's exact streamed "
          "full-period run, not recomputed here]")

    # the cross-rung summary
    W("\n=== the excess of the letter, and of every size, across the rungs ===")
    W("rung q' | F | F_2 | a_L | E(a_L) | #sizes | #E>3 | max E | sizes attaining max E")
    for k, e in res.items():
        rows = e["rows"]
        aLrow = next((x for x in rows if x["v"] == e["aL"]), None)
        mx = max(x["E"] for x in rows)
        W(f"{k} | {e['F']} | {e['F2']} | {e['aL']} | "
          f"{aLrow['E'] if aLrow else '-'} | {len(rows)} | "
          f"{sum(1 for x in rows if x['E'] > 3)} | {mx} | "
          f"{[x['v'] for x in rows if x['E'] == mx]}")

    json.dump(res, open(os.path.join(OUT, "pl_profile.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "pl_profile.txt"), "w").write(txt)
    print(f"wrote {OUT}/pl_profile.txt ({len(txt)} chars, {time.time()-t0:.1f}s)")
    for k, e in res.items():
        rows = e["rows"]
        aLrow = next((x for x in rows if x["v"] == e["aL"]), None)
        ul = [x for x in rows if x["u_leg"]]
        uf = [x for x in rows if x["u_full"]]
        print(f"q'={k:>2} F={e['F']:>2} aL={e['aL']:>2} "
              f"E(aL)={aLrow['E'] if aLrow else '-':>3} "
              f"E>3: {sum(1 for x in rows if x['E'] > 3):>2}/{len(rows):>2}  "
              f"U_leg {len(ul)} (bad {sum(1 for x in ul if x['E'] > 3)})  "
              f"U_full {len(uf)} (bad {sum(1 for x in uf if x['E'] > 3)})")


if __name__ == "__main__":
    main()
