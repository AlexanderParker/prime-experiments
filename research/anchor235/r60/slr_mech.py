"""slr_mech.py -- the MECHANISM of the short-letter row.

(a) THE GEAR-5 PAIR FILTER.  Three openings x0 < x1 < x2 with x1-x0 = a, x2-x1 = v need
    x0, x0+a, x0+a+v all open modulo every gear.  For a gear p the forbidden set for x0 is
    T_p u (T_p - a) u (T_p - a - v), of size at most 6; it can exhaust Z_p only for p = 5.  So
    gear 5 is the ONLY gear that can forbid an entire residue class of the neighbour a.  This
    section enumerates the achievable (a mod 5, v mod 5) exactly, states the count, and checks it
    against every realised adjacent pair of every machine to {5..29}.

(b) THE LOCAL CONTRAST.  delta(v) = r(v) - (r(v-1) + r(v+1))/2, stratified by c_5(v); a
    trend-free test of the gear-5 stratification of the row maxima.

(c) THE CLOSERS OF AN a_L-GAP.  Which gears can strike BOTH ends of a gap of size a_L (chain law,
    file 05 (C)): Leg(a_L) = {p : p | a_L (3a_L - 1)(3a_L + 1)} and 3a_L = q' -+ 1, so
    {3a_L - 1, 3a_L + 1} = {q', q' -+ 2}.  Reported inside and outside M at every rung.

(d) THE STRIKER CENSUS of the a_L-gaps: which gears of M block the interior columns of an
    a_L-gap, and which gear is the SOLE striker there, with counts over the full period.

(e) THE EXTREMAL OCCURRENCE (r(a_L), a_L): the three openings, their residues, and the striker
    map of every blocked column of both gaps.

Outputs results/slr_mech.txt.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import build_levels, u_of                    # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def teeth(p):
    u = u_of(p)
    return {u % p, (-u) % p}


def cost_closed(p, v):
    if v % p == 0:
        return 2
    if (3 * v - 1) % p == 0 or (3 * v + 1) % p == 0:
        return 3
    return 4


def achievable_mod(p):
    """{(a mod p, v mod p) : some x0 has x0, x0+a, x0+a+v all open mod p}."""
    O = [x for x in range(p) if x not in teeth(p)]
    S = set()
    for o0 in O:
        for o1 in O:
            for o2 in O:
                S.add(((o1 - o0) % p, (o2 - o1) % p))
    return S


def strikers(x, gears):
    return [g for g in gears if (x % g) in teeth(g)]


def main():
    lines = []
    W = lines.append
    t0 = time.time()

    # ---------------- (a) the gear-5 pair filter -----------------------------
    W("=== (a) THE GEAR-5 PAIR FILTER ===")
    for p in (5, 7, 11, 13):
        S = achievable_mod(p)
        miss = sorted(set((a, v) for a in range(p) for v in range(p)) - S)
        W(f"  gear {p}: teeth {sorted(teeth(p))}, openings "
          f"{[x for x in range(p) if x not in teeth(p)]}; "
          f"{len(S)} of {p*p} class pairs (a mod {p}, v mod {p}) achievable; "
          f"forbidden: {miss if len(miss) <= 12 else str(len(miss)) + ' pairs'}")
    S5 = achievable_mod(5)
    W("  the forbidden-neighbour count of a size v, from gear 5:")
    for v in range(5):
        bad = [a for a in range(5) if (a, v) not in S5]
        W(f"    v = {v} (mod 5): c_5 = {cost_closed(5, v) if v else 2}, "
          f"forbidden neighbour classes a = {bad} -> {len(bad)} of 5")
    W("  IDENTITY: #forbidden classes = c_5(v) - 2 (0 for v = 0 mod 5, 1 for v = +-2, 2 for "
      "v = +-1).")
    W("  and no gear p >= 7 forbids any class: |T_p u (T_p-a) u (T_p-a-v)| <= 6 < 7 <= p.")

    L = build_levels()
    W("\n  check against every realised adjacent pair (a,v) of every machine:")
    for n in range(0, 7):
        size = L[n].size
        mx = int(size.max())
        Wd = mx + 1
        r = np.empty_like(size)
        r[:-1] = size[1:]
        r[-1] = size[0]
        D = np.bincount(size * Wd + r, minlength=Wd * Wd).reshape(Wd, Wd)
        pairs = [(a, v) for a in range(Wd) for v in range(Wd) if D[a][v]]
        bad = [(a, v) for a, v in pairs if (a % 5, v % 5) not in S5]
        W(f"    M = {{5..{L[n].gears[-1]}}}: {len(pairs)} realised pairs, "
          f"{len(bad)} in a gear-5-forbidden class")
        assert not bad, bad[:5]

    # ---------------- (b) the local contrast ---------------------------------
    W("\n=== (b) THE LOCAL CONTRAST delta(v) = r(v) - (r(v-1)+r(v+1))/2, by c_5(v) ===")
    import json
    row = json.load(open(os.path.join(OUT, "slr_row.json")))
    m31 = json.load(open(os.path.join(OUT, "slr_m31.json"))) if os.path.exists(
        os.path.join(OUT, "slr_m31.json")) else None
    todo = [(k, row[k]["gears"][-1], {int(a): b for a, b in row[k]["r"].items()})
            for k in row]
    if m31:
        todo.append(("37", 31, {int(a): b for a, b in m31["r"].items()}))
    W("  machine | n(c=2) mean delta | n(c=3) mean delta | n(c=4) mean delta | ordered?")
    for k, top, r in todo:
        acc = {2: [], 3: [], 4: []}
        for v in sorted(r):
            if v - 1 in r and v + 1 in r:
                acc[cost_closed(5, v)].append(r[v] - (r[v - 1] + r[v + 1]) / 2)
        if min(len(a) for a in acc.values()) == 0:
            continue
        mu = {c: float(np.mean(acc[c])) for c in acc}
        W(f"  {{5..{top}}} | {len(acc[2])} {mu[2]:+.2f} | {len(acc[3])} {mu[3]:+.2f} | "
          f"{len(acc[4])} {mu[4]:+.2f} | {'YES' if mu[2] > mu[3] > mu[4] else 'no'}")

    # ---------------- (c) the closers of an a_L-gap --------------------------
    W("\n=== (c) THE CLOSERS OF AN a_L-GAP: gears that can strike BOTH ends ===")
    W("  rung q' | a_L | 3a_L | {3a_L-1, 3a_L+1} | Leg(a_L) all | in M | OUTSIDE M | q'+-2 prime?")
    for i in range(0, 9):
        gears = PRIMES[:i + 1]
        qn = PRIMES[i + 1]
        u = u_of(qn)
        aL = min((2 * u) % qn, qn - (2 * u) % qn)
        big = [p for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
                           71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
               if cost_closed(p, aL) <= 3]
        inm = [p for p in big if p in gears]
        out = [p for p in big if p not in gears]
        other = (3 * aL - 1) if 3 * aL - 1 != qn else (3 * aL + 1)
        isp = other > 1 and all(other % d for d in range(2, int(other ** 0.5) + 1))
        tagp = " (PRIME, the twin of q')" if isp else " (composite)"
        W(f"  {qn} | {aL} | {3*aL} = q'{'-' if 3*aL == qn-1 else '+'}1 | "
          f"{{{3*aL-1}, {3*aL+1}}} | {big} | {inm} | {out} | other = {other}{tagp}")
    W("  THE CLOSER LAW: {3a_L-1, 3a_L+1} = {q', q' -+ 2}; every prime factor of q' -+ 2 other")
    W("  than q' +2 itself is < q' hence already a gear of M, and the two ends of a gap of M are")
    W("  openings of M, so no gear of M strikes them.  Hence the ONLY gears that can close an")
    W("  a_L-gap of M are q' itself, and q'+2 when q' = 2 (mod 3) and q'+2 is prime.")

    # ---------------- (d)/(e) the striker census -----------------------------
    W("\n=== (d) STRIKER CENSUS OF THE a_L-GAPS (full periods, M = {5} .. {5..23}) ===")
    for n in range(0, 7):
        lv = L[n]
        qn = PRIMES[n + 1]
        u = u_of(qn)
        aL = min((2 * u) % qn, qn - (2 * u) % qn)
        gears = lv.gears
        idx = np.flatnonzero(lv.size == aL)
        if idx.size == 0:
            W(f"  M = {{5..{gears[-1]}}} q' = {qn} a_L = {aL}: no gap of that size")
            continue
        x = lv.O[idx]
        cnt = np.zeros((idx.size, aL - 1), dtype=np.int8)
        who = np.zeros((idx.size, aL - 1), dtype=np.int8)
        for gi, g in enumerate(gears):
            hit = np.zeros((idx.size, aL - 1), dtype=bool)
            for t in teeth(g):
                for j in range(1, aL):
                    hit[:, j - 1] |= ((x + j) % g == t)
            cnt += hit
            who[hit] = gi + 1
        sole = (cnt == 1)
        W(f"  M = {{5..{gears[-1]}}} q' = {qn} a_L = {aL}: {idx.size:,} a_L-gaps, "
          f"{aL-1} interior columns each")
        tot = {}
        for gi, g in enumerate(gears):
            tot[g] = int(((who == gi + 1) & sole).sum())
        allb = {g: int(0) for g in gears}
        for gi, g in enumerate(gears):
            h = np.zeros((idx.size, aL - 1), dtype=bool)
            for t in teeth(g):
                for j in range(1, aL):
                    h[:, j - 1] |= ((x + j) % g == t)
            allb[g] = int(h.sum())
        W(f"    columns blocked by gear (with multiplicity): "
          + " ".join(f"{g}:{allb[g]:,}" for g in gears))
        W(f"    columns where the gear is the SOLE striker:  "
          + " ".join(f"{g}:{tot[g]:,}" for g in gears))
        W(f"    interior columns with 1 / 2 / 3+ strikers: "
          f"{int((cnt==1).sum()):,} / {int((cnt==2).sum()):,} / {int((cnt>=3).sum()):,}")

    # (e) the extremal occurrence
    W("\n=== (e) THE EXTREMAL OCCURRENCE (r(a_L), a_L), striker map ===")
    for n in range(0, 7):
        lv = L[n]
        qn = PRIMES[n + 1]
        u = u_of(qn)
        aL = min((2 * u) % qn, qn - (2 * u) % qn)
        gears = lv.gears
        s = lv.size
        nxt = np.empty_like(s)
        nxt[:-1] = s[1:]
        nxt[-1] = s[0]
        cand = np.flatnonzero((nxt == aL) & (s > 0))
        if cand.size == 0:
            continue
        best = cand[np.argmax(s[cand])]
        a = int(s[best])
        x0 = int(lv.O[best])
        W(f"  M = {{5..{gears[-1]}}} q' = {qn}: extremal pair (a, a_L) = ({a}, {aL}) at x0 = {x0}")
        W(f"    openings x0={x0}, x1={x0+a}, x2={x0+a+aL}; residues mod gear "
          f"(tooth classes in brackets):")
        for g in gears:
            W(f"      gear {g:>2}: teeth {sorted(teeth(g))}  x0={x0%g} x1={(x0+a)%g} "
              f"x2={(x0+a+aL)%g}   c_g(a_L) = {cost_closed(g, aL)}")
        w1 = " ".join(f"{j}:{'+'.join(map(str, strikers(x0+j, gears)))}" for j in range(1, a))
        w2 = " ".join(f"{j}:{'+'.join(map(str, strikers(x0+a+j, gears)))}"
                      for j in range(1, aL))
        W(f"    left gap ({a}) blocked columns  (offset:strikers): {w1}")
        W(f"    right gap ({aL}) blocked columns (offset:strikers): {w2}")

    W(f"\n[total {time.time()-t0:.1f}s]")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "slr_mech.txt"), "w").write(txt)
    print(f"wrote {OUT}/slr_mech.txt ({len(txt)} chars)")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
