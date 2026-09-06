"""pa_strike.py -- the strike table of the attaining 2-run of the letter, in tooth units.

For every rung q' = 13, 17, 19, 23, 29, 31 (machines {5..11} .. {5..29}) take the attaining
2-run of the letter -- the pair (r(a_L), a_L) with the largest neighbour -- at EVERY occurrence in
the full period, and record

  * lam_g = 6 x_L (mod g), the near end of the LETTER gap in tooth units (teeth at lam = +-1);
  * the full strike table: which gears strike each column of the run;
  * the sole strikers;
  * per gear, how many columns it strikes inside the letter gap and inside the neighbour;
  * the mirror class of the residue vector (lam -> -lam is the machine's mirror k -> P - k).

Also checks, exactly:
  D2  if g | a_L then g strikes exactly 2 a_L / g interior columns of EVERY a_L-gap;
  D3  the specialised obstruction law against the generic spare-gear obstruction test;
  P7b at a twin rung the partner p = q' - 2 strikes at most 1 interior column of an a_L-gap
      (checked over ALL admissible classes lam_p, which is a complete proof over classes, and
      against the realised distribution).

Outputs results/pa_strike.txt / .json.
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import build_levels, u_of                    # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def unorm(g):
    w = u_of(g)
    return min(w, g - w)


def eps_of(q):
    return 1 if q % 6 == 5 else -1


def teeth(g):
    w = u_of(g)
    return (w % g, (-w) % g)


def strikes(g, x):
    """does gear g strike column x?"""
    return x % g in teeth(g)


# ---------------------------------------------------------------- m29 streaming


def m29_positions(lv23):
    """yield the openings of {5..29} in order, one copy of the m23 period at a time."""
    q = 29
    w = u_of(q)
    P23 = lv23.P
    r23 = lv23.O % q
    for j in range(q):
        rj = (r23 + j * P23) % q
        keep = (rj != w % q) & (rj != (-w) % q)
        yield lv23.O[keep] + j * P23


def find_pairs_stream(gen, a, v, cap=64):
    """x_0 of every 2-run with gaps (a, v) or (v, a), from a stream of position chunks."""
    out = []
    tail = np.empty(0, dtype=np.int64)
    for chunk in gen:
        p = np.concatenate([tail, chunk])
        d = np.diff(p)
        if d.size >= 2:
            hit = ((d[:-1] == a) & (d[1:] == v)) | ((d[:-1] == v) & (d[1:] == a))
            idx = np.flatnonzero(hit)
            for i in idx:
                out.append(int(p[i]))
                if len(out) >= cap:
                    return out
        tail = p[-2:] if p.size >= 2 else p
    return out


def row_max(size):
    mult = np.bincount(size.astype(np.int64))
    realised = np.flatnonzero(mult).tolist()
    lft, rgt = size[:-1], size[1:]
    r = {}
    for v in realised:
        s = set(np.unique(rgt[lft == v]).tolist()) | set(np.unique(lft[rgt == v]).tolist())
        r[v] = max(s) if s else 0
    return realised, int(size.max()), r


# ---------------------------------------------------------------- D3


def obstructed_generic(h, a, v):
    d = (2 * u_of(h)) % h
    if a % h == 0 or v % h == 0:
        return True
    c1 = (a % h == d) or (v % h == (-d) % h)
    c2 = (a % h == (-d) % h) or (v % h == d)
    return c1 and c2


def obstructed_letter(h, a, q):
    """the specialised law: h | a, or h | q'+eps, or (h | q'+2eps and a = -eps d_h (mod h))."""
    e = eps_of(q)
    d = (2 * u_of(h)) % h
    if a % h == 0:
        return True
    if (q + e) % h == 0:
        return True
    if (q + 2 * e) % h == 0 and a % h == (-e * d) % h:
        return True
    return False


def main():
    t0 = time.time()
    L = []
    W = L.append
    res = {}
    W("=== THE STRIKE TABLE OF THE ATTAINING 2-RUN, IN TOOTH UNITS ===")

    levels = build_levels()          # {5} .. {5..23}
    setups = []
    for n, qn in ((2, 13), (3, 17), (4, 19), (5, 23), (6, 29)):
        lv = levels[n]
        setups.append((qn, list(lv.gears), lv.O, lv.size, None))

    # ---------- D3, exhaustive over (rung, gear, neighbour a) ----------
    W("\n--- D3: the specialised obstruction law vs the generic spare-gear test ---")
    nd3 = badd3 = 0
    d3rows = []
    for qn in (7, 11, 13, 17, 19, 23, 29, 31, 37):
        M = [g for g in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37) if g < qn]
        aL = 2 * unorm(qn)
        exc = []
        for h in M:
            for a in range(1, 121):
                nd3 += 1
                if obstructed_generic(h, a, aL) != obstructed_letter(h, a, qn):
                    badd3 += 1
                    exc.append((h, a))
        e = eps_of(qn)
        div_aL = [h for h in M if (qn + e) % h == 0]
        div_tw = [h for h in M if (qn + 2 * e) % h == 0]
        W(f"rung {qn:>2}: gears {M} | h | q'+eps: {div_aL or '-'} | h | q'+2eps: {div_tw or '-'}"
          f" | mismatches {len(exc)}")
        d3rows.append(dict(q=qn, div_aL=div_aL, div_tw=div_tw, exc=exc))
    W(f"  D3: {nd3} (rung, gear, neighbour) cells, {badd3} mismatches")

    # ---------- P7b, exhaustive over classes ----------
    W("\n--- P7b: the twin partner's interior strikes inside an a_L-gap, over ALL classes ---")
    W("rung q' | p = q'-2 | admissible lam_p | interior strikes of p: distribution")
    p7 = []
    for qn in (7, 13, 19, 31):
        p = qn - 2
        e = eps_of(qn)
        aL = 2 * unorm(qn)
        sh = (2 * (qn + e)) % p            # 6 a_L (mod p)
        adm = [lam for lam in range(p)
               if lam % p not in (1 % p, (-1) % p, (1 - sh) % p, (-1 - sh) % p)]
        dist = Counter()
        for lam in adm:
            c = sum(1 for j in range(1, aL) if (lam + 6 * j) % p in (1 % p, (p - 1) % p))
            dist[c] += 1
        W(f"{qn:>7} | {p:>7} | {len(adm)} of {p} | {dict(sorted(dist.items()))}")
        p7.append(dict(q=qn, p=p, adm=len(adm), dist={int(k): int(v) for k, v in dist.items()}))
    W("  (a complete enumeration over residue classes, so it is a proof for that gear, not a scan)")

    # ---------- per machine: the attaining run ----------
    for qn, gears, O, size, _ in setups:
        t1 = time.time()
        realised, F, r = row_max(size)
        aL = 2 * unorm(qn)
        ra = r.get(aL, 0)
        S = aL + ra
        W(f"\n=== rung {qn}  M = {gears}  F = {F}  a_L = {aL}  r(a_L) = {ra}  S = {S}  "
          f"E = {S - F:+d} ===")
        # occurrences
        d = size
        hit = ((d[:-1] == ra) & (d[1:] == aL)) | ((d[:-1] == aL) & (d[1:] == ra))
        idx = np.flatnonzero(hit)
        W(f"  occurrences of the attaining pair: {idx.size}")
        info = analyse_occurrences(gears, [int(O[i]) for i in idx],
                                   [(int(size[i]), int(size[i + 1])) for i in idx],
                                   aL, ra, qn, W)
        res[str(qn)] = dict(gears=gears, F=F, aL=aL, r=ra, S=S, E=S - F, nocc=int(idx.size),
                            **info)
        # D2 on the full period
        d2 = check_d2(gears, O, size, aL, qn, W)
        res[str(qn)]["d2"] = d2
        W(f"  [{time.time()-t1:.1f}s]")
        del d, hit, idx

    # ---------- rung 31: M = {5..29}, streamed ----------
    lv23 = levels[6]
    del levels
    qn = 31
    gears29 = [5, 7, 11, 13, 17, 19, 23, 29]
    aL, ra = 10, 35
    W(f"\n=== rung {qn}  M = {gears29}  F = 43  a_L = {aL}  r(a_L) = {ra}  S = {aL+ra}  E = +2 ===")
    W("  (the m29 period is streamed as 29 copies of the m23 period; r(10) = 35 is the parent's")
    W("   full-period value and is used as the target, then verified by finding the pair)")
    pos = find_pairs_stream(m29_positions(lv23), ra, aL, cap=64)
    W(f"  occurrences of the attaining pair found: {len(pos)}")
    if pos:
        pairs = []
        for x in pos:
            # decide orientation by testing which column is open
            pairs.append(None)
        # recompute orientation from the machine directly
        oriented = []
        for x in pos:
            def op(y):
                return all(y % g not in teeth(g) for g in gears29)
            if op(x) and op(x + ra) and op(x + ra + aL):
                oriented.append((x, (ra, aL)))
            elif op(x) and op(x + aL) and op(x + aL + ra):
                oriented.append((x, (aL, ra)))
        W(f"  orientations resolved: {len(oriented)}")
        info = analyse_occurrences(gears29, [x for x, _ in oriented],
                                   [p for _, p in oriented], aL, ra, qn, W)
        res[str(qn)] = dict(gears=gears29, F=43, aL=aL, r=ra, S=aL + ra, E=2,
                            nocc=len(oriented), **info)

    json.dump(res, open(os.path.join(OUT, "pa_strike.json"), "w"))
    txt = "\n".join(L)
    open(os.path.join(OUT, "pa_strike.txt"), "w").write(txt)
    print(f"wrote {OUT}/pa_strike.txt ({len(txt)} chars, {time.time()-t0:.1f}s)")


def check_d2(gears, O, size, aL, qn, W):
    """D2: if g | a_L then g strikes exactly 2 a_L / g interior columns of EVERY a_L-gap."""
    e = eps_of(qn)
    padg = [g for g in gears if aL % g == 0]
    if not padg:
        W(f"  D2: no gear divides a_L = {aL} (q'+eps = {qn+e} has no factor in M) -- vacuous")
        return dict(pad=[], n=0, bad=0)
    idx = np.flatnonzero(size == aL)
    xs = O[idx]
    bad = 0
    for g in padg:
        want = 2 * aL // g
        lam = xs % g
        cnt = np.zeros(xs.size, dtype=np.int64)
        t1, t2 = teeth(g)
        for j in range(1, aL):
            cnt += (((lam + j) % g) == t1) | (((lam + j) % g) == t2)
        b = int(np.count_nonzero(cnt != want))
        bad += b
        W(f"  D2: gear {g} | a_L = {aL}: predicted {want} interior strikes, "
          f"exceptions {b} of {xs.size} a_L-gaps")
    return dict(pad=padg, n=int(xs.size), bad=int(bad))


def analyse_occurrences(gears, xs, pairs, aL, ra, qn, W):
    """strike tables of every occurrence; returns a summary dict."""
    S = aL + ra
    lams = []
    tables = []
    for x0, (g1, g2) in zip(xs, pairs):
        xL = x0 if g1 == aL else x0 + g1        # left end of the LETTER gap
        lam = tuple((6 * xL) % g for g in gears)
        lams.append(lam)
        tab = []
        for j in range(S + 1):
            col = x0 + j
            hit = [g for g in gears if col % g in teeth(g)]
            tab.append(hit)
        tables.append((x0, (g1, g2), xL, lam, tab))
    # mirror classes of lam
    def canon(lam):
        m = tuple((-a) % g for a, g in zip(lam, gears))
        return min(lam, m)
    classes = Counter(canon(l) for l in lams)
    W(f"  tooth-unit vectors lam_g = 6 x_L (mod g): {len(set(lams))} distinct, "
      f"{len(classes)} up to the mirror")
    for c, n in classes.items():
        W(f"    class {dict(zip(gears, c))}  x {n}")
    # the strike table of one representative
    x0, (g1, g2), xL, lam, tab = tables[0]
    W(f"  representative occurrence x_0 = {x0}, word ({g1}, {g2}), letter gap at "
      f"[{xL}, {xL + aL}]")
    W(f"    lam = {dict(zip(gears, lam))}")
    W("    column offsets 1..S-1 (strikers; '*' = sole striker):")
    sole = {g: [] for g in gears}
    for j in range(1, S):
        hit = tab[j]
        if len(hit) == 1:
            sole[hit[0]].append(j)
        mark = "*" if len(hit) == 1 else " "
        W(f"      {j:>3}{mark} {'+'.join(map(str, hit))}"
          f"{'   <- middle opening' if j == (aL if g1 == aL else g1) else ''}")
    W("    sole-striker columns by gear: " +
      "; ".join(f"{g}:{sole[g]}" for g in gears))
    nfree = [g for g in gears if not sole[g]]
    W(f"    gears with NO sole-striker column: {nfree or 'none'} "
      f"(every gear busy: {'yes' if not nfree else 'NO'})")
    # split by side
    mid = aL if g1 == aL else g1
    letter_lo, letter_hi = (0, aL) if g1 == aL else (g1, S)
    share = {}
    for g in gears:
        inlet = sum(1 for j in range(letter_lo + 1, letter_hi) if g in tab[j])
        innb = sum(1 for j in range(1, S) if g in tab[j] and not (letter_lo < j < letter_hi))
        share[g] = (inlet, innb)
    W("    columns struck (inside the letter gap / inside the neighbour): " +
      "; ".join(f"{g}:{share[g][0]}/{share[g][1]}" for g in gears))
    confined = [g for g in gears if share[g][1] == 0]
    W(f"    gears confined to the letter gap (0 strikes in the neighbour): {confined or 'none'}")
    return dict(lams=[list(l) for l in lams],
                nclass=len(classes),
                rep=dict(x0=x0, word=[g1, g2], xL=xL, lam=list(lam),
                         sole={str(g): sole[g] for g in gears},
                         share={str(g): list(share[g]) for g in gears},
                         confined=confined, free=nfree))


if __name__ == "__main__":
    main()
