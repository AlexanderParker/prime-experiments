"""Prover C, round 33 -- the chain statement from the teeth: I + (T) + (L).

Reuses prover B's vehicle (chain_family_r32.py: open_mask, gaps_of, letter_a, real_tooth) and adds,
per (member, incoming tooth) row:
  * the chain table Q*_J with the ARGMAX POSITION (opening index) of every cell,
  * the three "smallest unproved statements" of chain_statement.md section 4:
      S1  Phi(a)   = max flank sum over occurrences of the gap value a      (target F + b)
      S2  Phi(q')  = max flank sum over occurrences of the gap value q'     (target F)
      S3  Phi(a,b) = max flank sum over occurrences of (a,b) or (b,a)       (target F)
    with their occurrence counts, and the general J = 3 literal / padded cells,
  * the tooth separations sep_q = min(2v mod q, q - 2v mod q)  ((T) is sep_q >= 2 at every gear).
Modes:
  real                 the real machine m11..m23 (gate: F, F_2, Q*_J table vs record) + the three statements
  fam  <y..> [--procs] the full tooth-counterfactual family at level y (old teeth free, incoming tooth free)
  sub  <y> [--procs]   the (T)+(L) sub-family at level y, FULL sweep (no adjacent teeth, incoming tooth pinned)
  mech <y>             mechanism test on the pinned violators of level y (needs fam <y> rows)
  batch                fam 19 then sub 23, sequentially (background job)
Outputs: research/proof/chain_teeth_r33_<mode>_m<y>.json (rows) and the printed lines (log).
"""
import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter
from math import prod
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chain_family_r32 import (KNOWN_F, KNOWN_F2, REC_QSTAR, JMAX, gears_of, next_prime,  # noqa: E402
                              real_tooth, open_mask, gaps_of, letter_a, direct_F_new)


def sep_of(q, v):
    r = (2 * v) % q
    return min(r, q - r)


def qstar_table_pos(g, q1, a):
    """As chain_family_r32.qstar_table, plus the gap index i of the argmax (the run is
    gaps i-1 .. i+m, i.e. openings ops[i-1] .. ops[i+m])."""
    N = g.size
    b = q1 - a
    r = g % q1
    legal = (r == 0) | (r == a) | (r == b)
    s = np.zeros(N, dtype=np.int8)
    s[r == a] = 1
    s[r == b] = -1
    gE = np.concatenate([g, g[:JMAX + 3]])
    legalE = np.concatenate([legal, legal[:JMAX + 3]])
    sE = np.concatenate([s, s[:JMAX + 3]])
    gprev = np.roll(g, 1)
    F = int(g.max())
    F2 = int((g + np.roll(g, -1)).max())
    out = {}
    mask = legal.copy()
    lnz = s.copy()
    haspad = (s == 0) & legal
    csum = g.copy()
    m = 1
    while mask.any() and m <= JMAX:
        idx = np.flatnonzero(mask)
        span = gprev[idx] + csum[idx] + gE[idx + m]
        lit = ~haspad[idx]
        k = int(span.argmax())
        i = idx[k]
        out[m + 2] = dict(Q=int(span[k]), word=tuple(int(x) for x in gE[i:i + m]),
                          gL=int(gprev[i]), gR=int(gE[i + m]), literal=bool(lit[k]), i=int(i),
                          lit=int(span[lit].max()) if lit.any() else None,
                          pad=int(span[~lit].max()) if (~lit).any() else None, n=int(idx.size))
        nxt = idx + m
        ok = legalE[nxt] & ~((sE[nxt] != 0) & (lnz[idx] == sE[nxt]))
        sel = idx[ok]
        newl = sE[sel + m]
        lnz[sel] = np.where(newl != 0, newl, lnz[sel])
        haspad[sel] |= (newl == 0)
        csum[sel] += gE[sel + m]
        mask = np.zeros(N, dtype=bool)
        mask[sel] = True
        m += 1
    return F, F2, out


def three_statements(g, q1, a):
    """S1, S2, S3 flank envelopes with occurrence counts, plus the general J=3 literal/padded cells."""
    b = q1 - a
    gp = np.roll(g, 1)
    gn = np.roll(g, -1)
    gnn = np.roll(g, -2)
    fl = gp + gn
    res = {}
    for key, sel in (("A", g == a), ("Q", g == q1), ("B", g == b)):
        n = int(sel.sum())
        res["Phi" + key] = int(fl[sel].max()) if n else None
        res["n" + key] = n
    sel = ((g == a) & (gn == b)) | ((g == b) & (gn == a))
    n = int(sel.sum())
    res["PhiAB"] = int((gp + gnn)[sel].max()) if n else None
    res["nAB"] = n
    r = g % q1
    litsel = (r == a) | (r == b)
    padsel = (r == 0)
    res["Q3lit"] = int((fl + g)[litsel].max()) if litsel.any() else None
    res["Q3pad"] = int((fl + g)[padsel].max()) if padsel.any() else None
    return res


def row_of(y, teeth, g, q1, v1):
    a = letter_a(q1, v1)
    F, F2, tab = qstar_table_pos(g, q1, a)
    chain = max(r['Q'] for r in tab.values()) if tab else None
    row = dict(teeth=list(teeth), v1=v1, a=a, F=F, F2=F2, chain=chain, L=(max(tab) - 2 if tab else 0),
               viol={str(J): r['Q'] - F - q1 for J, r in tab.items() if r['Q'] > F + q1},
               argmax={str(J): [list(r['word']), r['gL'], r['gR'], bool(r['literal']), r['i']] for J, r in tab.items()},
               pair_ok=bool(F2 <= F + q1))
    row.update(three_statements(g, q1, a))
    return row


def member_rows(args):
    y, teeth, vlist = args
    gears = gears_of(y)
    q1 = next_prime(y)
    g = gaps_of(open_mask(gears, teeth, prod(gears)))
    return [row_of(y, teeth, g, q1, v1) for v1 in vlist]


def margins(row, q1):
    """the three statements' margins (None if vacuous)."""
    F, a = row['F'], row['a']
    b = q1 - a
    m1 = (F + b - row['PhiA']) if row['PhiA'] is not None else None
    m2 = (F - row['PhiQ']) if row['PhiQ'] is not None else None
    m3 = (F - row['PhiAB']) if row['PhiAB'] is not None else None
    return m1, m2, m3


def is_T(gears, teeth):
    return all(sep_of(q, v) >= 2 for q, v in zip(gears, teeth))


def report(rows, y, tag):
    gears = gears_of(y)
    q1 = next_prime(y)
    vp = real_tooth(q1)
    n = len(rows)
    viol = [r for r in rows if r['viol']]
    print(f"\n  == {tag}: {n} rows ==  chain violators {len(viol)} ({100 * len(viol) / max(n, 1):.2f}%);"
          f" pair violators {sum(1 for r in rows if not r['pair_ok'])}")
    if viol:
        exc = max(max(r['viol'].values()) for r in viol)
        print(f"  max excess {exc}; J cells {dict(sorted(Counter(J for r in viol for J in r['viol']).items()))}")
    # the three statements
    for key, name in (("m1", "S1 Phi(a) <= F+b"), ("m2", "S2 Phi(q') <= F"), ("m3", "S3 Phi(a,b) <= F")):
        ms = [(margins(r, q1)[int(key[1]) - 1], r) for r in rows]
        vac = sum(1 for m, _ in ms if m is None)
        val = [(m, r) for m, r in ms if m is not None]
        if not val:
            print(f"  {name}: vacuous at all {n} rows")
            continue
        fails = [(m, r) for m, r in val if m < 0]
        mn = min(val, key=lambda t: t[0])
        print(f"  {name}: vacuous {vac}, evaluated {len(val)}, FAILS {len(fails)}, min margin {mn[0]}"
              f" at teeth {tuple(mn[1]['teeth'])} v'={mn[1]['v1']} a={mn[1]['a']} F={mn[1]['F']}")
        if fails:
            fT = [r for m, r in fails if is_T(gears, r['teeth'])]
            fL = [r for m, r in fails if r['v1'] == vp]
            fTL = [r for m, r in fails if is_T(gears, r['teeth']) and r['v1'] == vp]
            print(f"      failing rows with (T): {len(fT)}; with (L): {len(fL)}; with both: {len(fTL)};"
                  f" a-distribution {dict(sorted(Counter(r['a'] for m, r in fails).items()))}")
    # which statement is tight, per row
    tight = Counter()
    for r in rows:
        ms = margins(r, q1)
        cand = [(m, k) for m, k in zip(ms, ("S1", "S2", "S3")) if m is not None]
        if cand:
            tight[min(cand)[1]] += 1
    print(f"  tight statement per row (smallest margin): {dict(tight)}")
    # margin vs min separation (PC7)
    if any(r['v1'] == vp for r in rows):
        pin = [r for r in rows if r['v1'] == vp and r['chain'] is not None]
        lo = [r['F'] + q1 - r['chain'] for r in pin if min(sep_of(q, v) / q for q, v in zip(gears, r['teeth'])) < 0.15]
        hi = [r['F'] + q1 - r['chain'] for r in pin if min(sep_of(q, v) / q for q, v in zip(gears, r['teeth'])) >= 0.25]
        if lo and hi:
            print(f"  PC7 pinned rows: mean chain margin, min sep/q < 0.15: {np.mean(lo):.2f} (n={len(lo)});"
                  f" >= 0.25: {np.mean(hi):.2f} (n={len(hi)})")
    sys.stdout.flush()


def save(rows, path):
    with open(path, "w") as fh:
        json.dump(rows, fh)
    print(f"  {len(rows)} rows written to {path}")


# ---------------------------------------------------------------- modes
def run_real():
    print("REAL MACHINE (round(q/6) teeth), pinned incoming tooth: gates + the three statements")
    for y in [11, 13, 17, 19, 23]:
        gears = gears_of(y)
        teeth = [real_tooth(q) for q in gears]
        q1 = next_prime(y)
        v1 = real_tooth(q1)
        t0 = time.time()
        g = gaps_of(open_mask(gears, teeth, prod(gears)))
        row = row_of(y, teeth, g, q1, v1)
        F, F2, a = row['F'], row['F2'], row['a']
        assert F == KNOWN_F[y] and F2 == KNOWN_F2[y]
        tab = {2: F2}
        tab.update({int(J): F + q1 + e for J, e in row['viol'].items()})
        got = {2: F2}
        got.update({int(J): F + q1 - (F + q1) + 0 for J in []})
        qs = {int(J): int(row['argmax'][J][1] + sum(row['argmax'][J][0]) + row['argmax'][J][2]) for J in row['argmax']}
        got.update(qs)
        ok = got == REC_QSTAR[y]
        m1, m2, m3 = margins(row, q1)
        print(f"  m{y} q'={q1} a={a} b={q1 - a} F={F} F_2={F2} budget {F + q1}: Q*_J {got} GATE G1 {'OK' if ok else 'MISMATCH'};"
              f" chain margin {F + q1 - row['chain']}  ({time.time() - t0:.1f}s)")
        print(f"      S1 Phi(a)={row['PhiA']} (n={row['nA']}) margin {m1};  S2 Phi(q')={row['PhiQ']} (n={row['nQ']}) margin {m2};"
              f"  S3 Phi(a,b)={row['PhiAB']} (n={row['nAB']}) margin {m3};  Phi(b)={row['PhiB']} (n={row['nB']});"
              f" Q3lit={row['Q3lit']} Q3pad={row['Q3pad']}; seps {[sep_of(q, v) for q, v in zip(gears, teeth)]}")
        sys.stdout.flush()


def run_fam(y, procs):
    gears = gears_of(y)
    q1 = next_prime(y)
    vfree = list(range(1, (q1 - 1) // 2 + 1))
    members = list(itertools.product(*[range(1, (q - 1) // 2 + 1) for q in gears]))
    print(f"\nFULL FAMILY m{y}: gears {gears}, q'={q1}, pinned v'={real_tooth(q1)} (a={letter_a(q1, real_tooth(q1))}),"
          f" {len(members)} members x {len(vfree)} incoming teeth = {len(members) * len(vfree)} rows, procs {procs}")
    t0 = time.time()
    tasks = [(y, m, vfree) for m in members]
    if procs > 1:
        with Pool(procs) as pool:
            res = pool.map(member_rows, tasks, chunksize=max(1, len(tasks) // (procs * 8)))
    else:
        res = [member_rows(t) for t in tasks]
    rows = [r for rr in res for r in rr]
    print(f"  computed {len(rows)} rows in {time.time() - t0:.0f}s")
    save(rows, os.path.join(HERE, f"chain_teeth_r33_fam_m{y}.json"))
    vp = real_tooth(q1)
    report(rows, y, "FREE incoming tooth, all old teeth")
    report([r for r in rows if r['v1'] == vp], y, "PINNED (L) only")
    report([r for r in rows if is_T(gears, r['teeth'])], y, "(T) only, incoming tooth free")
    report([r for r in rows if is_T(gears, r['teeth']) and r['v1'] == vp], y, "(T)+(L) sub-family")
    return rows


def run_sub(y, procs):
    gears = gears_of(y)
    q1 = next_prime(y)
    vp = real_tooth(q1)
    members = list(itertools.product(*[range(1, (q - 1) // 2) for q in gears]))   # excludes (q-1)/2: (T)
    print(f"\n(T)+(L) SUB-FAMILY m{y}, FULL SWEEP: gears {gears}, q'={q1}, v'={vp} (a={letter_a(q1, vp)}),"
          f" {len(members)} members, procs {procs}")
    t0 = time.time()
    tasks = [(y, m, [vp]) for m in members]
    rows = []
    nv = 0
    with Pool(procs) as pool:
        for k, rr in enumerate(pool.imap_unordered(member_rows, tasks, chunksize=2), 1):
            for r in rr:
                rows.append(r)
                if r['viol']:
                    nv += 1
                    J = max(r['viol'], key=r['viol'].get)
                    w, gL, gR, lit, i = r['argmax'][J]
                    print(f"  VIOLATOR #{nv} teeth {tuple(r['teeth'])} seps {[sep_of(q, v) for q, v in zip(gears, r['teeth'])]}"
                          f" F={r['F']} F_2={r['F2']} J={J} span {r['F'] + q1 + r['viol'][J]} = ({gL}) + {w} + ({gR})"
                          f" {'literal' if lit else 'padded'}; excess {r['viol'][J]}")
                    sys.stdout.flush()
            if k % 500 == 0:
                print(f"  progress {k}/{len(tasks)} members, {time.time() - t0:.0f}s, violators so far {nv}")
                sys.stdout.flush()
    print(f"  computed {len(rows)} rows in {time.time() - t0:.0f}s")
    save(rows, os.path.join(HERE, f"chain_teeth_r33_sub_m{y}.json"))
    report(rows, y, f"(T)+(L) sub-family m{y}, full sweep")
    marg = sorted(r['F'] + q1 - r['chain'] for r in rows if r['chain'] is not None)
    print(f"  chain margin: min {marg[0]}, 1st pct {marg[len(marg) // 100]}, median {marg[len(marg) // 2]}, max {marg[-1]};"
          f" L distribution {dict(sorted(Counter(r['L'] for r in rows).items()))}")
    # margin by min separation over gears >= 11
    by = {}
    for r in rows:
        if r['chain'] is None:
            continue
        s = min(sep_of(q, v) for q, v in zip(gears, r['teeth']) if q >= 11)
        by.setdefault(s, []).append(r['F'] + q1 - r['chain'])
    print("  chain margin by min separation over gears >= 11: " +
          "; ".join(f"sep {s}: n={len(v)} min {min(v)} mean {np.mean(v):.1f}" for s, v in sorted(by.items())))
    real = tuple(real_tooth(q) for q in gears)
    rr = [r for r in rows if tuple(r['teeth']) == real]
    if rr:
        r = rr[0]
        print(f"  real machine row: F={r['F']} F_2={r['F2']} chain {r['chain']} margin {r['F'] + q1 - r['chain']}")
    sys.stdout.flush()


def run_mech(y):
    """PC6: in each pinned violator, does the degenerate gear strike two consecutive columns inside
    the violating stretch?  Also: sole-coverer counts of every gear inside the stretch."""
    gears = gears_of(y)
    q1 = next_prime(y)
    vp = real_tooth(q1)
    rows = json.load(open(os.path.join(HERE, f"chain_teeth_r33_fam_m{y}.json")))
    viol = [r for r in rows if r['viol'] and r['v1'] == vp]
    print(f"\nMECHANISM at m{y}: {len(viol)} pinned violators")
    P = prod(gears)
    hit_adj = 0
    sole_deg = []
    sole_tab = Counter()
    for r in viol:
        teeth = r['teeth']
        deg = [q for q, v in zip(gears, teeth) if sep_of(q, v) == 1]
        mk = open_mask(gears, teeth, P)
        ops = np.flatnonzero(mk)
        J = max(r['viol'], key=r['viol'].get)
        w, gL, gR, lit, i = r['argmax'][J]
        m = len(w)
        x0 = int(ops[i - 1])          # opening before g_L
        xe = int(ops[(i + m + 1) % ops.size])   # opening after g_R (gap i+m is g_R, from ops[i+m] to ops[i+m+1])
        span = (xe - x0) % P
        assert span == gL + sum(w) + gR, (span, gL, w, gR)
        cols = [(x0 + t) % P for t in range(1, span)]      # interior columns
        # per gear strikes and sole coverage inside the stretch
        strikes = {q: np.array([(c % q) in (v % q, (-v) % q) for c in cols]) for q, v in zip(gears, teeth)}
        cover = sum(s.astype(int) for s in strikes.values())
        interior_open = [c for c, cv in zip(cols, cover) if cv == 0]
        sole = {q: int(((cover == 1) & strikes[q]).sum()) for q in gears}
        adj = any(strikes[q][t] and strikes[q][t + 1] for q in deg for t in range(len(cols) - 1))
        hit_adj += adj
        sole_deg.append(sum(sole[q] for q in deg))
        for q in gears:
            sole_tab[q] += sole[q]
        print(f"  teeth {tuple(teeth)} degenerate {deg} J={J} ({gL}) + {w} + ({gR}) = {span}"
              f" F={r['F']} budget {r['F'] + q1}; interior openings {len(interior_open)} (expected {m + 1});"
              f" degenerate gear strikes an adjacent pair inside: {adj}; sole counts {sole}")
    print(f"  PC6: {hit_adj} of {len(viol)} pinned violators have the degenerate gear striking two consecutive columns inside the stretch")
    print(f"  sole-coverer columns summed over violators, by gear: {dict(sole_tab)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "fam", "sub", "mech", "batch"])
    ap.add_argument("levels", nargs="*", type=int)
    ap.add_argument("--procs", type=int, default=4)
    A = ap.parse_args()
    if A.mode == "real":
        run_real()
    elif A.mode == "fam":
        for y in A.levels:
            run_fam(y, A.procs)
    elif A.mode == "sub":
        for y in A.levels:
            run_sub(y, A.procs)
    elif A.mode == "mech":
        for y in A.levels:
            run_mech(y)
    elif A.mode == "batch":
        run_fam(19, A.procs)
        run_sub(23, A.procs)
