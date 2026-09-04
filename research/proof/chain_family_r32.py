"""Prover B, round 32 -- the chain statement Q*_J(M) <= F(M) + q' on the real machine and on
the tooth-counterfactual family.

Everything is computed on ONE lower period, directly from the cyclic gap sequence:
  F      = max gap,  F_2 = max sum of two adjacent gaps (cyclic),
  Q*_J   = max over realised legal words w of length J-2 of  g_before + span(w) + g_after,
           a legal word being consecutive gaps all in {0, +a, -a} mod q' with the nonzero
           classes strictly alternating (padded letters transparent) -- T2 + T3.
Gates:
  G1  the recorded Q*_J table at m11..m23 (R68/R81, evenj_r29.REC_QSTAR) reproduced cell for cell;
  G2  max(F_2, max_J Q*_J) == F(M+q') by a DIRECT sieve of M+q' at m11..m19 (real teeth), and at
      every member of the m11 and m13 families (free incoming tooth) -- the attainment identity.
Family: old gears' teeth at +-v_q, v_q in 1..(q-1)/2 (the record's V(y)); incoming tooth either
FREE (v' in 1..(q'-1)/2) or PINNED to round(q'/6) (so a = 2u', 3a = q' -+ 1 hold exactly).

Usage:  uv run python research/proof/chain_family_r32.py real            # gates + real rows
        uv run python research/proof/chain_family_r32.py family 11 13 17 # family levels
        uv run python research/proof/chain_family_r32.py family 19 --procs 4
        uv run python research/proof/chain_family_r32.py realfree         # real M, free incoming tooth
"""
import argparse
import itertools
import sys
import time
from collections import Counter
from math import prod
from multiprocessing import Pool

import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
REC_QSTAR = {11: {2: 11, 3: 8}, 13: {2: 16, 3: 18}, 17: {2: 25, 3: 25},
             19: {2: 31, 3: 33, 4: 34}, 23: {2: 39, 3: 43}}
KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68}
JMAX = 24


def gears_of(y):
    return [p for p in PRIMES if p <= y]


def next_prime(y):
    return [p for p in PRIMES if p > y][0]


def real_tooth(q):
    u = round(q / 6)
    assert (6 * u) % q in (1, q - 1)
    return u


def open_mask(gears, teeth, P):
    m = np.ones(P, dtype=bool)
    for q, v in zip(gears, teeth):
        m[v % q::q] = False
        m[(-v) % q::q] = False
    return m


def gaps_of(mask):
    ops = np.flatnonzero(mask)
    g = np.diff(ops)
    wrap = mask.size - ops[-1] + ops[0]
    return np.append(g, wrap).astype(np.int64)      # cyclic gap sequence, length N = prod(q-2)


def letter_a(q1, v1):
    """smallest positive representative of the two literal classes +-2v' mod q'."""
    r = (2 * v1) % q1
    return min(r, q1 - r)


def qstar_table(g, q1, a):
    """F, F_2 and, for every J >= 3 with a realised legal word of length J-2, the record
    Q*_J with its argmax, plus the literal-only and padded-only maxima."""
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
        rec = dict(Q=int(span[k]), word=tuple(int(x) for x in gE[i:i + m]),
                   gL=int(gprev[i]), gR=int(gE[i + m]), literal=bool(lit[k]),
                   lit=int(span[lit].max()) if lit.any() else None,
                   pad=int(span[~lit].max()) if (~lit).any() else None,
                   n=int(idx.size))
        out[m + 2] = rec
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


def direct_F_new(gears, teeth, q1, v1):
    P = prod(gears) * q1
    mk = open_mask(gears + [q1], list(teeth) + [v1], P)
    return int(gaps_of(mk).max())


def summarize(F, F2, tab, q1):
    chain = max((r['Q'] for r in tab.values()), default=None)
    litmax = max((r['lit'] for r in tab.values() if r['lit'] is not None), default=None)
    padmax = max((r['pad'] for r in tab.values() if r['pad'] is not None), default=None)
    L = max(tab) - 2 if tab else 0
    viol = {J: r['Q'] - F - q1 for J, r in tab.items() if r['Q'] > F + q1}
    return dict(F=F, F2=F2, chain=chain, litmax=litmax, padmax=padmax, L=L, viol=viol,
                pair_ok=F2 <= F + q1,
                delta={J: r['Q'] - F2 for J, r in tab.items()},
                argmax={J: (r['word'], r['gL'], r['gR'], r['literal']) for J, r in tab.items()})


# ---------------------------------------------------------------- real machine
def run_real():
    print("REAL MACHINE, real teeth (round(q/6) at every gear)")
    for y in [11, 13, 17, 19, 23]:
        gears = gears_of(y)
        teeth = [real_tooth(q) for q in gears]
        q1 = next_prime(y)
        v1 = real_tooth(q1)
        a = letter_a(q1, v1)
        P = prod(gears)
        t0 = time.time()
        g = gaps_of(open_mask(gears, teeth, P))
        assert g.size == prod(q - 2 for q in gears) and int(g.sum()) == P
        F, F2, tab = qstar_table(g, q1, a)
        assert F == KNOWN_F[y] and F2 == KNOWN_F2[y], (y, F, F2)
        row = {2: F2}
        row.update({J: r['Q'] for J, r in tab.items()})
        rec = REC_QSTAR[y]
        ok = all(row.get(J) == rec[J] for J in rec) and all(J in rec for J in row)
        print(f"  m{y:<3} q'={q1:<3} a={a:<3} F={F:<3} F_2={F2:<3} Q*_J:",
              " ".join(f"J{J}={row[J]}" for J in sorted(row)),
              f"| recorded {rec} | GATE G1 {'OK' if ok else 'MISMATCH'}"
              f"  ({time.time() - t0:.1f}s)")
        for J in sorted(tab):
            r = tab[J]
            print(f"      J={J}: Q*={r['Q']}  argmax flanks ({r['gL']},{r['gR']}) word {r['word']}"
                  f" {'literal' if r['literal'] else 'padded'}; literal max {r['lit']}, padded max {r['pad']};"
                  f" {r['n']} realised words of length {J - 2}; Q*_J - (F+q') = {r['Q'] - F - q1}")
        if y <= 19:
            Fn = direct_F_new(gears, teeth, q1, v1)
            att = max([F2] + [r['Q'] for r in tab.values()])
            print(f"      GATE G2 direct sieve F(M+q') = {Fn}; max(F_2, max_J Q*_J) = {att};"
                  f" {'OK' if Fn == att else 'MISMATCH'}; chain statement margin F+q'-max_J Q*_J ="
                  f" {F + q1 - max(r['Q'] for r in tab.values())}")
        sys.stdout.flush()


# ---------------------------------------------------------------- family
def member_rows(args):
    y, teeth, vlist, direct = args
    gears = gears_of(y)
    q1 = next_prime(y)
    P = prod(gears)
    g = gaps_of(open_mask(gears, teeth, P))
    rows = []
    for v1 in vlist:
        a = letter_a(q1, v1)
        F, F2, tab = qstar_table(g, q1, a)
        sm = summarize(F, F2, tab, q1)
        sm['teeth'] = tuple(teeth)
        sm['v1'] = v1
        sm['a'] = a
        if direct:
            Fn = direct_F_new(gears, teeth, q1, v1)
            att = max([F2] + [r['Q'] for r in tab.values()])
            sm['G2'] = (Fn == att)
            sm['Fnew'] = Fn
        rows.append(sm)
    return rows


def run_family(y, procs, direct):
    gears = gears_of(y)
    q1 = next_prime(y)
    vp = real_tooth(q1)
    vfree = list(range(1, (q1 - 1) // 2 + 1))
    members = list(itertools.product(*[range(1, (q - 1) // 2 + 1) for q in gears]))
    print(f"\nFAMILY at m{y}: gears {gears}, q'={q1}, pinned tooth {vp} (a={letter_a(q1, vp)}),"
          f" {len(members)} old-teeth members x {len(vfree)} incoming teeth = {len(members) * len(vfree)} rows"
          f"{' [direct-sieve gate G2 on every row]' if direct else ''}")
    t0 = time.time()
    tasks = [(y, m, vfree, direct) for m in members]
    if procs > 1:
        with Pool(procs) as pool:
            res = pool.map(member_rows, tasks, chunksize=max(1, len(tasks) // (procs * 8)))
    else:
        res = [member_rows(t) for t in tasks]
    rows = [r for rr in res for r in rr]
    print(f"  computed {len(rows)} rows in {time.time() - t0:.0f}s")
    import json
    import os
    vrows = [dict(teeth=r['teeth'], v1=r['v1'], a=r['a'], F=r['F'], F2=r['F2'], L=r['L'],
                  pair_ok=bool(r['pair_ok']), viol={str(J): e for J, e in r['viol'].items()},
                  argmax={str(J): [list(w), gL, gR, bool(lit)] for J, (w, gL, gR, lit) in r['argmax'].items()})
             for r in rows if r['viol']]
    vpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"chain_family_r32_viol_m{y}.json")
    with open(vpath, "w") as fh:
        json.dump(vrows, fh)
    print(f"  {len(vrows)} violating rows written to {vpath}")
    if direct:
        bad = [r for r in rows if not r['G2']]
        print(f"  GATE G2 (attainment identity by direct sieve): {len(rows) - len(bad)}/{len(rows)} OK,"
              f" {len(bad)} mismatches")
    real_teeth = tuple(real_tooth(q) for q in gears)
    for mode, sel in [("FREE incoming tooth", rows), ("PINNED incoming tooth", [r for r in rows if r['v1'] == vp])]:
        n = len(sel)
        viol = [r for r in sel if r['viol']]
        pairbad = [r for r in sel if not r['pair_ok']]
        both = [r for r in viol if not r['pair_ok']]
        print(f"\n  == {mode}: {n} rows ==")
        print(f"  chain violators (some J>=3 with Q*_J > F+q'): {len(viol)}  ({100 * len(viol) / n:.2f}%)")
        print(f"  pair violators (F_2 > F+q'): {len(pairbad)}; chain violators with the pair statement HOLDING:"
              f" {len(viol) - len(both)} of {len(viol)}")
        if viol:
            exc = max(max(r['viol'].values()) for r in viol)
            worst = max(viol, key=lambda r: max(r['viol'].values()))
            Jc = Counter(J for r in viol for J in r['viol'])
            litc = sum(1 for r in viol if r['litmax'] is not None and r['litmax'] > r['F'] + q1)
            padc = sum(1 for r in viol if r['padmax'] is not None and r['padmax'] > r['F'] + q1)
            print(f"  max excess Q*_J - F - q' = {exc}; J-distribution of violating cells {dict(sorted(Jc.items()))};"
                  f" violators whose LITERAL max violates: {litc}; whose PADDED max violates: {padc}")
            Jw = max(worst['viol'], key=worst['viol'].get)
            w, gL, gR, lit = worst['argmax'][Jw]
            print(f"  worst: old teeth {worst['teeth']} v'={worst['v1']} a={worst['a']} F={worst['F']} F_2={worst['F2']}"
                  f" J={Jw} Q*={worst['F'] + q1 + worst['viol'][Jw]} = ({gL}) + {w} + ({gR}) {'literal' if lit else 'padded'}")
            # the smallest-F violator and a literal one, for the record
            for tag, cand in [("min-flank <= a literal J=3 violator",
                               [r for r in viol if 3 in r['viol'] and r['argmax'][3][3]
                                and min(r['argmax'][3][1], r['argmax'][3][2]) <= r['a']]),
                              ("literal J=4 violator", [r for r in viol if 4 in r['viol'] and r['argmax'][4][3]]),
                              ("padded J=3 violator", [r for r in viol if 3 in r['viol'] and not r['argmax'][3][3]])]:
                if cand:
                    r = cand[0]
                    Jw = 3 if 3 in r['viol'] and "J=3" in tag else 4
                    w, gL, gR, lit = r['argmax'][Jw]
                    print(f"  example {tag}: teeth {r['teeth']} v'={r['v1']} a={r['a']} F={r['F']} F_2={r['F2']}"
                          f" Q*_{Jw}={r['F'] + q1 + r['viol'][Jw]} = ({gL}) + {w} + ({gR}); pair_ok={r['pair_ok']}")
                else:
                    print(f"  example {tag}: none in this family")
        Ls = Counter(r['L'] for r in sel)
        print(f"  L distribution {dict(sorted(Ls.items()))}")
        # par trading along the maximising chain and Delta_J
        dmax = max((d for r in sel for d in r['delta'].values()), default=None)
        dmin = min((d for r in sel for d in r['delta'].values()), default=None)
        eps = [r['delta'][J] - r['delta'][J - 1] for r in sel for J in r['delta'] if J - 1 in r['delta']]
        eps3 = [r['delta'][3] for r in sel if 3 in r['delta']]
        print(f"  Delta_J = Q*_J - F_2 over the family: min {dmin}, max {dmax}; Delta_3 range"
              f" [{min(eps3) if eps3 else None}, {max(eps3) if eps3 else None}];"
              f" chain step Q*_J - Q*_(J-1) (= -eps): min {min(eps) if eps else None}, max {max(eps) if eps else None}")
        realrow = [r for r in sel if r['teeth'] == real_teeth and r['v1'] == vp]
        if realrow:
            r = realrow[0]
            print(f"  real machine row: F={r['F']} F_2={r['F2']} chain max {r['chain']} margin {r['F'] + q1 - r['chain']}"
                  f" viol={r['viol']} L={r['L']}")
        sys.stdout.flush()
    return rows


def run_realfree():
    print("\nREAL OLD GEARS, FREE INCOMING TOOTH v' (wrong incoming letters)")
    for y in [11, 13, 17, 19, 23]:
        gears = gears_of(y)
        teeth = [real_tooth(q) for q in gears]
        q1 = next_prime(y)
        g = gaps_of(open_mask(gears, teeth, prod(gears)))
        out = []
        for v1 in range(1, (q1 - 1) // 2 + 1):
            a = letter_a(q1, v1)
            F, F2, tab = qstar_table(g, q1, a)
            sm = summarize(F, F2, tab, q1)
            out.append((v1, a, sm['chain'], sm['chain'] - F - q1 if sm['chain'] else None, sm['L'], sm['viol']))
        print(f"  m{y} q'={q1} F={F} F_2={F2} budget F+q'={F + q1}: "
              + "; ".join(f"v'={v1}(a={a}): max_J Q*_J={c} ({'+' if e and e > 0 else ''}{e}) L={L}{' VIOL' if vi else ''}"
                          for v1, a, c, e, L, vi in out))
        sys.stdout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "family", "realfree", "ndpinned"])
    ap.add_argument("levels", nargs="*", type=int)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--direct", action="store_true")
    A = ap.parse_args() if sys.argv[1] != "ndpinned" else None
    if A is None:
        pass
    elif A.mode == "real":
        run_real()
    elif A.mode == "realfree":
        run_realfree()
    else:
        for y in A.levels:
            run_family(y, A.procs, A.direct)


# ---------------------------------------------------------------- non-degenerate pinned sample
def run_ndpinned(y, nsample, procs, seed=32):
    """Sample of the sub-family with NO adjacent teeth (v_q != (q-1)/2) and the incoming tooth
    PINNED, at a level whose full sweep is out of budget."""
    import json
    import os
    import random
    gears = gears_of(y)
    q1 = next_prime(y)
    vp = real_tooth(q1)
    rng = random.Random(seed)
    members = set()
    while len(members) < nsample:
        members.add(tuple(rng.randrange(1, (q - 1) // 2) for q in gears))   # excludes (q-1)/2
    members = sorted(members)
    print(f"\nNON-DEGENERATE PINNED SAMPLE at m{y}: {len(members)} members (seed {seed}), q'={q1}, v'={vp}")
    t0 = time.time()
    tasks = [(y, m, [vp], False) for m in members]
    with Pool(procs) as pool:
        res = pool.map(member_rows, tasks, chunksize=4)
    rows = [r for rr in res for r in rr]
    viol = [r for r in rows if r['viol']]
    pairbad = [r for r in rows if not r['pair_ok']]
    print(f"  computed {len(rows)} rows in {time.time() - t0:.0f}s; chain violators {len(viol)}; pair violators {len(pairbad)}")
    marg = [r['F'] + q1 - r['chain'] for r in rows if r['chain'] is not None]
    print(f"  chain margin F+q'-max_J Q*_J: min {min(marg)}, median {sorted(marg)[len(marg) // 2]}, max {max(marg)};"
          f" L distribution {dict(sorted(Counter(r['L'] for r in rows).items()))}")
    for r in viol[:5]:
        J = max(r['viol'], key=r['viol'].get)
        w, gL, gR, lit = r['argmax'][J]
        print(f"  VIOLATOR teeth {r['teeth']} F={r['F']} F_2={r['F2']} J={J} span {r['F'] + q1 + r['viol'][J]} = ({gL}) + {w} + ({gR})")
    vpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"chain_family_r32_ndpinned_m{y}.json")
    with open(vpath, "w") as fh:
        json.dump([dict(teeth=r['teeth'], F=r['F'], F2=r['F2'], chain=r['chain'], L=r['L'], pair_ok=bool(r['pair_ok']),
                        viol={str(J): e for J, e in r['viol'].items()}) for r in rows], fh)
    print(f"  rows written to {vpath}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ndpinned":
    y = int(sys.argv[2]); n = int(sys.argv[3]); procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    run_ndpinned(y, n, procs)
