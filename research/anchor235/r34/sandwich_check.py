"""Checks for branch 7a: the cycle sandwich F_c = floor((F-2)/5), the phase count of whole dead
cycles inside every record gap, the mirror pairing of record gaps, the increment scorecard,
the wall bound H_1, the closed form of the kill residues R_q, and the cycle-only chain law's
residue count.  Reads research/anchor235/r34/results/cycle_record_<q>.json.

Usage: uv run python research/anchor235/r34/sandwich_check.py
Writes research/anchor235/r34/results/sandwich_check.txt
"""
import os, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
E = (11, 13, 17, 19, 29, 31)
CORPUS_F = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
PRIMES = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]


def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def whole_cycles(start, gap):
    """Number of cycles j with 5j+2 >= start+1 and 5j+5 <= start+gap-1."""
    lo = -((-(start + 1 - 2)) // 5)          # ceil((start-1)/5)
    hi = (start + gap - 1 - 5) // 5
    return max(0, hi - lo + 1)


out = []
recs = {}
for fn in sorted(glob.glob(os.path.join(RES, "cycle_record_*.json"))):
    r = json.load(open(fn)); recs[r["qmax"]] = r

out.append("== 1. The sandwich F_c = floor((F-2)/5), exact full periods ==")
ok = True
for q in sorted(recs):
    r = recs[q]; F = r["F"]; Fc = max(r["F_c"]["max"], 0)
    pred = (F - 2) // 5
    ok &= (Fc == pred)
    out.append(f"  {{5..{q}}}: F = {F} (F mod 5 = {F % 5}), F_c = {Fc}, floor((F-2)/5) = {pred}  {'OK' if Fc == pred else 'FAIL'}; "
               f"runs attaining: {r['F_c']['n_max']}, record gaps by phase {r['record_gap_phases']}, dead cycles {r['dead_cycles']} of {r['period_cycles']} "
               f"({r['dead_cycles']/r['period_cycles']:.4f}; CRT {r['crt']['P_dead']:.4f})")
out.append(f"  sandwich holds at every computed machine: {ok}")

out.append("\n== 2. Whole dead cycles inside each record gap, by start phase (formula: phase 0 -> floor((F-1)/5), 2 -> floor((F-4)/5), 3 -> floor((F-3)/5)) ==")
for q in sorted(recs):
    r = recs[q]; F = r["F"]; P = r["period_columns"]
    starts = {s for s, L, ph in r["record_gaps"]}
    rows = []; mirror_ok = True; maxin = 0
    for s, L, ph in r["record_gaps"]:
        n = whole_cycles(s, L)
        f = {0: (F - 1) // 5, 2: (F - 4) // 5, 3: (F - 3) // 5}[ph]
        assert n == f, (q, s, L, ph, n, f)
        maxin = max(maxin, n)
        mirror_ok &= ((P - s - L) % P in starts)
        rows.append((ph, n))
    from collections import Counter
    out.append(f"  {{5..{q}}}: F = {F}: (phase, whole cycles) counts {dict(Counter(rows))}; max over record gaps = {maxin} = F_c {max(r['F_c']['max'],0)}: "
               f"{maxin == max(r['F_c']['max'],0)}; every record gap's mirror is a record gap: {mirror_ok}; "
               f"allowed phases for F mod 5 = {F % 5}: {[x for x in (0,2,3) if (x + F) % 5 in (0,2,3)]}")

out.append("\n== 3. Increment scorecard (F_c computed where a json exists, else floor((F-2)/5) from the corpus ladder) ==")
Fc = {}
for q in PRIMES:
    Fc[q] = max(recs[q]["F_c"]["max"], 0) if q in recs else (CORPUS_F[q] - 2) // 5
src = {q: ("exact" if q in recs else "ladder") for q in PRIMES}
out.append("  rung        F->F'   F'-F   F_c->F_c'  dF_c   q'/5   q'/15  dF_c<=q'/5  dF_c<=q'/15  dF_c/q'  class q' mod 30  source")
worst = 0
for a, b in zip(PRIMES, PRIMES[1:]):
    d = Fc[b] - Fc[a]; worst = max(worst, d / b)
    cls = b % 30; cls = cls if cls <= 15 else cls - 30
    out.append(f"  {a:>2}->{b:<2}   {CORPUS_F[a]:>3}->{CORPUS_F[b]:<3} {CORPUS_F[b]-CORPUS_F[a]:>4}   {Fc[a]:>2}->{Fc[b]:<2}     {d:>2}    {b/5:5.2f}  {b/15:5.2f}   {str(d <= b/5):<9}   {str(d <= b/15):<9}   {d/b:.3f}     {cls:+d}          {src[b]}")
out.append(f"  largest dF_c/q' on the record: {worst:.3f} (c = 1/5 holds everywhere; the best c on record is {worst:.3f})")
out.append("  sandwich bound on the increment: dF_c <= (F'-F+3)/5 + 1 always; with the budget F'-F <= q' this is q'/5 + 1.6, so the q'/5 line is the budget divided by 5")

out.append("\n== 4. The wall bound H_1(M) (longest run of cycles with <= 1 open slot) against F_c(M) + q'/5 ==")
for a, b in zip(PRIMES, PRIMES[1:]):
    if a in recs:
        H = recs[a]["H_1"]["max"]
        out.append(f"  M = {{5..{a}}}, q' = {b}: H_1 = {H} at {recs[a]['H_1']['runs'][:2]}; F_c(M) + q'/5 = {Fc[a] + b/5:.1f}; F_c(M+q') = {Fc[b]}; "
                   f"H_1 certifies the rung iff H_1 <= F_c + q'/5: {H <= Fc[a] + b/5}")

out.append("\n== 5. Closed form of the kill residues: R_q = -30^{-1} E (mod q) equals ((q m - 11) div 30) mod q over the six open m ==")
cls_m = {}
bad = 0; n = 0
for q in primes_upto(2000):
    if q < 7: continue
    ms = [m for m in range(1, 30) if (q * m) % 30 in (1, 11, 13, 17, 19, 29)]
    R1 = sorted(set(((q * m - 11) // 30) % q for m in ms))
    inv = pow(30, -1, q)
    R2 = sorted(set((-e * inv) % q for e in E))
    n += 1; bad += (R1 != R2)
    c = q % 30; c = c if c <= 15 else c - 30
    cls_m.setdefault(c, ms)
out.append(f"  primes 7..2000: {n} checked, {bad} mismatches; open multipliers by class: {dict(sorted(cls_m.items()))}")
out.append("  (cycle j holds a multiple of q at offset e iff 30j + e = 0 mod q iff j = -e/30 mod q; the class fixes which m gives which e, not the set)")
for q in (7, 11, 13, 17, 19, 23, 29, 31):
    inv = pow(30, -1, q)
    per = {e: (-e * inv) % q for e in E}
    out.append(f"    q = {q}: j-residue per offset e: {per}  -> R_q = {sorted(set(per.values()))}")

out.append("\n== 6. The chain law in cycle index alone: admissible residues of j2 - j1 (mod q') for two kills of q' in unknown slots ==")
diffs = sorted({a - b for a in E for b in E})
for q in PRIMES[1:]:
    inv = pow(30, -1, q)
    res = sorted({(d * inv) % q for d in diffs})
    out.append(f"  q' = {q}: {len(res)} of {q} residues admissible ({len(res)/q:.2f}); column chain law: 3 of {q} (0, +-2u')")
out.append(f"  E - E = {diffs} ({len(diffs)} values)")

out.append("\n== 7. Gear q kills at most one number per cycle for q >= 11; gear 7 double-kills at j = 2 mod 7 ==")
for q in (7, 11, 13, 17, 19, 23):
    inv = pow(30, -1, q)
    per = {}
    for e in E: per.setdefault((-e * inv) % q, []).append(e)
    dbl = {r: es for r, es in per.items() if len(es) > 1}
    out.append(f"  q = {q}: residues with two kills: {dbl}")

txt = "\n".join(out)
open(os.path.join(RES, "sandwich_check.txt"), "w").write(txt + "\n")
print(txt)
