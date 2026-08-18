"""Harvester round 16: does the WHOLE tolerance route transfer to general Polignac d?

Audit of constructor's five parts, per d class (3 | e vs 3 not dividing e).
Halved coordinates throughout (round-15 reconciliation: 1 slot = 3 halved = 6 member;
in halved units a padded link costs 3q' when 3 does not divide e, q' when 3 | e).

(A) word list finite + computable from a fixed modulus alone
(B) literal span bound  (C) padded count bound (done r15)  (E) flank exclusion
plus: LATERAL'S CORRIDOR LAW analogue (adjacent padded links need r, r+c, r+2c in E).
"""
import numpy as np
from sympy import primerange
from itertools import product

M = 105

def E_of(e):
    return np.array([all(n % q not in (0, (-e) % q) for q in (3, 5, 7))
                     for n in range(M)])

def cost_pad(e, q1):
    """cheapest padded link in HALVED units"""
    return q1 if e % 3 == 0 else 3 * q1

def letters(e, q1):
    """primitive literal letters in halved units (consecutive frame-admissible kills)"""
    n = np.arange(6 * q1)
    kill = (n % q1 == 0) | (n % q1 == (-e) % q1)
    adm = (n % 3 != 0) & (n % 3 != (-e) % 3)
    idx = np.flatnonzero(kill & adm)
    return sorted(set(np.diff(idx).tolist()))

def words_compatible(e, q1, kmax=7):
    """set of compatible literal chain patterns: tuples of letters realizable in E."""
    E = E_of(e)
    L = letters(e, q1)
    out = set()
    for k in range(2, kmax + 1):
        for w in product(L, repeat=k - 1):
            # chain positions from some start r, all must be exposed
            offs = [0]
            for x in w:
                offs.append(offs[-1] + x)
            if any(np.all([E[(r + o) % M] for o in offs]) for r in range(M)):
                out.add(w)
        if not any(len(w) == k - 1 for w in out):
            break
    return out

print("=== (A) is the WORD LIST a function of q' mod 105 alone? ===")
for d in (2, 4, 6, 12, 30):
    e = d // 2
    byclass, bad, tested = {}, 0, 0
    for q1 in primerange(max(11, e + 1), 700):
        if q1 % 3 == 0 or q1 % 5 == 0 or q1 % 7 == 0:
            continue
        c = q1 % M
        w = words_compatible(e, q1, kmax=5)
        if c in byclass:
            tested += 1
            if byclass[c] != w:
                bad += 1
        byclass.setdefault(c, w)
    sizes = sorted({len(v) for v in byclass.values()})
    print(f"  d={d:>2}: {len(byclass)} classes, {tested} repeat tests, "
          f"mismatches {bad}  | word-set sizes {sizes[:6]}  -> "
          f"{'FUNCTION OF q mod 105' if bad == 0 else 'NOT a function'}")

print("\n=== (B) literal span: letters per chain and span in frame units ===")
for d, cap in ((2, 6), (4, 6), (6, 6), (10, 6), (30, 10), (210, 12)):
    e = d // 2
    q1 = 41 if e % 41 else 43
    L = letters(e, q1)
    prim = [x for x in L if x + (q1 * (1 if e % 3 == 0 else 3) - x) in
            (q1 * (1 if e % 3 == 0 else 3),)][:2]
    per = q1 if e % 3 == 0 else 3 * q1
    print(f"  d={d:>3} cap={cap:>2}: letters {L[:3]} sum to frame period {per} "
          f"(= q' in the frame's own unit); max letters = cap-1 = {cap-1}; "
          f"literal span <= {((cap-1)+1)//2} x period")

print("\n=== LATERAL'S CORRIDOR LAW: adjacent padded links need r, r+c, r+2c in E ===")
print("  (twins q'=41 must give ZERO - lateral's proved case)")
for d in (2, 4, 6, 12, 30):
    e = d // 2
    E = E_of(e)
    feas, tot, zeroq = 0, 0, []
    for q1 in primerange(max(11, e + 1), 400):
        if q1 % 3 == 0 or q1 % 5 == 0 or q1 % 7 == 0:
            continue
        c = cost_pad(e, q1) % M
        ok = any(E[r] and E[(r + c) % M] and E[(r + 2 * c) % M] for r in range(M))
        tot += 1
        feas += ok
        if not ok:
            zeroq.append(q1)
    print(f"  d={d:>2}: double-padding FEASIBLE for {feas}/{tot} probes; "
          f"IMPOSSIBLE for {tot-feas} (e.g. q' = {zeroq[:6]})")

print("\n=== (E) both-flanks-maximal, machine-free exclusion ===")
for d in (2, 4, 6, 12):
    e = d // 2
    E = E_of(e)
    forb = tot = 0
    for q1 in (17, 19, 23, 29, 31, 37, 41):
        if q1 <= e:
            continue
        L = letters(e, q1)[:2]
        for w in [(L[0],), (L[1],), (L[0], L[1]), (L[1], L[0])]:
            offs = [0]
            for x in w:
                offs.append(offs[-1] + x)
            span = offs[-1]
            for Fr in range(M):
                if e % 3 and Fr % 3:
                    continue          # flanks are gaps: divisible by 3 when 3 not | e
                tot += 1
                ok = any(E[(r - Fr) % M] and E[(r + span + Fr) % M] and
                         all(E[(r + o) % M] for o in offs) for r in range(M))
                if not ok:
                    forb += 1
    print(f"  d={d:>2}: both-flanks-maximal FORBIDDEN in {forb}/{tot} "
          f"(word, F mod 105) pairs = {100*forb/tot:.0f}%")
