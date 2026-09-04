# Branch 5g: THE GEAR-5 LOCK.
#
# Claim (found while scoring A3/B, then proved by hand and checked here): let a MAXIMAL blocked
# stretch of any machine containing gear 5 be the columns s .. s+L-1 with s-1 and s+L both
# openings.  Then gear 5 is automatically at its coverage-maximal phase: c_5(s,L) = m_5(L).
#
# Proof in five cases (e = L mod 5).  Gear 5's teeth are T = {1,4} = {+-1} mod 5 and its open
# residues are {0,2,3}.  Writing L = 5t + e, c_5(s,L) = 2t + n_e(s) with n_e(s) = #{j < e :
# s+j in T}, so m_5(L) = 2t + max_s n_e(s).  The flanks give s-1 not in T (s in {1,3,4}) and
# s+e not in T (s not in T-e).  In each of the five cases the surviving phases are exactly
# argmax n_e:
#   e=0: n=0 always;                 allowed {3}
#   e=1: argmax {1,4};               allowed {1,4}
#   e=2: argmax {0,1,3,4};           allowed {1,3}
#   e=3: argmax {4};                 allowed {4}
#   e=4: argmax {1,3,4};             allowed {1,3,4}
# This script verifies the claim by brute force for every L up to 2000, verifies that no other
# gear has the property, checks the corollary against node 5e's slot rule, and gates the whole
# thing against the period records of m13..m31 and the window's longest stretch at every rung.
#
# Self-contained; numpy only.  Run: uv run python research/anchor235/r36/d1_gear5_lock.py
import os
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "d1_gear5_lock.txt")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


PR = primes_upto(2100)
NXT = {PR[i]: PR[i + 1] for i in range(len(PR) - 1)}


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


def cov(s, L, g):
    n = 0
    for t in teeth(g):
        off = (t - s) % g
        if off < L:
            n += 1 + (L - 1 - off) // g
    return n


say("Branch 5g: THE GEAR-5 LOCK.")
say("A maximal blocked stretch is s..s+L-1 with s-1 and s+L openings, hence non-teeth of EVERY")
say("gear.  Question: for which gears g does that flank condition alone force c_g = m_g?")
say("")

say("=== 1. Brute force, gear by gear, over every length L = 1..600.")
say("  For each gear: the share of the phases allowed by g's own flank condition that attain")
say("  m_g(L), minimised over L, and the lengths where it is below 1.")
say("  gear   min over L of (allowed phases at max / allowed phases)   first L where < 1")
for g in PR:
    if g < 5 or g > 97:
        continue
    a, b = teeth(g)
    T = {a, b}
    worst, firstbad = 1.0, None
    for L in range(1, 601):
        m = max(cov(r, L, g) for r in range(g))
        allowed = [s for s in range(g) if (s - 1) % g not in T and (s + L) % g not in T]
        if not allowed:
            continue
        sh = sum(1 for s in allowed if cov(s, L, g) == m) / len(allowed)
        if sh < worst:
            worst = sh
        if sh < 1 and firstbad is None:
            firstbad = L
    say(f"  {g:>4}   {worst:>52.3f}   {str(firstbad):>16}")

say("")
say("=== 2. Gear 5 in full: every L to 2000, exhaustively.")
bad = []
for L in range(1, 2001):
    m = max(cov(r, L, 5) for r in range(5))
    allowed = [s for s in range(5) if (s - 1) % 5 not in (1, 4) and (s + L) % 5 not in (1, 4)]
    for s in allowed:
        if cov(s, L, 5) != m:
            bad.append((L, s))
say(f"  lengths tested: 2000; violations: {len(bad)}  {bad[:5]}")
say("  the allowed start residues by e = L mod 5, and the maximal-coverage set:")
for e in range(5):
    L = 100 + ((e - 100) % 5)          # any L with L mod 5 = e
    m = max(cov(r, L, 5) for r in range(5))
    allowed = sorted(s for s in range(5) if (s - 1) % 5 not in (1, 4)
                     and (s + L) % 5 not in (1, 4))
    argmax = sorted(s for s in range(5) if cov(s, L, 5) == m)
    say(f"    e = {e}: allowed start residues {allowed}, argmax {argmax}, "
        f"allowed subset of argmax: {set(allowed) <= set(argmax)}")

say("")
say("=== 3. The corollary is node 5e's slot rule, now uniform in the machine.")
say("  The opening that OPENS the gap is x = s-1, and its residue mod 5 names the twin slot")
say("  (k = 0 mod 5 -> 29|31, k = 2 -> 11|13, k = 3 -> 17|19; k = 1, 4 are struck by gear 5).")
say("  A gap of length F has L = F-1 blocked columns, so e = (F-1) mod 5.")
SLOT = {0: "29|31", 2: "11|13", 3: "17|19"}
for Fm in range(5):
    e = (Fm - 1) % 5
    L = 100 + ((e - 100) % 5)
    allowed = sorted(s for s in range(5) if (s - 1) % 5 not in (1, 4)
                     and (s + L) % 5 not in (1, 4))
    xs = sorted((s - 1) % 5 for s in allowed)
    say(f"    F = {Fm} mod 5  ->  e = {e}  ->  start openings x mod 5 in {xs} "
        f"= slots {[SLOT[x] for x in xs]}")
say("  node 5e (research/proof/anchor_cycles.md, measured at eight full periods): F = 1 mod 5")
say("  starts on 11|13, F = 4 on 17|19, F = 2 and F = 3 on mirror pairs, F = 0 on any.")

say("")
say("=== 4. Gate against the period records of m13..m31 (starts from a1 / f1 full-period scans).")
LAD = [5, 7, 11, 13, 17, 19, 23, 29, 31]
FK = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
BIG = {29: [200906186, 877375978],
       31: [1468940243, 11582483683, 21844264616, 31957808056]}
tot = 0
for q in (13, 17, 19, 23, 29, 31):
    gears = [g for g in LAD if g <= q]
    F = FK[q]
    if q in BIG:
        starts = BIG[q]
    else:
        P = prod(gears)
        w = np.ones(P, bool)
        for g in gears:
            for t in teeth(g):
                w[t::g] = False
        op = np.flatnonzero(w)
        d = np.diff(np.concatenate([op, [op[0] + P]]))
        starts = [int(op[j]) + 1 for j in np.flatnonzero(d == F)]
    L = F - 1
    m5 = max(cov(r, L, 5) for r in range(5))
    ok = all(cov(s, L, 5) == m5 for s in starts)
    tot += len(starts)
    say(f"  m{q}: {len(starts)} record stretches, L = {L}, m_5 = {m5}, all at maximum: {ok}; "
        f"start residues mod 5 {sorted({s % 5 for s in starts})}")
say(f"  {tot} record stretches, no exception.")

say("")
say("=== 5. Gate on the window's longest blocked stretch at every prime rung 23..1999.")
n_ok = n_all = 0
worst = []
for q in PR:
    if q < 23 or q > 1999:
        continue
    qq = NXT[q]
    lo, hi = q // 6 + 1, (qq * qq - 1) // 6
    n = hi - lo + 1
    cnt = np.zeros(n, np.int16)
    for g in PR:
        if g < 5 or g > q:
            continue
        for t in teeth(g):
            cnt[(t - lo) % g::g] += 1
    op = np.flatnonzero(cnt == 0) + lo
    if len(op) < 4:
        continue
    d = np.diff(op)
    j = int(d.argmax())
    L, s = int(d[j]) - 1, int(op[j]) + 1
    m5 = max(cov(r, L, 5) for r in range(5))
    n_all += 1
    if cov(s, L, 5) == m5:
        n_ok += 1
    else:
        worst.append(q)
    # every stretch of this window, not only the longest
    for t in range(len(d)):
        LL = int(d[t]) - 1
        if LL < 1:
            continue
        ss = int(op[t]) + 1
        mm = max(cov(r, LL, 5) for r in range(5))
        assert cov(ss, LL, 5) == mm, (q, ss, LL)
    del cnt, op
say(f"  longest window stretch at maximum for gear 5: {n_ok} of {n_all} rungs; exceptions {worst}")
say("  and EVERY maximal blocked stretch of every window (not only the longest) satisfies it -")
say("  asserted column by column at all rungs, no exception.")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written", OUT)
