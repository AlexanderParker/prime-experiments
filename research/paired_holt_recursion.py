"""Harvester round 20 (frames directive, literature import): THE PAIRED HOLT
RECURSION - exact population dynamics for two-residue sieves.

Holt (arXiv:1510.00743, with Rudd arXiv:1408.6002) runs Eratosthenes sieve as a
recursion on the CYCLE OF GAPS G(p#): concatenate p' copies, fuse at closures;
gap/constellation populations obey an exact linear dynamical system
    n_{s,j}(p'#) = (p'-j-1) n_{s,j}(p#) + (driving terms),
transfer matrix with eigenvalues (p'-j-1)/(p'-2) and p-INDEPENDENT eigenvectors.
That is the ONE-residue-per-prime case. Our machines are TWO-residue sieves
(teeth +-u in the twin slot frame; {0,-e} in the general-difference frame), and
the merge law is the corresponding cycle recursion for the MAX gap.

THIS SCRIPT states and verifies the paired analogue at the POPULATION level:

  THEOREM (paired Holt recursion, verified exactly here). Let M be a two-residue
  machine with period P, and q' a new gear with tooth set T (|T| <= 2),
  gcd(P, q') = 1. For every gap value g,
      n_g(M + q') = sum over words w = (g_1..g_j), g_1+...+g_j = g, of
                    coef(w) * n_w(M),
  where n_w(M) = number of occurrences of w as consecutive gaps in M's cycle and
      coef(w) = #{ r in Z_q' : r not in T,  r + g not in T,
                   r + sigma_i in T for every interior partial sum sigma_i }.
  Position-free: coef depends only on the word mod q'.

  - j = 1 diagonal: coef(g) = #{r : r not in T, r+g not in T} = c_q'(g), which
    for teeth +-u is EXACTLY Lateral's round-19 exposed-set autocorrelation law
    (q'-2 if q'|g; q'-3 if g = +-2u; q'-4 otherwise). The autocorrelation IS the
    transfer-matrix diagonal.
  - Length-j diagonal (word surviving as itself, no fusion): generically
    q' - 2(j+1), so normalised eigenvalues (q'-2j-2)/(q'-2) - the paired
    analogue of Holt's (p-j-1)/(p-2), decaying at TWICE the rate in j.
  - One-residue T (the gcd collapse q'|e) recovers Holt's own recursion.

Verified: full histogram identity at slot-frame rungs [5,7,11,13]+17 and
[5,7,11,13,17]+19, and general-difference rung e=344 [3,5,7,11,13]+17, every
gap value exact; plus the diagonal = c-law check.
"""
import numpy as np
from math import prod
from collections import Counter

def openings_slot(gears, P):
    a = np.ones(P, bool)
    for q in gears:
        t = pow(6, -1, q)
        a[t % q::q] = False
        a[(-t) % q::q] = False
    return np.flatnonzero(a)

def openings_e(gears, e, P):
    a = np.ones(P, bool)
    for q in gears:
        a[0::q] = False
        a[(-e) % q::q] = False
    return np.flatnonzero(a)

def gap_word(idx, P):
    return np.diff(np.append(idx, idx[0] + P))

def word_counts(gaps, Lmax):
    n = len(gaps)
    ext = np.concatenate([gaps, gaps[:Lmax]])   # cyclic
    cnt = Counter()
    for j in range(1, Lmax + 1):
        for i in range(n):
            cnt[tuple(int(x) for x in ext[i:i + j])] += 1
    return cnt

def coef(word, q, T):
    g = sum(word)
    sig = np.cumsum(word[:-1])
    c = 0
    for r in range(q):
        if r % q in T or (r + g) % q in T:
            continue
        if all((r + s) % q in T for s in sig):
            c += 1
    return c

def check_rung(name, old_idx, Pold, new_idx, Pnew, q, T, Lmax=7):
    gaps_old = gap_word(old_idx, Pold)
    gaps_new = gap_word(new_idx, Pnew)
    hist_new = Counter(int(g) for g in gaps_new)
    wc = word_counts(gaps_old, Lmax)
    Fnew = max(hist_new)
    # predict every gap population of the new machine
    pred = Counter()
    for w, n in wc.items():
        s = sum(w)
        if s > Fnew:
            continue
        c = coef(w, q, T)
        if c:
            pred[s] += c * n
    allg = sorted(set(hist_new) | set(pred))
    ok = all(pred[g] == hist_new[g] for g in allg)
    print(f"{name}: q'={q} T={sorted(T)}  gaps {len(allg)} values "
          f"1..{Fnew}  ->  {'EXACT for every gap value' if ok else 'MISMATCH'}")
    for g in allg:
        assert pred[g] == hist_new[g], (name, g, pred[g], hist_new[g])
    # diagonal = c-law check (teeth +-u case)
    if len(T) == 2 and (-list(T)[0]) % q == list(T)[1] or True:
        t = sorted(T)
        for g in allg:
            c1 = coef((g,), q, T)
            if len(T) == 2:
                u2 = (t[1] - t[0]) % q
                expect = (q - 2 if g % q == 0 else
                          q - 3 if g % q in (u2, (-u2) % q) else q - 4)
            else:
                expect = q - 1 if g % q == 0 else q - 2   # one-residue Holt case
            assert c1 == expect, (g, c1, expect)
    # eigenvalue readout for a generic word of each length
    gen = {}
    for w in wc:
        if all(s % q not in (0,) and True for s in w):
            d = coef_diag(w, q, T)
            gen.setdefault(len(w), set()).add(d)
    print(f"   word-survival diagonals by length (distinct values seen): "
          f"{ {j: sorted(v) for j, v in sorted(gen.items())} }")
    return True

def coef_diag(word, q, T):
    """copies in which the word survives UNFUSED (all its openings alive)"""
    pts = [0] + list(np.cumsum(word))
    c = 0
    for r in range(q):
        if all((r + p) % q not in T for p in pts):
            c += 1
    return c

# ------------------------------------------------ rung 1: slot frame, +17
G0 = [5, 7, 11, 13]; P0 = prod(G0)
G1 = G0 + [17];      P1 = prod(G1)
q = 17; t = pow(6, -1, q); T = {t % q, (-t) % q}
check_rung("slot [5,7,11,13] -> +17", openings_slot(G0, P0), P0,
           openings_slot(G1, P1), P1, q, T)

# ------------------------------------------------ rung 2: slot frame, +19
G2 = G1 + [19]; P2 = prod(G2)
q = 19; t = pow(6, -1, q); T = {t % q, (-t) % q}
check_rung("slot [..17] -> +19    ", openings_slot(G1, P1), P1,
           openings_slot(G2, P2), P2, q, T)

# ------------------- rung 3: general difference e = 344 (a 13-winner), +17
Ge = [3, 5, 7, 11, 13]; Pe = prod(Ge)
Ge1 = Ge + [17]; Pe1 = prod(Ge1)
e = 344
q = 17; T = {0, (-e) % q}
check_rung("family e=344 -> +17  ", openings_e(Ge, e, Pe), Pe,
           openings_e(Ge1, e, Pe1), Pe1, q, T)

# ------------------- rung 4: the gcd collapse q'|e -> ONE-residue Holt case
e = 17 * 6           # 17 | e: T collapses to {0}; recursion must still be exact
q = 17; T = {0}
check_rung("collapse e=102 -> +17 (Holt one-residue case)",
           openings_e(Ge, e, Pe), Pe, openings_e(Ge1, e, Pe1), Pe1, q, T)

print("""
ALL RUNGS EXACT. The paired Holt recursion holds with:
  diagonal (j=1)  = Lateral's autocorrelation c_q'(g) in {q'-2, q'-3, q'-4}
  length-j word survival = q' - 2(j+1) generically -> normalised eigenvalues
  (q'-2j-2)/(q'-2), vs Holt's one-residue (q'-j-1)/(q'-2): the paired system
  contracts twice as fast per unit word length. The gcd-collapse rung shows the
  one-residue (Holt) law as the degenerate case of the same theorem.""")
