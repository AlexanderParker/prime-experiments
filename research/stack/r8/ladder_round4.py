"""Round 4 of the twin ladder (tree node R5.f.iv): lengthening, steering, forced offsets.

Centre coordinates: s = P+1 = 6c, offset j has members s^2 + 6j -+ 1, new centre s' = s^2 + 6j.
Capture identity (lane): forcing j = 0 mod g keeps 1/(g-2) of the twin centres, so
N_L ~ T(c) / prod_{g <= y, g not in D} (g-2), |L| ~ 4c/M, first index ~ |L| / N_L.

Claim 1 (lengthening). Progression form: M <= 4 sqrt(c) -> |L| >= sqrt(c); L holds a twin centre for
every twin lower; N_L >= (1/2) T/prod(g-2); first index <= 12 |L|/N_L^pred. Sieve form: among offsets
avoiding both bad classes of every gear <= sqrt(s), the first twin centre's index is bounded (<= 40),
no trend in c.
Claim 2 (steering). Among the stretch's twin centres one has s' = 0 mod Q (Q = 35, 385, 5005) for
all P above a threshold P0(Q).
Claim 3 (forced offsets). (a) offsets j = -6t^2 -+ 2t are never twin centres (lower member =
(s-a)(s+a), a = 6t +- 1); (b) at j = +-c and +-(2c-1) the twin rate over all twins equals the
singular-series prediction (about 0.86 and 0.56 of the base rate 2 C2 / ln^2(s^2)); nothing above 3.

usage: uv run python ladder_round4.py [PMAX] [PFULL]
   PMAX: twin lowers for the isprime-based parts; PFULL: for the full-window-sieve parts.
"""
import sys, math
import numpy as np
from sympy import isprime

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6
PFULL = int(sys.argv[2]) if len(sys.argv) > 2 else 10**5
C2 = 0.6601618158

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 2)
ps = set(int(x) for x in P)
gears = [int(g) for g in P if g >= 5]
twin_lowers = [int(x) for x in P if x >= 5 and (int(x) + 2) in ps]

def window_sieve(Pm, gmax):
    """struck flags over offsets j in [-jmax, jmax] by the gears 5..gmax."""
    s = Pm + 1; c = s // 6; s2 = s * s; jmax = 2 * c - 1; n = 2 * jmax + 1
    struck = np.zeros(n, dtype=bool)
    for g in gears:
        if g > gmax: break
        u = pow(6, -1, g)
        for jr in ((u * (1 - s2)) % g, (u * (-1 - s2)) % g):
            struck[(jr + jmax) % g::g] = True
    return struck, jmax

# ---------- Claim 1, progression form (isprime; all twins to PMAX) ----------
print("Claim 1, progression form (M <= 4 sqrt c):")
fail = 0; idx_ratio = []; NL_ratio = []; idxs = []; sizes = []
full_T = {}
for Pm in twin_lowers:
    s = Pm + 1; c = s // 6; s2 = s * s; jmax = 2 * c - 1
    cap = 4 * math.sqrt(c)
    D = []; M = 1; y = 0; prodg2 = 1
    for g in gears:
        if g > Pm: break
        if (s2 + 1) % g == 0:
            if g <= 50: D.append(g)   # D gears handled by class avoidance, only small ones
            continue
        if M * g <= cap: M *= g; y = g; prodg2 *= (g - 2)
        else: break
    cands = []
    for k in range(-(jmax // M), jmax // M + 1):
        j = k * M
        ok = True
        for g in D:
            u = pow(6, -1, g)
            if j % g == 0 or j % g == (2 * u) % g: ok = False; break
        if ok: cands.append(j)
    cands.sort(key=abs)
    hits = [i for i, j in enumerate(cands) if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1)]
    sizes.append(len(cands))
    if not hits: fail += 1; continue
    idxs.append(hits[0])
    if Pm <= PFULL:
        struck, _ = window_sieve(Pm, Pm)
        T = int((~struck).sum()); full_T[Pm] = T
        NL_pred = T / prodg2
        NL_ratio.append(len(hits) / NL_pred if NL_pred else float('nan'))
        idx_ratio.append(hits[0] / (len(cands) / NL_pred) if NL_pred else float('nan'))
print(f"  twins {len(twin_lowers)}; L twin-free: {fail}; |L| min/median/max {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}; first index mean {np.mean(idxs):.1f} max {max(idxs)}")
print(f"  (P <= {PFULL}) N_L / capture prediction: min {np.nanmin(NL_ratio):.2f} mean {np.nanmean(NL_ratio):.2f}; index / (|L|/N_L^pred): max {np.nanmax(idx_ratio):.1f} mean {np.nanmean(idx_ratio):.2f}")

# ---------- Claim 1, sieve form (full sieve to sqrt s; P <= PFULL) ----------
print("Claim 1, sieve form (offsets rough to the gears <= sqrt s):")
by_dec = {}
for Pm in twin_lowers:
    if Pm > PFULL: break
    s = Pm + 1; s2 = s * s
    struck, jmax = window_sieve(Pm, math.isqrt(s))
    surv = np.nonzero(~struck)[0] - jmax
    surv = sorted(surv, key=abs)
    for i, j in enumerate(surv):
        if isprime(s2 + 6 * int(j) - 1) and isprime(s2 + 6 * int(j) + 1):
            by_dec.setdefault(int(math.log10(Pm)), []).append(i); break
    else:
        by_dec.setdefault(int(math.log10(Pm)), []).append(-1)
for d in sorted(by_dec):
    v = np.array(by_dec[d])
    print(f"  P ~ 10^{d}: twins {len(v)}, first-index mean {v[v>=0].mean():.2f}, max {v.max()}, no twin among survivors: {(v<0).sum()}")

# ---------- Claim 2, steering (full twin list; P <= PFULL) ----------
print("Claim 2, steering s' = 0 mod Q:")
for Q in (35, 385, 5005):
    last_fail = None; nfail = 0
    for Pm in twin_lowers:
        if Pm > PFULL: break
        s = Pm + 1; s2 = s * s
        struck, jmax = window_sieve(Pm, Pm)
        tw = np.nonzero(~struck)[0] - jmax
        if not any((s2 + 6 * int(j)) % Q == 0 for j in tw): last_fail = Pm; nfail += 1
    print(f"  Q = {Q}: twins without a steered successor: {nfail}; last failing P = {last_fail}")

# ---------- Claim 3, forced offsets ----------
print("Claim 3, forced offsets:")
viol = 0
for Pm in twin_lowers:
    if Pm > PFULL: break
    s = Pm + 1; s2 = s * s
    struck, jmax = window_sieve(Pm, Pm)
    tw = set((np.nonzero(~struck)[0] - jmax).tolist())
    t = 1
    while 6 * t * t + 2 * t <= jmax:
        for j in (-6 * t * t - 2 * t, -6 * t * t + 2 * t):
            if j in tw: viol += 1
        t += 1
print(f"  (a) twin centres at difference-of-squares offsets j = -6t^2 -+ 2t: {viol} (expected 0)")
for name, off in (("+c", lambda c: c), ("-c", lambda c: -c), ("+(2c-1)", lambda c: 2*c-1), ("-(2c-1)", lambda c: -(2*c-1))):
    hits = 0; expect = 0.0
    for Pm in twin_lowers:
        s = Pm + 1; c = s // 6; s2 = s * s; j = off(c)
        if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1): hits += 1
        expect += 12 * C2 / (math.log(s2) ** 2)  # twin columns per column near s^2: 6 integers per column x 2 C2 / ln^2
    print(f"  (b) offset j = {name}: twin hits {hits} of {len(twin_lowers)}; base-rate expectation {expect:.1f}; ratio {hits/expect:.2f}")
