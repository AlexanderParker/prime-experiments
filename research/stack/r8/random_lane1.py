"""Random lane, round 1 (tree node R5.f.xiii): three angles on the rung.

Angle 1 (universal clearance): for a twin centre s, s != +-1 mod g, so s^2 mod g lies in
Q_g = {x^2 : x != +-1} (size (g-1)/2); U_g = {j mod g : 1 - 6j and -1 - 6j both outside Q_g} is
the set of offset classes gear g can NEVER strike at any twin centre.  A_B = intersection over
5 <= g <= B (CRT) is a fixed class set, the same at every level.  Test: U_g nonempty for g <= 61;
verify on twin centres; payoff = rung rate inside A_13 vs overall.
Angle 2 (forest): s' has at most one parent s (a twin centre in (sqrt(s')-1, sqrt(s')+1), an
interval of length 2 with at most one multiple of 6).  Test: uniqueness; fraction with a parent by
decade; the accounting identity #{s' <= Y with a parent} = sum over s <= sqrt(Y)+1 of #{rungs of s
<= Y} at Y = 10^6.
Angle 3 (neighbours): t^2 - s^2 = (t - s)(t + s) translates the strike pattern between consecutive
twin centres; six excluded classes s != +-1, +-1 - d+, +-1 + d- (mod g) can force s mod g.  Test:
translation check; forcing rate per g; first-rung index by forced s^2 mod 7.

usage: uv run python random_lane1.py [PMAX] [PFULL]
"""
import sys, math, collections
import numpy as np
from sympy import isprime

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6
PFULL = int(sys.argv[2]) if len(sys.argv) > 2 else 10**4

def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.nonzero(s)[0]

P = primes_upto(PMAX + 2)
ps = set(int(x) for x in P)
gears = [int(g) for g in P if g >= 5]
centres = [int(x) + 1 for x in P if x >= 5 and (int(x) + 2) in ps]   # s = P + 1

# ---------------- Angle 1 ----------------
def Q(g): return {pow(x, 2, g) for x in range(g) if x % g not in (1, g - 1)}
U = {}
for g in gears:
    if g > 61: break
    q = Q(g)
    U[g] = [j for j in range(g) if (1 - 6 * j) % g not in q and (-1 - 6 * j) % g not in q]
print("Angle 1 (universal clearance): |U_g| for g = 5..61: " + ", ".join(f"{g}:{len(U[g])}" for g in U))
viol = 0; checks = 0
for s in centres:
    if s > 10**4 + 1: break
    c = s // 6; s2 = s * s; jmax = 2 * c - 1
    for g in U:
        if g > s - 2: continue
        for j in range(-jmax, jmax + 1):
            if j % g in U[g]:
                checks += 1
                if (s2 + 6 * j - 1) % g == 0 or (s2 + 6 * j + 1) % g == 0: viol += 1
print(f"   verification (twin centres <= 10^4, all offsets in U_g): checks {checks}, violations {viol}")
# payoff: rung rate inside A_13 vs overall, twin centres <= PFULL
M = 5 * 7 * 11 * 13
A = [a for a in range(M) if all(a % g in U[g] for g in (5, 7, 11, 13))]
tot_all = tot_A = hit_all = hit_A = 0
for s in centres:
    if s > PFULL + 1: break
    c = s // 6; s2 = s * s; jmax = 2 * c - 1
    for j in range(-jmax, jmax + 1):
        tw = isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1)
        tot_all += 1; hit_all += tw
        if j % M in A: tot_A += 1; hit_A += tw
print(f"   |A_13| = {len(A)} of {M} classes (density {len(A)/M:.4f}); twin centres <= {PFULL}: rung rate overall {hit_all/tot_all:.5f} ({hit_all}/{tot_all}), inside A_13 {hit_A/tot_A:.5f} ({hit_A}/{tot_A}); enhancement {(hit_A/tot_A)/(hit_all/tot_all):.2f} (naive prod 1/(1-2/g) over 5,7,11,13 = {1/((1-2/5)*(1-2/7)*(1-2/11)*(1-2/13)):.2f})")

# ---------------- Angle 2 ----------------
cset = set(centres)
two_parents = 0; with_parent = collections.Counter(); total = collections.Counter(); parents = {}
for sp in centres:
    r = math.isqrt(sp)
    cands = [x for x in (r - 1, r, r + 1) if x % 6 == 0 and x in cset and (x - 1) ** 2 < sp - 1 and sp + 1 < (x + 1) ** 2]
    if len(cands) > 1: two_parents += 1
    d = int(math.log10(sp)); total[d] += 1
    if cands: with_parent[d] += 1; parents[sp] = cands[0]
print(f"Angle 2 (forest): twin centres {len(centres)}; with two parents: {two_parents}; fraction with a parent by decade: " +
      ", ".join(f"10^{d}: {with_parent[d]}/{total[d]}" for d in sorted(total)))
Y = PMAX
left = sum(1 for sp in centres if sp in parents and sp <= Y)
right = 0
for s in centres:
    if (s - 1) ** 2 >= Y: break
    c = s // 6; s2 = s * s; jmax = 2 * c - 1
    for j in range(-jmax, jmax + 1):
        sp = s2 + 6 * j
        if sp <= Y and sp in cset: right += 1
print(f"   accounting identity at Y = {Y}: descent {left}, ascent {right}, {'EQUAL' if left == right else 'MISMATCH'}")
children = collections.Counter(parents.values())
print(f"   child counts among parents <= {Y}: " + ", ".join(f"{k} children: {v}" for k, v in sorted(collections.Counter(children.values()).items())))
roots = [s for s in centres if s not in parents]
print(f"   roots (no parent) among twin centres <= {Y}: {len(roots)}; first roots {roots[:12]}")

# ---------------- Angle 3 ----------------
trans_viol = 0; forced = collections.Counter(); forced_ok = 0; forced_bad = 0; sq_forced = collections.Counter()
by_phase7 = collections.defaultdict(list)
for k in range(1, len(centres) - 1):
    r, s, t = centres[k - 1], centres[k], centres[k + 1]
    if t > PMAX: break
    dm, dp = s - r, t - s
    for g in gears:
        if g > 61: break
        if g >= r - 1: continue
        if (t * t - s * s) % g != ((t - s) * (s + t)) % g: trans_viol += 1
        excl = {1 % g, (-1) % g, (1 - dp) % g, (-1 - dp) % g, (1 + dm) % g, (-1 + dm) % g}
        surv = [x for x in range(g) if x not in excl]
        if len(surv) == 1:
            forced[g] += 1
            if s % g == surv[0]: forced_ok += 1
            else: forced_bad += 1
        if len(surv) >= 1 and len({(x * x) % g for x in surv}) == 1: sq_forced[g] += 1
    # first-rung index among 61-rough offsets by (forced?) s^2 mod 7, twin centres <= 10^5
    if s <= 10**5:
        c = s // 6; s2 = s * s; jmax = 2 * c - 1; n = 2 * jmax + 1
        struck = np.zeros(n, dtype=bool)
        for g in gears:
            if g > 61: break
            u = pow(6, -1, g)
            for jr in ((u * (1 - s2)) % g, (u * (-1 - s2)) % g): struck[(jr + jmax) % g::g] = True
        bo = np.nonzero(~struck)[0] - jmax; bo = bo[np.argsort(np.abs(bo), kind='stable')]
        for i, j in enumerate(bo):
            j = int(j)
            if isprime(s2 + 6 * j - 1) and isprime(s2 + 6 * j + 1):
                by_phase7[s2 % 7].append(i); break
print(f"Angle 3 (neighbours): translation violations {trans_viol}; s mod g forced by the six exclusions (count, all agree?): " +
      ", ".join(f"g={g}: {forced[g]}" for g in sorted(forced)) + f"; agreements {forced_ok}, violations {forced_bad}")
print("   s^2 mod g forced: " + ", ".join(f"g={g}: {sq_forced[g]}" for g in sorted(sq_forced)))
print("   first-rung index (61-rough list) by s^2 mod 7 (twin centres <= 10^5): " + ", ".join(f"{q}: mean {np.mean(v):.2f} (n={len(v)})" for q, v in sorted(by_phase7.items())))
