"""Invariants of the charge set in turns 1..M, with the fuel model selectable:
  real : fuel = 1 or a prime above Q (the exhaust);
  F    : the counterfactual F = {f > Q : gcd(f, q#) = 1, f = 1 (mod 3)} (turn_ledger.md V2);
  adv  : the one-tooth free-phase manifold adversary: each manifold prime p in (q, Q] removes ONE class c_p,
         chosen greedily to cover every pure-imprint pair (n, n+2 both coprime to q#) in (Q, (M+1)Q];
         unused gears keep phase zero (c_p = 0). The fuel is then {f > Q : gcd(f, q#) = 1, f != c_p mod p for all p}.
A charge is s x f (air s q-smooth, f = 1 or in the fuel); the charge set is the pairs (n, n+2) both charges.
Computes: per-turn P_m, T_m; the pure charges per pure-imprint class mod q# and the mirror pairing (E-P5);
F_odd(q), the manifold's run ceiling, the longest run of consecutive open odd numbers, the most charges in a
window of length q# (E-P6); the charge-unit walk between consecutive twins per turn (E-P7); balance of the fuel
on the engine's teeth, mirror closure of the family set, the one-tooth and phase-zero flags (E-P8).

usage: uv run python research/valves/r2/invariants.py q Q [M] [real|F|adv]
"""
import sys, os, math, json, heapq
from collections import Counter
import numpy as np

q = int(sys.argv[1]); Q = int(sys.argv[2])
M = int(sys.argv[3]) if len(sys.argv) > 3 else 60
model = sys.argv[4] if len(sys.argv) > 4 else "real"
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)
N = (M + 1) * Q + 2


def sieve(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


isprime = sieve(N + 3)
primes = np.flatnonzero(isprime)
engine = [int(p) for p in primes if p <= q]
odd_engine = [p for p in engine if p > 2]
qsharp = math.prod(engine)
gears = [int(p) for p in primes if q < p <= Q]
qprime = gears[0]

nn = np.arange(0, N + 3, dtype=np.int64)
rest = nn.copy()
for p in engine:
    while True:
        m = (rest % p == 0); m[0] = False
        if not m.any():
            break
        rest[m] //= p
air = np.where(nn > 0, nn // np.maximum(rest, 1), 0); fuel = rest
coprime = np.ones(N + 3, dtype=bool)
for p in engine:
    coprime[::p] = False
coprime[0] = False

# ---- the fuel model
used = {}
if model == "real":
    fuel_ok = (fuel > Q) & isprime[np.minimum(fuel, N + 3)]
elif model == "F":
    fuel_ok = (fuel > Q) & (fuel % 3 == 1)
elif model == "adv":
    cand = np.flatnonzero(coprime[Q + 1:N - 1] & coprime[Q + 3:N + 1]) + Q + 1   # pure-imprint pairs in (Q, (M+1)Q]
    cand = cand[cand <= (M + 1) * Q]
    remaining = cand.copy()
    n_cand = len(cand)
    # lazy greedy: upper bound per gear = its last computed best coverage (coverage only falls)
    heap = []
    for p in gears:
        h = np.bincount(remaining % p, minlength=p) + np.bincount((remaining + 2) % p, minlength=p)
        c = int(h.argmax()); heap.append((-int(h[c]), p, c))
    heapq.heapify(heap)
    rounds = 0
    while len(remaining):
        if not heap:                       # every gear used and pairs survive: the adversary's reach ends here
            break
        negb, p, c = heapq.heappop(heap)
        h = np.bincount(remaining % p, minlength=p) + np.bincount((remaining + 2) % p, minlength=p)
        c = int(h.argmax()); b = int(h[c])
        if heap and b < -heap[0][0]:
            heapq.heappush(heap, (-b, p, c)); continue
        used[p] = c
        kill = (remaining % p == c) | ((remaining + 2) % p == c)
        remaining = remaining[~kill]
        rounds += 1
    ok = np.ones(N + 3, dtype=bool)
    for p in gears:
        c = used.get(p, 0)
        ok[c::p] = False
    fuel_ok = (fuel > Q) & ok[np.minimum(fuel, N + 3)]
    if len(remaining):
        reach = int((remaining.min() - 1) // Q) - 1
        print(f"adversary EXHAUSTED: all {len(gears)} gears used, {len(remaining)} of {n_cand} pure-imprint pairs survive; the first survivor "
              f"{int(remaining.min())} is in turn {reach + 1}, so the reach R({q}, {Q}) = {reach} turns (every turn <= {reach} emptied)")
    print(f"adversary: {n_cand} pure-imprint pairs in (Q, {(M+1)*Q}] covered by {len(used)} of {len(gears)} gears "
          f"(one class each; nonzero phase at {sum(1 for c in used.values() if c)}); smallest used {min(used)}, largest {max(used)}; "
          f"sum 2/p over used gears {sum(2/p for p in used):.3f}")
else:
    raise SystemExit("model must be real, F or adv")
opn = (fuel == 1) | fuel_ok
opn[0] = False
opn[:Q + 1] = False              # the zone starts above Q; embers below Q are not charges
# a charge below Q+1 is irrelevant; pairs need n > Q
lo, hi = Q + 1, (M + 1) * Q
pair = np.flatnonzero(opn[lo:hi + 1] & opn[lo + 2:hi + 3]) + lo
s1 = air[pair]; s2 = air[pair + 2]; f1 = fuel[pair]; f2 = fuel[pair + 2]
pure = (s1 == 1) & (s2 == 1)
turn = (pair - 1) // Q
out = {"q": q, "Q": Q, "M": M, "model": model}

# ---- per-turn ledger
T = np.bincount(turn, minlength=M + 1)[1:M + 1]; P = np.bincount(turn[pure], minlength=M + 1)[1:M + 1]
print(f"model={model} q={q} Q={Q} M={M}: charges {len(pair)}, pure {int(pure.sum())}; turns with P_m=0: {int((P == 0).sum())} of {M}; "
      f"P_m m=1..12: {P[:12].tolist()}; T_m m=1..12: {T[:12].tolist()}")
out["P"] = P.tolist(); out["T"] = T.tolist()

# ---- E-P5: pure charges per pure-imprint class mod q#, and the mirror pairing
res = np.arange(qsharp)
pure_classes = [int(r) for r in res if math.gcd(int(r), qsharp) == 1 and math.gcd(int(r) + 2, qsharp) == 1]
mirror = {r: (-r - 2) % qsharp for r in pure_classes}
fixed = [r for r in pure_classes if mirror[r] == r]
cls = Counter((pair[pure] % qsharp).tolist())
counts = {r: cls.get(r, 0) for r in pure_classes}
print(f"E-P5: pure imprint has {len(pure_classes)} classes mod {qsharp} (prod(p-2) = {math.prod(p - 2 for p in odd_engine)}); "
      f"mirror-fixed classes: {fixed} (the spoke class -1 = {qsharp - 1}); pure charges per class: min {min(counts.values())}, "
      f"max {max(counts.values())}, mean {np.mean(list(counts.values())):.1f}; fixed-class count {counts.get(qsharp - 1, 0)}")
pairs_seen = set(); mm = []
for r in pure_classes:
    s = mirror[r]
    if s != r and (s, r) not in pairs_seen:
        pairs_seen.add((r, s)); mm.append((r, s, counts[r], counts[s]))
if mm:
    dev = max(abs(a - b) for _, _, a, b in mm); tot = sum(counts.values())
    print(f"E-P5: {len(mm)} mirror pairs of classes; max |count(r) - count(-r-2)| = {dev} against sqrt(N) = {math.sqrt(max(tot,1)):.0f}; "
          f"first pairs: {mm[:4]}")
out["class_counts"] = counts; out["mirror_pairs"] = mm; out["fixed"] = fixed

# ---- E-P6: the pigeonhole
# F_odd(q): most consecutive odd numbers with no pure start (cyclic over the odd residues mod q#)
pc = sorted(pure_classes)                     # all odd since coprime to q# (2 | q#)
gaps = [((pc[(i + 1) % len(pc)] - pc[i]) % qsharp) // 2 for i in range(len(pc))]
F_odd = max(gaps)
ceiling = qprime - 1
# longest run of consecutive open odd numbers in (Q, (M+1)Q]
odd_idx = np.arange(lo | 1, hi + 1, 2)
o = opn[odd_idx].astype(np.int8)
# run lengths
d = np.diff(np.concatenate([[0], o, [0]]))
starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1)
runs = ends - starts
longest = int(runs.max()) if len(runs) else 0
nlong = int((runs == longest).sum())
# do the longest runs contain a twin?
tw_in = 0
pure_set = set(pair[pure].tolist())
for a, b in zip(starts[runs == longest], ends[runs == longest]):
    ns = odd_idx[a:b]
    if any(int(x) in pure_set for x in ns[:-1]):
        tw_in += 1
# most charges in a window of length q#
ind = np.zeros(hi - lo + 2, dtype=np.int32); ind[pair - lo] = 1
cs = np.concatenate([[0], np.cumsum(ind)])
win = cs[qsharp:] - cs[:-qsharp]
max_win = int(win.max()); burnt_res = qsharp - len(pure_classes)
# charge PAIRS (n, n+2 both charges) per window of length q#, against the burnt odd residues q#/2 - prod(p-2)
indp = np.zeros(hi - lo + 2, dtype=np.int32); indp[pair - lo] = 1
csp = np.concatenate([[0], np.cumsum(indp)])
winp = csp[qsharp:] - csp[:-qsharp]
max_winp = int(winp.max()); burnt_odd = qsharp // 2 - len(pure_classes)
print(f"E-P6 (pairs): most charge pairs in a window of length q# = {max_winp} against burnt odd residues q#/2 - prod(p-2) = {burnt_odd} "
      f"(a window of q# integers holds q#/2 = {qsharp // 2} odd numbers, so at most {qsharp // 2 - 1} consecutive pairs; the manifold allows "
      f"at most q'-2 = {qprime - 2} consecutive open pairs at step 2)")
out["max_win_pairs"] = max_winp; out["burnt_odd"] = burnt_odd
print(f"E-P6: F_odd({q}) = {F_odd} (a run of {F_odd + 1} consecutive open odd numbers must hold a twin); manifold ceiling q'-1 = {ceiling} "
      f"(q' = {qprime}); pigeonhole margin (needed - allowed) = {F_odd + 1 - ceiling}; measured longest run of consecutive open odd "
      f"numbers {longest} ({nlong} runs, {tw_in} of them hold a twin); most charges in a window of length q# = {max_win} against "
      f"burnt residues q# - prod(p-2) = {burnt_res}")
out["F_odd"] = F_odd; out["ceiling"] = ceiling; out["longest_run"] = longest; out["max_win"] = max_win; out["burnt_res"] = burnt_res

# ---- E-P7: the charge-unit walk between consecutive twins
pi = np.flatnonzero(pure)
if len(pi) >= 2:
    walk = np.diff(pi)                       # charges from one twin to the next (1 + burnt between)
    wt = turn[pi[:-1]]
    rows = []
    for m in [1, 2, 3, 5, 9, 15, 30, 45, 60]:
        if m > M:
            break
        sel = walk[wt == m]
        if len(sel):
            rows.append((m, int(sel.max()), float(sel.mean()), float(sel.max() / sel.mean())))
    print("E-P7: charge-unit walk twin -> next twin, per turn (m, max, mean, max/mean): " + "; ".join(f"({a}, {b}, {c:.2f}, {d:.1f})" for a, b, c, d in rows)
          + f"; overall max {int(walk.max())} in turn {int(wt[walk.argmax()])}")
    # the geometric scale: with pure share p_m = P_m / T_m per charge, the largest of P_m geometric gaps is about log(P_m) / (-log(1 - p_m))
    geo = []
    for m, mx, mean, _ in rows:
        Pm, Tm = int(P[m - 1]), int(T[m - 1])
        if 1 < Pm < Tm:
            pm = Pm / Tm; scale = math.log(Pm) / (-math.log(1 - pm))
            geo.append((m, mx, round(scale, 1), round(mx / scale, 2)))
    print("E-P7 (tail): per turn (m, measured max, geometric scale log(P_m)/(-log(1 - P_m/T_m)), ratio): " + "; ".join(str(g) for g in geo))
    out["walk_geo"] = geo
    out["walk"] = rows
else:
    print("E-P7: fewer than two pure charges; the walk to the next pure charge is unbounded")

# ---- E-P8: what the fuel has
frange = np.arange(Q + 1, hi + 3)
fm = frange[fuel_ok[Q + 1:hi + 3] & coprime[Q + 1:hi + 3]]       # the fuel set in range
bal = {}
for p in odd_engine:
    h = np.bincount(fm % p, minlength=p)[1:]
    bal[p] = (int(h.min()), int(h.max()))
fam = Counter(zip(s1[~pure & (f1 > 1) & (f2 > 1)].tolist(), s2[~pure & (f1 > 1) & (f2 > 1)].tolist()))
mirrored = sum(1 for (a, b) in fam if (b, a) in fam)
div_by_manifold = 0
for p in gears:
    if p > hi:
        break
    div_by_manifold += int(((fm % p) == 0).sum())
phase_zero = (model == "real") or (model == "adv" and all(c == 0 for c in used.values()))
print(f"E-P8: fuel members in range {len(fm)}; balance per engine gear (min, max class count): {bal}; "
      f"fuelled families {len(fam)}, present with their mirror {mirrored}; fuel members divisible by a manifold prime: {div_by_manifold}; "
      f"one class per manifold prime: {'yes' if model in ('real', 'adv') else 'no (gear 3 removes two classes, the manifold removes none)'}; "
      f"phase zero at every manifold prime: {'yes' if phase_zero else 'no'}"
      + (f" ({sum(1 for c in used.values() if c)} gears at nonzero phase)" if model == 'adv' else ""))
out["balance"] = bal; out["families"] = len(fam); out["mirrored"] = mirrored; out["div_by_manifold"] = div_by_manifold
if model == "adv":
    out["used"] = {int(p): int(c) for p, c in used.items()}
with open(os.path.join(outdir, f"inv_{model}_q{q}_Q{Q}_M{M}.json"), "w") as f:
    json.dump(out, f)
