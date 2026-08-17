"""Round 2 lateral: close the overcount/lone anomaly into exact identities,
then exact phase-space extremality for tiny machines.

Part A - the anomaly as a theorem. For the round-1 20-gear set (10 twin pairs,
269..601) and window [1, K], K = 60000, V = 6K+1:

  REAL side (pure divisor census, no window array, no phases):
    cnt(k) = omega_G(6k-1) + omega_G(6k+1)  (omega_G = # gears dividing)
    overcount = sum (cnt-1 over killed slots) = SAME + B where
      SAME = sum over members v <= V of (omega_G(v) - 1)+  = semiprime census
             (# squarefree gear-products with a member multiple in window;
             triple products >> V so pairs only)
      B    = # slots with BOTH members gearful (split census: Bezout-pinned,
             contains the 10 twin own-slots u'(p))
    lone = # slots with cnt = 1, also pure census.

  RANDOM side (closed form, no simulation). Phases v_q uniform on
  [1,(q-1)/2] independent; for q not dividing k exactly one v hits k, so
    P(q kills k) = 2/(q-1), independent across gears.
  Hence exact expectations:
    E[marks]     = sum_q (K - floor(K/q)) * 2/(q-1)
    E[distinct]  = sum_k 1 - prod_{q not| k} (1 - 2/(q-1))
    E[overcount] = E[marks] - E[distinct]
    E[lone]      = sum_k sum_{q not| k} p_q prod_{q' != q, not| k} (1 - p_q')
  (k <= K < any product of two gears, so k is divisible by at most one gear.)

  Identity checks: census == window-array metrics EXACTLY; closed-form
  expectations == Monte Carlo within se. The z = +6.1 / z = -5.9 anomalies are
  then the difference of two formulas, i.e. theorems, not mysteries.

Part B - extremality, exact. Tiny machines {5,7} (y=7, K=8), {5,7,11} (y=11,
K=20), {5,7,11,13} (y=13, K=28): enumerate the WHOLE phase space (mirror-
preserving teeth +-v per gear; also the full unconstrained 2-teeth space for
the sets where it is small) and rank the real phase vector on overcount, lone,
survivors, max stride. Answers argmax/argmin exactly, ties counted.

Run: uv run python research/overcount_census.py
"""
import itertools
import random
from collections import defaultdict

from tooth_sharing import isprime, uprime, teeth_real, teeth_synth, window_metrics

TWINS = [269, 281, 311, 347, 419, 431, 461, 521, 569, 599]
G = [q for p in TWINS for q in (p, p + 2)]
K = 60000

# ---------- part A1: real machine as divisor census ----------

def real_census(gears, K):
    V = 6 * K + 1
    memdiv = defaultdict(list)            # member value -> gear divisors
    for q in gears:
        for m in range(q, V + 1, q):
            r = m % 6
            if r == 5 and m <= 6 * K - 1:
                memdiv[m].append(q)       # left member of slot (m+1)/6
            elif r == 1 and m >= 7:
                memdiv[m].append(q)       # right member of slot (m-1)/6
    cnt = defaultdict(int)
    sides = defaultdict(lambda: [0, 0])   # slot -> [omega(left), omega(right)]
    for v, qs in memdiv.items():
        if v % 6 == 5:
            k = (v + 1) // 6
            sides[k][0] += len(qs)
        else:
            k = (v - 1) // 6
            sides[k][1] += len(qs)
        cnt[k] += len(qs)
    marks = sum(len(qs) for qs in memdiv.values())
    distinct = len(cnt)
    overcount = marks - distinct
    lone = sum(1 for c in cnt.values() if c == 1)
    same = sum(len(qs) - 1 for qs in memdiv.values() if len(qs) >= 1)
    both = sum(1 for l, r in sides.values() if l and r)
    twin_pins = sum(1 for p in TWINS
                    if sides[uprime(p)][0] and sides[uprime(p)][1])
    cntdist = defaultdict(int)
    for c in cnt.values():
        cntdist[c] += 1
    return dict(marks=marks, distinct=distinct, overcount=overcount, lone=lone,
                same=same, both=both, twin_pins=twin_pins,
                survivors=K - distinct, cntdist=dict(sorted(cntdist.items())))

# ---------- part A2: random-phase expectations in closed form ----------

def random_expect(gears, K):
    p = {q: 2 / (q - 1) for q in gears}
    Emarks = sum((K - K // q) * p[q] for q in gears)
    prod_all = 1.0
    for q in gears:
        prod_all *= 1 - p[q]
    sum_all = sum(p[q] / (1 - p[q]) for q in gears)
    # slots divisible by no gear vs by exactly one gear (k <= K < q*q' always)
    n_mult = {q: K // q for q in gears}
    n0 = K - sum(n_mult.values())
    Edist = n0 * (1 - prod_all)
    Elone = n0 * prod_all * sum_all
    for q in gears:                       # gear q shields its multiples
        pr = prod_all / (1 - p[q])
        sm = sum_all - p[q] / (1 - p[q])
        Edist += n_mult[q] * (1 - pr)
        Elone += n_mult[q] * pr * sm
    return dict(marks=Emarks, distinct=Edist, overcount=Emarks - Edist,
                lone=Elone, survivors=K - Edist)

def partA(draws=200, seed=7):
    print("=" * 72)
    print(f"PART A: the anomaly as two exact formulas ({len(G)} gears, K={K})")
    # real: census vs window array, must agree EXACTLY
    cen = real_census(G, K)
    arr = window_metrics([(q, teeth_real(q)) for q in G], K)
    print("  A1 real machine, divisor census vs window array (identity check):")
    for key in ('marks', 'overcount', 'survivors', 'lone'):
        tag = "OK" if cen[key] == arr[key] else "MISMATCH"
        print(f"    {key:>10}: census {cen[key]:>6}  array {arr[key]:>6}   {tag}")
    print(f"    decomposition: overcount = SAME {cen['same']} (semiprime census: "
          f"member multiples of gear pairs qq' <= 6K+1)")
    print(f"                             + B    {cen['both']} (split slots, both members "
          f"gearful; {cen['twin_pins']} of them = the 10 twin own-slots u'(p))")
    print(f"    kill-multiplicity distribution over killed slots: {cen['cntdist']}")
    # random: closed form vs Monte Carlo
    exp = random_expect(G, K)
    rng = random.Random(seed)
    acc = defaultdict(float); acc2 = defaultdict(float)
    for _ in range(draws):
        gt = [(q, teeth_synth(q, rng.randrange(1, (q - 1) // 2 + 1))) for q in G]
        m = window_metrics(gt, K)
        for key in ('marks', 'overcount', 'survivors', 'lone'):
            acc[key] += m[key]; acc2[key] += m[key] ** 2
    print(f"  A2 random phases, closed-form expectation vs {draws} Monte Carlo draws:")
    for key in ('marks', 'overcount', 'survivors', 'lone'):
        mean = acc[key] / draws
        var = acc2[key] / draws - mean ** 2
        se = (var / draws) ** 0.5
        z = (mean - exp[key]) / se if se else 0.0
        print(f"    {key:>10}: formula {exp[key]:>9.2f}  MC {mean:>9.2f} "
              f"(se {se:5.2f})  z {z:+5.2f}")
    print("  A3 the anomaly, now a difference of formulas:")
    for key in ('overcount', 'lone'):
        print(f"    {key:>10}: real (census) {cen[key]:>6}  -  E[random] (formula) "
              f"{exp[key]:>8.2f}  =  {cen[key]-exp[key]:+8.2f}")
    print(f"    surplus anatomy: SAME {cen['same']} deterministic (every pair product "
          f"qq' <= 6K+1 is a member) + B {cen['both']} Bezout-pinned, vs the "
          f"phase-uniform expectation; lone deficit is the same slots not being lone.")

# ---------- part B: exact extremality on tiny machines ----------

def normalize(ts, q):
    a, b = sorted(t % q for t in ts)
    return (a, b)

def enum_space(gears, K, full=False):
    spaces = []
    for q in gears:
        if full:
            spaces.append([(a, b) for a in range(q) for b in range(a + 1, q)])
        else:
            spaces.append([(v, q - v) for v in range(1, (q - 1) // 2 + 1)])
    real = tuple(normalize(teeth_real(q), q) for q in gears)
    rows = {}
    for combo in itertools.product(*spaces):
        key = tuple(normalize(t, q) for t, q in zip(combo, gears))
        m = window_metrics(list(zip(gears, combo)), K)
        rows[key] = m
    return rows, real

def rank_report(gears, K, full):
    rows, real = enum_space(gears, K, full)
    n = len(rows)
    space = "FULL 2-teeth" if full else "mirror +-v"
    print(f"  gears {gears}, K={K}, {space} space: {n} configs, "
          f"real vector {real}")
    for key, want in (('overcount', 'max'), ('lone', 'min'),
                      ('survivors', 'max'), ('maxstride', 'max')):
        vals = sorted(r[key] for r in rows.values())
        rv = rows[real][key]
        lo, hi = vals[0], vals[-1]
        ties_hi = vals.count(hi); ties_lo = vals.count(lo)
        if want == 'max':
            verdict = ("ARGMAX" if rv == hi else f"not argmax (rank "
                       f"{sum(1 for v in vals if v > rv)+1} of {n} values)")
            extra = f"max {hi} x{ties_hi}"
        else:
            verdict = ("ARGMIN" if rv == lo else f"not argmin (rank "
                       f"{sum(1 for v in vals if v < rv)+1} from bottom of {n})")
            extra = f"min {lo} x{ties_lo}"
        print(f"    {key:>10}: real {rv:>3}  range [{lo},{hi}]  {extra:>10}  -> {verdict}")

def partB():
    print("=" * 72)
    print("PART B: exact phase-space extremality, tiny machines")
    rank_report([5, 7], 8, full=False)
    rank_report([5, 7], 8, full=True)
    rank_report([5, 7, 11], 20, full=False)
    rank_report([5, 7, 11], 20, full=True)
    rank_report([5, 7, 11, 13], 28, full=False)
    # robustness of the overcount verdict across window lengths, {5,7,11} mirror
    print("  window sweep, {5,7,11}, mirror space, overcount verdict per K:")
    out = []
    for Kw in range(10, 41, 5):
        rows, real = enum_space([5, 7, 11], Kw, full=False)
        vals = [r['overcount'] for r in rows.values()]
        rv = rows[real]['overcount']
        out.append(f"K={Kw}:{rv}/{max(vals)}{'*' if rv == max(vals) else ''}")
    print("    real/max (* = argmax): " + "  ".join(out))

if __name__ == "__main__":
    partA()
    partB()
