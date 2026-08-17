"""Tooth-sharing downstream effect: does "twins at scale sqrt(N)" measurably
change coverage at scale N through the machine's own laws?

Setting: twin gears (p, p+2) share their low tooth u' = round(p/6) (the
tooth-sharing law, 17c). In centred coordinates EVERY gear's teeth are +-u',
so a twin pair shares BOTH tooth values: of the 4 within-pair double-kill CRT
classes mod P = p(p+2), two are PINNED at k = +-u' mod P. A generic gear pair
has its 4 classes at arbitrary CRT positions. Density is identical (4 per
period - forced by the prod(q-2) law, so sharing can NEVER change survivor
counts over a full period, only their positions). The whole candidate
mechanism is therefore positional, and the quantitative law to test is:

    LAW (redistribution): in window [1, K], sharing the tooth phase of a pair
    (p, p') changes the expected within-pair wasted (double) kills by
        Delta = 1 - 2R/P,   P = p p',  R = K mod P
    (+ O(v/P) corrections). Sign flips at R = P/2: pairs with period > 2K
    waste MORE in-window when sharing; pairs with period < 2K (many full
    periods inside) average to zero.

Parts:
 1. anatomy - verify the pinned classes and that both are SPLIT kills
    (each twin gear kills a different member of the same slot).
 2. the redistribution law, pair by pair: real teeth (A) vs shared-random-
    phase (B) vs independent-random-phase (C) synthetic gears.
 3. full 20-gear set (10 twin pairs): A vs B vs C populations on global
    metrics: overcount, survivors, max stride, lone-killer (fragile proxy).
 4. the literal experiment: RICH (20 twin gears) vs POOR (20 matched
    non-twin primes), low window + random windows, with the tooth-count
    confound decomposed.

Run: uv run python research/tooth_sharing.py
"""
import random
from math import prod

# ---------- basics ----------

def isprime(n):
    if n < 2: return False
    d = 2
    while d * d <= n:
        if n % d == 0: return False
        d += 1
    return True

def uprime(q):
    u = pow(6, -1, q)
    return min(u, q - u)

def teeth_real(q):
    u = pow(6, -1, q)
    return (u, q - u)          # = {u', q - u'} = +-u' centred

def teeth_synth(q, v):
    return (v, q - v)          # synthetic gear: teeth at +-v

def crt2(r1, m1, r2, m2):
    """x = r1 mod m1, x = r2 mod m2 -> x mod m1*m2 (m1, m2 coprime)."""
    g = pow(m1, -1, m2)
    return (r1 + m1 * ((r2 - r1) * g % m2)) % (m1 * m2)

def count_in_window(x, P, K):
    """#{k in [1,K] : k = x mod P}, x in [0,P)."""
    if x == 0:
        return K // P
    if x > K:
        return 0
    return (K - x) // P + 1

def pair_coincidences(p, pp, tp, tpp, K):
    """Exact # slots in [1,K] killed by both gears, given tooth sets."""
    total = 0
    seen = set()
    for a in tp:
        for b in tpp:
            x = crt2(a % p, p, b % pp, pp)
            if x not in seen:      # classes can coincide only if tooth sets degenerate
                seen.add(x)
                total += count_in_window(x, p * pp, K)
    return total

# ---------- part 1: anatomy of the pinned classes ----------

def part1():
    print("=" * 72)
    print("PART 1: pinned-class anatomy, twin gear pairs p <= 1000")
    twins = [p for p in range(5, 1000) if isprime(p) and isprime(p + 2)]
    bad = 0
    for p in twins:
        pp = p + 2
        up = uprime(p)
        assert uprime(pp) == up, f"u' differs at {p}"
        P = p * pp
        classes = sorted({crt2(a % p, p, b % pp, pp)
                          for a in teeth_real(p) for b in teeth_real(pp)})
        # full closed form: classes are exactly {+-u', +-u'(p+1)} mod P, and the
        # mixed class is the twin-product slot: 6*u'(p+1) - 1 = (p+1)^2 - 1 = P
        expect = sorted({up % P, (P - up) % P, (up * (p + 1)) % P,
                         (P - up * (p + 1)) % P})
        pinned = (classes == expect) and (6 * up * (p + 1) - 1 == P)
        # split-kill check at both pinned slots: each gear kills a different member
        ok = True
        for k in (up, P - up):
            l, r = 6 * k - 1, 6 * k + 1
            kills_p = (l % p == 0) or (r % p == 0)
            kills_pp = (l % pp == 0) or (r % pp == 0)
            same_member = (l % p == 0 and l % pp == 0) or (r % p == 0 and r % pp == 0)
            if not (kills_p and kills_pp and not same_member):
                ok = False
        if not (len(classes) == 4 and pinned and ok):
            bad += 1
            print(f"  FAIL at ({p},{pp}): classes={classes}")
    print(f"  {len(twins)} twin pairs: the 4 CRT double-kill classes are exactly "
          f"{{+-u', +-u'(p+1)}} mod p(p+2); +-u' are SPLIT kills (each gear a "
          f"different member); the mixed class IS the twin-product slot "
          f"(6*u'(p+1)-1 = p(p+2)). failures: {bad}")
    # show one example
    p, pp = 269, 271; up = uprime(p); P = p * pp
    k = P - up
    print(f"  example (269,271): u'={up}; pinned slots k={up} "
          f"(pair ({6*up-1},{6*up+1}) = the gears' own pair) and k={k}: "
          f"6k-1 = {6*k-1} = {pp} * {(6*k-1)//pp}, 6k+1 = {6*k+1} = {p} * {(6*k+1)//p}")

# ---------- part 2: the redistribution law, pair by pair ----------

def part2(K=60000, draws=400, seed=1):
    print("=" * 72)
    print(f"PART 2: redistribution law, window [1,{K}], {draws} draws per config")
    print("  A = real teeth (shared at u'), B = shared random phase, C = independent")
    print("  predicted E[B]-E[C] = 1 - 2R/P  (R = K mod P)")
    rng = random.Random(seed)
    pairs = [(101,103),(137,139),(179,181),(197,199),(239,241),
             (269,271),(281,283),(311,313),(347,349),(419,421),
             (431,433),(461,463),(521,523),(569,571),(599,601)]
    print(f"  {'pair':>12} {'P=pp2':>8} {'R/P':>6} {'A':>4} {'E[B]':>7} {'E[C]':>7} "
          f"{'B-C':>7} {'pred':>7}")
    tot_meas, tot_pred = 0.0, 0.0
    for p, pp in pairs:
        P = p * pp
        R = K % P
        A = pair_coincidences(p, pp, teeth_real(p), teeth_real(pp), K)
        sB = sC = 0
        for _ in range(draws):
            v = rng.randrange(1, (p - 1) // 2 + 1)
            sB += pair_coincidences(p, pp, teeth_synth(p, v), teeth_synth(pp, v), K)
            v1 = rng.randrange(1, (p - 1) // 2 + 1)
            v2 = rng.randrange(1, (pp - 1) // 2 + 1)
            sC += pair_coincidences(p, pp, teeth_synth(p, v1), teeth_synth(pp, v2), K)
        eB, eC = sB / draws, sC / draws
        pred = 1 - 2 * R / P
        tot_meas += eB - eC; tot_pred += pred
        print(f"  ({p:>4},{pp:>4}) {P:>8} {R/P:>6.3f} {A:>4} {eB:>7.3f} {eC:>7.3f} "
              f"{eB-eC:>7.3f} {pred:>7.3f}")
    print(f"  totals: measured sum(B-C) = {tot_meas:+.3f}, predicted = {tot_pred:+.3f}")

# ---------- window metrics ----------

def window_metrics(gearteeth, K, lo=0):
    """Metrics over slots lo+1 .. lo+K. gearteeth = list of (q, (t1,t2))."""
    cnt = bytearray(K + 1)                 # cnt[i] = kills at slot lo+i
    marks = 0
    for q, ts in gearteeth:
        for t in ts:
            k = (t - lo - 1) % q + lo + 1  # first slot > lo congruent to t
            i = k - lo
            while i <= K:
                cnt[i] += 1
                marks += 1
                i += q
    distinct = sum(1 for i in range(1, K + 1) if cnt[i])
    lone = sum(1 for i in range(1, K + 1) if cnt[i] == 1)
    survivors = K - distinct
    # max stride between consecutive survivors
    maxstride, last = 0, None
    for i in range(1, K + 1):
        if not cnt[i]:
            if last is not None:
                maxstride = max(maxstride, i - last)
            last = i
    return dict(marks=marks, distinct=distinct, overcount=marks - distinct,
                survivors=survivors, lone=lone, maxstride=maxstride)

# ---------- part 3: full-set A vs B vs C ----------

TWINS = [269, 281, 311, 347, 419, 431, 461, 521, 569, 599]

def part3(K=60000, draws=300, seed=2):
    print("=" * 72)
    print(f"PART 3: 20-gear set (10 twin pairs {TWINS}), window [1,{K}]")
    rng = random.Random(seed)
    gears = [q for p in TWINS for q in (p, p + 2)]
    A = window_metrics([(q, teeth_real(q)) for q in gears], K)
    keys = ('overcount', 'survivors', 'maxstride', 'lone', 'marks')

    def population(shared):
        rows = []
        for _ in range(draws):
            gt = []
            for p in TWINS:
                if shared:
                    v = rng.randrange(1, (p - 1) // 2 + 1)
                    gt += [(p, teeth_synth(p, v)), (p + 2, teeth_synth(p + 2, v))]
                else:
                    v1 = rng.randrange(1, (p - 1) // 2 + 1)
                    v2 = rng.randrange(1, (p + 1) // 2 + 1)
                    gt += [(p, teeth_synth(p, v1)), (p + 2, teeth_synth(p + 2, v2))]
            rows.append(window_metrics(gt, K))
        return rows

    B, C = population(True), population(False)

    def stats(rows, key):
        vals = [r[key] for r in rows]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return m, var ** 0.5

    print(f"  {'metric':>10} {'A(real)':>8} {'E[B] shared':>12} {'E[C] indep':>11} "
          f"{'B-C':>7} {'se':>6} {'z(A in C)':>10}")
    for key in keys:
        mB, sB = stats(B, key)
        mC, sC = stats(C, key)
        se = (sB ** 2 / draws + sC ** 2 / draws) ** 0.5
        z = (A[key] - mC) / sC if sC else float('nan')
        print(f"  {key:>10} {A[key]:>8} {mB:>12.2f} {mC:>11.2f} "
              f"{mB-mC:>7.2f} {se:>6.2f} {z:>10.2f}")

# ---------- part 4: literal rich vs poor, real primes ----------

def matched_controls(rich):
    """One non-twin prime near each rich gear, no two controls twin each other."""
    used = set(rich)
    out = []
    for q in rich:
        for d in range(1, 200):
            for cand in (q - d, q + d):
                if cand < 5 or cand in used or not isprime(cand):
                    continue
                if isprime(cand - 2) or isprime(cand + 2):
                    continue        # exclude any twin-pair member entirely
                used.add(cand)
                out.append(cand)
                break
            else:
                continue
            break
    return out

def part4(K=60000, nwin=100, seed=3):
    print("=" * 72)
    rich = [q for p in TWINS for q in (p, p + 2)]
    poor = matched_controls(rich)
    print(f"PART 4: literal rich-vs-poor, real primes, window length {K}")
    print(f"  RICH (10 twin pairs): {rich}")
    print(f"  POOR (matched, no twin members): {sorted(poor)}")
    print(f"  sum 2/q: rich {sum(2/q for q in rich):.6f}  poor {sum(2/q for q in poor):.6f}")
    mr = window_metrics([(q, teeth_real(q)) for q in rich], K)
    mp = window_metrics([(q, teeth_real(q)) for q in poor], K)
    print(f"  low window [1,{K}]:")
    print(f"    {'':>6} {'marks':>7} {'overcount':>9} {'survivors':>9} {'maxstride':>9} {'lone':>6}")
    print(f"    {'rich':>6} {mr['marks']:>7} {mr['overcount']:>9} {mr['survivors']:>9} "
          f"{mr['maxstride']:>9} {mr['lone']:>6}")
    print(f"    {'poor':>6} {mp['marks']:>7} {mp['overcount']:>9} {mp['survivors']:>9} "
          f"{mp['maxstride']:>9} {mp['lone']:>6}")
    print(f"    survivor diff {mr['survivors']-mp['survivors']:+d} decomposes as "
          f"marks diff {mp['marks']-mr['marks']:+d} + overcount diff "
          f"{mr['overcount']-mp['overcount']:+d}  (tooth-count confound explicit)")
    rng = random.Random(seed)
    diffs = {k: [] for k in ('overcount', 'survivors', 'maxstride', 'lone')}
    for _ in range(nwin):
        a = rng.randrange(10 ** 6, 10 ** 8)
        wr = window_metrics([(q, teeth_real(q)) for q in rich], K, lo=a)
        wp = window_metrics([(q, teeth_real(q)) for q in poor], K, lo=a)
        for k in diffs:
            diffs[k].append(wr[k] - wp[k])
    print(f"  {nwin} random windows [a, a+{K}), rich - poor:")
    for k, v in diffs.items():
        m = sum(v) / len(v)
        s = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        print(f"    {k:>10}: mean {m:+8.2f}  sd {s:7.2f}  se {s/len(v)**0.5:6.2f}")

if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
