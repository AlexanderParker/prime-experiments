"""Round 4 lateral: the MASTER SUPPLY FORMULA - overcount, B, and the
Constructor's n2 as exact prefix-graded floor arithmetic, no sieving.

MASTER FORMULA. For gears 5..y, window prefix [1, t], members <= 6t+1:
write cnt(k) = omega(6k-1) + omega(6k+1). Using (c-1)[c>=1] = c - 1 + [c=0]
and full inclusion-exclusion on both members simultaneously, every count
collapses to ONE signed sum over coprime pairs of squarefree gear products
(s_L, s_R), s_L | 6k-1, s_R | 6k+1, each pair a single CRT class mod s_L*s_R:

    overcount(t) = sum_{|s_L|+|s_R| >= 2} (-1)^{#gears} N(s_L, s_R; t)

where N(s_L,s_R;t) = floor((t - x)/M) + 1 if x <= t else 0, M = s_L*s_R, and
x = class representative (closed form / one CRT). Term taxonomy:
    one-sided terms  (s,1),(1,s), |s|>=2  ->  SAME  (same-member stacking)
    (q,q') singles                        ->  PAIRSPLIT (the gap-law classes)
    both-sided, >= 3 gears                ->  -CORR (multi-gear-side overlap)
Round 3's CORR is now formula: CORR_incidence(t) = PAIRSPLIT(t) - B(t) with
    B(t) = sum_{both sides nonempty} (-1)^{#gears} N(s_L, s_R; t)
         = # slots <= t with BOTH members gearful.

THE CONSTRUCTOR'S n2 (both members composite). In the window every composite
member has a gear factor (horizon), and the only PRIME gearful members are the
gears themselves, sitting exactly at their self-block slots u'(q). Hence

    n2(t) = B(t) - U(t),   U(t) = #{u'(q) <= t : partner member gearful}

(each element of U is a B-slot with a prime member; twin pairs give one shared
slot). Corollary bridge:   overcount(t) = SAME(t) + U(t) + n2(t).

This file verifies formula == sieved census AT EVERY PREFIX t (max abs diff
over all t in [1,K]) for y = 101 and y = 211, then prints the supply-arrival
schedule (u' pins / semiprimes / splits by gap / higher terms) and the
multiplicity spectrum.

Run: uv run python research/supply_formula.py   (from repo root)
"""
from collections import defaultdict

from tooth_sharing import isprime, uprime, crt2
from split_gap_law import primes

# ---------- term enumeration ----------

def products_upto(gears, cap):
    """All squarefree products of gears with value <= cap: (value, mask, #gears)."""
    out = []
    n = len(gears)

    def dfs(idx, val, mask, cnt):
        for i in range(idx, n):
            v = val * gears[i]
            if v > cap:
                break
            out.append((v, mask | (1 << i), cnt + 1))
            dfs(i + 1, v, mask | (1 << i), cnt + 1)
    dfs(0, 1, 0, 0)
    return out

def build_terms(gears, K):
    """All master-formula terms with at least one event in [1,K]:
    (x, M, sign, nL, nR, vL, vR)."""
    cap = 6 * K + 1
    prods = products_upto(gears, cap)
    u6 = {v: pow(6, -1, v) for v, _, _ in prods}
    entries = prods + [(1, 0, 0)]
    terms = []
    for vL, mL, nL in entries:
        if vL != 1 and vL > 6 * K - 1:
            continue                          # cannot divide a left member
        for vR, mR, nR in entries:
            if nL + nR < 2 or (mL & mR):
                continue
            if vL == 1:
                x, M = vR - u6[vR], vR
            elif vR == 1:
                x, M = u6[vL], vL
            else:
                M = vL * vR
                x = crt2(u6[vL], vL, (vR - u6[vR]) % vR, vR)
            if 1 <= x <= K:
                terms.append((x, M, (-1) ** (nL + nR), nL, nR, vL, vR))
    return terms

# ---------- prefix arrays: formula side ----------

def formula_arrays(terms, K):
    names = ('oc', 'same', 'pair', 'B')
    delta = {nm: [0] * (K + 1) for nm in names}
    for x, M, sg, nL, nR, vL, vR in terms:
        if nL == 0 or nR == 0:
            tags = ('oc', 'same')
        elif nL == 1 and nR == 1:
            tags = ('oc', 'pair', 'B')
        else:
            tags = ('oc', 'B')
        z = x
        while z <= K:
            for tg in tags:
                delta[tg][z] += sg
            z += M
    for a in delta.values():
        for i in range(1, K + 1):
            a[i] += a[i - 1]
    return delta

def u_slots(gears, K):
    """Sorted u'(q) slots whose PARTNER member is gearful (pure arithmetic)."""
    out = set()
    for q in gears:
        k = uprime(q)
        if k > K:
            continue
        partner = 6 * k + 1 if q == 6 * k - 1 else 6 * k - 1
        if any(partner % g == 0 for g in gears if g <= partner):
            out.add(k)
    return sorted(out)

# ---------- census side (sieve, ground truth) ----------

def census_arrays(gears, K):
    oml = [0] * (K + 1)
    omr = [0] * (K + 1)
    for q in gears:
        for m in range(q, 6 * K + 2, q):
            r = m % 6
            if r == 5 and m <= 6 * K - 1:
                oml[(m + 1) // 6] += 1
            elif r == 1 and 7 <= m <= 6 * K + 1:
                omr[(m - 1) // 6] += 1
    gs = set(gears)
    oc = [0] * (K + 1)
    B = [0] * (K + 1)
    n2 = [0] * (K + 1)
    mult = defaultdict(int)
    for k in range(1, K + 1):
        c = oml[k] + omr[k]
        oc[k] = oc[k - 1] + (c - 1 if c >= 1 else 0)
        both = oml[k] >= 1 and omr[k] >= 1
        B[k] = B[k - 1] + both
        n2[k] = n2[k - 1] + (both and (6 * k - 1) not in gs and (6 * k + 1) not in gs)
        if c:
            mult[c] += 1
    return oc, B, n2, dict(sorted(mult.items()))

# ---------- the verification ----------

def verify_scale(y):
    gears = primes(5, y)
    K = (y * y - 1) // 6
    terms = build_terms(gears, K)
    F = formula_arrays(terms, K)
    U = u_slots(gears, K)
    n2f = [0] * (K + 1)
    ui = 0
    for t in range(1, K + 1):
        while ui < len(U) and U[ui] <= t:
            ui += 1
        n2f[t] = F['B'][t] - ui
    ocC, BC, n2C, mult = census_arrays(gears, K)
    d_oc = max(abs(F['oc'][t] - ocC[t]) for t in range(1, K + 1))
    d_B = max(abs(F['B'][t] - BC[t]) for t in range(1, K + 1))
    d_n2 = max(abs(n2f[t] - n2C[t]) for t in range(1, K + 1))
    corr = F['pair'][K] - F['B'][K]
    print(f"  y={y}: {len(terms)} terms; EVERY prefix t in [1,{K}]: "
          f"max|d| overcount {d_oc}, B {d_B}, n2 {d_n2}")
    print(f"        t=K components: SAME {F['same'][K]}  PAIRSPLIT {F['pair'][K]}  "
          f"CORR_inc {corr}  B {F['B'][K]}  U {len(U)}  n2 {n2f[K]}  "
          f"overcount {F['oc'][K]}")
    print(f"        bridge oc = SAME + U + n2: {F['same'][K]} + {len(U)} + {n2f[K]} "
          f"= {F['same'][K] + len(U) + n2f[K]}  "
          f"[{'OK' if F['same'][K] + len(U) + n2f[K] == F['oc'][K] else 'MISMATCH'}]")
    print(f"        kill-multiplicity spectrum (census): {mult}")
    return gears, K, terms, F, U, n2f

# ---------- the availability schedule ----------

def schedule(gears, K, terms, F, U, n2f, band=60, deciles=10):
    print(f"  supply-arrival schedule, bottom band t <= {band}:")
    events = []
    for x, M, sg, nL, nR, vL, vR in terms:
        z = x
        while z <= band:
            if nL == 0 or nR == 0:
                v = vL if vR == 1 else vR
                kind = f"SAME {v}={'*'.join(str(g) for g in sorted(gfac(v, gears)))}"
            elif nL == 1 and nR == 1:
                g = abs(vR - vL)
                kind = (f"PIN ({min(vL,vR)},{max(vL,vR)})" if g == 2
                        else f"split g={g} ({vL},{vR})")
            else:
                kind = f"hub ({vL}|{vR}) sign {sg:+d}"
            events.append((z, kind))
            z += M
    for z, kind in sorted(events):
        star = " <- n2 starts" if n2f[z] == 1 and n2f[z - 1] == 0 else ""
        print(f"    t={z:>3}: {kind}{star}")
    print(f"  first n2 slot (both members composite): "
          f"t={next(t for t in range(1, K + 1) if n2f[t] > 0)}")
    print(f"  U slots (prime-member B slots, all in bottom band): {U}")
    print(f"  decile curves t, SAME, PAIRSPLIT, B, n2, overcount:")
    for d in range(1, deciles + 1):
        t = K * d // deciles
        print(f"    t={t:>6}: {F['same'][t]:>6} {F['pair'][t]:>6} "
              f"{F['B'][t]:>6} {n2f[t]:>6} {F['oc'][t]:>6}")

def gfac(v, gears):
    out = []
    for g in gears:
        if v % g == 0:
            out.append(g)
    return out

if __name__ == "__main__":
    print("=" * 72)
    print("PART 1+3: master formula vs census at EVERY prefix; n2 = B - U bridge")
    for y in (101, 211):
        gears, K, terms, F, U, n2f = verify_scale(y)
        if y == 211:
            print("=" * 72)
            print("PART 2: availability schedule (y=211)")
            schedule(gears, K, terms, F, U, n2f)
