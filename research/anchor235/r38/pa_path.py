"""Branch W.a - the path from q^2 taken apart (part 1: the blocker sequence, the layer nest,
the bucket vector at the landing, the character law).

The path is the columns k_0 = (q^2-1)/6, k_0+1, ... up to the landing.  Column k_0 + i carries
q^2 + 6i - 2 and q^2 + 6i, so gear g strikes offset i iff

    i = (2 - q^2) u_g (mod g)      [lower member]      or      i = (-q^2) u_g (mod g)   [upper]

with u_g = 6^{-1} mod g.  Everything below is computed from those two progressions per gear.

Writes results/pa_path.txt.
"""
import os
from math import isqrt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "pa_path.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


QMAX = 20000
FEATURE = [53, 59, 137, 2593, 4637]          # sixth = longest walk in range, added later


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


FL = sieve(QMAX + 10)
GEARS = [p for p in range(5, QMAX + 1) if FL[p]]
UU = [pow(6, -1, g) for g in GEARS]
NG = len(GEARS)
say("gears 5..%d: %d" % (QMAX, NG))


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


F_LOW = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}

NEG = 128            # how far left of k_0 we probe for the containing stretch
PROBE = 80           # offsets 0..PROBE-1 whose depth is recorded for every q (no selection)


def build(q, nq, POS):
    """marks[j] = sorted list of (gear, member) striking offset i = j - NEG, for
    -NEG <= i < POS.  member 0 = lower (6k-1), 1 = upper (6k+1)."""
    qq = q * q
    W = NEG + POS
    marks = [[] for _ in range(W)]
    for idx in range(nq):
        g = GEARS[idx]
        u = UU[idx]
        r = qq % g
        a = ((2 - r) * u) % g          # lower-member offset class
        b = ((-r) * u) % g             # upper-member offset class
        for off, mem in ((a, 0), (b, 1)):
            j = (off + NEG) % g
            while j < W:
                marks[j].append((g, mem))
                j += g
    return marks


rows = []
depth_sum = [0] * PROBE          # summed depth at offset i over all q
depth_cnt = [0] * PROBE
char_violation = 0
char_examples = []
mod8_violation = 0
g5_off1 = 0
g5_pattern_bad = 0
Lmod5_bad = 0
land_hist = {}

for qi, q in enumerate(GEARS):
    nq = qi + 1
    POS = 768
    while True:
        marks = build(q, nq, POS)
        L = None
        for i in range(1, POS):
            if not marks[i + NEG]:
                L = i
                break
        if L is not None:
            break
        POS *= 2
    k0 = (q * q - 1) // 6
    # ---- blocker sequence on 0..L-1
    word = [marks[i + NEG][0][0] for i in range(L)]          # smallest striker per offset
    depth = [len(marks[i + NEG]) for i in range(L)]
    strikes = sum(depth)
    allg = set()
    for i in range(L):
        for g, m in marks[i + NEG]:
            allg.add(g)
    sole = sorted({marks[i + NEG][0][0] for i in range(L) if len(marks[i + NEG]) == 1})
    # member struck: lower only / upper only / both
    memcls = [0, 0, 0]
    for i in range(L):
        ms = {m for g, m in marks[i + NEG]}
        memcls[0 if ms == {0} else (1 if ms == {1} else 2)] += 1
    # ---- depth at fixed offsets, no path selection
    for i in range(min(PROBE, POS)):
        depth_sum[i] += len(marks[i + NEG])
        depth_cnt[i] += 1
    # ---- left extension / containing stretch
    e = 0
    while e < NEG and marks[NEG - 1 - e]:
        e += 1
    # ---- layer nest: min striker per offset, survivors by layer
    mins = [marks[i + NEG][0][0] for i in range(L)] + [10 ** 9]     # landing survives all
    T_all = max(mins[:L])
    T_int = max(mins[1:L]) if L > 1 else 0
    # largest gap of the layer-g survivor set, for the layers that matter
    maxgap = {}
    order = sorted(range(L), key=lambda i: mins[i])
    pos = 0
    cur = set(range(L + 1))
    for g in sorted(set(mins[:L])):
        while pos < L and mins[order[pos]] <= g:
            cur.discard(order[pos])
            pos += 1
        s = sorted(cur)
        mg = max((s[t + 1] - s[t] for t in range(len(s) - 1)), default=0)
        maxgap[g] = (len(s), mg)
    # ---- bucket vector at the landing
    qq = q * q
    best = 10 ** 9
    bestg = 0
    dq = 0
    rank = 0
    buck = [0] * nq
    for idx in range(nq):
        g = GEARS[idx]
        u = UU[idx]
        r = qq % g
        a = ((2 - r) * u) % g
        b = ((-r) * u) % g
        dist = min((a - L) % g, (b - L) % g)
        buck[idx] = dist
        if dist < best:
            best = dist
            bestg = g
        if g == q:
            dq = dist
    for idx in range(nq):
        if buck[idx] > dq:
            rank += 1
    # ---- gear-5 checks
    if q >= 7:
        r5 = qq % 5
        want = {1, 4} if r5 == 1 else {1, 3}
        got = set()
        for i in range(-NEG, POS):
            for g, m in marks[i + NEG]:
                if g == 5:
                    got.add(i % 5)
        if got != want:
            g5_pattern_bad += 1
        if 5 not in [g for g, m in marks[1 + NEG]]:
            g5_off1 += 1
        if L % 5 not in ({0, 2, 3} if r5 == 1 else {0, 2, 4}):
            Lmod5_bad += 1
    # ---- first column's strikers mod 8
    for g, m in marks[NEG]:
        if g != q and g % 8 not in (1, 7):
            mod8_violation += 1
    land_hist[L] = land_hist.get(L, 0) + 1
    rows.append(dict(q=q, nq=nq, k0=k0, L=L, word=word, depth=depth, strikes=strikes,
                     ndist=len(allg), nword=len(set(word)), sole=sole, e=e,
                     T_all=T_all, T_int=T_int, maxgap=maxgap, memcls=memcls,
                     stopper=bestg, stopdist=best, dq=dq, rank=rank,
                     marks=marks if q in FEATURE else None))

say("paths computed:", len(rows))
Ls = sorted(r["L"] for r in rows)
say("walk length: min %d, median %d, max %d at q = %d" % (
    Ls[0], Ls[len(Ls) // 2], Ls[-1], max(rows, key=lambda r: r["L"])["q"]))
LONGEST = max(rows, key=lambda r: r["L"])["q"]
say("landings all twin pairs:",
    all(is_prime(6 * (r["k0"] + r["L"]) - 1) and is_prime(6 * (r["k0"] + r["L"]) + 1)
        for r in rows))

# =====================================================================  P2, P3, P4
say("")
say("=== the anchor's fixed pattern on the path (P2, P3) and the first column (P4) ===")
say("gear 5 misses offset 1 at:", g5_off1, "of", len(rows) - 1, "paths (q >= 7)")
say("gear 5's offset class set != {1,4} / {1,3} by q^2 mod 5 at:", g5_pattern_bad, "paths")
say("L mod 5 outside its predicted class set at:", Lmod5_bad, "paths")
say("gears != q striking offset 0 that are not +-1 mod 8:", mod8_violation)
byq5 = {}
for r in rows:
    if r["q"] == 5:
        continue
    c = (r["q"] * r["q"]) % 5
    byq5.setdefault(c, []).append(r["L"] % 5)
for c in sorted(byq5):
    from collections import Counter
    say("  q^2 = %d mod 5 (%d paths): L mod 5 histogram %s" % (
        c, len(byq5[c]), sorted(Counter(byq5[c]).items())))

# =====================================================================  P1, P5
say("")
say("=== the offset character law (P1) and the depth profile (P5) ===")


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def chi(i, g):
    """how many of -6i, 2-6i are nonzero quadratic residues mod g"""
    c = 0
    for T in (-6 * i, 2 - 6 * i):
        if legendre(T, g) == 1:
            c += 1
    return c


# P1 gate: rebuild the strike sets over offsets 0..PROBE-1 for every q and check that
# every (gear, offset) hit has chi > 0 or the gear divides the target.
viol = 0
vex = []
for r in rows:
    q = r["q"]
    if q > 500:
        continue
    qq = q * q
    nq = r["nq"]
    for idx in range(nq):
        g = GEARS[idx]
        if g == q:
            continue
        u = UU[idx]
        rr = qq % g
        for off, T in ((((2 - rr) * u) % g, 2), (((-rr) * u) % g, 0)):
            i = off
            while i < PROBE:
                # gear g strikes offset i; the target is T - 6i
                if legendre(T - 6 * i, g) != 1:
                    viol += 1
                    if len(vex) < 5:
                        vex.append((q, g, i))
                i += g
say("hits at an offset whose target is a non-residue for that gear:", viol, vex[:5])

lam = []
for i in range(PROBE):
    s = 0.0
    for idx in range(NG):
        g = GEARS[idx]
        if g > 5000:
            break
        s += 2.0 * chi(i, g) / (g - 1)
    lam.append(s)
meas = [depth_sum[i] / depth_cnt[i] if depth_cnt[i] else 0 for i in range(PROBE)]


def spearman(x, y):
    n = len(x)
    rx = [0] * n
    ry = [0] * n
    for arr, out in ((x, rx), (y, ry)):
        order = sorted(range(n), key=lambda t: arr[t])
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for t in range(i, j + 1):
                out[order[t]] = avg
            i = j + 1
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[t] - mx) * (ry[t] - my) for t in range(n))
    den = (sum((rx[t] - mx) ** 2 for t in range(n)) * sum((ry[t] - my) ** 2 for t in range(n))) ** .5
    return num / den if den else 0.0


say("Spearman(lambda(i), measured mean depth at offset i) over i = 1..%d: %.4f"
    % (PROBE - 1, spearman(lam[1:], meas[1:])))
say("offset : lambda : measured mean depth : times a landing")
for i in list(range(1, 25)):
    say("  %3d   %6.3f   %6.3f   %6d" % (i, lam[i], meas[i], land_hist.get(i, 0)))
lo = sorted(range(1, PROBE), key=lambda i: lam[i])[:8]
hi = sorted(range(1, PROBE), key=lambda i: -lam[i])[:8]
say("lowest-lambda offsets:", [(i, round(lam[i], 3), land_hist.get(i, 0)) for i in lo])
say("highest-lambda offsets:", [(i, round(lam[i], 3), land_hist.get(i, 0)) for i in hi])
tot = sum(land_hist.values())
say("landings on the 8 lowest-lambda offsets: %d of %d (%.3f); on the 8 highest: %d (%.3f)" % (
    sum(land_hist.get(i, 0) for i in lo), tot, sum(land_hist.get(i, 0) for i in lo) / tot,
    sum(land_hist.get(i, 0) for i in hi), sum(land_hist.get(i, 0) for i in hi) / tot))

# =====================================================================  the word, depth, strikes
say("")
say("=== the blocker sequence: word, depth profile, strike count (item 1) ===")
say("total strikes on the path against L: sum L = %d, sum strikes = %d, ratio %.4f" % (
    sum(r["L"] for r in rows), sum(r["strikes"] for r in rows),
    sum(r["strikes"] for r in rows) / sum(r["L"] for r in rows)))
from collections import Counter
dh = Counter()
for r in rows:
    for d in r["depth"]:
        dh[d] += 1
tt = sum(dh.values())
say("depth histogram over all %d path columns: %s" % (tt, sorted(dh.items())))
say("  depth-1 columns: %.4f of all" % (dh[1] / tt))
wl = Counter()
for r in rows:
    for w in r["word"]:
        wl[w] += 1
say("smallest-striker letters, top 10:", wl.most_common(10))
say("distinct letters per path: min %d, median %d, max %d" % (
    min(r["nword"] for r in rows), sorted(r["nword"] for r in rows)[len(rows) // 2],
    max(r["nword"] for r in rows)))
say("distinct strikers (any depth) per path: min %d, median %d, max %d" % (
    min(r["ndist"] for r in rows), sorted(r["ndist"] for r in rows)[len(rows) // 2],
    max(r["ndist"] for r in rows)))
say("sole strikers per path: min %d, median %d, max %d" % (
    min(len(r["sole"]) for r in rows), sorted(len(r["sole"]) for r in rows)[len(rows) // 2],
    max(len(r["sole"]) for r in rows)))
mc = [0, 0, 0]
for r in rows:
    for t in range(3):
        mc[t] += r["memcls"][t]
say("columns struck on the lower member only / upper only / both: %d / %d / %d (%.4f / %.4f / %.4f)"
    % (mc[0], mc[1], mc[2], mc[0] / tt, mc[1] / tt, mc[2] / tt))

# =====================================================================  P6 the tail
say("")
say("=== the path inside its maximal blocked stretch (P6, item 4) ===")
es = sorted(r["e"] for r in rows)
say("left extension e (blocked columns immediately below k_0): min %d, median %d, max %d"
    % (es[0], es[len(es) // 2], es[-1]))
say("k_0 - 1 open (the path IS the whole stretch): %d of %d paths"
    % (sum(1 for r in rows if r["e"] == 0), len(rows)))
say("e >= 1 at %.4f of paths; e >= L at %.4f" % (
    sum(1 for r in rows if r["e"] >= 1) / len(rows),
    sum(1 for r in rows if r["e"] >= r["L"]) / len(rows)))
fr = sorted(r["e"] / (r["e"] + r["L"]) for r in rows)
say("position of k_0 in its stretch, e/(e+L): min %.3f, median %.3f, max %.3f" % (
    fr[0], fr[len(fr) // 2], fr[-1]))
say("stretch length e+L: median %d, max %d" % (
    sorted(r["e"] + r["L"] for r in rows)[len(rows) // 2],
    max(r["e"] + r["L"] for r in rows)))
say("paths whose left extension hit the probe wall (e = %d):" % NEG,
    sum(1 for r in rows if r["e"] == NEG))

# =====================================================================  P9, P10 layer nest
say("")
say("=== the layer nest (P9, P10) ===")
say("largest gear removing an INTERIOR survivor (the last fusion): = q at %d paths"
    % sum(1 for r in rows if r["T_int"] == r["q"]))
say("  those q:", [r["q"] for r in rows if r["T_int"] == r["q"]][:20])
say("largest gear removing ANY survivor (incl. offset 0): = q at %d paths (the square gate)"
    % sum(1 for r in rows if r["T_all"] == r["q"]))
bad10 = []
for r in rows:
    for g, (ns, mg) in r["maxgap"].items():
        if g in F_LOW and mg >= F_LOW[g]:
            bad10.append((r["q"], g, mg, F_LOW[g]))
say("layers where the survivor set's largest gap reaches the lower record F_g:", len(bad10))
say("  first few:", bad10[:8])
say("fusion count (pieces joined by the last-fusion gear) - distribution:")
fus = Counter()
for r in rows:
    if r["L"] > 1:
        t = r["T_int"]
        # survivors at layer just below t, minus the ones t removes
        cnt = sum(1 for i in range(1, r["L"]) if r["word"][i] == t)
        fus[cnt + 1] += 1
say("  ", sorted(fus.items()))

# =====================================================================  P7 bucket at landing
say("")
say("=== the bucket vector at the landing (item 2, P7) ===")
sd = sorted(r["stopdist"] for r in rows)
say("stopper distance (nearest tooth past the landing): min %d, median %d, max %d" % (
    sd[0], sd[len(sd) // 2], sd[-1]))
say("stopper distance = 1 at %d of %d paths" % (
    sum(1 for r in rows if r["stopdist"] == 1), len(rows)))
say("stopper gear histogram, top 8:", Counter(r["stopper"] for r in rows).most_common(8))
say("stopper distance > 1 forces landing = 2 mod 5:",
    all((r["k0"] + r["L"]) % 5 == 2 for r in rows if r["stopdist"] > 1),
    " (count %d)" % sum(1 for r in rows if r["stopdist"] > 1))
rk = sorted(r["rank"] / max(1, GEARS.index(r["q"])) for r in rows)
say("top gear's rank by bucket distance at the landing (0 = farthest), normalised:"
    " min %.4f, median %.4f, max %.4f" % (rk[0], rk[len(rk) // 2], rk[-1]))
say("top gear in the farthest decile at %.4f of paths; the single farthest at %.4f" % (
    sum(1 for x in rk if x <= 0.1) / len(rk), sum(1 for r in rows if r["rank"] == 0) / len(rows)))
say("d - L (the top gear's own bucket at the landing): median %d, min %d" % (
    sorted(r["dq"] for r in rows)[len(rows) // 2], min(r["dq"] for r in rows)))

# =====================================================================  featured paths
say("")
say("=== full paths (item 1: every column, every striker) ===")
FEATURE2 = FEATURE + [LONGEST]
for r in rows:
    if r["q"] not in FEATURE2:
        continue
    q = r["q"]
    mk = r["marks"]
    if mk is None:
        nq = r["nq"]
        POS = r["L"] + 4
        mk = build(q, nq, POS)
    say("")
    say("q = %d, k_0 = %d (q^2 = %d), L = %d, landing column %d -> twin %d | %d"
        % (q, r["k0"], q * q, r["L"], r["k0"] + r["L"],
           6 * (r["k0"] + r["L"]) - 1, 6 * (r["k0"] + r["L"]) + 1))
    say("  left extension e = %d, containing stretch %d columns, q^2 sits at %.3f of it"
        % (r["e"], r["e"] + r["L"], r["e"] / (r["e"] + r["L"])))
    say("  offset | strikers (gear:member, L=6k-1 U=6k+1) | smallest | depth")
    for i in range(r["L"] + 1):
        ss = mk[i + NEG]
        txt = " ".join("%d%s" % (g, "L" if m == 0 else "U") for g, m in ss)
        say("   %4d  | %-56s | %6s | %d" % (i, txt[:56], ss[0][0] if ss else "-", len(ss)))
    say("  word:", " ".join(str(w) for w in r["word"]))
    say("  sole strikers:", r["sole"])
    say("  stopper: gear %d at distance %d; top gear's bucket d - L = %d"
        % (r["stopper"], r["stopdist"], r["dq"]))

LOG.close()
