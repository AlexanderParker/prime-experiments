"""R2.a self-feeding, part 3: what the newest gear does to its own walk.

For every prime q in 5..5000, the layered walk from the column holding q^2 under {5..q}
(the 9c object).  New measurements:

 W1  the walk starts on the -u tooth of q (6 k0 = q^2 - 1 = -1 mod q), so gear q's next
     strike is exactly d_q = 2 u_q columns higher: the top gear strikes the walk interval
     exactly once - at its first column - whenever L < 2 u_q.  Exception count and the
     largest q at which it fails.
 W2  T = the top hop layer (largest smallest-striker over the traversed columns).  Candidate:
     T = q iff q^2 - 2 is prime (the square gate).  Exception count, and the mechanism of
     each exception.
 W3  how many distinct gears the walk uses, and how large the largest one is.
 W4  the same for the pair born at the landing: it strikes the NEXT level's walk interval
     once, at its first column, and its next strike is 2k = (g+1)/3 columns on.

Writes results/sf_walks.txt.
"""
import os
from math import isqrt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "sf_walks.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


FL = sieve(25_000_100)
PR = [i for i in range(2, 5001) if FL[i]]
GEARS = [p for p in PR if p >= 5]
say("gears to 5000:", len(GEARS))


def min_striker(x, ng):
    lo, hi = 6 * x - 1, 6 * x + 1
    for i in range(ng):
        p = GEARS[i]
        if lo % p == 0 or hi % p == 0:
            return p
    return 0


rows = []
for q in GEARS:
    ng = 0
    while ng < len(GEARS) and GEARS[ng] <= q:
        ng += 1
    k0 = (q * q - 1) // 6
    x = k0
    strik = []
    while True:
        s = min_striker(x, ng)
        if s == 0:
            break
        strik.append(s)
        x += 1
    c = pow(6, -1, q)                       # teeth are +-c; u_q = min(c, q-c) = round(q/6)
    u = min(c, q - c)
    d = (2 * c) % q                         # k0 = -c (mod q); distance up to the other tooth
    rows.append(dict(q=q, k0=k0, k=x, L=x - k0, strik=strik, u=u, c=c, d=d,
                     T=max(strik) if strik else 0))
say("walks computed:", len(rows), " (every prime gear 5..5000)")
say("walk length: min %d, median %d, max %d at q = %d" % (
    min(r["L"] for r in rows), sorted(r["L"] for r in rows)[len(rows) // 2],
    max(r["L"] for r in rows), max(rows, key=lambda r: r["L"])["q"]))
say("every landing a twin pair:", all(FL[6 * r["k"] - 1] and FL[6 * r["k"] + 1] for r in rows))

# ------------------------------------------------------------------ W1
say("")
say("=== W1. the top gear strikes its own walk exactly once ===")
say("6 k0 = q^2 - 1 = -1 (mod q) and 6 c = 1, so k0 = -c: the walk starts ON a tooth of q.")
say("The other tooth is +c, so the next strike of q is d = 2c mod q columns higher:")
say("  d = 2 u_q       for q = 5 mod 6   (u_q = round(q/6) = c)")
say("  d = q - 2 u_q   for q = 1 mod 6   (c = q - u_q)")
mis = 0
for r in rows:
    q = r["q"]
    if r["k0"] % q != (-r["c"]) % q:
        mis += 1
    nxt = min(i for i in range(1, q + 1)
              if (r["k0"] + i) % q in (r["c"] % q, (-r["c"]) % q))
    if nxt != r["d"]:
        mis += 1
say("  k0 = -c and next strike at d: exceptions over the 667 walks:", mis)
bad = [r for r in rows if r["L"] >= r["d"]]
say("walks with L >= d (the top gear strikes its own walk interval twice):", len(bad),
    "of", len(rows))
say("  they are (q, L, d):", [(r["q"], r["L"], r["d"]) for r in bad])
rest = [r for r in rows if r["q"] > (max(r["q"] for r in bad) if bad else 0)]
top = max(rest, key=lambda r: r["L"] / r["d"])
say("  above the last exception the tightest walk is q = %d: L = %d against d = %d"
    " (ratio %.4f)" % (top["q"], top["L"], top["d"], top["L"] / top["d"]))
say("  L/d over those: median %.4f, 90th pct %.4f, max %.4f" % (
    sorted(r["L"] / r["d"] for r in rest)[len(rest) // 2],
    sorted(r["L"] / r["d"] for r in rest)[int(0.9 * len(rest))],
    max(r["L"] / r["d"] for r in rest)))
hitcount = 0
for r in rows:
    hits = sum(1 for x in range(r["k0"], r["k"] + 1)
               if x % r["q"] in (r["c"] % r["q"], (-r["c"]) % r["q"]))
    if hits != 1 + (1 if r["L"] >= r["d"] else 0):
        hitcount += 1
say("  direct count of gear-q strikes inside [k0, landing] against the rule: exceptions",
    hitcount)

# ------------------------------------------------------------------ W2
say("")
say("=== W2. the top hop layer and the square gate ===")
gate = [r for r in rows if FL[r["q"] * r["q"] - 2]]
say("q with q^2 - 2 prime (square gate open):", len(gate))
e1 = [r for r in gate if r["T"] != r["q"]]
e2 = [r for r in rows if r["T"] == r["q"] and not FL[r["q"] * r["q"] - 2]]
say("  gate open but T != q:", len(e1), [r["q"] for r in e1][:10])
say("  gate closed but T == q:", len(e2), [r["q"] for r in e2][:20])
def ngof(q):
    n = 0
    while n < len(GEARS) and GEARS[n] <= q:
        n += 1
    return n


for r in e2[:6]:
    q = r["q"]
    cols = [x for x in range(r["k0"], r["k"]) if min_striker(x, ngof(q)) == q]
    say("    q=%d: columns in the walk whose smallest striker is q:" % q, cols,
        " (6x+1)/q =", [(6 * c + 1) // q if (6 * c + 1) % q == 0 else (6 * c - 1) // q for c in cols])
say("  so T = q iff the square gate is open OR the walk reaches a column whose member is")
say("  q times a prime >= q - the second clause needs L >= d, the W1 threshold.")

# ------------------------------------------------------------------ W3
say("")
say("=== W3. how many gears the walk uses ===")
nd = [len(set(r["strik"])) for r in rows]
say("distinct hop layers per walk: min %d, median %d, max %d" % (
    min(nd), sorted(nd)[len(nd) // 2], max(nd)))
say("gears of the machine: pi(q) - 2; fraction used, median: %.4f" % (
    sorted(len(set(r["strik"])) / max(1, len([p for p in GEARS if p <= r["q"]])) for r in rows)[len(rows) // 2]))
big = [r for r in rows if r["T"] > isqrt(r["q"])]
say("walks whose largest hop layer exceeds sqrt(q):", len(big), "of", len(rows))
say("hops made by gears > sqrt(q):",
    sum(sum(1 for s in r["strik"] if s * s > r["q"]) for r in rows),
    " of", sum(len(r["strik"]) for r in rows))
say("hops by layer 5:", sum(sum(1 for s in r["strik"] if s == 5) for r in rows),
    " layer 7:", sum(sum(1 for s in r["strik"] if s == 7) for r in rows))

# ------------------------------------------------------------------ W4
say("")
say("=== W4. the pair born at the landing, acting on the NEXT level's walk ===")
say("pair (g, g+2) born at column k; next walk starts at k0' = k(g-1) = 6k^2 - 2k;")
say("gear g strikes k0' (its member is g^2) and next at k0' + 2k = 6k^2 = the twin-product")
say("column; gear g+2 strikes 6k^2 and nothing in between.")
chk = 0
bad4 = 0
brute = 0
for r in rows:
    k = r["k"]
    g = 6 * k - 1
    if not (FL[g] and FL[g + 2]):
        continue
    chk += 1
    k0p = k * (g - 1)
    h = g + 2
    if (6 * k0p + 1) != g * g:
        bad4 += 1
    # residues: k0p = -k (mod g) and -3k (mod h); teeth of both gears are {+-k}
    if k0p % g != (-k) % g or k0p % h != (-3 * k) % h:
        bad4 += 1
    # the predicted next strike of each gear is at offset 2k
    if (6 * (k0p + 2 * k) + 1) % g and (6 * (k0p + 2 * k) - 1) % g:
        bad4 += 1
    if (6 * (k0p + 2 * k) + 1) % h and (6 * (k0p + 2 * k) - 1) % h:
        bad4 += 1
    if k <= 3000:                      # brute-force confirmation on the small pairs
        brute += 1
        for i in range(1, 2 * k):
            c = k0p + i
            if (6 * c - 1) % g == 0 or (6 * c + 1) % g == 0 or                (6 * c - 1) % h == 0 or (6 * c + 1) % h == 0:
                bad4 += 1
                break
say("landings checked:", chk, " brute-forced over the whole 2k gap:", brute, " failures:", bad4)
say("so the two newest gears strike the next walk interval exactly once each, both at the")
say("distance 2k = (g+1)/3, provided the next walk is shorter than 2k.")
nxtL = {r["q"]: r["L"] for r in rows}
ex = 0
tot = 0
for r in rows:
    k = r["k"]
    g = 6 * k - 1
    if g in nxtL:
        tot += 1
        if nxtL[g] >= 2 * k:
            ex += 1
say("landings whose own walk we also computed (g <= 5000):", tot, " with L(g) >= 2k:", ex)

LOG.close()
