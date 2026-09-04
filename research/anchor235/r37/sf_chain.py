"""R2.a self-feeding, part 2: the chain of landings.

g_0 = q;  g_{n+1} = the landing of the layered walk from the column holding g_n^2 under the
machine {5..g_n} - i.e. the first twin pair at or above g_n^2, whose shared first tooth is
the landing column.  Chains from every prime q <= 200, iterated while g_n < 10^6 (so the
landing stays inside 12 digits).

Per level we record: the walk length, the blocked columns' smallest strikers (the layer each
hop is made at), the top hop layer T = max of those, the layer of the first and last hop, and
the landing's residues.

Also: P2.2, the square-gate rule for the FIRST hop, over every prime q <= 5000.

Writes results/sf_chain.txt.
"""
import os
from math import isqrt
from collections import Counter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "sf_chain.txt"), "w")


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


LIM = 1_000_003
FL = sieve(LIM)
P = [i for i in range(2, LIM + 1) if FL[i]]
say("primes to", LIM, ":", len(P))
GEARS = [p for p in P if p >= 5]


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def min_striker(x, top, ng):
    """smallest gear <= top dividing 6x-1 or 6x+1; 0 if the column is open under {5..top}.
    ng = number of gears <= top (precomputed index)."""
    lo, hi = 6 * x - 1, 6 * x + 1
    for i in range(ng):
        p = GEARS[i]
        if lo % p == 0 or hi % p == 0:
            return p
    return 0


def walk(g):
    """from the column holding g^2, under {5..g}: returns (k0, landing, striker list)."""
    ng = 0
    while ng < len(GEARS) and GEARS[ng] <= g:
        ng += 1
    k0 = (g * g - 1) // 6
    x = k0
    strik = []
    while True:
        s = min_striker(x, g, ng)
        if s == 0:
            return k0, x, strik
        strik.append(s)
        x += 1


# ------------------------------------------------------------------ the chains
say("")
say("=== the chains: g_{n+1} = first twin pair at or above g_n^2 ===")
say("q -> levels (g_n, walk length L_n, top hop layer T_n, first hop, last hop)")

chains = {}
for q in [p for p in P if 5 <= p <= 200]:
    g = q
    lev = []
    while g < 1_000_000:
        k0, k, strik = walk(g)
        gn = 6 * k - 1
        assert is_prime(gn) and is_prime(gn + 2), (g, gn)
        lev.append(dict(g=g, k0=k0, k=k, L=k - k0, T=max(strik) if strik else 0,
                        first=strik[0] if strik else 0, last=strik[-1] if strik else 0,
                        nblocked=len(strik), strik=strik, gnext=gn))
        g = gn
    lev.append(dict(g=g, L=None))
    chains[q] = lev
    say("q=%-4d" % q, " -> ".join(
        "%d(L=%s,T=%s,f=%s,l=%s)" % (r["g"], r["L"], r["T"], r["first"], r["last"])
        if r["L"] is not None else "%d" % r["g"] for r in lev))

# ------------------------------------------------------------------ P2.6 merges
say("")
say("=== P2.6 chain merges ===")
seen = {}
merges = 0
for q, lev in chains.items():
    for r in lev:
        seen.setdefault(r["g"], set()).add(q)
multi = {g: s for g, s in seen.items() if len(s) > 1 and g > 200}
say("values of g_n (above the starting range) reached by more than one chain:", len(multi))
if multi:
    for g, s in list(multi.items())[:10]:
        say("   ", g, "from", sorted(s))

# ------------------------------------------------------------------ P2.4 top layer
say("")
say("=== P2.4 the top hop layer T_n along a chain ===")
say("q: T_0, T_1, ...   (with g_n for scale)")
for q, lev in chains.items():
    if len([r for r in lev if r["L"] is not None]) >= 2:
        say("  q=%-4d" % q, "T:", [r["T"] for r in lev if r["L"] is not None],
            " g:", [r["g"] for r in lev if r["L"] is not None])
rows = [r for lev in chains.values() for r in lev if r["L"] is not None]
say("levels computed:", len(rows))
say("T_n / g_n over all levels: min %.4f max %.4f" % (
    min(r["T"] / r["g"] for r in rows), max(r["T"] / r["g"] for r in rows)))
say("T_n <= sqrt(g_n)?  exceptions:", sum(1 for r in rows if r["T"] > isqrt(r["g"])), "of", len(rows))
say("T_n == g_n (the top gear itself makes the deepest hop)?  count:",
    sum(1 for r in rows if r["T"] == r["g"]))
# is T_{n+1} a function of T_n?
pairs = []
for lev in chains.values():
    ls = [r for r in lev if r["L"] is not None]
    for a, b in zip(ls, ls[1:]):
        pairs.append((a["T"], b["T"]))
d = {}
conf = 0
for a, b in pairs:
    if a in d and d[a] != b:
        conf += 1
    d[a] = b
say("consecutive (T_n, T_{n+1}) pairs:", len(pairs), " distinct T_n with conflicting successors:", conf)

# ------------------------------------------------------------------ P2.5 walk lengths
say("")
say("=== P2.5 walk lengths along a chain ===")
mono = 0
tot = 0
for q, lev in chains.items():
    ls = [r["L"] for r in lev if r["L"] is not None]
    if len(ls) >= 2:
        tot += 1
        if all(x < y for x, y in zip(ls, ls[1:])):
            mono += 1
say("chains with >= 2 levels:", tot, " strictly increasing walk lengths:", mono)
say("L_n vs g_n:  L, g pairs, all levels (L, log10 g):")
say("  " + ", ".join("(%d,%.1f)" % (r["L"], __import__("math").log10(r["g"])) for r in rows[:40]))
say("L_n <= g_n?  exceptions:", sum(1 for r in rows if r["L"] > r["g"]))
say("max L over all levels:", max(r["L"] for r in rows), " min:", min(r["L"] for r in rows))

# ------------------------------------------------------------------ P2.3 residues
say("")
say("=== P2.3 residues of g_{n+1} modulo the hop gears of level n ===")
cnt = Counter()
zero = 0
tot_r = 0
tooth = 0
for r in rows:
    for p in set(r["strik"]):
        u = pow(6, -1, p)
        kk = r["k"] % p
        tot_r += 1
        if (6 * r["gnext"]) % p == 0:
            zero += 1
        if kk in (u % p, (-u) % p):
            tooth += 1
say("(level, hop gear) pairs:", tot_r)
say("  landing column on a hop gear's tooth:", tooth, "(must be 0: the landing is an opening)")
say("  hop gear dividing g_{n+1} or g_{n+1}+2:", zero, "(must be 0: same statement)")
say("  so the only residue constraint is non-divisibility - the definition of an opening.")
say("residue of g_{n+1} mod g_n is the excess d = g_{n+1} - g_n^2 = 6 L_n - 2:")
say("  " + ", ".join("%d==6*%d-2" % ((r["gnext"] - r["g"] * r["g"]), r["L"]) for r in rows[:12]))
say("  exceptions to g_{n+1} - g_n^2 == 6 L_n - 2:",
    sum(1 for r in rows if r["gnext"] - r["g"] * r["g"] != 6 * r["L"] - 2))

# ------------------------------------------------------------------ P2.2 square gate
say("")
say("=== P2.2 the first hop of the walk and the square gate ===")
bad = 0
tot2 = 0
gate = 0
for q in [p for p in P if 5 <= p <= 5000]:
    tot2 += 1
    x = (q * q - 1) // 6
    ng = 0
    while ng < len(GEARS) and GEARS[ng] <= q:
        ng += 1
    s = min_striker(x, q, ng)
    sq2 = q * q - 2
    lp = 0
    for p in GEARS:
        if p * p > sq2:
            break
        if sq2 % p == 0:
            lp = p
            break
    pred = q if lp == 0 else min(lp, q)
    if s != pred:
        bad += 1
    if is_prime(sq2):
        gate += 1
say("primes q in 5..5000:", tot2)
say("  first blocked column of the walk is always the one holding q^2 (upper member q^2).")
say("  its smallest striker = q iff q^2-2 is prime, else lpf(q^2-2):  exceptions", bad)
say("  q with q^2-2 prime (the square gate open):", gate, "of", tot2)

# ------------------------------------------------------------------ hop layer census
say("")
say("=== hop layers over all levels (which gear each traversed column is counted at) ===")
c = Counter()
for r in rows:
    c.update(r["strik"])
tot3 = sum(c.values())
say("blocked columns traversed:", tot3)
say("  by layer: " + ", ".join("%d:%d" % (g, n) for g, n in sorted(c.items())[:15]))
say("  layers above 100:", sum(n for g, n in c.items() if g > 100),
    " above 1000:", sum(n for g, n in c.items() if g > 1000))
say("  largest layer used:", max(c))

LOG.close()
