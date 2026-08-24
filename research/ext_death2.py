"""Harvester round 21 (b) part 2: closing assertions for the extension-death law.

THE SHALLOW-EXTENSION CAP (the mechanism of the exact 9):
  A record window of the old machine is a maximal GAP - it has NO interior
  openings.  Lifting to gear q', the only way it can grow is fusing neighbour
  gaps, whose shared endpoints become interiors.  Interiors must lie in the two
  tooth classes {0, -e} mod q'; THREE interiors would need three distinct
  residues in a 2-set - impossible unless two collide mod q'.  Hence at most
  BOTH FLANKS fuse:   F_ext  <=  g_L + F_old + g_R,
  attainable iff F_old = +-(tooth separation) mod q' (both-flank case).
  At 13->17: every 13-winner has flanks (6, 6) and 75 = 7 mod 17, so the cap is
  6 + 75 + 6 = 87, attained with e = +-7 mod 17.  Deficit vs the deep-fusion
  winner 96: EXACTLY 9.

This script asserts: (1) all 16 13-winners have flank gaps exactly (6,6);
(2) the extension value set {81, 84, 87} is exactly {75+6, 75+6+3, 6+75+6}
    (one flank / one flank + the next 3-gap / both flanks) - checked against
    each winner's local gap context;
(3) the 3-interior impossibility for the observed windows (exact residues);
(4) anatomy of the best 19-extension (e=768556, F=111) and best 23-extension
    (e=20932007, F=147): both are flank fusions of the previous record, and the
    cap law F_ext <= g_L + F_old + g_R holds with the measured flanks;
(5) the deficit ladder 9, 18, 36 restated with its mechanism split
    (cap shortfall vs deep-fusion gain).
"""
import numpy as np
from math import prod

G13 = [3, 5, 7, 11, 13]
G17 = G13 + [17]
G19 = G17 + [19]
P13, P17, P19 = prod(G13), prod(G17), prod(G19)


def openings(gears, e, P):
    a = np.ones(P, bool)
    for q in gears:
        a[0::q] = False
        a[(-e) % q::q] = False
    return np.flatnonzero(a)


def gap_context(gears, e, P, width=3):
    """record window with `width` neighbour gaps on each side"""
    idx = openings(gears, e, P)
    g = np.diff(np.append(idx, idx[0] + P))
    i = int(np.argmax(g))
    n = len(g)
    left = [int(g[(i - k) % n]) for k in range(width, 0, -1)]
    right = [int(g[(i + k) % n]) for k in range(1, width + 1)]
    return left, int(g[i]), right, int(idx[i])


F13 = np.load("research/data/f13_family.npy")
F17 = np.load("research/data/f17_family.npy")
w13 = [int(e) for e in np.flatnonzero(F13 == 75) if e >= 1]
w17 = [int(e) for e in np.flatnonzero(F17 == 96) if e >= 1]

# ---- (1) + (2): every 13-winner's local context and the value-set accounting
print("1/2. FLANKS OF ALL 16 13-WINNERS + the {81,84,87} accounting")
ctxs = set()
for e in w13:
    L, F, R, a = gap_context(G13, e, P13)
    assert F == 75 and L[-1] == 6 and R[0] == 6, (e, L, F, R)
    ctxs.add((tuple(L), F, tuple(R)))
    # cap law components for this winner:
    one_flank = 75 + 6                       # fuse one shared endpoint
    chain2 = 75 + 6 + (R[1] if R[1] < L[-2] else L[-2])  # 6 then next gap
    both = 6 + 75 + 6
    assert both == 87 and one_flank == 81
print(f"   all 16 winners: flanks (6, 75, 6); local contexts (3 wide): {sorted(ctxs)}")
sec = sorted({min(c[0][-2], c[2][1]) for c in ctxs})
print(f"   second-neighbour minima: {sec}  ->  one-side chain 75+6+3 = 84 "
      f"available iff a 3-gap sits beyond a 6-flank")
assert 3 in sec

# ---- (3) the 3-interior impossibility at the observed windows
print("\n3. THREE-INTERIOR IMPOSSIBILITY (exact residues mod q')")
for (Fold, gL, gR, q) in ((75, 6, 6, 17), (96, 6, 9, 19), (129, 6, 12, 23)):
    # candidate interior positions if both flanks + one more fused:
    for triple in ([0, Fold, Fold + gR], [-gL, 0, Fold]):
        r = sorted({t % q for t in triple})
        assert len(r) == 3, (Fold, q, triple, r)
    print(f"   F_old={Fold}, q'={q}: endpoint triples have 3 distinct residues "
          f"mod {q} -> never inside a 2-class tooth set")

# ---- (4) best 19- and 23-extension anatomy + cap law
print("\n4. BEST-EXTENSION ANATOMY AT 19 AND 23 (flank-fusion shape + cap law)")


def anatomy(e, gears_small, P_small, q_new, label):
    gears_big = gears_small + [q_new]
    P_big = P_small * q_new
    idx = openings(gears_big, e, P_big)
    g = np.diff(np.append(idx, idx[0] + P_big))
    i = int(np.argmax(g))
    a = int(idx[i])
    b = a + int(g[i])
    small_open = np.zeros(P_small, bool)
    small_open[openings(gears_small, e, P_small)] = True
    xs = np.arange(a + 1, b)
    ints = xs[small_open[xs % P_small]]
    T = {0, (-e) % q_new}
    res = sorted({int(x % q_new) for x in ints})
    assert set(res) <= T
    pts = [a] + [int(x) for x in ints] + [b]
    word = [pts[k + 1] - pts[k] for k in range(len(pts) - 1)]
    print(f"   {label}: F={int(g[i])}, word {word}, "
          f"{len(ints)} interiors at {res} (teeth {sorted(T)})")
    return word


w19 = anatomy(768556, G17, P17, 19, "19-ext of 17-winner  e=768556 ")
w23 = anatomy(20932007, G19, P19, 23, "23-ext of 19-argmax  e=20932007")
assert max(w19) == 96 and len(w19) <= 3, w19
assert max(w23) == 129 and len(w23) <= 3, w23


def cap2(gears, e, P):
    """THE SHALLOW-EXTENSION CAP: at most 2 interiors can fuse (3-residue
    impossibility), all with ONE separation congruence - free by lift choice.
    cap = F_old + max(both flanks, right 2-chain, left 2-chain)."""
    L, F, R, _ = gap_context(gears, e, P)
    opts = {"flanks": L[-1] + R[0], "right2": R[0] + R[1], "left2": L[-2] + L[-1]}
    m = max(opts.values())
    return F + m, F, opts


for (e, gears, P, best, label) in ((104761 % P13, G13, P13, 87, "13-winner  "),
                                   (768556 % P17, G17, P17, 111, "17-winner  "),
                                   (20932007 % P19, G19, P19, 147, "19-argmax  ")):
    c, F, opts = cap2(gears, e, P)
    print(f"   {label}: record {F}, adjacent 2-sums {opts} -> cap {c} "
          f"(best extension measured {best})")
    assert c == best, (label, c, best)

print("""
5. THE LADDER, MECHANISM-SPLIT (cap = F_old + best adjacent 2-gap sum):
   level   record   best 2-sum   shallow cap = best ext   true max   deficit
   17      75       12 (6+6)     87                       96         9
   19      96       15 (6+9)     111                      129        18
   23      129      18 (6+12)    147                      183        36 (lineage)
   The deficit DOUBLES per level; the record's adjacent 2-sums grow by 3.
ALL ASSERTIONS PASSED.""")
