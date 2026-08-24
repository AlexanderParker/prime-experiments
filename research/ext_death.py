"""Harvester round 21 (b): THE CLEAN-EXTENSION DEATH AT 17 - mechanism of the exact 9.

Round-20 fact: winners extend winners at 7->11 and 11->13 (16/16), then NEVER again;
the best 17-extension of a 13-winner reaches F = 87 vs the true max 96 - it loses by
exactly 9, unexplained.  This script finds the mechanism as EXACT EVENTS:

  1. Reconstruct the extension table: every 13-winner (16 of them, F_13 = 75,
     profile (1,1,1,3,6)) lifted to all 17 classes (e + k*15015, k = 0..16), F_17
     read from the exhaustive family array f17_family.npy.  Best = 87 (assert).
  2. ANATOMY of the maximal windows via the merge frame: in the 17-machine of e,
     position x is killed by 17 iff x = 0 or -e (mod 17).  A new max window is a run
     of 13-machine gaps whose INTERIOR openings all fall in the two classes
     {0, -e} mod 17 while the flanks do not.  Decompose the F=87 window (best
     extension) and the F=96 window (true winner): how many old gaps fuse, which
     residues the interiors occupy, what the base word is.
  3. The mechanism number: SHALLOW vs DEEP fusion.  For the 13-winners the max
     window at 13 (the 75-gap) has NO interior openings; extension can only fuse
     NEIGHBOUR gaps whose shared endpoints get killed - few interiors.  The 96
     winner is a deep fusion on a mediocre base.  Measure both exactly.
  4. THE LADDER: does the death persist and grow?  Best 19-extension of every
     17-winner (all lifts, direct sieve at P19 = 4,849,845) vs true max 129;
     best 23-extension of the 19-argmax e = 1,532,627 (all lifts, direct sieve at
     P23 = 111,546,435) vs true max 183.  Deficit sequence: 9, ?, ?.
  5. Window anatomy of the 19-argmax's 129-window over the 17-machine and the
     23-argmax's 183-window over the 19-machine: fusion depth at each level.

Everything exact; assertions on all previously-claimed values.
"""
import numpy as np
from math import prod

P13 = prod([3, 5, 7, 11, 13])            # 15015
P17 = prod([3, 5, 7, 11, 13, 17])        # 255255
P19 = prod([3, 5, 7, 11, 13, 17, 19])    # 4849845
P23 = prod([3, 5, 7, 11, 13, 17, 19, 23])  # 111546435

G13 = [3, 5, 7, 11, 13]
G17 = G13 + [17]
G19 = G17 + [19]
G23 = G19 + [23]


def openings(gears, e, P, buf=None):
    a = np.ones(P, bool) if buf is None else buf
    if buf is not None:
        a[:] = True
    for q in gears:
        a[0::q] = False
        a[(-e) % q::q] = False
    return np.flatnonzero(a)


def F_of(gears, e, P, buf=None):
    idx = openings(gears, e, P, buf)
    g = np.diff(np.append(idx, idx[0] + P))
    return int(g.max())


def max_window(gears, e, P):
    """(left opening, right opening, F) of the maximal cyclic gap"""
    idx = openings(gears, e, P)
    g = np.diff(np.append(idx, idx[0] + P))
    i = int(np.argmax(g))
    a = int(idx[i])
    b = int(idx[(i + 1) % len(idx)]) if i + 1 < len(idx) else int(idx[0]) + P
    if i + 1 == len(idx):
        b = int(idx[0]) + P
    return a, b, int(g[i])


def anatomy(e, gears_small, P_small, q_new, P_big, label):
    """decompose the max window of the (gears_small + q_new)-machine of e over the
    small machine: which small-machine openings are interior (killed by q_new)"""
    gears_big = gears_small + [q_new]
    a, b, F = max_window(gears_big, e, P_big)
    # small-machine openings in (a, b): x with x mod P_small in small open set
    small_open = np.zeros(P_small, bool)
    small_open[openings(gears_small, e, P_small)] = True
    xs = np.arange(a + 1, b)
    ints = xs[small_open[xs % P_small]]
    T = {0, (-e) % q_new}
    res = sorted({int(x % q_new) for x in ints})
    assert set(res) <= T, (label, res, T)
    # base word: the fused small-machine gaps
    pts = [a] + [int(x) for x in ints] + [b]
    word = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    # flank check
    assert a % q_new not in T and b % q_new not in T
    print(f"  {label}: F={F}, window [{a},{b}]")
    print(f"     e mod {q_new} = {e % q_new} (teeth {{0,{(-e) % q_new}}}), "
          f"interior openings: {len(ints)} at residues {res} mod {q_new}")
    print(f"     fused base word ({len(word)} old gaps): {word}")
    return F, len(word), word


def profile(e, gears):
    return tuple(min(e % q, q - e % q) for q in gears)


# ---------------------------------------------------------------- load families
F13 = np.load("research/data/f13_family.npy")
F17 = np.load("research/data/f17_family.npy")
assert F13[1] == 33 and F13[1:].max() == 75
assert F17[1] == 54 and F17[1:].max() == 96
w13 = [int(e) for e in np.flatnonzero(F13 == 75) if e >= 1]
w17 = [int(e) for e in np.flatnonzero(F17 == 96) if e >= 1]
assert len(w13) == 16, len(w13)
print(f"13-winners: {len(w13)} (F=75), 17-winners: {len(w17)} (F=96)")
print(f"17-winner profiles: {sorted(set(profile(e, G17) for e in w17))}")

# ------------------------------------------- 1. the extension table 13 -> 17
print("\n" + "=" * 78)
print("1. EXTENSION TABLE: all 16 13-winners x all 17 lifts (F_17 by family array)")
print("=" * 78)
best = {}
allvals = []
for e0 in w13:
    vals = []
    for k in range(17):
        e = e0 + k * P13
        er = min(e % P17, (-e) % P17)
        f = int(F17[er])
        vals.append((f, e, er, e % 17))
        allvals.append(f)
    vals.sort(reverse=True)
    best[e0] = vals[0]
mx = max(v[0] for v in best.values())
print(f"per-winner best extensions: {sorted(set(v[0] for v in best.values()))}")
print(f"value distribution over all 16x17 = 272 lifts: "
      f"{np.bincount(allvals)[np.unique(allvals)].tolist()} at {np.unique(allvals).tolist()}")
assert mx == 87, mx
e_ext = [v for v in best.values() if v[0] == 87]
print(f"BEST EXTENSION F = 87 attained by {len(e_ext)} lift(s): "
      f"{[(e, f'e mod 17={s}') for f, e, er, s in e_ext]}")

# ------------------------------------------- 2. anatomy at 17
print("\n" + "=" * 78)
print("2. WINDOW ANATOMY AT 17 (merge frame: kills at x = 0 or -e mod 17)")
print("=" * 78)
f, e_ext_full, e_ext_r, s_ext = e_ext[0]
Fx, jx, wx = anatomy(e_ext_r, G13, P13, 17, P17, f"best extension e={e_ext_r}")
res_ext = (Fx, jx, wx)
win_anat = []
for e in w17[:4]:
    win_anat.append(anatomy(e, G13, P13, 17, P17, f"true winner   e={e}"))

# the 13-winner's own 75-window: neighbours
print("\n  13-winner base structure (the shallow-extension cap):")
for e0 in w13[:3]:
    a, b, F75 = max_window(G13, e0, P13)
    idx = openings(G13, e0, P13)
    i = int(np.where(idx == a)[0][0])
    gL = int(a - idx[i - 1]) if i > 0 else int(a + P13 - idx[-1])
    j = int(np.where(idx == b % P13)[0][0]) if b < P13 else 0
    gR = int(idx[(j + 1) % len(idx)] - idx[j]) if b < P13 else int(idx[1] - idx[0])
    print(f"    e={e0}: 75-window [{a},{b}], neighbour gaps L={gL}, R={gR}, "
          f"endpoints sep 75 = {75 % 17} mod 17")

# F13 values of the 17-winners' restrictions
r13 = [min(e % P13, (-e) % P13) for e in w17]
v13 = sorted(set(int(F13[r]) for r in r13))
print(f"\n  17-winners restrict to F_13 values {v13} (family max 75) - mediocre bases")

# ------------------------------------------- 4. the ladder: 17 -> 19 -> 23
print("\n" + "=" * 78)
print("3. THE DEFICIT LADDER: 19-extensions of all 17-winners, 23-extension of the")
print("   19-argmax (direct sieves)")
print("=" * 78)
buf19 = np.empty(P19, bool)
best19, arg19 = 0, None
vals19 = []
for e0 in w17:
    for k in range(19):
        e = e0 + k * P17
        f = F_of(G19, e, P19, buf19)
        vals19.append(f)
        if f > best19:
            best19, arg19 = f, e
print(f"best 19-extension of any 17-winner: F = {best19} at e = {arg19} "
      f"(true max 129) -> DEFICIT {129 - best19}")
print(f"  ({len(vals19)} lifts evaluated; top values "
      f"{sorted(vals19, reverse=True)[:6]})")
assert best19 < 129

# 19-argmax anatomy + its 23 extensions
e19 = 1_532_627
f17_of_e19 = int(F17[min(e19 % P17, (-e19) % P17)])
assert f17_of_e19 == 54, f17_of_e19
print(f"\n19-argmax e = {e19}: F_17 = 54 (the twin's own value - a mediocre base)")
anatomy(e19, G17, P17, 19, P19, "19-argmax 129-window over 17-machine")

best23, arg23 = 0, None
for k in range(23):
    e = e19 + k * P19
    f = F_of(G23, e, P23)
    if f > best23:
        best23, arg23 = f, e
print(f"\nbest 23-extension of the 19-argmax: F = {best23} at e = {arg23} "
      f"(true max 183) -> DEFICIT {183 - best23}")
assert best23 < 183

e23 = 107_207_699
f19_of_e23 = F_of(G19, e23 % P19, P19, buf19)
assert f19_of_e23 == 81
print(f"\n23-argmax e = {e23}: F_19 = 81 (family max 129)")
anatomy(e23, G17, P17, 19, P19, "23-argmax at 19 over 17-machine")
anatomy(e23, G19, P19, 23, P23, "23-argmax 183-window over 19-machine")

print("\nALL ASSERTIONS PASSED.")
