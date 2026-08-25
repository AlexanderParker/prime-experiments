"""Round 23 (mechanic): THE DELETION-LADDER BOUND, and its free corollaries.

LEMMA.  For any machine M (gears 5..y) and any r NEW gears q_1..q_r (primes
not in M),   F_{r+1}(M)  <=  F(M + q_1 + ... + q_r).

PROOF.  Let x_0 < ... < x_{r+1} be consecutive M-openings realising F_{r+1}(M),
so x_{r+1} - x_0 = F_{r+1}(M) and the r interiors x_1..x_r are the only
M-openings strictly inside.  The new machine's openings are
{ x : x mod P in E(M),  x mod q_i not in {+-6^{-1}} for each i }.  Translating
the window by j periods P sends x_i to x_i + jP, and the map
j |-> (jP mod q_1, ..., jP mod q_r) is a bijection onto the product of the
Z_{q_i} (CRT; P is invertible mod each q_i).  So choose the single j for which
x_i + jP = 6^{-1} mod q_i for every i: then EVERY interior of the translated
window is killed, by its own gear.  The endpoints may or may not survive; if
they do the new gap is exactly F_{r+1}(M), if they do not the new gap that
contains the window is strictly longer.  Either way
F(M + q_1 + ... + q_r) >= F_{r+1}(M).   [] (r = 1 is merge-law.md's
"F(M+q') >= F2(M) unconditionally"; the content here is that r new gears buy
r levels of the F_j ladder, one designated kill each.)

USE: it converts the corpus F-ladder into FREE EXACT UPPER BOUNDS on the F_j
of machines nobody can scan - e.g. F_2(41) <= F(43) = 103 turned an 11-value
SAT descent into what it is, and F_3(41) <= F(47) <= F(53) = 145 caps
Constructor's A_kill(41) enumeration without any computation at all.

This script asserts the lemma at every (M, j) where both sides are known
exactly, and prints the corollaries.
"""
GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

# F(M), slot frame, EXACT (corpus twin ladder F(2,y)/3 + independent scans)
F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
     41: 91, 43: 103, 53: 145}
# F_j(M), j = 1..: exact full-period values (mechanic C11 + r21 upgrades)
FJ = {13: [11, 16, 23, 26, 28, 31],
      17: [18, 25, 28, 33, 35, 40],
      19: [25, 31, 35, 38, 47, 50],
      23: [34, 39, 50, 58, 65, 77, 83, 88],
      29: [43, 55, 65, 70, 85, 90],
      31: [58, 68, 85, 90, 92, 97, 104, 110],
      37: [88, 90, 97, 105, 113, 120]}


def ahead(y, r):
    """the machine r gears above y"""
    i = GEARS.index(y)
    return GEARS[i + r] if i + r < len(GEARS) else None


checks = tight = 0
print("LEMMA CHECK   F_j(M) <= F(M + (j-1) more gears)")
print("   M    j   F_j(M)   target machine   F(target)   margin")
for y, row in sorted(FJ.items()):
    for j in range(2, len(row) + 1):
        t = ahead(y, j - 1)
        if t is None or t not in F:
            print(f"  {y:3d}  {j:3d}   {row[j-1]:6d}   {t}"
                  f"{'':13s}  unknown     -   (gives F({t}) >= {row[j-1]})")
            continue
        lhs, rhs = row[j - 1], F[t]
        assert lhs <= rhs, (y, j, lhs, t, rhs)
        checks += 1
        if lhs == rhs:
            tight += 1
        print(f"  {y:3d}  {j:3d}   {lhs:6d}   {t:^15d}  {rhs:6d}"
              f"      {rhs-lhs:+4d}{'   TIGHT' if lhs == rhs else ''}")
print(f"\n  {checks} checks, ALL PASS (asserted); {tight} attained with equality")

print("\nFREE COROLLARIES (no computation), for machines past the scan wall:")
for y, j in [(41, 2), (41, 3), (41, 4), (43, 2), (43, 3), (47, 2), (37, 4)]:
    t = ahead(y, j - 1)
    cap = F.get(t)
    if cap is None:
        # fall back to monotonicity F(t) <= F(t') for a bigger known machine
        bigger = [z for z in GEARS if z > t and z in F]
        if bigger:
            cap = F[min(bigger)]
            print(f"  F_{j}({y}) <= F({t}) <= F({min(bigger)}) = {cap}"
                  f"   (monotone in the gear set)")
        continue
    print(f"  F_{j}({y}) <= F({t}) = {cap}")
print("\n  and in the other direction, lower bounds on unscanned F:")
print(f"  F(47) >= F_4(37) = {FJ[37][3]};  F(47) >= F_6(29) = {FJ[29][5]};"
      f"  F(47) >= F_5(31) = {FJ[31][4]}")
print("  (all weaker than the exact SAT witness F(47) >= 118)")

print("\n(D) AT ALPHA=3, DECIDED FROM THE CORPUS LADDER ALONE:")
print("   step        F(new)        budget F(old)+q'     verdict")
steps = [(19, 23), (23, 29), (29, 31), (31, 37), (37, 41), (41, 43),
         (43, 47), (47, 53)]
for old, new in steps:
    fo = F.get(old)
    fn = F.get(new)
    lo_hi = ""
    if fn is None:                      # F(47): bracket it
        bigger = min(z for z in GEARS if z > new and z in F)
        fn_up = F[bigger]
        lo_hi = f"<= F({bigger}) = {fn_up}"
        ok = fn_up <= fo + new
        print(f"  {old:2d}->{new:2d}    F({new}) {lo_hi:18s} {fo+new:6d}"
              f"        {'HOLDS' if ok else '??'}  margin >= {fo+new-fn_up:+d}")
        continue
    if fo is None:                      # F(47) on the budget side
        lo = 118
        ok = fn <= lo + new
        print(f"  {old:2d}->{new:2d}    F({new}) = {fn:5d}      "
              f">= {lo+new:5d}        {'HOLDS' if ok else '??'}"
              f"  margin >= {lo+new-fn:+d}")
        continue
    ok = fn <= fo + new
    assert ok, (old, new)
    print(f"  {old:2d}->{new:2d}    F({new}) = {fn:5d}      {fo+new:6d}"
          f"        HOLDS  margin {fo+new-fn:+d}")
print("\n  Every step through 47->53 is decided TRUE without any criterion.")
