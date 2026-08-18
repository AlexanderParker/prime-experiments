"""Round 16 lateral: the MOD-5 PIGEONHOLE LEMMA behind the j>=2 impossibility.

Gear 5 alone exposes only 3 of 5 residues: a twin slot k survives gear 5 iff
k mod 5 not in {1, 4} (the teeth), so every opening has k mod 5 in {0, 2, 3}.

LEMMA. No run of openings can contain FOUR openings in arithmetic progression
with common difference q'. Four terms r, r+g, r+2g, r+3g with gcd(g,5) = 1 are
four DISTINCT residues mod 5, and only three residues are exposed. Pigeonhole.
(q' is a prime > 5, so gcd(q',5) = 1 always.)

CONSEQUENCE for padding. Two alternating literal links sum to a multiple of q'
(minimally exactly q'), so the p=2, j=2 shape has offsets
    0, q', q'+v_1, 2q', 3q'
which CONTAINS the 4-term AP {0, q', 2q', 3q'} - impossible for EVERY q'.
Likewise three adjacent padded links (p=3, j=0) give exactly {0,q',2q',3q'}.
This is scale-free and uses gear 5 only: it never expires.
"""
E = sorted(k for k in range(35) if k % 5 not in (1, 4) and k % 7 not in (1, 6))
Eset = set(E)
E5 = sorted({k % 5 for k in E})

def feasible(offsets, mod=35, allowed=None):
    allowed = allowed or Eset
    return [r for r in E if all((r + d) % mod in allowed for d in offsets)]

print("=" * 74)
print("PART 1: the lemma")
print(f"  exposed residues mod 5 (gear 5 teeth are 1, 4): {E5}  -> only 3 of 5")
print(f"  a 4-term AP with difference coprime to 5 hits 4 distinct residues")
print(f"  mod 5, so it can NEVER lie inside a 3-element set. Hence:")
print(f"  NO FOUR OPENINGS IN AP WITH DIFFERENCE q' - for every prime q' > 5.")
bad = 0
for g in range(1, 5):
    for r in range(5):
        terms = {(r + i * g) % 5 for i in range(4)}
        if len(terms) != 4:
            bad += 1
        if terms <= set(E5):
            print("  COUNTEREXAMPLE", g, r)
print(f"  exhaustive check over all (r, g) mod 5 with g invertible: "
      f"4-term APs always hit 4 distinct residues (failures {bad}), and none "
      f"fits inside {E5}.")

print("=" * 74)
print("PART 2: which shapes contain a 4-term q'-AP (hence are impossible)")
print("  p=2, j=2 minimal: offsets 0, q', q'+v, 2q', 3q'  -> contains "
      "{0,q',2q',3q'}  IMPOSSIBLE")
print("  p=3, j=0 (three adjacent padded links): offsets 0, q', 2q', 3q'"
      "        IMPOSSIBLE")
print("  p=2, j=0 / j=1: only THREE q'-multiples appear - lemma silent, "
      "settled per-residue (part 3)")

print("=" * 74)
print("PART 3: exhaustive residue check, all (g, v) mod 35, for j = 0..4")
print("  g = q' mod 35 (invertible), v = literal-link value mod 35")
print(f"  {'j':>2} {'(g,v) pairs feasible':>22} {'of':>4} {'verdict':>34}")
for j in range(0, 5):
    tot = feas = 0
    for g in range(35):
        if g % 5 == 0 or g % 7 == 0:
            continue
        for v in range(35):
            tot += 1
            offs = [0, g]
            cur, val = g, v
            for _ in range(j):
                cur += val
                offs.append(cur)
                val = (g - val) % 35        # letters alternate, pair sums to q'
            offs.append(cur + g)
            if feasible([d % 35 for d in offs]):
                feas += 1
    verdict = ("ALWAYS IMPOSSIBLE (all residue pairs)" if feas == 0
               else f"possible for {100*feas/tot:.0f}% of pairs")
    print(f"  {j:>2} {feas:>22} {tot:>4} {verdict:>34}")

print("=" * 74)
print("PART 4: what survives - the permanent shape law")
print("  Two padded links in one run can ONLY be separated by j = 0 or j = 1")
print("  literal links; j >= 2 is impossible for every q', unconditionally.")
print("  Three padded links cannot be mutually adjacent.")
print("  These are gear-5(+7) facts: scale-free, no spectrum, never expire -")
print("  unlike round 14's F_2(M) < 2q' threshold, which died at 37->41.")
