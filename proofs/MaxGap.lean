/-
The mod-3 endpoint law for the maximal coverable run.

Harvester section 12 (rust2/src/bin/maxgap_pruned.rs) derives, and uses to
prune the F(2,53) covering search:

> at the maximal coverable run `M`, both bounding positions are uncovered
> (else the run extends) and gear 3 is used at the maximum; gear 3 leaves
> exactly ONE residue class mod 3 uncovered, so both bounding positions lie
> in it, giving `M ≡ 2 (mod 3)` and `F = M + 1 ≡ 0 (mod 3)` unconditionally.

The arithmetic core is `uncovered_span_mod_three`: gear 3 blocks two distinct
classes mod 3, so the survivors form a single class and any two of them are
congruent. Everything else in the derivation - maximality forcing the two
bounding positions to be uncovered, and gear 3 being active - is search
bookkeeping, taken here as hypotheses.

All thirteen known exact values (33 .. 309) are ≡ 0 mod 3, as recorded.
-/

import Mathlib.Tactic.Linarith

namespace MaxGap

/-- **Gear 3 leaves one class.** If two distinct residues mod 3 are blocked,
any two unblocked positions are congruent mod 3, so their span is divisible
by 3. -/
theorem uncovered_span_mod_three {c1 c2 a b : ℕ} (hne : c1 % 3 ≠ c2 % 3)
    (ha1 : a % 3 ≠ c1 % 3) (ha2 : a % 3 ≠ c2 % 3)
    (hb1 : b % 3 ≠ c1 % 3) (hb2 : b % 3 ≠ c2 % 3) :
    (b - a) % 3 = 0 := by
  omega

/-- **`F ≡ 0 (mod 3)`.** The maximal coverable run `M` is bounded below by
an uncovered position `a` and above by an uncovered position `b = a + M + 1`;
both avoid gear 3's two blocked classes, so `3 ∣ M + 1 = F`. -/
theorem F_zero_mod_three {c1 c2 a M : ℕ} (hne : c1 % 3 ≠ c2 % 3)
    (ha1 : a % 3 ≠ c1 % 3) (ha2 : a % 3 ≠ c2 % 3)
    (hb1 : (a + M + 1) % 3 ≠ c1 % 3) (hb2 : (a + M + 1) % 3 ≠ c2 % 3) :
    (M + 1) % 3 = 0 := by
  omega

/-- Equivalently `M ≡ 2 (mod 3)`: the run length itself is two short of a
multiple of 3. -/
theorem M_two_mod_three {c1 c2 a M : ℕ} (hne : c1 % 3 ≠ c2 % 3)
    (ha1 : a % 3 ≠ c1 % 3) (ha2 : a % 3 ≠ c2 % 3)
    (hb1 : (a + M + 1) % 3 ≠ c1 % 3) (hb2 : (a + M + 1) % 3 ≠ c2 % 3) :
    M % 3 = 2 := by
  omega

/-- The pruning rule the search uses: a run length that is not `≡ 2 (mod 3)`
can never be the maximum, so only every third length needs a certificate. -/
theorem not_max_of_mod_three {c1 c2 a M : ℕ} (hne : c1 % 3 ≠ c2 % 3)
    (ha1 : a % 3 ≠ c1 % 3) (ha2 : a % 3 ≠ c2 % 3)
    (hb1 : (a + M + 1) % 3 ≠ c1 % 3) (hb2 : (a + M + 1) % 3 ≠ c2 % 3)
    (hM : M % 3 ≠ 2) : False := by
  omega

end MaxGap
