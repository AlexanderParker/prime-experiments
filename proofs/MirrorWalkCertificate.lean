/-
MirrorWalkCertificate (round 65, 2026-09-17): the window statement for every machine below
1.28 × 10⁸, from four landings.

A landing serves every machine from its square root up to itself (`chain_covers_upto`), so a
handful of twin pairs settles an enormous range of machines.  This file carries the first four
links of the greedy chain - each the largest twin centre below the square of the one before -

    12,  108,  11352,  128845110

and with them proves, with no hypotheses and no sorries, that every machine from 11 to 128845108
has a twin prime pair inside its window `(q, q²]`.

The four numbers are landings of the one-flip family: `t = 2 M k` is reached from home by the
mirror of product `M`, so 108 = 2·6·9 is the mirror {2,3} at period 9, and 11352 = 2·2838·2 the
mirror of product 2838 at period 2.

research/stack/r8/chain_certificate.py carries the chain further: nine landings settle every
machine from 11 to 10²⁵⁹, the last found 49508 columns below the square of the one before.
-/
import MirrorWalkChain

namespace MirrorWalk

/-- The first four links of the greedy chain. -/
def cert : ℕ → ℕ
  | 0 => 12
  | 1 => 108
  | 2 => 11352
  | _ => 128845110

/-- **The window statement below 1.28 × 10⁸, proved.**  Every machine from 11 to 128845108 has a
twin prime pair inside its window. -/
theorem window_statement_below :
    ∀ q : ℕ, 11 ≤ q → q < 128845109 →
      ∃ m : ℕ, 1 ≤ m ∧ q < 6 * m - 1 ∧ 6 * m + 1 ≤ q ^ 2 ∧
        (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro q hlo hhi
  have hchain : ∀ n < 3, cert (n + 1) + 1 < (cert n - 1) ^ 2 := by
    intro n hn
    interval_cases n <;> simp [cert] <;> norm_num
  have hsix : ∀ n, n ≤ 3 → 6 ∣ cert n := by
    intro n hn
    interval_cases n <;> simp [cert] <;> norm_num
  have h6 : ∀ n, n ≤ 3 → 6 ≤ cert n := by
    intro n hn
    interval_cases n <;> simp [cert]
  have htwin : ∀ n, n ≤ 3 → TwinCenter (cert n) := by
    intro n hn
    interval_cases n <;>
      refine ⟨?_, ?_⟩ <;> simp [cert, TwinCenter] <;> norm_num
  exact window_statement_upto (N := 3) hchain hsix h6 htwin (by simp [cert]; omega)
    (by simp [cert]; omega)

end MirrorWalk
