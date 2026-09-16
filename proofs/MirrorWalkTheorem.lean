/-
MirrorWalkTheorem (round 59, 2026-09-17): the construction as one theorem with one hypothesis.

The settle walk is a fixed finite procedure:
  * start at home `(-1, 1)`, open to every gear;
  * visit the gears `q` down to `7`, at gear `g` flipping about the mirror `{2, 3, g}`, whose
    move is `12 g k` for a period `k ≤ K` in either direction, choosing a period that keeps
    every gear visited so far off its two teeth (such a period exists at every step of the first
    cut: `keeping_move_free`);
  * at the first step past the cut, take a candidate open to every gear of the machine.
Every part is proved except the last, and this file states the whole as one implication:

    the handover column is open to the gears of the prefix   (the walk's invariant)
  + some candidate of the last step is open to every gear    (`StepOpen`, the one hypothesis)
  => a twin prime pair inside the window `(q, q²]`.

The hypothesis is measured to hold at the tail's FIRST step at every machine from 200 to 1200
(research/stack/r8/tail_steps_needed.py), with a margin flat in `q` of about five open
candidates once `K = (ln q)²`.
-/
import MirrorWalkConditional

namespace MirrorWalk

open SquareColumn

/-- The walk's data at the handover: the column, the stride of the last step, the period bound,
and the machine's top gear. -/
structure Handover where
  c : ℤ        -- the column the prefix hands over
  s : ℤ        -- the stride of the last step (`12 g` for the mirror `{2, 3, g}`)
  K : ℕ        -- the period bound
  P : ℕ        -- the machine's top gear (the window is `(P, P²]`)

/-- **The construction's theorem.**  With `G` holding every prime from 5 below the machine's top
gear, the step hypothesis at the handover gives a twin prime pair inside the window. -/
theorem construction_twin {G : Finset ℕ} (H : Handover)
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < H.P → r ∈ G)
    (hstep : StepOpen G H.c H.s H.K H.P) :
    ∃ m : ℕ, 1 ≤ m ∧ 6 * m + 1 < H.P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime :=
  walk_twin_of_stepOpen hfull hstep

/-- **The window statement, from the construction.**  If every machine's walk meets its step
hypothesis, every machine's window holds a twin prime pair. -/
theorem window_statement_of_stepOpen
    (H : ∀ (P : ℕ), P.Prime → 5 ≤ P → ∃ (G : Finset ℕ) (h : Handover),
        h.P = P ∧ (∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) ∧ StepOpen G h.c h.s h.K h.P) :
    ∀ (P : ℕ), P.Prime → 5 ≤ P →
      ∃ m : ℕ, 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro P hP h5
  obtain ⟨G, h, hPq, hfull, hstep⟩ := H P hP h5
  subst hPq
  exact construction_twin h hfull hstep

end MirrorWalk
