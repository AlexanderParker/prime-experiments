/-
WORD LEGALITY: THE WORD REDUCTION (R89) AND THE SAME-TOOTH LEMMA (R90) AS
KERNEL THEOREMS (Formalist, round 30).

Constructor's definitions, quoted verbatim from docs/proof-search/constructor.md:

  R81. "Define a WORD-LEGAL J-WINDOW of M as J consecutive gaps
       (g_L, w_1..w_{J-2}, g_R) whose J-2 middles are each 0 or +-2c mod q' (T2)
       with the nonzero classes strictly alternating, padded middles transparent
       (T3); Q*_J = max span, Delta_J = Q*_J - F_2(M), Phi_J = max flank sum."

  R89. "Let Lambda(M) be the legal letters and a LEGAL WORD of M a run of
       consecutive gaps all in Lambda(M) whose nonzero T3 classes strictly
       alternate; L(M) = length of the longest REALISED legal word.  Then for
       every J >= 3
           Q*_J(M; q') > -inf  <=>  L(M) >= J-2,   so  J_max(M) = L(M) + 2
                                                   and  A_kill(M -> q') = L(M) + 1.
       PROOF. (=>) the J-2 middles of a word-legal J-window ARE J-2 consecutive
       gaps of M, each legal, alternating - a realised legal word. (<=) any
       occurrence of such a word, plus the gaps immediately before and after, IS
       a word-legal J-window (legality constrains only middles).  For A_kill: a
       k-chain's k-1 interiors are a legal word by T2/T3, and conversely a
       realised legal word of length k-1 is killable in full at some translate
       by R68's CRT step."

  R90. "A middle of class 0 (padded) leaves the tooth fixed; a middle of class
       +-1 flips it.  So the middle span x_{J-1} - x_1 = (t_{J-1} - t_1)c is
           = 0 mod q'   <=>   the number of NON-PADDED middles is even.
       For a LITERAL even-J window all J-2 middles are non-padded: the first and
       last killed opening sit on the SAME TOOTH, the middle span is 0 mod q'."

  Mechanic, round 25 (the same predicate from the census side): "WORD LEGALITY
  (each middle gap in V mod q', and the induced letter word's prefix sums of
  range <= 1)", V = {0, a, b}, a = 2u', b = q' - 2u', u' = 6^{-1} mod q'.

HOW THE PHRASES ARE RENDERED HERE (everything is over `ZMod q`, `q` the new
gear, `c` its tooth `u' = 6^{-1}`; NO primality and NO `6c = 1` is assumed
unless a theorem states it):

  * "legal letter" (T2)          `Cls` = {pad, up, down}, values `0, 2c, -2c`
                                 (`Cls.val`).  `up` is the step from tooth `-c`
                                 to tooth `+c`, `down` the reverse, `pad` stays.
  * "nonzero classes strictly    `Alt t w`: read the word with the current
     alternate" (T3)             tooth `t`; `pad` keeps it, `up` needs `-` and
                                 moves to `+`, `down` the reverse.  `Legal w` =
                                 some starting tooth works.  `legal_iff_noRepeat`
                                 proves this IS "no two consecutive nonzero
                                 classes are equal" (`NoRepeat`), Constructor's
                                 wording, and `alt_iff_prefixSum` proves it is
                                 Mechanic's "prefix sums of range <= 1".
  * "killed at one phase"        `KilledAt c r x` = `x - r = c or -c`, i.e.
                                 `AnchorChain.OnTeeth c (x - r)`.
  * "realised legal word of M"   `RealisedWord c op m`: some index `i` and some
                                 legal word `w` of length `m` with the residues
                                 of the gaps `op (i+k+1) - op (i+k)` equal to
                                 the letters (`WordAt`).  `op` is the machine's
                                 opening enumeration (`Machine37.opSeq37`, ...).
  * "word-legal J-window"        `WindowLegal c op i J`: the J-2 middles at
                                 `i+1 .. i+J-2` are a legal word; `QstarNonempty
                                 c op J` = some window exists = "Q*_J > -inf".
  * "k-chain"                    `Chain c op k n`: the k consecutive openings
                                 `op n .. op (n+k-1)` are all on teeth at ONE
                                 phase `r` (Mechanic's parts (A)+(B); part (C),
                                 the joint realisability at the CRT slot, is
                                 R68's step and is NOT needed for anything here).

THEOREMS (all hypothesis-free unless a hypothesis is named):
  killable_iff       ∃ r killing every point  <->  the differences are a legal word
  chain_iff_word     Chain (k+1) n  <->  a legal word of length k at n     (R89, A_kill half)
  qstar_iff_word     Q*_J > -inf  <->  RealisedWord (J-2)        [hper: the gap residues
                                                                  are periodic, N > 0]
  jmax               with L realised and L+1 not: Q*_J > -inf <-> J <= L + 2 [hper]
  akill              with L realised and L+1 not: a k-chain exists <-> k <= L + 1
  same_tooth         Alt t w -> (sum of letters = 0 <-> even # non-padded)  [2c ≠ 0]
  same_tooth_window  the middle span of a window is 0 mod q  <-> even # non-padded middles
  two_mul_ne_zero    6c = 1 and 1 < q give 2c ≠ 0 (so the hypothesis is discharged at
                     every real gear); val_injective: the three letters are distinct
                     when 6c = 1 and 2 < q.

The periodicity hypothesis `hper` is used ONLY in the (<=) direction of R89's
first equivalence, to move a realised word that starts at index 0 to one that
has a gap before it - exactly the "plus the gaps immediately before" of
Constructor's proof.  Every machine in the ledger has it (`Machine11.g11_shift`,
`Periodic.op_shift`); it is a hypothesis here because this file is abstract in
the machine.
-/

import AnchorChain

namespace WordLegal

/-! ## 1. Letters, words, alternation -/

/-- The three legal letter classes (T2): padded (`0`), up (`+2c`), down (`-2c`). -/
inductive Cls
  | pad
  | up
  | down
  deriving DecidableEq, Repr

variable {q : ℕ}

/-- The residue a letter stands for: `0`, `2c`, `-2c`. -/
def Cls.val (c : ZMod q) : Cls → ZMod q
  | .pad => 0
  | .up => 2 * c
  | .down => -(2 * c)

/-- The tooth of a killed opening: `true` is `+c`, `false` is `-c`. -/
def tooth (c : ZMod q) (t : Bool) : ZMod q := if t then c else -c

/-- **T3 alternation**, read with the current tooth: a padded letter keeps the
tooth, `up` needs tooth `-` and moves to `+`, `down` the reverse. -/
def Alt : Bool → List Cls → Prop
  | _, [] => True
  | t, .pad :: w => Alt t w
  | t, .up :: w => t = false ∧ Alt true w
  | t, .down :: w => t = true ∧ Alt false w

/-- The tooth after reading the word (meaningful under `Alt`). -/
def endTooth : Bool → List Cls → Bool
  | t, [] => t
  | t, .pad :: w => endTooth t w
  | _, .up :: w => endTooth true w
  | _, .down :: w => endTooth false w

/-- A LEGAL WORD: some starting tooth makes the alternation consistent. -/
def Legal (w : List Cls) : Prop := ∃ t, Alt t w

/-- The number of non-padded letters. -/
def nonpad : List Cls → ℕ
  | [] => 0
  | .pad :: w => nonpad w
  | _ :: w => nonpad w + 1

/-- Constructor's wording: "the nonzero classes strictly alternate" = no two
consecutive nonzero classes are equal; `last` is the last nonzero class seen. -/
def NoRepeat : Option Cls → List Cls → Prop
  | _, [] => True
  | last, .pad :: w => NoRepeat last w
  | last, .up :: w => last ≠ some .up ∧ NoRepeat (some .up) w
  | last, .down :: w => last ≠ some .down ∧ NoRepeat (some .down) w

theorem alt_iff_noRepeat : ∀ (t : Bool) (w : List Cls),
    Alt t w ↔ NoRepeat (some (if t then Cls.up else Cls.down)) w
  | _, [] => Iff.rfl
  | t, .pad :: w => alt_iff_noRepeat t w
  | t, .up :: w => by
      cases t
      · show (false = false ∧ Alt true w) ↔ (some Cls.down ≠ some Cls.up ∧ NoRepeat (some Cls.up) w)
        rw [alt_iff_noRepeat true w]; simp
      · show (true = false ∧ Alt true w) ↔ (some Cls.up ≠ some Cls.up ∧ NoRepeat (some Cls.up) w)
        simp
  | t, .down :: w => by
      cases t
      · show (false = true ∧ Alt false w) ↔ (some Cls.down ≠ some Cls.down ∧ NoRepeat (some Cls.down) w)
        simp
      · show (true = true ∧ Alt false w) ↔ (some Cls.up ≠ some Cls.down ∧ NoRepeat (some Cls.down) w)
        rw [alt_iff_noRepeat false w]; simp

/-- **`Legal` IS Constructor's "nonzero classes strictly alternate".** -/
theorem legal_iff_noRepeat : ∀ w : List Cls, Legal w ↔ NoRepeat none w
  | [] => ⟨fun _ => trivial, fun _ => ⟨true, trivial⟩⟩
  | .pad :: w => by
      show (∃ t, Alt t w) ↔ NoRepeat none w
      exact legal_iff_noRepeat w
  | .up :: w => by
      show (∃ t, t = false ∧ Alt true w) ↔ (none ≠ some Cls.up ∧ NoRepeat (some Cls.up) w)
      constructor
      · rintro ⟨t, rfl, h⟩
        exact ⟨by simp, by simpa using (alt_iff_noRepeat true w).mp h⟩
      · rintro ⟨-, h⟩
        exact ⟨false, rfl, (alt_iff_noRepeat true w).mpr (by simpa using h)⟩
  | .down :: w => by
      show (∃ t, t = true ∧ Alt false w) ↔ (none ≠ some Cls.down ∧ NoRepeat (some Cls.down) w)
      constructor
      · rintro ⟨t, rfl, h⟩
        exact ⟨by simp, by simpa using (alt_iff_noRepeat false w).mp h⟩
      · rintro ⟨-, h⟩
        exact ⟨true, rfl, (alt_iff_noRepeat false w).mpr (by simpa using h)⟩

/-- The class of a letter as an integer, `up = +1`, `down = -1`, `pad = 0`. -/
def Cls.sgn : Cls → ℤ
  | .pad => 0
  | .up => 1
  | .down => -1

/-- Prefix sums of the class word, from a starting value `s`. -/
def prefixOK : ℤ → List Cls → Prop
  | _, [] => True
  | s, a :: w => (s + a.sgn = 0 ∨ s + a.sgn = 1) ∧ prefixOK (s + a.sgn) w

/-- **Mechanic's "prefix sums of range <= 1"**: the running class sum stays in
`{0, 1}` (tooth `-` is `0`, tooth `+` is `1`) exactly when the word alternates. -/
theorem alt_iff_prefixSum : ∀ (t : Bool) (w : List Cls),
    Alt t w ↔ prefixOK (if t then 1 else 0) w
  | _, [] => Iff.rfl
  | t, .pad :: w => by
      show Alt t w ↔ ((if t then (1:ℤ) else 0) + 0 = 0 ∨ (if t then (1:ℤ) else 0) + 0 = 1) ∧
        prefixOK ((if t then (1:ℤ) else 0) + 0) w
      rw [alt_iff_prefixSum t w]
      cases t <;> simp
  | t, .up :: w => by
      show (t = false ∧ Alt true w) ↔ ((if t then (1:ℤ) else 0) + 1 = 0 ∨ (if t then (1:ℤ) else 0) + 1 = 1) ∧
        prefixOK ((if t then (1:ℤ) else 0) + 1) w
      rw [alt_iff_prefixSum true w]
      cases t <;> simp
  | t, .down :: w => by
      show (t = true ∧ Alt false w) ↔ ((if t then (1:ℤ) else 0) + -1 = 0 ∨ (if t then (1:ℤ) else 0) + -1 = 1) ∧
        prefixOK ((if t then (1:ℤ) else 0) + -1) w
      rw [alt_iff_prefixSum false w]
      cases t <;> simp

/-- Prefixes of legal words are legal (the alternation is a fold). -/
theorem alt_take : ∀ (m : ℕ) (t : Bool) (w : List Cls), Alt t w → Alt t (w.take m)
  | 0, _, _, _ => trivial
  | _ + 1, _, [], h => h
  | m + 1, t, .pad :: w, h => by
      rw [List.take_succ_cons]; exact alt_take m t w h
  | m + 1, t, .up :: w, h => by
      rw [List.take_succ_cons]
      obtain ⟨ht, h'⟩ : t = false ∧ Alt true w := h
      exact ⟨ht, alt_take m true w h'⟩
  | m + 1, t, .down :: w, h => by
      rw [List.take_succ_cons]
      obtain ⟨ht, h'⟩ : t = true ∧ Alt false w := h
      exact ⟨ht, alt_take m false w h'⟩

theorem legal_take (m : ℕ) {w : List Cls} (h : Legal w) : Legal (w.take m) :=
  let ⟨t, ht⟩ := h; ⟨t, alt_take m t w ht⟩

/-! ## 2. The same-tooth lemma (R90) -/

theorem sum_eq_tooth_sub (c : ZMod q) : ∀ (t : Bool) (w : List Cls), Alt t w →
    (w.map (Cls.val c)).sum = tooth c (endTooth t w) - tooth c t
  | t, [], _ => by simp [endTooth]
  | t, .pad :: w, h => by
      have ih := sum_eq_tooth_sub c t w h
      simp only [List.map_cons, List.sum_cons, Cls.val, endTooth, ih, zero_add]
  | t, .up :: w, h => by
      obtain ⟨rfl, h'⟩ : t = false ∧ Alt true w := h
      have ih := sum_eq_tooth_sub c true w h'
      simp only [List.map_cons, List.sum_cons, Cls.val, endTooth, ih]
      generalize endTooth true w = e
      cases e <;> simp [tooth] <;> try ring
  | t, .down :: w, h => by
      obtain ⟨rfl, h'⟩ : t = true ∧ Alt false w := h
      have ih := sum_eq_tooth_sub c false w h'
      simp only [List.map_cons, List.sum_cons, Cls.val, endTooth, ih]
      generalize endTooth false w = e
      cases e <;> simp [tooth] <;> try ring

/-- Under alternation the end tooth is the start tooth iff the number of
non-padded letters is even: each non-padded letter flips the tooth. -/
theorem endTooth_eq_iff : ∀ (t : Bool) (w : List Cls), Alt t w →
    (endTooth t w = t ↔ Even (nonpad w))
  | _, [], _ => by simp [endTooth, nonpad]
  | t, .pad :: w, h => by
      show (endTooth t w = t ↔ Even (nonpad w))
      exact endTooth_eq_iff t w h
  | t, .up :: w, h => by
      obtain ⟨rfl, h'⟩ : t = false ∧ Alt true w := h
      show (endTooth true w = false ↔ Even (nonpad w + 1))
      rw [Nat.even_add_one, ← endTooth_eq_iff true w h']
      cases endTooth true w <;> simp
  | t, .down :: w, h => by
      obtain ⟨rfl, h'⟩ : t = true ∧ Alt false w := h
      show (endTooth false w = true ↔ Even (nonpad w + 1))
      rw [Nat.even_add_one, ← endTooth_eq_iff false w h']
      cases endTooth false w <;> simp

/-- **THE SAME-TOOTH LEMMA (R90).**  For a legal word the sum of its letters is
`0` iff the number of non-padded letters is even.  Needs only `2c ≠ 0`. -/
theorem same_tooth (c : ZMod q) (h2 : (2 : ZMod q) * c ≠ 0) {t : Bool} {w : List Cls}
    (h : Alt t w) : (w.map (Cls.val c)).sum = 0 ↔ Even (nonpad w) := by
  rw [sum_eq_tooth_sub c t w h, ← endTooth_eq_iff t w h]
  generalize endTooth t w = e
  cases e <;> cases t
  · simp [tooth]
  · simp only [tooth]
    constructor
    · intro h; exfalso; apply h2; simp at h; linear_combination -h
    · intro h; cases h
  · simp only [tooth]
    constructor
    · intro h; exfalso; apply h2; simp at h; linear_combination h
    · intro h; cases h
  · simp [tooth]

/-- `2c ≠ 0` at every real gear: from `6c = 1` alone (and `q > 1`). -/
theorem two_mul_ne_zero {c : ZMod q} (h6 : (6 : ZMod q) * c = 1) (hq : 1 < q) :
    (2 : ZMod q) * c ≠ 0 := by
  intro h
  have h1 : ((1 : ℕ) : ZMod q) = 0 := by
    push_cast
    linear_combination -h6 + 3 * h
  have := Nat.le_of_dvd one_pos ((ZMod.natCast_eq_zero_iff 1 q).mp h1)
  omega

theorem four_mul_ne_zero {c : ZMod q} (h6 : (6 : ZMod q) * c = 1) (hq : 2 < q) :
    (4 : ZMod q) * c ≠ 0 := by
  intro h
  have h2 : ((2 : ℕ) : ZMod q) = 0 := by
    push_cast
    linear_combination -2 * h6 + 3 * h
  have := Nat.le_of_dvd (by norm_num) ((ZMod.natCast_eq_zero_iff 2 q).mp h2)
  omega

/-- The three legal letters are DISTINCT residues at every real gear, so the
class of a legal gap is well defined. -/
theorem val_injective {c : ZMod q} (h6 : (6 : ZMod q) * c = 1) (hq : 2 < q) :
    Function.Injective (Cls.val c) := by
  intro a b hab
  have h2 := two_mul_ne_zero h6 (by omega)
  have h4 := four_mul_ne_zero h6 hq
  cases a <;> cases b
  all_goals (try rfl)
  all_goals simp only [Cls.val] at hab
  · exact absurd (by linear_combination -hab) h2
  · exact absurd (by linear_combination hab) h2
  · exact absurd (by linear_combination hab) h2
  · exact absurd (by linear_combination hab) h4
  · exact absurd (by linear_combination -hab) h2
  · exact absurd (by linear_combination -hab) h4

/-! ## 3. Killability = legality (the residue half of R89) -/

/-- `x` is killed at phase `r`: its class relative to `r` is a tooth. -/
def KilledAt (c r x : ZMod q) : Prop := x - r = c ∨ x - r = -c

theorem killedAt_iff_onTeeth (c r x : ZMod q) :
    KilledAt c r x ↔ AnchorChain.OnTeeth c (x - r) := Iff.rfl

/-- Consecutive differences of a list of residues. -/
def diffs : List (ZMod q) → List (ZMod q)
  | a :: b :: l => (b - a) :: diffs (b :: l)
  | _ => []

/-- (=>) with the head's tooth carried along. -/
theorem word_of_killed (c r : ZMod q) : ∀ (l : List (ZMod q)) (a : ZMod q) (t : Bool),
    a - r = tooth c t → (∀ y ∈ l, KilledAt c r y) →
    ∃ w : List Cls, Alt t w ∧ diffs (a :: l) = w.map (Cls.val c)
  | [], _, _, _, _ => ⟨[], trivial, rfl⟩
  | b :: l, a, t, ha, hk => by
      have hb : KilledAt c r b := hk b (by simp)
      have hrest : ∀ y ∈ l, KilledAt c r y := fun y hy => hk y (by simp [hy])
      rcases hb with hb | hb
      · obtain ⟨w, hw, hd⟩ := word_of_killed c r l b true (by simpa [tooth] using hb) hrest
        cases t with
        | false =>
            refine ⟨.up :: w, ⟨rfl, hw⟩, ?_⟩
            rw [show diffs (a :: b :: l) = (b - a) :: diffs (b :: l) from rfl, hd, List.map_cons]
            congr 1
            simp only [tooth] at ha; simp at ha
            show b - a = 2 * c
            linear_combination hb - ha
        | true =>
            refine ⟨.pad :: w, hw, ?_⟩
            rw [show diffs (a :: b :: l) = (b - a) :: diffs (b :: l) from rfl, hd, List.map_cons]
            congr 1
            simp only [tooth] at ha; simp at ha
            show b - a = 0
            linear_combination hb - ha
      · obtain ⟨w, hw, hd⟩ := word_of_killed c r l b false (by simpa [tooth] using hb) hrest
        cases t with
        | true =>
            refine ⟨.down :: w, ⟨rfl, hw⟩, ?_⟩
            rw [show diffs (a :: b :: l) = (b - a) :: diffs (b :: l) from rfl, hd, List.map_cons]
            congr 1
            simp only [tooth] at ha; simp at ha
            show b - a = -(2 * c)
            linear_combination hb - ha
        | false =>
            refine ⟨.pad :: w, hw, ?_⟩
            rw [show diffs (a :: b :: l) = (b - a) :: diffs (b :: l) from rfl, hd, List.map_cons]
            congr 1
            simp only [tooth] at ha; simp at ha
            show b - a = 0
            linear_combination hb - ha

/-- (<=): a legal word starting at `a` with tooth `t` is killed at phase
`a - tooth t`. -/
theorem killed_of_word (c : ZMod q) : ∀ (w : List Cls) (t : Bool) (l : List (ZMod q)) (a : ZMod q),
    Alt t w → diffs (a :: l) = w.map (Cls.val c) →
    ∀ y ∈ a :: l, KilledAt c (a - tooth c t) y
  | _, t, [], a, _, _ => by
      intro y hy
      simp at hy
      subst hy
      unfold KilledAt tooth
      cases t <;> simp
  | w, t, b :: l, a, hw, hd => by
      cases w with
      | nil => simp [diffs] at hd
      | cons w0 w' =>
        rw [show diffs (a :: b :: l) = (b - a) :: diffs (b :: l) from rfl, List.map_cons,
          List.cons.injEq] at hd
        obtain ⟨h0, hd'⟩ := hd
        have hhead : KilledAt c (a - tooth c t) a := by
          unfold KilledAt tooth
          cases t <;> simp
        cases w0 with
        | pad =>
            have hw' : Alt t w' := hw
            have ih := killed_of_word c w' t l b hw' hd'
            have hr : b - tooth c t = a - tooth c t := by
              simp only [Cls.val] at h0
              linear_combination h0
            intro y hy
            rcases List.mem_cons.mp hy with rfl | hy
            · exact hhead
            · rw [← hr]; exact ih y hy
        | up =>
            obtain ⟨rfl, hw'⟩ : t = false ∧ Alt true w' := hw
            have ih := killed_of_word c w' true l b hw' hd'
            have hr : b - tooth c true = a - tooth c false := by
              simp only [Cls.val] at h0
              simp only [tooth]; simp
              linear_combination h0
            intro y hy
            rcases List.mem_cons.mp hy with rfl | hy
            · exact hhead
            · rw [← hr]; exact ih y hy
        | down =>
            obtain ⟨rfl, hw'⟩ : t = true ∧ Alt false w' := hw
            have ih := killed_of_word c w' false l b hw' hd'
            have hr : b - tooth c false = a - tooth c true := by
              simp only [Cls.val] at h0
              simp only [tooth]; simp
              linear_combination h0
            intro y hy
            rcases List.mem_cons.mp hy with rfl | hy
            · exact hhead
            · rw [← hr]; exact ih y hy

/-- **KILLABILITY = LEGALITY.**  A list of residues can be put on the teeth of
one phase exactly when its consecutive differences form a legal word.  This is
Constructor's "legality = existence of a tooth assignment" (R68), machine-free
and hypothesis-free. -/
theorem killable_iff (c : ZMod q) (x : List (ZMod q)) :
    (∃ r, ∀ y ∈ x, KilledAt c r y) ↔ ∃ w : List Cls, Legal w ∧ diffs x = w.map (Cls.val c) := by
  constructor
  · rintro ⟨r, hr⟩
    cases x with
    | nil => exact ⟨[], ⟨true, trivial⟩, rfl⟩
    | cons a l =>
      have ha := hr a (by simp)
      have hl : ∀ y ∈ l, KilledAt c r y := fun y hy => hr y (by simp [hy])
      rcases ha with ha | ha
      · obtain ⟨w, hw, hd⟩ := word_of_killed c r l a true (by simpa [tooth] using ha) hl
        exact ⟨w, ⟨true, hw⟩, hd⟩
      · obtain ⟨w, hw, hd⟩ := word_of_killed c r l a false (by simpa [tooth] using ha) hl
        exact ⟨w, ⟨false, hw⟩, hd⟩
  · rintro ⟨w, ⟨t, hw⟩, hd⟩
    cases x with
    | nil => exact ⟨0, by simp⟩
    | cons a l => exact ⟨a - tooth c t, killed_of_word c w t l a hw hd⟩

/-! ## 4. The word reduction (R89) over a machine's opening enumeration -/

section Machine

variable (c : ZMod q) (op : ℕ → ℕ)

/-- The residue of the `n`-th gap, `op (n+1) - op n`. -/
def gapRes (n : ℕ) : ZMod q := (op (n + 1) : ZMod q) - (op n : ZMod q)

theorem gapRes_eq_cast {n : ℕ} (h : op n ≤ op (n + 1)) :
    gapRes (q := q) op n = ((op (n + 1) - op n : ℕ) : ZMod q) := by
  unfold gapRes; rw [Nat.cast_sub h]

/-- The gaps from index `i` on spell the word `w`. -/
def WordAt : ℕ → List Cls → Prop
  | _, [] => True
  | i, a :: w => gapRes (q := q) op i = Cls.val c a ∧ WordAt (i + 1) w

/-- "a realised legal word of length `m`" (R89's `L(M) >= m`). -/
def RealisedWord (m : ℕ) : Prop :=
  ∃ (i : ℕ) (w : List Cls), w.length = m ∧ Legal w ∧ WordAt c op i w

/-- "a word-legal `J`-window starting at gap `i`": its `J-2` middles, the gaps
`i+1 .. i+J-2`, are a legal word (R81). -/
def WindowLegal (i J : ℕ) : Prop :=
  ∃ w : List Cls, w.length = J - 2 ∧ Legal w ∧ WordAt c op (i + 1) w

/-- "`Q*_J > -inf`": some word-legal `J`-window exists. -/
def QstarNonempty (J : ℕ) : Prop := ∃ i, WindowLegal c op i J

/-- "a `k`-chain at `n`": the `k` consecutive openings `op n .. op (n+k-1)` are
all on the teeth of ONE phase `r` of the new gear. -/
def Chain (k n : ℕ) : Prop :=
  ∃ r : ZMod q, ∀ j, j < k → KilledAt c r (op (n + j) : ZMod q)

/-- The residues of `k` consecutive openings from `n`. -/
def resList (n : ℕ) : ℕ → List (ZMod q)
  | 0 => []
  | k + 1 => (op n : ZMod q) :: resList (n + 1) k

/-- The residues of `k` consecutive gaps from `n`. -/
def gapResList (n : ℕ) : ℕ → List (ZMod q)
  | 0 => []
  | m + 1 => gapRes (q := q) op n :: gapResList (n + 1) m

theorem mem_resList (n : ℕ) : ∀ (k : ℕ) (y : ZMod q),
    y ∈ resList (q := q) op n k ↔ ∃ j, j < k ∧ y = (op (n + j) : ZMod q)
  | 0, y => by simp [resList]
  | k + 1, y => by
      simp only [resList, List.mem_cons, mem_resList (n + 1) k]
      constructor
      · rintro (rfl | ⟨j, hj, rfl⟩)
        · exact ⟨0, by omega, by simp⟩
        · exact ⟨j + 1, by omega, by rw [show n + (j + 1) = n + 1 + j by omega]⟩
      · rintro ⟨j, hj, rfl⟩
        cases j with
        | zero => left; simp
        | succ j => right; exact ⟨j, by omega, by rw [show n + (j + 1) = n + 1 + j by omega]⟩

theorem diffs_resList (n : ℕ) : ∀ k, diffs (resList (q := q) op n (k + 1)) = gapResList op n k
  | 0 => rfl
  | k + 1 => by
      show gapRes op n :: diffs (resList op (n + 1) (k + 1)) = gapRes op n :: gapResList op (n + 1) k
      rw [diffs_resList (n + 1) k]

theorem gapResList_length (n : ℕ) : ∀ k, (gapResList (q := q) op n k).length = k
  | 0 => rfl
  | k + 1 => by simp [gapResList, gapResList_length (n + 1) k]

theorem wordAt_iff : ∀ (w : List Cls) (n : ℕ),
    WordAt c op n w ↔ gapResList (q := q) op n w.length = w.map (Cls.val c)
  | [], n => by simp [WordAt, gapResList]
  | a :: w, n => by
      simp only [WordAt, List.length_cons, gapResList, List.map_cons, List.cons.injEq]
      rw [wordAt_iff w (n + 1)]

/-- **R89, the `A_kill` half**: a `(k+1)`-chain at `n` IS a legal word of length
`k` at `n`.  Hypothesis-free. -/
theorem chain_iff_word (k n : ℕ) :
    Chain c op (k + 1) n ↔ ∃ w : List Cls, w.length = k ∧ Legal w ∧ WordAt c op n w := by
  have h1 : Chain c op (k + 1) n ↔ ∃ r, ∀ y ∈ resList (q := q) op n (k + 1), KilledAt c r y := by
    unfold Chain
    constructor
    · rintro ⟨r, hr⟩
      refine ⟨r, fun y hy => ?_⟩
      obtain ⟨j, hj, rfl⟩ := (mem_resList op n (k + 1) y).mp hy
      exact hr j hj
    · rintro ⟨r, hr⟩
      exact ⟨r, fun j hj => hr _ ((mem_resList op n (k + 1) _).mpr ⟨j, hj, rfl⟩)⟩
  rw [h1, killable_iff, diffs_resList]
  constructor
  · rintro ⟨w, hw, hd⟩
    have hlen : w.length = k := by
      have := congrArg List.length hd
      simpa [gapResList_length] using this.symm
    exact ⟨w, hlen, hw, (wordAt_iff c op w n).mpr (by rw [hlen]; exact hd)⟩
  · rintro ⟨w, hlen, hw, hwa⟩
    refine ⟨w, hw, ?_⟩
    have := (wordAt_iff c op w n).mp hwa
    rw [hlen] at this
    exact this

/-- (=>) of R89: a word-legal window's middles are a realised legal word. -/
theorem word_of_window {J : ℕ} (h : QstarNonempty c op J) : RealisedWord c op (J - 2) := by
  obtain ⟨i, w, hl, hw, hwa⟩ := h
  exact ⟨i + 1, w, hl, hw, hwa⟩

/-- (<=) of R89 when the word has a gap before it. -/
theorem window_of_word {i : ℕ} (hi : 1 ≤ i) {w : List Cls} (hl : Legal w)
    (hwa : WordAt c op i w) : WindowLegal c op (i - 1) (w.length + 2) := by
  refine ⟨w, by omega, hl, ?_⟩
  rw [show i - 1 + 1 = i by omega]
  exact hwa

theorem wordAt_shift {N : ℕ} (hper : ∀ n, gapRes (q := q) op (n + N) = gapRes op n) :
    ∀ (w : List Cls) (i : ℕ), WordAt c op i w → WordAt c op (i + N) w
  | [], _, _ => trivial
  | a :: w, i, ⟨h1, h2⟩ =>
      ⟨by rw [hper]; exact h1,
       by rw [show i + N + 1 = (i + 1) + N by omega]; exact wordAt_shift hper w (i + 1) h2⟩

/-- **R89, first equivalence**: `Q*_J > -inf  <->  a realised legal word of
length `J-2` exists`.  The periodicity of the gap residues (`hper`, period
`N > 0`) is used only to give a word that starts at index `0` a gap before it. -/
theorem qstar_iff_word {N : ℕ} (hper : ∀ n, gapRes (q := q) op (n + N) = gapRes op n)
    (hN : 0 < N) (J : ℕ) : QstarNonempty c op J ↔ RealisedWord c op (J - 2) := by
  constructor
  · exact word_of_window c op
  · rintro ⟨i, w, hl, hw, hwa⟩
    cases i with
    | zero =>
        refine ⟨N - 1, w, hl, hw, ?_⟩
        rw [show N - 1 + 1 = 0 + N by omega]
        exact wordAt_shift c op hper w 0 hwa
    | succ i => exact ⟨i, w, hl, hw, hwa⟩

theorem wordAt_take : ∀ (m : ℕ) (w : List Cls) (i : ℕ), WordAt c op i w → WordAt c op i (w.take m)
  | 0, _, _, _ => trivial
  | _ + 1, [], _, h => h
  | m + 1, a :: w, i, ⟨h1, h2⟩ => by
      rw [List.take_succ_cons]
      exact ⟨h1, wordAt_take m w (i + 1) h2⟩

/-- A realised legal word of length `m` gives one of every length `m' <= m`. -/
theorem realisedWord_mono {m m' : ℕ} (h : RealisedWord c op m) (hm : m' ≤ m) :
    RealisedWord c op m' := by
  obtain ⟨i, w, hl, hw, hwa⟩ := h
  refine ⟨i, w.take m', ?_, legal_take m' hw, wordAt_take c op m' w i hwa⟩
  rw [List.length_take, hl]
  omega

/-- **`J_max = L + 2`** (R89).  If a legal word of length `L` is realised and
none of length `L+1` is, then `Q*_J > -inf` exactly for `J <= L + 2`. -/
theorem jmax {N L : ℕ} (hper : ∀ n, gapRes (q := q) op (n + N) = gapRes op n) (hN : 0 < N)
    (hL : RealisedWord c op L) (hL1 : ¬ RealisedWord c op (L + 1)) (J : ℕ) :
    QstarNonempty c op J ↔ J ≤ L + 2 := by
  rw [qstar_iff_word c op hper hN]
  constructor
  · intro h
    by_contra hc
    push Not at hc
    exact hL1 (realisedWord_mono c op h (by omega))
  · intro h
    exact realisedWord_mono c op hL (by omega)

/-- **`A_kill = L + 1`** (R89).  Under the same two facts, a `k`-chain exists
exactly for `1 <= k <= L + 1`.  No periodicity needed. -/
theorem akill {L : ℕ} (hL : RealisedWord c op L) (hL1 : ¬ RealisedWord c op (L + 1))
    (k : ℕ) (hk : 1 ≤ k) : (∃ n, Chain c op k n) ↔ k ≤ L + 1 := by
  obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  have hcw : (∃ n, Chain c op (k + 1) n) ↔ RealisedWord c op k := by
    simp only [chain_iff_word]
    constructor
    · rintro ⟨n, w, hl, hw, hwa⟩; exact ⟨n, w, hl, hw, hwa⟩
    · rintro ⟨n, w, hl, hw, hwa⟩; exact ⟨n, w, hl, hw, hwa⟩
  rw [hcw]
  constructor
  · intro h
    by_contra hc
    push Not at hc
    exact hL1 (realisedWord_mono c op h (by omega))
  · intro h
    exact realisedWord_mono c op hL (by omega)

/-- The middle span of a word telescopes to the sum of its letters. -/
theorem middle_span : ∀ (w : List Cls) (i : ℕ), WordAt c op i w →
    (op (i + w.length) : ZMod q) - (op i : ZMod q) = (w.map (Cls.val c)).sum
  | [], i, _ => by simp
  | a :: w, i, ⟨h1, h2⟩ => by
      rw [List.map_cons, List.sum_cons, ← middle_span w (i + 1) h2, ← h1, List.length_cons,
        show i + (w.length + 1) = i + 1 + w.length by omega]
      unfold gapRes
      ring

/-- **R90 for a window**: the middle span `x_{J-1} - x_1` of a word-legal
window is `0 mod q` iff the number of non-padded middles is even. -/
theorem same_tooth_window (h2 : (2 : ZMod q) * c ≠ 0) {t : Bool} {w : List Cls} {i : ℕ}
    (hw : Alt t w) (hwa : WordAt c op (i + 1) w) :
    (op (i + 1 + w.length) : ZMod q) - (op (i + 1) : ZMod q) = 0 ↔ Even (nonpad w) := by
  rw [middle_span c op w (i + 1) hwa]
  exact same_tooth c h2 hw

/-- **R90, literal even case**: a literal (no padded middle) word of even length
has middle span `0 mod q` - its first and last killed openings sit on the same
tooth. -/
theorem literal_even_span (h2 : (2 : ZMod q) * c ≠ 0) {t : Bool} {w : List Cls} {i : ℕ}
    (hw : Alt t w) (hwa : WordAt c op (i + 1) w) (hlit : ∀ a ∈ w, a ≠ Cls.pad)
    (heven : Even w.length) :
    (op (i + 1 + w.length) : ZMod q) - (op (i + 1) : ZMod q) = 0 := by
  rw [same_tooth_window c op h2 hw hwa]
  have : nonpad w = w.length := by
    clear hw hwa heven
    induction w with
    | nil => rfl
    | cons a w ih =>
        have ha : a ≠ Cls.pad := hlit a (by simp)
        have ih' := ih (fun b hb => hlit b (by simp [hb]))
        cases a with
        | pad => exact absurd rfl ha
        | up => simp [nonpad, ih']
        | down => simp [nonpad, ih']
  rw [this]
  exact heven

end Machine

end WordLegal
