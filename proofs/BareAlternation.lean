/-
THE BARE-ALTERNATION NECESSARY CONDITION (Formalist, round 31).

Round 31's ONE LEMMA, kernel half.  Lateral's round-30 observation (item 79):

  "call (a,b,a) admissible if some residue mod 5 (and mod 7) carries
   r, r+a, r+a+b, r+2a+b outside the gear's tooth pair.  The real machine's
   alternation is NOT admissible at 13->17 (6,11,6), 17->19 (6,13,6) and
   23->29 (10,19,10) - so its L <= 2 there is decided by gears 5 and 7 alone -
   and IS admissible at 19->23 (8,15,8)."

  P(L >= 3 | NOT admissible) = 0 in 21,357 bare-letter deep words.

THIS FILE PROVES THE NECESSITY DIRECTION, and computes the inadmissible set.

WHAT IS PROVED HERE
  1. THE NECESSARY CONDITION (`no_gapWord`), over an ABSTRACT machine.  If a
     gear `g` with teeth `{u, g-u}` blocks the opening predicate `E`, and `op`
     enumerates openings of `E`, then a run of consecutive gaps spelling the
     value word `gs` at index `i` puts every prefix-sum offset of `gs` on an
     OPEN slot, so the translate `op i mod g` of the offset set fits inside
     `E_g = Z_g \ {u, g-u}`.  Contrapositive: if NO translate fits
     (`fitsB g u (offsets gs) = false`, a finite Bool test) then the word `gs`
     is realised NOWHERE.  This is `AlternationOrder.fitsAt` in machine
     coordinates: there the offsets are built from the class `c = q' mod 210`,
     here they are the actual gap values, and `bareFits_eq_fits` proves the two
     agree at all 48 classes.
  2. THE SET S (`bareAlt_inadmissible_iff`), by `decide` over the 48 classes
     `c` mod 210 coprime to 210.  With the BARE letters
         a = (c -+ 1)/3   (`aOfClass`; the sign from `c mod 6`, R74's `2u'`)
         b = c - a        (`bOfClass`;  the other bare letter, `q' - 2u'`)
     the bare 3-word is (a,b,a) or (b,a,b), offsets [0,a,c,c+a] / [0,b,c,c+b].
     Both a and b reduce mod 5 and mod 7 to functions of `c` alone (5 and 7
     divide 210), so ADMISSIBILITY IS CLASS-DETERMINED even though a and b
     themselves are not (`a = (q' -+ 1)/3` grows with q').  |S| = 28.
  3. THE CROSS-CHECKS, in the kernel: `bareAdm c 3 = AlternationOrder.survMax
     c 4` (round 29's independent vehicle, built from `aMod`/`inv3` instead of
     Nat division), `bareAdm c 3 = false <-> AlternationOrder.psMax c <= 3`,
     and hence `<-> LiteralCapTable.capC c <= 3` through R74's
     `ps_max_eq_capC`.  `S` is also closed under `c -> 210 - c` (Lateral's
     mirror at the class level): that map sends `a -> 70 - a` and
     `b -> 140 - b`, so both offset sets negate mod 35 and each gear's tooth
     pair is closed under negation - so the one-start-letter sets `S_A`
     (32 classes) and `S_B` (36) are mirror-closed too, with the SAME start
     letter.  `S = S_A ∩ S_B`, `|S| = 28`.

WHAT IS **NOT** PROVED HERE, and it is the honest boundary.  The conclusion is
about BARE words only: runs of consecutive gaps whose VALUES are exactly `a`
or `b`.  A legal word may use a padded letter (gap `= q'`) or a non-bare
literal (gap `= a + q'`, `b + q'`, ...), and those are untouched.  So this
bounds `L_bare`, NOT `L`.  The corpus table says the gap is real:

    M          11 13 17 19 23 29 31 37 41 43 47 53
    q'         13 17 19 23 29 31 37 41 43 47 53 59
    L           1  1  1  2  1  3  3  2  2  2  4  3
    L_bare cap  1  1  1  3  2  3  5  1  1  1  5  2      (= psMax(q' mod 210) - 1)

at M = 37, 41, 43 and 53 the machine's L EXCEEDS the bare cap, so the deep
words there are NOT bare - a prediction, not a defect (see the round-31 block).

Gate: `research/bare_alt_r31.py` reproduces S, S_A, S_B, the psMin
distribution and the table from an independent Python implementation
(6 assertion gates, exit 0).
-/

import AlternationOrder

namespace BareAlt

/-! ## 1. The necessary condition, over an abstract machine -/

/-- Gear `g` with teeth `{u, g - u}` BLOCKS the opening predicate `E`: no open
slot of `E` sits on either tooth.  (For the real machine `u = 6⁻¹ mod g`, so
gear 5 has teeth `{1,4}` and gear 7 has teeth `{6,1}` - `Machine11.expT`.) -/
def Blocks (E : ℕ → Prop) (g u : ℕ) : Prop :=
  ∀ k, E k → k % g ≠ u ∧ k % g ≠ g - u

/-- **The finite fit test at one gear.**  Some translate `t` of the offset list
avoids both teeth of `g`. -/
def fitsB (g u : ℕ) (offs : List ℕ) : Bool :=
  (List.range g).any fun t =>
    offs.all fun o => ((t + o) % g != u) && ((t + o) % g != g - u)

/-- **The necessary condition.**  Open slots at `k + offs` force a fit. -/
theorem fitsB_of_open {E : ℕ → Prop} {g u : ℕ} (hg : 0 < g) (hb : Blocks E g u)
    {k : ℕ} {offs : List ℕ} (hk : ∀ o ∈ offs, E (k + o)) : fitsB g u offs = true := by
  unfold fitsB
  rw [List.any_eq_true]
  refine ⟨k % g, List.mem_range.mpr (Nat.mod_lt _ hg), ?_⟩
  rw [List.all_eq_true]
  intro o ho
  have hmod : (k % g + o) % g = (k + o) % g := Nat.mod_add_mod k g o
  have h := hb _ (hk o ho)
  rw [hmod]
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq]
  exact ⟨h.1, h.2⟩

/-- Contrapositive: no translate fits ⇒ those offsets are never all open. -/
theorem not_open_of_not_fits {E : ℕ → Prop} {g u : ℕ} (hg : 0 < g)
    (hb : Blocks E g u) {offs : List ℕ} (h : fitsB g u offs = false) (k : ℕ) :
    ¬ (∀ o ∈ offs, E (k + o)) := by
  intro hk
  rw [fitsB_of_open hg hb hk] at h
  exact Bool.noConfusion h

/-! ## 2. Gap words and their prefix-sum offsets -/

/-- The prefix-sum offsets of a value word: `offsets [a,b,a] = [0,a,a+b,a+b+a]`
(one more point than there are letters). -/
def offsets : List ℕ → List ℕ
  | [] => [0]
  | g :: gs => 0 :: (offsets gs).map (g + ·)

/-- "the gaps of the machine from index `i` on are exactly the values `gs`":
a REALISED word, in gap VALUES, not residues. -/
def GapWordAt (op : ℕ → ℕ) : ℕ → List ℕ → Prop
  | _, [] => True
  | i, g :: gs => op (i + 1) = op i + g ∧ GapWordAt op (i + 1) gs

/-- Every prefix-sum offset of a realised word sits on an opening. -/
theorem open_of_gapWord {E : ℕ → Prop} {op : ℕ → ℕ} (hE : ∀ n, E (op n)) :
    ∀ (gs : List ℕ) (i : ℕ), GapWordAt op i gs → ∀ o ∈ offsets gs, E (op i + o) := by
  intro gs
  induction gs with
  | nil =>
      intro i _ o ho
      simp only [offsets, List.mem_singleton] at ho
      subst ho
      simpa using hE i
  | cons g gs ih =>
      intro i h o ho
      simp only [offsets, List.mem_cons, List.mem_map] at ho
      rcases ho with rfl | ⟨o', ho', rfl⟩
      · simpa using hE i
      · have hx := ih (i + 1) h.2 o' ho'
        rw [h.1] at hx
        simpa [Nat.add_assoc] using hx

/-- **THE LEMMA, abstract half.**  A gear that blocks `E`, and an offset set of
`gs` that fits nowhere in that gear, forbid `gs` at EVERY index. -/
theorem no_gapWord {E : ℕ → Prop} {op : ℕ → ℕ} (hE : ∀ n, E (op n))
    {g u : ℕ} (hg : 0 < g) (hb : Blocks E g u) {gs : List ℕ}
    (h : fitsB g u (offsets gs) = false) (i : ℕ) : ¬ GapWordAt op i gs :=
  fun hw => not_open_of_not_fits hg hb h (op i) (open_of_gapWord hE gs i hw)

/-! ## 3. The bare alternation -/

/-- The alternating value word: `altWord a b 3 = [a, b, a]`. -/
def altWord (a b : ℕ) : ℕ → List ℕ
  | 0 => []
  | n + 1 => a :: altWord b a n

/-- The bare alternation of `m` letters, started with `a`, fits at gears 5
(teeth `{1,4}`) AND 7 (teeth `{6,1}`). -/
def bareFits (a b m : ℕ) : Bool :=
  fitsB 5 1 (offsets (altWord a b m)) && fitsB 7 6 (offsets (altWord a b m))

/-- ADMISSIBLE: some start letter's `m`-letter bare alternation fits at both
gears.  A realised bare word of length `m` is either `(a,b,a,..)` or
`(b,a,b,..)`, so this is the disjunction. -/
def bareAdmAB (a b m : ℕ) : Bool := bareFits a b m || bareFits b a m

/-- **THE LEMMA, machine half.**  If the bare alternation of length `m` is
inadmissible at gear `g` (either start letter), no run of `m` consecutive gaps
of the machine is a bare alternation. -/
theorem no_bare_run {E : ℕ → Prop} {op : ℕ → ℕ} (hE : ∀ n, E (op n))
    (h5 : Blocks E 5 1) (h7 : Blocks E 7 6) {a b m : ℕ}
    (h : bareAdmAB a b m = false) (i : ℕ) :
    ¬ GapWordAt op i (altWord a b m) ∧ ¬ GapWordAt op i (altWord b a m) := by
  have hA : fitsB 5 1 (offsets (altWord a b m)) = false ∨
      fitsB 7 6 (offsets (altWord a b m)) = false := by
    unfold bareAdmAB bareFits at h
    revert h
    cases fitsB 5 1 (offsets (altWord a b m)) <;>
      cases fitsB 7 6 (offsets (altWord a b m)) <;> simp
  have hB : fitsB 5 1 (offsets (altWord b a m)) = false ∨
      fitsB 7 6 (offsets (altWord b a m)) = false := by
    unfold bareAdmAB bareFits at h
    revert h
    cases fitsB 5 1 (offsets (altWord b a m)) <;>
      cases fitsB 7 6 (offsets (altWord b a m)) <;> simp
  constructor
  · rcases hA with hA | hA
    · exact no_gapWord hE (by norm_num) h5 hA i
    · exact no_gapWord hE (by norm_num) h7 hA i
  · rcases hB with hB | hB
    · exact no_gapWord hE (by norm_num) h5 hB i
    · exact no_gapWord hE (by norm_num) h7 hB i

/-! ## 4. Longer runs contain shorter ones -/

theorem gapWordAt_take {op : ℕ → ℕ} : ∀ (gs : List ℕ) (i m : ℕ),
    GapWordAt op i gs → GapWordAt op i (gs.take m) := by
  intro gs
  induction gs with
  | nil => intro i m _; cases m <;> trivial
  | cons g gs ih =>
      intro i m h
      cases m with
      | zero => trivial
      | succ m => exact ⟨h.1, ih (i + 1) m h.2⟩

theorem altWord_take : ∀ (n m a b : ℕ), m ≤ n → (altWord a b n).take m = altWord a b m := by
  intro n
  induction n with
  | zero => intro m a b hm; interval_cases m; rfl
  | succ n ih =>
      intro m a b hm
      cases m with
      | zero => rfl
      | succ m =>
          simp only [altWord, List.take_succ_cons, List.cons.injEq, true_and]
          exact ih m b a (by omega)

/-- **`L_bare <= m - 1`**: no realised bare alternating run of `m` gaps, and
hence none of any greater length. -/
theorem no_bare_run_ge {E : ℕ → Prop} {op : ℕ → ℕ} (hE : ∀ n, E (op n))
    (h5 : Blocks E 5 1) (h7 : Blocks E 7 6) {a b m : ℕ}
    (h : bareAdmAB a b m = false) {n : ℕ} (hn : m ≤ n) (i : ℕ) :
    ¬ GapWordAt op i (altWord a b n) ∧ ¬ GapWordAt op i (altWord b a n) := by
  have key := no_bare_run hE h5 h7 h i
  constructor
  · intro hw
    exact key.1 (by
      have := gapWordAt_take (op := op) (altWord a b n) i m hw
      rwa [altWord_take n m a b hn] at this)
  · intro hw
    exact key.2 (by
      have := gapWordAt_take (op := op) (altWord b a n) i m hw
      rwa [altWord_take n m b a hn] at this)

/-! ## 5. The set S, by `decide` over the 48 classes mod 210 -/

/-- The bare letter in the class `+d'`: `3a = c - 1` when `c = 1 mod 6`, else
`3a = c + 1` (`AlternationOrder.aMod` without the reduction mod a gear). -/
def aOfClass (c : ℕ) : ℕ := if c % 6 = 1 then (c - 1) / 3 else (c + 1) / 3

/-- The other bare letter, `b = q' - a` - class-determined mod 5 and mod 7
because 5 and 7 divide 210. -/
def bOfClass (c : ℕ) : ℕ := c - aOfClass c

/-- Admissibility of the `m`-letter bare alternation of the class `c`. -/
def bareAdm (c m : ℕ) : Bool := bareAdmAB (aOfClass c) (bOfClass c) m

set_option maxRecDepth 40000

/-- **THE INADMISSIBLE SET S**: the 28 classes mod 210 at which NEITHER bare
3-word `(a,b,a)` nor `(b,a,b)` has a translate fitting inside the exposed sets
of gears 5 and 7. -/
def S : List ℕ :=
  [11, 13, 17, 19, 29, 41, 43, 47, 59, 71, 73, 79, 101, 103, 107, 109, 131,
    137, 139, 151, 163, 167, 169, 181, 191, 193, 197, 199]

/-- **S, kernel-checked.** -/
theorem bareAlt_inadmissible_iff : ∀ c < 210, Nat.gcd c 210 = 1 →
    (bareAdm c 3 = false ↔ c ∈ S) := by decide

/-- 28 classes of 48. -/
theorem S_card : S.length = 28 := by decide

/-- S is closed under the mirror `c ↦ 210 - c` (Lateral's record mirror at the
level of the class). -/
theorem S_mirror : ∀ c < 210, Nat.gcd c 210 = 1 →
    (bareAdm c 3 = false ↔ bareAdm (210 - c) 3 = false) := by decide

/-- Each start letter's set is mirror-closed on its own, with the SAME start
letter: `c ↦ 210 - c` sends `a ↦ 70 - a` and `b ↦ 140 - b`, so both offset
sets negate mod 35, and each gear's tooth pair is closed under negation. -/
theorem S_half_mirror : ∀ c < 210, Nat.gcd c 210 = 1 →
    (bareFits (aOfClass c) (bOfClass c) 3 =
        bareFits (aOfClass (210 - c)) (bOfClass (210 - c)) 3 ∧
      bareFits (bOfClass c) (aOfClass c) 3 =
        bareFits (bOfClass (210 - c)) (aOfClass (210 - c)) 3) := by decide

/-! ### PSORD: Constructor's round-31 table in the kernel

`psord c` is the longest bare alternation length, in LETTERS, that is
admissible at gears 5 and 7 for the class `c` - Constructor's `PSORD`
(`docs/novel/bare-word-uniform-cap.md`).  `bareAdm_downward` makes the count a
maximum.  `S = {c : psord c ≤ 2}`. -/

/-- Admissibility is downward closed in the length (the offset sets nest). -/
theorem bareAdm_downward : ∀ c < 210, Nat.gcd c 210 = 1 → ∀ m < 9,
    bareAdm c (m + 2) = true → bareAdm c (m + 1) = true := by decide

/-- The bare order in letters (a count over lengths 1..9, a maximum by
`bareAdm_downward`). -/
def psord (c : ℕ) : ℕ := ((List.range 9).filter fun m => bareAdm c (m + 1)).length

/-- **`L_bare <= PSORD <= 5`, finite half**: no class admits a 6-letter bare
alternation. -/
theorem psord_le_five : ∀ c < 210, Nat.gcd c 210 = 1 → psord c ≤ 5 := by decide

/-- **PSORD = 4 is EMPTY**: a class admitting a 4-letter bare alternation
admits a 5-letter one. -/
theorem psord_ne_four : ∀ c < 210, Nat.gcd c 210 = 1 → psord c ≠ 4 := by decide

/-- The 24 classes of order 1: not even a bare PAIR occurs. -/
theorem psord_eq_one_iff : ∀ c < 210, Nat.gcd c 210 = 1 →
    (psord c = 1 ↔ c ∈ [11, 13, 17, 19, 41, 43, 47, 71, 73, 79, 101, 103, 107,
      109, 131, 137, 139, 163, 167, 169, 191, 193, 197, 199]) := by decide

/-- The 4 classes of order 2. -/
theorem psord_eq_two_iff : ∀ c < 210, Nat.gcd c 210 = 1 →
    (psord c = 2 ↔ c ∈ [29, 59, 151, 181]) := by decide

/-- The 6 classes of order 5 - R74's exceptional six again. -/
theorem psord_eq_five_iff : ∀ c < 210, Nat.gcd c 210 = 1 →
    (psord c = 5 ↔ c ∈ [37, 53, 83, 127, 157, 173]) := by decide

/-- **S is exactly the order-≤2 set** (Constructor's definition of S). -/
theorem S_iff_psord : ∀ c < 210, Nat.gcd c 210 = 1 →
    (c ∈ S ↔ psord c ≤ 2) := by decide

/-- `psord` is one less than round 29's point count. -/
theorem psord_succ_eq_psMax : ∀ c < 210, Nat.gcd c 210 = 1 →
    psord c + 1 = AlternationOrder.psMax c := by decide

/-! ### Cross-checks against round 29's independent vehicle -/

/-- The two vehicles agree letter by letter: this file builds the offsets from
`(c ∓ 1)/3` in ℕ, `AlternationOrder.fitsAt` builds them from `3⁻¹ mod g`. -/
theorem bareFits_eq_fits : ∀ c < 210, Nat.gcd c 210 = 1 →
    bareFits (aOfClass c) (bOfClass c) 3 = AlternationOrder.fits c 4 true ∧
      bareFits (bOfClass c) (aOfClass c) 3 = AlternationOrder.fits c 4 false := by
  decide

/-- Admissibility IS `AlternationOrder.survMax` at 4 points. -/
theorem bareAdm_eq_survMax : ∀ c < 210, Nat.gcd c 210 = 1 →
    bareAdm c 3 = AlternationOrder.survMax c 4 := by decide

/-- Inadmissibility IS the maximising order being at most 3. -/
theorem inadmissible_iff_psMax : ∀ c < 210, Nat.gcd c 210 = 1 →
    (bareAdm c 3 = false ↔ AlternationOrder.psMax c ≤ 3) := by decide

/-- ... and hence, through R74's `ps_max_eq_capC`, the literal cap being at
most 3: **S = the classes whose literal cap is <= 3**. -/
theorem inadmissible_iff_capC (c : ℕ) (hc : c < 210) (hg : Nat.gcd c 210 = 1) :
    bareAdm c 3 = false ↔ LiteralCapTable.capC c ≤ 3 := by
  rw [inadmissible_iff_psMax c hc hg, AlternationOrder.ps_max_eq_capC c hc hg]

/-! ### The bare letters of a real gear are the class's bare letters

`a = (q' ∓ 1)/3` and `q' ≡ c (mod 210)` give `a ≡ aOfClass c (mod 70)`, so
`a` and `aOfClass c` agree mod 5 and mod 7 - which is all `fitsB` sees. -/

theorem aOfClass_mod_five (q a : ℕ) (ha : 3 * a + 1 = q ∨ 3 * a = q + 1) :
    a % 5 = aOfClass (q % 210) % 5 := by
  unfold aOfClass
  split <;> omega

theorem aOfClass_mod_seven (q a : ℕ) (ha : 3 * a + 1 = q ∨ 3 * a = q + 1) :
    a % 7 = aOfClass (q % 210) % 7 := by
  unfold aOfClass
  split <;> omega

theorem bOfClass_mod_five (q a b : ℕ) (ha : 3 * a + 1 = q ∨ 3 * a = q + 1)
    (hb : a + b = q) : b % 5 = bOfClass (q % 210) % 5 := by
  unfold bOfClass aOfClass
  split <;> omega

theorem bOfClass_mod_seven (q a b : ℕ) (ha : 3 * a + 1 = q ∨ 3 * a = q + 1)
    (hb : a + b = q) : b % 7 = bOfClass (q % 210) % 7 := by
  unfold bOfClass aOfClass
  split <;> omega

/-! ## 6. `fitsB` sees the offsets only mod `g` -/

theorem all_map_mod (g t u : ℕ) : ∀ o : List ℕ,
    (o.map (· % g)).all (fun x => ((t + x) % g != u) && ((t + x) % g != g - u)) =
      o.all (fun x => ((t + x) % g != u) && ((t + x) % g != g - u)) := by
  intro o
  induction o with
  | nil => rfl
  | cons x xs ih => simp only [List.map_cons, List.all_cons, ih, Nat.add_mod_mod]

theorem fitsB_map_mod (g u : ℕ) (o : List ℕ) :
    fitsB g u (o.map (· % g)) = fitsB g u o := by
  unfold fitsB
  refine congrArg _ (funext fun t => ?_)
  exact all_map_mod g t u o

theorem fitsB_congr {g u : ℕ} {o o' : List ℕ}
    (h : o.map (· % g) = o'.map (· % g)) : fitsB g u o = fitsB g u o' := by
  rw [← fitsB_map_mod g u o, ← fitsB_map_mod g u o', h]

/-- The 4-point offset set of a bare 3-word, spelled out. -/
theorem offsets_bare3 (a b : ℕ) :
    offsets (altWord a b 3) = [0, a, a + b, a + (b + a)] := rfl

theorem fitsB_bare3_congr {g u a b a' b' : ℕ} (hga : a % g = a' % g)
    (hgb : b % g = b' % g) :
    fitsB g u (offsets (altWord a b 3)) = fitsB g u (offsets (altWord a' b' 3)) := by
  refine fitsB_congr ?_
  rw [offsets_bare3, offsets_bare3]
  have h1 : (a + b) % g = (a' + b') % g := by
    conv_lhs => rw [Nat.add_mod, hga, hgb]
    conv_rhs => rw [Nat.add_mod]
  have h2 : (a + (b + a)) % g = (a' + (b' + a')) % g := by
    conv_lhs => rw [Nat.add_mod, Nat.add_mod b a, hga, hgb]
    conv_rhs => rw [Nat.add_mod, Nat.add_mod b' a']
  simp only [List.map_cons, List.map_nil]
  rw [hga, h1, h2]

theorem bareAdmAB_congr {a b a' b' : ℕ}
    (h5a : a % 5 = a' % 5) (h5b : b % 5 = b' % 5)
    (h7a : a % 7 = a' % 7) (h7b : b % 7 = b' % 7) :
    bareAdmAB a b 3 = bareAdmAB a' b' 3 := by
  unfold bareAdmAB bareFits
  rw [fitsB_bare3_congr h5a h5b, fitsB_bare3_congr h7a h7b,
    fitsB_bare3_congr h5b h5a, fitsB_bare3_congr h7b h7a]

/-! ## 7. THE LEMMA, assembled: the class decides the bare 3-word -/

/-- **`q' mod 210 ∈ S ⇒ M has no realised bare word of length 3`.**
`a` and `b` are the bare letters of the incoming gear `q'` (`3a = q' ∓ 1`,
`a + b = q'`); `E` is the machine's opening predicate, blocked by gears 5 and
7 at their teeth; `op` enumerates openings.  No index carries the consecutive
gaps `(a,b,a)` or `(b,a,b)`, hence (by `no_bare_run_ge`) none of any greater
length: `L_bare(M) ≤ 2`. -/
theorem no_bare3_of_class_mem {E : ℕ → Prop} {op : ℕ → ℕ} (hE : ∀ n, E (op n))
    (h5 : Blocks E 5 1) (h7 : Blocks E 7 6) {q a b : ℕ}
    (ha : 3 * a + 1 = q ∨ 3 * a = q + 1) (hab : a + b = q)
    (hS : q % 210 ∈ S) (hlt : q % 210 < 210) (hg : Nat.gcd (q % 210) 210 = 1)
    (i : ℕ) :
    ¬ GapWordAt op i (altWord a b 3) ∧ ¬ GapWordAt op i (altWord b a 3) := by
  have hcls : bareAdm (q % 210) 3 = false :=
    (bareAlt_inadmissible_iff _ hlt hg).mpr hS
  have hbridge : bareAdmAB a b 3 = bareAdm (q % 210) 3 :=
    bareAdmAB_congr (aOfClass_mod_five q a ha) (bOfClass_mod_five q a b ha hab)
      (aOfClass_mod_seven q a ha) (bOfClass_mod_seven q a b ha hab)
  exact no_bare_run hE h5 h7 (hbridge.trans hcls) i

end BareAlt
