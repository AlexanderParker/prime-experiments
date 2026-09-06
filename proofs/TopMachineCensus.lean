import TopMachineWalk
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.GroupWithZero.Finset
import Mathlib.Algebra.BigOperators.Group.Finset.Powerset

/-!
# The gap census law (L22 / register W22) in the kernel

Round 37 (Formalist).  `research/proof/top_machine_2.md` L22 and the kernel shape
in `research/proof/top_machine_7.md` section 5: K1 (the local characterisation),
K2 (inclusion-exclusion over the interior positions), K3 (the CRT product per
subset), assembled into

    N_d(G) = sum over S subset of [1, d-1] of (-1)^|S| prod_{g in G} (g - |E_g(S)|),
    E_g(S) = ({0, -2, -d, -d-2} u {-j, -(j+2) : j in S}) mod g .

`N_d(G)` is the number of residues `n` mod the wheel `W = prod G` such that `n`
and `n + d` are consecutive open pairs: both open, every `n + j` with
`0 < j < d` struck.

Hypotheses as the proofs needed them: every gear positive, gears pairwise
coprime.  No size hypothesis (`E_g(S)` is a `Finset` image, so its card is
whatever it is), no primality.

Zero sorries; no `native_decide`, no `Lean.ofReduceBool`, no `decide`.
-/
namespace TopMachine

/-! ## 0. The object: consecutive open pairs -/

/-- **`n` and `n + d` are consecutive open pairs** (raw-line form): both open,
nothing open strictly between them. -/
def ConsecOpen (G : Finset ℕ) (d : ℕ) (n : ℤ) : Prop :=
  IsOpen G n ∧ IsOpen G (n + d) ∧ ∀ z : ℤ, n < z → z < n + d → ¬ IsOpen G z

/-- The residue form on `ℕ`, over the interior positions `Finset.Ioo 0 d`. -/
def ConsecOpenN (G : Finset ℕ) (d n : ℕ) : Prop :=
  OpenN G n ∧ OpenN G (n + d) ∧ ∀ j ∈ Finset.Ioo 0 d, ¬ OpenN G (n + j)

instance decConsecOpenN (G : Finset ℕ) (d n : ℕ) : Decidable (ConsecOpenN G d n) := by
  unfold ConsecOpenN; infer_instance

/-- The two forms agree on natural arguments. -/
theorem consecOpen_natCast (G : Finset ℕ) (d n : ℕ) :
    ConsecOpen G d (n : ℤ) ↔ ConsecOpenN G d n := by
  unfold ConsecOpen ConsecOpenN
  rw [open_natCast, show ((n : ℤ) + (d : ℤ)) = ((n + d : ℕ) : ℤ) by push_cast; ring,
    open_natCast]
  refine and_congr Iff.rfl (and_congr Iff.rfl ?_)
  constructor
  · intro h j hj
    have hj' := Finset.mem_Ioo.mp hj
    rw [← open_natCast]
    push_cast
    exact h _ (by omega) (by omega)
  · intro h z h1 h2
    obtain ⟨j, hj⟩ := Int.eq_ofNat_of_zero_le (show (0 : ℤ) ≤ z - n by omega)
    have hz : z = (n : ℤ) + j := by omega
    subst hz
    have := h j (Finset.mem_Ioo.mpr ⟨by omega, by omega⟩)
    rw [← open_natCast] at this
    push_cast at this
    exact this

/-- The census: the number of residues mod the wheel at which `n` and `n + d`
are consecutive open pairs. -/
def Ncount (G : Finset ℕ) (d : ℕ) : ℕ :=
  ((Finset.range (∏ g ∈ G, g)).filter (ConsecOpenN G d)).card

/-! ## 1. The forbidden residue sets `E_g(S)` -/

/-- The offsets `x` at which `n + x` must not be a multiple of the gear: the
four boundary offsets `0, 2, d, d + 2` and, for each `j ∈ S`, `j` and `j + 2`. -/
def Offs (d : ℕ) (S : Finset ℕ) : Finset ℕ :=
  ({0, 2, d, d + 2} : Finset ℕ) ∪ S ∪ S.image (· + 2)

/-- `E_g(S)`: the residues `n mod g` forbidden by gear `g`, i.e. `-x mod g` for
`x` in `Offs d S`, written with the forward offset `off g x = (-x) mod g`. -/
def E (d : ℕ) (S : Finset ℕ) (g : ℕ) : Finset ℕ :=
  (Offs d S).image (fun x : ℕ => off g x)

theorem mem_Offs {d : ℕ} {S : Finset ℕ} {x : ℕ} :
    x ∈ Offs d S ↔
      x = 0 ∨ x = 2 ∨ x = d ∨ x = d + 2 ∨ x ∈ S ∨ ∃ j ∈ S, j + 2 = x := by
  unfold Offs
  simp only [Finset.mem_union, Finset.mem_insert, Finset.mem_singleton, Finset.mem_image,
    or_assoc]

theorem offs_zero_mem (d : ℕ) (S : Finset ℕ) : 0 ∈ Offs d S := by
  rw [mem_Offs]; exact Or.inl rfl

theorem offs_two_mem (d : ℕ) (S : Finset ℕ) : 2 ∈ Offs d S := by
  rw [mem_Offs]; exact Or.inr (Or.inl rfl)

theorem offs_d_mem (d : ℕ) (S : Finset ℕ) : d ∈ Offs d S := by
  rw [mem_Offs]; exact Or.inr (Or.inr (Or.inl rfl))

theorem offs_d_two_mem (d : ℕ) (S : Finset ℕ) : d + 2 ∈ Offs d S := by
  rw [mem_Offs]; exact Or.inr (Or.inr (Or.inr (Or.inl rfl)))

theorem offs_mem_of_mem {d : ℕ} {S : Finset ℕ} {j : ℕ} (hj : j ∈ S) : j ∈ Offs d S := by
  rw [mem_Offs]; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl hj))))

theorem offs_add_two_mem_of_mem {d : ℕ} {S : Finset ℕ} {j : ℕ} (hj : j ∈ S) :
    j + 2 ∈ Offs d S := by
  rw [mem_Offs]; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨j, hj, rfl⟩))))

/-- `E_g(S)` lies in `range g`. -/
theorem E_subset_range {g : ℕ} (hg : 0 < g) (d : ℕ) (S : Finset ℕ) :
    E d S g ⊆ Finset.range g := by
  intro r hr
  obtain ⟨x, _, rfl⟩ := Finset.mem_image.mp hr
  exact Finset.mem_range.mpr (off_lt hg _)

theorem E_card_le {g : ℕ} (hg : 0 < g) (d : ℕ) (S : Finset ℕ) : (E d S g).card ≤ g := by
  have := Finset.card_le_card (E_subset_range hg d S)
  rwa [Finset.card_range] at this

/-- `E_g(∅)` is the four-tooth set of the pair correlation (round 34). -/
theorem E_empty_eq_corrTeeth (d g : ℕ) : E d ∅ g = CorrTeeth g d := by
  unfold E Offs CorrTeeth
  simp only [Finset.image_empty, Finset.union_empty, Finset.image_insert,
    Finset.image_singleton, Nat.cast_zero, Nat.cast_ofNat, Nat.cast_add]

/-! ## 2. K1: the local characterisation -/

/-- `n ≡ -x (mod g)` iff `g | n + x`. -/
theorem mod_eq_off_iff {g : ℕ} (hg : 0 < g) (n x : ℕ) :
    n % g = off g (x : ℤ) ↔ (n + x) % g = 0 := by
  have hr : n % g < g := Nat.mod_lt n hg
  have e : n + x = g * (n / g) + (n % g + x) := by
    rw [← Nat.add_assoc, Nat.div_add_mod]
  rw [← Nat.dvd_iff_mod_eq_zero, e, Nat.dvd_add_right (Nat.dvd_mul_right g (n / g)),
    ← Int.natCast_dvd_natCast, Nat.cast_add, add_comm, dvd_iff_off_eq hg hr (x : ℤ)]
  exact eq_comm

/-- Gear `g` strikes the pair `n + x` iff `n` sits on one of the two residues
`-x`, `-(x + 2)` mod `g`. -/
theorem strikesR_add_iff {g : ℕ} (hg : 0 < g) (n x : ℕ) :
    StrikesR g (n + x) ↔
      (n % g = off g (x : ℤ) ∨ n % g = off g ((x + 2 : ℕ) : ℤ)) := by
  unfold StrikesR
  rw [mod_eq_off_iff hg, mod_eq_off_iff hg, Nat.add_assoc]

theorem mem_E_iff {g : ℕ} (hg : 0 < g) (d : ℕ) (S : Finset ℕ) (n : ℕ) :
    n % g ∈ E d S g ↔ ∃ x ∈ Offs d S, (n + x) % g = 0 := by
  unfold E
  rw [Finset.mem_image]
  constructor
  · rintro ⟨x, hx, hx'⟩; exact ⟨x, hx, (mod_eq_off_iff hg n x).mp hx'.symm⟩
  · rintro ⟨x, hx, hx'⟩; exact ⟨x, hx, ((mod_eq_off_iff hg n x).mpr hx').symm⟩

/-- **K1, one gear.**  `n` avoids `E_g(S)` iff gear `g` strikes neither `n` nor
`n + d` nor any `n + j` with `j ∈ S`. -/
theorem not_mem_E_iff {g : ℕ} (hg : 0 < g) (d : ℕ) (S : Finset ℕ) (n : ℕ) :
    n % g ∉ E d S g ↔
      ¬ StrikesR g n ∧ ¬ StrikesR g (n + d) ∧ ∀ j ∈ S, ¬ StrikesR g (n + j) := by
  rw [mem_E_iff hg]
  unfold StrikesR
  constructor
  · intro h
    refine ⟨?_, ?_, fun j hj => ?_⟩
    · rintro (h0 | h2)
      · exact h ⟨0, offs_zero_mem d S, by simpa using h0⟩
      · exact h ⟨2, offs_two_mem d S, h2⟩
    · rintro (hd | hd2)
      · exact h ⟨d, offs_d_mem d S, hd⟩
      · exact h ⟨d + 2, offs_d_two_mem d S, by rwa [← Nat.add_assoc]⟩
    · rintro (hj' | hj')
      · exact h ⟨j, offs_mem_of_mem hj, hj'⟩
      · exact h ⟨j + 2, offs_add_two_mem_of_mem hj, by rwa [← Nat.add_assoc]⟩
  · rintro ⟨h0, hd, hS⟩ ⟨x, hx, hx'⟩
    rcases mem_Offs.mp hx with rfl | rfl | rfl | rfl | hxS | ⟨j, hj, rfl⟩
    · exact h0 (Or.inl (by simpa using hx'))
    · exact h0 (Or.inr hx')
    · exact hd (Or.inl hx')
    · exact hd (Or.inr (by rwa [Nat.add_assoc]))
    · exact (hS x hxS) (Or.inl hx')
    · exact (hS j hj) (Or.inr (by rwa [Nat.add_assoc]))

/-- **`n` avoids the forbidden set `F g` at every gear.**  The general shape of
K3's count; `E d S` is the instance the census needs. -/
def AvoidN (G : Finset ℕ) (F : ℕ → Finset ℕ) (n : ℕ) : Prop := ∀ g ∈ G, n % g ∉ F g

instance decAvoidN (G : Finset ℕ) (F : ℕ → Finset ℕ) (n : ℕ) : Decidable (AvoidN G F n) := by
  unfold AvoidN; infer_instance

/-- **K1, all gears.**  `n` avoids `E_g(S)` on every gear iff `n`, `n + d` and
every `n + j` (`j ∈ S`) are open. -/
theorem avoidN_E_iff {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g) (d : ℕ) (S : Finset ℕ) (n : ℕ) :
    AvoidN G (E d S) n ↔ OpenN G n ∧ OpenN G (n + d) ∧ ∀ j ∈ S, OpenN G (n + j) := by
  unfold AvoidN OpenN
  constructor
  · intro h
    refine ⟨fun g hg => ?_, fun g hg => ?_, fun j hj g hg => ?_⟩
    · exact ((not_mem_E_iff (hG0 g hg) d S n).mp (h g hg)).1
    · exact ((not_mem_E_iff (hG0 g hg) d S n).mp (h g hg)).2.1
    · exact ((not_mem_E_iff (hG0 g hg) d S n).mp (h g hg)).2.2 j hj
  · rintro ⟨h0, hd, hS⟩ g hg
    exact (not_mem_E_iff (hG0 g hg) d S n).mpr ⟨h0 g hg, hd g hg, fun j hj => hS j hj g hg⟩

/-- The boundary clause: `E_g(∅)` avoided on every gear iff `n` and `n + d` are
both open (the pair-correlation predicate of round 34). -/
theorem avoidN_E_empty_iff {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g) (d n : ℕ) :
    AvoidN G (E d ∅) n ↔ OpenN G n ∧ OpenN G (n + d) := by
  rw [avoidN_E_iff hG0]
  simp

/-- Position `n + x` is struck iff some gear has `n` on `-x` or `-(x + 2)`. -/
theorem not_openN_add_iff {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g) (n x : ℕ) :
    ¬ OpenN G (n + x) ↔
      ∃ g ∈ G, (n % g = off g (x : ℤ) ∨ n % g = off g ((x + 2 : ℕ) : ℤ)) := by
  unfold OpenN
  simp only [not_forall, not_not, exists_prop]
  constructor
  · rintro ⟨g, hg, hs⟩; exact ⟨g, hg, (strikesR_add_iff (hG0 g hg) n x).mp hs⟩
  · rintro ⟨g, hg, hs⟩; exact ⟨g, hg, (strikesR_add_iff (hG0 g hg) n x).mpr hs⟩

/-- **K1 as the branch states it.**  `n` is consecutive-open at distance `d` iff
no gear has `n` in `E_g(∅) = {0, -2, -d, -d-2} mod g`, and for every interior
position `j` some gear has `n` in `{-j, -(j+2)} mod g`. -/
theorem consecOpenN_iff_residues {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g) (d n : ℕ) :
    ConsecOpenN G d n ↔
      AvoidN G (E d ∅) n ∧
      ∀ j ∈ Finset.Ioo 0 d,
        ∃ g ∈ G, (n % g = off g (j : ℤ) ∨ n % g = off g ((j + 2 : ℕ) : ℤ)) := by
  unfold ConsecOpenN
  rw [avoidN_E_empty_iff hG0, and_assoc]
  refine and_congr Iff.rfl (and_congr Iff.rfl ?_)
  exact forall_congr' fun j => forall_congr' fun _ => not_openN_add_iff hG0 n j

/-- K1 in the shape K2 consumes: boundary avoided, every interior position
NOT open. -/
theorem consecOpenN_iff_avoid {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g) (d n : ℕ) :
    ConsecOpenN G d n ↔
      AvoidN G (E d ∅) n ∧ ∀ j ∈ Finset.Ioo 0 d, ¬ OpenN G (n + j) := by
  unfold ConsecOpenN
  rw [avoidN_E_empty_iff hG0, and_assoc]

/-! ## 3. K2: inclusion-exclusion over a finite set of positions -/

/-- **K2, the general identity.**  For a finite set `A` of points, a finite set
`J` of positions and a predicate `P j n` ("position `j` is unstruck at `n`"),
the number of points of `A` at which every position of `J` is struck is the
alternating sum, over subsets `S ⊆ J`, of the number of points of `A` at which
every position of `S` is unstruck.  Pointwise it is
`prod_{j ∈ J} (1 - [P j n]) = sum_S (-1)^|S| prod_{j ∈ S} [P j n]`
(`Finset.prod_add`), summed over `A`. -/
theorem card_filter_forall_not {α ι : Type*} (A : Finset α) (J : Finset ι)
    (P : ι → α → Prop) [∀ j, DecidablePred (P j)] :
    ((A.filter (fun n => ∀ j ∈ J, ¬ P j n)).card : ℤ)
      = ∑ S ∈ J.powerset,
          (-1 : ℤ) ^ S.card * ((A.filter (fun n => ∀ j ∈ S, P j n)).card : ℤ) := by
  classical
  have hpt : ∀ n, (if (∀ j ∈ J, ¬ P j n) then (1 : ℤ) else 0)
      = ∑ S ∈ J.powerset, (-1 : ℤ) ^ S.card * (if (∀ j ∈ S, P j n) then 1 else 0) := by
    intro n
    have h1 : (if (∀ j ∈ J, ¬ P j n) then (1 : ℤ) else 0)
        = ∏ j ∈ J, ((-1) * (if P j n then (1 : ℤ) else 0) + 1) := by
      rw [← Finset.prod_boole]
      refine Finset.prod_congr rfl fun j _ => ?_
      by_cases h : P j n <;> simp [h]
    rw [h1, Finset.prod_add]
    refine Finset.sum_congr rfl fun S _ => ?_
    rw [Finset.prod_const_one, mul_one, Finset.prod_mul_distrib, Finset.prod_const,
      Finset.prod_boole]
  rw [← Finset.sum_boole, Finset.sum_congr rfl (fun n _ => hpt n), Finset.sum_comm]
  refine Finset.sum_congr rfl fun S _ => ?_
  rw [← Finset.mul_sum, Finset.sum_boole]

/-! ## 4. K3: the CRT product for per-gear forbidden sets -/

theorem avoidN_congr {G : Finset ℕ} {F : ℕ → Finset ℕ} {W x y : ℕ} (hdvd : ∀ g ∈ G, g ∣ W)
    (h : x % W = y % W) : AvoidN G F x ↔ AvoidN G F y := by
  unfold AvoidN
  refine forall_congr' fun g => forall_congr' fun hg => ?_
  rw [← Nat.mod_mod_of_dvd x (hdvd g hg), ← Nat.mod_mod_of_dvd y (hdvd g hg), h]

theorem avoidN_insert {a : ℕ} {s : Finset ℕ} {F : ℕ → Finset ℕ} {n : ℕ} :
    AvoidN (insert a s) F n ↔ (n % a ∉ F a ∧ AvoidN s F n) := by
  constructor
  · intro h
    exact ⟨h a (Finset.mem_insert_self a s), fun g hg => h g (Finset.mem_insert_of_mem hg)⟩
  · rintro ⟨h1, h2⟩ g hg
    rcases Finset.mem_insert.mp hg with rfl | hg'
    · exact h1
    · exact h2 g hg'

/-- **K3, the general CRT product.**  For positive pairwise coprime gears and
any assignment `F` of a forbidden residue set to each gear, the residues mod the
wheel avoiding `F g` at every gear are counted by the product over gears of the
number of residues mod `g` outside `F g`.  `wheel_count` (`F g = {0, -2}`) and
`corr_prod` (`F g = CorrTeeth g d`) are the two earlier instances. -/
theorem card_avoid_prod : ∀ (G : Finset ℕ), (∀ g ∈ G, 0 < g) →
    (∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) → ∀ (F : ℕ → Finset ℕ),
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => AvoidN G F n)).card
      = ∏ g ∈ G, ((Finset.range g).filter (fun r => r ∉ F g)).card := by
  classical
  intro G
  induction G using Finset.induction_on with
  | empty => intro _ _ F; simp [AvoidN]
  | insert a s hasnot ih =>
      intro hG0 hcop F
      have hmemA : a ∈ insert a s := Finset.mem_insert_self a s
      have ha0 : 0 < a := hG0 a hmemA
      have hs0 : ∀ g ∈ s, 0 < g := fun g hg => hG0 g (Finset.mem_insert_of_mem hg)
      have hscop : ∀ g ∈ s, ∀ h ∈ s, g ≠ h → Nat.Coprime g h := fun g hg h hh hne =>
        hcop g (Finset.mem_insert_of_mem hg) h (Finset.mem_insert_of_mem hh) hne
      have hP0 : 0 < ∏ g ∈ s, g :=
        Finset.prod_pos fun i hi => hs0 i hi
      have hcopr : Nat.Coprime a (∏ g ∈ s, g) :=
        Nat.Coprime.prod_right fun i hi =>
          hcop a hmemA i (Finset.mem_insert_of_mem hi) (by rintro rfl; exact hasnot hi)
      have hdvd : ∀ g ∈ s, g ∣ (∏ g ∈ s, g) := fun g hg => Finset.dvd_prod_of_mem _ hg
      rw [Finset.prod_insert hasnot, Finset.prod_insert hasnot]
      have hfil : (Finset.range (a * ∏ g ∈ s, g)).filter (fun n => AvoidN (insert a s) F n)
          = (Finset.range (a * ∏ g ∈ s, g)).filter
              (fun n => n % a ∉ F a ∧ AvoidN s F n) :=
        Finset.filter_congr fun n _ => avoidN_insert
      have hres : ((Finset.range a).filter (fun n => n % a ∉ F a)).card
          = ((Finset.range a).filter (fun r => r ∉ F a)).card := by
        congr 1
        exact Finset.filter_congr fun r hr => by
          rw [Nat.mod_eq_of_lt (Finset.mem_range.mp hr)]
      rw [hfil,
        card_filter_crt ha0 hP0 hcopr (fun n => n % a ∉ F a) (AvoidN s F)
          (fun x y h => by rw [h])
          (fun x y h => avoidN_congr hdvd h),
        hres, ih hs0 hscop F]

/-- Residues mod `g` outside a set `F ⊆ range g`: `g - |F|` of them. -/
theorem card_range_filter_not_mem {g : ℕ} {F : Finset ℕ} (hF : F ⊆ Finset.range g) :
    ((Finset.range g).filter (fun r => r ∉ F)).card = g - F.card := by
  rw [← Finset.sdiff_eq_filter, Finset.card_sdiff_of_subset hF, Finset.card_range]

/-- **K3 for the census sets.**  For fixed `S`, the residues mod the wheel
avoiding `E_g(S)` at every gear number `prod_g (g - |E_g(S)|)`. -/
theorem card_avoid_E {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (d : ℕ) (S : Finset ℕ) :
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => AvoidN G (E d S) n)).card
      = ∏ g ∈ G, (g - (E d S g).card) := by
  rw [card_avoid_prod G hG0 hcop (E d S)]
  exact Finset.prod_congr rfl fun g hg =>
    card_range_filter_not_mem (E_subset_range (hG0 g hg) d S)

/-- The `S = ∅` term is the pair correlation of round 34 (L44): both open
counts `prod_g (g - |CorrTeeth g d|)`. -/
theorem pair_corr_teeth {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (d : ℕ) :
    ((Finset.range (∏ g ∈ G, g)).filter
        (fun n => OpenN G n ∧ OpenN G (n + d))).card
      = ∏ g ∈ G, (g - (CorrTeeth g d).card) := by
  have hfil : (Finset.range (∏ g ∈ G, g)).filter (fun n => OpenN G n ∧ OpenN G (n + d))
      = (Finset.range (∏ g ∈ G, g)).filter (fun n => AvoidN G (E d ∅) n) :=
    Finset.filter_congr fun n _ => (avoidN_E_empty_iff hG0 d n).symm
  rw [hfil, card_avoid_E hG0 hcop d ∅]
  exact Finset.prod_congr rfl fun g _ => by rw [E_empty_eq_corrTeeth]

/-! ## 5. L22, THE GAP CENSUS LAW (register W22) -/

/-- **L22 / W22, the gap census law.**  For positive pairwise coprime gears,

    N_d(G) = sum over S ⊆ [1, d-1] of (-1)^|S| prod_{g ∈ G} (g - |E_g(S)|) .

K1 (`consecOpenN_iff_avoid`, `avoidN_E_iff`) rewrites the census predicate as
"boundary residues avoided, every interior position struck"; K2
(`card_filter_forall_not`) turns "every interior position struck" into the
alternating sum over subsets `S` of "every position of `S` unstruck"; K3
(`card_avoid_E`) counts each term as a CRT product. -/
theorem gap_census {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (d : ℕ) :
    (Ncount G d : ℤ)
      = ∑ S ∈ (Finset.Ioo 0 d).powerset,
          (-1 : ℤ) ^ S.card * ((∏ g ∈ G, (g - (E d S g).card) : ℕ) : ℤ) := by
  unfold Ncount
  have h1 : (Finset.range (∏ g ∈ G, g)).filter (ConsecOpenN G d)
      = ((Finset.range (∏ g ∈ G, g)).filter (fun n => AvoidN G (E d ∅) n)).filter
          (fun n => ∀ j ∈ Finset.Ioo 0 d, ¬ OpenN G (n + j)) := by
    rw [Finset.filter_filter]
    exact Finset.filter_congr fun n _ => consecOpenN_iff_avoid hG0 d n
  rw [h1, card_filter_forall_not]
  refine Finset.sum_congr rfl fun S _ => ?_
  rw [Finset.filter_filter]
  have h2 : (Finset.range (∏ g ∈ G, g)).filter
      (fun n => AvoidN G (E d ∅) n ∧ ∀ j ∈ S, OpenN G (n + j))
      = (Finset.range (∏ g ∈ G, g)).filter (fun n => AvoidN G (E d S) n) :=
    Finset.filter_congr fun n _ => by
      rw [avoidN_E_empty_iff hG0, avoidN_E_iff hG0, and_assoc]
  rw [h2, card_avoid_E hG0 hcop d S]

/-- L22 with the product taken in `ℤ`: `prod_g ((g : ℤ) - |E_g(S)|)`. -/
theorem gap_census_int {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (d : ℕ) :
    (Ncount G d : ℤ)
      = ∑ S ∈ (Finset.Ioo 0 d).powerset,
          (-1 : ℤ) ^ S.card * ∏ g ∈ G, ((g : ℤ) - ((E d S g).card : ℤ)) := by
  rw [gap_census hG0 hcop d]
  refine Finset.sum_congr rfl fun S _ => ?_
  congr 1
  rw [Nat.cast_prod]
  exact Finset.prod_congr rfl fun g hg => by rw [Nat.cast_sub (E_card_le (hG0 g hg) d S)]

/-- The census on the raw line, `ℤ` form of the predicate, is the same number. -/
theorem gap_census_raw {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (d : ℕ) :
    (((Finset.range (∏ g ∈ G, g)).filter (fun n : ℕ => ConsecOpenN G d n)).card : ℤ)
      = ∑ S ∈ (Finset.Ioo 0 d).powerset,
          (-1 : ℤ) ^ S.card * ((∏ g ∈ G, (g - (E d S g).card) : ℕ) : ℤ) :=
  gap_census hG0 hcop d

/-! ## 6. The `d = 4` check: the alternating sum is identically zero -/

/-- At `d = 4` the interior position `2` adds the offsets `2` and `4`, already
among the boundary offsets `{0, 2, 4, 6}`: adding `2` to `S` changes nothing. -/
theorem offs_four_insert_two (S : Finset ℕ) : Offs 4 (insert 2 S) = Offs 4 S := by
  ext x
  rw [mem_Offs, mem_Offs]
  simp only [Finset.mem_insert]
  constructor
  · rintro (h | h | h | h | (h | h) | ⟨j, (rfl | hj), rfl⟩)
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (Or.inl h))
    · exact Or.inr (Or.inr (Or.inr (Or.inl h)))
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h))))
    · exact Or.inr (Or.inr (Or.inl rfl))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨j, hj, rfl⟩))))
  · rintro (h | h | h | h | h | ⟨j, hj, rfl⟩)
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (Or.inl h))
    · exact Or.inr (Or.inr (Or.inr (Or.inl h)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (Or.inr h)))))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨j, Or.inr hj, rfl⟩))))

theorem E_four_insert_two (S : Finset ℕ) (g : ℕ) : E 4 (insert 2 S) g = E 4 S g := by
  unfold E; rw [offs_four_insert_two]

/-- **L4 from the census formula.**  At `d = 4` the terms pair off, `S` against
`S ∪ {2}`, with equal products and opposite signs, so the alternating sum is
identically zero: `N_4(G) = 0` for every positive pairwise coprime gear set.
(`no_gap_four` is the same fact by the direct argument; this derives it from
the algebra of L22.) -/
theorem gap_four_zero {G : Finset ℕ} (hG0 : ∀ g ∈ G, 0 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) : Ncount G 4 = 0 := by
  have h := gap_census hG0 hcop 4
  have hIoo : Finset.Ioo 0 4 = insert 2 ({1, 3} : Finset ℕ) := by
    ext x
    simp only [Finset.mem_Ioo, Finset.mem_insert, Finset.mem_singleton]
    omega
  have h2 : (2 : ℕ) ∉ ({1, 3} : Finset ℕ) := by simp
  rw [hIoo, Finset.sum_powerset_insert h2, ← Finset.sum_add_distrib] at h
  have hz : ∀ S ∈ ({1, 3} : Finset ℕ).powerset,
      (-1 : ℤ) ^ S.card * ((∏ g ∈ G, (g - (E 4 S g).card) : ℕ) : ℤ)
        + (-1 : ℤ) ^ (insert 2 S).card
            * ((∏ g ∈ G, (g - (E 4 (insert 2 S) g).card) : ℕ) : ℤ) = 0 := by
    intro S hS
    have h2S : 2 ∉ S := fun hm => h2 (Finset.mem_powerset.mp hS hm)
    simp only [E_four_insert_two, Finset.card_insert_of_notMem h2S, pow_succ]
    ring
  rw [Finset.sum_eq_zero hz] at h
  exact_mod_cast h

end TopMachine
