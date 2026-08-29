/-
THE MIRROR, IN THE KERNEL (round 26) - Lateral's round-25/26 parity laws,
the two halves that are pure arithmetic.

Lateral's lane produced two elementary facts about the involution
`k |-> -k` on slots, and offered them as this lane's cheapest kernel targets:

  * THE OPENING SET IS CLOSED UNDER THE MIRROR.  Slot `k` carries the pair
    `(6k - 1, 6k + 1)` and slot `P - k` carries `(6P - (6k+1), 6P - (6k-1))`,
    so with `q | P` the mirror EXCHANGES the two members' divisibility
    conditions - and a gear blocks a slot iff it divides EITHER member, a
    condition symmetric in the two.  `mirror_gear`, one gear, any period.

  * `g_1* = 1`: THE ANTIPODAL SLOT IS OPEN AT EVERY MACHINE (Lateral round
    26).  Their argument is `6 * (P+1)/2 = 3 (mod q)` against teeth at
    `6u = +-1`; the arithmetic is even shorter without residues -
    `6 * ((P+1)/2) = 3P + 3`, so the antipodal slot's members are
    `3P + 2` and `3P + 4`, and a gear dividing either would divide `2` or
    `4`.  `antipode_open`, one gear, any period, NO CASE ANALYSIS AND NO
    SCAN - it is the same five lines at every machine, for ever.

  * THE SELF-MIRROR WINDOW IS UNIQUE.  A depth-`j` window at index `t` is
    self-mirror iff `2t + j = 0 (mod N)`, and `N = prod (q-2)` is ODD, so
    there is AT MOST ONE such `t` per depth (`self_mirror_unique`).  That is
    the half the route consumes: "at most one exceptional window", which
    turns a first-moment bound of FEWER THAN TWO into a proof of NONE.

WHAT IS NOT HERE, stated because the corollary is the point of the lever:
"every configuration occurs an EVEN number of times except the self-mirror
one" needs a counting step - a fixed-point-free involution on a `Finset` has
even cardinality - which this file does not build.  What is proved here is
the involution and the uniqueness of its fixed point, not the parity of the
counts.
-/

import Machine29
import Machine11

namespace Mirror

/-! ## 1. The mirror exchanges the two members of a slot -/

/-- **The mirror, one gear.**  With `q | P` and `1 <= k < P`, gear `q`
divides the lower member of the mirrored slot exactly when it divides the
UPPER member of the original, and vice versa.  A gear blocks a slot iff it
divides either member, so blocking - hence opening - is mirror-invariant. -/
theorem mirror_gear {q P k : ℕ} (hqP : q ∣ P) (hk1 : 1 ≤ k) (hk2 : k < P) :
    ((q ∣ Census.lo (P - k)) ↔ (q ∣ Census.hi k)) ∧
      ((q ∣ Census.hi (P - k)) ↔ (q ∣ Census.lo k)) := by
  obtain ⟨c, rfl⟩ := hqP
  have h6 : q * (6 * c) = 6 * (q * c) := by
    rw [← Nat.mul_assoc, ← Nat.mul_assoc, Nat.mul_comm q 6]
  have hlo : Census.lo (q * c - k) + Census.hi k = q * (6 * c) := by
    simp only [Census.lo, Census.hi]; omega
  have hhi : Census.hi (q * c - k) + Census.lo k = q * (6 * c) := by
    simp only [Census.lo, Census.hi]; omega
  have hd : q ∣ q * (6 * c) := ⟨6 * c, rfl⟩
  constructor
  · constructor
    · intro h
      have h2 : q ∣ Census.lo (q * c - k) + Census.hi k := by rw [hlo]; exact hd
      exact (Nat.dvd_add_right h).mp h2
    · intro h
      have h2 : q ∣ Census.hi k + Census.lo (q * c - k) := by
        rw [Nat.add_comm, hlo]; exact hd
      exact (Nat.dvd_add_right h).mp h2
  · constructor
    · intro h
      have h2 : q ∣ Census.hi (q * c - k) + Census.lo k := by rw [hhi]; exact hd
      exact (Nat.dvd_add_right h).mp h2
    · intro h
      have h2 : q ∣ Census.lo k + Census.hi (q * c - k) := by
        rw [Nat.add_comm, hhi]; exact hd
      exact (Nat.dvd_add_right h).mp h2

/-- **The opening set of machine 11 is closed under the mirror.** -/
theorem mirror_exposed11 {k : ℕ} (hk1 : 1 ≤ k) (hk2 : k < 385) :
    Machine11.Exposed11 (385 - k) ↔ Machine11.Exposed11 k := by
  obtain ⟨m5l, m5h⟩ := mirror_gear (q := 5) (P := 385) ⟨77, rfl⟩ hk1 hk2
  obtain ⟨m7l, m7h⟩ := mirror_gear (q := 7) (P := 385) ⟨55, rfl⟩ hk1 hk2
  obtain ⟨m11l, m11h⟩ := mirror_gear (q := 11) (P := 385) ⟨35, rfl⟩ hk1 hk2
  unfold Machine11.Exposed11
  rw [m5l, m5h, m7l, m7h, m11l, m11h]
  constructor
  · rintro ⟨a, b, c, d, e, f⟩; exact ⟨b, a, d, c, f, e⟩
  · rintro ⟨a, b, c, d, e, f⟩; exact ⟨b, a, d, c, f, e⟩

/-- **The opening set of machine 29 is closed under the mirror.**  The same
lemma at eight gears; the period is `1,078,282,205`. -/
theorem mirror_exposed29 {k : ℕ} (hk1 : 1 ≤ k) (hk2 : k < 1078282205) :
    Machine29.Exposed29 (1078282205 - k) ↔ Machine29.Exposed29 k := by
  obtain ⟨a1, a2⟩ := mirror_gear (q := 5) (P := 1078282205) ⟨215656441, rfl⟩ hk1 hk2
  obtain ⟨b1, b2⟩ := mirror_gear (q := 7) (P := 1078282205) ⟨154040315, rfl⟩ hk1 hk2
  obtain ⟨c1, c2⟩ := mirror_gear (q := 11) (P := 1078282205) ⟨98025655, rfl⟩ hk1 hk2
  obtain ⟨d1, d2⟩ := mirror_gear (q := 13) (P := 1078282205) ⟨82944785, rfl⟩ hk1 hk2
  obtain ⟨e1, e2⟩ := mirror_gear (q := 17) (P := 1078282205) ⟨63428365, rfl⟩ hk1 hk2
  obtain ⟨f1, f2⟩ := mirror_gear (q := 19) (P := 1078282205) ⟨56751695, rfl⟩ hk1 hk2
  obtain ⟨g1, g2⟩ := mirror_gear (q := 23) (P := 1078282205) ⟨46881835, rfl⟩ hk1 hk2
  obtain ⟨h1, h2⟩ := mirror_gear (q := 29) (P := 1078282205) ⟨37182145, rfl⟩ hk1 hk2
  unfold Machine29.Exposed29 Machine23.Exposed23 Machine19.Exposed19
  rw [a1, a2, b1, b2, c1, c2, d1, d2, e1, e2, f1, f2, g1, g2, h1, h2]
  constructor
  · rintro ⟨⟨⟨p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12⟩, q1, q2⟩, r1, r2⟩
    exact ⟨⟨⟨p2, p1, p4, p3, p6, p5, p8, p7, p10, p9, p12, p11⟩, q2, q1⟩, r2, r1⟩
  · rintro ⟨⟨⟨p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12⟩, q1, q2⟩, r1, r2⟩
    exact ⟨⟨⟨p2, p1, p4, p3, p6, p5, p8, p7, p10, p9, p12, p11⟩, q2, q1⟩, r2, r1⟩

/-! ## 2. `g_1* = 1`: the antipodal slot is open at every machine -/

/-- **THE ANTIPODAL SLOT IS OPEN** (Lateral's `g_1* = 1`).  If `2s = P + 1`
and `q | P` with `q >= 5`, then the antipodal slot's members are `3P + 2` and
`3P + 4`, so gear `q` divides neither: it would have to divide `2` or `4`.
No residues, no case analysis, no machine. -/
theorem antipode_open {q P s : ℕ} (hq : 5 ≤ q) (hqP : q ∣ P) (hs : 2 * s = P + 1) :
    ¬ (q ∣ Census.lo s) ∧ ¬ (q ∣ Census.hi s) := by
  obtain ⟨c, rfl⟩ := hqP
  have h3 : q * (3 * c) = 3 * (q * c) := by
    rw [← Nat.mul_assoc, ← Nat.mul_assoc, Nat.mul_comm q 3]
  have hlo : Census.lo s = q * (3 * c) + 2 := by simp only [Census.lo]; omega
  have hhi : Census.hi s = q * (3 * c) + 4 := by simp only [Census.hi]; omega
  have key : ∀ r, 0 < r → r < q → ¬ (q ∣ q * (3 * c) + r) := by
    intro r hr0 hrq h
    obtain ⟨d, hd⟩ := h
    have hlt : 3 * c < d := by
      by_contra hcon
      have hle : d ≤ 3 * c := by omega
      have := Nat.mul_le_mul (Nat.le_refl q) hle
      omega
    have hstep : 3 * c + 1 ≤ d := by omega
    have h2 := Nat.mul_le_mul (Nat.le_refl q) hstep
    have hexp : q * (3 * c + 1) = q * (3 * c) + q := by
      rw [Nat.mul_add, Nat.mul_one]
    omega
  exact ⟨by rw [hlo]; exact key 2 (by omega) (by omega),
    by rw [hhi]; exact key 4 (by omega) (by omega)⟩

/-- The antipodal slot of machine 11 (`193`, since `2 * 193 = 386`) is an
opening - from `antipode_open` at each gear, with no computation. -/
theorem antipode_exposed11 : Machine11.Exposed11 193 := by
  obtain ⟨a1, a2⟩ := antipode_open (q := 5) (P := 385) (s := 193)
    (by omega) ⟨77, rfl⟩ (by omega)
  obtain ⟨b1, b2⟩ := antipode_open (q := 7) (P := 385) (s := 193)
    (by omega) ⟨55, rfl⟩ (by omega)
  obtain ⟨c1, c2⟩ := antipode_open (q := 11) (P := 385) (s := 193)
    (by omega) ⟨35, rfl⟩ (by omega)
  exact ⟨a1, a2, b1, b2, c1, c2⟩

/-- The antipodal slot of machine 29 (`539,141,103`) is an opening.  This is
the fact that makes `W_1(g)` even for every `g >= 2` - the maximal gap never
occurs exactly once. -/
theorem antipode_exposed29 : Machine29.Exposed29 539141103 := by
  obtain ⟨a1, a2⟩ := antipode_open (q := 5) (P := 1078282205) (s := 539141103)
    (by omega) ⟨215656441, rfl⟩ (by omega)
  obtain ⟨b1, b2⟩ := antipode_open (q := 7) (P := 1078282205) (s := 539141103)
    (by omega) ⟨154040315, rfl⟩ (by omega)
  obtain ⟨c1, c2⟩ := antipode_open (q := 11) (P := 1078282205) (s := 539141103)
    (by omega) ⟨98025655, rfl⟩ (by omega)
  obtain ⟨d1, d2⟩ := antipode_open (q := 13) (P := 1078282205) (s := 539141103)
    (by omega) ⟨82944785, rfl⟩ (by omega)
  obtain ⟨e1, e2⟩ := antipode_open (q := 17) (P := 1078282205) (s := 539141103)
    (by omega) ⟨63428365, rfl⟩ (by omega)
  obtain ⟨f1, f2⟩ := antipode_open (q := 19) (P := 1078282205) (s := 539141103)
    (by omega) ⟨56751695, rfl⟩ (by omega)
  obtain ⟨g1, g2⟩ := antipode_open (q := 23) (P := 1078282205) (s := 539141103)
    (by omega) ⟨46881835, rfl⟩ (by omega)
  obtain ⟨h1, h2⟩ := antipode_open (q := 29) (P := 1078282205) (s := 539141103)
    (by omega) ⟨37182145, rfl⟩ (by omega)
  exact ⟨⟨⟨a1, a2, b1, b2, c1, c2, d1, d2, e1, e2, f1, f2⟩, g1, g2⟩, h1, h2⟩

/-! ## 3. At most one self-mirror window per depth -/

/-- **THE SELF-MIRROR WINDOW IS UNIQUE.**  A depth-`j` window at index `t` is
its own mirror image iff `2t + j = 0 (mod N)`; `N = prod (q - 2)` is ODD, so
two solutions below `N` coincide.  This is the half of Lateral's parity law
the live route consumes: at most ONE exceptional window per depth, so a
counting bound of "fewer than two" proves "none". -/
theorem self_mirror_unique {N j t1 t2 : ℕ} (hN : N % 2 = 1)
    (h1 : t1 < N) (h2 : t2 < N)
    (e1 : (2 * t1 + j) % N = 0) (e2 : (2 * t2 + j) % N = 0) : t1 = t2 := by
  -- both are multiples of `N`, so `N` divides twice their difference
  have hd1 : N ∣ 2 * t1 + j := Nat.dvd_of_mod_eq_zero e1
  have hd2 : N ∣ 2 * t2 + j := Nat.dvd_of_mod_eq_zero e2
  have hsym : ∀ a b : ℕ, b ≤ a → a < N → b < N →
      N ∣ 2 * a + j → N ∣ 2 * b + j → a = b := by
    intro a b hba haN hbN hda hdb
    have hsub : N ∣ (2 * a + j) - (2 * b + j) := Nat.dvd_sub hda hdb
    have heq : (2 * a + j) - (2 * b + j) = 2 * (a - b) := by omega
    rw [heq] at hsub
    obtain ⟨c, hc⟩ := hsub
    have hclt : c < 2 := by
      by_contra hcon
      have h2c : 2 ≤ c := by omega
      have := Nat.mul_le_mul (Nat.le_refl N) h2c
      omega
    interval_cases c <;> omega
  rcases Nat.le_total t2 t1 with h | h
  · exact hsym t1 t2 h h1 h2 hd1 hd2
  · exact (hsym t2 t1 h h2 h1 hd2 hd1).symm

/-- `N = prod (q - 2)`, the openings per period, is ODD at every machine of
the ladder - the hypothesis `self_mirror_unique` needs.  `135, 1485, 22275,
378675, 7952175, 214708725, 6226553025` at machines 11, 13, 17, 19, 23, 29,
31; the last two are the gate-asserted gap counts of round 25. -/
theorem periods_odd :
    135 % 2 = 1 ∧ 1485 % 2 = 1 ∧ 22275 % 2 = 1 ∧ 378675 % 2 = 1 ∧
      7952175 % 2 = 1 ∧ 214708725 % 2 = 1 ∧ 6226553025 % 2 = 1 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num⟩

/-! ## 4. The counting half (round 27)

Round 26 left the lever's second half open and named it: "every configuration
occurs an EVEN number of times except the self-mirror one" needs a
fixed-point-free-involution counting lemma, which round 26 did not build.
It is built here, and it needs nothing from the machine. -/

/-- **A FIXED-POINT-FREE INVOLUTION PAIRS A FINSET UP.**  If `f` maps `s` into
itself, is an involution there and fixes nothing, then `s` has even
cardinality.  Structural induction: remove `a` and `f a` - two distinct
elements - and the hypotheses restrict to what is left, because `f x = f a`
forces `x = a` and `f x = a` forces `x = f a`. -/
theorem even_card_involution {α : Type*} [DecidableEq α] (f : α → α) :
    ∀ (n : ℕ) (s : Finset α), s.card ≤ n →
      (∀ a ∈ s, f a ∈ s) → (∀ a ∈ s, f (f a) = a) → (∀ a ∈ s, f a ≠ a) →
      s.card % 2 = 0 := by
  intro n
  induction n with
  | zero => intro s hle _ _ _; omega
  | succ m ih =>
      intro s hle hmap hinv hfix
      rcases Finset.eq_empty_or_nonempty s with rfl | hne
      · simp
      obtain ⟨a, ha⟩ := hne
      have hbs : f a ∈ s := hmap a ha
      have hab : f a ≠ a := hfix a ha
      have hb' : f a ∈ s.erase a := Finset.mem_erase.mpr ⟨hab, hbs⟩
      have hc1 : (s.erase a).card = s.card - 1 := Finset.card_erase_of_mem ha
      have hc2 : ((s.erase a).erase (f a)).card = (s.erase a).card - 1 :=
        Finset.card_erase_of_mem hb'
      have hpos : 1 ≤ (s.erase a).card := Finset.card_pos.mpr ⟨f a, hb'⟩
      have hspos : 1 ≤ s.card := Finset.card_pos.mpr ⟨a, ha⟩
      have hsub : ∀ x ∈ (s.erase a).erase (f a), x ∈ s := by
        intro x hx
        exact Finset.mem_of_mem_erase (Finset.mem_of_mem_erase hx)
      have hmap' : ∀ x ∈ (s.erase a).erase (f a),
          f x ∈ (s.erase a).erase (f a) := by
        intro x hx
        have hx1 : x ≠ f a := (Finset.mem_erase.mp hx).1
        have hx2 : x ∈ s.erase a := (Finset.mem_erase.mp hx).2
        have hx3 : x ≠ a := (Finset.mem_erase.mp hx2).1
        have hxs : x ∈ s := Finset.mem_of_mem_erase hx2
        refine Finset.mem_erase.mpr ⟨?_, Finset.mem_erase.mpr ⟨?_, hmap x hxs⟩⟩
        · intro hc
          apply hx3
          have h1 : f (f x) = x := hinv x hxs
          rw [hc, hinv a ha] at h1
          exact h1.symm
        · intro hc
          apply hx1
          have h1 : f (f x) = x := hinv x hxs
          rw [hc] at h1
          exact h1.symm
      have := ih ((s.erase a).erase (f a)) (by omega) hmap'
        (fun x hx => hinv x (hsub x hx)) (fun x hx => hfix x (hsub x hx))
      omega

/-! ## 5. The parity law in the form the route consumes

`m` is the mirror on window INDICES, `L` the window's length.  Nothing below
knows what a machine is: the only inputs are that `m` is an involution of
`range N` preserving `L`, and that its unique fixed point does not carry the
length being counted. -/

/-- **EVERY WINDOW LENGTH OCCURS AN EVEN NUMBER OF TIMES**, except possibly
the one carried by the self-mirror window.  This is the half round 26 named
and did not have; with `self_mirror_unique` it is the whole lever. -/
theorem window_count_even {N g : ℕ} (m L : ℕ → ℕ)
    (hlt : ∀ t, t < N → m t < N)
    (hmm : ∀ t, t < N → m (m t) = t)
    (hL : ∀ t, t < N → L (m t) = L t)
    (hg : ∀ t, t < N → m t = t → L t ≠ g) :
    (((Finset.range N).filter (fun t => L t = g)).card) % 2 = 0 := by
  set s := (Finset.range N).filter (fun t => L t = g) with hs
  have hmem : ∀ t, t ∈ s ↔ (t < N ∧ L t = g) := by
    intro t
    simp only [hs, Finset.mem_filter, Finset.mem_range]
  refine even_card_involution m s.card s (le_refl _) ?_ ?_ ?_
  · intro t ht
    obtain ⟨h1, h2⟩ := (hmem t).mp ht
    exact (hmem (m t)).mpr ⟨hlt t h1, by rw [hL t h1]; exact h2⟩
  · intro t ht
    exact hmm t ((hmem t).mp ht).1
  · intro t ht
    obtain ⟨h1, h2⟩ := (hmem t).mp ht
    intro hc
    exact hg t h1 hc h2

/-- **THE ENDPOINT LEVER.**  With the self-mirror window located (round 26's
`self_mirror_unique` supplies its uniqueness, and Lateral's address formula
its value), any length the exceptional window does NOT carry occurs an even
number of times - so a counting bound of "at most one" proves "none".  In
particular an ADJACENT EQUAL PAIR `(F, F)` - length `2F` at depth 2 - can
never occur exactly once. -/
theorem adjacent_equal_even {N F t0 : ℕ} (m L : ℕ → ℕ)
    (hlt : ∀ t, t < N → m t < N)
    (hmm : ∀ t, t < N → m (m t) = t)
    (hL : ∀ t, t < N → L (m t) = L t)
    (ht0 : t0 < N) (hf0 : m t0 = t0)
    (huniq : ∀ t, t < N → m t = t → t = t0)
    (hexc : L t0 ≠ 2 * F) :
    (((Finset.range N).filter (fun t => L t = 2 * F)).card) % 2 = 0 :=
  window_count_even m L hlt hmm hL
    (fun t ht hfix => by rw [huniq t ht hfix]; exact hexc)

/-- The same statement with the conclusion the route quotes: a configuration
whose count is at most one, and which the exceptional window does not carry,
does not occur at all. -/
theorem none_of_at_most_one {N F t0 : ℕ} (m L : ℕ → ℕ)
    (hlt : ∀ t, t < N → m t < N)
    (hmm : ∀ t, t < N → m (m t) = t)
    (hL : ∀ t, t < N → L (m t) = L t)
    (ht0 : t0 < N) (hf0 : m t0 = t0)
    (huniq : ∀ t, t < N → m t = t → t = t0)
    (hexc : L t0 ≠ 2 * F)
    (hone : (((Finset.range N).filter (fun t => L t = 2 * F)).card) ≤ 1) :
    (((Finset.range N).filter (fun t => L t = 2 * F)).card) = 0 := by
  have h := adjacent_equal_even m L hlt hmm hL ht0 hf0 huniq hexc
  omega

end Mirror
