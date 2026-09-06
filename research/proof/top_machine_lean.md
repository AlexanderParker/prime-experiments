# The top machine in the kernel (Formalist, round 32)

Lean 4 / mathlib formalisation of `research/proof/top_machine_1.md` section 4,
on the raw line, in the top machine's own pair coordinate.

Files: `proofs/TopMachine.lean` (core, no project dependency),
`proofs/TopMachineWheel.lean` (wheel count + conjugacy, imports `Census`).
Registered in `proofs/lakefile.toml` as `lean_lib`s and in `defaultTargets`;
audited from `proofs/AxiomCheck.lean`.

**Zero sorries. No `native_decide`, no `Lean.ofReduceBool`, no `decide` at
all** - every theorem below is an ordinary proof, so nothing depends on a
gear set being small enough to enumerate.

## The definitions

```lean
def Strikes (g : ℕ) (n : ℤ) : Prop := (g : ℤ) ∣ n ∨ (g : ℤ) ∣ (n + 2)
def IsOpen (G : Finset ℕ) (n : ℤ) : Prop := ∀ g ∈ G, ¬ Strikes g n
def StrikesR (g r : ℕ) : Prop := r % g = 0 ∨ (r + 2) % g = 0     -- residue form
def OpenN (G : Finset ℕ) (n : ℕ) : Prop := ∀ g ∈ G, ¬ StrikesR g n
def ColOpen (G : Finset ℕ) (k : ℤ) : Prop :=                      -- column coordinate
  ∀ g ∈ G, ¬ ((g : ℤ) ∣ 6 * k - 1 ∨ (g : ℤ) ∣ 6 * k + 1)
```

`G` is an arbitrary `Finset ℕ`. **No theorem assumes primality**; each carries
the exact hypothesis it needs (`3 ≤ g`, `5 ≤ g`, `g % 2 = 1`, pairwise
coprime). The owner's construction (primes above `q`, so `g ≥ 7`) satisfies
all of them.

## Verification before formalising

Every statement was reproduced first on a brute-force model of the machine
(scratch script, mirroring `research/topmachine/r1/wheel.py`, `pairwise.py`,
`ladder.py`, `validate.py`), with **zero exceptions**:

- arcs and count `g - 2` for every gear 5..37;
- partner law over 6 gears, whole periods; no gap 4 in 8 wheels;
- wheel counts 495, 1485, 2805, 5355, 7425, 25245, 135, 126225 - all
  `= prod (g - 2)`;
- shield / `n = 2` / `n = -4` / clump: 8 wheels;
- mirror: 0 mismatches; the 8 affine maps at `W = 1001` are exactly
  `n ↦ c(n+1) - 1` with `c = ±1` mod each gear;
- longest run `= q' - 3` and longest step-2 chain `= q' - 2` in 7 wheels;
- chain law and merge law: 0 exceptions;
- parity law `F = 2m - (m mod 2)`: 16 scanned wheels (`m = 2,3,4`), matching
  `ladder.py`'s convention **record = number of consecutive struck pairs**
  (confirmed against `validate.py: F_scan`, cyclic over the whole period);
- conjugacy `n ↦ 6^{-1}(n+1)`: 0 mismatches.

## The ledger

| law | Lean name (namespace `TopMachine`) | status | hypothesis |
|---|---|---|---|
| L1 teeth | `strikesR_iff` | proved | `3 ≤ g` |
| L1 count `g-2` | `card_open_residues` | proved | `3 ≤ g` |
| L2 arcs `(g-3, 1)` | `open_residues` | proved | `3 ≤ g` |
| L2 shield `n = -1` | `not_strikes_neg_one` | proved | `2 ≤ g` |
| L3 partner law | `partner` | proved | none |
| L3 dominoes `{x, x+2}` | `strikes_iff_domino` | proved | none |
| L4 forbidden gap 4 | `open_of_open_add_four`, `no_gap_four` | proved | none |
| L5 wheel count `prod (g-2)` | `wheel_count` | proved | gears `≥ 3`, pairwise coprime |
| L5 CRT engine | `card_filter_crt` | proved | `a, b` coprime, residue-invariant predicates |
| L6 shield open | `shield_open` | proved | gears `≥ 2` |
| L6 antipode `n = 2`, `n = -4` | `two_open`, `neg_four_open` | proved | gears `≥ 5` |
| L6 origin clump | `clump_open`, `origin_clump` | proved | gears `≥ q' ≥ 3` |
| L7 mirror | `strikes_mirror`, `open_mirror` | proved | none |
| L7 unique fixed point | `mirror_fixed_iff` | proved | none |
| L8 sufficiency | `strikes_affine`, `open_affine` | proved | `c = ±1` mod each gear |
| L8 necessity, one gear | `affine_teeth` | proved | `g` odd, `g ∤ c` |
| L8 adjacency | `affine_step` (+ `affine_one`, `affine_neg_one`) | proved | none |
| L8 group is exactly `(Z/2)^m` | - | R32: **will not close**; **closed in R33**, see the R33 section | - |
| L10 run `< q' - 2` | `no_long_run`, `run_lt` | proved | `5 ≤ q'`, `q' ∈ G` |
| L10 run `q' - 3` attained | `run_attained` | proved | gears `≥ q' ≥ 3` |
| L10 chain `< q' - 1` | `no_long_chain2`, `chain2_lt` | proved | `q'` odd, `3 ≤ q'`, `q' ∈ G` |
| L10 chain `q' - 2` attained | `chain2_attained` | proved | `q'` odd, gears `≥ q' ≥ 3` |
| L12 chain law | `chain_law` | proved | none |
| L13 merge law | `merge_law` | proved | none |
| L17 parity, upper bound | `parity_core`, `parity_upper` | proved | gears odd and `> 2m + 1` |
| L17 parity, attainment | - | R32: **will not close**; **closed in R33**, see the R33 section | - |
| L19 conjugacy | `strikes_iff_col`, `conjugacy` | proved | `6k ≡ n+1` mod each gear |
| L19 the column exists | `exists_column` | proved | `gcd(6, W) = 1`, gears `∣ W` |
| L19 against `Census.lo/hi` | `conjugacy_census` | proved | `1 ≤ k` |

Supporting lemmas, all proved: `strikes_natCast`, `open_natCast`,
`eq_zero_of_dvd_of_abs_lt`, `not_dvd_of_abs_lt`, `dvd_iff_of_dvd_sub`,
`dvd_of_dvd_two_mul`, `mod_add_two_eq_zero_iff`, `open_of_small`,
`window_pair`, `card_even_range`, `card_odd_range`, `strikesR_congr`,
`openN_congr`, `openN_insert`.

## The two honest "will not close in this round" verdicts

**L8, "the symmetry group is exactly `(Z/2)^m`".** The two halves that are
mathematics are in the kernel: sufficiency for every gear set
(`open_affine`), and per-gear necessity (`affine_teeth`: an affine map with
`g ∤ c` that preserves gear `g`'s struck set has `(c, b) ≡ (1, 0)` or
`(-1, -2)` mod `g` - the branch's one-line argument, that the map must
permute the tooth pair `{0, -2}`). What is missing is the *assembly*: from
"preserves the open set of `G`" to "preserves each gear's struck set"
(needs the CRT surjectivity of `ℤ_W → ∏ ℤ_g` to isolate one gear while the
others miss), and the count `2^m` (needs the same CRT bijection on the sign
vector). That is a `Finset`-indexed CRT existence lemma, not built here; the
2-modulus version `card_filter_crt` is, and is the natural base for it.
The adjacency half needs nothing: `affine_step` shows the map translates
consecutive pairs by exactly `c`, so adjacency survives iff `c = ±1` in
`ℤ_W`.

**L17, attainment (`F_top ≥ 2m - (m mod 2)`).** The upper bound - the
mathematical content of the branch's tiling argument - is closed
(`parity_upper`), and it is closed in the sharp form: `L + (m mod 2) ≤ 2m`,
so `2m` for even `m` and `2m - 1` for odd `m`, with the *parity* argument in
the kernel (each gear's strikes inside a window of length `≤ g - 2` lie in
one distance-2 domino, hence in ONE parity class; the even and odd positions
are therefore covered by disjoint gear pools of at most two positions each,
and for odd `m` the two ceilings `⌈m/2⌉` cannot both be paid out of `m`
gears). Attainment needs a *construction*: an explicit assignment of
`⌈m/2⌉ + ⌊m/2⌋` gears to the dominoes of `[0, 2m)` plus a `Finset`-indexed
CRT to realise the phase vector. Same missing lemma as L8. Not attempted
this round; the branch's evidence for attainment is the 170-case cover
search in `ladder.py`, not a proof.

## What did NOT transfer from the existing project files

`MergeLaw.lean` and `TwoTeeth.lean` are written for the bottom machine's
teeth `{u, q - u}` with letters `{2u, q - 2u}`. Setting `d = 2` there does
not specialise cleanly: `TwoTeeth.Kill q u x := x % q = u ∨ x % q = q - u`
has its teeth symmetric about `0`, whereas the top machine's teeth are `0`
and `-2` - an *offset* pair, not a symmetric one, so `kill_spacing`'s
`{2u, q - 2u}` becomes `{2, g - 2}` only after a shift of coordinate that
also moves `MergeLaw`'s `pos`/`kap` enumeration. The top machine's L12/L13
are two- and six-line proofs stated directly (`chain_law`, `merge_law`), so
the reuse would have cost more than it saved. `MergeLaw.newgap_le` remains
the right vehicle if a spectrum/qualifying-bound ladder is ever wanted on
the raw line; nothing here blocks that.

`Gear.lean` has no `open` predicate (it is the per-gear ledger line `R q S`
over `minFac`); the project's column-coordinate members are `Census.lo` and
`Census.hi`, which is what `conjugacy_census` is stated against.

## Build and audit

```
cd C:/dev/primes/proofs
~/.elan/bin/lake.exe build TopMachine TopMachineWheel
```

Result: **green**, 1001 jobs, no warnings, no errors.

| target | build time |
|---|---|
| `TopMachine` | 6.9 s (cold), 6.0 s (rebuild) |
| `TopMachineWheel` | 5.9 s (cold), 5.0 s (rebuild) |

Both are ordinary elaboration - no kernel scan, no `decide` - so peak memory
is a normal `lean.exe` (well under 1 GB); neither needs the babysitter.

Axiom audit over **all 59 declarations** of the two files (`lake env lean` on
a local `#print axioms` file, and the same block appended to
`proofs/AxiomCheck.lean` behind `import TopMachine` / `import
TopMachineWheel`):

- every declaration is `[propext, Classical.choice, Quot.sound]` or smaller;
- `Strikes`, `IsOpen`, `StrikesR`, `OpenN`, `ColOpen` depend on no axioms;
- `partner`, `strikes_iff_domino`, `strikesR_congr` depend on `[propext]`
  only; `affine_teeth`, `chain_law`, `strikes_mirror`, `strikesR_iff` on
  `[propext, Quot.sound]`;
- **no `sorryAx`, no `Lean.ofReduceBool`, no `Lean.trustCompiler`.**

Only `parity_core` uses choice essentially (`choose` picks a striker per
position); everything else inherits `Classical.choice` from mathlib's
`Finset`/`omega` plumbing.

---

# Round 33: the Finset-indexed CRT, and the two laws it closes

New file: `proofs/TopMachineCrt.lean` (imports `TopMachineWheel`), registered as
a `lean_lib` in `proofs/lakefile.toml`, in `defaultTargets`, and audited from
`proofs/AxiomCheck.lean`.  **29 new declarations, zero sorries, no
`native_decide`, no `decide`, no `Lean.ofReduceBool`.**

## The lemma that was missing

```lean
theorem exists_crt : ∀ (G : Finset ℕ),
    (∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) → ∀ (r : ℕ → ℤ),
    ∃ n : ℤ, ∀ g ∈ G, (g : ℤ) ∣ n - r g

theorem crt_unique : ∀ (G : Finset ℕ),
    (∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) → ∀ n n' : ℤ,
    (∀ g ∈ G, (g : ℤ) ∣ n - n') → ((∏ g ∈ G, g : ℕ) : ℤ) ∣ n - n'
```

Existence by induction on the `Finset`: the new modulus is coprime to the
product of the old ones, so the old solution is corrected by a multiple of that
product; the correction factor is Bezout (`exists_inv_of_coprime`, from
`Nat.gcd_eq_gcd_ab`).  Uniqueness by the same induction with
`IsCoprime.mul_dvd`.  Mathlib's `ZMod.chineseRemainder` and
`Nat.chineseRemainderOfList` were not used: the first is a ring equivalence that
would have to be transported back to `ℤ`-divisibility at every use, the second
is a list, not a `Finset`, statement.  The direct induction is twenty lines and
states exactly what the two laws need.

Supporting lemma - the per-gear choice that makes CRT useful for L8:

```lean
theorem exists_avoiding (g : ℕ) (V : Finset ℤ) (hV : V.card < g) :
    ∃ x : ℤ, ∀ v ∈ V, ¬ (g : ℤ) ∣ x - v
```

(pigeonhole of `Finset.range g` against the image of `V` in residues).

## L17, attainment - and the parity law as an EQUALITY

The construction, checked against `research/topmachine/r1/cover.py`'s pool
pieces `{x, x + 2}` and its `minpieces` count (`ceil(len/2)` per parity chain)
before formalising: give gear `j` the domino anchor

```lean
def anchor (m j : ℕ) : ℕ :=
  if j < (m + 1) / 2 then 4 * j else 4 * (j - (m + 1) / 2) + 1
```

- the first `ceil(m/2)` gears tile the EVEN positions of `[0, L)` with the
  dominoes `{0,2}, {4,6}, ...`;
- the remaining `floor(m/2)` gears tile the ODD positions with `{1,3}, {5,7},
  ...`;
- `anchor_covers` proves the tiling covers `[0, 2m - (m mod 2))` exactly (`omega`
  after the case split on the parity of the position).

Setting each gear's residue by CRT to `-(anchor + 2)` puts its two teeth on its
own domino:

```lean
theorem parity_attained {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    ∃ n : ℤ, ∀ i : ℕ, i < 2 * G.card - G.card % 2 → ¬ IsOpen G (n + (i : ℤ))

theorem parity_law {G : Finset ℕ} (hodd : ∀ g ∈ G, g % 2 = 1)
    (hbig : ∀ g ∈ G, 2 * G.card + 1 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    IsGreatest {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))}
      (2 * G.card - G.card % 2)
```

`parity_law` is L17 in full: the longest run of consecutive struck pairs is
EXACTLY `2m - (m mod 2)`.  Upper bound `parity_upper` (round 32), lower bound
`parity_attained`.  The lower bound needs NO size and NO oddness hypothesis - a
gear always strikes both ends of the domino its residue names - so oddness and
`g > 2m + 1` are used only by the upper bound.

Pre-formalisation check of the construction (scratch script: CRT solve, then
brute-force strike test): all 389 gear sets of `m = 1..11` consecutive primes
from `[5, 200)` with `q' > 2m + 1`; every position of `[0, L)` struck in every
case, **0 failures**, with the covering claim `anchor_covers` checked
independently.

## L8, the symmetry group is exactly `(Z/2)^m`

Four parts, together the whole statement.

**Necessity** (the assembly that was missing), in three steps:

```lean
theorem symm_not_dvd_mul {G : Finset ℕ} (h2 : ∀ g ∈ G, 2 ≤ g) (hcop : ...)
    {c b : ℤ} (hpres : ∀ n : ℤ, IsOpen G (c * n + b) ↔ IsOpen G n)
    {g : ℕ} (hg : g ∈ G) : ¬ (g : ℤ) ∣ c

theorem isolate ... (h5 : ∀ g ∈ G, 5 ≤ g) (hcop) (hunit) (hg : g ∈ G) (a : ℤ) :
    ∃ N : ℤ, (g : ℤ) ∣ N - a ∧
      ∀ h ∈ G, h ≠ g → ¬ Strikes h N ∧ ¬ Strikes h (c * N + b)

theorem affine_gear ... : Strikes g (c * n + b) ↔ Strikes g n
```

- `symm_not_dvd_mul`: if a gear divided `c`, then `n` and `n + tP` (`P` the
  product of the OTHER gears) would have images congruent modulo EVERY gear, so
  openness would be invariant under `n ↦ n + tP`; but `P` is invertible mod `g`,
  so some `t` slides the always-open shield `n = -1` onto `g`'s own tooth.  No
  primality, no size beyond `g ≥ 2`.
- `isolate`: the CRT with an avoidance choice.  At each other gear only four
  residues are forbidden (`0` and `-2` for `n`; their two preimages under
  `x ↦ c x + b` for the image) and the gear has at least five, so
  `exists_avoiding` supplies a residue; CRT assembles them together with the
  demanded class at `g`.
- `affine_gear`: with every other gear missing both `n` and its image, "open" IS
  "`g` does not strike", so preserving the open set is preserving `g`'s struck
  set.  `affine_teeth` (round 32) then finishes, per gear.

```lean
theorem affine_group_of_unit (hodd) (h5) (hcop)
    (hunit : ∀ g ∈ G, ∃ c' : ℤ, (g : ℤ) ∣ c * c' - 1)
    (hpres : ∀ n : ℤ, IsOpen G (c * n + b) ↔ IsOpen G n) :
    ∀ g ∈ G, ((g : ℤ) ∣ c - 1 ∧ (g : ℤ) ∣ b) ∨ ((g : ℤ) ∣ c + 1 ∧ (g : ℤ) ∣ b + 2)

theorem affine_group (hp : ∀ g ∈ G, Nat.Prime g) (h5 : ∀ g ∈ G, 5 ≤ g)
    (hpres : ∀ n : ℤ, IsOpen G (c * n + b) ↔ IsOpen G n) :
    ∀ g ∈ G, ((g : ℤ) ∣ c - 1 ∧ (g : ℤ) ∣ b) ∨ ((g : ℤ) ∣ c + 1 ∧ (g : ℤ) ∣ b + 2)

theorem affine_group_form (hp) (h5) (hpres) :
    (∀ g ∈ G, (g : ℤ) ∣ c - 1 ∨ (g : ℤ) ∣ c + 1) ∧ (∀ g ∈ G, (g : ℤ) ∣ b - (c - 1))
```

`affine_group_form` is the branch's own wording: every symmetry is
`n ↦ c(n + 1) - 1` (that is, `b = c - 1` mod every gear) with `c = ±1` modulo
every gear.

**Sufficiency**: `open_affine` (round 32).  **Realisability** - every sign
vector occurs, by CRT:

```lean
theorem exists_symmetry (hcop) (ε : ℕ → ℤ) (hε : ∀ g ∈ G, ε g = 1 ∨ ε g = -1) :
    ∃ c : ℤ, (∀ g ∈ G, (g : ℤ) ∣ c - ε g) ∧
      ∀ n : ℤ, IsOpen G (c * (n + 1) - 1) ↔ IsOpen G n
```

**The count** `2 ^ m`, by the same CRT counting engine `card_filter_crt` that
gives `wheel_count`, run on the sign predicate (`SignR g n := n % g = 1 ∨
(n + 1) % g = 0`; two residues per gear, `1` and `g - 1`):

```lean
theorem sign_count : ∀ (G : Finset ℕ), (∀ g ∈ G, 3 ≤ g) → (hcop) →
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => SignsN G n)).card = 2 ^ G.card
```

with `signR_iff_dvd` the bridge from the residue form to
`(g:ℤ) ∣ c - 1 ∨ (g:ℤ) ∣ c + 1`.

## The one hypothesis that is not derived: invertibility at a COMPOSITE gear

`affine_group_of_unit` carries `hunit`: `c` is invertible modulo each gear -
the affine map is a bijection of `ℤ_W`, which is what "symmetry" means.  That is
NOT derived in this generality.  What IS derived, for arbitrary pairwise coprime
gears, is the weaker `symm_not_dvd_mul` (no gear divides `c`).  For a PRIME gear
the two are the same statement, which is why `affine_group` needs no `hunit`;
for a composite gear with `1 < gcd(c, g) < g` the shift argument of
`symm_not_dvd_mul` yields no contradiction and the counting behind
`exists_avoiding` would need `2 + 2 gcd(c, g) < g`.  The owner's gears are the
primes above `q`, so `affine_group` covers the construction, and
`affine_group_of_unit` is the general statement.  This is the ONLY place in the
three files where primality is used at all.

Brute-force check of the whole L8 package before formalising: gears
`{5, 7, 11}`, `W = 385`, all `385 x 385` affine maps of `ℤ_W` - exactly
**8 = 2^3** preserve the open set, every one of them with `c` a unit, `c = ±1`
mod each gear, and `b = c - 1` mod `W`.  (Round 32 had the same result at
`W = 1001`.)

## Build and audit

```
cd C:/dev/primes/proofs
~/.elan/bin/lake.exe build TopMachine TopMachineWheel TopMachineCrt
```

Result: **green**, 1390 jobs, no warnings, no errors.

| target | build time |
|---|---|
| `TopMachineCrt` | 6.0 s elaboration, 9.5 s wall for the cold module |
| `TopMachine`, `TopMachineWheel` | unchanged, cached (6.9 s / 5.9 s cold in R32) |

Ordinary elaboration - no kernel scan, no `decide` - so peak memory is a normal
`lean.exe`, well under 1 GB; no babysitter needed.

Axiom audit over **all 29 new declarations** (`lake env lean` on a local
`#print axioms` file, and the same block appended to `proofs/AxiomCheck.lean`
behind `import TopMachineCrt`):

- every declaration is `[propext, Classical.choice, Quot.sound]` or smaller;
- `anchor`, `SignR`, `decSignR` depend on no axioms; `signR_congr` on
  `[propext]` only; `anchor_covers` and `strikes_congr` on
  `[propext, Quot.sound]`;
- **no `sorryAx`, no `Lean.ofReduceBool`, no `Lean.trustCompiler`.**

Choice is used essentially in `isolate` (`choose` picks the avoiding residue at
each gear) and in `parity_attained` (`choose` picks each gear's anchor residue);
elsewhere it is inherited from mathlib's `Finset` plumbing.

## What the ledger now says

Both round-32 "will not close" verdicts are discharged:

| law | Lean name (namespace `TopMachine`) | status |
|---|---|---|
| L8 group is exactly `(Z/2)^m` | `symm_not_dvd_mul`, `isolate`, `affine_gear`, `affine_group_of_unit`, `affine_group`, `affine_group_form`, `exists_symmetry`, `sign_count` | proved (necessity unconditional for prime gears; general form under invertibility) |
| L17 parity, attainment | `parity_attained`, and the equality `parity_law` | proved |

---

# Round 34: the walk - the next open pair, the next twin candidate, the holes

New file: `proofs/TopMachineWalk.lean` (imports `TopMachineCrt`), registered as a
`lean_lib` in `proofs/lakefile.toml`, in `defaultTargets`, and audited from
`proofs/AxiomCheck.lean`.  Branch document: `research/proof/top_machine_3.md`,
section 4, laws **L30, L31, L34, L35, L44, L45**.  **70 new declarations, zero
sorries, no `native_decide`, no `decide`, no `Lean.ofReduceBool`.**

## The one new primitive: the forward offset

```lean
def off (g : ℕ) (y : ℤ) : ℕ := ((-y) % (g : ℤ)).toNat
```

`off g y` is the least `j >= 0` with `g | y + j`, i.e. `(-y) mod g`.  Every
residue in the branch's closed forms is one of these, which is what makes the
pair machine and the triple machine share one set of lemmas:

| lemma | statement | hypothesis |
|---|---|---|
| `off_lt` | `off g y < g` | `0 < g` |
| `dvd_add_off` | `g` divides `y + off g y` | `0 < g` |
| `off_eq_of_dvd` | `j < g` and `g` divides `y + j` imply `off g y = j` | `0 < g` |
| `off_eq_iff` | `off g y = off g z` iff `g` divides `y - z` | `0 < g` |
| `off_zero`, `off_two` | `off g 0 = 0`, `off g 2 = g - 2` | `0 < g` / `3 <= g` |

In this vocabulary gear `g` strikes the pair `x + j` (for `j < g`) iff
`j = off g x` or `j = off g (x+2)` - the branch's `a_g`, `b_g` - and strikes the
NUMBER `x + j` iff `j = off g x`.

## L30/L31 - THE NEXT OPEN PAIR

```lean
def Res (G : Finset ℕ) (x : ℤ) : Finset ℕ :=
  G.biUnion (fun g => ({off g x, off g (x + 2)} : Finset ℕ))
def mexS (G : Finset ℕ) (x : ℤ) : ℕ := Nat.find (exists_not_mem_nat (Res G x))

theorem mex_form {G : Finset ℕ} (hbig : ∀ g ∈ G, 2 * G.card < g) (x : ℤ) :
    IsLeast {j : ℕ | IsOpen G (x + (j : ℤ))} (mexS G x)
```

`mex_form` is L30 in full: **the next open pair at or after `x` is exactly
`x + mexS G x`**, `mexS` being the mex of the `2m` listed residues.  It is an
`IsLeast`, so both halves are in the kernel:

- `not_open_of_lt_mexS` (every `j < mexS` is struck) - hypothesis `0 < g` only;
- `open_mexS` (the mex position is open) - **this is the half that needs
  `2m < g`**, exactly as the branch's proof says: `mexS_le` gives `mexS <= 2m`
  (a set of `2m` numbers cannot contain all of `0..2m`, by
  `res_card_le : (Res G x).card <= 2 * G.card` against
  `Finset.range (2m+1) ⊆ Res G x`), so the mex is below every gear, and for
  `r < g` the only strikes of `g` in `[x, x+g)` are its two listed residues.

Sharpness is NOT claimed in Lean (the branch's evidence for it is the 36
mismatches at `{7,11,13,17}`, a computation, not a theorem).

```lean
theorem mexS_le (G : Finset ℕ) (x : ℤ) : mexS G x ≤ 2 * G.card
theorem mexS_le_parity {G : Finset ℕ} (hodd : ∀ g ∈ G, g % 2 = 1)
    (hbig : ∀ g ∈ G, 2 * G.card + 1 < g) (x : ℤ) :
    mexS G x ≤ 2 * G.card - G.card % 2
```

`mexS_le` is the plain `2m` bound.  `mexS_le_parity` is **L31 in its sharp
form**, and it is proved by REUSE, not by redoing the covering count: every
position below the mex is struck, so `mexS` is a run of consecutive struck
pairs, and round 32's `parity_upper` (L17) bounds such a run by
`2m - (m mod 2)`.  The branch's own argument (the listed numbers inside
`[0, 2m]` form at most `m` same-parity pairs `{a, a-2}`) is the same content
already in the kernel as `parity_core`.

## L34/L35 - THE NEXT TWIN CANDIDATE, and the record `3m`

The single-number view is new in this file:

```lean
def StrikesN (g : ℕ) (n : ℤ) : Prop := (g : ℤ) ∣ n
def OpenNum (G : Finset ℕ) (n : ℤ) : Prop := ∀ g ∈ G, ¬ StrikesN g n
def IsStart (G : Finset ℕ) (n : ℤ) : Prop :=
  OpenNum G n ∧ OpenNum G (n + 1) ∧ OpenNum G (n + 2)
def Res3 (G : Finset ℕ) (x : ℤ) : Finset ℕ :=
  G.biUnion (fun g => ({off g x, off g (x+1), off g (x+2)} : Finset ℕ))
def mexT (G : Finset ℕ) (x : ℤ) : ℕ := Nat.find (exists_not_mem_nat (Res3 G x))

theorem triple_mex_form {G : Finset ℕ} (hbig : ∀ g ∈ G, 3 * G.card < g) (x : ℤ) :
    IsLeast {j : ℕ | IsStart G (x + (j : ℤ))} (mexT G x)
theorem mexT_le (G : Finset ℕ) (x : ℤ) : mexT G x ≤ 3 * G.card
```

Same three lemmas as the pair case with three teeth (`not_start_of_mem_res3`,
`start_of_not_mem_res3`, `res3_card_le`); the location bound is `3m`.

**The record, as an equality:**

```lean
theorem triple_upper {G : Finset ℕ} (hbig : ∀ g ∈ G, 3 * G.card < g) {n : ℤ} {L : ℕ}
    (hL : ∀ i : ℕ, i < L → ¬ IsStart G (n + (i : ℤ))) : L ≤ 3 * G.card

theorem triple_attained {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    ∃ n : ℤ, ∀ i : ℕ, i < 3 * G.card → ¬ IsStart G (n + (i : ℤ))

theorem triple_law {G : Finset ℕ} (hbig : ∀ g ∈ G, 3 * G.card + 3 ≤ g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    IsGreatest {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsStart G (n + (i : ℤ))}
      (3 * G.card)
```

`triple_law` is **L35 in full: `F_3(G) = 3m` exactly**, stated as an
`IsGreatest` in the shape of `parity_law`.

Two things to note about the proof, because they differ from the branch's
write-up in a way that is worth recording.

1. The **upper bound falls straight out of the mex form** - `triple_upper` is
   three lines: if `3m` consecutive positions were all non-starts, the mex at
   the first of them would exceed `3m`, contradicting `mexT_le`.  The branch's
   window argument ("the trace is a solid interval of at most 3 cells, so
   `L <= 3m`") is the same fact, and the mex route needs only `3m < g`, one
   notch weaker than the branch's `3m + 3 <= g`.  `triple_law` is nevertheless
   stated with `3m + 3 <= g` to keep the branch's hypothesis.
2. The **attainment needs no size hypothesis at all**, exactly as in
   `parity_attained`: the anchor is the solid triomino `[3j, 3j+3)`, gear `j`
   is asked by `exists_crt` to divide `n + 3j + 2`, and that single multiple
   kills all three starts `3j`, `3j+1`, `3j+2` (the multiple sits at offset
   `2`, `1`, `0` from the start).  The covering step that in `anchor_covers`
   needed a parity split here is just `j = i / 3`.

**The contrast is the branch's point, and it is now visible in the two Lean
files side by side.**  `parity_attained` tiles with the GAPPED domino
`{x, x+2}`, which lives in one parity class, so `anchor` must split the gears
into an even pool and an odd pool and one cell is lost when `m` is odd
(`parity_law` = `2m - (m mod 2)`).  `triple_attained` tiles with the SOLID
triomino, which needs no split and loses nothing (`triple_law` = `3m`).  The
parity defect is a property of the separation, not of the tooth count.

## L45 - the holes of the consecutive census

```lean
theorem start_of_start_add_two  (h0 : IsStart G x) (h2 : IsStart G (x+2)) : IsStart G (x+1)
theorem start_of_start_add_three (h0 : IsStart G x) (h3 : IsStart G (x+3)) : IsStart G (x+1)

theorem no_start_gap_two_three {G : Finset ℕ} {x d : ℤ} (hd : d = 2 ∨ d = 3)
    (h0 : IsStart G x) (hdd : IsStart G (x + d)) :
    ∃ y : ℤ, x < y ∧ y < x + d ∧ IsStart G y
theorem no_start_gap ... (hmid : ∀ z, x < z → z < x + d → ¬ IsStart G z) : False
```

**No hypothesis whatever** - not even `0 < g`.  If `x` and `x + d` are both
run-of-three starts with `d` in `{2, 3}`, the three unstruck numbers of each
overlap enough to make `x + 1` a start as well, so the two are never
CONSECUTIVE: the twin-candidate gap census has holes exactly at 2 and 3.  This
is a shorter proof than the branch's (which argues through the gear at
`x = -3 (mod g)`); the branch's version is the stronger statement that the next
candidate is at `x+1` or at `x+4` or beyond, and the overlap argument gives the
hole directly.

`no_pair_gap_four` restates round 32's `open_of_open_add_four` (L4) in the same
"a pair lies strictly between" shape, so the two views' holes read alike.

## L44 - the correlation is a true product

```lean
def BothR (g d r : ℕ) : Prop := ¬ StrikesR g r ∧ ¬ StrikesR g (r + d)
def BothN (G : Finset ℕ) (d n : ℕ) : Prop := ∀ g ∈ G, BothR g d n

theorem corr_prod : ∀ (G : Finset ℕ), (∀ g ∈ G, 0 < g) → (hcop) → ∀ d : ℕ,
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => BothN G d n)).card
      = ∏ g ∈ G, ((Finset.range g).filter (fun r => BothR g d r)).card
```

The product form runs on the **same CRT counting engine as `wheel_count`**
(`card_filter_crt`, with `bothR_congr` / `bothN_congr` for residue invariance
and `bothN_insert` for the induction step): "both open" is a per-gear
condition, so it factors.

The per-gear factor is `g` minus the number of DISTINCT forbidden offsets, and
the forbidden set is written in the `off` vocabulary as the branch writes it,
`{0, -2} u {-d, -d-2}`:

```lean
def CorrTeeth (g d : ℕ) : Finset ℕ := {off g 0, off g 2, off g (d:ℤ), off g ((d:ℤ)+2)}
theorem card_both_residues {g d : ℕ} (hg : 0 < g) :
    ((Finset.range g).filter (fun r => BothR g d r)).card = g - (CorrTeeth g d).card
```

The four coincidence lemmas are exactly the branch's case list, each an
`off_eq_iff` away from a divisibility (`off_zero_ne_off_d`,
`off_zero_ne_off_d_two`, `off_two_ne_off_d`, `off_two_ne_off_d_two`, plus
`off_zero_ne_off_two` and `off_d_ne_off_d_two`, which are `g` not dividing 2):

| case | collisions | `(CorrTeeth g d).card` | factor | Lean name |
|---|---|---|---|---|
| `g` divides `d` | both pairs merge | 2 | `g - 2` | `corrTeeth_card_of_dvd` (`3 <= g`) |
| `g` divides `d + 2` | `0` with `-d-2` | 3 | `g - 3` | `corrTeeth_card_of_dvd_add` (`5 <= g`) |
| `g` divides `d - 2` | `-2` with `-d` | 3 | `g - 3` | `corrTeeth_card_of_dvd_sub` (`5 <= g`) |
| otherwise | none | 4 | `g - 4` | `corrTeeth_card_generic` (`5 <= g`) |

`5 <= g` is used in exactly one place: the cases `g | d +- 2` need `g` not to
divide 4, to rule out the SECOND collision (`g | d+2` together with `g | 2-d`
gives `g | 4`).

```lean
def corrCoeff (g d : ℕ) : ℕ :=
  if d % g = 0 then g - 2
  else if (d + 2) % g = 0 ∨ (d + g - 2) % g = 0 then g - 3
  else g - 4

theorem pair_corr {G : Finset ℕ} (h5 : ∀ g ∈ G, 5 ≤ g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (d : ℕ) :
    ((Finset.range (∏ g ∈ G, g)).filter
        (fun n => OpenN G n ∧ OpenN G (n + d))).card = ∏ g ∈ G, corrCoeff g d
```

`pair_corr` is L44 in full, stated directly on `OpenN` (the bridge is
`bothN_iff`).  `corrCoeff` is written with `%` rather than integer
divisibility so that it is computable; `card_both_residues_eval` carries the
three bridges (`d % g = 0` iff `g | d` over the integers, and likewise for
`d + 2` and for `d + g - 2`, which is `d - 2` modulo `g`).

## Verification before formalising

Scratch script (brute force over full wheel periods), mirroring
`research/topmachine/r3/core.py` / `walk.py`:

- **pair mex form**: `{11,13,17}`, `{13,17,19}`, `{17,19,23}` - every position
  of every period, **0 mismatches**; max walk 5 in all three, `= 2m - (m mod 2)`;
- **triple mex form**: `{13,17,19}`, `{17,19,23}` - **0 mismatches**; max walk
  **9 = 3m** in both, and the number of run-of-three starts is `2240` and
  `4480`, both `= prod (g - 3)` (the branch's table);
- **correlation**: `{11,13,17}`, `{13,17,19}`, `d = 0..39` - `B(d)` against
  `prod corrCoeff g d`, **0 mismatches** (including `B(1) = prod(g-4)` and
  `B(2) = prod(g-3)`, the L15 special cases).

## Build and audit

```
cd C:/dev/primes/proofs
~/.elan/bin/lake.exe build TopMachine TopMachineWheel TopMachineCrt TopMachineWalk
```

Result: **green**, 1394 jobs, no warnings, no errors.

| target | build time |
|---|---|
| `TopMachineWalk` | 7.5 s cold (10.1 s wall for the four targets) |
| `TopMachine`, `TopMachineWheel`, `TopMachineCrt` | unchanged, cached |

Ordinary elaboration throughout - no kernel scan, no `decide` - so peak memory
is a normal `lean.exe`, well under 1 GB; no babysitter needed.

Axiom audit over **all 70 new declarations** (`lake env lean` on a local
`#print axioms` file, and the same block appended to `proofs/AxiomCheck.lean`
behind `import TopMachineWalk`):

- every declaration is `[propext, Classical.choice, Quot.sound]` or smaller;
- `off`, `StrikesN`, `BothR`, `decBothR`, `corrCoeff` depend on no axioms;
  `bothR_congr` on `[propext]` only; `CorrTeeth` on `[propext, Quot.sound]`;
- **no `sorryAx`, no `Lean.ofReduceBool`, no `Lean.trustCompiler`.**

Choice is used essentially only in `triple_attained` (`choose` picks each
gear's anchor residue, as in `parity_attained`); elsewhere it is inherited
from mathlib's `Finset` / `Nat.find` plumbing.

## What the ledger now says

| law | Lean name (namespace `TopMachine`) | status | hypothesis |
|---|---|---|---|
| L30 mex form | `mex_form` (`IsLeast`), `mexS_le` | proved | `2m < g` (openness half only) |
| L31 location bound | `mexS_le_parity` | proved | gears odd, `2m + 1 < g` |
| L34 triple mex form | `triple_mex_form` (`IsLeast`), `mexT_le` | proved | `3m < g` |
| L35 triple record `= 3m` | `triple_upper`, `triple_attained`, `triple_law` (`IsGreatest`) | proved | `3m + 3 <= g`, pairwise coprime |
| L44 correlation product | `corr_prod`, `card_both_residues_eval`, `pair_corr` | proved | `5 <= g`, pairwise coprime |
| L45 triple holes `d = 2, 3` | `start_of_start_add_two/three`, `no_start_gap_two_three`, `no_start_gap` | proved | none |
| L45 pair hole `d = 4` | `no_pair_gap_four` (restating `open_of_open_add_four`) | proved | none |

**Not attempted this round**, and named here so the gap is a first-class
output rather than a silence: L32 (the general mex form for gear sets with
small gears - it needs the mex over the truncated union
`{a_g, b_g} + g Z_{>=0}`, a different `Finset` construction, not a
strengthening of `Res`), L33 (the counting bound, which needs the harmonic sum
`H_S` and so rationals), L36-L39 (the walk distribution `C(j)`, its closed
form, the hop law and the nested form), L40-L43 (everything spectral and
bitwise - they need roots of unity and a DFT, which nothing in these four
files has), and the L22 gap census of `top_machine_2.md`.

---

# Round 35: the stack of machines, and the exhaust

New file: `proofs/MachineStack.lean` (imports `TopMachineWalk` and
`Mathlib.NumberTheory.Bertrand`), registered as a `lean_lib` in
`proofs/lakefile.toml`, in `defaultTargets`, and audited from
`proofs/AxiomCheck.lean`.  Branch: `research/proof/theory_tree.md` node R4.b.viii
(the owner's stack) and `research/proof/top_machine_4.md` L46 (the zone law).
Proof document: `docs/proofs/23-stack-and-exhaust.md`.  **48 new declarations,
zero sorries, no `native_decide`, no `decide`, no `Lean.ofReduceBool`.**

## The object: tiers and cuts

```lean
def gearsIoc (a b : ℕ) : Finset ℕ := (Finset.Ioc a b).filter Nat.Prime
def primesLE (C : ℕ) : Finset ℕ := gearsIoc 1 C

def cut (q : ℕ) : ℕ → ℕ
  | 0 => q
  | 1 => ∏ g ∈ gearsIoc 1 q, g
  | (k + 2) => ∏ g ∈ gearsIoc (cut q k) (cut q (k + 1)), g

def tier (q : ℕ) : ℕ → Finset ℕ
  | 0 => ∅
  | 1 => gearsIoc 1 q
  | (k + 2) => gearsIoc (cut q k) (cut q (k + 1))

def stack (q k : ℕ) : Finset ℕ := (Finset.range (k+1)).biUnion (fun j => tier q (j+1))
def Spans (g : ℕ) (M : Finset ℕ) : Prop := (∏ h ∈ M, h) ≤ g
def Smooth (q n : ℕ) : Prop := ∀ p : ℕ, p.Prime → p ∣ n → p ≤ q
def CutMono (q k : ℕ) : Prop := ∀ j < k, cut q j ≤ cut q (j + 1)
```

The owner's vocabulary: tier 1 is the motor, tier 2 the wheels, `cut q k` is tier
`k`'s period and the lower edge of tier `k + 2` (`tier_prod`: the product over
`tier q (k+1)` is `cut q (k+1)`, `rfl` after a case split).  The strike and open
predicates are round 32/34's (`Strikes`, `IsOpen`, `StrikesN`, `OpenNum`), reused
unchanged; `isOpen_iff_openNum` is the bridge (a pair is open iff both members
are).

## Stride containment, and where it stops

```lean
theorem card_dvd_window_le_one {g P : ℕ} (hgP : P < g) (x : ℤ) :
    (((Finset.range P).filter (fun (j : ℕ) => (g : ℤ) ∣ x + (j : ℤ)))).card ≤ 1
theorem card_strikes_window_le_two {g P : ℕ} (hgP : P < g) (x : ℤ) :
    (((Finset.range P).filter (fun (j : ℕ) => Strikes g (x + (j : ℤ))))).card ≤ 2
theorem spans_two_below {q k g : ℕ} (hg : g ∈ tier q (k + 3)) : Spans g (tier q (k + 1))
theorem stride_containment {q k g : ℕ} (hg : g ∈ tier q (k + 3)) (x : ℤ) :
    (((Finset.range (∏ h ∈ tier q (k + 1), h)).filter
      (fun (j : ℕ) => Strikes g (x + (j : ℤ)))).card) ≤ 2
theorem tier_pattern_repeats (q k : ℕ) (n : ℤ) :
    IsOpen (tier q (k+1)) (n + ((∏ h ∈ tier q (k+1), h : ℕ) : ℤ)) ↔ IsOpen (tier q (k+1)) n
```

Two teeth, so two positions: `card_strikes_window_le_two` is the union of the two
one-multiple counts at `x` and at `x + 2`.  `tier_pattern_repeats` is the other
half of containment - the spanned tier's whole pattern comes round inside one
stride (`isOpen_add_period`, a shift by any common multiple of the gears).

The non-containment side, which is where the construction rule shows its teeth:

```lean
theorem lt_prod_of_two_le {G : Finset ℕ} (h2 : ∀ g ∈ G, 2 ≤ g) (hcard : 2 ≤ G.card)
    {g : ℕ} (hg : g ∈ G) : g < ∏ h ∈ G, h
theorem not_spans_self {q k g : ℕ} (hcard : 2 ≤ (tier q (k+1)).card)
    (hg : g ∈ tier q (k+1)) : ¬ Spans g (tier q (k+1))
theorem gear_le_period_below {q k g : ℕ} (hg : g ∈ tier q (k+2)) : g ≤ ∏ h ∈ tier q (k+1), h
theorem not_spans_below {q k g : ℕ} (hcard : 2 ≤ (tier q (k+1)).card)
    (hg : g ∈ tier q (k+2)) : ¬ Spans g (tier q (k+1))
```

Spanning is a threshold, not a matter of degree: it starts exactly two tiers down.
One tier down it fails for every tier with two or more gears, and the only escape
is the degenerate single-gear tier (its "silent top"), where the gear IS the
period.

## The exhaust cap

```lean
theorem exhaust_home_or_echo {C n p : ℕ} (h1 : C < n) (h2 : n ≤ C ^ 2)
    (hpC : C < p) (hpn : p ∣ n) : n = p ∨ ∃ r ∈ primesLE C, r ∣ n
theorem openNum_iff_prime {C n : ℕ} (hC : 2 ≤ C) (h1 : C < n) (h2 : n ≤ C ^ 2) :
    OpenNum (primesLE C) (n : ℤ) ↔ n.Prime
theorem open_iff_twin {C n : ℕ} (hC : 2 ≤ C) (h1 : C < n) (h2 : n + 2 ≤ C ^ 2) :
    IsOpen (primesLE C) (n : ℤ) ↔ (n.Prime ∧ (n + 2).Prime)
```

**Primality of the exhaust gear is never used** - the kernel shows the cap is a
statement about ANY divisor above the cut: `n = p m` with `m >= 2` forces
`m (C+1) <= C C`, hence `m < C`, hence a prime factor of `n` at or below the cut.
That is one hypothesis weaker than the branch's wording.

Stack form:

```lean
theorem stack_eq_primesLE (q : ℕ) : ∀ k, CutMono q k → stack q k = primesLE (cut q k)
theorem exhaust_gear_gt_cut {q k j g : ℕ} (hmono : CutMono q j) (hjk : k ≤ j)
    (hg : g ∈ tier q (j + 2)) : cut q k < g
theorem exhaust_silent {q k j n g : ℕ} (hmono : CutMono q j) (hjk : k ≤ j)
    (hmk : CutMono q k) (hg : g ∈ tier q (j + 2)) (hgn : g ∣ n)
    (h1 : cut q k < n) (h2 : n ≤ (cut q k) ^ 2) :
    n = g ∨ ∃ r ∈ stack q k, r ∣ n
theorem stack_open_iff_twin {q k n : ℕ} (hmono : CutMono q k) (hC : 2 ≤ cut q k)
    (h1 : cut q k < n) (h2 : n + 2 ≤ (cut q k) ^ 2) :
    IsOpen (stack q k) (n : ℤ) ↔ (n.Prime ∧ (n + 2).Prime)
```

## The hypothesis that is NOT derived, and why it cannot be here

`CutMono` (the cuts are nondecreasing) is a prime-density statement about
`(cut q j, cut q (j+1)]`, not stack arithmetic, and it is **false at the bottom
for `q = 2, 3`**: the cuts at `q = 3` are `3, 6, 5, 1, ...`, so tier 3 is the
single gear `{5}` and tier 4 is EMPTY; at `q = 5` they are `5, 30, 215656441, ...`
and climb.  So it is carried explicitly.  The FIRST step is unconditional, by
Bertrand:

```lean
theorem le_prod_primesLE (n : ℕ) : n ≤ ∏ p ∈ primesLE n, p
theorem cut_zero_le_one (q : ℕ) : cut q 0 ≤ cut q 1
theorem cutMono_one (q : ℕ) : CutMono q 1
theorem stack_one (q : ℕ) : stack q 1 = primesLE (cut q 1)
theorem wheels_open_iff_twin {q n : ℕ} (hq : 2 ≤ q) (h1 : cut q 1 < n)
    (h2 : n + 2 ≤ (cut q 1) ^ 2) :
    IsOpen (stack q 1) (n : ℤ) ↔ (n.Prime ∧ (n + 2).Prime)
```

`le_prod_primesLE` (`n <= n#`) is a strong induction: `n = p m` with `p` least;
`m = 1` gives `n` prime and a factor of `n#`; otherwise `p <= m`, Bertrand supplies
a prime in `(m, 2m]`, contained in `(m, n]` and uncounted in `m#`, and
`n# >= p' m# > m m >= p m = n`.  `wheels_open_iff_twin` is the two-machine window
statement with **no hypothesis beyond `q >= 2`**: motor plus wheels leave a pair
open on `(q#, (q#)^2]` iff it is a twin prime.

## The zone laws of the wheels (top_machine_4.md L46)

```lean
theorem smooth_zone_num {q Q n : ℕ} (hn : 0 < n) (hnQ : n ≤ Q) :
    OpenNum (gearsIoc q Q) (n : ℤ) ↔ Smooth q n
theorem smooth_zone {q Q n : ℕ} (hn : 0 < n) (hnQ : n + 2 ≤ Q) :
    IsOpen (gearsIoc q Q) (n : ℤ) ↔ (Smooth q n ∧ Smooth q (n + 2))
theorem wheels_smooth_zone {q n : ℕ} (hn : 0 < n) (hnQ : n + 2 ≤ cut q 1) :
    IsOpen (tier q 2) (n : ℤ) ↔ (Smooth q n ∧ Smooth q (n + 2))
theorem quiet_zone {q Q n : ℕ} (hn : 0 < n) (hnQ : n ≤ Q ^ 2) :
    OpenNum (gearsIoc q Q) (n : ℤ) ↔
      ∃ s P : ℕ, n = s * P ∧ Smooth q s ∧ (P = 1 ∨ (P.Prime ∧ Q < P))
```

`smooth_zone` is L46 exactly (`n <= Q - 2` in the branch's form).  `quiet_zone` is
the manager's reading of R4.b.vii, arithmetic half: on `(Q, Q^2]` open means
`q`-smooth times at most one prime above `Q`, the "at most one" coming from
`p r >= (Q+1)^2 > Q^2` for two prime factors above `Q`.  `wheels_smooth_zone` is
the same law read on `tier q 2` (definitional: the wheels are
`gearsIoc q (cut q 1)`).

## Verification

Bounded brute force (sympy, scratch script): the cuts at `q = 2, 3, 5, 7, 11`
(degeneracy at 2 and 3 as stated); `n <= n#` for `n < 400`, 0 failures;
open-iff-twin at `C = 30` over all `n` in `(30, 898]`, 0 failures; the exhaust cap
at `C = 30` over all `n` in `(30, 900]` and every prime factor, 0 failures; the
zone law and the quiet zone at `q = 5, Q = 30` over `n <= 900`, 0 failures; stride
containment for gears 31, 37, 101, 1009 over 250 windows of length 30, 0 failures.

## Build and audit

```
cd C:/dev/primes/proofs
~/.elan/bin/lake.exe build TopMachine TopMachineWheel TopMachineCrt TopMachineWalk MachineStack
```

Result: **green**, 2244 jobs, no warnings, no errors; `MachineStack` 9.3 s cold
(15.7 s wall for the five targets), ordinary elaboration - no kernel scan, no
`decide` - peak memory a normal `lean.exe`, no babysitter.

Axiom audit over **all 48 new declarations** (`lake env lean` on a local
`#print axioms` file, and the same block appended to `proofs/AxiomCheck.lean`
behind `import MachineStack`):

- every declaration is `[propext, Classical.choice, Quot.sound]` or smaller;
- `decStrikesInt` and `Smooth` depend on `[propext]` only; `Spans` and
  `strikes_add_period` on `[propext, Quot.sound]`;
- **no `sorryAx`, no `Lean.ofReduceBool`, no `Lean.trustCompiler`.**

Choice is inherited from mathlib's `Finset` and `Nat.minFac` plumbing; nothing
here uses it essentially.

## What the ledger now says

| statement | Lean name (namespace `TopMachine`) | status | hypothesis |
|---|---|---|---|
| stride containment, size | `spans_two_below` | proved | none (definitional) |
| stride containment, count | `card_dvd_window_le_one`, `card_strikes_window_le_two`, `stride_containment` | proved | `P < g` |
| the spanned pattern repeats | `isOpen_add_period`, `tier_pattern_repeats` | proved | gears divide the shift |
| no self-span, no span one tier down | `lt_prod_of_two_le`, `not_spans_self`, `gear_le_period_below`, `not_spans_below` | proved | tier has 2 or more gears |
| exhaust cap (home strike or echo) | `exhaust_home_or_echo` | proved | `C < n <= C^2`, `C < p`, `p` divides `n` |
| open iff prime / iff twin | `openNum_iff_prime`, `open_iff_twin` | proved | `2 <= C < n`, `n (+2) <= C^2` |
| a stack prefix is the primes below the cut | `stack_eq_primesLE` | proved | `CutMono q k` |
| the exhaust is above the cut, and silent | `exhaust_gear_gt_cut`, `exhaust_silent` | proved | `CutMono` |
| the stack's window statement | `stack_open_iff_twin` | proved | `CutMono q k`, `2 <= cut q k` |
| `q <= q#`, and the motor+wheels case | `le_prod_primesLE`, `cutMono_one`, `stack_one`, `wheels_open_iff_twin` | proved | `2 <= q` for the last |
| zone law L46 | `smooth_zone_num`, `smooth_zone`, `wheels_smooth_zone` | proved | `0 < n`, `n + 2 <= Q` |
| quiet zone (R4.b.vii, arithmetic half) | `quiet_zone` | proved | `0 < n <= Q^2` |

**Not attempted**, and named so the gap is an output: `CutMono` itself beyond the
first step (it needs a lower bound on the product of the primes in
`(cut q k, cut q (k+1)]`, i.e. an iterated-Bertrand or Chebyshev argument, and it
is false at `q = 2, 3`); nonemptiness of a tier at height `k >= 2` (same input);
anything about WHERE the openings are on the quiet zone - the file caps the search
space and adds no bound on the in-use machine, which is the instrument still
missing at every cut.

---

# Round 36: the loaded record rule, and `CutMono` discharged

New file: `proofs/TopMachineRecord.lean` (imports `TopMachineWalk`), registered as a
`lean_lib` in `proofs/lakefile.toml`, in `defaultTargets`, and audited from
`proofs/AxiomCheck.lean`.  Branch document: `research/proof/top_machine_7.md`, laws
**L67, L68, L69** and the boundary corollary (**L71**).  Second, smaller piece: an
addendum at the end of `proofs/MachineStack.lean` transcribing the exhaust branch's
**X12/X13** (`research/proof/exhaust_1.md` 3.3), which discharges round 35's carried
hypothesis `CutMono`.  **49 + 14 = 63 new declarations, zero sorries, no
`native_decide`, no `decide`, no `Lean.ofReduceBool`.**

## The objects

```lean
def trace (g : ℕ) (n : ℤ) (L : ℕ) : Finset ℕ :=          -- what gear g shows in [0, L)
  (Finset.range L).filter (fun c => Strikes g (n + (c : ℤ)))
def Piece (x : ℕ) : Finset ℕ := {x, x + 2}                -- a distance-2 domino
def CoveredBy (S : Finset ℕ) (k : ℕ) : Prop :=
  ∃ P : Finset ℕ, P.card ≤ k ∧ S ⊆ P.biUnion Piece
noncomputable def domCost (S : Finset ℕ) : ℕ := sInf {k | CoveredBy S k}   -- D(S)
def run (a j : ℕ) : Finset ℕ := (Finset.range j).image (fun i => a + 2 * i)
def core  (G : Finset ℕ) (L : ℕ) : Finset ℕ := G.filter (fun g => g ≤ L + 1)
def tailG (G : Finset ℕ) (L : ℕ) : Finset ℕ := G.filter (fun g => L + 1 < g)
def uncovered (C : Finset ℕ) (n : ℤ) (L : ℕ) : Finset ℕ :=
  (Finset.range L).filter (fun i => ∀ g ∈ C, ¬ Strikes g (n + (i : ℤ)))
def Coverable (G : Finset ℕ) (L : ℕ) : Prop :=
  ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))
```

`Coverable G L` is literally the membership predicate of `parity_law`'s set, so the
two rounds' record statements are about the same object with no bridge.

## L67, the piece law - and where the branch's wording is one notch off

```lean
theorem trace_subset_domino (hg : L + 1 < g) : ∃ c : ℕ, trace g n L ⊆ ({c, c + 2} : Finset ℕ)
theorem trace_card_le_two   (hg : L + 1 < g) : (trace g n L).card ≤ 2
theorem trace_one_parity    (hg : L + 1 < g) : x ∈ trace g n L → y ∈ trace g n L → x % 2 = y % 2
```

The tail piece: a gear above the boundary shows a subset of ONE domino, hence at most
two cells, hence one parity class.  Proof by the round-32 lemma `window_pair` against
the trace's minimum - no new machinery, and **no hypothesis except `L + 1 < g`**.

The ends-joining half needed the `off` vocabulary of round 34:

```lean
theorem trace_subset_off (hg : 0 < g) (hL : L ≤ g) :
    trace g n L ⊆ ({off g n, off g (n + 2)} : Finset ℕ)
theorem off_pair_diff (hg : 2 ≤ g) (n : ℤ) :
    (off g n : ℤ) - (off g (n + 2) : ℤ) = 2 ∨ (off g n : ℤ) - (off g (n + 2) : ℤ) = 2 - g
```

`off_pair_diff` is the whole content: a gear's two teeth are at separation `2`, or at
separation `g - 2` the other way round, and **nothing else**.  From it:

```lean
theorem trace_crosses_parity_iff (hodd : g % 2 = 1) (h5 : 5 ≤ g) (h4 : 4 ≤ L) (hgL : L ≤ g) :
    (∃ (n : ℤ) (x y : ℕ), x ∈ trace g n L ∧ y ∈ trace g n L ∧ x % 2 ≠ y % 2) ↔ g ≤ L + 1

theorem ends_join_iff (h4 : 4 ≤ L) (hgL : L ≤ g) :
    (∃ n : ℤ, (0 : ℕ) ∈ trace g n L ∧ (L - 1) ∈ trace g n L) ↔ g = L + 1
```

**A finding, and it is a distinction the branch does not draw.**  L67's reading is "a
gear can join the two ends of the window iff `g <= L + 1`".  That is TRUE for
"joining the ends" read as *crossing parity* (`trace_crosses_parity_iff`: the crossing
separation is `g - 2`, which fits in the window exactly when `g <= L + 1`), and FALSE
for "joining the ends" read literally as *striking both `0` and `L - 1`*: that happens
only at `g = L + 1` (`ends_join_iff`).  At `g = L` the gear's parity-crossing piece is
the WRAP pair `{0, L - 2}` or `{1, L - 1}` - which is what the branch's own table says
- so `g = L` crosses parity without touching both endpoints.  Both statements are in
the kernel; the tail hypothesis `g > L + 1` is the right one either way, because it is
the negation of the crossing condition.

Also, `trace_crosses_parity_iff` is stated for `L <= g`, and that restriction is
necessary, not cosmetic: for `g < L` a gear may fail to reach the ends at all (e.g.
`g = 5`, `L = 10`: `5` divides none of `L - 1`, `L + 1`, `L - 3`), so "iff `g <= L+1`"
is a statement about gears at least as big as the window - exactly the regime L67's
case list is written in.

## L68, the matching lemma

```lean
theorem piece_one_parity : y ∈ Piece x → y % 2 = x % 2          -- dominoes never cross parity
theorem card_le_two_mul_of_coveredBy : CoveredBy S k → S.card ≤ 2 * k   -- the lower bound
theorem domCost_union_parity (hA : ∀ x ∈ A, x % 2 = 0) (hB : ∀ x ∈ B, x % 2 = 1) :
    domCost (A ∪ B) = domCost A + domCost B                     -- the parity split
theorem domCost_run (a j : ℕ) : domCost (run a j) = (j + 1) / 2  -- ceil(j/2) per run
```

The branch states L68 as "cost = sum over the maximal step-2 runs of `ceil(run/2)`".
**What is in the kernel is that statement's mechanism, plus the exact value for the
sets the rest of the round needs**, not the general maximal-run formula:

- the lower bound is proved in the general `Finset` form (`card_le_two_mul_of_coveredBy`
  - a piece meets any set in at most two cells, so `k >= |S| / 2`);
- the parity split is proved in the general form, and it is the half of L68 that the
  branch's matching argument really turns on: pieces never cross parity, so a cover
  splits into two disjoint pools and the costs ADD;
- per run the value is exact, both bounds: lower by cardinality (`|run| = j`), upper by
  the explicit tiling `a, a + 4, a + 8, ...` (`domCost_run`).

The general "sum over maximal runs" identity would need a `Finset`-level definition of
maximal runs and an induction over the run decomposition; it was not attempted, and it
is NOT needed anywhere below - the record rule is stated with `domCost` itself, and the
boundary corollary needs only two runs of opposite parity.  That is the honest scope:
**L68's lower bound and parity mechanism are general; its closed form is proved on
runs and on unions of two opposite-parity runs.**

## L69, the loaded record rule - both directions

```lean
theorem cost_le_tail_of_coverable {G : Finset ℕ} {L : ℕ} {n : ℤ}
    (hstruck : ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))) :
    domCost (uncovered (core G L) n L) ≤ (tailG G L).card

theorem coverable_of_cost_le_tail {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) {L : ℕ} {n : ℤ}
    (hU : domCost (uncovered (core G L) n L) ≤ (tailG G L).card) : Coverable G L

theorem loaded_record_rule (hcop) (L : ℕ) :
    Coverable G L ↔ ∃ n : ℤ, domCost (uncovered (core G L) n L) ≤ (tailG G L).card
```

**The hypotheses are weaker than the branch's, in both directions, and that is the
round's main finding about L69.**  The branch carries "pairwise coprime odd gears
`g >= 3`" throughout.  In the kernel:

- **necessity carries NO hypothesis at all** - not coprimality, not oddness, not
  primality, not a size condition.  Every gear that strikes an uncovered cell is a tail
  gear by definition, and `trace_subset_domino` (whose only hypothesis, `L + 1 < g`, IS
  the definition of the tail) puts its whole trace inside one piece.  The cover of `U`
  is then the image of the tail under "the anchor of your piece", so at most `t(L)`
  pieces.
- **sufficiency carries pairwise coprimality and nothing else.**  In particular no size
  hypothesis on the tail gears is used: a gear given the residue `-(c + 2)` strikes
  both `c` and `c + 2` whatever its size, exactly as in `parity_attained`.  A tail
  gear's EXTRA strikes inside the window (which is what `g > L + 1` rules out) can only
  help a cover, never hurt it.  So the tail hypothesis is needed for the rule to be an
  *iff* - it is what makes necessity true - and is free in the sufficiency half.

Assembly, in the shape `parity_law` already uses:

```lean
theorem record_set_eq (hcop) :
    {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))}
      = {L : ℕ | ∃ n : ℤ, domCost (uncovered (core G L) n L) ≤ (tailG G L).card}
theorem record_isGreatest_iff (hcop) (F : ℕ) :
    IsGreatest {L | ∃ n, ∀ i < L, ¬ IsOpen G (n + i)} F ↔ IsGreatest {L | cost ≤ t} F
```

`F_top` is therefore closed in the form the branch asks for: the record length is the
greatest coverable length iff it is the greatest length whose minimal core cost is
affordable.  (The existence of a greatest element is not asserted in general - it is
supplied for free wheels by `parity_law_of_rule` below and, in general, by any bound on
the record; `coverable_mono` records that coverability is downward closed, the
monotonicity `wheelrec.py` assumed.)

## The boundary corollary, and the parity law re-proved

```lean
theorem boundary_cost (L : ℕ) : domCost (Finset.range L) = 2 * (L / 4) + min (L % 4) 2
theorem boundary_greatest (m : ℕ) :
    IsGreatest {L : ℕ | domCost (Finset.range L) ≤ m} (2 * m - m % 2)

theorem parity_law_of_rule (hodd : ∀ g ∈ G, g % 2 = 1) (hbig : ∀ g ∈ G, 2 * G.card + 1 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    IsGreatest {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))}
      (2 * G.card - G.card % 2)
theorem rule_gives_parity_law (hodd) (hbig) (hcop) :
    IsGreatest {L : ℕ | ∃ n, domCost (uncovered (core G L) n L) ≤ (tailG G L).card}
      (2 * G.card - G.card % 2)
theorem parity_law_agrees (hodd) (hbig) (hcop) {F : ℕ}
    (hF : IsGreatest {L | ∃ n, ∀ i < L, ¬ IsOpen G (n + i)} F) : F = 2 * G.card - G.card % 2
```

`boundary_cost` is the branch's P5 (`0,1,2,2,2,3,4,4,4,5,6,6,6` at `L = 0..12`), proved
by splitting `[0, L)` into its even and odd runs (`range_eq_runs`), costing each by
`domCost_run`, and one `omega`.  `boundary_greatest` is P6.

`parity_law_of_rule` is **a second, independent proof of L17** - it uses neither
`parity_upper` nor `parity_attained`:

- lower: at `L = 2m - (m mod 2)` every gear is above `L + 1`, so the core is empty, the
  uncovered set is the whole window, its cost is exactly `m`, and the rule's sufficiency
  places one domino per gear by CRT;
- upper: if a longer run existed, restriction gives one at `L0 + 1`; **oddness is used
  exactly once and exactly here** - at even `m` the hypothesis `g > 2m + 1` leaves
  `g = 2m + 2` possible in principle, and it is oddness that pushes `g` to `2m + 3` so
  that the core is still empty at `L0 + 1` - and then the rule's necessity gives
  `D([0, L0+1)) <= m`, while the closed form gives `m + 1`.

The only shared ancestor of the two proofs is `window_pair`; the covering route never
counts fibres of a striker map.  `parity_law_agrees` states that any greatest element of
the record set is that value, so round 33's `parity_law` and this round's rule cannot
disagree.

## `CutMono` is a theorem from `q = 5` (X12/X13), in `MachineStack.lean`

Round 35 left `CutMono` (the cuts are nondecreasing) as a carried hypothesis, noting it
is a prime-density statement and false at `q = 2, 3`.  The exhaust branch supplied a
proof; transcribed:

```lean
theorem prod_gearsIoc_split (hac : a ≤ c) (hcb : c ≤ b) :
    ∏ p ∈ gearsIoc a b, p = (∏ p ∈ gearsIoc a c, p) * (∏ p ∈ gearsIoc c b, p)
theorem exists_prime_mem_gearsIoc (ha : 0 < a) (hb : 2 * a ≤ b) :
    ∃ p, p ∈ gearsIoc a b ∧ a < p ∧ p ≤ 2 * a
theorem four_mul_lt_prod_gearsIoc : ∀ b a : ℕ,
    ((16 ≤ a ∧ 4 * a ≤ b) ∨ (5 ≤ a ∧ 8 * a ≤ b)) → 4 * b < ∏ p ∈ gearsIoc a b, p
theorem thirty_le_cut_one (hq5 : 5 ≤ q) : 30 ≤ cut q 1
theorem eight_mul_le_cut_one (hq : Nat.Prime q) (hq7 : 7 ≤ q) : 8 * q ≤ cut q 1
theorem cut_two_gt_four_mul (hq : Nat.Prime q) (hq5 : 5 ≤ q) : 4 * cut q 1 < cut q 2
theorem cut_succ_gt_four_mul (hq : Nat.Prime q) (hq5 : 5 ≤ q) (hk : 1 ≤ k) :
    4 * cut q k < cut q (k + 1)
theorem cutMono_of_five_le (hq : Nat.Prime q) (hq5 : 5 ≤ q) (k : ℕ) : CutMono q k
```

**What the Lean proof needed, against the branch's write-up.**  The branch proves
Lemma A (a dyadic product bound with `t = floor(log_2(b/a))`) and then Lemma B by a
real-number comparison of `(t-1) log_2 a + t(t-1)/2` against `t + 1`.  The
transcription replaces both by ONE strong induction on `b` with a halving step, and no
logarithm appears:

- `b >= 16a`: split at `c = b / 2`, use the induction hypothesis on `(a, c]` and one
  Bertrand prime in `(c, b]`;
- `8a <= b < 16a`: three Bertrand primes, `(a, 2a]`, `(2a, 4a]`, `(4a, 8a]`, product
  above `8a^3 >= 64a > 4b`;
- `b < 8a`: then the hypothesis must be the `a >= 16` branch, and two Bertrand primes
  give `2a^2 >= 32a > 4b`.

The hypothesis is the branch's, unchanged: `a >= 16` with `b >= 4a`, or `a >= 5` with
`b >= 8a`.  The base case `cut q 2 > 4 cut q 1` splits exactly as X13 says: for
`q >= 7` it is the dyadic lemma with `b = q# >= 8q` (`eight_mul_le_cut_one`: the motor
carries `2, 3, 5` and `q`, so `q# >= 30q`), and at `q = 5` it is the finite evaluation
`7 * 11 * 13 = 1001 > 120 = 4 * 30`, with `cut 5 1 = 30` proved by `interval_cases`
(`gearsIoc_one_five`), NOT by `decide`.  The induction then carries `30 <= cut q (k+1)`
alongside, which is what supplies the `a >= 16` branch at every later step.

`cutMono_of_five_le` needs `q` prime and `5 <= q` and nothing else; `j = 0` is round
35's `cut_zero_le_one` (Bertrand, unconditional).  With it, `stack_eq_primesLE`,
`exhaust_gear_gt_cut`, `exhaust_silent` and `stack_open_iff_twin` are unconditional for
every prime base `q >= 5`.

## Build and audit

```
cd C:/dev/primes/proofs
~/.elan/bin/lake.exe build TopMachine TopMachineWheel TopMachineCrt TopMachineWalk MachineStack TopMachineRecord
```

Result: **green**, 2246 jobs, no warnings, no errors.

| target | build time |
|---|---|
| `TopMachineRecord` | 7.8 s cold |
| `MachineStack` (with the X12 addendum) | 10.0 s cold (was 9.3 s) |
| the rest | unchanged, cached (13.3 s wall for the six targets) |

Ordinary elaboration throughout - no kernel scan, no `decide` - peak memory a normal
`lean.exe`, no babysitter.

Axiom audit over **all 63 new declarations** (`lake env lean` on a local `#print axioms`
file, and the same block appended to `proofs/AxiomCheck.lean` behind
`import TopMachineRecord`):

- every declaration is `[propext, Classical.choice, Quot.sound]` or smaller;
- `decStrikesRec` depends on `[propext]` only; `Piece`, `core`, `tailG` on
  `[propext, Quot.sound]`;
- **no `sorryAx`, no `Lean.ofReduceBool`, no `Lean.trustCompiler`.**

Choice is used essentially in `domCost` (`sInf` over a non-decidable predicate) and in
`coverable_of_cost_le_tail` (`choose` picks each gear's demanded residue, as in
`parity_attained`) and `cost_le_tail_of_coverable` (`choose` picks each tail gear's
piece anchor); elsewhere it is inherited from mathlib's `Finset` plumbing.

## What the ledger now says

| law | Lean name (namespace `TopMachine`) | status | hypothesis |
|---|---|---|---|
| L67 tail piece | `trace_subset_domino`, `trace_card_le_two`, `trace_one_parity` | proved | `L + 1 < g` |
| L67 two teeth, two separations | `trace_subset_off`, `off_pair_diff` | proved | `L <= g` / `2 <= g` |
| L67 ends-joining = parity crossing | `trace_crosses_parity_iff` | proved | `g` odd, `5 <= g`, `4 <= L <= g` |
| L67 end pair `{0, L-1}` | `ends_join_iff` | proved, and SHARPER (`g = L + 1`, not `g <= L + 1`) | `4 <= L <= g` |
| L68 pieces keep to one parity | `piece_one_parity` | proved | none |
| L68 lower bound | `card_le_two_mul_of_coveredBy` | proved | none |
| L68 parity split | `domCost_union_parity` | proved | the two classes |
| L68 cost of a run | `domCost_run` | proved | none |
| L68 general maximal-run formula | - | **not attempted** (not needed; needs a run decomposition) | - |
| L69 necessity | `cost_le_tail_of_coverable` | proved | **none** |
| L69 sufficiency | `coverable_of_cost_le_tail` | proved | pairwise coprime |
| L69 the rule | `loaded_record_rule`, `record_set_eq`, `record_isGreatest_iff` | proved | pairwise coprime |
| L69 monotonicity of coverability | `coverable_mono` | proved | none |
| L71 empty-core cost, closed form | `boundary_cost`, `boundary_greatest` | proved | none |
| L71 parity law, second proof | `parity_law_of_rule`, `rule_gives_parity_law`, `parity_law_agrees` | proved | gears odd, `2m + 1 < g`, pairwise coprime |
| X12 `CutMono` from `q = 5` | `four_mul_lt_prod_gearsIoc`, `cut_two_gt_four_mul`, `cut_succ_gt_four_mul`, `cutMono_of_five_le` | proved | `q` prime, `5 <= q` |

**Not attempted this round**, named so the gap is an output: L70 (the capacity bound -
one `omega` away from `card_le_two_mul_of_coveredBy` plus a core-gear count, but the
core count `2 ceil(L/g)` needs a per-gear window census that is not in the kernel);
L73/L74 (the moment vanishing and `r(d) = D(d-1)` - they need the multilinear expansion
over `{0,1}^{d-1}` and a Boolean-cube Mobius inversion, which nothing in these files
has); L22 via K1-K3 (K2's general `Finset.powerset` inclusion-exclusion is the one new
piece of machinery; K1 and K3 are mechanical on `card_filter_crt`).  The round's time
went to L67-L69, the boundary corollary, and the `CutMono` transcription requested
mid-round.


---

# Round 37: the gap census law L22 (W22) in the kernel

New file: `proofs/TopMachineCensus.lean` (imports `TopMachineWalk`, so the whole
`TopMachine` stack below it), registered as a `lean_lib` in `proofs/lakefile.toml`, in
`defaultTargets`, and audited from `proofs/AxiomCheck.lean`.  Source: L22 of
`research/proof/top_machine_2.md` in the kernel shape of `research/proof/top_machine_7.md`
section 5 (K1, K2, K3).  Register row **W22**; the `d = 4` corollary is the
register's "`4` the only identically zero length" (W24) and L26's "`d = 4` exception",
now a one-line consequence of the formula.  **41 new declarations, zero sorries, no
`native_decide`, no `decide`, no `Lean.ofReduceBool`.**  The ledger is now 310
declarations across seven libs.

## The objects

```lean
-- n and n + d are consecutive open pairs: both open, nothing open strictly between
def ConsecOpen (G : Finset ℕ) (d : ℕ) (n : ℤ) : Prop :=
  IsOpen G n ∧ IsOpen G (n + d) ∧ ∀ z : ℤ, n < z → z < n + d → ¬ IsOpen G z
def ConsecOpenN (G : Finset ℕ) (d n : ℕ) : Prop :=                       -- residue form
  OpenN G n ∧ OpenN G (n + d) ∧ ∀ j ∈ Finset.Ioo 0 d, ¬ OpenN G (n + j)
theorem consecOpen_natCast : ConsecOpen G d (n : ℤ) ↔ ConsecOpenN G d n

-- the census N_d(G): residues mod the wheel at which n, n + d are consecutive open
def Ncount (G : Finset ℕ) (d : ℕ) : ℕ :=
  ((Finset.range (∏ g ∈ G, g)).filter (ConsecOpenN G d)).card

-- the offsets n must avoid, and E_g(S) = their negatives mod g (off g x = (-x) mod g)
def Offs (d : ℕ) (S : Finset ℕ) : Finset ℕ := ({0, 2, d, d + 2} : Finset ℕ) ∪ S ∪ S.image (· + 2)
def E (d : ℕ) (S : Finset ℕ) (g : ℕ) : Finset ℕ := (Offs d S).image (fun x : ℕ => off g x)

-- the general per-gear avoidance predicate behind K3
def AvoidN (G : Finset ℕ) (F : ℕ → Finset ℕ) (n : ℕ) : Prop := ∀ g ∈ G, n % g ∉ F g
```

`E d S g` is a `Finset` image, so `|E_g(S)|` is whatever the image's card is; no
distinctness hypothesis is stated anywhere, and none was needed.  `E d ∅ g = CorrTeeth g d`
(`E_empty_eq_corrTeeth`), the four-tooth set of round 34's pair correlation.

## K1, the local characterisation

```lean
theorem mod_eq_off_iff (hg : 0 < g) (n x : ℕ) : n % g = off g (x : ℤ) ↔ (n + x) % g = 0
theorem strikesR_add_iff (hg : 0 < g) (n x : ℕ) :
    StrikesR g (n + x) ↔ (n % g = off g (x : ℤ) ∨ n % g = off g ((x + 2 : ℕ) : ℤ))
theorem not_mem_E_iff (hg : 0 < g) (d S n) :
    n % g ∉ E d S g ↔ ¬ StrikesR g n ∧ ¬ StrikesR g (n + d) ∧ ∀ j ∈ S, ¬ StrikesR g (n + j)
theorem avoidN_E_iff (hG0 : ∀ g ∈ G, 0 < g) (d S n) :
    AvoidN G (E d S) n ↔ OpenN G n ∧ OpenN G (n + d) ∧ ∀ j ∈ S, OpenN G (n + j)
theorem avoidN_E_empty_iff (hG0) (d n) : AvoidN G (E d ∅) n ↔ OpenN G n ∧ OpenN G (n + d)
theorem not_openN_add_iff (hG0) (n x) :
    ¬ OpenN G (n + x) ↔ ∃ g ∈ G, (n % g = off g (x : ℤ) ∨ n % g = off g ((x + 2 : ℕ) : ℤ))
-- K1 as the branch states it
theorem consecOpenN_iff_residues (hG0) (d n) :
    ConsecOpenN G d n ↔ AvoidN G (E d ∅) n ∧
      ∀ j ∈ Finset.Ioo 0 d, ∃ g ∈ G, (n % g = off g (j : ℤ) ∨ n % g = off g ((j + 2 : ℕ) : ℤ))
-- K1 in the shape K2 consumes
theorem consecOpenN_iff_avoid (hG0) (d n) :
    ConsecOpenN G d n ↔ AvoidN G (E d ∅) n ∧ ∀ j ∈ Finset.Ioo 0 d, ¬ OpenN G (n + j)
```

Hypothesis: gears positive (`0 < g`), nothing else.  The one arithmetic fact is
`mod_eq_off_iff`, which is `dvd_iff_off_eq` (round 34) after splitting `n = g (n/g) + n % g`.

## K2, inclusion-exclusion over a finite set of positions

```lean
theorem card_filter_forall_not {α ι : Type*} (A : Finset α) (J : Finset ι)
    (P : ι → α → Prop) [∀ j, DecidablePred (P j)] :
    ((A.filter (fun n => ∀ j ∈ J, ¬ P j n)).card : ℤ)
      = ∑ S ∈ J.powerset, (-1 : ℤ) ^ S.card * ((A.filter (fun n => ∀ j ∈ S, P j n)).card : ℤ)
```

General in the point set `A`, the position set `J` and the predicate `P` ("position `j`
is unstruck at `n`"); nothing about the machine enters.  Proof: pointwise
`prod_{j in J} (1 - [P j n]) = sum_S (-1)^|S| prod_{j in S} [P j n]` by
`Finset.prod_add` (with `f = -[P]`, `g = 1`) and `Finset.prod_boole`, then
`Finset.sum_boole` on both sides and `Finset.sum_comm`.  Mathlib's own
`Finset.inclusion_exclusion_card_inf_compl` needs a `Fintype` ambient type and was not
used.

## K3, the CRT product for per-gear forbidden sets

```lean
theorem card_avoid_prod : ∀ (G : Finset ℕ), (∀ g ∈ G, 0 < g) →
    (∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) → ∀ (F : ℕ → Finset ℕ),
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => AvoidN G F n)).card
      = ∏ g ∈ G, ((Finset.range g).filter (fun r => r ∉ F g)).card
theorem card_range_filter_not_mem (hF : F ⊆ Finset.range g) :
    ((Finset.range g).filter (fun r => r ∉ F)).card = g - F.card
theorem card_avoid_E (hG0) (hcop) (d S) :
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => AvoidN G (E d S) n)).card
      = ∏ g ∈ G, (g - (E d S g).card)
-- the S = ∅ term is the pair correlation (L44) with its teeth counted as a Finset
theorem pair_corr_teeth (hG0) (hcop) (d) :
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => OpenN G n ∧ OpenN G (n + d))).card
      = ∏ g ∈ G, (g - (CorrTeeth g d).card)
```

`card_avoid_prod` is the same `Finset.induction_on` + `card_filter_crt` argument as
`wheel_count` (round 32) and `corr_prod` (round 34), with an arbitrary forbidden set
`F g` per gear; `wheel_count` is the instance `F g = {0, g - 2}` and `corr_prod` the
instance `F g = CorrTeeth g d` (`pair_corr_teeth` shows the latter through the new
lemma).  Hypotheses: gears positive, pairwise coprime.

## L22 / W22, the gap census law

```lean
theorem gap_census (hG0 : ∀ g ∈ G, 0 < g) (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (d : ℕ) :
    (Ncount G d : ℤ)
      = ∑ S ∈ (Finset.Ioo 0 d).powerset,
          (-1 : ℤ) ^ S.card * ((∏ g ∈ G, (g - (E d S g).card) : ℕ) : ℤ)
theorem gap_census_int (hG0) (hcop) (d) :          -- the product taken in ℤ
    (Ncount G d : ℤ)
      = ∑ S ∈ (Finset.Ioo 0 d).powerset, (-1 : ℤ) ^ S.card * ∏ g ∈ G, ((g : ℤ) - ((E d S g).card : ℤ))
theorem gap_census_raw (hG0) (hcop) (d) :          -- the filter written out
    (((Finset.range (∏ g ∈ G, g)).filter (fun n : ℕ => ConsecOpenN G d n)).card : ℤ) = ...
```

Proof: `consecOpenN_iff_avoid` rewrites the census filter as
`(filter (AvoidN G (E d ∅))).filter (∀ j ∈ Ioo 0 d, ¬ OpenN G (n + j))`;
`card_filter_forall_not` with `P j n := OpenN G (n + j)` gives the alternating sum;
for each `S`, `avoidN_E_empty_iff` + `avoidN_E_iff` fold the term's filter into
`AvoidN G (E d S)`, and `card_avoid_E` counts it.  **Hypotheses as the proof needed
them: every gear positive, gears pairwise coprime.**  No `g >= 5`, no `g > d + 2`, no
primality, no odd-gear hypothesis: the size hypotheses of the branch are only needed to
EVALUATE `|E_g(S)|` (W23's universality), not to state or prove the law.

## The `d = 4` check: the alternating sum is identically zero

```lean
theorem offs_four_insert_two (S : Finset ℕ) : Offs 4 (insert 2 S) = Offs 4 S
theorem E_four_insert_two (S : Finset ℕ) (g : ℕ) : E 4 (insert 2 S) g = E 4 S g
theorem gap_four_zero (hG0) (hcop) : Ncount G 4 = 0
```

At `d = 4` the interior position `2` contributes the offsets `2` and `4`, both already
among the boundary offsets `{0, 2, 4, 6}`, so `E_g(S u {2}) = E_g(S)` for every `S` and
every `g`; `Finset.sum_powerset_insert` pairs each `S ⊆ {1, 3}` with `S u {2}`, the two
products agree and the signs differ, and the sum is `0`.  This is L4 (`no_gap_four`,
round 32) re-derived from the algebra of L22 rather than from the direct argument - the
check that the formula's algebra is captured, and L26's "`d = 4` exception" as
`top_machine_7.md` section 5 reads it.

## Build and audit

```
cd C:/dev/primes/proofs
~/.elan/bin/lake.exe build TopMachine TopMachineWheel TopMachineCrt TopMachineWalk MachineStack TopMachineRecord TopMachineCensus
```

Result: **green**, 2248 jobs, no warnings, no errors.  `TopMachineCensus` 6.6 s cold
(1393 jobs on its own), the other six cached; ordinary elaboration throughout, a normal
`lean.exe`, no kernel scan.

Axiom audit over **all 41 new declarations** (`lake env lean` on a scratch
`#print axioms` file, and the same block appended to `proofs/AxiomCheck.lean` behind
`import TopMachineCensus`): every declaration is
`[propext, Classical.choice, Quot.sound]`; **no `sorryAx`, no `Lean.ofReduceBool`, no
`Lean.trustCompiler`.**  Choice is inherited from mathlib's `Finset` plumbing; nothing
in the file chooses.

## What the ledger now says

| law | Lean name (namespace `TopMachine`) | status | hypothesis |
|---|---|---|---|
| L22 / W22 the gap census law | `gap_census`, `gap_census_int`, `gap_census_raw` | **proved** | gears positive, pairwise coprime |
| K1 local characterisation | `consecOpenN_iff_residues`, `consecOpenN_iff_avoid`, `avoidN_E_iff`, `not_mem_E_iff`, `strikesR_add_iff`, `mod_eq_off_iff` | proved | `0 < g` |
| K2 inclusion-exclusion over positions | `card_filter_forall_not` | proved, general (any `A`, `J`, `P`) | none |
| K3 CRT product per forbidden set | `card_avoid_prod`, `card_avoid_E`, `card_range_filter_not_mem` | proved | gears positive, pairwise coprime |
| the `S = ∅` term = L44 pair correlation | `pair_corr_teeth`, `E_empty_eq_corrTeeth`, `avoidN_E_empty_iff` | proved | gears positive, pairwise coprime |
| raw line = residue form | `consecOpen_natCast` | proved | none |
| L4 / W24 `d = 4` identically zero, FROM the formula | `gap_four_zero`, `E_four_insert_two`, `offs_four_insert_two` | proved | gears positive, pairwise coprime |
| L15 `d = 1`, L9, L18 as instances | - | not written out (they are `gap_census` at `d = 1` and the evaluations of `|E_g(S)|`, W23) | - |

**Not attempted this round**, named so the gap is an output: L73 (the moment vanishing
`M_k(d) = 0` for `k < r(d)`) and L74 (`r(d) = D(d-1)`).  The moment form L25 writes
`N_d = sum_k (-1)^k sigma_{m-k} M_k(d)` with `M_k(d) = sum_e c_e(d) e^k`, which needs
W23's evaluation `|E_g(S)| = e(S)` for gears `> d + 2` (a per-subset distinctness count
on `off g`, the `corrTeeth_card_*` lemmas of round 34 generalised from four offsets to
`4 + 2|S|`), then the expansion of `prod (g - e(S))` in elementary symmetric functions
of the gears, and finally the vanishing itself, a Boolean-cube Mobius inversion over
`{0,1}^{d-1}`.  None of the three pieces is in the kernel; the first is mechanical, the
third is the branch's actual content.  The time went to K1-K3, the assembly and the
`d = 4` derivation.
