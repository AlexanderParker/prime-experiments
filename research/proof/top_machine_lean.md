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
