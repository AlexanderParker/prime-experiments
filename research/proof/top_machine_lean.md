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
| L8 group is exactly `(Z/2)^m` | - | **will not close** (see below) | - |
| L10 run `< q' - 2` | `no_long_run`, `run_lt` | proved | `5 ≤ q'`, `q' ∈ G` |
| L10 run `q' - 3` attained | `run_attained` | proved | gears `≥ q' ≥ 3` |
| L10 chain `< q' - 1` | `no_long_chain2`, `chain2_lt` | proved | `q'` odd, `3 ≤ q'`, `q' ∈ G` |
| L10 chain `q' - 2` attained | `chain2_attained` | proved | `q'` odd, gears `≥ q' ≥ 3` |
| L12 chain law | `chain_law` | proved | none |
| L13 merge law | `merge_law` | proved | none |
| L17 parity, upper bound | `parity_core`, `parity_upper` | proved | gears odd and `> 2m + 1` |
| L17 parity, attainment | - | **will not close** (see below) | - |
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
