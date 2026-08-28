# Formalist workstream - cumulative findings, rounds 1-24

Compacted 2026-08-29 into one cumulative summary over all 24 rounds. Full
verbatim logs: `archive/formalist-full-r1-19.md` and
`archive/formalist-full-r20-24.md` (the pre-compaction copy of this file).
Rounds 12, 14, 19 have no entries in the r1-19 log - the lane was not briefed.

Mandate: kernel-check the twin-prime proof-search results in Lean 4 / mathlib -
exact statements, zero sorries, axiom-audited; verify every claim against the
`research/*.py` tooling BEFORE formalising; record honest "will not close"
verdicts as first-class outputs.

## 0. Final state (round-24 close)

Build **GREEN at 1372 jobs**, 55 targets / 107 libs, **zero sorries**, zero
warnings in owned files, **no `native_decide`, no `Lean.ofReduceBool` anywhere
in the ledger**. Axiom audit over **273 declarations**: standard three or
smaller. Jobs: 1252 (r17) -> 1254 (r18/19) -> 1276 (r20) -> 1302 (r21) -> 1322
(r22) -> 1334 (r23) -> 1372 (r24).

- **(D) at alpha = 3 kernel-proved, hypothesis-free, at FIVE consecutive steps**:
  11->13, 13->17, 17->19, 19->23, 23->29.
- **17->19 proved twice** (merge law; a 37-integer LP covering certificate)
  sharing only the definition of the machine. 11->13 and 13->17 also have second
  covering-certificate proofs.
- **(D) has a depth-quantifier-free form**, with exhibited potentials h11, h13,
  h17, h19 at all four scannable rungs.
- **The per-rung period-scan vehicle is measured dead past 23->29** (~170 h at
  29->31); replacements are the sandwich lemma and bounded-state longest-path
  certificates.

Process note (r24): the engineering agent was lost to an API limit after its
build completed but before filing. The block was rebuilt from disk by a
successor - build re-run green, axiom audit re-run in its own process,
scaffolding re-classified against lakefile + import graph, the headline
performance claim re-measured as a controlled A/B. Nothing taken on the lost
agent's word.

## 1. Infrastructure facts (consolidated)

### 1.1 Lake and the ledger

- **Lake does NOT glob `proofs/`.** Every root file needs its own `[[lean_lib]]`
  in `proofs/lakefile.toml` PLUS an entry in `defaultTargets`. Sibling modules
  imported by a root (PolignacCap split, `MachineNNS0..S16` slice families) must
  ALSO each be a `lean_lib` or imports fail with "unknown module prefix".
- Build `~/.elan/bin/lake.exe build` from `proofs/`; audit
  `~/.elan/bin/lake.exe env lean AxiomCheck.lean`. `lake env lean <file>` needs
  no registration if its imports are built - how measurements/dry-checks avoid
  perturbing the ledger.
- Lake keys on CONTENT HASHES, not mtimes: `touch` + rebuild returns "up to
  date" in 1 s.
- A killed lake leaves `.lake/config/N/lakefile.olean.lock` stale - remove it or
  invocations hang. Lake runs `git rev-parse` + `git diff --exit-code HEAD` per
  invocation (slow on a busy repo); `lakefile.toml` edits trigger a big
  trace/replay pass; `lake env lean` fails transiently with "failed to read
  ....olean.private" during a parallel build - retry, don't debug.
- Read the WHOLE failed-target list, not the tail (r20 rebuilt 3 of 10 because
  `tail -5` truncated it).
- `#print axioms` blocks appended to `AxiomCheck.lean` need their imports
  spliced at the TOP (imports precede the first command).
- The mathlib cache is PARTIAL: `Mathlib.Data.Finite.Basic` is not built (no
  `Finite (Fin n)` instance); `Mathlib.Data.ZMod.Basic` is.

### 1.2 Memory, parallelism, the round-24 measurements

**Binding constraint on kernel scans is RAM: ~5 GB PER LEAN WORKER even in the
good encoding.** Controlled A/B, same slice (e=16,f=18), same box, strictly
sequential, neither run boosted:

    encoding                     file              wall            peak RSS
    position-indexed (new)   qsliceIdx 16 18    65 s / 81 s         5.38 GB
    offset-walk (round 23)   qslice23  16 18    >= 1,780 s          8.80 GB
                                                (ABORTED, unfinished)

- Six-way = ~32 GB on a 16 GB box: hopeless by 2x in EITHER encoding. Two-way =
  ~10.8 GB, ~5 GB for the OS - marginal, matching the babysitter guard firing to
  `allowed=1` 44x and `allowed=0` 15x.
- **Parallelism budget for a kernel scan is 2, set by ~5 GB/worker, not by cores
  or slices.** (The older qualitative rule - at most 2-3 targets per `lake build`
  invocation, sequentially; lake 5.0.0 has no jobs flag - is the same rule.)
- **Size scans in peak RSS per declaration, and MEASURE it**: a "~0.4 GB" draft
  figure written on plausibility was wrong by 13x and cost 81 s to check. R23's
  cost model had no memory column - that is what made six-way look reasonable.

**Livelock post-mortem: six-way parallel completed NOTHING in 11 hours.**
`Machine23Idx.olean` 07:30:39, dry-scan 08:04, first of 17 slice modules not
until 19:15:33 and only after a babysitter took over at 18:45. ~10.75 h produced
ZERO completed modules; the same work finished in **3 h 36 min** serialised to
2. Thrashing is a livelock, not a slowdown: each worker gets 2-10% CPU, none
reaches its `.olean`, lake's next invocation has nothing to reuse. **A
kernel-scan build that does not fit in RAM makes NEGATIVE progress.**

**`research/lean_babysitter.py` - default for any multi-module scan.** Keeps at
most `MAX_RUN`(=2) `lean.exe` RUNNING, SUSPENDS the rest, resumes
longest-suspended-first. Transferable points: (i) suspension is REVERSIBLE and
loses no work (state in the pagefile, finished `.olean`s reused) - **killing
workers loses hours, suspending loses nothing**; (ii) it calls `EmptyWorkingSet`
(psapi) so suspended pages move out IMMEDIATELY, which is what returns physical
RAM; (iii) it scales runners by AVAILABLE RAM (<1.5 GB -> 1, <0.75 GB -> 0 until
recovery above 2.0 GB) - RAM bottomed at **221 MB**, 22 pids over 7 supervisor
restarts; (iv) rank runners by LARGEST resident set first - the biggest worker
is most memory-warm and closest to finishing, so it frees the most RAM soonest.

**Sweep-resilient resume loops.** A process sweep struck TWICE in r21, killing
both slice loops mid-run; **skip-if-built resume loops meant zero loss both
times.** Drive any multi-hour slice family with a loop that checks for the
`.olean` before invoking lake.

**Priority boosting is worth 2.3x and should be the default here**: same
mini-slice 28 s -> 12 s with `lean.exe` raised to `High` by a poll loop beside
the build (box at 100% CPU from ~8 other-lane `python3.12` on 14 cores / 20
threads; r21 saw background-started leans starved until raised to AboveNormal):

    for i in $(seq 1 N); do powershell -NoProfile -Command "Get-Process lean \
      -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'High' }"; \
      sleep 2; done

**Killed-run timing artifact: an unfinished job's wrapper writes a
finished-looking number.** The wrapper wrote `OLD_OFFSET_SLICE_SECONDS=1785` for
a run killed at 1,780 s; the log had no lean output because the theorem never
elaborated. The line dates the WRAPPER, not the job (mechanic's rule 17).
Defences: check the process list, and check the job's own output (theorem,
`.olean`) exists. Corollary: **mini-slice extrapolation is not linear here** -
memory pressure grows with the declaration, so r23's 420 s/slice (143 tuples x
35; ~966 s unboosted) was an UNDERESTIMATE; the real slice was unfinished at
1,780 s.

### 1.3 Kernel-scan scaling and encoding

- **~5e3 tuples per DECLARATION max**, and only a handful of heavy
  `decide +kernel` per MODULE - lake gives each module its own process; both
  limits are per-process state, not total work. Eight decides in one file:
  >2.3 GB, 20+ min, though each alone was 17-60 s. Split into modules under one
  root. (6 heavy decides/file, 3 files per invocation, stayed inside the rule.)
- **The rule of thumb is about DISTINCT SUB-COMPUTATIONS, not quantifier range**:
  `CoveringCert2.cert17` covers 85,085 tuples in seconds - only 53 distinct `S`
  and 240 distinct `P` sums are ever evaluated.
- **The kernel SHARES structurally identical subterms - measure, don't assume.**
  Control: 23 IDENTICAL iterations cost 1x (1.35 s), 23 NEARLY-identical ones
  21x (12 s). **What must be shared has to be SYNTACTICALLY free of the loop
  variable, not merely equal in value.** Cheapest "is my encoding paying for
  repetition?" diagnostic in the lane.
- **A diagnostic ladder beats a single probe**: four mini-slice variants
  (calibration / one phase / two clauses / everything) with `profiler true` and
  `profiler.threshold 20` located the cost in one 2.5-min run; r20's single
  canary ran 25 min and said only "too slow".
- **seekT walk encoding: 12x.** Read MULTIPLE window facts off ONE walk.
  `countP`-per-fact re-walks per fact; a `seekT` chain visits each slot once,
  stops at the last needed opening, and turns extraction into EQUATIONS (no
  Nodup/pigeonhole). 97 s vs 1150 s per slice under identical load. **Recipe:
  walk to the k-th next opening, assert positions.**
- **Position-index fix: >=22x per slice, 3.6x end to end.** Index the opening
  chain by POSITION not offset (`w19 a b c d e f k` = offset of the k-th
  machine-19 opening after the base). For literal `k` the term does not mention
  the gear-23 phase `g`, so all 23 phases reduce the SAME closed term and the
  m19 walk is evaluated once per CRT tuple; the phase loop only SELECTS
  positions. Predicted ~5x; measured >=22x/slice (65-81 s vs an aborted
  >=1,780 s) and 3 h 36 min vs a 13 h estimate at MAX_RUN=2. Only ~1.6x on
  memory.
- **Scan-first-check trick**: make the FIRST clause `Nat.ble o1 F`. It
  re-derives `F_1 <= F` from the same walk - the fuel-sufficiency fact
  `seek_next` needs - so a new machine's scan imports no bound from an earlier
  scan.
- **CRT-tuple recipe**: direct `decide` over residues mod 5005 does not
  terminate; scan the CRT TUPLE (all `a<5, b<7, c<11, d<13`, shifts mod each
  gear separately) - same 5005 cases, single-digit moduli, 12.4 s. General for
  any period that is a product of small primes.
- Big literal tables: plain `decide` hits elaborator maxRecDepth;
  `maxRecDepth 8192` (20000 for CoveringCert) + `decide +kernel` stays
  axiom-clean. `decide +kernel` handles `Finset.sum` fine.
- **Speed stack (38x)**: allocation-free fuel-recursive Bool over Nat state
  (list allocation dominates kernel time); restrict starts to the exposed set
  (2-7x); measure fuel instead of guessing.
- Load-bearing times: m13 period scan 12.4 s; `forbidden_pairs_count` 22 s; m17
  slice ~16 s; PolignacCap classes 17-60 s post-38x (10:48 before); m19 F4-walk
  ~13 s/slice, 246 s per 19-slice file, ~70 min/machine; m13 4-step walk fuel 11,
  37 s; m17 6-step walk fuel 18, ~35 s/slice; assemblies 19-26 s.

### 1.4 Dry-elaboration discipline

- **Sorry'd-assembly dry-check**: copy the root, swap slice imports for the core
  import, `sorry` the assembly theorem, `lake env lean`. R20's entire 300-line
  root elaborated before any slice finished, so it compiled first try. Zero
  kernel cost.
- **Mega-dry file**: concatenate all new modules (imports stripped) over
  already-built imports, one `lake env lean` - four files' elaboration at zero
  kernel cost, no lock conflict with a running build; caught 3 real bugs in r21.
  Build with `grep -v "^import "` per source: **trimming by line offsets silently
  eats `namespace` lines.** R22 skipped it; 3 of its 4 failures would have been
  caught free.
- **Axiom-stubbed dry file** (`proofs/DryScan2.lean`): stub the expensive fact as
  `axiom qsliceIdxAll : ...`, develop the whole consumer against the stub while
  the multi-hour scan runs, delete on landing. **Standing rule: delete or
  `_`-prefix such a file the moment its scan lands - an `axiom` in `proofs/` is
  one lakefile edit from silently entering the ledger.**

## 2. Kernel-checked theorems, by file

Axiom footprint is the standard three `[propext, Classical.choice, Quot.sound]`
unless noted (audit in section 3). Signatures verbatim from the round logs (a
few logged abbreviated, marked `...`; full statements in the files). Later
rounds' logs use `N` for `ℕ`, `and`/`or`/`->` for `∧`/`∨`/`→`, `|` for `∣`.

### 2.1 proofs/Horizon.lean

Interior form: gears strictly below y decide the OPEN window (y, y*y) - sharper
than BlockedSlots.survivor_iff_twin (q <= y, closed).

```lean
theorem exists_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (hnp : ¬ m.Prime) : ∃ p, p.Prime ∧ p < y ∧ p ∣ m

theorem prime_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m) : m.Prime

theorem twin_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hwin : m + 2 < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m ∧ ¬ p ∣ (m + 2)) :
    m.Prime ∧ (m + 2).Prime
```

### 2.2 proofs/Layer.lean

Bertrand avoided by carrying `y'^2 <= y^3` as an explicit hypothesis (holds for
consecutive primes from y = 3; the caller discharges it).

```lean
theorem slot_cap {q m : ℕ} (hq : 3 ≤ q) : ¬ (q ∣ m ∧ q ∣ m + 2)

theorem minFac_lt_or_eq {y y' m : ℕ}
    (hnext : ∀ q, q.Prime → y < q → q < y' → False)
    (h1 : 1 < m) (hnp : ¬ m.Prime) (hm : m < y' * y') :
    m.minFac < y ∨ m.minFac = y

theorem eq_mul_prime_of_minFac_eq {y m : ℕ} (h1 : 1 < m)
    (hfac : m.minFac = y) (hlow : y * y < m) (hhigh : m < y * y * y) :
    ∃ c, c.Prime ∧ y < c ∧ m = y * c

theorem layer_novelty {y y' m : ℕ}
    (hnext : ∀ q, q.Prime → y < q → q < y' → False)
    (hthin : y' * y' ≤ y * y * y)
    (hnp : ¬ m.Prime) (hlow : y * y < m) (hhigh : m < y' * y') :
    (∃ p, p.Prime ∧ p < y ∧ p ∣ m) ∨ ∃ c, c.Prime ∧ y < c ∧ m = y * c
```

### 2.3 proofs/Supply.lean

Root partition by minFac; the window hypothesis is per-member, so S need not be
an interval.

```lean
theorem minFac_mem_gears {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (hnp : ¬ m.Prime) : m.minFac ∈ (Finset.range y).filter Nat.Prime

theorem card_composites_eq_sum_roots {y : ℕ} (S : Finset ℕ)
    (hS : ∀ m ∈ S, y < m ∧ m < y * y) :
    (S.filter fun m => ¬ m.Prime).card =
      ∑ p ∈ (Finset.range y).filter Nat.Prime,
        (S.filter fun m => ¬ m.Prime ∧ m.minFac = p).card

theorem card_eq_primes_add_sum_roots {y : ℕ} (S : Finset ℕ)
    (hS : ∀ m ∈ S, y < m ∧ m < y * y) :
    S.card = (S.filter Nat.Prime).card
      + ∑ p ∈ (Finset.range y).filter Nat.Prime,
          (S.filter fun m => ¬ m.Prime ∧ m.minFac = p).card

theorem roots_ne {m : ℕ} (h1 : 1 < m) (hodd : ¬ 2 ∣ m) :
    m.minFac ≠ (m + 2).minFac
```

### 2.4 proofs/Census.lean

Slot k carries (6k-1, 6k+1) as `lo`/`hi`; counters tied to real `Nat.Prime`, all
over arbitrary `T : Finset Nat`. `n0_eq_zero_iff` is exactly Condition X.

```lean
theorem census_partition : n0 T + n1 T + n2 T = T.card
theorem comps_eq         : compsIn T = n1 T + 2 * n2 T
theorem primes_add_comps : primesIn T + compsIn T = 2 * T.card
theorem primes_eq        : primesIn T = n1 T + 2 * n0 T
theorem n0_eq_zero_iff   : n0 T = 0 ↔ ∀ k ∈ T, ¬ ((lo k).Prime ∧ (hi k).Prime)
theorem census_pinned (h0 : n0 T = 0) :
    n1 T = primesIn T ∧ n2 T = T.card - primesIn T
theorem census_pinned_add (h0 : n0 T = 0) : n2 T + primesIn T = T.card
theorem census_pinned_prefix (t) (hX : ∀ k < t, ¬ ((lo k).Prime ∧ (hi k).Prime)) :
    n1 (range t) = primesIn (range t) ∧ n2 (range t) = t - primesIn (range t)
```

### 2.5 proofs/Bridge.lean

Supply side (root partition over members) joined to demand side (slot census) -
the X-consistency equation's LHS skeleton, end to end.

```lean
def members (T : Finset ℕ) : Finset ℕ := T.image lo ∪ T.image hi

theorem card_members        : (members T).card = 2 * T.card
theorem card_comps_members  : ((members T).filter fun m => ¬ m.Prime).card = Census.compsIn T
theorem card_primes_members : ((members T).filter fun m => m.Prime).card = Census.primesIn T

theorem sum_roots_eq_census {y} (T) (hwin : ∀ k ∈ T, y < lo k ∧ hi k < y * y) :
    (∑ p ∈ (Finset.range y).filter Nat.Prime,
      ((members T).filter fun m => ¬ m.Prime ∧ m.minFac = p).card)
      = Census.n1 T + 2 * Census.n2 T

theorem sum_roots_pinned {y} (T) (hwin) (h0 : Census.n0 T = 0) :
    Σ_p R_p = Census.primesIn T + 2 * (T.card - Census.primesIn T)

theorem slot_roots_ne {k} (hk : 1 ≤ k) : (lo k).minFac ≠ (hi k).minFac
```

### 2.6 proofs/Gear.lean

Per-gear ledger line R, cap, onset (shadow law), and the semiprime refinement:
below q^3 one gear's line IS a prime count.

```lean
def R (q : ℕ) (S : Finset ℕ) : ℕ :=
  (S.filter fun m => ¬ m.Prime ∧ m.minFac = q).card

theorem supply_eq_sum_R (hS : window) :
    (S.filter fun m => ¬ m.Prime).card = ∑ p ∈ (range y).filter Nat.Prime, R p S
theorem sum_R_eq_census (hwin : slot window) :
    (∑ p ∈ (range y).filter Nat.Prime, R p (Bridge.members T))
      = Census.n1 T + 2 * Census.n2 T
theorem R_le_card_multiples : R q S ≤ (S.filter fun m => q ∣ m).card
theorem R_prefix_le (hq : 0 < q) :
    R q (Bridge.members (Finset.range t)) ≤ 6 * t / q + 2
theorem sq_le_of_minFac_eq (h1 : 1 < m) (hnp : ¬ m.Prime)
    (hfac : m.minFac = q) : q * q ≤ m
theorem R_eq_zero_of_below_sq (hS : ∀ m ∈ S, 1 < m ∧ m < q * q) : R q S = 0

theorem semiprime_of_fiber (hq : q.Prime) (h1 : 1 < m) (hnp : ¬ m.Prime)
    (hfac : m.minFac = q) (hcube : m < q * q * q) :
    ∃ c, c.Prime ∧ q ≤ c ∧ m = q * c

theorem not_prime_mul (hq : q.Prime) (hc : c.Prime) : ¬ (q * c).Prime
theorem minFac_mul (hq : q.Prime) (hc : c.Prime) (hqc : q ≤ c) :
    (q * c).minFac = q

def partners (q : ℕ) (S : Finset ℕ) : Finset ℕ :=
  (S.filter fun m => ¬ m.Prime ∧ m.minFac = q).image (· / q)

theorem R_eq_card_partners (q S) : R q S = (partners q S).card   -- unconditional
theorem mem_partners (hq : q.Prime) (hS : ∀ m ∈ S, 1 < m ∧ m < q * q * q) :
    c ∈ partners q S ↔ c.Prime ∧ q ≤ c ∧ q * c ∈ S
theorem window_bounds (hwin) (hy : 1 ≤ y) (hthin : y * y ≤ q * q * q) :
    ∀ m ∈ S, 1 < m ∧ m < q * q * q
```

Boundary: m = q^2 is rooted at q with partner q itself (hence `q <= c`);
`minFac 0 = 2`, so the `1 < m` guard is required or gear 2's fiber absorbs 0.

### 2.7 proofs/Placement.lean

Where each large-gear line sits. `slotOf m = (m+1)/6` covers both sign classes
with no case split; statements prefer `Ico 1 t` (slot 0 is degenerate).

```lean
theorem prime_mod_six (hp : p.Prime) (h5 : 5 ≤ p) : p % 6 = 1 ∨ p % 6 = 5
theorem sign_law (ha : a % 6 = 1 ∨ a % 6 = 5) (hb : b % 6 = 1 ∨ b % 6 = 5) :
    ((a * b) % 6 = 1 ↔ a % 6 = b % 6)
theorem unit_mul (ha) (hb) : (a * b) % 6 = 1 ∨ (a * b) % 6 = 5

def slotOf (m : ℕ) : ℕ := (m + 1) / 6
theorem lo_slotOf (hm : m % 6 = 5) : Census.lo (slotOf m) = m
theorem hi_slotOf (hm : m % 6 = 1) : Census.hi (slotOf m) = m
theorem mem_members_iff_slot (hm : m % 6 = 1 ∨ m % 6 = 5) :
    m ∈ Bridge.members T ↔ slotOf m ∈ T

theorem slot_injOn_partners (hq : q.Prime) (h5 : 5 ≤ q)
    (hS : ∀ m ∈ S, 1 < m ∧ m < q * q * q) :
    Set.InjOn (fun c => slotOf (q * c)) (Gear.partners q S)
theorem card_slots_of_line (hq) (h5) (hS) :
    ((Gear.partners q S).image fun c => slotOf (q * c)).card = Gear.R q S
theorem R_slots_eq (hq : q.Prime) (h5 : 5 ≤ q) (hcube : 6 * t ≤ q * q * q) :
    Gear.R q (Bridge.members (Finset.Ico 1 t))
      = ((Finset.range (6 * t)).filter fun c =>
          c.Prime ∧ q ≤ c ∧ slotOf (q * c) ∈ Finset.Ico 1 t).card
```

### 2.8 proofs/Corridor.lean

The (5,7) corridor: 32-cap, twin-product pin, endpoint/adjacency laws, packing
floor. Slot 1 IS the twin (5,7) - the unique class slot with both members prime,
hence the `k >= 2` guards. Cross-verified against
`research/topgap_endpoint_law.py`.

```lean
-- The 32-cap
theorem exists_class_in_run (a) :
    ∃ k, a ≤ k ∧ k < a + 33 ∧ (k % 35 = 1 ∨ k % 35 = 34)
theorem both_composite_of_class (hk : 2 ≤ k) (h : k % 35 = 1 ∨ k % 35 = 34) :
    ¬ (Census.lo k).Prime ∧ ¬ (Census.hi k).Prime
theorem both_composite_in_run (ha : 2 ≤ a) :
    ∃ k, a ≤ k ∧ k < a + 33 ∧ ¬ (lo k).Prime ∧ ¬ (hi k).Prime
theorem double_slot_in_run (ha : 2 ≤ a) :
    ∃ k, a ≤ k ∧ k < a + 33 ∧ Census.slotComps k = 2
theorem prime_adjacent_run_le (ha : 2 ≤ a)
    (hrun : ∀ k, a ≤ k → k < a + L → (lo k).Prime ∨ (hi k).Prime) : L ≤ 32

-- The pin unified (re-exports Polignac.twin_product_slot through Census.lo)
theorem product_slotOf (hu : 6 * u = p + 1) :
    Placement.slotOf (p * (p + 2)) = u * (p + 1)
theorem product_slotOf_sq (hu) : slotOf (p * (p + 2)) = 6 * (u * u)
theorem twin_product_pin (hu) :
    slotOf (p*(p+2)) = u*(p+1) ∧ Census.lo (u*(p+1)) = p*(p+2)
      ∧ p ∣ lo (u*(p+1)) ∧ (p+2) ∣ lo (u*(p+1))

-- Endpoint / adjacency laws
def Exposed (k) : Prop := ¬5∣lo k ∧ ¬5∣hi k ∧ ¬7∣lo k ∧ ¬7∣hi k
def exposedSet : Finset ℕ := {0,2,3,5,7,10,12,17,18,23,25,28,30,32,33}

theorem exposed_iff_mem (hk : 1 ≤ k) : Exposed k ↔ k % 35 ∈ exposedSet
theorem endpoint_law (ha : 1 ≤ a) (h1 : Exposed a) (h2 : Exposed (a+G)) :
    a % 35 ∈ exposedSet.filter fun r => (r + G) % 35 ∈ exposedSet
theorem endpoint_law_34 (hG : G % 35 = 34) ... :
    a % 35 = 3 ∨ a % 35 = 18 ∨ a % 35 = 33

def allowed3 (g1 g2) : Finset ℕ :=
  exposedSet.filter fun r => (r+g1)%35 ∈ exposedSet ∧ (r+g1+g2)%35 ∈ exposedSet
theorem adjacency_law ... : a % 35 ∈ allowed3 (g1 % 35) (g2 % 35)
theorem no_chain_of_forbidden (hf : allowed3 ... = ∅) ... : False
theorem forbidden_first_examples : allowed3 1 1 = ∅ ∧ ...
theorem forbidden_pairs_count :
    ((range 35 ×ˢ range 35).filter fun p => allowed3 p.1 p.2 = ∅).card = 294

theorem n2_packing (ha : 2 ≤ a) : W / 33 ≤ Census.n2 (Finset.Ico a (a + W))
```

`n2_packing` uses `Classical.choice` via `choose` (a `Nat.find` variant would be
choice-free). `forbidden_pairs_count` is kernel `decide` - no `ofReduceBool`.

### 2.9 proofs/Machine13.lean - the y=13 alpha1 certificate

Verified against `research/strata_adjacency.py` on all 5005 residues first.
Tiers A+B+C all closed (at fixed y the period scan subsumes B and C; tier A kept
separate: machine-free, scales). Shorthand:

```lean
theorem gap_le      ... : b - a <= 11          -- F_k(13) <= 11
theorem pair_sum_le ... : c - a <= 16          -- F2_k(13) <= 16
theorem gap11_realized   : openings 122,133 with nothing between  -- F  = 11
theorem pair16_realized  : openings 117,122,133 (gaps 5,11)       -- F2 = 16
theorem alpha1_certificate ... : 3 * (c - a) <= 3 * 11 + 1 * 17
theorem lemma1_at_13       ... : (c - a) - 11 <= 1 * 17
theorem tierA_forbidden : allowed3 of (6,11),(8,11),(11,6),(11,8),(11,11) = empty
theorem tierA_kills / no_11_11_chain : those chains cannot exist at all
```

### 2.10 proofs/MaxGap.lean - F = 0 mod 3

`uncovered_span_mod_three` (two distinct blocked classes mod 3 leave one, so any
two survivors are congruent), `F_zero_mod_three` (3 | M+1 = F),
`M_two_mod_three`, `not_max_of_mod_three` (pruning rule: a length not = 2 mod 3
can never be maximal). Search bookkeeping - maximality forcing both bounding
positions uncovered, gear 3 active - taken as hypotheses.

### 2.11 proofs/LiteralCap.lean - the twin literal cap

Verified against `research/literal_cap_gap_d.py` first (48 invertible classes mod
210, cap spectrum {2:24, 3:4, 4:14, 6:6}).

```lean
def sOf (c : ℕ) : ℕ := (if c % 6 = 1 then (c - 1) / 3 else (c + 1) / 3) % 35
def wpos (t s r ph i : ℕ) : ℕ :=
  (r + ((i + ph) / 2) * t + (if (i + ph) % 2 = 1 then s else 0)) % 35

theorem no_run_seven :          -- THE FINITE CHECK
    ∀ c < 210, Nat.gcd c 210 = 1 →
      ∀ r < 35, ∀ ph < 2, run7 (c % 35) (sOf c) r ph = false

theorem s_eq (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) :
    (2 * u) % 35 = sOf (q % 210)

theorem literal_chain_le_six    -- THE CAP
    (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) (hq : Nat.gcd q 210 = 1)
    (hph : ph < 2) (hr : 1 ≤ r)
    (hE : ∀ i < L, Corridor.Exposed (member r q u ph i)) : L ≤ 6

theorem cap_six_classes_sharp : -- SHARPNESS
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ hasRun6 c = true)
      = {37, 53, 83, 127, 157, 173}
```

At most 6 members, at every gear, NO bound on q'; 6 attained at exactly six
classes, so it cannot be lowered. Stated as "no class admits SEVEN consecutive
exposed walk members" - the sharpest form that stays linear.

### 2.12 proofs/LiteralCapTable.lean - the (A) word-list enumeration, CLOSED

Caps recomputed in the corridor frame and cross-checked against
`literal_cap_gap_d.py`'s 140-step max-run (48/48 classes, zero mismatches;
start-anchored = anywhere-in-walk because a shifted start is another (r,ph)
pair); every realized chain length in `research/data/fuel_census.csv` respects
its class cap, saturating at q' = 19 and 31.

```lean
def runL (t s r ph L : N) : Bool        -- run6/run7 at any length
def hasRunL (c L : N) : Bool
def capC (c : N) : N                    -- the explicit 48-class table

theorem cap_table_maximal :             -- upper: NO run of capC c + 1
    forall c < 210, Nat.gcd c 210 = 1 ->
      forall r < 35, forall ph < 2,
        runL (c % 35) (LiteralCap.sOf c) r ph (capC c + 1) = false
theorem cap_table_realized :            -- exact: a run of capC c EXISTS
    forall c < 210, Nat.gcd c 210 = 1 -> hasRunL c (capC c) = true

theorem literal_chain_le_capC {q u r ph L : N}
    (hu : 6 * u + 1 = q or 6 * u = q + 1) (hq : Nat.gcd q 210 = 1)
    (hph : ph < 2) (hr : 1 <= r)
    (hE : forall i < L, Corridor.Exposed (LiteralCap.member r q u ph i)) :
    L <= capC (q % 210)
theorem word_length_lt_capC ... : ell < capC (q % 210)  -- (A) in word form

theorem cap_two_classes   : ... = {11, 13, ..., 199}    -- 24 classes
theorem cap_three_classes : ... = {29, 59, 151, 181}
theorem cap_four_classes  : ... = {1, 23, 31, 61, 67, 89, 97, 113, 121,
                                   143, 149, 179, 187, 209}
theorem cap_six_classes   : ... = {37, 53, 83, 127, 157, 173}
theorem no_cap_five       : ... = empty  -- the spectrum {2,3,4,6} has a hole
theorem cap_spectrum_counts : cards 24 / 4 / 14 / 6
theorem hasRunL_mono, capC_le_six

theorem tripled_teeth_antipode   -- the T3 law, for 6u = q -+ 1:
    ... : {3u, q - 3u} = {(q-1)/2, (q+1)/2}   -- exact integers, every gear
```

The word list of R21/R26 - alternating words over `{2u', q'-2u'}`, two per
length, lengths `1 .. capC-1` - is COMPLETE as a kernel-checked function of
`q' mod 210` alone, exact in both directions; **(A) is FULLY kernel-checked.**
`tripled_teeth_antipode` (two-line omega proof) upgrades what lateral asserted
numerically to 100,000.

### 2.13 proofs/Machine17.lean

85085-tuple scan chunked into 34 explicit slice theorems assembled with
`interval_cases`. The 25 is tight (24 fails). Shorthand:

```lean
theorem gap_le      ... : b - a <= 18            -- F_k(17) = 18
theorem pair_sum_le ... : c - a <= 25            -- F2_k(17) = 25
theorem alpha1_certificate ... : 9 * (c - a) <= 9 * 18 + 4 * 19   -- 225 <= 238
theorem lemma1_at_17       ... : 3 * ((c - a) - 18) <= 4 * 19
```

### 2.14 proofs/TierA.lean - corridor law for chains of any length

`carrier` generalises `Corridor.allowed3` to any length; cost independent of the
machine - the piece that scales past the scans.

```lean
def offsets : List ℕ → List ℕ                    -- partial sums
def carrier (steps : List ℕ) : Finset ℕ          -- residues carrying the chain
theorem mem_carrier_of_chain : chain of openings → base residue in carrier
theorem no_chain_of_carrier_empty : carrier = ∅ → no such chain, anywhere
def flanked (F) (w) : List ℕ := F :: (w ++ [F])
theorem no_maximal_flanks : carrier (flanked F w) = ∅ → no both-maximal flanks
theorem padding_count_le / padding_at_most_one_below_onset

theorem no_adjacent_equal_padded (hc : carrier [q, q] = ∅) ... : False
theorem no_adjacent_padded_41 : carrier [41, 41] = ∅
theorem equal_padding_forbidden_classes :
    ((Finset.range 35).filter fun g => Nat.gcd g 35 = 1 ∧ carrier [g, g] = ∅)
      = {1, 4, 6, 9, 11, 16, 19, 24, 26, 29, 31, 34}
theorem equal_padding_forbidden_card : ... .card = 12
theorem padding_shape_dichotomy : ∀ g < 35, Nat.gcd g 35 = 1 →
    (carrier [g, g] = ∅ ↔
      carrier [g, (2*g) % 35] ≠ ∅ ∧ carrier [(2*g) % 35, g] ≠ ∅)

theorem onset_gate (hg : 0 < g) (hdvd : q ∣ g) (hF : g ≤ F) : q ≤ F
theorem padding_three_not_excluded : 13 * q ≤ 6 * F → 6 * (3*q) ≤ 6*F + 5*q
```

Flank steps closed by corridor arithmetic alone: 11->13 (w=(4),F=7), 13->17
(w=(6),F=11), 17->19 (w=(13),F=18), 23->29 (w=(19),F=34), 29->31 (w=(10),F=43).
`flanks_17_19` is sharp: each flank alone feasible mod 35, both together not.
**The honest exception is itself a theorem: `flanks_19_23_nonempty : carrier
(flanked 25 [8]) = {0, 5, 7, 12}`.** Carriers checked against
`research/flank_tierA_fix.py`. `onset_gate`: a padded link's interior gap is a
positive multiple of q' and is one of M's gaps, so q' <= F(M) - **padding cannot
exist below onset**. `padding_count_le` is `p <= F/q + 5/6`, a bound that GROWS;
`padding_three_not_excluded`: once F >= (13/6)q the budget stops excluding three
padded links (lateral's p = 3 at 41->43).

### 2.15 PolignacCap - the all-d literal cap

Files: `PolignacCapCore.lean` (defs + coprime lemma),
`PolignacCap{1,3,5,7,15,21,35,105}.lean` (one gcd class each), `PolignacCap.lean`
(root: `capOf`, `capOf_le_twelve`). Harvester's halved-coordinate frame: position
n is the pair (2n+1, 2n+1+2e) for d = 2e; gear q blocks n = 0, -e (mod q); gear 3
FILTERS the candidate list. The cap depends only on gcd(e,105), so eight theorems
cover EVERY even gap:

    gcd(e,105)    1    5    7    3   21   35   15  105
    cap           6    6    6    6    6    6   10   12

Each cap numerically sharp (scan fails at cap-1); all eight spectra reproduced
independently first; the twin row reproduces constructor's mod-35 table. **12 is
the absolute ceiling over all Polignac gaps**; gcd = 3 (densest gaps) still caps
at 6. |E_e| matches Hardy-Littlewood: prod over q in {3,5,7} of (q - r_q),
r_q = 1 if q | e else 2.

```lean
theorem exists_mul_mod_eq {n t : ℕ} (hn : 0 < n) (h : Nat.Coprime t n)
    {r : ℕ} (hr : r < n) : ∃ j, j < n ∧ (j * t) % n = r
```

Prerequisite of the single-cycle reduction (one orbit-length walk replaces the
whole start set - 37x, verified exact) - on the shelf, unused.

### 2.16 proofs/Spectrum.lean - the bridge identity and the qualifying spectrum

The load-bearing formal step of constructor's decomposition of (D). Nothing
empirical is assumed inside the file.

```lean
def windowSum (g : ℕ → ℕ) (a j : ℕ) : ℕ := ∑ i ∈ Finset.range j, g (a + i)
def SpectrumBound (g : ℕ → ℕ) (j Fj : ℕ) : Prop := ∀ a, windowSum g a j ≤ Fj

theorem merged_eq (g a l) :
    g a + windowSum g (a+1) l + g (a+l+1) = windowSum g a (l+2)
theorem merged_le_spectrum (h : SpectrumBound g (l+2) Fj) :
    g a + windowSum g (a+1) l + g (a+l+1) ≤ Fj
theorem merged_le_spectrum_succ (h : SpectrumBound g ((l+1)+1) Fk) : ...
theorem merged_le_of_shallow (hl : l + 2 ≤ 4)
    (h4 : SpectrumBound g 4 F4) (hflat : F4 ≤ F + q) :
    g a + windowSum g (a+1) l + g (a+l+1) ≤ F + q

-- the qualifying spectrum (mechanic's Q_j, research/qspec_table.py)
def Qualifying (g : N -> N) (u a j : N) : Prop :=
  forall i, 1 <= i -> i + 1 < j -> 2 * u <= g (a + i)
def QualBound (g : N -> N) (u j Qj : N) : Prop :=
  forall a, Qualifying g u a j -> windowSum g a j <= Qj

theorem qualifying_of_word (hw : forall i < l, 2 * u <= g (a + 1 + i)) :
    Qualifying g u a (l + 2)
theorem merged_le_qual (hQ : QualBound g u (l + 2) Qj) (hw) :
    g a + windowSum g (a + 1) l + g (a + l + 1) <= Qj
theorem merged_le_of_qual_flat (hQ) (hflat : Qj <= F + q) (hw) : merged <= F + q
theorem merged_le_of_qual_flat_all (Q : N -> N)
    (hQ : forall j, QualBound g u j (Q j)) (hflat : forall j, Q j <= F + q) :
    forall a l, (forall i < l, 2 * u <= g (a + 1 + i)) -> merged <= F + q
theorem merged_le_of_corrected            -- R31's two-part lambda form
    (hflat : Fj <= F + q + lam * l * L)
    (hsupp : forall b, Qualifying g u b (l+2) ->
               windowSum g b (l+2) + lam*l*L <= Fj)
    (hw) : merged <= F + q
theorem alphabet_ge_floor : 2 * u <= q - 2 * u  -- both literal letters
theorem padded_ge_floor   : 2 * u <= q          -- and padded letters qualify
```

`merged_eq`: a word of l gaps plus its two flanks spans exactly l+2 = k+1
CONSECUTIVE gaps, so merged length is a window sum bounded by the spectrum value.
`merged_le_of_qual_flat_all` is the word-free criterion: `Q_j <= F + q'` at every
depth gives (D) for every floor-respecting word of every length - no k_win, no
fuel, no word list in the statement; `Q_j = 0` discharges deep depths free.

### 2.17 proofs/MergeLaw.lean - R39 as a two-machine kernel statement

Constructor's exact qualmax criterion, abstract in the machine: `pos` = the old
machine's opening enumeration, `kap` = the kill predicate on opening indices,
teeth `{u, q-u}`. Consumers are `SpectrumBound`/`QualBound` instances the
per-machine scans provide; nothing empirical inside.

```lean
theorem sub_mod_eq (hxy : x <= y) (hal : al < q) (hbe : be < q)
    (hx : x % q = al) (hy : y % q = be) : (y - x) % q = (q + be - al) % q
def MergedWindow (kap : N -> Prop) (a j : N) : Prop :=
  0 < j and not kap a and not kap (a + j) and forall i, 0 < i -> i < j -> kap (a + i)
theorem interior_gap_mod ... :        -- RESIDUE NECESSITY
    g (a+i) % q = 0 or g (a+i) % q = 2*u or g (a+i) % q = q - 2*u
theorem floor_of_mod (hG : 0 < G) (h4u : 4*u <= q) (h : ...) : 2*u <= G
theorem newgap_le            -- THE CORE: merged window sum <= B whenever
    (hF2 : SpectrumBound g 2 F2) (hF2B : F2 <= B)      -- F2 <= B and
    (hQ : forall j, 3 <= j -> QualBound g u j (Q j)) (hQB : forall j, Q j <= B)
    (hmw : MergedWindow kap a j) : Spectrum.windowSum g a j <= B
theorem newgap_le_max ... : windowSum g a j <= max F2 Qmax   -- R39 verbatim
theorem D_of_qualmax  ... : windowSum g a j <= F + qp        -- (D) form

-- the per-step bookkeeping factored out once: a rung is a 15-line instantiation
theorem pos_le_add (hmono : forall m, pos m <= pos (m+1)) (a j) : pos a <= pos (a + j)
theorem windowSum_telescope (hg) (hmono) (a j) :
    Spectrum.windowSum g a j = pos (a + j) - pos a
theorem newgap_le_step {ExO ExN Kap : N -> Prop} {posO posN g : N -> N}
    (hg) (hOmono) (hOpos) (hOex) (hOsurj)          -- old machine's enumeration
    (hNpos) (hNmono) (hNex) (hNempty)              -- new machine's enumeration
    (hnk) (hkn) (hsub)                             -- the step relation
    (hteeth : forall x, Kap x -> x % q = u or x % q = q - u)
    (hu : 0 < u) (h4u : 4 * u <= q)
    (hF2 : Spectrum.SpectrumBound g 2 F2) (hF2B : F2 <= B)
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g u j (Q j)) (hQB : forall j, Q j <= B)
    (n : N) : posN (n + 1) - posN n <= B
```

### 2.18 proofs/TwoTeeth.lean - the kill-spacing law, T1-T5

Constructor's `docs/novel/two-teeth-kill-spacing.md` T1-T5. Verified numerically
first for every prime gear 5..199 (`check_two_teeth.py`, zero mismatches).

```lean
def Kill (q u x : N) : Prop := x % q = u or x % q = q - u
theorem next_kill_of_lo ... : y - x = q - 2*u and y % q = q - u  -- exact form
theorem next_kill_of_hi ... : y - x = 2*u and y % q = u
theorem kill_spacing    ... : y - x = 2*u or y - x = q - 2*u   -- {2u', q'-2u'}
theorem kill_spacing_min .. : 2*u <= y - x                     -- min ~ q'/3
theorem kill_period     ... : z - x = q       -- alternation: pairs sum to q'
theorem gear_side (hu6 : 6*u+1 = q or 6*u = q+1) (hq : 5 <= q) : 0 < u and 4*u < q
theorem teeth_letters   ... : 2*u + (q - 2*u) = q and ((6*u)%q = q-1 or = 1) -- T1
theorem spacing_from_lo ... : ((y-x)%q = 0 and y%q = u) or
    ((y-x)%q = q-2*u and y%q = q-u)           -- T2+T3, padding-transparent
theorem spacing_from_hi ... : (stay) or ((y-x)%q = 2*u and y%q = u)
theorem kills_gap_ge    ... : 2*u <= y - x    -- T4 general: ANY two kills
theorem fuel_span_cap   ... : 2*u*(k-1) <= x (k-1) - x 0       -- T5
theorem fuel_le         ... : k <= 1 + (x (k-1) - x 0) / (2*u) -- ~3L/q'
```

### 2.19 Machine11 / Machine13Q / Machine17Q / Machine19 / Machine19Q

Each is the `seekT`-walk recipe at a new machine: one walk per opening reads the
whole ladder plus the depth refutation; `seek_next` proves the walk computes
`nextOp` exactly (extraction needs no witness pigeonhole - the chain IS the
consecutive openings); `opSeq_surj` completes the enumeration, which is what lets
MergeLaw be instantiated on a real machine.

```lean
-- the shared enumeration development (Machine19 shown; identical at each machine)
theorem exists_exposed_above (k) : exists m, k < m and Exposed19 m
def nextOp (k) := Nat.find (exists_exposed_above k)
def opSeq : N -> N                       -- the openings in increasing order
def g19 (n) := opSeq (n + 1) - opSeq n   -- the gap word
theorem windowSum_g19 :
    Spectrum.windowSum g19 a j = opSeq (a + j) - opSeq a
theorem seek_next (hx : 1 <= x) (hE : Exposed19 (x + s)) :
    x + seekT (x%5) (x%7) (x%11) (x%13) (x%17) (x%19) 25 s = nextOp (x + s)
theorem opSeq_surj (hm : 1 <= m) (hE : Exposed19 m) : exists n, opSeq n = m

-- Machine19.lean (+ Machine19Core, Machine19S0..S16: 323 slices of 5005 tuples)
theorem sliceAll : forall e < 17, forall f < 19, slice e f = true  -- [propext] ONLY
theorem gap_le      ... : b - a <= 25     -- F_k(19)  = 25
theorem pair_sum_le ... : c - a <= 31     -- F2_k(19) = 31
theorem quad_sum_le ... : e - a <= 38     -- F4_k(19) = 38
theorem alpha1_certificate ... : 9 * (c - a) <= 9 * 25 + 4 * 23  -- 279 <= 317
theorem lemma1_at_19       ... : 3 * ((c - a) - 25) <= 4 * 23
theorem shallow_flatness   ... : e - a <= 25 + 23  -- F_4 <= F+q' (38 <= 48)
theorem spectrum_four      : Spectrum.SpectrumBound g19 4 38
theorem spectrum_four_flat : Spectrum.SpectrumBound g19 4 (25 + 23)
theorem D_of_shallow_word {a l : N} (hl : l + 2 <= 4) :
    g19 a + Spectrum.windowSum g19 (a + 1) l + g19 (a + l + 1) <= 25 + 23

-- Machine19Q (Machine19QCore + Machine19QS0..16, 323 slices): EVERY depth
theorem qsliceAll : ...                                    -- [propext] ONLY
theorem chain_facts (n : N) :
    opSeq (n+3) - opSeq n <= 35 and opSeq (n+5) - opSeq n <= 47 and
      not (8 <= g19 n and 8 <= g19 (n+1) and 8 <= g19 (n+2) and 8 <= g19 (n+3))
theorem no_big_run (n) : not (four consecutive g19 gaps all >= 8)  -- Q_6 = 0
theorem spectrum_ladder : F_1..F_5 <= 25, 31, 35, 38, 47 over g19
theorem qual_bound_all : forall j, 3 <= j -> Spectrum.QualBound g19 4 j 47
theorem qual_five_flat : Spectrum.QualBound g19 4 5 (25 + 23)
theorem D_of_word {a l : N} (hw : forall i < l, 8 <= g19 (a + 1 + i)) :
    g19 a + Spectrum.windowSum g19 (a+1) l + g19 (a+l+1) <= 25 + 23

-- Machine11 (gears {5,7,11}, period 385; ONE 385-tuple kernel check, NO AXIOMS)
theorem qasm : qslice = true
theorem chain_facts (n) : opSeq (n+1) - opSeq n <= 7 and opSeq (n+2) - opSeq n <= 11
  and opSeq (n+3) - opSeq n <= 16 and opSeq (n+4) - opSeq n <= 18
  and ((4 <= g11 (n+1) and 4 <= g11 (n+2) and 4 <= g11 (n+3)) -> opSeq (n+5) - opSeq n <= 20)
  and not (4 <= g11 n and 4 <= g11 (n+1) and 4 <= g11 (n+2) and 4 <= g11 (n+3))
theorem spectrum_ladder : F_1..F_4 <= 7, 11, 16, 18   -- over g11
theorem qual_bound_all : forall j, 3 <= j -> Spectrum.QualBound g11 2 j 20

-- Machine13Q (period 5005; ONE 5005-tuple kernel check, NO AXIOMS)
theorem qasm : qslice = true
theorem chain_facts (n) : opSeq (n+1) - opSeq n <= 11 and opSeq (n+3) - opSeq n <= 23
  and opSeq (n+4) - opSeq n <= 26 and not (6 <= g13 n and 6 <= g13 (n+1) and 6 <= g13 (n+2))
theorem spectrum_ladder : F_1..F_4 <= 11, 16, 23, 26  -- F_2 = round-11 certificate
theorem qual_bound_all : forall j, 3 <= j -> Spectrum.QualBound g13 3 j 26

-- Machine17Q (period 85085; 17 slices of 5005)
theorem qsliceAll : forall e < 17, qslice e = true   -- [propext, Quot.sound]
theorem chain_facts (n) : ... F_1 <= 18, F_3 <= 28, F_4 <= 33, F_5 <= 35,
  the qualifying depth-6 bound Q_6 <= 34, and no five consecutive gaps all >= 6
theorem spectrum_ladder : F_1..F_5 <= 18, 25, 28, 33, 35
theorem qual_bound_all : forall j, 3 <= j -> Spectrum.QualBound g17 3 j 35
```

`D_of_word` is (D) at alpha = 3 at machine 19 for EVERY word length - r20's
shallowness hypothesis GONE (depths 2-5 flat under 48 by the kernel ladder;
depths >= 6 empty by `no_big_run`).

Full-period ladders behind these, all independently measured in Python first:

    machine   F_1..F_8                             Q_j(floor)              max run
    11        7, 11, 16, 18, 23, 26, 28, 30        16, 18, 20, 0 (fl 4)    3
    13        11, 16, 23, 26, 28, 31, 34, 38       18, 23, 0     (fl 6)    2
    17        18, 25, 28, 33, 35, 40, 43, 48       28, 31, 32, 34, 0 (6)   4
    19        25, 31, 35, 38, 47, ...              35, 37, 38, 0 (fl 8)    3
    23        34, 39, 50, 58, 65, 77, 83, 88       43, 50, 55, 60, 0 (10)  4

**Where the qualifying restriction earns its keep**, per machine: at m13 the
unconditional ladder already clears the budget at both live depths (F_3 = 23,
F_4 = 26 <= 28) and qualifying only kills j >= 5; at m11 it first bites at depth
5 (F_5 = 23 > 20, Q_5 = 20); at m17 at depth 6 (F_6 = 40 > 37, Q_6 = 34). **The
criterion is NOT a uniform improvement - it is a one-or-two-depth patch on the
unconditional spectrum, and the patched depth moves UP with the machine.**

### 2.20 proofs/Machine23.lean + Machine23Q.lean + Machine29.lean

```lean
-- Machine23.lean: (D) at 19->23, END TO END, NO HYPOTHESES
def Killed23 (k : N) : Prop := k % 23 = 4 or k % 23 = 19   -- teeth {u', 23-u'}
def Exposed23 (k : N) : Prop := Exposed19 k and not(23 | lo k) and not(23 | hi k)
def g23 (n : N) : N := opSeq23 (n+1) - opSeq23 n    -- machine 23's own gaps
theorem merge_alphabet (hk1 : Killed23 x) (hk2 : Killed23 y) (hxy : x < y)
    (hle : y - x <= 25) : y - x = 8 or y - x = 15 or y - x = 23
theorem g23_le (n : N) : g23 n <= 47
theorem D_at_19_23 (n : N) : g23 n <= 25 + 23

-- Machine23Q.lean: machine 23's enumeration is complete (no new scan)
theorem opSeq23_strict_mono {a b : N} (h : a < b) : opSeq23 a < opSeq23 b
theorem windowSum_g23 (a j : N) :
    Spectrum.windowSum g23 a j = opSeq23 (a + j) - opSeq23 a
theorem opSeq23_surj {m : N} (hm : 1 <= m) (hE : Exposed23 m) : exists n, opSeq23 n = m

-- Machine29.lean: gear 29's teeth are {5, 24}  (6 * 5 = 30 = 29 + 1)
def Killed29 (k : N) : Prop := k % 29 = 5 or k % 29 = 24
def Exposed29 (k : N) : Prop :=
  Exposed23 k and not (29 | Census.lo k) and not (29 | Census.hi k)
def g29 (n : N) : N := opSeq29 (n + 1) - opSeq29 n
theorem killed29_iff {k : N} (hk : 1 <= k) :
    Killed29 k <-> (29 | Census.lo k or 29 | Census.hi k)
theorem merge_alphabet {x y : N} (hk1 : Killed29 x) (hk2 : Killed29 y)
    (hxy : x < y) (hle : y - x <= 34) : y - x = 10 or y - x = 19 or y - x = 29
theorem g29_le (hF2 : Spectrum.SpectrumBound g23 2 39)
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g23 5 j 60) (n : N) : g29 n <= 60
theorem D_at_23_29 (hF2 : Spectrum.SpectrumBound g23 2 39)
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g23 5 j 60) (n : N) : g29 n <= 34 + 29
```

`g23_le` = `MergeLaw.newgap_le` instantiated with m19's kernel bounds through
`opSeq_surj`. **No m23 period scan was needed for `D_at_19_23` - the merge law
replaced a 37.2M scan.** Machine 29's enumeration is discharged at period
`1,078,282,205 = 37,182,145 * 29`, so the 23->29 rung reduced to exactly
`F_2(23) <= 39` and `Q_j(23;10) <= 60`, both discharged in 2.21. Merge letters
{10, 19, 29}, all at or above the floor 2u'' = 10.

### 2.21 Machine23Idx.lean + Machine23IdxS0..S16.lean + Machine23Scan.lean

The fifth rung, hypothesis-free: 323 kernel decisions, position-indexed encoding.

```lean
-- Machine23Scan.lean (namespace Machine23)
theorem qsliceIdxAll : forall e < 17, forall f < 19, qsliceIdx e f = true
theorem qokIdxAll {a b c d e f g : N} (ha : a < 5) (hb : b < 7) (hc : c < 11)
    (hd : d < 13) (he : e < 17) (hf : f < 19) (hg : g < 23)
    (hop : Machine19.atT a b c d e f 0 = true) :
    qokIdx a b c d e f g = true
theorem W_eq {x m : N} (hx : 1 <= x) (hm : Machine19.opSeq m = x) :
    forall k, x + W x k = Machine19.opSeq (m + k)
theorem next23_step {x m k : N} (hx : 1 <= x) (hm : Machine19.opSeq m = x)
    (hp : NS x 5 k <= k + 5) :
    nextOp23 (x + W x k) = x + W x (NS x 5 k)
theorem chain_facts23 (n : N) :
    (opSeq23 (n + 1) - opSeq23 n <= 34) and
    (opSeq23 (n + 2) - opSeq23 n <= 39) and
    (10 <= g23 (n + 1) -> opSeq23 (n + 3) - opSeq23 n <= 60) and
    (10 <= g23 (n + 1) -> 10 <= g23 (n + 2) -> opSeq23 (n + 4) - opSeq23 n <= 60) and
    (10 <= g23 (n + 1) -> 10 <= g23 (n + 2) -> 10 <= g23 (n + 3) ->
      opSeq23 (n + 5) - opSeq23 n <= 60) and
    (10 <= g23 (n + 1) -> 10 <= g23 (n + 2) -> 10 <= g23 (n + 3) -> 10 <= g23 (n + 4) ->
      opSeq23 (n + 6) - opSeq23 n <= 60) and
    not (10 <= g23 n and 10 <= g23 (n + 1) and 10 <= g23 (n + 2) and 10 <= g23 (n + 3) and
        10 <= g23 (n + 4))
theorem spectrum23_one : Spectrum.SpectrumBound g23 1 34          -- F(23) <= 34
theorem spectrum23_two : Spectrum.SpectrumBound g23 2 39          -- F_2(23) <= 39
theorem qual23_all : forall j, 3 <= j -> Spectrum.QualBound g23 5 j 60
theorem D_23_29 (n : N) : Machine29.g29 n <= 34 + 29   -- (D) at 23->29, NO HYPOTHESES
theorem g29_le_60 (n : N) : Machine29.g29 n <= 60      -- R39's form, margin 3
```

`qual23_all`'s u = 5 is gear 29's floor 2u'' = 10, matching `chain_facts23`'s
guards clause for clause. **The scan covers the full period exactly**: 17 modules
x 19 slices = 323 slices; x 5,005 m19 tuples = 1,616,615 CRT tuples; x 23 phases =
37,182,145 = 5*7*11*13*17*19*23 = machine 23's period, each residue once; openings
3*5*9*11*15*17*21 = 7,952,175, chain Bool true at every one (simulated first in
numpy and as a literal transcription of the Lean defs). **The scan certifies its
own fuel**: each clause is preceded by `p_i <= p_(i-1) + 5`; at most 4 m19
positions separate consecutive m23 openings, so fuel 5 never hits the sentinel,
and the values read off are `F_1..F_6(23) = 34, 39, 50, 58, 65, 77`. Bridge: m23's
openings are m19's off gear 23's teeth; `next23_step` is the whole content.

### 2.22 proofs/Ladder.lean - THE (D) LADDER

Every conjunct is a theorem about that machine's OWN gap sequence, no hypotheses.

```lean
theorem D_at_11_13 (n : N) : Machine13.g13 n <= 7 + 13
theorem D_at_13_17 (n : N) : Machine17.g17 n <= 11 + 17
theorem D_at_17_19 (n : N) : Machine19.g19 n <= 18 + 19
theorem D_ladder :
    (forall n, Machine13.g13 n <= 7 + 13) and (forall n, Machine17.g17 n <= 11 + 17) and
      (forall n, Machine19.g19 n <= 18 + 19) and (forall n, Machine23.g23 n <= 25 + 23)
-- R39's own form, per rung (the value the criterion actually produces):
theorem g13_le       (n : N) : Machine13.g13 n <= 20   -- max(F2=11, maxQ=20)
theorem g17_le       (n : N) : Machine17.g17 n <= 26   -- max(F2=16, maxQ=26)
theorem g19_le_of_17 (n : N) : Machine19.g19 n <= 35   -- max(F2=25, maxQ=35)
theorem criterion_arith : max 39 60 <= 34 + 29 and max 90 91 <= 88 + 41  -- no axioms

-- hypothesis-explicit instantiations above the scannable range
theorem D_at_23_29 ... (hteeth : forall x, Kap x -> x % 29 = 5 or x % 29 = 24)
    (hF2 : Spectrum.SpectrumBound g 2 39)                   -- F_2(23) = 39
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g 5 j (Q j)) (hQm : forall j, Q j <= 60)
    (n : N) : posN (n + 1) - posN n <= 34 + 29
theorem D_at_37_41 ... (hteeth : forall x, Kap x -> x % 41 = 7 or x % 41 = 34)
    (hF2 : Spectrum.SpectrumBound g 2 90)                   -- F_2(37) = 90
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g 7 j (Q j)) (hQm : forall j, Q j <= 91)
    (n : N) : posN (n + 1) - posN n <= 88 + 41
```

The five hypothesis-free rungs: `D_at_11_13`, `D_at_13_17`, `D_at_17_19`,
`Machine23.D_at_19_23`, `Machine23.D_23_29`.

    step     criterion max(F2, max_j Q_j)   budget F+q'   margin   floor 2u'
    11->13   max(11, 20) = 20               20             0 TIGHT   4
    13->17   max(16, 26) = 26               28             2         6
    17->19   max(25, 35) = 35               37             2         6
    19->23   max(31, 47) = 47               48             1         8
    23->29   max(39, 60) = 60               63             3        10

11->13 is EXACTLY tight: `Q_5(11; 4) = 20 = F(11) + 13`. `g19_le_of_17` derives an
m19 gap bound from m17's period scan ALONE (35, vs the sharp 25 m19's own scan
gives) - **the point of a rung is that the merge law reaches the next machine
without seeing it.** Independent confirmation of the corrected C13 row: m23's
spectra were re-derived here over the full period before writing `D_at_23_29` -
**F(23) = 34, F_2(23) = 39, Q_j(23;10) = 43, 50, 55, 60, 0 for j = 3..7, longest
run of gaps >= 10 is 4**; mechanic's corrected row reproduces EXACTLY, the
pre-2026-08-24 row 50/50/49/0/0 is confirmed wrong.

### 2.23 proofs/DepthSum.lean - the depth-sum identity at machine 13

Both halves of `sum_j W_j(g) = prod_q c_q(g)`, kernel-checked; the glue is not
(verdict 11). Verified first over the full period for g = 0..59 (`depthsum13.py`).

```lean
theorem window_depth_unique (hg) (hmono : forall m, pos m < pos (m+1))
    (h1 : Spectrum.windowSum g a j1 = gap) (h2 : Spectrum.windowSum g a j2 = gap) :
    j1 = j2
def reachSet (g) (gap N J) : Finset N
theorem mem_reachSet : a in reachSet g gap N J <->
    a < N and exists j, 1 <= j and j < J and Spectrum.windowSum g a j = gap
theorem depth_partition (hg) (hmono) (gap N J) :
    sum over j in Finset.Ico 1 J of
        ((Finset.range N).filter fun a => Spectrum.windowSum g a j = gap).card
      = (reachSet g gap N J).card
-- the RHS at machine 13, over the whole period (all of these: NO AXIOMS)
theorem local_factor_5  : forall gap < 40, cq 5 1 4 gap   = 5 - nuq 5 gap
theorem local_factor_7  : forall gap < 40, cq 7 6 1 gap   = 7 - nuq 7 gap
theorem local_factor_11 : forall gap < 40, cq 11 2 9 gap  = 11 - nuq 11 gap
theorem local_factor_13 : forall gap < 40, cq 13 11 2 gap = 13 - nuq 13 gap
theorem depth_sum_at_13 : forall gap < 40,
    pairCount13 gap = cq 5 1 4 gap * cq 7 6 1 gap * cq 11 2 9 gap * cq 13 11 2 gap
theorem depth_sum_hl_form : forall gap < 40, pairCount13 gap
    = (5 - nuq 5 gap) * (7 - nuq 7 gap) * (11 - nuq 11 gap) * (13 - nuq 13 gap)
```

`window_depth_unique` IS lateral's bijection ("every opening pair at lag g is the
endpoint pair of exactly one window"), abstract - no machine, no arithmetic, just
strict monotonicity. `local_factor_*` is harvester's
`c_q(g) = q - nu_q({0, 2, 6g, 6g+2})`: **the machine's transfer diagonal IS the
Hardy-Littlewood prime-quadruplet local factor**, at four gears.

### 2.24 proofs/Potential.lean + Potential19.lean + PotentialLadder.lean

(D) with no depth quantifier. From constructor's R46 Kleene generator
(`docs/novel/kleene-generator.md`): `F(M+q') = L^T (x) K* (x) R` has as corollary
that (D) holds IFF a POTENTIAL `h` exists satisfying three ONE-STEP, ONE-OPENING
inequalities. Kernel-checked is the direction that does proof work - **a
potential CERTIFIES the bound, at every chain length, by one induction.**

```lean
def IsPotential {St : Type*} (Step : St -> St -> Prop) (d e h : St -> N) (B : N) : Prop :=
  (forall x, d x <= h x) and (forall x y, Step x y -> d x + h y <= h x)
    and (forall x, e x + h x <= B)
theorem chain_le_potential (hC1) (hC2) :
    forall (l : N) (p : N -> St), (forall k, k < l -> Step (p k) (p (k+1))) ->
      sum over k in Finset.range (l+1) of d (p k) <= h (p 0)
theorem D_of_potential (hP : IsPotential Step d e h B) (l) (p) (hstep) :
    e (p 0) + sum over k in Finset.range (l+1) of d (p k) <= B
theorem windowSum_succ_left (g b l) :
    Spectrum.windowSum g b (l+1) = g b + Spectrum.windowSum g (b+1) l
theorem tail_le_potential (hC1) (hC2) : forall l b, (forall i < l, 2*u <= g (b+i)) ->
    Spectrum.windowSum g b l + g (b+l) <= h b
theorem merged_le_of_potential {g h u F q}
    (hC1 : forall i, g i <= h i)
    (hC2 : forall i, 2*u <= g i -> g i + h (i+1) <= h i)
    (hC3 : forall i, g i + h (i+1) <= F + q)
    {a l} (hw : forall i < l, 2*u <= g (a+1+i)) :
    g a + Spectrum.windowSum g (a+1) l + g (a+l+1) <= F + q
```

`merged_le_of_potential`'s hypotheses contain NO quantifier over `l`; its
conclusion holds for every `l`. The state type stays general because
constructor's states are `(opening, tooth)` pairs, not indices. Exhibited
certificates - `Potential19.lean` (h19), `PotentialLadder.lean` (h11, h13, h17):

```lean
def h19 (i : N) : N :=            -- the qualifying tail from i
  if 8 <= g19 i then
    (if 8 <= g19 (i+1) then
      (if 8 <= g19 (i+2) then g19 i + g19 (i+1) + g19 (i+2) + g19 (i+3)
        else g19 i + g19 (i+1) + g19 (i+2))
      else g19 i + g19 (i+1))
    else g19 i
theorem h19_C1 (i) : g19 i <= h19 i
theorem h19_C2 (i) (hq : 8 <= g19 i) : g19 i + h19 (i+1) <= h19 i
theorem h19_C3 (i) : g19 i + h19 (i+1) <= 25 + 23
theorem D_of_word_potential {a l} (hw : forall i < l, 8 <= g19 (a+1+i)) :
    g19 a + Spectrum.windowSum g19 (a+1) l + g19 (a+l+1) <= 25 + 23

def h11 (i : N) : N   -- machine 11's qualifying tail, floor 4, unfolds <= 4
def h13 (i : N) : N   -- machine 13's, floor 6, unfolds <= 3
def h17 (i : N) : N   -- machine 17's, floor 6, unfolds <= 5
theorem h11_C1 / h11_C2 / h11_C3, h13_C1/C2/C3, h17_C1/C2/C3
theorem D_of_word_11 {a l : N} (hw : forall i < l, 4 <= Machine11.g11 (a+1+i)) :
    Machine11.g11 a + Spectrum.windowSum Machine11.g11 (a+1) l
      + Machine11.g11 (a+l+1) <= 7 + 13
theorem D_of_word_13 ... <= 11 + 17
theorem D_of_word_17 ... <= 18 + 19
theorem potential_ladder : the three, collected
```

    rung      potential   floor 2u'   tail depth   budget F + q'
    11 -> 13  h11             4           4          7 + 13 = 20
    13 -> 17  h13             6           3         11 + 17 = 28
    17 -> 19  h17             6           5         18 + 19 = 37
    19 -> 23  h19             8           4         25 + 23 = 48

**(C2) holds with EQUALITY in every branch at every machine, and its deepest
branch is always exactly that machine's `no_big_run`** (`Q_J = 0`: floor gaps in
a row force the next not to qualify, so the tail terminates); **(C3)'s cases are
always that machine's own spectrum ladder**, with the deepest case at m11 and m17
supplied by the CONDITIONAL rung of `chain_facts` (`Q_5(11) <= 20`, `Q_6(17) <=
34`) rather than an unconditional `F_j` - where the qualifying restriction earns
its keep. **The recipe generalises**: at any machine whose qualifying runs are
bounded - what `Q_J = 0` says, and every machine scanned has such a J (11: J=6,
13: J=5, 17: J=7, 19: J=6, 23: J=7) - the TAIL FUNCTION unfolded to depth J-2 IS
a potential. **Tail depths do NOT grow with the machine: 4, 3, 5, 4.** Each is a
separate finite object; a potential valid at every machine at once is still NOT
known, which is what constructor's negative at 29->31 is about.

### 2.25 proofs/CoveringCert.lean - a (D) rung proved a second way, scan-free

`F(19) <= 37 = F(17) + 19` from THIRTY-SEVEN INTEGERS, no period of machine 19
built, nothing shared with the merge-law route.

```lean
def ywList : List N := [115, 169, 265, ..., 265, 169, 115]     -- 37 weights
def totY : N := sum over i in Finset.range 37 of yw i
def S (bq : N -> N -> Bool) (r : N) : N                        -- single-gear sum
def PP (bq : N -> N -> Bool) (r5 rq : N) : N                   -- pair sum
theorem tot_eq : totY = 9757                                   -- decide +kernel
theorem S5_le  : forall r < 5,  S b5 r <= 3905                 -- six of these
theorem S19_le : forall r < 19, S b19 r <= 1115
theorem P7_ge  : forall a < 5, forall b < 7, 1101 <= PP b7 a b -- five of these
theorem P19_ge : forall a < 5, forall b < 19, 272 <= PP b19 a b
theorem cert_signs :
    3905+2796+1821+1648+1204+1115 < 9757 + (1101+552+548+276+272)  -- NO AXIOMS
theorem kounias (a b c d e f : Bool) (h : (a || b || c || d || e || f) = true)
    (w : N) : w + (the five (if a && x then w else 0))
                <= (the six (if x then w else 0))
theorem cover_bound (hcov : forall i < 37, the six-way or at i) :
    totY + (PP b7 r5 r7 + ... + PP b19 r5 r19)
      <= S b5 r5 + S b7 r7 + S b11 r11 + S b13 r13 + S b17 r17 + S b19 r19
theorem no_cover (h5 : r5 < 5) ... (hcov) : False
theorem no_37_run {p : N} (hp : 1 <= p) : exists i < 37, Machine19.Exposed19 (p + i)
theorem F19_le_37 (n : N) : Machine19.g19 n <= 37
theorem D_17_19_lp (n : N) : Machine19.g19 n <= 18 + 19
```

`no_37_run` and `F19_le_37` depend on NOTHING but the certificate arithmetic and
`Machine19.exposed19_iff`: no slice, no `sliceAll`, no merge law, no `Spectrum`.
**So 17->19 has TWO kernel proofs whose only common ancestor is the definition of
the machine.** Three facts that appeared only on formalising: (i) **supported on
ONE distinguished gear** - all 37 weights on rows `(i, 5)`, Kounias with `k = 5`
everywhere (the 222-row 7-pair LP optimum uses 37 rows, 5 pairs), which is what
keeps the Lean statement small; (ii) **a palindrome**, `y_i = y_{36-i}` - the
machine's mirror symmetry in the dual; (iii) over denominator 1101 it reads
`12489 < 9757 + 2749`, margin 0.14%. Cost: 11 `decide +kernel`, SECONDS, against a
period scan costing hours. **The only upper bound on a Jacobsthal-type maximal gap
here that a kernel checks without enumerating a period.**

### 2.26 proofs/CoveringCert2.lean - consistent certificates at 11->13, 13->17

The LP-duality thread's finding: round 22's relaxation - and the whole classical
Bonferroni/Kounias family - drops MARGINAL CONSISTENCY, and restoring it at degree
2 closes 11->13 and 13->17, which no amount of DEGREE does (at m13 the
inconsistent relaxation is feasible at degree 2, 3 AND 4). **Where the
inconsistency is here**: `CoveringCert.cover_bound` bounds each block separately,
letting gear 5 use one phase in `S_5` and another in the pair minima. **The fix
keeps the phases under one quantifier**: `sum y <= max over PHASE TUPLES of
[ S_5(r5) + sum_j (S_j(rj) - P_j(r5,rj)) ]` - finite, decidable, strictly
stronger; the `k = 5` STAR case of marginal consistency (only gear 5's marginal is
tied, since every row uses gear 5 as its distinguished event). **No dual
multiplier for a consistency equation ever appears.**

```lean
theorem cert13 : forall a < 5, forall b < 7, forall c < 11, forall d < 13,
    S13 b5 a + (S13 b7 b - P13 b7 a b) + (S13 b11 c - P13 b11 a c)
      + (S13 b13 d - P13 b13 a d) < 22                      -- decide +kernel
theorem cover13 (hcov : forall i < 20, the four-way or at i) :
    T13 + (P13 b7 r5 r7 + P13 b11 r5 r11 + P13 b13 r5 r13)
      <= S13 b5 r5 + S13 b7 r7 + S13 b11 r11 + S13 b13 r13
theorem no_20_run {p : N} (hp : 1 <= p) : exists i < 20, Machine13.Exposed13 (p + i)
theorem F13_le_20 (n : N) : Machine13.g13 n <= 20
theorem D_11_13_lp (n : N) : Machine13.g13 n <= 7 + 13
theorem cert17 : forall a < 5, forall b < 7, forall c < 11, forall d < 13,
    forall e < 17, ... < 94                                 -- decide +kernel
theorem F17_le_28 (n : N) : Machine17.g17 n <= 28
theorem D_13_17_lp (n : N) : Machine17.g17 n <= 11 + 17
theorem lp_ladder :   -- three consecutive rungs, one vehicle
    (forall n, Machine13.g13 n <= 7 + 13) and (forall n, Machine17.g17 n <= 11 + 17)
      and (forall n, Machine19.g19 n <= 18 + 19)
```

    rung      width  weights                     sum  max over tuples  margin  tuples
    11 -> 13    20   20 integers, EIGHTEEN 1s     22        21           1      5,005
    13 -> 17    28   28 integers, all in [2,5]    94        92           2     85,085
    17 -> 19    37   37 integers (CoveringCert) 9757     12489-2749=9740  17  1,616,615

Both new ones are PALINDROMES again. Against the thread's fully-consistent dual at
11->13 (106 integers over denominator 37, 2,868 rational ops), the phase-tied form
is **20 small integers, eighteen of them 1**, integer arithmetic with no
denominators. Verified exactly over every phase tuple before formalising
(`consistent_cert.py`, `gen_consistent.py`; 3,850 and 59,767 DISTINCT coefficient
vectors among the 5,005 and 85,085 tuples). Three consecutive (D) rungs by
covering certificates, sharing nothing with the merge law; 11->13 and 13->17 need
consistency, 17->19 does not. Honest range 7->11 .. 17->19, 19->23 undecided;
**it cannot reach 23->29** (verdict 12b).

## 3. Axiom audit

Default is the standard three `[propext, Classical.choice, Quot.sound]`.
Departures, all confirmed by `lake env lean AxiomCheck.lean`:

**NO AXIOMS AT ALL:**

    Machine13.w11, Machine13.w16                  the two round-11 period scans
    PolignacCap.cap_gcd_{1,3,5,7,15,21,35,105}    all eight class caps
    PolignacCap.capOf_le_twelve                   the absolute ceiling
    TierA.padding_count_le                        p <= F/q + 5/6
    Machine11.qasm                                the 385-tuple check
    Machine13.qasm                                the 5005-tuple check
    Ladder.criterion_arith
    DepthSum.local_factor_{5,7,11,13}
    DepthSum.depth_sum_at_13, DepthSum.depth_sum_hl_form
    CoveringCert.cert_signs

**`[propext]` ALONE:**

    Placement.sign_law
    TierA.onset_gate
    Machine17.w18All, Machine17.w25All            85,085-tuple period scan
    Machine19.sliceAll                            1,616,615-slot period scan
    Machine19.qsliceAll                           the qualifying scan
    Machine23.qsliceIdxAll                        323-slice position-indexed scan

**`[propext, Quot.sound]`:**

    Layer.slot_cap
    Placement.prime_mod_six
    Corridor: all round-9 theorems EXCEPT double_slot_in_run
    MaxGap: all four theorems
    TierA.padding_at_most_one_below_onset
    Machine17.qsliceAll                           85,085 tuples
    Machine29.merge_alphabet

**Standard three**, notably the whole extraction layer above the 23->29 scan:

    Machine23.next23_step      [propext, Classical.choice, Quot.sound]
    Machine23.chain_facts23    [propext, Classical.choice, Quot.sound]
    Machine23.spectrum23_one   [propext, Classical.choice, Quot.sound]
    Machine23.spectrum23_two   [propext, Classical.choice, Quot.sound]
    Machine23.qual23_all       [propext, Classical.choice, Quot.sound]
    Machine23.D_23_29          [propext, Classical.choice, Quot.sound]
    Machine23.g29_le_60        [propext, Classical.choice, Quot.sound]

**Shape to aim for**: 323 `decide +kernel` declarations collected by
`interval_cases` carry essentially no logical baggage - `qsliceIdxAll` needs
`propext` ALONE. The standard three enter only in the extraction above the scan;
which mathlib lemma introduces them was not traced, so no cause is claimed.
`Corridor.n2_packing` uses `Classical.choice` via `choose`. `Nat.find` is fully
usable throughout: the opening predicate is decidable, so `opSeq`/`g19` are
computable and `find_spec`/`find_min` give consecutiveness with no extra choice.

## 4. Honest "will not close" verdicts

1. **Tier A does not close 19->23.** `flanks_19_23_nonempty : carrier (flanked 25
   [8]) = {0, 5, 7, 12}` - a theorem, not an omission. The mod-385 and direct
   tiers are genuinely needed there; anyone building on tier A must carry this.
2. **(E) both-flanks-maximal exclusion is off-target for (D).** Measured FS_max
   is attained at MID-SIZE flanks, never maximal ones (29->31: max FS = 48 at
   flanks (18,30), F = 43). The theorems stand as corridor facts but rule out a
   configuration that never binds.
3. **The monotone-envelope / F_j spectrum route will not close (D).** Flatness
   FAILS at 29->31 (the 5-window max sits 42 above F where 31 is allowed), so
   `F_j` was deliberately not formalised in that form. If wanted: replace the
   literal 2 in `Machine17.pair25T` by j. (`Q_j` is a different object and is
   what carries the ladder.)
4. **Residue laws cannot cap sizes** - the mod-105 / mod-385 corridor transfer is
   not a route to size bounds.
5. **"Cap <= 6 for ALL (t,s) pairs mod 35" is FALSE**: over all 1225 pairs the
   spectrum runs {2,3,4,5,6,8,10,140}. The restriction to invertible classes mod
   210 does real work - the cap is not a property of the exposed set alone.
6. **q < y <= q^2 is insufficient for the c-prime semiprime conclusion**
   (counterexample q = 5, y = 25, m = 175 = 5*35). Honest regime m < q^3;
   `Gear.window_bounds` is the adapter.
7. **Tier-C wall: measured, REVISED, superseded.** R15's "no further than machine
   19" was an encoding artefact - the 38x stack put m19 in single-digit minutes,
   and the merge law made an m23 scan unnecessary for (D) at 19->23. The real
   wall is verdict 17.
8. **R18 correction of r15 padding claims**: the count bound is budget
   arithmetic, NOT constant; `F < q` is not "the onset condition" but (by
   `onset_gate`) precisely the regime where NO padded link exists. The theorems
   were hypothesis-explicit and never false; headings overclaimed, and were
   restated.
9. **THE MERGE LAW IS ONE-STEP - structural, not budgetary.** R39 consumes an
   `F_2` and a qualifying spectrum of the OLD machine and produces a bound on the
   NEW machine's SINGLE gaps - not the form the next rung needs. At the live
   step: R39 gives `g23 <= 47`, so the best merge-law-only bound on `F_2(23)` is
   `2*47 = 94` against the `<= 63` required (true value 39). Chaining depth-j
   bounds is WORSE: a depth-j m23 window relaxes to an m19 window with `j-1`
   unconstrained interiors and the loss compounds linearly in j (three qualifying
   m19 blocks admit `47+10+47 = 104` against the true `Q_3(23;10) = 43`). **No
   function of the old machine's marginal data supplies the next rung's input;
   each rung needs its own scan.** (Constructor's counting boundary R41 in
   formal-lane form; the marked spectrum / sandwich lemma is the answer.)
10. **A rung's bound is not the machine's F.** Every rung is true but untight (20
    vs F(13)=11; 26 vs 18; 35 vs 25; 47 vs 34; 60 vs 34). (D) at alpha = 3 is all
    that is claimed and all the route needs; exact F still needs the machine's
    own certificate.
11. **The depth-sum identity's glue was not built.** `depth_partition` counts
    window STARTS in an index range; `depth_sum_at_13` counts openings in a SLOT
    range. Relating them needs "one period of `Machine13.opSeq` = one period of
    residues" (`opSeq (n + 1485) = opSeq n + 5005`, 1485 openings/period).
    Routine but real; named rather than half-done.
12. **The Kleene identity itself was NOT formalised, nor the converse of the
    potential form.** `F(M+q') = L^T (x) K* (x) R` is an EQUALITY needing
    max-plus matrix machinery plus the machine's own `K`; the converse ("a
    potential always exists") uses nilpotency of `K`, `h` being the least
    super-solution, needing the finite path bound as a Finset construction.
    **Only the certificate direction is claimed** - the one a proof consumes, and
    `Potential19` shows it is not vacuous. Recorded verbatim from constructor so
    nobody reads more into these files: the generator is arity-free but NOT YET
    machine-free - bounded-state certificates certify 19->23 (45 <= 48) and FAIL
    at 29->31 (99/99/91 against a budget of 74). **The files make the target
    statement precise; they do not prove (D).**
    - **12b. The covering vehicle cannot reach 23->29.** A level-2 certificate at
      m23 at width 63 was named as the cheap way to skip the m23 scan; the
      LP-duality thread settled it negatively - **23->29 is VACUOUS at degree 2**
      (the uniform product measure is a global distribution, hence feasible for
      the CONSISTENT relaxation too), so "consistency buys WIDTH, not MACHINES".
      Their rung-ratio row `B(y)/F(y)` = 2.29, 1.82, 1.56, 1.48, 1.41, 1.47,
      1.28, 1.08, 1.42 is why, and is worth carrying: **a certificate vehicle
      must be NEAR-TIGHT at every step, so any bound whose gap grows with the
      machine stops proving rungs long before it stops proving anything.** This
      lane's margins are 4.5%, 2.1%, 0.14% - shrinking exactly as predicted.
    - **12c. The published marked-spectrum numbers were inflated; one verdict
      reverses.** Re-derived from the written DEFINITION, not the code:

          step (floor)     J:      2     3     4     5     6     7
          11->13 (a=6)  Q_J(13)   16    18    23     0     0     0    exact
                        published 16    23    23     0     0     0
          13->17 (a=6)  Q_J(17)   25    28    31    32    34     0    exact
                        published 25    28    32    33     -     -
          17->19 (a=8)  Q_J(19)   31    35    37    38     0     0    exact
                        published 31    35    38    38     -     -
          19->23 (a=10) Q_J(23)   39    43    50    55    60     0    exact
                        published 39    50    50    55    60     0
          29->31 (a=10) Q_J(29)   55    65    68    71    71    71    exact
                        published 55    65    68    85    73    73

      **The corrected marked spectrum equals the exact `Q_J(new)` in all 30
      entries of all five steps** - not merely a tight relaxation, exact
      entrywise. The discrepancy is ONE LINE OF A DP: `research/marked_qspec.py`
      places `J-1` marks and returns success the moment the count is reached,
      never checking that interiors AFTER the last mark are killed, so it accepts
      windows whose tail holds an opening neither marked nor killed. Re-running
      correct code with that one check disabled REPRODUCES THE PUBLISHED ROW
      DIGIT FOR DIGIT at every step - the proof of diagnosis. Consequences: the
      19->23 verdict STANDS and was conservative (the error bites at J = 3, not
      at the maximum); **the 29->31 verdict REVERSES** - corrected `max = 71 <=
      74`, so the rung is NOT lost by this route, and the J = 5 entry that
      carried the whole verdict was the DP artefact. Triple-sourced: mechanic
      retracted the row, constructor proved the SANDWICH LEMMA forcing the
      equality (`Q_J(new) <= Q^[J](old) <= max_{j<=J} Q_j(new)`, hence
      `max_J Q^[J](old) = max_J Q_J(new)` always), and this lane located the DP
      line.
13. **A Lean kernel cannot share a walk across a phase loop** (r23; since FIXED -
    kept because the mechanism recurs). Factoring m23's period as (m19 tuple) x
    (phase) is EXACT, but the m19 walking is IDENTICAL for 21 of 23 phases and
    cannot be shared: hoisting the phase-free walk out explicitly gave NO
    improvement (11 s vs 12 s), while a control making the loop body
    `g`-INDEPENDENT collapsed the cost to 1.35 s. **The kernel DOES share
    structurally identical subterms; the walks after the first hop are indexed by
    `g` even where they compute the same number** - a pure term-rewriting kernel
    has no way to say "evaluate once, then branch". The named fix
    (position-indexing) was built in r24 and over-delivered. The corrected marked
    spectrum does NOT remove this: it replaces the phase quantifier by a phase
    MAXIMUM, which in a functional kernel encoding still costs one walk per phase
    (mechanic's Python avoids that with an incremental mutable coverage array; a
    kernel term cannot).
14. **`F_2(23) <= 63` cannot be got from `F(23) <= 34` by doubling** - the
    obvious shortcut misses by five: `2*34 = 68 > 63` (and with `g23 <= 47` it is
    94). The depth-2 fact genuinely needs its own scan.
15. **A_4's soundness is not itself kernel-checkable at machine 29; any Lean
    statement must carry it as a hypothesis.** Constructor's A_4 (state = last
    three GAP VALUES, phase-free, 14,368 states / 3,513 edges at m29, exact at
    all seven scannable steps) has edge relation "this 4-tuple of consecutive
    gaps is REALISED in the period" - a full-period claim about 1,078,282,205
    slots, uncertifiable in the kernel by any method in the lane. What CAN be:
    the longest-path value over an EXPLICIT edge set, with "E contains every
    realised 4-tuple" as a named hypothesis a census discharges. No theorem must
    be worded to suggest (D) is proved in general: A_4 is per-machine, and
    constructor's own machine-free result in the same report is a NEGATIVE
    (MF_3 mod 35 = MF_3 mod 385 = MF_4 mod 35 at all seven steps,
    15/31/47/111/105/125/211 against budgets 20/28/37/48/63/74/95).
16. **A_4 at 29->31 in the hypothesis-explicit shape was NOT STARTED** - a
    not-attempted, not a will-not-close. `grep -rn "A4\|A_4" proofs/*.lean` is
    EMPTY and there is no `Machine31`; r24 was consumed by the 23->29 rung and
    the agent hit an API limit. Recorded plainly so no one reads the round's
    success as covering it. Target shape unchanged:
    `theorem D_of_A4 (E : Finset (N x N x N x N)) (hE : every realised 4-tuple of
    consecutive gaps is in E) (hlp : longest path over E <= 58) : (D) at 29->31`.
17. **THE PER-RUNG SCAN VEHICLE ENDS AT 23->29, AND THE REASON IS THE PERIOD, NOT
    THE ENCODING.** R24's scan factored m23's 37,182,145-slot period as (m19
    tuple) x (phase), reusing the 323 slices m19 already had. The same trick at
    29->31 factors 1,078,282,205 = 37,182,145 x 29 as (m23 tuple) x (phase) -
    **but there is no 323-slice m23 slice family to reuse; the outer index is
    37,182,145 / 5,005 = 7,429 slices.** At the MEASURED 65 s per slice that is a
    FLOOR of 134 h, and **~170 h** once the inner loop is 29-fold rather than
    23-fold - before the walk itself being longer, so the true figure is higher.
    **The position-index fix does not repeat: its saving came from sharing ONE
    m19 walk across 23 phases, and that walk gets ~23x longer at the next
    machine.** WHAT REPLACES IT: everything past 23->29 must come from **the
    SANDWICH LEMMA** (supplies every rung from the OLD machine's period, and by
    12c is exact entrywise) **or from a certificate over a bounded-state
    abstraction** (A_4/A_5 longest path over an explicit edge set, the edge set
    supplied by mechanic's dictionary-transfer superset). Measured verdict, not a
    judgment. (Correction: r23 recorded the direct m23 slice family as "7,434
    slices"; it is 7,429 - 37,182,145 / 5,005 divides exactly.)
18. **`proofs/DryScan2.lean` is scaffolding and is NOT in the ledger** - not in
    `lakefile.toml`, nothing imports it, not among the 1372 jobs. It contains
    `axiom qsliceIdxAll : forall e < 17, forall f < 19, qsliceIdx e f = true`.
    Its header says "NOT part of the ledger; deleted before close"; it was NOT
    deleted (the agent died first) and is **RECOMMENDED FOR DELETION at the next
    commit**. Standing rule in 1.4. (R23 drafts `DryScan.lean`, `PsliceOld.lean`,
    `Machine23Probe.lean` do NOT exist on disk; r19 drafts
    `Machine19Core`/`Machine19Probe` were later registered and are in the ledger.)

## 5. Failed approaches and standing lessons

### 5.1 Tactic-level

- `omega` cannot see nonlinear atoms (`y*y`, variable products) - use `linarith`
  (treats `y*y` as opaque) for window inequalities; derive `q*c != 1` from
  `Nat.dvd_one`. It cannot relate variable*variable products (link them with an
  explicit `Nat.mul_succ`/`mul_comm` equation), nor mix `Nat.find` with a def
  wrapping it (state the find-fact in the def's terms explicitly).
- **One-shot `omega` dies at 5 simultaneous dvd atoms and TIMES OUT at 12
  simultaneous `%`-atoms** (200,000 heartbeats) though each is individually
  trivial. Same fix twice: introduce per-gear facts separately (`have q5 : (p % 5
  + i) % 5 = (p + i) % 5 := by omega`), rewrite with them, finish the
  propositional step with `tauto` (for the dvd version: generalize `r = k%35`
  then `interval_cases r <;> decide`) - **normalise instead of searching.**
- An 8-conjunct Bool/Prop bridge iff times out at 1M heartbeats under `tauto` OR
  `omega` even when each half is fast. Fix: `simp only [expT, Bool.and_eq_true,
  bne_iff_ne, ne_eq, and_assoc]`.
- When a product of two variables is bounded on one side, `interval_cases` the
  bounded one first: `((i+ph)/2) * q` becomes `literal * q`, linear.
- Index-shift goals over window sums: do the shifts as explicit `rw`s - `congr 1;
  omega` and `norm_num` both fail; omega cannot close until the g-atoms are
  syntactically identical.
- **State gap facts as `opSeq` differences, never as walk offsets.** `have g3 :
  g11 (n+3) = o4 - o3 := by simp only [g11]; omega` FAILS - `simp only` unfolds
  to `opSeq (n+3+1) - opSeq (n+3)` and omega sees `opSeq (n+3+1)` as an atom
  distinct from the chain's `opSeq (n+4)`. Working shape: `have g3 : g11 (n+3) =
  opSeq (n+4) - opSeq (n+3) := by simp only [g11]` - simp's index normalisation
  closes it by rfl. **This will bite any future rung.**
- **`rcases h with a | b; . tac1; . tac2` inside a term-mode `by` is wrong** -
  `;` sequences over ALL goals so focusing bullets misfire. Use `<;>` with a
  tactic closing both branches (`rcases hk with h | h <;> omega`).
- **Component projections into a 12-conjunct `Exposed` are position-sensitive and
  the kernel catches swaps immediately** - gear q's `lo` and `hi` teeth are
  DIFFERENT residues (19: `lo` at `k % 19 = 16`, `hi` at 3), so `Or.inl`/`Or.inr`
  must match `killed_iff`'s order. Two swapped pairs were caught by
  application-type-mismatch, not by a false theorem.
- **`seek_next` needs `hnE : Exposed (nextOp (x+s))` in scope** - the trailing
  `rwa` closes with `assumption`. Dropping that line cost two rebuild cycles.
- 4-witness extraction from a `countP` fact (pre-`seekT` method): convert with
  `List.countP_eq_length_filter`, rcases to `w::x::y::z::rest` (length
  contradictions close short cases), get distinctness by three
  `List.nodup_cons.mp` unpackings, derive per-witness 4-way disjunctions by a
  `by_contra` cascade, and let one `omega` do the pigeonhole (256 cases - fine).
- This mathlib: `Finset.range_subset` has a different shape (supply subsets
  pointwise); `Finset.sum_const_nat` is the Nat-native collapse (`smul_eq_mul`
  missing under minimal imports); `windowSum_mono` needs
  `Mathlib.Algebra.Order.BigOperators.Group.Finset`; `ring` unavailable under
  minimal imports (use `Nat.mul_succ`); `Nat.modEq_iff_dvd'` needs
  `Mathlib.Data.Nat.ModEq` (cached).
- Deprecations: `if_pos`/`if_neg` (prove the two unfolding equations once with
  `split` - `seekT_succ_pos`/`_neg` - and `rw` with them); `push_neg` -> `push
  Not` (warns, failing the zero-warning invariant); `Set.mem_setOf_eq` ->
  `Set.mem_ofPred_eq`.

### 5.2 Kernel-computation-level

- At 85085 tuples the limit is tuples PER DECLARATION, not total:
  `decidableBallLT` over all coords blows the proof TERM (2 GB+); one Bool with 5
  nested `List.all` makes the term `rfl` but evaluation never finishes; `(all e <
  17, slice e = true)` by `decide +kernel` is still > 600 s (**a Prop quantifier
  over Bool slices does NOT behave like separate declarations**). What works:
  explicit slice theorems + `interval_cases` assembly.
- Test a new lemma in a scratch file before placing it upstream of multi-minute
  decides. This caught the dead `Finite.injective_iff_surjective` route for
  `exists_mul_mod_eq` (missing `Finite (Fin n)` instance); the working route is
  `ZMod n` units (`ZMod.unitOfCoprime`).
- Scaling rules, the sharing law and the encoding recipes: 1.3. Memory and
  parallelism: 1.2.

### 5.3 Modelling-level (the kernel caught real errors)

- First `pairT` formulation quantified over ALL window starts, not openings -
  `decide` reported it FALSE (1296 counterexamples confirmed). **The real F2
  statement requires the window to start at an opening.**
- Gear-3 skip semantics in the halved frame: gear 3 FILTERS the candidate list; a
  3-inadmissible kill is SKIPPED and the run continues across it. Treating gear 3
  like gears 5/7 gives max caps 2/4 instead of 6/10/12.
- `minFac 0 = 2`: without `1 < m`, 0 lands in gear 2's fiber and the shadow law
  is false.
- **Verify every claim against the `research/*.py` tooling BEFORE formalising.**
  This caught all of the above, validated every table, and produced the two
  corrections this lane is proudest of (the C13 row; the marked-spectrum DP bug).
  Where a claim is load-bearing, re-derive it from the written DEFINITION rather
  than from another lane's code - that is what located the DP line.
- Adopted correction: **`F(29) = 43`, not 46.** Checked against every statement
  in this lane: no contamination (the only use is the 29->31 budget 43 + 31 = 74,
  already right).

## 6. Cross-lane kernel candidates on offer (received r24, none actioned)

1. **MECHANIC - the dictionary-transfer superset, `research/dict_transfer.py`**:
   exactly the `hE` shape verdict 15 asked for. A window of M+q' is an M-window
   plus one free phase, kills decided by partial sums mod q', so walking machine
   M's dictionary in its order-m closure yields a certified SUPERSET of machine
   (M+q')'s m-tuple dictionary WITH NO SCAN. At 29->31: **715,697 rows,
   containment proved by construction**, verified exhaustively. Superset edges
   keep the max-plus closure sound, so this discharges `hE` without the
   1.08e9-slot realisability claim that made verdict 15 a negative. Highest-value
   item: converts open target 1 from "needs a census we cannot kernel-check" into
   "needs a longest-path `decide +kernel` over an explicit 715,697-row digraph".
2. **CONSTRUCTOR - `A_5(23)` survivor closure = 55**: a finite integer digraph
   yielding the R53 integer `F_2(29) = 55` with no machine-29 scan. Cheaper than
   A_4-at-29->31, same rung. Also offered: **the survivor identity at 11->13**, a
   finite max-plus closure equal to 16 - the right size for a first, cheap
   statement of the generator in Lean.
3. **LATERAL - the m13 covering dual**: weights over a 793-row system with
   denominator 2081, exact rationals already, integer check only. Directly
   comparable to `CoveringCert2.cert13` (20 integers, eighteen of them 1), so it
   doubles as an independent check of the phase-tied form; plus **LP(MF) = closure
   at one step**, a longest-path duality over an explicit integer digraph.
4. **LP-DUALITY THREAD**: composed certificates 2-3x SMALLER than r23's at the
   same four rungs (562 / 1,456 / 3,303 / 8,179 ops vs 464 / 2,868 / 9,091 /
   25,413), same integers-over-one-denominator shape plus one integer recursive
   row. Their r23 `W*` correction leaves the Lean certificates untouched - checked
   and agreed: a valid certificate stays valid whatever `W*` is, and nothing in
   this lane consumed `W*`.

## 7. Open formalisation targets (round-25 priority order)

1. **A_4 / A_5 as a longest-path certificate, on MECHANIC's transferred edge set**
   (verdicts 15, 16; cross-lane 1). **Top target**: the expensive half (`hE`) is a
   finished artifact in another lane, and the rest is a `decide +kernel` longest
   path this lane has the tooling for. Take `A_5(23) -> F_2(29) = 55` (cross-lane
   2) as the cheap first instance if the 715,697-row digraph is too large for one
   declaration.
2. **The SANDWICH LEMMA, formalised** (constructor R51): `Q_J(new) <= Q^[J](old)
   <= max_{j<=J} Q_j(new)`. Promoted by verdict 17 - not merely the best
   value-per-line object in the lane but the **ONLY route past 23->29 that does
   not need a per-rung scan**, the per-rung vehicle being measured dead at ~170 h
   for the next rung. With it the marked spectrum supplies EVERY rung from the OLD
   machine's period. The `<=` half is a relaxation argument; the `>=` half (extend
   a relaxed window to the nearest survivor on each side) is the content.
   Abstract, machine-free, needs no scan.
3. **The survivor identity at 11->13** (cross-lane 2) - small, finite, the first
   Lean statement of the generator.
4. **The m13 covering dual** (cross-lane 3) - independent check of the phase-tied
   certificate form, plus the LP(MF) = closure duality.
5. **The depth-sum glue at m13** (verdict 11): `opSeq (n + 1485) = opSeq n + 5005`
   and the re-indexing it enables. Finishes the identity at one machine.
6. **Harvester's paired-Holt coef rung** (5005 -> 85085): coef position-freeness
   is near-definitional; the rung verification is a machine-17-scale scan with
   word extraction. The subterm-sharing finding makes the 85,085-fold quantifier
   cheap - **likely much easier than its size suggests.**
7. **Suppression-corrected flatness wired at a further machine**: prove a
   `SpectrumBound g 4 F4` instance from a period scan at 31/37/41, contingent on
   mechanic's two halves surviving there.
8. **CRT collapse / single-cycle reduction**: one orbit-length walk replaces the
   whole start set (37x) whenever the step is invertible mod the modulus;
   `PolignacCapCore.exists_mul_mod_eq` is already proved. On the shelf for any
   machine truly out of reach otherwise.
9. Housekeeping: delete `proofs/DryScan2.lean` (verdict 18).

WITHDRAWN / DONE:
- ~~A level-2 covering certificate at machine 23 at width 63~~ - WITHDRAWN,
  refuted by the LP-duality thread (verdict 12b).
- ~~The machine-23 chain scan by the index encoding~~ - **DONE** (r24, 2.21); it
  delivered the fifth rung.
- ~~The (A) word-list enumeration gap~~ - **DONE** (`LiteralCapTable`, 2.12); (A)
  is now FULLY kernel-checked.
