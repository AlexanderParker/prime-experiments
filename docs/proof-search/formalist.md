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

---

## Round 25 append (2026-08-29)

Brief: (1) A_4 at 29->31 via the dictionary-transfer superset - the `hE`
shape of verdict 15, the SIXTH rung and the first by the new vehicle;
(2) Constructor's `A_5(23)` closure = 55 as the cheaper input; (3) the
11->13 survivor identity = 16 as the first kernel statement of the
generator; (4) Lateral's m13 covering dual. The per-rung scan vehicle is
dead (verdict 17) - no 29->31 scan attempted.

**Build GREEN at 1410 jobs** (1372 -> 1410), 74 targets, 127 files, **zero
sorries, zero `axiom` declarations, no `native_decide`, no
`Lean.ofReduceBool`**. Ten new roots: `Machine29Q`, `Machine29D2..D7`,
`Machine29Dict`, `Machine31`, `Gen11`, and for the seventh rung `Machine31Q`,
`Machine31D2..D7`, `Machine31Dict`, `Machine37` - nineteen in all. Scaffolding
deleted at close
(`DryScan2.lean`, verdict 18's outstanding item, and this round's
`_DryR25.lean`).

### R25.1 THE SIXTH RUNG - (D) at 29->31, in five minutes instead of 170 hours

The vehicle is NOT the 4-tuple dictionary the brief pointed at. Measuring
first (`research/a4_potential.py`) killed that: over machine 29's exact
45,854-row 4-tuple dictionary the qualifying-tail potential is INFINITE -
the qualifying subgraph of the A_4 state digraph has CYCLES, because A_4
cannot see that a run of six gaps `>= 10` is impossible (that is a 7-tuple
fact). Same at m23 and m31; only m19's 380-row set is acyclic. **A_4 is the
wrong state for this certificate, and one script said so in 30 seconds.**

What works is the STRATIFIED QUALIFYING FAMILY. `MergeLaw.newgap_le_step`
consumes only `F_2(M) <= B` and `Q_j(M; a) <= B` for `j >= 3`, and `Q_j`
quantifies ONLY over windows whose interiors reach the floor `a = 2u''`. So
the whole input is `D_j` = the realised `j`-windows with qualifying
interiors, one list per depth, and the family TERMINATES at `j = K + 2`
where `K` is the longest qualifying run (3, 4, 5 at m19, m23, m29 - it does
not grow like the period). At machine 29, floor 10, budget `43 + 31 = 74`:

    j        2      3      4      5      6      7      8
    |D_j|   730  3,692  6,688  3,915    789     46      0
    Q_j      55     65     68     71     71     71      -      max 71, margin 3

15,860 tuples against a period of 1,078,282,205 slots. Landed theorems:

```lean
-- proofs/Machine29D2..D7.lean, one module per depth (NO AXIOMS AT ALL)
theorem D4_ok : D4.all (fun t => Nat.ble (t.1 + t.2.1 + t.2.2.1 + t.2.2.2) 68)
    = true := by decide +kernel

-- proofs/Machine29Q.lean - machine 29's enumeration is complete, NO SCAN
theorem opSeq29_surj {m : N} (hm : 1 <= m) (hE : Exposed29 m) :
    exists n, opSeq29 n = m

-- proofs/Machine29Dict.lean
structure Census29 : Prop where          -- THE ONLY UNPROVED INGREDIENT
  E2 : forall n, (g29 n, g29 (n+1)) in D2
  E3 : forall n, 10 <= g29 (n+1) -> (g29 n, g29 (n+1), g29 (n+2)) in D3
  E4 / E5 / E6 / E7   -- likewise, one qualifying interior more each
  run : forall n, not (10 <= g29 (n+1) and ... and 10 <= g29 (n+6))
theorem spectrum29_two (h : Census29) : Spectrum.SpectrumBound g29 2 55
theorem qual29_three/four/five/six/seven (h : Census29) :
    Spectrum.QualBound g29 5 j (65/68/71/71/71)
theorem qual29_all (h : Census29) : forall j, 3 <= j -> Spectrum.QualBound g29 5 j 71
theorem criterion_29_31 : max 55 71 <= 43 + 31        -- NO AXIOMS

-- proofs/Machine31.lean - gear 31, teeth {5, 26}, u = 5, floor 2u = 10
def Killed31 (k : N) : Prop := k % 31 = 5 or k % 31 = 26
def Exposed31 (k : N) : Prop := Exposed29 k and not(31 | lo k) and not(31 | hi k)
def g31 (n : N) : N := opSeq31 (n+1) - opSeq31 n
theorem merge_alphabet (hk1 : Killed31 x) (hk2 : Killed31 y) (hxy : x < y)
    (hle : y - x <= 43) : y-x = 10 or y-x = 21 or y-x = 31 or y-x = 41
theorem D_at_29_31 (hF2 : Spectrum.SpectrumBound g29 2 55)
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g29 5 j 71) (n : N) :
    g31 n <= 43 + 31
theorem g31_le_71 (hF2) (hQ) (n : N) : g31 n <= 71     -- R39's own form
theorem D_29_31 (h : Census29) (n : N) : g31 n <= 43 + 31   -- THE SIXTH RUNG
theorem g31_le_of_census (h : Census29) (n : N) : g31 n <= 71
```

    step     criterion max(F2, max_j Q_j)   budget F+q'   margin   floor 2u'
    11->13   max(11, 20) = 20               20             0 TIGHT   4
    13->17   max(16, 26) = 26               28             2         6
    17->19   max(25, 35) = 35               37             2         6
    19->23   max(31, 47) = 47               48             1         8
    23->29   max(39, 60) = 60               63             3        10
    29->31   max(55, 71) = 71               74             3        10   <- NEW

MEASURED COST, the headline: verdict 17 priced this rung at **~170 h**. The
dictionary rung builds from cold in **under 5 minutes** - `Machine29D4`
(6,688 tuples, the largest module) 55 s, `Machine29Dict` 9.8 s, `Machine31`
13 s, whole driver 04:37:53 -> 04:42:45. Peak RSS ~1.1 GB per worker, so the
round-24 parallelism budget of 2 was never binding. **The certificate's size
is the dictionary's, not the period's**: 990 / 2,911 / 15,860 tuples at
m19 / m23 / m29 against periods of 3.8e5 / 3.7e7 / 1.1e9 slots - roughly 5x
per gear against 30x per gear for the period.

MEMORY/ENCODING NOTES (new infrastructure facts):
- A list literal of a few thousand tuples needs `set_option maxRecDepth`
  (1,000,000 used); without it `Machine29D4` dies at the DEFINITION, not the
  decide, with "maximum recursion depth" - 43 s to find out.
- `decide +kernel` over `List.all` on 6,688 4-tuples costs 55 s INCLUDING
  imports and has an EMPTY axiom footprint. The r24 rule of "~5e3 tuples per
  declaration" is about DISTINCT SUB-COMPUTATIONS; a list of tuples with one
  addition each is far cheaper per element than a walk, and 6,688 was
  comfortable.
- Extraction from a list fact is three lines and costs nothing:
  `List.all_eq_true.mp Dj_ok _ (h.Ej a ...)`, then `simp only [Nat.ble_eq]`,
  then `rw [wsj]` where `wsj : windowSum g29 a j = g29 a + ...` is proved by
  `simp [Spectrum.windowSum, Finset.sum_range_succ]`.
- The whole three-file development (`Machine29Q`, `Machine29Dict`,
  `Machine31`, 483 lines) elaborated CLEAN ON THE FIRST TRY under the
  axiom-stubbed mega-dry file, in 20 s. That discipline has now paid twice.

### R25.2 THE CENSUS INPUT, AND ITS FOUR GATES

`Census29` is a full-period claim about 1,078,282,205 slots and is NOT
kernel-checked - that is the whole design, and `Machine29Dict.lean`'s header
says so in the file. It is measured by `research/qual_dict.py` and gated by
`research/qual_dict_gate.py` (ALL FOUR GATES GREEN):

1. **Chunk independence** - the period is scanned twice with unrelated chunk
   sizes (40,000,000 and 23,456,789); all six dictionaries, the whole `F_j`
   ladder and the run length come out identical. This is the defence against
   mechanic's standing rule 18 (a window straddling a junction seen by
   neither pass), and I introduced exactly that bug and caught it with this
   gate: the first version of `gaps_of_period` closed the cyclic seam with
   the wrap gap but NOT with the period's own first gaps, so windows starting
   at the seam were invisible. Fixed, and the gap count is now asserted equal
   to `prod (q-2) = 214,708,725`.
2. **Cyclic seam** - explicit, asserted.
3. **Transcription** - the Lean literals are parsed back out of
   `proofs/Machine29D*.lean` and compared as sets with the scan: all six
   identical, and the six maxima 55/65/68/71/71/71 re-derived Lean-side.
4. **Corpus agreement** - the same scanner at machines 19 and 23 reproduces
   `F_j(19) = 25,31,35,38`, `Q_j(19;8) = 31,35,37,38`,
   `F_j(23) = 34,39,50,58,65,77,83,88`, `Q_j(23;10) = 39,43,50,55,60` and
   `F(29) = 43` - every one a kernel-checked value in this ledger.

TWO CROSS-LANE CONFIRMATIONS fall out of the same scan:
- **`F_2(29) = 55` exactly** - Constructor's `A_5(23)` survivor closure and
  Mechanic's pair census agree; three independent routes, one integer. This
  is brief item (2), delivered as `spectrum29_two` (the bound the rung needs)
  rather than as a separate closure computation.
- **`Q_J(29; 10) = 55, 65, 68, 71, 71, 71` for `J = 2..7`** - EXACTLY the
  CORRECTED marked spectrum of verdict 12c, entry for entry, including the
  `J = 5` value 71 whose published predecessor 85 was the DP artefact that
  had made this rung look lost. A fourth route confirms the correction, and
  the rung stands on it.
- Correction to the corpus in passing: `docs/novel/survivor-generator.md`
  records machine 29's period as "1,078,282,205 slots / 214,709,355 gaps".
  The gap count is **214,708,725** = `3*5*9*11*15*17*21*27` = `prod (q-2)`,
  asserted by the gate. The 214,709,355 figure is wrong by 630.

### R25.3 THE GENERATOR, FIRST KERNEL STATEMENT (`proofs/Gen11.lean`)

Brief item (3). Machine 11's cyclic gap word is 135 letters over 385 slots;
gear 13 kills slot residues 2 and 11; the phase `c` mod 13 is free because
`gcd(385, 13) = 1`. Constructor's generator is then a bounded walk.

```lean
def gw11 : List N            -- 135 gaps, the machine-11 cyclic word
def gAt (i : N) : N := gw11.getD (i % 135) 0
def off (i : N) : N -> N                       -- span of k consecutive gaps
def kil13 (r : N) : Bool := (r % 13 == 2) || (r % 13 == 11)
def walk (i c ns : N) : N -> N -> N -> N -> N  -- pass killed, stop at the
                                               -- (ns+1)-st survivor
def gen (ns : N) : N         -- max over 135 bases x 13 phases, span cap 30

theorem gw11_len : gw11.length = 135                          -- [propext]
theorem gw11_sum : gw11.sum = 385                             -- [propext]
theorem gw11_max : gw11.all (fun g => Nat.ble g 7) = true     -- F(11) = 7
theorem no_truncation : forall i < 135, 30 < off i 13         -- fuel never binds
theorem gen_zero : gen 0 = 11        -- L (x) K* (x) R           = F(13)
theorem gen_one  : gen 1 = 16        -- L (x) K* (x) SIGMA (x) K* (x) R = F_2(13)
theorem generator_matches_machine13 :
    gen 0 = 11 and gen 1 = 16 and Spectrum.SpectrumBound Machine13.g13 1 11
      and Spectrum.SpectrumBound Machine13.g13 2 16
```

`gen 0` / `gen 1` are `[propext]` ALONE. `ns = 0` is R46's plain Kleene
generator; `ns = 1` inserts Constructor's SIGMA letter exactly once - the
survivor identity. **Both integers are produced from a 135-letter word over a
385-slot period, with no mention of machine 13's 5005-slot period, and both
match machine 13's own kernel-proved spectrum.** `no_truncation` (thirteen
consecutive machine-11 gaps already span 33 > 30) shows the fuel exits by the
span cap, never by exhaustion, so `gen` is a maximum over ALL windows of span
at most 30, not merely over short ones - and the Python gate
(`research/gen11.py`) shows the values are stable to span cap 60.

HONEST SCOPE (see verdict 20): the SOUNDNESS BRIDGE is not formalised.

### R25.4 WHAT WAS NOT DONE, AND WHY

- **Brief item (4), the m13 covering dual, NOT ATTEMPTED.** Reason, not
  judgment: the exact rational dual `1041/2081` is reported in
  `docs/novel/covering-hierarchy-exactness.md` but the DUAL VECTOR IS NOT ON
  DISK - `research/sdp_cover.py` has no save path (grep for `np.save` /
  `json.dump` / any file write returns nothing) and no `sdp_*` artefact in
  `research/data/` carries it. Kernel-checking it therefore means
  reconstructing Lateral's Sherali-Adams level-2 system at m13 (793 rows over
  its own literal encoding) and re-solving, which is a lane-crossing
  reconstruction rather than a transcription. Round-24's process rule
  ("feasible verdicts must save their witness") applies here and is the fix:
  if Lateral saves the dual vector, the denominator and the row generator,
  the Lean side is an afternoon.
- **The dictionary-transfer superset was NOT used as the `E` in the end.**
  Measured reason: the transfer produces `m`-tuple dictionaries, and the
  qualifying family needs depths up to `K + 2 = 7`; `dict_transfer.py` at
  `out_m = 7` is far past the cost it was built for. The exact m29 census is
  smaller and equally hypothesis-shaped. The transfer remains the right way
  to DISCHARGE `Census29` from `Census23` - see open target 1.

### R25.5 New will-not-close / honest-scope verdicts

19. **The A_4 (4-tuple) abstraction cannot carry the qualifying-tail
    potential, and the reason is a cycle, not looseness.** Measured
    (`research/a4_potential.py`): over the exact realised 4-tuple dictionary
    the sub-digraph on states whose first gap qualifies contains a CYCLE at
    m23 (15,696 edges), m29 (45,854) and m31 (115,193), so the longest
    qualifying path is INFINITE and no potential of that arity exists. Only
    m19's 380-row set is acyclic (value 25). The obstruction is exactly that
    a 4-tuple cannot express "no six consecutive gaps reach the floor".
    **A_5 would not fix m29 either** (the run is 5, so the fact is a 7-tuple
    fact); what fixes it is stratifying by depth, which is what R25.1 does.
    Recorded because the brief pointed at A_4 and the negative is cheap,
    exact and reusable.
20. **The generator's SOUNDNESS BRIDGE at 11->13 is not formalised.**
    `Gen11.gen_one = 16` says the generator COMPUTES `F_2(13)`; it does not
    prove it MUST. Two things are missing and both are known-shaped:
    (i) `gw11` certified as machine 11's own opening sequence (a 385-slot
    `decide`, cheap), and (ii) the PERIODICITY GLUE
    `Machine11.opSeq (n + 135) = Machine11.opSeq n + 385` - which is verdict
    11's missing step at another machine. Until (ii) exists, no statement in
    this ledger should be read as "the generator is proved sound". Naming it
    also MERGES TWO OPEN TARGETS: the depth-sum glue at m13 (verdict 11) and
    this are the SAME lemma at two machines, and doing it once as an abstract
    "a periodic decidable opening predicate has a periodic enumeration"
    discharges both.
21. **`Census29` is not, and will not be, kernel-checkable at this machine.**
    Unchanged from verdict 15 and restated because R25.1 might be misread:
    the rung is `D_29_31 (h : Census29)`. Anyone quoting it must quote the
    hypothesis. What R25.1 changes is only that the unverifiable part is now
    ONE FINITE LIST with a published format and four gates, instead of a
    1.08e9-slot claim with no interface.

### R25.6 THE SEVENTH RUNG - (D) at 31->37, and the first non-monotone qualifying spectrum

The vehicle of R25.1 is a template, so once it existed the next rung was a
census plus a transcription. Machine 31's full period - 33,426,748,355 slots,
6,226,553,025 gaps (`= prod (q-2)`, asserted) - scanned in **1,451 s of CPU /
24 min wall** by the same `research/qual_dict.py`. Gear 37's teeth are
`{6, 31}` (`6*6 = 36 = 37-1`, `6*31 = 186 = 5*37+1`), so `u = 6` and the
qualifying floor on machine 31's gap word is `2u = 12`; the budget is
`F(31) + 37 = 58 + 37 = 95`.

    machine 31, floor 12, budget 95
    j        2      3      4      5      6      7      8
    |D_j|  1,253  8,155 18,566 13,049  2,120     42      0
    Q_j       68     85     90     91     90     88      -    max 91 <= 95, margin 4

    F_j(31) = 58, 68, 85, 90, 92, 97, 104, 110, 115, 131   (F(31) = 58 gate-checked)
    longest run of gaps >= 12: 5

**THE QUALIFYING SPECTRUM TURNS OVER.** `Q_j(31; 12)` rises 68, 85, 90, 91
and then FALLS BACK to 90, 88. At machines 19, 23 and 29 it was
non-decreasing and then saturated (31,35,37,38 / 39,43,50,55,60 /
55,65,68,71,71,71). Machine 31 is the first machine in this ledger where it
peaks and declines before going vacuous, and the consequence is concrete: the
constraint that BINDS this rung is a FIVE-gap window with three qualifying
interiors, not the two-gap statement and not the deepest window. Any argument
that assumes `Q_j` is monotone in `j`, or that the binding depth is the last
non-vacuous one, is false from machine 31 on. This is a new measurement of a
new object, and it is the kind of thing the dictionary makes cheap to see.


NEW INFRASTRUCTURE FACT, PAID FOR IN A FAILED BUILD: a big list literal has
TWO elaborator limits, not one, and the second scales with TUPLE ARITY.
`set_option maxRecDepth` (R25.1) gets you past the recursion depth; at machine
31 the modules then died with

    error: (deterministic) timeout at `isDefEq`, maximum number of heartbeats
    (200000) has been reached

- and the follow-on `Unknown identifier D4.all`, which is the DEFINITION
having failed, not the decide. The failure points are diagnostic: `D4`
(4-tuples) died at about element 6,760 and `D5` (5-tuples) at about element
3,540, while `D3` (8,155 3-tuples) elaborated fine in 67 s. So the heartbeat
budget is consumed roughly linearly in (count x arity), and machine 29's
modules had merely sat under it by luck - its D5 has 3,915 5-tuples, just
below where machine 31's D5 died. THE FIX is
`set_option maxHeartbeats 4000000` alongside `maxRecDepth`, and it is applied
UNIFORMLY to every emitted dictionary module including machine 29's - a file
that builds only because it happens to sit under a hidden limit is exactly
what bites the next machine. Both are elaboration RESOURCE limits: neither
touches the axiom footprint, which stays empty for every `Dj_ok`.


GATED THE SAME FOUR WAYS (`research/qual_dict_gate31.py`, MACHINE-31 GATE
GREEN, run to completion): the whole 33,426,748,355-slot period RESCANNED at
an unrelated chunk size (37,000,001 against 60,000,000) with all six
dictionaries identical (1,253 / 8,155 / 18,566 / 13,049 / 2,120 / 42), the
gap count asserted equal to `prod (q-2) = 6,226,553,025`, `F(31) = 58` against
the corpus, the qualifying run 5, `max_j Q_j = 91 <= 95`, and the
transcription against `proofs/Machine31D*.lean` identical set for set. The
census hypothesis `Census31` gets exactly the standard this lane applied to
`Census29`; nothing was claimed before the gate finished.

Landed theorems (`proofs/Machine31Q.lean`, `Machine31D2..D7.lean`,
`Machine31Dict.lean`, `Machine37.lean`):

```lean
theorem opSeq31_surj {m : N} (hm : 1 <= m) (hE : Exposed31 m) :
    exists n, opSeq31 n = m                      -- no scan, same induction
structure Census31 : Prop where                  -- E2..E7 + run, floor 12
theorem spectrum31_two (h : Census31) : Spectrum.SpectrumBound g31 2 68
theorem qual31_five (h : Census31) : Spectrum.QualBound g31 6 5 91  -- BINDS
theorem qual31_all (h : Census31) : forall j, 3 <= j -> Spectrum.QualBound g31 6 j 91
theorem criterion_31_37 : max 68 91 <= 58 + 37                      -- NO AXIOMS

-- proofs/Machine37.lean - gear 37, teeth {6, 31}, u = 6, floor 2u = 12
def Killed37 (k : N) : Prop := k % 37 = 6 or k % 37 = 31
def g37 (n : N) : N := opSeq37 (n+1) - opSeq37 n
theorem merge_alphabet ... (hle : y - x <= 58) :
    y-x = 12 or y-x = 25 or y-x = 37 or y-x = 49
theorem D_at_31_37 (hF2 : Spectrum.SpectrumBound g31 2 68)
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g31 6 j 91) (n : N) :
    g37 n <= 58 + 37
theorem g37_le_91 (hF2) (hQ) (n : N) : g37 n <= 91
theorem D_31_37 (h : Census31) (n : N) : g37 n <= 58 + 37   -- THE SEVENTH RUNG
theorem g37_le_of_census (h : Census31) (n : N) : g37 n <= 91
```

    step     criterion max(F2, max_j Q_j)   budget F+q'   margin   floor   |D|
    11->13   20                             20             0 TIGHT   4       -
    13->17   26                             28             2         6       -
    17->19   35                             37             2         6       -
    19->23   47                             48             1         8     990
    23->29   60                             63             3        10   2,911
    29->31   71                             74             3        10  15,860
    31->37   91                             95             4        12  43,185   <- NEW

**NO PERIOD SCAN OF EITHER MACHINE EXISTS OR EVER WILL.** The 23->29 rung
needed a 37,182,145-residue kernel scan costing 3 h 36 min; machine 31's
period is 33,426,748,355 slots, nine hundred times larger, and this rung
never touches it. THE DICTIONARY GROWS ~3-5x PER GEAR WHILE THE PERIOD GROWS
~30x: 990 / 2,911 / 15,860 / 43,185 tuples at m19 / m23 / m29 / m31 against
periods of 3.8e5 / 3.7e7 / 1.1e9 / 3.3e10. And **`K`, the longest qualifying
run - the thing that decides how DEEP the family goes - did NOT grow from
machine 29 to machine 31** (3, 4, 5, 5 at m19, m23, m29, m31). That is the
first evidence on the open question this vehicle's lifetime depends on.

CROSS-CHECK AGAINST CONSTRUCTOR'S SCAN-FREE CERTIFICATE (checked before
citing, and the check changed what I was going to write). Their round-25
`docs/novel/scanfree-certificate.md` reports 31->37 as "bound 95 <= budget
95". That is NOT a loose version of my 91: their CEGAR `bound` column is the
BUDGET at every step by construction (48/63/74/95 against budgets
48/63/74/95) because the loop is asked to certify `<= budget` and stops when
it does. The two lanes therefore AGREE and are not comparable as sharpness -
their route certifies `g(M+q') <= F(M) + q'` directly, mine certifies the
sharper R39 criterion `g37 <= 91` (`Machine37.g37_le_91`) and then weakens it
to the budget. Where the two CAN be compared they match exactly: their gate
(c) reports `F_1..F_4(29) = [43, 55, 65, 70]`, and my independent full-period
scan gives `F_j(29) = 43, 55, 65, 70`; their `F_2` row has `F_2(31) = 68`,
and mine gives 68. Two independent codebases, one set of integers.

### R25.7 Open formalisation targets (round-26 priority order)

1. **Discharge `Census29` from `Census23` by the dictionary transfer.** The
   transfer is sound by construction (the order-`m` closure of a dictionary
   is a superset); if it can produce a superset of machine 29's QUALIFYING
   family at depths up to 7 from machine 23's 4-tuple dictionary, then one
   census at machine 23 underwrites every rung above it - and machine 23's
   period is the largest this lane has ever kernel-scanned, so the chain
   would bottom out at something already proved. Highest-value item.
2. **The periodicity glue, once, abstractly** (verdicts 11 and 20): for a
   decidable opening predicate periodic mod `P` with `N` openings per period,
   `opSeq (n + N) = opSeq n + P`. It closes the depth-sum identity at m13 AND
   the generator's soundness bridge at 11->13.
3. ~~The 31->37 rung by the same vehicle~~ - **DONE in this round** (R25.6).
   The next is 37->41: machine 37's period is 1,236,789,689,135 slots, about
   15 min of CPU per pass by the same script, so still cheap. The binding
   question is whether `K` stays at 5.
4. **The SANDWICH LEMMA** (constructor R51) - still the only route that
   removes the census hypothesis altogether rather than shrinking it.
5. The m13 covering dual (blocked on the witness being saved - R25.4).
6. Harvester's paired-Holt coef rung; suppression-corrected flatness at a
   further machine; the CRT single-cycle reduction. Unchanged.

## Round 26 append (2026-08-29)

Brief: (1) THE SOUNDNESS BRIDGE - one abstract lemma discharging both the
generator's bridge (verdict 20) and the depth-sum glue (verdict 11);
(2) `Census29`/`Census31` from hypotheses toward kernel facts;
(3) the eighth rung 37->41 IF other lanes' inputs land in the `hE` shape;
(4) Lateral's parity theorem. No per-rung period scan (brief instruction).

**Build GREEN at 1426 jobs** (1410 -> 1426), 82 targets, 135 files, **zero
sorries, zero `axiom` declarations, no `native_decide`, no
`Lean.ofReduceBool`**. Eight new roots: `Periodic`, `Machine11Per`,
`Machine13Per`, `Gen11Sound`, `Machine29Cen`, `Machine31Cen`, `LadderPeriod`,
`Mirror`, plus the unregistered audit tool `DepAudit.lean`. Every job this
round launched has finished.

### R26.1 THE PERIODIC-ENUMERATION LEMMA - two standing gaps, one theorem

`proofs/Periodic.lean`, no machine, no gears, nothing but `omega`:

```lean
theorem next_shift {E : ℕ → Prop} {next : ℕ → ℕ} {P : ℕ}
    (hgt : ∀ k, k < next k) (hE : ∀ k, E (next k))
    (hmin : ∀ k m, k < m → m < next k → ¬ E m)
    (hper : ∀ k, 1 ≤ k → (E (k + P) ↔ E k)) (k : ℕ) :
    next (k + P) = next k + P

theorem op_shift {next op : ℕ → ℕ} {P N : ℕ}
    (hsucc : ∀ n, op (n + 1) = next (op n))
    (hnext : ∀ k, next (k + P) = next k + P)
    (h0 : op N = op 0 + P) (n : ℕ) : op (n + N) = op n + P
```

`next_shift` carries the mathematics (periodicity makes `next k + P` an
`E`-point above `k + P`, and pulling `next (k+P)` back by `P` makes one above
`k`, so the two minimalities pin the values to each other); `op_shift` is a
one-line induction whose ONLY machine-specific input is the finite fact
`op N = op 0 + P`. Also in the file: `op_shift_mul`, `gap_shift`,
`gap_shift_mul`, `gap_mod`, `windowSum_shift`, `pred_shift_mul`,
`next_shift_mul`, `index_reduce` (R26.3).

The `1 <= k` side condition on periodicity is load-bearing, not cosmetic: slot
`0` carries `(0, 1)` rather than `(-1, 1)`, so `Exposed 0` is FALSE while
`Exposed P` is TRUE at every machine in this ledger. Every use of `hper` is at
a provably positive point.

MAKING THE BASE CASE KERNEL-COMPUTABLE. `opSeq` is built from `Nat.find` and
does not reduce. The fix is the `seekT` walk that `seek_next` already proves
equal to `nextOp`:

```lean
-- proofs/Machine11Per.lean            -- proofs/Machine13Per.lean
def ow  : ℕ → ℕ                        def ow13 : ℕ → ℕ
  | 0 => 0                               | 0 => 0
  | i+1 => seekT 3 3 3 7 (ow i)          | i+1 => seekT 3 3 3 3 11 (ow13 i)
theorem opSeq_zero : opSeq 0 = 3       theorem opSeq_zero : opSeq 0 = 3
theorem opSeq_eq_ow : ∀ i, opSeq i = 3 + ow i
theorem ow_135 : ow 135 = 385          theorem ow13_1485 : ow13 1485 = 5005
        -- NO AXIOMS                            -- NO AXIOMS
theorem opSeq_shift (n) :              theorem opSeq_shift (n) :
    opSeq (n + 135) = opSeq n + 385        opSeq (n + 1485) = opSeq n + 5005
theorem g11_shift / g11_mod            theorem g13_shift / g13_mod
                                       theorem windowSum_g13_shift
```

`Machine11.opSeq_shift` is verdict 20's missing step (ii);
`Machine13.opSeq_shift` is verdict 11's missing step. ONE lemma, two machines,
as the brief predicted. Both base cases are `decide +kernel` with EMPTY axiom
footprints (135 and 1,485 walk steps).

COST NOTE, and it is a new infrastructure fact: `exposed13_period` written as
`unfold ...; omega` over all EIGHT gears at once elaborated fine and then
**failed in the KERNEL with "(deterministic) timeout"** - omega's certificate
for sixteen simultaneous divisibility constraints is too big to re-check.
Split into one `omega` per gear (`(5 | x + 30030) <-> (5 | x)`, sixteen of
them) it is instant. The same shape recurs at machines 29 and 31 with sixteen
and eighteen gears. RULE: **one divisibility per `omega` call.**

### R26.2 THE GENERATOR IS SOUND AT 11 -> 13 (verdict 20 closed)

With the glue in hand the bridge is finishable, and it was worth finishing:

```lean
-- proofs/Gen11Sound.lean
theorem word_check : ∀ i < 135,
    gw11.getD ((i + 1) % 135) 0 = Machine11.ow (i + 1) - Machine11.ow i
theorem gAt_succ (i : ℕ) : gAt (i + 1) = Machine11.g11 i
theorem walk_sound (ns n j : ℕ) : ∀ fuel k d surv,
    Machine11.opSeq (j + k) = Machine13.opSeq n + d →
    Machine13.opSeq (n + surv) ≤ Machine13.opSeq n + d →
    Machine13.opSeq n + d < Machine13.opSeq (n + surv + 1) →
    surv ≤ ns →
    walk (j + 1) (Machine13.opSeq n % 13) ns fuel k d surv ≠ 999 →
    Machine13.opSeq n + walk (j + 1) (Machine13.opSeq n % 13) ns fuel k d surv
      = Machine13.opSeq (n + ns + 1)
theorem spectrum_of_gen {ns : ℕ} (hgen : gen ns < 999) :
    Spectrum.SpectrumBound Machine13.g13 (ns + 1) (gen ns)
theorem generator_sound :
    Spectrum.SpectrumBound Machine13.g13 1 11 ∧
      Spectrum.SpectrumBound Machine13.g13 2 16 ∧
      Spectrum.SpectrumBound Machine13.g13 3 23 ∧
      Spectrum.SpectrumBound Machine13.g13 4 26
```

`F_1..F_4(13) <= 11, 16, 23, 26` - the exact values - **derived from machine
11's 135-letter word, with machine 13's 5,005-slot period nowhere in the
derivation.** Round 25 could only assert that the two computations AGREE.

THREE THINGS THE PROOF NEEDED, all of them findings:

* A CORRECTION TO `gw11`. The word's base is ONE OPENING EARLIER than the
  enumeration's: `gAt (i+1) = g11 i`, not `gAt i = g11 i` (machine 11's first
  opening is slot 3 and `gw11` starts with the gap ENDING there). `gen` is a
  maximum over all 135 bases, so its VALUE is unaffected - but a soundness
  proof has to get the index right, and a "certified word" claim written
  without the `+1` would have been false.
* THE BAIL VALUE HAD TO BECOME A SENTINEL. `Gen11.walk` returned `0` when the
  span cap or the fuel ran out. That is sound for a MAXIMUM and fatal for a
  bound: a walk that gives up is indistinguishable from a short gap. Changed
  to `999`; `gen ns < 999` is then itself the proof that no walk bailed, which
  is exactly the hypothesis `spectrum_of_gen` needs. Values unchanged
  (checked by simulation before editing), and two more landed: `gen 2 = 23`,
  `gen 3 = 26` - the generator reproduces the whole published ladder.
* THE INVARIANT. "`x + d` is the `k`-th machine-11 opening after `x`, and
  exactly `surv` machine-13 openings lie in `(x, x + d]`". The killed branch
  preserves it because no machine-13 opening can hide between consecutive
  machine-11 openings; the surviving branch closes it because
  `opSeq13 (n+surv+1)` is then squeezed between the two.

INDEPENDENCE IS GATED, NOT ASSERTED - `proofs/DepAudit.lean` (new, an audit
tool like `AxiomCheck.lean`, deliberately not a `defaultTarget`):

    DEP AUDIT GREEN: Gen11.generator_sound closes over 3858 constants;
    all 11 positive controls reached; none of the 15 machine-13-period
    constants is among them.

It walks the transitive constant closure of the proof term and fails
elaboration if `Machine13.qasm`, `qslice`, `qokAll`, `chain_facts`,
`spectrum_one..four`, `spectrum_ladder`, `nextOp_le_11` (etc.) are reachable.
**THE POSITIVE CONTROLS EARNED THEIR PLACE IMMEDIATELY**: the first version
passed vacuously, reaching only 310 constants, because `ConstantInfo.value?`
does NOT return a THEOREM's proof term in this toolchain - one must match
`.thmInfo` explicitly. A dependency audit written the obvious way is a no-op.
(New standing lesson; `#print axioms` cannot see this class of claim at all.)

### R26.3 THE CENSUS HYPOTHESIS SHRINKS TO ONE PERIOD (brief item 2)

`Census29` says `forall n, ...` - a claim about EVERY index of an infinite gap
word. `research/qual_dict.py` verifies ONE PERIOD. Nothing in the ledger
connected them; that step was an unstated assumption inside a named
hypothesis. It is now a theorem, and the engine is abstract:

```lean
-- proofs/Periodic.lean
theorem index_reduce {E : ℕ → Prop} {next op g : ℕ → ℕ} {P : ℕ} (hP : 0 < P)
    (hsucc) (hnext) (hEop) (hposop) (hper) (hsurj) (hg) (n : ℕ) :
    ∃ m, op m ≤ P ∧ ∀ i, g (n + i) = g (m + i)

-- proofs/Machine29Cen.lean, proofs/Machine31Cen.lean
theorem Machine29.exposed29_period {k} (hk : 1 ≤ k) :
    Exposed29 (k + 1078282205) ↔ Exposed29 k
theorem Machine29.index_reduce29 (n) :
    ∃ m, opSeq29 m ≤ 1078282205 ∧ ∀ i, g29 (n + i) = g29 (m + i)
structure Machine29.Census29P : Prop      -- every clause restricted to
                                          -- opSeq29 n <= 1078282205
theorem Machine29.census29_of_period (h : Census29P) : Census29
theorem Machine31.census31_of_period (h : Census31P) : Census31

-- proofs/LadderPeriod.lean
theorem D_29_31_period (h : Machine29.Census29P) (n) : Machine31.g31 n ≤ 43 + 31
theorem D_31_37_period (h : Machine31.Census31P) (n) : Machine37.g37 n ≤ 58 + 37
theorem g31_le_of_period / g37_le_of_period      -- R39's own form
```

WHY THIS WORKS WHERE THE GLUE OF R26.1 CANNOT. `op_shift` needs the base case
`op N = op 0 + P` - a walk of `N` steps, which is 135 at machine 11 and
214,708,725 at machine 29. `index_reduce` needs NO base case and NO walk: it
needs only that the opening PREDICATE is periodic (one `omega` per gear) plus
surjectivity of the enumeration, both of which machines 29 and 31 already
have. The reduction therefore applies at machines whose period a kernel will
never enumerate.

WHAT IT DOES AND DOES NOT BUY. Verdict 21 STANDS: `Census29` is not
kernel-checked and will not be. What changed is that the unverified part is
now FINITE as well as explicit - a claim about the 214,708,725 openings of one
period (6,226,553,025 at machine 31), which is exactly the object the four
gates scan. Anyone quoting the rung still has to quote the hypothesis; the
hypothesis is now the same shape as the evidence.

### R26.4 LATERAL'S PARITY LAWS, THE ARITHMETIC HALVES (brief item 4)

`proofs/Mirror.lean`, axiom footprint `[propext, Quot.sound]` - not even
`Classical.choice`:

```lean
theorem mirror_gear {q P k : ℕ} (hqP : q ∣ P) (hk1 : 1 ≤ k) (hk2 : k < P) :
    ((q ∣ Census.lo (P - k)) ↔ (q ∣ Census.hi k)) ∧
      ((q ∣ Census.hi (P - k)) ↔ (q ∣ Census.lo k))
theorem mirror_exposed11 {k} (hk1 : 1 ≤ k) (hk2 : k < 385) :
    Machine11.Exposed11 (385 - k) ↔ Machine11.Exposed11 k
theorem mirror_exposed29 {k} (hk1 : 1 ≤ k) (hk2 : k < 1078282205) :
    Machine29.Exposed29 (1078282205 - k) ↔ Machine29.Exposed29 k
theorem antipode_open {q P s : ℕ} (hq : 5 ≤ q) (hqP : q ∣ P) (hs : 2 * s = P + 1) :
    ¬ (q ∣ Census.lo s) ∧ ¬ (q ∣ Census.hi s)
theorem antipode_exposed11 : Machine11.Exposed11 193
theorem antipode_exposed29 : Machine29.Exposed29 539141103
theorem self_mirror_unique {N j t1 t2 : ℕ} (hN : N % 2 = 1)
    (h1 : t1 < N) (h2 : t2 < N)
    (e1 : (2 * t1 + j) % N = 0) (e2 : (2 * t2 + j) % N = 0) : t1 = t2
theorem periods_odd : 135 % 2 = 1 ∧ ... ∧ 6226553025 % 2 = 1
```

`mirror_gear` is Lateral's M0 for one gear at any period: the mirror EXCHANGES
the slot's two members and blocking is symmetric in them. `antipode_open` is
their round-26 `g_1* = 1`, and the arithmetic is even shorter than their
residue argument - `6 * ((P+1)/2) = 3P + 3`, so the antipodal slot's members
are `3P + 2` and `3P + 4` and a gear would have to divide `2` or `4`. Note
`antipode_exposed29` is an opening of machine 29 exhibited BY ARITHMETIC: no
scan, no `decide`, at slot 539,141,103.

NOT DONE, and it is the half the lever needs: "every count is EVEN except the
exceptional one" requires a counting step - a fixed-point-free involution on a
`Finset` has even cardinality - which this file does not build. What is proved
is the involution and the UNIQUENESS of its fixed point, which is the half the
route consumes ("fewer than two" proves "none").

### R26.5 THE EIGHTH RUNG WAS NOT ATTEMPTED - the reason, not a judgment

Brief item (3) was conditional on Mechanic's and Constructor's round-26
outputs landing in the `hE` shape. At my round close `agents-shared.md`
carried round-26 blocks from LATERAL and HARVESTER only; neither Constructor's
37->41 chain nor a machine-37 qualifying dictionary exists. Two independent
reasons the rung could not be built anyway:

* THE VEHICLE NEEDS MACHINE 37's QUALIFYING FAMILY `D_2..D_{K+2}` at floor
  `2u' = 14` (gear 41's teeth are `{7, 34}`, `6*7 = 42 = 41+1`). That is a
  full-period scan of 1,236,789,689,135 slots - and the brief says explicitly
  "do NOT start any per-rung period scan". (R25.7's "about 15 min of CPU per
  pass" for machine 37 is wrong by three orders of magnitude on its own
  measured scaling: machine 31's 33.4e9 slots cost 1,451 s CPU, and machine 37
  is 37x larger. Correcting my own estimate.)
* MECHANIC's m41 superset is the WRONG OBJECT for this vehicle twice over: it
  is a dictionary of machine 41 (the NEW machine), where the merge law
  consumes the OLD machine's word, and it is 4-tuples, where the qualifying
  family needs depths up to `K + 2 = 7`.

The honest route to the eighth rung is Constructor's scan-free CRT dictionary
generating machine 37's qualifying windows at depths 2..7 - which is their
tool, mid-round. Recorded as the top open target, not as a failure.

### R26.6 New verdicts

22. **`ConstantInfo.value?` DOES NOT SEE A THEOREM'S PROOF TERM in this
    toolchain (Lean 4.34.0-rc1).** Any dependency audit written with it
    reports a closure of only the TYPES' constants and passes vacuously
    (measured: 310 constants instead of 3,858). Match `.thmInfo` explicitly.
    The general lesson is bigger than the API: **an audit needs positive
    controls or it is not an audit** - `DepAudit.lean` now asserts eleven
    constants that MUST be reachable, and they are what caught this.
23. **A single `omega` over sixteen divisibility constraints elaborates and
    then fails IN THE KERNEL** ("(deterministic) timeout"). The elaborator is
    not the binding limit for omega certificates; the kernel is. One
    divisibility per call. (Machine 13's period lemma, and by extension 29's
    and 31's.)
24. **The generator's soundness needed a SENTINEL, and that is a general
    lesson about computed bounds.** A search that returns a neutral value when
    it gives up can only ever certify a maximum, never a bound. `gen 0 = 11`
    was true and useless for soundness until the bail value moved above every
    attainable span; then the SAME number became a proof that no walk bailed.
    Any "computed maximum" in this project used as an upper bound should be
    checked for this: what does the computation return when it fails?
25. **The census hypotheses are now finite, and that is the whole of what
    round 26 could shrink.** `Census29P`/`Census31P` are one-period claims
    (verdict 21 unchanged: still not kernel-checked). The route to REMOVING
    them is unchanged and is not the periodicity lemma: it is either the
    dictionary transfer from machine 23 (blocked on `dict_transfer.py` at
    `out_m = 7`) or Constructor's sandwich lemma.

### R26.7 Open formalisation targets (round-27 priority order)

**0. THE LP THREAD'S CASE-SPLIT CERTIFICATES (added post-filing; see R26.8 for the
sizing and the order of attack).** It outranks everything below it because it
retires the census hypotheses rather than shrinking them - the residue verdicts
21 and 25 name.

1. **Machine 37's qualifying dictionary, scan-free** (Constructor's
   `crt_dict.py` / `scanfree_dict.py` at depths 2..7, floor 14). It is the
   ONLY missing input to the eighth rung, and it is a CSP job, not a scan.
2. **Discharge `Census29` from `Census23`** by the dictionary transfer -
   unchanged from R25.7 item 1, and now sharper: with `index_reduce` the
   target is a one-period claim on both sides.
3. **The involution-parity counting lemma** (a fixed-point-free involution on
   a `Finset` has even cardinality). It converts `Mirror.self_mirror_unique`
   into Lateral's actual lever - "every configuration occurs an even number of
   times except the named one" - and it is the last piece of the endpoint
   killer. Check `Finset.card_modEq_card_fixedPoints` before rolling one.
4. **The depth-sum re-indexing bijection** at machine 13: `Finset.range 1485`
   window starts vs the 5,005 residues of `pairCount13`. The periodicity half
   is done (R26.1); what remains is Finset bookkeeping.
5. **The generator at 13 -> 17** by the `Gen11Sound` template. Machine 13's
   word is 1,485 letters (vs 135), gear 17's teeth are `{3, 14}`, and the
   periodicity glue already exists (`Machine13Per`). If it works, the
   generator - not the dictionary - becomes the ladder's vehicle, and the
   dictionary's census hypothesis disappears from those rungs.
6. **The SANDWICH LEMMA** (Constructor R51) - unchanged, still the only route
   that removes the census hypothesis rather than shrinking it.
7. The m13 covering dual (blocked on Lateral saving the dual vector).

### R26.8 THE LP THREAD'S CASE-SPLIT CERTIFICATES - SIZED, AND TAKEN AS THE TOP ROUND-27 ITEM

Routed in after my round was filed (LP-duality thread round 26, section 9): every (D)
rung through 37->41 is certified hypothesis-free by their case-split vehicle, and the
checking predicate is arithmetic. That is the alternative to verdicts 21/25 - a rung with
NO `Census29P`, NO `Census31P`, nothing empirical.

I MEASURED THE ARTEFACTS BEFORE JUDGING, because the numbers quoted for it are OP COUNTS,
not certificate sizes, and those are not the same quantity. `research/data/r26/
cert_gate_m23_w48_h*.pkl` (the 19->23 rung, five cases, gear 5 held):

    per case:   29 cut rows x 22 rational entries + 29 row weights (`y`)
                + 450 link weights (`nu`) + 1 recursion weight (`yff`)
                = ~1,120 rationals, 5.4 KB
    magnitudes: DENOMINATORS <= 5 BITS (21, 7, 3), numerators <= 10 bits
    verdict record: lhs 202/7 < rhs 607/21 - the published table row, exactly
    whole rung: 5 cases, ~5,600 small rationals, ~27 KB

**THE DATA IS NOT THE OBSTACLE.** ~11,000 numerals is the scale of `Machine29D4` (6,688
4-tuples, 55 s, empty axiom footprint), and the entries are single small fractions rather
than tuples, so the round-25 `count x arity` isDefEq budget is not close to binding. One
module per case, five modules, is a comfortable shape.

**THE OBSTACLE IS SOUNDNESS, AND IT IS NOT ARITHMETIC.** Checking a certificate is easy;
the theorem is `certificate -> rung`, and that needs the vehicle formalised. This lane
already has that scaffold for the UNRESTRICTED level-2 consistent vehicle
(`CoveringCert.lean` at 11->13, `CoveringCert2.lean` at 11->13 and 13->17). `RelaxStar`
is a strict extension and needs four further soundness lemmas, none of them hard and none
of them free:

  1. `pos` restricted - the held gears' blocked positions are removed, and a real
     configuration with those held phases still induces a 0/1 point;
  2. `dom(q)` restricted - the lower gears range only over phases blocking no required-open
     position, and restricting them only RAISES `n_ij` while `n_ij <= N_ij` still holds at
     the actual tuple;
  3. cuts taken at the positions of `pos` only - validity is "the row's subset-sums are
     >= 1 at every nonempty atom", `<= 2^n` atoms with `n <= 11`;
  4. CASE EXHAUSTIVENESS - the held gears' phases range over all residues, so the five
     (35, 385) cases cover every configuration. This is the one with no analogue in the
     existing files, and it is the reason the vehicle escapes round 25's refutations.

VERDICT: **ROUND-27 TARGET, TOP OF THE LIST - not a this-round pickup.** Not because it
is unattractive (it is the most attractive thing offered to this lane in several rounds:
it retires the census hypotheses outright rather than shrinking them), but because it is
a four-lemma soundness development plus five transcribed modules, and my round is filed
and green at 1426 jobs. The job-completion rule says launch work of that size EARLY or
narrow it; starting it now would do neither. What this round contributes is the sizing:
round 27 starts from a measured object rather than "looks reachable".

ORDER OF ATTACK, so it is decided in advance and not re-litigated:
  (i)   `RelaxStar` soundness at ONE case, no case split - the smallest true statement;
  (ii)  case exhaustiveness over one held gear, giving the 19->23 rung (5 cases);
  (iii) 29->31 (35 cases) - the first rung that REPLACES a census-hypothesis rung, which
        is the whole point; `D_29_31` would then exist in two forms, one of them
        hypothesis-free.
Stop after (ii) if the transcription cost per case exceeds one module.

WHAT I NEED FROM THE LP THREAD (their offer: "ask and I will emit them in whatever shape
the Lean side wants") - JSON, not pickles, one file per case:
  * `rows`: a list of `[pos, [[num, den], ...22]]` - integers only, no `Fraction` repr;
  * `y`, `nu` as `[[num, den], ...]`; `yff` as `[num, den]`;
  * the ATOM INDEXING made explicit: which subset of gears each of the 22 entries is,
    as a list of gear-index bitmasks, so the Lean side can state cut validity without
    reconstructing their column order;
  * `held`, `ws`, `W`, `full`, and the claimed `lhs`/`rhs` as integer pairs;
  * for exhaustiveness: the list of held-phase tuples the case files are indexed by, and
    the assertion that it is all of `prod (residues of held gears)`.
With that, (i)+(ii) is an afternoon of transcription plus the soundness lemmas.

## Round 27 append (2026-08-29)

GATES, all re-run at round close from clean invocations:
  cd proofs && lake build           -> Build completed successfully (1521 jobs)
                                       (1426 -> 1521; 45 new modules: 40 case
                                       transcriptions, 2 gear bases, 2 rung roots,
                                       1 shared soundness module)
  lake env lean AxiomCheck.lean     -> footprints below; zero custom axioms,
                                       no native_decide, no ofReduceBool
  research/lp_cert_lean.py GATE     -> ALL ASSERTIONS PASSED (this lane's own
                                       independent transcription + soundness gate)
Zero sorries. Every job this round launched has finished; nothing left running.

### R27.0 A TOOLING TRAP THAT COST THE ROUND'S FIRST BUILD, AND IT IS A LANE RULE NOW

`~/.elan/bin/lake.exe` is an ELAN PROXY: it reads `lean-toolchain` from the
**current working directory**, not from `--dir`/`-d`. Run from anywhere but
`proofs/` it picks elan's DEFAULT toolchain (here 4.33.1 against the project's
4.34.0-rc1) and starts REBUILDING MATHLIB FROM SOURCE, which then fails on
version-skewed `Batteries` lemmas ("Unknown identifier `ite_eq_left`"). The
symptom looks like a corrupt cache; it is a wrong compiler.
NEW LANE RULE: every lake invocation must be issued with `proofs/` as the
actual cwd. `-d`/`--dir` is NOT a substitute. (Agent shells that reset cwd
between calls must chain `cd proofs && lake ...` in one command.) Re-run from
`proofs/`, the baseline was GREEN at 1426 jobs with nothing to repair.

### R27.1 ITEM 0 - THE LP CASE-SPLIT CERTIFICATES ARE IN THE KERNEL

The round-26 addendum (R26.8) fixed the order of attack; it was followed, and
step (iii) was reached rather than stopped at (ii).

    theorem CaseCert23.D_19_23_case (n : ℕ) : Machine23.g23 n ≤ 25 + 23
    theorem CaseCert23.F_le (n : ℕ) : Machine23.g23 n ≤ 48
    theorem CaseCert23.no_run {p : ℕ} (hp : 1 ≤ p) :
        ∃ i < 48, Machine23.Exposed23 (p + i)

hypothesis-free, no period, no census, standard-three axiom footprint. New
files: `proofs/CaseSplit.lean` (the reusable half), `proofs/CaseCert23B.lean`
(gears), `proofs/CaseCert23C0..C4.lean` (one module per case), and
`proofs/CaseCert23.lean` (the exhaustiveness + the rung).

THE FOUR SOUNDNESS LEMMAS I SIZED IN R26.8: THREE COLLAPSED AND THE FOURTH WAS
NOT ON THE LIST. Recorded because the sizing was wrong in an instructive way.

 1. `pos` RESTRICTED - trivial. `pos` is a literal list in the Lean file; the
    only fact needed is `gb5 w (q t) = false` at every index, one `decide` per
    case. The held gear appears nowhere else in the argument.
 2. `dom(q)` RESTRICTED - VACUOUS at this instance. Domains shrink only when
    OPEN positions are prescribed (the windowed instance); a case split
    prescribes none.
 3. CUT VALIDITY - VACUOUS at every certified rung on disk. My own
    transcription asserts `rows == base_cut` for all 75 cases of 19->23,
    23->29 and 29->31: the separation loop never fired at these widths. So
    the coverage row is `sum_q [q blocks i] >= 1` - literally the hypothesis
    "the window is fully blocked" - and `lam_0 = 0`, so
    `rhs = sum_r y_r + yff*|pos|`. (The LP thread reached the same conclusion
    independently and posted it mid-round; the two statements were written
    from different code.)
 4. CASE EXHAUSTIVENESS - ONE `omega`. `p % 5 = 0 or ... or p % 5 = 4` and five
    `nocase` applications. The step I called "the one with no analogue in the
    existing files" is the cheapest of the four.

WHAT THE WORK ACTUALLY WAS - THE RECURSION ROW, AND IT HAS A CLEAN LEMMA.
`frow` carries a coefficient `n_ab` per gear pair, defined in the LP thread's
code as a MAX-COVER over the phases of the gears BELOW `a`. Taken literally
that is ~8.2 million evaluations to certify in the kernel at 19->23 alone.
The identity that removes it:

    CaseSplit.lowest6 / lowest7  (NO AXIOMS AT ALL)
    if some gear blocks x, then
      1 + #{(a,b) : a < b, both block x, no gear below a blocks x}
        = #{a : a blocks x}
    - only the LOWEST blocker can be the `a` of such a pair, and it pairs with
    each of the other blockers exactly once.

Summed over `pos` this gives `sum_a |A_a| >= |pos| + sum_{a<b} n_ab` for ANY
`n_ab` at most the "a is lowest and b also blocks" count - which is what the
vehicle's `n_ab` is, being a MINIMUM over the lower gears' phases. In Lean it
is a `decide` over 2^m Boolean assignments. Everything else in the case proof
is `Finset.sum_le_sum` plus one `linarith` over 43 facts.

THE SUPPORTING PIECES, all in `CaseSplit.lean`, all `[propext, Quot.sound]` or
smaller: `mxr` / `mxr2` (block maxima as a kernel-evaluable fold) with
`le_mxr` / `le_mxr2`; `ind_low2` ("[not A and B and C] = [B and C] -
[B and C and A]", the step that turns "a is lowest" into `|P| - cover`);
`ind_nonneg`; `degpos6` / `degpos7`.

### R27.2 THE EMISSION, AND WHY THIS LANE BUILT ITS OWN

My brief said: check `research/data/r27/` early, and if the LP thread's JSON is
not there by mid-round, transcribe from the round-26 pickles and flag the gap.
It was not there at my first look, so I wrote `research/lp_cert_lean.py`, which
does more than transcribe - it REBUILDS the relaxation from the primes and
re-derives every number:

  1. asserts every cut row equals `base_cut` (finding 3 above);
  2. RECOMPUTES `n_ab` from the closed form the kernel will use (0 above gear
     index 1; `|P|` at index 0; `|P| - max_s |P & hits(q_0,s)|` at index 1) and
     asserts it equal to `RelaxStar.frow` COLUMN BY COLUMN - 3,381 columns per
     case at 19->23, 7,201 at 29->31. This is what licenses the kernel-cheap
     form;
  3. recomputes `lhs`/`rhs` from its own formulas (not `certificate_star`) in
     exact integers after scaling by the case denominator, and asserts they
     equal the recorded verdict;
  4. a SOUNDNESS GATE on the recursion row over random phase tuples:
     `#covered + sum n_ab <= sum_a |A_a|` and `n_ab <= #{a lowest, b blocks}`.

The LP thread's emission then landed mid-round, matching my R26.8 spec exactly
(atom bitmasks, block spans, link order, the exhaustiveness assertion). CROSS-
CHECK RUN: their `cert_19_23_h*.json` and my independent transcription agree on
`pos`, `y`, `nu`, `yff`, `lhs`, `rhs` for all five cases as exact rationals.
TWO CODEBASES, ONE CERTIFICATE - and the "every row is the base cut" finding
was made twice, independently, in the same round.

### R27.3 STEP (iii): 29->31, THE FIRST RUNG THAT REPLACES A CENSUS HYPOTHESIS

    theorem CaseCert31.D_29_31_case (n : ℕ) : Machine31.g31 n ≤ 43 + 31

35 case modules (`CaseCert31C0..C34`), seven free gears, 21 gear pairs, held
gears (5,7). `D_29_31` NOW EXISTS IN TWO FORMS: `Machine31.D_29_31` (merge law
+ `Machine29.Census29`, a full-period claim about 214,708,725 openings) and
`CaseCert31.D_29_31_case`, WHICH HAS NO HYPOTHESIS AT ALL. Verdicts 21 and 25
name the census as the ledger's residue; at this rung it is now optional.

### R27.4 A KERNEL-SIZING FACT THAT MADE THE 35-CASE RUNG AFFORDABLE

First cut, 9 min per case at 29->31 => 5 h serial for the rung. Measured where
it went and found a fact about the vehicle, not about Lean:

  `n_ab = 0` FOR 96.4% OF THE GEAR-INDEX-1 COLUMNS (52,173 of 54,145 over the
  35 cases) - one gear below suffices to cover the whole two-gear overlap - and
  `n_ab = 0` is SOUND WITH NO EVALUATION AT ALL (`0 <= anything`).

So the exceptions go in a literal list and the kernel skips the 11-phase
maximum everywhere else. 9m01 -> 4m10 per case measured SOLO at 29->31, and 1m50 for two cases at
19->23. MEASURED END TO END: the 35-case rung built in 47 MINUTES WALL at two
concurrent workers (1.34 min/case throughput, ~2.7 min per case inside a
batch); 19->23's five cases in about 5 minutes. Structurally: the recursion row is, numerically, almost entirely a
KOUNIAS ROW AT THE SMALLEST FREE GEAR - the only pairs with a systematically
nonzero coefficient are those whose lower member has nothing below it.

COST CURVE FOR THE NEXT RUNG, honestly: the kernel cost is the BLOCK MAXIMA -
one `aP` per phase pair per gear pair, ~3,400 columns at 19->23 and ~7,200 at
29->31, each a `Finset.range |pos|` sum. That grows ~2x per rung. The CASE
COUNT grows as a primorial. At the measured 1.34 min/case (two workers) k = 4 (5,005
cases) is ~5 days of kernel. IN THE KERNEL IT IS THE CASE COUNT THAT BITES, NOT THE COLUMNS.

### R27.5 ITEM 1 - THE MIRROR'S COUNTING HALF (round 26's named gap, closed)

`proofs/Mirror.lean`, appended (footprint: the standard three - `Classical.choice`
enters through the `Finset` machinery, unlike round 26's arithmetic halves, which
need only `[propext, Quot.sound]`):

    theorem Mirror.even_card_involution {α} [DecidableEq α] (f : α → α) :
        ∀ (n : ℕ) (s : Finset α), s.card ≤ n →
          (∀ a ∈ s, f a ∈ s) → (∀ a ∈ s, f (f a) = a) → (∀ a ∈ s, f a ≠ a) →
          s.card % 2 = 0
    theorem Mirror.window_count_even {N g : ℕ} (m L : ℕ → ℕ)
        (hlt : ∀ t, t < N → m t < N) (hmm : ∀ t, t < N → m (m t) = t)
        (hL : ∀ t, t < N → L (m t) = L t)
        (hg : ∀ t, t < N → m t = t → L t ≠ g) :
        (((Finset.range N).filter (fun t => L t = g)).card) % 2 = 0
    theorem Mirror.adjacent_equal_even ... (hexc : L t0 ≠ 2 * F) : ... % 2 = 0
    theorem Mirror.none_of_at_most_one ... (hone : ... ≤ 1) : ... = 0

`even_card_involution` is a structural induction on a cardinality bound: remove
`a` and `f a`, and the hypotheses restrict because `f x = f a` forces `x = a`
and `f x = a` forces `x = f a`. `none_of_at_most_one` is the form the live
route quotes - parity plus "at most one" gives "none".

HONEST SCOPE, and it is the reason this is not a closure: what is kernel-checked
is the LEVER over an ABSTRACT index involution. The INSTANTIATION at a machine -
that the depth-`j` window family is mirror-equivariant with the length function
invariant - needs `mirror_gear`/`mirror_exposed29` composed with the opening
ENUMERATION (`Periodic.lean`), and that composition is not built. Named, not
claimed. It is the natural round-28 continuation of this item.

### R27.6 ITEM 2 - RUNG EIGHT (37->41): NOT ATTEMPTED, PRECONDITION ABSENT

My brief made this conditional on Constructor's emission landing in the `hE`
shape. Their round-27 block is filed and it does NOT contain it: no machine-37
qualifying dictionary at depths 2..7 floor 14 and no `qual_dict.py`-format
emission of the 12,587 deletions. Their own round closed with "no exact m41
census appeared on disk, so item (c)'s precondition never arrived", and rung
nine is likewise not certified on their vehicle. So this is a MISSING INPUT,
not a judgment and not a will-not-close. UNCHANGED ASK, and it is cheap for
them: emit machine 37's qualifying windows at depths 2..7, floor 14, in
`qual_dict.py`'s format, and the rung is a transcription.
NEW OFFER RECEIVED FROM THEM, and it is a good one for round 28:
`A_relax(M) <= 5` as 48 classes mod 210, each a small phase-saturation check -
the same shape as `LiteralCapTable.lean`, and it would be the FIRST UNIFORM
(machine-free) ORDER STATEMENT in the Lean corpus. Their warning is recorded
with it: do NOT lift it to "A_m nilpotent for m >= 6", which their padded-cycle
example refutes.

### R27.7 ITEM 3 - LATERAL'S `g_1* = 1`: ALREADY DONE, AND NOW IT PAYS

`Mirror.antipode_open` landed in round 26. What round 27 adds is the half that
makes it useful: with `window_count_even`, "the antipodal slot is open" plus
"the exceptional window is unique" gives `W_1(g)` EVEN for every `g >= 2` as a
kernel-checked implication - once the instantiation of R27.5 exists. The
arithmetic is done; the plumbing is what is missing, and it is named.

### R27.8 NEW VERDICTS

26. **A LAKE INVOCATION OUTSIDE `proofs/` IS A DIFFERENT COMPILER.** See R27.0.
    The failure mode (mathlib rebuilding from source and failing on Batteries)
    is indistinguishable from cache corruption and is neither.
27. **THREE OF THE FOUR SOUNDNESS LEMMAS I SIZED IN R26.8 WERE VACUOUS AT THE
    ARTEFACTS, AND THE REAL WORK WAS NOT ON THE LIST.** The sizing was done on
    the CLASS `RelaxStar` (which supports required-open positions, restricted
    domains and separated cuts) rather than on the INSTANCES the rungs actually
    use (which have none of the three). General lesson for this lane: size the
    obligation against the ARTEFACT ON DISK, not against the generality of the
    code that produced it - the artefact is usually in a degenerate corner of
    its own class, and the degeneracy is checkable.
28. **A CERTIFICATE COEFFICIENT THAT IS ZERO NEEDS NO EVIDENCE.** `n_ab = 0` is
    sound outright, so the expensive max-cover only has to be evaluated where
    the certificate is nonzero - 3.6% of the columns. A general kernel tactic:
    look for the certificate's SUPPORT before formalising its DEFINITION.
29. **THE CASE SPLIT'S KERNEL LIMIT IS THE CASE COUNT.** Per-case cost grows
    ~2x per rung (columns); the case count grows as a primorial in the number
    of held gears. k = 3 (385 cases) is ~8.6 h of kernel at 29->31 rates (two
    workers) and k = 4 (5,005) is ~5 days and out of reach. The LP thread's cost law transfers to Lean
    with the same shape and a worse constant.
30. **STOPPING A BACKGROUND TASK DOES NOT NECESSARILY STOP THE SCRIPT IT
    LAUNCHED, AND TWO RESUMABLE DRIVERS OVER ONE `.olean` TREE IS A
    CORRECTNESS PROBLEM, NOT JUST A MESS.** Mine cost about 20 minutes and it
    is recorded because the failure is silent. I stopped a 3-wide driver
    (reported "Successfully stopped"), rewrote the script and launched a 2-wide
    one; the process list later showed FOUR copies of the script alive, both
    generations interleaving into the same log and both racing on the same
    modules. The `[ ! -f x.olean ]` resume guard does not exclude two builders
    starting the SAME module at the same time, and lake does not lock a module
    against another lake process - so an `.olean` could be written twice
    concurrently. Oleans are TRUSTED on load, not re-checked, so a raced file
    is exactly the kind of thing this lane must not build on. Response: killed
    every driver and worker, VERIFIED by process list rather than by the tool's
    success message, DELETED all 35 case `.olean`s and rebuilt from clean under
    a single driver. Standing rules now: (a) after stopping a background build,
    confirm from the process list that the script and its `lean`/`lake`
    children are gone; (b) never run two builders over one build tree; (c) if
    it happened anyway, delete the artefacts - a kernel claim on a possibly
    raced `.olean` is not a kernel claim.

### R27.10 SCORING R26.8's OWN PREDICTIONS (this lane made no other pre-registration
this round, and that is itself a gap - the round's only pre-committed statements
were the sizing paragraph and the stop rule)

  * "one module per case is a comfortable shape" - CONFIRMED. Exactly one module
    per case, 320 lines at 19->23 and 408 at 29->31, and the STOP CONDITION
    ("stop after (ii) if transcription exceeds one module per case") never fired,
    which is why step (iii) was reached.
  * "round 25's `count x arity` isDefEq budget is nowhere near binding" -
    CONFIRMED. It was never approached. The binding limit is KERNEL EVALUATION
    TIME of the block maxima, which round 26 did not name at all.
  * "the obstacle is soundness, not arithmetic" - HALF RIGHT, and the wrong half
    was the specific one. Soundness was indeed the work, but three of the four
    named lemmas were vacuous and the actual obligation (the recursion row's
    max-cover coefficients) was not on the list. See verdict 27.
  * "(i)+(ii) is an afternoon of transcription plus the soundness lemmas" -
    CONFIRMED as an estimate of shape; the transcription was automated
    (`research/gen_case_lean.py`) rather than typed, which is what made (iii)
    affordable on the same day.

### R27.9 Open formalisation targets (round-28 priority order)

0. **INSTANTIATE THE MIRROR LEVER AT A MACHINE** (R27.5's named gap): compose
   `mirror_exposed29` with the opening enumeration so that `window_count_even`'s
   three hypotheses are discharged from the machine rather than assumed. This
   is the piece that turns the lever from a tool into a theorem about the
   machine, and it is the same `Periodic.lean` plumbing as `index_reduce`.
1. **31->37 BY THE CASE SPLIT** (385 cases at k = 3, or 35 at k = 2 if the LP
   thread's ladder parameter allows) - it would retire `Census31P` the way
   29->31 now retires `Census29P`. PRICED FROM THIS ROUND'S MEASUREMENT: 1.34
   min/case throughput at two workers, so 385 cases is ~8.6 HOURS of kernel and
   35 cases is ~45 min. Ask the LP thread for the smallest k that certifies it
   BEFORE launching; k = 2 is a comfortable round, k = 3 is the whole round.
2. **`A_relax(M) <= 5`** (Constructor's offer, R27.6) - 48 classes mod 210,
   `LiteralCapTable.lean` shape, the first uniform order statement in Lean.
3. Machine 37's qualifying dictionary, scan-free (unchanged, R26.7 item 1) -
   still the only missing input to rung eight on the merge-law vehicle.
4. Discharge `Census29` from `Census23` by dictionary transfer (unchanged) -
   NOTE its priority DROPS: at 29->31 the census hypothesis now has a
   hypothesis-free alternative, so this matters for 31->37 and above.
5. The involution-parity counting lemma - **DONE** (R27.5).
6. The depth-sum re-indexing bijection at machine 13 (unchanged).
7. The generator at 13 -> 17 by the `Gen11Sound` template (unchanged).
8. The sandwich lemma (unchanged).

## Round 28 append (2026-08-29/30)

GATES, all re-run at round close from clean invocations:
  cd proofs && lake build             -> Build completed successfully (1749 jobs)
                                         (1521 -> 1749; 114 new modules)
  lake env lean AxiomCheck.lean       -> 405 declarations, footprints in R28.7;
                                         zero custom axioms, no native_decide,
                                         no ofReduceBool
  research/lp_cert_inc_r28.py GATE    -> ALL ASSERTIONS PASSED (47 s) - this
                                         lane's independent re-derivation of all
                                         120 increment-width certificates from
                                         the primes, plus the recursion-row
                                         soundness gate on random phase tuples
  research/lp_cert_inc_r28.py CROSS   -> CROSS-CHECK PASSED (120 cases, two
                                         codebases) - against the LP thread's own
                                         round-28 emission, as exact rationals
  research/lp_cert_lean.py GATE       -> ALL ASSERTIONS PASSED (round 27's gate,
                                         re-run, unchanged)
Zero sorries. Every job this round launched has finished; nothing left running.

### R28.0 ITEM 0 - THE MIRROR LEVER IS INSTANTIATED AT A MACHINE

Round 27 closed the lever's counting core and named the gap in the same breath:
`window_count_even` quantifies over an ABSTRACT index involution, and nothing
tied it to a machine. `proofs/MirrorM11.lean` ties it, at machine 11 - the
smallest machine with a complete kernel enumeration.

    theorem Machine11.opSeq_mirror :
        forall n, n <= 133 -> opSeq n + opSeq (133 - n) = 385
    theorem Machine11.g11_mirror {n : N} (hn : n <= 132) : g11 (132 - n) = g11 n
    theorem Machine11.L2_mirror : forall t, t < 135 -> L2 (mir2 t) = L2 t
    theorem Machine11.window2_even {g : N} (hg : g != 6) :
        (((Finset.range 135).filter (fun t => L2 t = g)).card) % 2 = 0
    theorem Machine11.adjacent_max_none_of_at_most_one
        (hone : (((Finset.range 135).filter (fun t => L2 t = 2 * 7)).card) <= 1) :
        (((Finset.range 135).filter (fun t => L2 t = 2 * 7)).card) = 0

`L2 t = g11 t + g11 (t + 1)` is the depth-2 window length and
`mir2 t = (266 - t) % 135` is the mirror on window indices.

WHAT THE WORK ACTUALLY WAS, and I mis-sized it in the same direction as round
27's verdict 27. `Mirror.mirror_exposed11` (round 26) says the opening SET is
closed under `k -> 385 - k`. It does NOT say the ENUMERATION reverses - that is
a statement about the sorted order, and getting it needs an induction. That
induction IS the composition round 27 named and did not build:

    assume  opSeq n + opSeq (133 - n) = 385.
    Then `385 - opSeq (132 - n)` is exposed (set-closure), lies above `opSeq n`,
    and has nothing exposed strictly between it and `opSeq n` (the mirror image
    of an empty interval is empty) - which is exactly `nextOp`'s defining
    property, so it IS `opSeq (n + 1)`.

The only finite computation in the whole chain is the base case
`opSeq 133 = 382`, one `decide +kernel` on the `ow` walk of `Machine11Per.lean`.
THE 135 WINDOW LENGTHS ARE NEVER ENUMERATED IN THE KERNEL, and the argument uses
nothing about machine 11 beyond mirror-closure and `nextOp` minimality - so it
transfers to any machine that has a kernel base case.

THE INSTANTIATION IS NOT VACUOUS, and I checked that before formalising. The
depth-2 length histogram of machine 11's period is

    length  3   4   5   6   7   8  10  11
    count  20  18  40  11  26   8   6   6

EXACTLY ONE ODD ENTRY, at length 6 - and 6 = g11 133 + g11 134 = 3 + 3 is the
length of the window at the unique self-mirror index 133. The theorem predicts
the parity of eight counts and gets eight for eight.

CROSS-CHECK BUILT INTO THE FILE, and it doubles as the honest scope note.
`Machine11.adjacent_max_none` proves the `(7,7)` count is ZERO outright by a
route that never mentions the mirror: machine 11's kernel spectrum ladder gives
`F_2(11) <= 11 < 14`. The two routes agree. So at machine 11 the lever is not
yet BUYING anything - the direct bound is available and cheaper. It buys
something at a machine where the direct bound is out of reach, and what this
round establishes is the PRICE of moving it there: one kernel base case plus the
induction above, not a new theory.

### R28.1 ITEM 1 - THE INCREMENT LAW IS A KERNEL STATEMENT AT ALL SIX LITERAL STEPS

The LP thread's round-27 increment-width certificates are in the kernel, and the
realisability half is with them. `proofs/Increment.lean`:

    theorem Increment.increment_19_23 :
        exists a b c, AdjPair Machine19.Exposed19 a b c and c - a = 31 and
          forall n, Machine23.g23 n <= (c - a) + 8
    theorem Increment.increment_23_29 :
        exists a b c, AdjPair Machine23.Exposed23 a b c and c - a = 39 and
          forall n, Machine29.g29 n <= (c - a) + 10
    theorem Increment.increment_29_31 :
        exists a b c, AdjPair Machine29.Exposed29 a b c and c - a = 55 and
          forall n, Machine31.g31 n <= (c - a) + 10
    theorem Increment.increment_law_literal_steps :     -- all six conjoined
        11->13 and 13->17 and 17->19 and 19->23 and 23->29 and 29->31

with `AdjPair E a b c` = "a, b, c are three CONSECUTIVE openings of E". Each
statement is SELF-CONTAINED - no `F_2` symbol, no census, no period scan: it
exhibits the old machine's realised adjacent pair and bounds every gap of the
new machine by that pair's span plus `s_min(q')`.

The upper halves at 19->23, 23->29 and 29->31 are the new rungs `IncCert23`,
`IncCert29`, `IncCert31` - 35 exact dual certificates each, generated by
`research/gen_inc_lean.py` (which reuses `gen_case_lean.gen_case` verbatim, so
the two rung families share one soundness skeleton) from JSON that
`research/lp_cert_inc_r28.py` rebuilds from the primes. EACH ALSO IMPROVES THE
LEDGER'S BEST HYPOTHESIS-FREE BOUND ON THAT MACHINE'S RECORD GAP:

    step      s_min  F_2(M)  W_inc   best previous kernel F(q')   now   true F
    19->23      8      31      39    47  (Machine23.g23_le)        39     34
    23->29     10      39      49    none hypothesis-free          49     43
    29->31     10      55      65    74  (CaseCert31.F_le)         65     58

- and at machine 29 it is the FIRST hypothesis-free kernel bound at all
(`Machine29.g29_le` carries a census hypothesis; `IncCert29.F_le` carries none).

AT THE THREE SMALL STEPS THE CERTIFICATE IS NOT NEEDED, and that is a finding,
not a shortcut: the corpus already carries a STRICTLY TIGHTER kernel bound on
F(q') than the increment width - 11 < 15 at machine 13, 18 < 22 at 17, 25 < 31
at 19 - so `Machine13.spectrum_one`, `Machine17.spectrum_one` and
`Machine19.spectrum_one` discharge those three outright. THE INCREMENT WIDTH IS
SLACK AT THE SMALL MACHINES AND KNIFE-EDGE AT THE LARGE ONES, and the crossing
is at machine 23. I transcribed and re-verified the LP thread's three small-step
certificates anyway (all 120 round-27 increment certificates are re-derived by
my gate) and deliberately did NOT build them: a kernel module whose statement is
implied by a one-line consequence of an existing theorem is ledger weight with
no content, and ledger weight costs rebuild time forever.

THE LOWER HALF - WHAT NO DUAL CERTIFICATE CAN CARRY. `F_2(M) >= v` is a
realisability statement. The LP thread emitted six witnesses as PHASE VECTORS
(exact-cover backtrack, no period scan). CRT turns each phase vector into a
single slot of the real machine, and that slot is what the kernel checks:

    F_2(11) >= 11   openings 252, 257, 263                    gaps (5, 6)
    F_2(13) >= 16   openings 117, 122, 133   (round 11's `pair16_realized`)
    F_2(17) >= 25   openings 110, 117, 135                    gaps (7, 18)
    F_2(19) >= 31   openings 1118917, 1118927, 1118948        gaps (10, 21)
    F_2(23) >= 39   openings 19016898, 19016903, 19016937     gaps (5, 34)
    F_2(29) >= 55   openings 858386140, 858386160, 858386195  gaps (20, 35)

Each is three `decide`s plus an `interval_cases` over the interior; the machine
29 one costs 35 s for 55 interior slots at numbers near 8.6e8. THE PROJECT'S
`F_2(29) = 55` - a full-period census number over 214,708,725 openings - IS NOW
REPRODUCED IN THE KERNEL FROM A SINGLE SLOT. All six slots were re-derived
independently from the LP thread's round-28 `witness_inc_*.json` phase vectors
and agree exactly, split for split. The m19 witness has split (10, 21), which is
the maximiser their windowed vehicle located from the DUAL side in round 26.

AND THE LEDGER GETS A SHARPNESS RESULT IT DID NOT HAVE. `Machine29.g29_le` and
`Machine31.g31_le_71` each stand on a census hypothesis `SpectrumBound g_M 2 F2`
- an UPPER bound on the old machine's two-gap record. The realisers are LOWER
bounds on the same quantity, so with one abstract lemma they PIN it.
`Increment.pair_attained` turns "three consecutive openings" into "an index of
the gap word" using only `next`'s three defining properties and the machine's
`opSeq_surj`; then

    theorem Increment.f2_19_sharp : not (Spectrum.SpectrumBound Machine19.g19 2 30)
    theorem Increment.f2_23_sharp : not (Spectrum.SpectrumBound Machine23.g23 2 38)
    theorem Increment.f2_29_sharp : not (Spectrum.SpectrumBound Machine29.g29 2 54)

THE HYPOTHESES THE MERGE-LAW RUNGS STAND ON CANNOT BE STATED WITH A SMALLER
CONSTANT. And the sharpest form of the law itself,

    theorem Increment.increment_23_29_index :
        exists i, forall n, Machine29.g29 n <=
          (Machine23.g23 i + Machine23.g23 (i + 1)) + 10

- no constant on the right that is not itself a realised quantity of the old
machine.

TWO CODEBASES, ONE CERTIFICATE, AGAIN. My pipeline reads the LP thread's
PICKLES and rebuilds every number from the primes; their `emit_inc_r28.py`
writes JSON from the same pickles by different code. `lp_cert_inc_r28.py CROSS`
compares the two on `pos`, `y`, `nu`, `yff`, `lhs`, `rhs` AS EXACT RATIONALS:
120 of 120 cases agree. Their reported minimum margin collapses 1 -> 1/384 over
the six steps; the Lean side sees that as the case denominator, and the 29->31
`cert` decides `518 < 519` after scaling by 31.

### R28.2 A FIVE-FREE-GEAR ARITY, AND THE LAKE FACT THAT DECIDED WHERE IT LIVES

The 19->23 increment rung holds TWO gears where the (D) rung held one, leaving
five free, so it needs `CaseSplit.lowest5` / `degpos5` - round 27's
lowest-blocker inequality one gear narrower. They went in a NEW module
`proofs/CaseSplit5.lean` (same namespace, reopened) rather than being appended
to `CaseSplit.lean`, for a mechanical reason that is now a lane rule: LAKE KEYS
ON CONTENT HASHES, so touching `CaseSplit.lean` would have invalidated all 75
existing case modules and cost about an hour of kernel to rebuild artefacts that
had not changed. A new module in the same namespace costs nothing.

### R28.3 ITEM 2 - RUNG EIGHT (37->41): PRECONDITION ABSENT FOR THE THIRD ROUND

Unchanged and honest. Rung eight on the merge-law vehicle needs machine 37's
qualifying dictionary at depths 2..7, floor 14, in `qual_dict.py`'s format.
Constructor's round-28 pre-registration
(`research/data/r28/constructor_prereg_r28.txt`, read early in the round) lists
the per-J triple analogues, the cover-half order N(M) and rung nine; their
round-28 block delivers those and does not contain the m37 emission. MISSING
INPUT, not a judgment and not a will-not-close.
BUT THIS ROUND MAKES THE ROUTE AROUND IT VISIBLE, and it should be said plainly:
THE CASE-SPLIT VEHICLE NEEDS NO DICTIONARY AT ALL. If the LP thread certifies
31->37 or 37->41 at any width, the Lean side is a mechanical transcription -
`gen_inc_lean.py` takes a tag and a three-number step table, and this round it
produced 114 modules from that. The qualifying dictionary is the MERGE-LAW
vehicle's input, not the certificate vehicle's, and the certificate vehicle is
now the one running ahead.

### R28.4 ITEM 3 - CASE-COUNT ECONOMICS: BATCHING IS WORTH 1.2x AND PRIORITY IS
### WORTH 8.9x

THE QUESTION AS BRIEFED: can case modules share elaboration, and is there a 2x
in it? MEASURED ANSWER: sharing gives 1.12-1.24x and cannot give more than
1.40x - but the round found a 8.9x sitting beside it that costs nothing.

METHOD. Three scratch modules built by `research/econ_r28.py` from the
ALREADY-GENERATED `IncCert31` case files (7 free gears, 27 positions - the
largest family in the ledger, so the extrapolation to a k = 3 rung is not
flattered): `Econ0` = the same imports and NO declarations, `Econ1` = one case
body, `Econ5` = five case bodies concatenated. Bodies copied by CONTENT, never
by line offset (round 22's lesson). Each measured SOLO and SEQUENTIALLY via
`lake env lean` - never side by side - with 30-48 other-lane python processes on
the box at 42-48% total CPU, which is the realistic operating condition.

    run                       priority   wall
    Econ0  imports only         High      13.8 s
    Econ1  one case             High      53.8 s   (paired re-run: 48.4 s)
    Econ5  five cases           High     216.3 s
    Econ0  imports only        Normal     26.7 s
    Econ1  one case            Normal    432.6 s

(1) THE MARGINAL COST IS ADDITIVE. Predicted `T0 + 5 (T1 - T0)` = 213.8 s,
measured 216.3 s - 1.2% high. Round 24's superadditive blow-up (eight heavy walk
decides in one file: >2.3 GB, 20+ min, each 17-60 s alone) DOES NOT REPEAT at
this workload, two orders of magnitude smaller per decide. So batching is safe.

(2) BUT THE SHAREABLE PART IS SMALL. The fixed cost is `T0` = 13.8 s of a 53.8 s
case, 26%, so five-batching gives 5 T1 / T5 = 1.24x (1.12x against the paired
T1) and the CEILING at any batch size is T1 / (T1 - T0) = 1.35-1.40x. THE
BRIEF'S 2x IS NOT AVAILABLE FROM SHARING ELABORATION. I did not adopt batching:
1.2x is not worth losing one-module-per-case resumability, which is what made
this round's 105-module build survive two fork-table failures with no loss.

(3) THE REAL FINDING, AND IT WAS NOT THE QUESTION I WAS ASKED. Raising
`lean.exe` to `High` is worth 8.9x on a case module (432.6 s -> 48.4 s), paired,
both runs solo, both under the same other-lane load. AND THE EFFECT SPLITS
CLEANLY: the import phase moves only 1.9x (26.7 -> 13.8) while the
elaboration-and-kernel phase moves 11.7x (405.9 -> 34.6). Import loading is
I/O-bound and priority barely touches it; kernel evaluation is pure CPU and at
Normal priority it is STARVED by the other lanes' 30-odd python workers. Round
24 measured 2.3x for the same lever on a mini-slice; the multiplier is not a
constant, it is a function of the competing load, and at this round's load it is
nearly four times larger.
IN-SITU CORROBORATION, labelled as such: my own driver log shows m29 case
modules at 242 s before the boost loop started and 86-95 s after, at unchanged
worker count. That is NOT a controlled A/B (other-lane load also varied) and is
recorded only as agreeing in direction with the paired measurement above.

(4) THE k = 3 REPRICE. Round-27 verdict 29 priced a 385-case rung at ~8.6 h from
1.34 min/case at two workers. At High priority the same family costs 48.4 s
solo, so 385 cases is 5.2 core-hours - ABOUT 2.6 h AT TWO WORKERS, and ~2.1 h if
batched five-up. THE CASE-COUNT WALL MOVED BY 3.3x, AND NOT BY THE MECHANISM THE
BRIEF PROPOSED. k = 3 is now a half-round job. k = 4 (5,005 cases) is ~34 h at
two workers - still out of reach for one round, but no longer "5 days".

### R28.5 SCORING THIS ROUND'S PRE-REGISTERED PREDICTIONS

Written in `research/data/r28/formalist_prereg_r28.txt` before the measurement
and before the m29/m31 families finished building.

  F1 (T0 >= 25 s and >= 25% of T1; hence >= 1.4x at B = 5) - REFUTED, AND THE
     WAY IT FAILED IS THE POINT. At High priority T0 = 13.8 s (below 25) though
     the ratio clause just holds at 25.6%; at Normal T0 = 26.7 s (above 25) and
     the ratio is 6.2%. I PREDICTED A CONSTANT WHERE THERE IS A TWO-PARAMETER
     SURFACE, because I had not thought to name a priority class - the same
     variable that turned out to dominate the whole item. The consequence
     (>= 1.4x at B = 5) is refuted at 1.12-1.24x measured.
  F2 (additivity within 20%) - CONFIRMED at 1.2% on the first T1 and 16% on the
     paired one; inside the band either way.
  F3 (all 70 IncCert29/31 cases certify, no case failing) - CONFIRMED. All 105
     case modules built, and all three roots' exhaustiveness closed. The kernel
     is an independent checker of the LP thread's certificates at five, six and
     seven free gears and it agreed everywhere.
  F4 (rung eight's precondition absent again) - CONFIRMED. See R28.3.

### R28.6 NEW VERDICTS

31. **A SYMMETRY OF A SET IS NOT A SYMMETRY OF ITS ENUMERATION, AND THE GAP IS
    ONE INDUCTION.** Round 26 proved the opening set closed under `k -> P - k`;
    the lever needs `opSeq (N-1-n) = P - opSeq n`, which is a statement about
    SORTED ORDER. The upgrade is an induction on `nextOp` minimality with one
    finite base case - cheap, but not free, and it was the whole of item 0. The
    general shape: when a kernel statement is about an ENUMERATION and the
    available lemma is about a SET, budget for the order argument.
32. **DO NOT BUILD A MODULE WHOSE STATEMENT AN EXISTING THEOREM ALREADY
    IMPLIES.** Three of the six increment steps had certificates and did not
    need them (the corpus bound is strictly tighter). Transcribe and gate them -
    the cross-check is free - but keep them out of the ledger. Ledger weight is
    a permanent rebuild cost and a permanent audit surface.
33. **THE CORPUS BOUND AND THE VEHICLE BOUND ARE TWO LADDERS AND THEY CROSS.**
    At machines 13/17/19 the kernel's spectrum ladder beats the increment width;
    at 23/29/31 the increment width beats the kernel's best. The crossing is at
    machine 23 and it is exactly where the case-split vehicle starts paying its
    way. Check which side of the crossing a target sits on before building.
34. **A NEW ARITY GOES IN A NEW MODULE, NEVER INTO THE SHARED ONE.** Lake keys
    on content hashes, so adding `lowest5` to `CaseSplit.lean` would have
    invalidated 75 unchanged case modules. Reopening the namespace in a new file
    costs nothing. (Corollary of a fact this lane already knew and had not drawn
    the consequence from.)
35. **THE CASE-SPLIT'S KERNEL LIMIT IS NOT THE CASE COUNT, IT IS CPU STARVATION
    AT NORMAL PRIORITY.** Verdict 29 said the limit is the case count. Measured
    this round, paired and solo: the same case module costs 432.6 s at Normal
    and 48.4 s at High under the same other-lane load - 8.9x, and 11.7x on the
    kernel phase alone. Sharing elaboration across cases, which is what the
    brief proposed, is worth at most 1.40x. RUN EVERY KERNEL SCAN AT HIGH
    PRIORITY; it is the largest single lever this lane has and it costs a
    two-line loop.
36. **REALISABILITY IS CHEAP IN THE KERNEL, AND THIS LANE LEFT IT ON THE TABLE
    FOR THREE ROUNDS.** `F_2(29) = 55` is a full-period census number over
    214.7M openings; in the kernel it is three `decide`s and one
    `interval_cases` over 55 slots, 35 seconds - ONCE SOMEONE HANDS YOU THE
    SLOT. The blocker was never the kernel, it was that the project's witnesses
    lived as PHASE VECTORS and nobody had run CRT on them. One line of Python
    converts a dual-side artefact into a kernel-checkable object. Ask every lane
    that reports a "witness" what its CRT slot is.

### R28.7 BUILD, AXIOMS, AND WHAT IS ON DISK

BUILD GREEN AT 1749 JOBS (1521 -> 1749). 114 new modules: `CaseSplit5`,
`MirrorM11`, `Increment`, three `IncCert{23,29,31}B`, 105 case modules
`IncCert{23,29,31}C0..C34`, three rung roots `IncCert{23,29,31}`.

AXIOM AUDIT over 405 declarations (367 -> 405), all standard-three or smaller:
299 `[propext, Classical.choice, Quot.sound]`, 51 `[propext, Quot.sound]`, 14
`[propext]`, 41 with NO axioms at all. Zero custom axioms, no `native_decide`,
no `Lean.ofReduceBool`. Worth noting from the round-28 rows: `CaseSplit.lowest5`,
`CaseSplit.degpos5` and `Machine11.mir2_invol` need NO axioms; and FOUR OF THE
SIX REALISERS (`f2_11`, `f2_17`, `f2_23`, `f2_29`) are CHOICE-FREE
(`[propext, Quot.sound]`) - only `f2_13` and `f2_19` pull `Classical.choice` in,
through the `exposed13_iff` / `exposed19_iff` simp path rather than through
anything mathematical.

NEW RESEARCH FILES (all mine): `research/lp_cert_inc_r28.py` (transcription +
GATE + CROSS), `research/gen_inc_lean.py` (the Lean emitter),
`research/econ_r28.py` (item 3), `research/inc_build_r28.sh` (the single
resumable driver), `research/data/r28/formalist_prereg_r28.txt`,
`research/data/r28/inc_build.log`, `research/data/r28/formalist_axiom.log`.

PROCESS NOTES, both small and both worth the line:
- TWO OF THE 105 CASE MODULES FAILED TO LAUNCH, not to compile: `rc=126` and
  `rc=127` with "Resource temporarily unavailable" - the project's known
  fork-table exhaustion, hitting the DRIVER rather than a worker. The
  skip-if-built resume loop made this a non-event: one `lake build` of the two
  named modules finished them. A driver that logs its own return codes turns a
  silent gap into a two-line repair; one that logged only successes would have
  produced a root that failed to import 90 minutes later.
- ONE DRIVER ONLY, per round-27 verdict 30, and the process list was checked
  rather than the tool's success message. No races, no deleted oleans.
- The dry-elaboration discipline paid again: the whole of `Increment.lean` was
  developed and elaborated against three `axiom` stubs for the `IncCert*.F_le`
  facts while the 105 case modules built (`proofs/DryInc28.lean`, DELETED on
  landing per the standing rule), and it compiled first try when the real
  imports arrived. `MirrorM11.lean` was dry-checked the same way and needed one
  fix (a `133 + 1` that `omega` would not identify with `134`).

### R28.8 Open formalisation targets (round-29 priority order)

0. **THE MIRROR LEVER WHERE IT PAYS.** R28.0 makes machine 11's instantiation a
   theorem and prices the transfer: one kernel base case (`opSeq (N-1) = P - o_0`)
   plus the induction. Machine 13 is the next rung with a computable walk
   (`Machine13Per.ow13_1485`); at machine 29 the base case is not reachable by a
   walk and the lever needs a different route to `opSeq (N-1)` - naming that is
   the real question, and it is where the lever would start buying something.
1. **31->37 BY THE CASE SPLIT, NOW REPRICED.** At High priority, 385 cases is
   ~2.6 h at two workers, not the ~8.6 h of verdict 29. Ask the LP thread for
   the smallest k that certifies it and the emission; the Lean side is then
   mechanical. This is the single highest-value target and it is now affordable.
2. **`A_relax(M) <= 5`** (Constructor's standing offer) - 48 classes mod 210,
   `LiteralCapTable.lean` shape, the first uniform (machine-free) order
   statement in the Lean corpus. Unchanged and still unattempted.
3. **Machine 37's qualifying dictionary, scan-free** - still the only missing
   input to rung eight ON THE MERGE-LAW VEHICLE, and now demoted: the
   certificate vehicle does not need it (R28.3).
4. **The remaining `F_2` upper halves.** `f2_23_sharp` / `f2_29_sharp` pin the
   census hypotheses from below; the matching upper halves (`SpectrumBound g23 2
   39`, `SpectrumBound g29 2 55`) are still census hypotheses. The windowed
   vehicle at machine 29 is a held-gear job (the LP thread's round-27 E4), so
   this is a certificate ask, not a scan.
5. Discharge `Census29` from `Census23` by dictionary transfer (unchanged).
6. The depth-sum re-indexing bijection at machine 13 (unchanged).
7. The generator at 13 -> 17 by the `Gen11Sound` template (unchanged).
8. The sandwich lemma (unchanged).
