# The certificate arity ladder for the maximal gap, and its Mertens threshold

Status: PROVED (the potential bound and its tightness; the bounded-state
no-go T1; the arity-1 MERTENS no-go T2, exact rational arithmetic) +
SCRIPT-VERIFIED LP ladder with every feasible certificate re-checked directly
against the machine (`research/potential_arity.py`, log
`research/data/potential_arity.log`). The arity-r threshold for r ≥ 2 is
CONJECTURED with its derivation and the gap in it stated. Established round
23 (Lateral). Prior-art check: NOT YET CHECKED (section 6).

## 1. WHAT IT IS

Plain language. `nilpotent-invariants.md` proves that every unitary invariant
of the machine's blocked-walk operator is a function of the gap histogram, so
no operator invariant can bound the maximal gap `F(M)` non-circularly. Exactly
one frame escapes, because it is a CERTIFICATE rather than an invariant: a
POTENTIAL (a Lyapunov function) on the slot line. This document builds that
frame, shows it is exactly tight, and then measures the only thing about it
that can fail - HOW MANY GEARS THE POTENTIAL MUST SEE AT ONCE. That number is
the round's spine question ("does the arity stabilise?") asked as a proof
obligation rather than a census: an infeasibility verdict rules out EVERY
certificate of that arity, not one attempt.

**THE CERTIFICATE (proved, tight).** For `h : Z_P → R` write

        (*)   h(k) - h(k-1) ≥ 1   for every BLOCKED slot k.

Along a run of `L` consecutive blocked slots `h` increases by at least `L`,
and a gap of `g` slots is a run of `g-1` blocked slots, so

        F(M) ≤ 1 + osc(h),   osc(h) := max h - min h,

for EVERY `h` satisfying (*). It is TIGHT: `h(k) =` (distance back to the
previous opening) satisfies (*) with `osc = F - 1` exactly (asserted at
machines 11/13/17/19). So `F` is exactly the optimum of a linear program over
potentials. The multiplicative form `w = exp(h/t)` is a Schur test on
`A = BS + (BS)^T`; the tropical limit `t → 0` is Constructor's max-plus
potential inequality. The frame loses nothing.

**THE ARITY CLASSES.** The optimal `h` above is "distance to the previous
opening" - a function of ALL gears jointly, and precisely the window
indicator whose Schmidt rank round 22 measured growing. Restrict:

        LEVEL-r class:  h(k) = Σ_{|U| = r} x_U( k mod ∏U ),  U ⊆ gears.

Level `r` contains level `r-1` (a function of `k mod q` is a function of
`k mod qq'`). Level 1 is a per-gear potential; level `m` (all gears) is the
full class and returns `F` exactly.

**T1 (BOUNDED-STATE NO-GO, one line).** If `h` depends only on `k mod m` for
a PROPER divisor `m` of `P`, (*) is infeasible outright: every residue class
mod `m` contains a blocked slot (the large gears block inside every class -
asserted for `m = 35, 385` at machines 11-19), so (*) forces
`h(r) - h(r-1) ≥ 1` for all `r mod m`, and summing round the `m`-cycle gives
`0 ≥ m`. A state that has forgotten any gear cannot see that a slot is
blocked. This is the structural reason bounded-state certificates mod
35/385/5005 cannot bound `F` - the same wall Constructor's bounded-state
certificates hit at 23→29.

**T2 (MERTENS NO-GO AT ARITY 1, proved).** A level-1 potential exists only if

        σ(y) := Σ_{5 ≤ q ≤ y} 1/q  <  1/2.

Proof. `D(k) = h(k)-h(k-1) = Σ_q f_q(k mod q)` with each `f_q` of zero mean
(a difference over a full cycle). Put `S_q = Σ_{t ∈ teeth(q)} f_q(t)` and
`Σ = Σ_q S_q/(q-2)`; by CRT the residues are independent and uniform, and
openings are the all-exposed slots, so the mean of `D` over OPEN slots is
exactly `-Σ`.
(i) `mean_{Z_P} D = 0` and `D ≥ 1` on blocked give
`0 ≥ -π_o Σ + (1-π_o)`, i.e. `Σ ≥ (1-π_o)/π_o > 0`, `π_o = ∏(1-2/q)`.
(ii) For each gear `q` and each tooth `t`, the slots with `k ≡ t (mod q)` and
every other gear exposed are blocked and nonempty (CRT); averaging (*) over
them gives `f_q(t) - Σ + S_q/(q-2) ≥ 1`. Summing the two teeth,
`S_q · q/(q-2) ≥ 2(1+Σ)`, i.e. `S_q/(q-2) ≥ 2(1+Σ)/q`. Summing over gears,
`Σ ≥ 2σ(1+Σ)`, i.e. `Σ(1 - 2σ) ≥ 2σ`. If `σ ≥ 1/2` the left side is `≤ 0 <
2σ`. Contradiction. ∎

`σ(11) = 167/385 = 0.43377` but `σ(13) = 2556/5005 = 0.51069`, and `σ`
DIVERGES, so **ARITY-1 CERTIFICATES DIE AT MACHINE 13 AND NEVER RETURN**.

**THE MEASURED LADDER** (LP over the level-r class, HiGHS; every FEASIBLE
verdict re-checked by rebuilding `h` and testing (*) at every blocked slot,
so no bound depends on trusting the solver; INFEASIBLE verdicts are LP
infeasibility on the full row set):

    machine  F     arity 1        arity 2          arity 3        full
    y = 11   7   23.902 (3.41x)  7.753 (1.11x)   7.000 (1.00x)   = arity 3
    y = 13  11   INFEASIBLE     17.980 (1.63x)   11.000 (1.00x)  11 exactly
    y = 17  18   INFEASIBLE     37.102 (2.06x)      -            18 exactly
    y = 19  25   INFEASIBLE     FEASIBLE, <=195.5   -            25 exactly

(entries are the certified bound `1 + osc*` with `bound/F` in brackets. The
m19 arity-2 cell is a PROVED feasibility - a certificate found on a 4,836-row
subsample and then verified against all 1,237,940 blocked slots of the full
period, `min step = 1.0000` - but the OPTIMAL bound there was not computed:
the osc-minimising LP exceeds memory at full row count, and the row-generation
version did not converge within the round's budget on a box running ~20 other
jobs from other lanes. So 195.5 is a valid certificate, not the optimum.)

Two things are visible and both matter. First, arity 1 dies exactly where T2
says it must. Second, WHERE A FIXED ARITY SURVIVES, ITS QUALITY DECAYS: the
arity-2 bound is 1.11x, 1.63x, 2.06x the truth at machines 11, 13, 17. A
fixed-arity certificate does not merely become harder, it becomes
asymptotically vacuous while remaining feasible.

**THE CONJECTURED THRESHOLD, AND WHY IT IS INTERESTING.** The same averaging
carried out for a level-r class gives, with `a_U` the mean of the ANOVA
component `D_U` over the all-exposed set and `A = Σ_U a_U = E[D | open] ≤
-(1-π_o)/π_o < 0`:

        Σ_{U ∋ q} a_U ≤ (2A - 2)/q   for every gear q,
        hence  Σ_U |U| a_U ≤ (2A - 2) σ.

For `r = 1` the left side is exactly `A` and the contradiction closes (that
is T2). For `r ≥ 2` closing it needs a lower bound on `Σ_U |U| a_U` in terms
of `A`; `Σ_U |U| a_U ≥ r A` holds whenever the `a_U` are all `≤ 0`, and then
`σ ≥ r/2` is contradictory. THE GAP IS EXACTLY THAT SIGN CONDITION - stated
here rather than papered over. Taking the threshold at face value,

        LEVEL r IS INFEASIBLE ONCE σ(y) ≥ r/2,

which fits every measured cell above. The thresholds are doubly exponential:

        level 1 dies at y = 13       (σ = 0.5107)
        level 2 dies at y = 109      (σ = 1.0076)
        level 3 dies at y = 2741     (σ = 1.5002)
        level 4 dies at y = 483281   (σ = 2.0000)

so the REQUIRED ARITY is `r*(y) ≈ 2σ(y) ≈ 2 log log y` - unbounded but doubly
logarithmically slow.

**A PRE-REGISTERED PREDICTION OF MINE, REFUTED BY MY OWN MEASUREMENT.**
Before running the ladder I wrote into `research/potential_arity.py`'s
docstring: "P2 - the minimal feasible arity r*(y) GROWS: r*(11) = 1,
r*(13) = 2, and r*(19) >= 3." The machine-19 arity-2 cell is FEASIBLE (a
certificate found on a 4,836-row subsample and then verified against all
1,237,940 blocked slots of the full period, `min step = 1.0000`), so
r*(19) = 2 and P2 IS FALSE. The correction is the threshold law itself: r*
does grow, but only when `σ` crosses the next half-integer, i.e. DOUBLY
LOGARITHMICALLY - level 2 survives to `y ≈ 109`, not to 19. My guess was
right in direction and badly wrong in rate, and the rate is the whole point:
a fixed arity stays FEASIBLE for a very long time while its BOUND becomes
worthless, which is why feasibility alone is the wrong statistic to watch.

**THE "MERTENS ROOM" AND THE BOUND QUALITY** (measured, 4 points, suggestive
not established). Define the room `R = r - 2σ(y)`, which T2 says must be
positive at `r = 1` and which the threshold law says must be positive in
general. The measured bound quality tracks it:

    cell        r - 2σ(y)    bound/F
    m11, r=1      0.132       3.41
    m11, r=2      1.132       1.11
    m13, r=2      0.979       1.63
    m17, r=2      0.861       2.06

i.e. the certificate degrades as the room closes and (on this evidence) blows
up as `R → 0`, rather than failing abruptly. That is a stronger statement
than feasibility alone: a fixed arity is asymptotically vacuous well before
it becomes infeasible.

**THE CONVERGENCE THAT MAKES THIS WORTH RECORDING.** The project's LP-duality
thread, working on a completely different certificate family (covering/Farkas
duals for the (D) rungs, not potentials for `F`), independently found that
the required degree grows like `2·S1(y)` with `S1` the same reciprocal-prime
sum, and used it to identify Constructor's measured truncation arity 3 (m19,
m23) → 4 (m29) as literally that quantity. Two unrelated certificate frames
therefore produce the SAME arity law, `r* ∝ Σ_{q ≤ y} 1/q`. That is much
stronger evidence for "no fixed-arity rule exists" than either frame alone,
and it names the arithmetic source: THE DIVERGENCE OF THE RECIPROCAL PRIME
SUM - the same divergence that makes the sieve hard in the first place.

## 2. WHY IT MIGHT BE NOVEL

The potential bound itself is elementary (it is the standard "Lyapunov
function bounds the longest path" argument, and in the max-plus reading it is
a Kleene-star potential). What appears unrecorded:

- posing the maximal-gap problem of a wheel/Jacobsthal sieve as an LP over
  potentials, with the observation that the LP is EXACTLY TIGHT so the whole
  difficulty is the certificate's ARITY;
- the arity hierarchy itself as a measurable object, with a proved level-1
  no-go whose threshold is a MERTENS sum - a certificate class that dies
  exactly when `Σ 1/q` crosses `1/2`;
- the resulting law `required arity ≈ 2 Σ_{q ≤ y} 1/q ≈ 2 log log y`, matched
  independently by a different certificate family in the same project;
- the one-line bounded-state no-go T1, which explains a family of failed
  bounded-state certificates rather than reporting them.

## 3. PROOF / STATUS

PROVED: the potential bound and its tightness; T1; T2 (exact rational
arithmetic, `σ` computed as a `Fraction`).
SCRIPT-VERIFIED (`research/potential_arity.py`, assertion-gated):
* part 1 asserts T1's hypothesis (every class mod 35 and mod 385 contains a
  blocked slot) at machines 11-19 and prints `σ` exactly;
* part 5 asserts that `h = ` distance-to-previous-opening satisfies (*) and
  has `osc = F-1` EXACTLY at machines 11-19 (integers);
* the ladder solves each level-r LP with HiGHS and then REBUILDS `h` from the
  solution and checks `min_{k blocked} (h(k)-h(k-1)) ≥ 1` over the FULL
  period before reporting any bound; the reported `1 + osc` is therefore a
  valid certificate independently of the solver.
CONJECTURED: the `σ ≥ r/2` threshold for `r ≥ 2` (derivation above, sign
condition named). MEASURED: the quality decay 1.11x → 1.63x → 2.06x.

Cost note: the m17 arity-2 LP is 1078 variables against 62,810 constraints
and took 291 s; larger cells use row generation (solve on a subset, verify on
the full period, add violated rows) - a certificate that passes the full
verification is a proof no matter which rows the LP saw.

## 4. IMPLICATIONS

Inside the project:
- it answers the round's spine question from the CERTIFICATE side and agrees
  with the two counting-side answers (Constructor's truncation arity,
  Lateral's Schmidt-rank growth): no fixed-arity rule exists, and now with a
  proof for arity 1 and a threshold law for the rest;
- T1 explains, in one line, why bounded-state certificates mod 35/385/5005
  cannot bound `F` - relevant directly to Constructor's 23→29 failures;
- it supplies a new, checkable certificate FORMAT: any `h` satisfying (*) is
  a valid maximal-gap bound, verifiable in one pass over the period and
  kernel-checkable at a fixed machine (a finite integer inequality);
- it says where to spend effort: not on richer INVARIANTS (they are all the
  histogram) but on certificates whose arity grows with the machine, i.e. on
  generators, which is exactly Constructor's Kleene-star route.

Outside: a sieve-theoretic hierarchy in which the minimum "interaction order"
of a valid certificate is controlled by `Σ_{p ≤ y} 1/p`, giving a concrete
arithmetic mechanism for the failure of bounded-order/local certificates.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Jacobsthal / Ziller-Morack `h_2` (the LP optimum IS `F`, so any level-r
solution is a certified upper bound); requirement (D) and Wall V (the arity
law is Wall V with an arithmetic source); the parity/Mertens barrier
(`Σ 1/p` divergence as the mechanism). Open: prove or refute the `σ ≥ r/2`
threshold for `r ≥ 2`; find the smallest class (not necessarily an arity
class) that certifies `F` at a given machine; and transport the certificate
across a merge step - `h_new` from `h_old` plus a gear-`q'` part would be the
merge law in certificate form, which is not attempted here.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"Lyapunov function bound longest path in a periodic 0/1 sequence"; "potential
function certificate Jacobsthal function upper bound"; "linear programming
hierarchy Chinese remainder residue classes interaction order"; "Sherali-Adams
/ Sum-of-squares degree lower bound covering integer programs primes";
"junta / low-degree certificate for a CRT-structured Boolean function";
"Mertens sum threshold for local certificates in sieve theory". Expected
nearest art: LP/SOS degree lower bounds in proof complexity (the arity ladder
is a degree hierarchy for a CRT-structured problem); the delta to check is
the sieve statement, the `σ ≥ 1/2` proof, and the `2 Σ 1/q` arity law.
