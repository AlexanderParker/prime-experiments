# The leftover at depth u (branch R4.d.i.c)

Prover, round 6 of the stack line, 2026-09-10. Parent: R4.d.i.c, the node spawned by the reading of
research/proof/dead_branches_reopened_4.md candidate (c): "the P-against-P1P2 reading is one link's
slice of the Omega-census of core-free members at depth u; the record's depth grows along the chain
1.35, 1.69, 2.37, about 4 next; above depth 3 the Liouville sign no longer separates the types."
The brief is the last paragraph of that file. Scripts research/stack/r6/leftover_depth_u.py (the grid, the Omega-census, the run counts and the
fuels), leftover_runs.py (the second measurement: four calibrations of the model's p, and the
by-K table) and leftover_tail.py (the third: one fixed core, the gap-length bands and the
variance ratio); outputs
research/stack/r6/results/ (untracked; every number this document uses is in this document).
Nothing here is committed by the prover.

Vocabulary (unchanged, the raw line). A SLOT is (n, n + 2) with n = 5 mod 6; a STRETCH of L' slots
is L' consecutive slots; a section is [c, c') with c' the square of the first prime p_k at or above
c. For a stretch length L' the CORE is the primes in [5, t'] with t' = 6L' + 1, and a member is
CORE-FREE if no core prime divides it (equivalently the member is t'-rough); a slot is CORE-OPEN
if both its members are core-free; K is the number of core-open slots of a stretch. The DEPTH of a
number n for the core t' is u = ln n / ln t'. Omega(m) is the number of prime factors of m with
multiplicity. A stretch is TWIN-FREE if none of its slots is a twin. A RUN is a maximal block of
consecutive twin-free starts, i.e. one twin gap of at least L' slots; a START is one position of
the sliding stretch (N16: the start count is not the event count).

Laws. The register is one project-wide numbering. S8-S9 are in stacked_squares.md, S10-S14 in
core_leftover.md, S15-S16 in dead_branches_reopened_4.md, and S18-S19 were issued in
base_and_step.md by the parallel Part II lane, which reserved S17 for this branch. This branch
therefore issues **S17, S20, S21 and S22**.

Prior results checked before opening: docs/novel/README.md (read first, per the standing
direction), research/proof/theory_tree.md node R4.d.i and its children, core_leftover.md sections
1-6 (S10-S14), step_evidence.md sections 5-7, dead_branches_reopened_4.md in full (N12-N16,
S15-S16), proofs/CoreLeftover.lean round 40 (twin_of_rough, primeOrSemiprime_of_rough_lt_cube,
crossing_core / exists_run_ge_iff). What is on record: the census of core-free members by PRIMALITY
by depth at the section's own record length only (Table 3 of dead_branches_reopened_4.md, one L'
per section); the independence of the types at the section's record length (N15); the depth law
S15 and the tail-empty dichotomy S16. What is NOT on record and what this branch can find: the
census by Omega (not by primality) as a function of depth; whether the type share is a function of
u alone or also of L' (the section's own machine leaving a trace); the exact Omega thresholds in u;
the twin-free run count against the independent-slot model in the two-parameter region (L', u)
rather than at one point per section; and the first behaviour of the leftover at depth above 3,
where the two-prime lemma (and with it the whole "P against P1 P2" reading of the step) expires.

---

## 1. Pre-registered (written before any script of round 6 ran)

### Theory

T17 (the type share is a function of depth alone). The core is a sieve by all primes up to t' and
nothing else; a core-free member is a t'-rough number; the only parameter of a rough number's
factorisation profile is u = ln n / ln t'. Therefore the Omega-census of core-free members, and the
PP share among core-open slots, should be functions of u alone, identical (to sampling error) at
u = 3 whether that u is reached with a short stretch high in the section or a long stretch at the
same depth. If instead the share at fixed u depends on L', something other than the sieve's depth
is acting: that would be the first trace of the section's own construction inside the leftover.

T18 (the Omega thresholds are exact and arithmetic). A t'-rough number with Omega >= j is at least
q^j with q = nextprime(t'). So Omega >= 2 begins exactly at q^2 (u = 2 ln q / ln t') and Omega >= 3
exactly at q^3 (u = 3 ln q / ln t'), with no exceptions, and the profile of the census is pinned at
those two points. This is the extension of the two-prime lemma
(CoreLeftover.primeOrSemiprime_of_rough_lt_cube) and of S15.

T19 (the count side stays independent above depth 3). N15 and N16 found the leftover's types
binomial given K and the twin-free run count on the independent-slot prediction at every length
from L/4 to L, at one depth per section. If the step at the core is ROOT in the count, that must
persist in the whole (L', u) region, including the bins where Omega = 3 members exist, where the
"P against P1 P2" dichotomy is false and the sign-count cannot see the types at all. A bin off the
independent prediction is the finding.

T20 (the fuel identity is the Omega = 2 statement and nothing more). "The fuel of a composite
core-free member is a prime of the section's own machine (P2 > n / p_k)" holds for Omega = 2 by
arithmetic: the smaller factor P1 <= sqrt(n) < sqrt(c') = p_k. For Omega = 3 the complement of the
largest factor is a product of two primes above t', at least q^2 > p_k for every L' on the grid, so
the same statement is false for EVERY Omega = 3 member. The fuel identity is therefore a restatement
of the depth being below 3, not an independent structure.

### Predictions and scorecard (verdicts filled in section 7)

| # | prediction (brief's numbering) | refuted by | verdict |
|---|---|---|---|
| P1 | the PP share among core-open slots depends on u alone: at a common u-bin the share agrees across all L' on the grid to within sampling | a systematic spread across L' at fixed u larger than the sampling error |  **held to 2.7%, refuted as exact** (section 3.1, S21) |
| P2 | Omega = 3 core-free members appear exactly from n >= nextprime(t')^3 (u = 3 ln q / ln t' > 3), Omega = 2 exactly from nextprime(t')^2, with 0 exceptions; Omega = 4 from nextprime(t')^4 | one member below its threshold |  **held exactly**, with the sharper threshold nextprime(t')^j (section 4, S17) |
| P3 | the twin-free RUN count is on the independent-slot prediction in every (L', u) bin above depth 2, including bins holding Omega = 3 members | a bin whose run count is off the prediction beyond the model's own simulated spread |  **held, one deviation recorded** (section 5, S22) |
| P4 | the fuels of the composite core-free members on twin-free stretches satisfy P2 > n / p_k with 0 exceptions | one composite member with n / (largest prime factor) >= p_k |  **refuted**, exception count = the Omega >= 3 count (section 6) |
| P5 (prover's, against P4) | P4 holds for every Omega = 2 member and fails for every Omega = 3 member, so its exception count is exactly the Omega = 3 count on twin-free stretches | an Omega = 3 member satisfying P2 > n / p_k, or an Omega = 2 member failing it |  **held**, threshold verified in both directions (section 6, S20) |
| P6 (prover's) | the prime share among core-free members at depth u follows one curve in u across both sections and all L' (Buchstab-type profile; prior art, not derived here) | two sections disagreeing at a common u beyond sampling |  **held** (section 3.1) |
| P7 (prover's) | min K over twin-free stretches falls to 0 for the short L' (the core alone covers a stretch of L' slots somewhere in the section: S12, R(t') >= L') and is positive for the long ones | min K > 0 at every L' on the grid, or min K = 0 at L' = 579 on base 3 |  **held**, crossing between L' = 200 and 300, matching S12 (section 5.1) |
| P8 (owner's reading, from the unstick file) | confirmation of P1-P3 files the step at the core as ROOT in the (L, u) coordinate: the count of PP among independent leftovers | a refutation of P1 or P3 |  **held**; filed ROOT in (L, u) (section 7) |

### Setup as pre-registered

Sections: base 3, section 4 = [16129, 260,467,321), 43,408,531 slots, p_k = 16139; base 7,
section 3 = [2809, 7,946,761), 1,323,991 slots, p_k = 53 for the lower cut and 2819 as the section's
first gear. Stretch lengths L' in {50, 75, 100, 150, 200, 300, 400, 579}, core t' = 6L' + 1 each
time. Depth bins of width 0.1 in u. Every count exact over every start of the section (no
sampling); the only sampled object is the independent-slot model's own null distribution, which is
simulated (20 draws per L') to give the spread against which an observed run count is judged.

---

## 2. Setup, and the checks that the machinery reproduces the record

Two sections, every start of each, no sampling.

| section | slots | twins | record L | first gear p_k | c' = p_k^2 |
|---|---|---|---|---|---|
| base 3, section 4 = [16129, 260,467,321) | 43,408,531 | 1,027,948 | 579 slots | 16,139 | 260,467,321 |
| base 7, section 3 = [2809, 7,946,761) | 1,323,991 | 48,249 | 254 slots | 2,819 | 7,946,761 |

The slot holding the next cut is dropped (core_leftover.py's convention). For each stretch length
L' the core is the primes in [5, t'], t' = 6L' + 1, so a member is core-free exactly when it is
t'-rough; the depth of a slot is u = ln n / ln t' taken at the lower member; bins are 0.1 wide in u.
For each member the smallest prime factor is computed once, by running every prime up to
sqrt(c') over its two residue classes in descending order; a member with no such factor is prime.
Omega and the largest prime factor then follow by dividing out the smallest prime factor and
testing primality against the sieve, which terminates in at most three steps here. Core-free is
"smallest prime factor absent or above t'", with the extra clause that a member at most t' is never
core-free (it is divisible by itself); that clause bites only when t' exceeds sqrt(c'), which on
this grid happens only at base 7, L' = 579.

The independent-slot model, stated exactly. Each core-open slot is a twin independently with
probability p, every other slot never; p is estimated from the data as the PP share among the
core-open slots of the slot's own depth bin. Then the predicted number of twin-free stretches
starting at x is the product of (1 - p) over the core-open slots of the stretch, and the predicted
number of RUNS (N16: the run is the event, the start is not) is that product times the probability
that the slot just before the stretch is a twin, summed over x. The same model is also SIMULATED
(20 draws, each a fresh Bernoulli assignment on the section's own core-open pattern) to get the
spread the model itself has, so an observed count is judged against the model's own sd and not
against a Poisson guess.

Checks against numbers already on the record, before any new reading was taken:

| quantity | this round | on record | source |
|---|---|---|---|
| base 3, L' = 579: mean K over all starts | 20.3561 | 20.356 | core_leftover.md 3.1 |
| base 3, L' = 579: min K over twin-free stretches | 12 | 12 (the record stretch) | core_leftover.md 3.1 |
| base 3: composite core-free members of the section at t = 3475 | 3,130,924 (all Omega = 2) | 3,130,924 | dead_branches_reopened_4.md |
| base 3: prime members of the section | 14,218,065 | 14,218,065 gears of machine 4 | step_evidence.md 1 |
| base 7, L' = 254: mean K | 10.3927 | 10.393 | core_leftover.md 3.1 |
| base 7: composite core-free members at t = 1525 | 33,456 (all Omega = 2) | 33,456 | dead_branches_reopened_4.md |
| base 7, L' = 254: twin-free starts / runs | 1 / 1 | 1 | step_evidence.md 6 |
| base 3: min K over twin-free stretches is 0 up to L' = 200 and 1 at L' = 300 | 0, 0, 0, 0, 0, 1, 1, 12 | the crossing L0 = 278 (S12) | core_leftover.md 3.2 |
| base 7, L' = 254: the record's one composite has its fuel in the tail | largest factor below the cut for 1 of 1 | 4,871,171 = 2039 x 2389 | dead_branches_reopened_4.md Table 2 |

The depth actually reachable. On the base-3 section the deepest slot for a given L' sits at
u = ln(2.6 x 10^8) / ln(6L' + 1): 3.396 at L' = 50, 3.171 at 75, 3.028 at 100, and below 3 from
L' = 150 on. Depth 4 is NOT reachable with L' >= 50; it needs t' <= 127, so a separate diagnostic
grid L' = 18, 20, 25 was run for the Omega = 4 clause of P2 alone (t' = 109, 121, 151; deepest slot
at u = 4.131, 4.108, 3.855). On base 7 the deepest slot is at u = 2.784 (L' = 50), so that section
reaches no Omega = 3 member at any L' on the grid; it is the control in which the whole leftover is
P or P1 P2.


## 3. Tables per (L', u)

### 3.1 Does the type census depend on depth alone? (P1, P6)

The PP share among core-open slots, at each depth bin, for each stretch length. t' runs from 301 to
3475, a factor 11.5, and at a fixed u the different L' are looking at different numbers: u = 2.5 is
n = 1.3 x 10^6 at t' = 301 and n = 7.0 x 10^8 at t' = 3475. Base 3.

| u | L' = 50 | 75 | 100 | 150 | 200 | 300 | 400 | 579 |
|---|---|---|---|---|---|---|---|---|
| < 2.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2.0 | 0.9080 | 0.8958 | 0.9038 | 0.9031 | 0.9029 | 0.9074 | 0.9059 | 0.9001 |
| 2.1 | 0.7716 | 0.7763 | 0.7766 | 0.7701 | 0.7722 | 0.7702 | 0.7691 | 0.7647 |
| 2.2 | 0.6711 | 0.6798 | 0.6794 | 0.6715 | 0.6727 | 0.6698 | 0.6685 | 0.6660 |
| 2.3 | 0.5991 | 0.6071 | 0.5968 | 0.5966 | 0.5946 | 0.5927 | 0.5935 | 0.5988* |
| 2.4 | 0.5480 | 0.5415 | 0.5392 | 0.5350 | 0.5330 | 0.5342 | 0.5356* | - |
| 2.5 | 0.4956 | 0.4905 | 0.4888 | 0.4851 | 0.4862 | 0.4886* | - | - |
| 2.6 | 0.4545 | 0.4516 | 0.4477 | 0.4468 | 0.4458 | - | - | - |
| 2.7 | 0.4193 | 0.4173 | 0.4147 | 0.4131 | 0.4247* | - | - | - |
| 2.8 | 0.3903 | 0.3880 | 0.3874 | 0.3927* | - | - | - | - |
| 2.9 | 0.3655 | 0.3641 | 0.3622 | - | - | - | - | - |
| 3.0 | 0.3432 | 0.3427 | 0.3491* | - | - | - | - | - |
| 3.1 | 0.3238 | 0.3256* | - | - | - | - | - | - |
| 3.2 | 0.3051 | - | - | - | - | - | - | - |
| 3.3 | 0.2873* | - | - | - | - | - | - | - |

(* marks each column's topmost bin, which the section's top cuts short; a truncated bin holds only
the lower part of its depth range and therefore reads high. Those cells are excluded from the
comparison below.)

The same table for the prime share among core-free members (P6):

| u | L' = 50 | 75 | 100 | 150 | 200 | 300 | 400 | 579 |
|---|---|---|---|---|---|---|---|---|
| < 2.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2.0 | 0.9525 | 0.9529 | 0.9507 | 0.9498 | 0.9509 | 0.9517 | 0.9511 | 0.9486 |
| 2.1 | 0.8802 | 0.8801 | 0.8783 | 0.8770 | 0.8772 | 0.8773 | 0.8767 | 0.8746 |
| 2.2 | 0.8223 | 0.8222 | 0.8199 | 0.8184 | 0.8186 | 0.8185 | 0.8179 | 0.8159 |
| 2.3 | 0.7751 | 0.7738 | 0.7723 | 0.7709 | 0.7707 | 0.7704 | 0.7699 | 0.7743* |
| 2.4 | 0.7350 | 0.7343 | 0.7327 | 0.7310 | 0.7309 | 0.7306 | 0.7323* | - |
| 2.5 | 0.7013 | 0.7004 | 0.6986 | 0.6973 | 0.6971 | 0.6996* | - | - |
| 2.6 | 0.6723 | 0.6712 | 0.6697 | 0.6683 | 0.6681 | - | - | - |
| 2.7 | 0.6468 | 0.6458 | 0.6443 | 0.6431 | 0.6522* | - | - | - |
| 2.8 | 0.6243 | 0.6235 | 0.6221 | 0.6272* | - | - | - | - |
| 2.9 | 0.6047 | 0.6037 | 0.6024 | - | - | - | - | - |
| 3.0 | 0.5866 | 0.5857 | 0.5917* | - | - | - | - | - |
| 3.1 | 0.5692 | 0.5713* | - | - | - | - | - | - |
| 3.2 | 0.5524 | - | - | - | - | - | - | - |
| 3.3 | 0.5367* | - | - | - | - | - | - | - |

The spread across L' at fixed u, truncated top bins excluded, with the binomial sampling error of
the two extreme entries:

| u | lengths compared | smallest PP share | largest | spread | shortest L' | longest L' |
|---|---|---|---|---|---|---|
| 2.0 | 8 | 0.8958 | 0.9080 | 1.34% | 50: 0.9080 +- 0.0106 | 579: 0.9001 +- 0.0011 |
| 2.1 | 8 | 0.7647 | 0.7766 | 1.54% | 50: 0.7716 +- 0.0113 | 579: 0.7647 +- 0.0010 |
| 2.2 | 8 | 0.6660 | 0.6798 | 2.03% | 50: 0.6711 +- 0.0093 | 579: 0.6660 +- 0.0007 |
| 2.3 | 7 | 0.5927 | 0.6071 | 2.37% | 50: 0.5991 +- 0.0071 | 400: 0.5935 +- 0.0007 |
| 2.4 | 6 | 0.5330 | 0.5480 | 2.74% | 50: 0.5480 +- 0.0054 | 300: 0.5342 +- 0.0007 |
| 2.5 | 5 | 0.4851 | 0.4956 | 2.12% | 50: 0.4956 +- 0.0040 | 200: 0.4862 +- 0.0008 |
| 2.6 | 5 | 0.4458 | 0.4545 | 1.92% | 50: 0.4545 +- 0.0030 | 200: 0.4458 +- 0.0005 |
| 2.7 | 4 | 0.4131 | 0.4193 | 1.46% | 50: 0.4193 +- 0.0022 | 150: 0.4131 +- 0.0005 |
| 2.8 | 3 | 0.3874 | 0.3903 | 0.75% | 50: 0.3903 +- 0.0017 | 100: 0.3874 +- 0.0007 |
| 2.9 | 3 | 0.3622 | 0.3655 | 0.91% | 50: 0.3655 +- 0.0012 | 100: 0.3622 +- 0.0005 |

So depth alone carries the type share to within 0.75% to 2.7% relative across an elevenfold change
of the core, and the residual is not noise: it is a smooth monotone drift downwards as t' grows,
several sampling errors wide at the ends. That is the size and the sign of the classical
second-order term of a rough-number density in 1 / ln t', not a dependence on the section: at any
fixed t' the two sections agree exactly wherever they overlap, because "core-free member with
Omega = j at depth u" is a property of the number and of t' alone and does not know which section
the number is in (a depth bin lying wholly inside the two
sections' overlap [16129, 7,946,761) holds literally the same slots in both tables: at L' = 50,
bin u = 2.5 has 201,599 slots, 118,235 core-free members of which 82,921 prime and 35,314 with
Omega = 2, 15,267 core-open slots and PP share 0.495644 in both. An identity, not evidence).

Prior art, one line, not derived here: the prime share among y-rough numbers up to x is
asymptotically 1 / (u omega(u)) with omega Buchstab's function, which on 2 <= u <= 3 is
1 / (1 + ln(u - 1)) - 0.7115 at u = 2.5, 0.6091 at u = 2.9, 0.5906 at u = 3.0, against the measured
0.6986-0.7013, 0.6024-0.6047 and 0.5857-0.5866; the measured profile sits 1-3% below the leading
term and converges to it as t' grows, in the same direction as the residual above. The PP share
among core-open slots is the twin analogue of the same profile; it has no classical closed form
located, and the measured profile above is the object.

**S21 (THE TYPE CENSUS AT DEPTH; measured, 0 exceptions to the depth-only form beyond the
classical residual).** The Omega-census of core-free members and the PP share among core-open slots
are functions of u = ln n / ln t' alone, to 2.7% relative over t' from 301 to 3475, with a residual
that is monotone in t', of the size of the classical 1/ln t' correction, and identical in bins with
and without Omega = 3 members. No dependence on the section exists at all: at fixed t' the census
is a property of the raw line.

### 3.2 The full tables

Rows are depth bins of width 0.1; the bins below 2.0 are collapsed into one row because S15 makes
every core-open slot there a twin exactly (prime share and PP share 1.000000 in every one of them,
0 exceptions on both sections). "tf runs" is the observed number of twin-free stretches counted as
runs (one per twin gap of at least L' slots) whose start falls in the bin; "predicted" is the
independent-slot analytic count with the bin's own PP share; "model sd" is the spread of the same
model over 20 simulated draws; "min K on tf" is the smallest core leftover over the twin-free
stretches starting in the bin.

### Base 3, section 4 = [16129, 260,467,321)

**L' = 50** (t' = 301, q = nextprime(t') = 307, core gears 60, tail gears 1815, starts 43408482). Core-free members 25402926 (Omega 1 / 2 / 3 / 4: 14218065 / 10991941 / 192920 / 0); core-open slots 3272959; PP 1027948; mean K 3.7699; min K over twin-free stretches 0; twin-free starts 12619848 (predicted 12621069.1); twin-free RUNS 307536 (predicted 307495.22; model's own spread 307405.80 +- 343.16; z +0.12).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 12412 | 6892 | 6892 | 0 | 0 | 1.000000 | 837 | 1.000000 | - | 16 | 16.00 | - | - | 0 |
| 2.0 | 11620 | 6237 | 5941 | 296 | 0 | 0.952541 | 750 | 0.908000 | 3.221 | 27 | 26.12 | 2.32 | +0.38 | 0 |
| 2.1 | 20562 | 11385 | 10021 | 1364 | 0 | 0.880193 | 1379 | 0.771574 | 3.357 | 55 | 59.61 | 5.48 | -0.84 | 0 |
| 2.2 | 36384 | 20624 | 16960 | 3664 | 0 | 0.822343 | 2548 | 0.671115 | 3.500 | 135 | 135.79 | 5.61 | -0.14 | 0 |
| 2.3 | 64384 | 37080 | 28741 | 8339 | 0 | 0.775108 | 4704 | 0.599065 | 3.654 | 280 | 274.28 | 9.12 | +0.63 | 0 |
| 2.4 | 113928 | 66340 | 48761 | 17579 | 0 | 0.735017 | 8549 | 0.548017 | 3.752 | 538 | 529.36 | 13.03 | +0.66 | 0 |
| 2.5 | 201599 | 118235 | 82921 | 35314 | 0 | 0.701324 | 15267 | 0.495644 | 3.786 | 1063 | 1056.47 | 24.12 | +0.27 | 0 |
| 2.6 | 356733 | 210150 | 141294 | 68856 | 0 | 0.672348 | 27267 | 0.454505 | 3.822 | 2014 | 2031.47 | 24.10 | -0.73 | 0 |
| 2.7 | 631249 | 372405 | 240889 | 131516 | 0 | 0.646847 | 48454 | 0.419264 | 3.838 | 3881 | 3859.61 | 41.16 | +0.52 | 0 |
| 2.8 | 1117013 | 658650 | 411220 | 247430 | 0 | 0.624338 | 85355 | 0.390299 | 3.821 | 7135 | 7187.35 | 47.85 | -1.09 | 0 |
| 2.9 | 1976580 | 1163165 | 703353 | 459812 | 0 | 0.604689 | 150688 | 0.365510 | 3.812 | 13112 | 13236.15 | 75.34 | -1.65 | 0 |
| 3.0 | 3497608 | 2052155 | 1203694 | 847406 | 1055 | 0.586551 | 265136 | 0.343152 | 3.790 | 24254 | 24168.95 | 87.23 | +0.98 | 0 |
| 3.1 | 6189105 | 3623657 | 2062554 | 1550256 | 10847 | 0.569191 | 466753 | 0.323831 | 3.771 | 43699 | 43746.94 | 103.96 | -0.46 | 0 |
| 3.2 | 10951727 | 6404051 | 3537660 | 2820064 | 46327 | 0.552410 | 824506 | 0.305112 | 3.764 | 78579 | 78708.79 | 145.09 | -0.89 | 0 |
| 3.3 | 18227627 | 10651900 | 5717164 | 4800045 | 134691 | 0.536727 | 1370766 | 0.287276 | 3.760 | 132748 | 132458.33 | 180.72 | +1.60 | 0 |

**L' = 75** (t' = 451, q = nextprime(t') = 457, core gears 85, tail gears 1790, starts 43408457). Core-free members 23825047 (Omega 1 / 2 / 3 / 4: 14218065 / 9583846 / 23136 / 0); core-open slots 2878730; PP 1027948; mean K 4.9738; min K over twin-free stretches 0; twin-free starts 6773255 (predicted 6766579.8); twin-free RUNS 163647 (predicted 163809.34; model's own spread 163806.60 +- 258.99; z -0.63).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 31212 | 16382 | 16382 | 0 | 0 | 1.000000 | 1900 | 1.000000 | - | 12 | 12.00 | - | - | 0 |
| 2.0 | 28562 | 14297 | 13624 | 673 | 0 | 0.952927 | 1584 | 0.895833 | 4.156 | 21 | 22.91 | 1.54 | -1.24 | 0 |
| 2.1 | 52628 | 27232 | 23967 | 3265 | 0 | 0.880104 | 3080 | 0.776299 | 4.388 | 55 | 58.95 | 4.09 | -0.97 | 0 |
| 2.2 | 96968 | 51365 | 42231 | 9134 | 0 | 0.822175 | 6006 | 0.679820 | 4.646 | 117 | 131.45 | 6.01 | -2.41 | 0 |
| 2.3 | 178668 | 96188 | 74428 | 21760 | 0 | 0.773776 | 11426 | 0.607124 | 4.796 | 286 | 293.41 | 17.44 | -0.42 | 0 |
| 2.4 | 329203 | 179248 | 131613 | 47635 | 0 | 0.734251 | 21562 | 0.541462 | 4.913 | 652 | 667.19 | 19.49 | -0.78 | 0 |
| 2.5 | 606572 | 332754 | 233071 | 99683 | 0 | 0.700430 | 40162 | 0.490513 | 4.966 | 1525 | 1488.59 | 29.62 | +1.23 | 0 |
| 2.6 | 1117636 | 615689 | 413225 | 202464 | 0 | 0.671159 | 74474 | 0.451621 | 4.998 | 3041 | 3109.88 | 44.59 | -1.54 | 0 |
| 2.7 | 2059283 | 1136331 | 733839 | 402492 | 0 | 0.645797 | 138024 | 0.417275 | 5.027 | 6422 | 6440.29 | 51.25 | -0.36 | 0 |
| 2.8 | 3794344 | 2092758 | 1304811 | 787947 | 0 | 0.623489 | 254194 | 0.387964 | 5.024 | 12952 | 12970.81 | 76.20 | -0.25 | 0 |
| 2.9 | 6991197 | 3847750 | 2322721 | 1525029 | 0 | 0.603657 | 466091 | 0.364116 | 5.000 | 25728 | 25735.38 | 101.42 | -0.07 | 0 |
| 3.0 | 12881615 | 7068232 | 4139685 | 2925023 | 3524 | 0.585675 | 853738 | 0.342708 | 4.971 | 50461 | 50451.08 | 137.71 | +0.07 | 0 |
| 3.1 | 15240643 | 8346821 | 4768468 | 3558741 | 19612 | 0.571292 | 1006489 | 0.325590 | 4.953 | 62375 | 62427.40 | 140.85 | -0.37 | 0 |

**L' = 100** (t' = 601, q = nextprime(t') = 607, core gears 108, tail gears 1767, starts 43408432). Core-free members 22855648 (Omega 1 / 2 / 3 / 4: 14218065 / 8637377 / 206 / 0); core-open slots 2649036; PP 1027948; mean K 6.1026; min K over twin-free stretches 0; twin-free starts 3649011 (predicted 3642980.0); twin-free RUNS 88748 (predicted 88344.56; model's own spread 88318.85 +- 260.86; z +1.55).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 57512 | 28958 | 28958 | 0 | 0 | 1.000000 | 3227 | 1.000000 | - | 5 | 5.00 | - | - | 0 |
| 2.0 | 53952 | 25865 | 24591 | 1274 | 0 | 0.950744 | 2714 | 0.903832 | 5.027 | 10 | 12.40 | 2.25 | -1.07 | 0 |
| 2.1 | 102305 | 50671 | 44504 | 6167 | 0 | 0.878293 | 5507 | 0.776648 | 5.383 | 43 | 38.00 | 4.77 | +1.05 | 0 |
| 2.2 | 193992 | 98322 | 80618 | 17704 | 0 | 0.819939 | 11019 | 0.679372 | 5.681 | 100 | 107.29 | 8.17 | -0.89 | 0 |
| 2.3 | 367852 | 189608 | 146426 | 43182 | 0 | 0.772256 | 21630 | 0.596764 | 5.880 | 295 | 288.15 | 12.41 | +0.55 | 0 |
| 2.4 | 697522 | 363681 | 266452 | 97229 | 0 | 0.732653 | 41664 | 0.539195 | 5.973 | 708 | 723.25 | 25.15 | -0.61 | 0 |
| 2.5 | 1322655 | 694650 | 485268 | 209382 | 0 | 0.698579 | 80180 | 0.488750 | 6.062 | 1754 | 1678.40 | 31.64 | +2.39 | 0 |
| 2.6 | 2508040 | 1322706 | 885786 | 436920 | 0 | 0.669677 | 153650 | 0.447725 | 6.126 | 3825 | 3851.89 | 41.28 | -0.65 | 0 |
| 2.7 | 4755766 | 2511876 | 1618521 | 893355 | 0 | 0.644347 | 291826 | 0.414675 | 6.136 | 8479 | 8435.57 | 77.66 | +0.56 | 0 |
| 2.8 | 9017977 | 4760609 | 2961774 | 1798835 | 0 | 0.622142 | 552982 | 0.387376 | 6.132 | 18049 | 18001.72 | 111.29 | +0.42 | 0 |
| 2.9 | 17100030 | 9007132 | 5425885 | 3581247 | 0 | 0.602399 | 1044805 | 0.362173 | 6.110 | 38345 | 38124.73 | 142.88 | +1.54 | 0 |
| 3.0 | 7230928 | 3801570 | 2249282 | 1552082 | 206 | 0.591672 | 439832 | 0.349056 | 6.083 | 17135 | 17078.17 | 86.61 | +0.66 | 0 |

**L' = 150** (t' = 901, q = nextprime(t') = 907, core gears 152, tail gears 1723, starts 43408382). Core-free members 21532901 (Omega 1 / 2 / 3 / 4: 14218065 / 7314836 / 0 / 0); core-open slots 2350966; PP 1027948; mean K 8.1239; min K over twin-free stretches 0; twin-free starts 1059017 (predicted 1056089.8); twin-free RUNS 25444 (predicted 25428.21; model's own spread 25473.95 +- 153.63; z +0.10).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 132612 | 62943 | 62943 | 0 | 0 | 1.000000 | 6550 | 1.000000 | - | 3 | 3.00 | - | - | 0 |
| 2.0 | 131859 | 59523 | 56533 | 2990 | 0 | 0.949767 | 6035 | 0.903065 | 6.868 | 4 | 4.16 | 1.35 | -0.12 | 0 |
| 2.1 | 260366 | 121429 | 106498 | 14931 | 0 | 0.877039 | 12504 | 0.770074 | 7.205 | 28 | 24.57 | 3.45 | +0.99 | 0 |
| 2.2 | 514109 | 245530 | 200945 | 44585 | 0 | 0.818413 | 25897 | 0.671468 | 7.555 | 60 | 63.41 | 6.52 | -0.52 | 0 |
| 2.3 | 1015146 | 493100 | 380120 | 112980 | 0 | 0.770878 | 52621 | 0.596568 | 7.775 | 203 | 206.10 | 11.97 | -0.26 | 0 |
| 2.4 | 2004480 | 984842 | 719897 | 264945 | 0 | 0.730977 | 106414 | 0.534986 | 7.963 | 631 | 585.68 | 20.68 | +2.19 | 0 |
| 2.5 | 3957987 | 1958743 | 1365833 | 592910 | 0 | 0.697301 | 213400 | 0.485150 | 8.087 | 1632 | 1587.28 | 50.19 | +0.89 | 0 |
| 2.6 | 7815297 | 3883807 | 2595503 | 1288304 | 0 | 0.668288 | 424438 | 0.446779 | 8.146 | 4088 | 4038.46 | 56.05 | +0.88 | 0 |
| 2.7 | 15431831 | 7679905 | 4939319 | 2740586 | 0 | 0.643148 | 841165 | 0.413126 | 8.176 | 9810 | 9930.80 | 76.70 | -1.57 | 0 |
| 2.8 | 12144844 | 6043079 | 3790474 | 2252605 | 0 | 0.627242 | 661942 | 0.392693 | 8.176 | 8985 | 8984.75 | 77.41 | +0.00 | 0 |

**L' = 200** (t' = 1201, q = nextprime(t') = 1213, core gears 195, tail gears 1680, starts 43408332). Core-free members 20593937 (Omega 1 / 2 / 3 / 4: 14218065 / 6375872 / 0 / 0); core-open slots 2150422; PP 1027948; mean K 9.9078; min K over twin-free stretches 0; twin-free starts 310557 (predicted 306242.9); twin-free RUNS 7608 (predicted 7535.22; model's own spread 7558.40 +- 47.47; z +1.53).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 237712 | 108217 | 108217 | 0 | 0 | 1.000000 | 10926 | 1.000000 | - | 1 | 1.01 | - | - | 0 |
| 2.0 | 248127 | 107379 | 102112 | 5267 | 0 | 0.950949 | 10298 | 0.902894 | 8.303 | 3 | 2.38 | 1.15 | +0.54 | 0 |
| 2.1 | 504231 | 225587 | 197889 | 27698 | 0 | 0.877218 | 22240 | 0.772167 | 8.822 | 6 | 8.98 | 1.76 | -1.69 | 0 |
| 2.2 | 1024668 | 469501 | 384351 | 85150 | 0 | 0.818637 | 47338 | 0.672652 | 9.240 | 37 | 34.65 | 5.98 | +0.39 | 1 |
| 2.3 | 2082273 | 970359 | 747899 | 222460 | 0 | 0.770745 | 99460 | 0.594621 | 9.553 | 146 | 130.86 | 11.69 | +1.30 | 0 |
| 2.4 | 4231479 | 1994959 | 1458082 | 536877 | 0 | 0.730883 | 207088 | 0.532986 | 9.788 | 413 | 416.91 | 17.84 | -0.22 | 0 |
| 2.5 | 8598972 | 4083861 | 2846822 | 1237039 | 0 | 0.697091 | 426478 | 0.486180 | 9.919 | 1251 | 1235.36 | 28.73 | +0.54 | 0 |
| 2.6 | 17474345 | 8333365 | 5567694 | 2765671 | 0 | 0.668121 | 874890 | 0.445798 | 10.013 | 3548 | 3525.30 | 50.17 | +0.45 | 0 |
| 2.7 | 9006724 | 4300709 | 2804999 | 1495710 | 0 | 0.652218 | 451704 | 0.424667 | 10.030 | 2203 | 2179.77 | 37.77 | +0.61 | 0 |

**L' = 300** (t' = 1801, q = nextprime(t') = 1811, core gears 277, tail gears 1598, starts 43408232). Core-free members 19304316 (Omega 1 / 2 / 3 / 4: 14218065 / 5086251 / 0 / 0); core-open slots 1889548; PP 1027948; mean K 13.0588; min K over twin-free stretches 1; twin-free starts 28216 (predicted 26077.9); twin-free RUNS 619 (predicted 632.47; model's own spread 630.20 +- 21.15; z -0.64).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 537912 | 231274 | 231274 | 0 | 0 | 1.000000 | 22130 | 1.000000 | - | 0 | 0.00 | - | - | - |
| 2.0 | 603404 | 246758 | 234836 | 11922 | 0 | 0.951685 | 22149 | 0.907445 | 11.013 | 0 | 0.07 | 0.44 | -0.16 | - |
| 2.1 | 1276907 | 540319 | 474004 | 66315 | 0 | 0.877267 | 50421 | 0.770195 | 11.847 | 0 | 1.63 | 1.37 | -1.19 | - |
| 2.2 | 2702154 | 1171492 | 958874 | 212618 | 0 | 0.818507 | 111769 | 0.669783 | 12.408 | 8 | 8.61 | 2.57 | -0.24 | 3 |
| 2.3 | 5718246 | 2521178 | 1942396 | 578782 | 0 | 0.770432 | 244575 | 0.592722 | 12.831 | 38 | 38.73 | 5.72 | -0.13 | 1 |
| 2.4 | 12100772 | 5398363 | 3943837 | 1454526 | 0 | 0.730562 | 529990 | 0.534250 | 13.139 | 138 | 150.85 | 12.81 | -1.00 | 1 |
| 2.5 | 20469136 | 9194932 | 6432844 | 2762088 | 0 | 0.699608 | 908514 | 0.488613 | 13.315 | 435 | 432.58 | 15.05 | +0.16 | 2 |

**L' = 400** (t' = 2401, q = nextprime(t') = 2411, core gears 355, tail gears 1520, starts 43408132). Core-free members 18421718 (Omega 1 / 2 / 3 / 4: 14218065 / 4203653 / 0 / 0); core-open slots 1720432; PP 1027948; mean K 15.8533; min K over twin-free stretches 1; twin-free starts 2855 (predicted 2281.8); twin-free RUNS 83 (predicted 54.23; model's own spread 55.35 +- 6.65; z +4.33).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 958111 | 395886 | 395886 | 0 | 0 | 1.000000 | 36335 | 1.000000 | - | 0 | 0.00 | - | - | - |
| 2.0 | 1131733 | 446122 | 424315 | 21807 | 0 | 0.951119 | 38770 | 0.905855 | 13.702 | 0 | 0.00 | 0.00 | - | - |
| 2.1 | 2464808 | 1004900 | 881043 | 123857 | 0 | 0.876747 | 90087 | 0.769079 | 14.620 | 0 | 0.20 | 0.22 | -0.88 | - |
| 2.2 | 5368102 | 2242401 | 1834019 | 408382 | 0 | 0.817882 | 205997 | 0.668505 | 15.350 | 2 | 1.75 | 1.89 | +0.13 | 7 |
| 2.3 | 11691286 | 4968017 | 3825075 | 1142942 | 0 | 0.769940 | 464255 | 0.593467 | 15.884 | 21 | 10.42 | 3.22 | +3.29 | 1 |
| 2.4 | 21794491 | 9364392 | 6857727 | 2506665 | 0 | 0.732320 | 884988 | 0.535577 | 16.242 | 60 | 41.86 | 5.67 | +3.20 | 2 |

**L' = 579** (t' = 3475, q = nextprime(t') = 3491, core gears 485, tail gears 1390, starts 43407953). Core-free members 17348989 (Omega 1 / 2 / 3 / 4: 14218065 / 3130924 / 0 / 0); core-open slots 1526144; PP 1027948; mean K 20.3561; min K over twin-free stretches 12; twin-free starts 1 (predicted 30.5); twin-free RUNS 1 (predicted 0.70; model's own spread 0.65 +- 0.81; z +0.36).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 2009914 | 790820 | 790820 | 0 | 0 | 1.000000 | 69103 | 1.000000 | - | 0 | 0.00 | - | - | - |
| 2.0 | 2535750 | 956447 | 907295 | 49152 | 0 | 0.948610 | 79351 | 0.900115 | 18.119 | 0 | 0.00 | 0.00 | - | - |
| 2.1 | 5730610 | 2235839 | 1955445 | 280394 | 0 | 0.874591 | 191741 | 0.764703 | 19.373 | 0 | 0.02 | 0.00 | - | - |
| 2.2 | 12950784 | 5176784 | 4223500 | 953284 | 0 | 0.815854 | 455511 | 0.666043 | 20.365 | 0 | 0.13 | 0.37 | -0.36 | - |
| 2.3 | 20181473 | 8189099 | 6341005 | 1848094 | 0 | 0.774323 | 730438 | 0.598826 | 20.956 | 1 | 0.55 | 0.76 | +0.59 | 12 |

### Base 7, section 3 = [2809, 7,946,761)

**L' = 50** (t' = 301, q = nextprime(t') = 307, core gears 60, tail gears 347, starts 1323942). Core-free members 775816 (Omega 1 / 2 / 3 / 4: 536043 / 239773 / 0 / 0); core-open slots 100181; PP 48249; mean K 3.7833; min K over twin-free stretches 0; twin-free starts 188750 (predicted 190342.5); twin-free RUNS 7186 (predicted 7198.79; model's own spread 7199.15 +- 47.48; z -0.27).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 14632 | 8360 | 8360 | 0 | 0 | 1.000000 | 1043 | 1.000000 | - | 16 | 16.00 | - | - | 0 |
| 2.0 | 11620 | 6237 | 5941 | 296 | 0 | 0.952541 | 750 | 0.908000 | 3.221 | 27 | 26.12 | 2.24 | +0.39 | 0 |
| 2.1 | 20562 | 11385 | 10021 | 1364 | 0 | 0.880193 | 1379 | 0.771574 | 3.357 | 55 | 59.61 | 5.90 | -0.78 | 0 |
| 2.2 | 36384 | 20624 | 16960 | 3664 | 0 | 0.822343 | 2548 | 0.671115 | 3.500 | 135 | 135.79 | 4.31 | -0.18 | 0 |
| 2.3 | 64384 | 37080 | 28741 | 8339 | 0 | 0.775108 | 4704 | 0.599065 | 3.654 | 280 | 274.28 | 9.89 | +0.58 | 0 |
| 2.4 | 113928 | 66340 | 48761 | 17579 | 0 | 0.735017 | 8549 | 0.548017 | 3.752 | 538 | 529.36 | 17.14 | +0.50 | 0 |
| 2.5 | 201599 | 118235 | 82921 | 35314 | 0 | 0.701324 | 15267 | 0.495644 | 3.786 | 1063 | 1056.47 | 19.56 | +0.33 | 0 |
| 2.6 | 356733 | 210150 | 141294 | 68856 | 0 | 0.672348 | 27267 | 0.454505 | 3.822 | 2014 | 2031.47 | 23.72 | -0.74 | 0 |
| 2.7 | 504149 | 297405 | 193044 | 104361 | 0 | 0.649095 | 38674 | 0.421162 | 3.836 | 3058 | 3069.70 | 35.07 | -0.33 | 0 |

**L' = 75** (t' = 451, q = nextprime(t') = 457, core gears 85, tail gears 322, starts 1323917). Core-free members 717839 (Omega 1 / 2 / 3 / 4: 536043 / 181796 / 0 / 0); core-open slots 85806; PP 48249; mean K 4.8607; min K over twin-free stretches 0; twin-free starts 69651 (predicted 71370.3); twin-free RUNS 2663 (predicted 2668.09; model's own spread 2664.75 +- 43.66; z -0.12).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 33432 | 17850 | 17850 | 0 | 0 | 1.000000 | 2106 | 1.000000 | - | 12 | 12.00 | - | - | 0 |
| 2.0 | 28562 | 14297 | 13624 | 673 | 0 | 0.952927 | 1584 | 0.895833 | 4.156 | 21 | 22.91 | 3.24 | -0.59 | 0 |
| 2.1 | 52628 | 27232 | 23967 | 3265 | 0 | 0.880104 | 3080 | 0.776299 | 4.388 | 55 | 58.95 | 5.69 | -0.69 | 0 |
| 2.2 | 96968 | 51365 | 42231 | 9134 | 0 | 0.822175 | 6006 | 0.679820 | 4.646 | 117 | 131.45 | 6.35 | -2.28 | 0 |
| 2.3 | 178668 | 96188 | 74428 | 21760 | 0 | 0.773776 | 11426 | 0.607124 | 4.796 | 286 | 293.41 | 13.37 | -0.55 | 0 |
| 2.4 | 329203 | 179248 | 131613 | 47635 | 0 | 0.734251 | 21562 | 0.541462 | 4.913 | 652 | 667.19 | 18.33 | -0.83 | 0 |
| 2.5 | 604530 | 331659 | 232330 | 99329 | 0 | 0.700509 | 40042 | 0.490435 | 4.968 | 1520 | 1482.17 | 25.87 | +1.46 | 0 |

**L' = 100** (t' = 601, q = nextprime(t') = 607, core gears 108, tail gears 299, starts 1323892). Core-free members 679389 (Omega 1 / 2 / 3 / 4: 536043 / 143346 / 0 / 0); core-open slots 76843; PP 48249; mean K 5.8038; min K over twin-free stretches 0; twin-free starts 25595 (predicted 26930.5); twin-free RUNS 1010 (predicted 1007.51; model's own spread 1003.35 +- 19.20; z +0.13).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 59732 | 30426 | 30426 | 0 | 0 | 1.000000 | 3433 | 1.000000 | - | 5 | 5.00 | - | - | 0 |
| 2.0 | 53952 | 25865 | 24591 | 1274 | 0 | 0.950744 | 2714 | 0.903832 | 5.027 | 10 | 12.40 | 2.31 | -1.04 | 0 |
| 2.1 | 102305 | 50671 | 44504 | 6167 | 0 | 0.878293 | 5507 | 0.776648 | 5.383 | 43 | 38.00 | 5.71 | +0.88 | 0 |
| 2.2 | 193992 | 98322 | 80618 | 17704 | 0 | 0.819939 | 11019 | 0.679372 | 5.681 | 100 | 107.29 | 6.61 | -1.10 | 0 |
| 2.3 | 367852 | 189608 | 146426 | 43182 | 0 | 0.772256 | 21630 | 0.596764 | 5.880 | 295 | 288.14 | 11.19 | +0.61 | 0 |
| 2.4 | 546158 | 284497 | 209478 | 75019 | 0 | 0.736310 | 32540 | 0.543700 | 5.958 | 557 | 556.68 | 16.28 | +0.02 | 0 |

**L' = 150** (t' = 901, q = nextprime(t') = 907, core gears 152, tail gears 255, starts 1323842). Core-free members 627558 (Omega 1 / 2 / 3 / 4: 536043 / 91515 / 0 / 0); core-open slots 65649; PP 48249; mean K 7.4371; min K over twin-free stretches 0; twin-free starts 3212 (predicted 3853.6); twin-free RUNS 133 (predicted 144.44; model's own spread 145.10 +- 7.35; z -1.56).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 134832 | 64411 | 64411 | 0 | 0 | 1.000000 | 6756 | 1.000000 | - | 3 | 3.00 | - | - | 0 |
| 2.0 | 131859 | 59523 | 56533 | 2990 | 0 | 0.949767 | 6035 | 0.903065 | 6.868 | 4 | 4.16 | 1.32 | -0.12 | 0 |
| 2.1 | 260366 | 121429 | 106498 | 14931 | 0 | 0.877039 | 12504 | 0.770074 | 7.205 | 28 | 24.57 | 2.99 | +1.15 | 0 |
| 2.2 | 514109 | 245530 | 200945 | 44585 | 0 | 0.818413 | 25897 | 0.671468 | 7.555 | 60 | 63.40 | 5.42 | -0.63 | 0 |
| 2.3 | 282825 | 136665 | 107656 | 29009 | 0 | 0.787736 | 14457 | 0.624265 | 7.666 | 38 | 49.31 | 5.71 | -1.98 | 0 |

**L' = 200** (t' = 1201, q = nextprime(t') = 1213, core gears 195, tail gears 212, starts 1323792). Core-free members 593561 (Omega 1 / 2 / 3 / 4: 536043 / 57518 / 0 / 0); core-open slots 58789; PP 48249; mean K 8.8794; min K over twin-free stretches 0; twin-free starts 390 (predicted 532.3); twin-free RUNS 12 (predicted 21.02; model's own spread 22.85 +- 4.97; z -1.82).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 239932 | 109685 | 109685 | 0 | 0 | 1.000000 | 11132 | 1.000000 | - | 1 | 1.01 | - | - | 0 |
| 2.0 | 248127 | 107379 | 102112 | 5267 | 0 | 0.950949 | 10298 | 0.902894 | 8.303 | 3 | 2.38 | 0.98 | +0.63 | 0 |
| 2.1 | 504231 | 225587 | 197889 | 27698 | 0 | 0.877218 | 22240 | 0.772167 | 8.822 | 6 | 8.98 | 2.61 | -1.14 | 0 |
| 2.2 | 331701 | 150910 | 126357 | 24553 | 0 | 0.837300 | 15119 | 0.704147 | 9.115 | 2 | 8.66 | 3.79 | -1.76 | 1 |

**L' = 300** (t' = 1801, q = nextprime(t') = 1811, core gears 277, tail gears 130, starts 1323692). Core-free members 554875 (Omega 1 / 2 / 3 / 4: 536043 / 18832 / 0 / 0); core-open slots 51479; PP 48249; mean K 11.6614; min K over twin-free stretches - (no twin-free stretch of this length exists); twin-free starts 0 (predicted 4.4); twin-free RUNS 0 (predicted 0.17; model's own spread 0.15 +- 0.37; z -0.46).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 540132 | 232742 | 232742 | 0 | 0 | 1.000000 | 22336 | 1.000000 | - | 0 | 0.00 | - | - | - |
| 2.0 | 603404 | 246758 | 234836 | 11922 | 0 | 0.951685 | 22149 | 0.907445 | 11.013 | 0 | 0.07 | 0.00 | - | - |
| 2.1 | 180455 | 75375 | 68465 | 6910 | 0 | 0.908325 | 6994 | 0.831284 | 11.623 | 0 | 0.10 | 0.37 | -0.27 | - |

**L' = 400** (t' = 2401, q = nextprime(t') = 2411, core gears 355, tail gears 52, starts 1323592). Core-free members 538727 (Omega 1 / 2 / 3 / 4: 536043 / 2684 / 0 / 0); core-open slots 48691; PP 48249; mean K 14.7043; min K over twin-free stretches - (no twin-free stretch of this length exists); twin-free starts 0 (predicted 0.0); twin-free RUNS 0 (predicted 0.00; model's own spread 0.00 +- 0.00; degenerate).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 960331 | 397354 | 397354 | 0 | 0 | 1.000000 | 36541 | 1.000000 | - | 0 | 0.00 | - | - | - |
| 2.0 | 363660 | 141373 | 138689 | 2684 | 0 | 0.981015 | 12150 | 0.963621 | 13.360 | 0 | 0.00 | 0.00 | - | - |

**L' = 579** (t' = 3475, q = nextprime(t') = 3491, core gears 485, tail gears 0, starts 1323413). Core-free members 535965 (Omega 1 / 2 / 3 / 4: 535965 / 0 / 0 / 0); core-open slots 48236; PP 48236; mean K 21.0871; min K over twin-free stretches - (no twin-free stretch of this length exists); twin-free starts 0 (predicted 0.0); twin-free RUNS 0 (predicted 0.00; model's own spread 0.00 +- 0.00; degenerate).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 1323991 | 535965 | 535965 | 0 | 0 | 1.000000 | 48236 | 1.000000 | - | 0 | 0.00 | - | - | - |

**L' = 254** (t' = 1525, q = nextprime(t') = 1531, core gears 239, tail gears 168, starts 1323738). Core-free members 569499 (Omega 1 / 2 / 3 / 4: 536043 / 33456 / 0 / 0); core-open slots 54183; PP 48249; mean K 10.3927; min K over twin-free stretches 1; twin-free starts 1 (predicted 39.7); twin-free RUNS 1 (predicted 1.74; model's own spread 2.25 +- 1.65; z -0.45).

| u | slots | core-free | Om=1 | Om=2 | Om=3 | prime share | core-open | PP share | mean K | tf runs | predicted | model sd | z | min K on tf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 2.0 | 387136 | 170817 | 170817 | 0 | 0 | 1.000000 | 16815 | 1.000000 | - | 0 | 0.00 | - | - | - |
| 2.0 | 419102 | 175613 | 166847 | 8766 | 0 | 0.950083 | 16297 | 0.900718 | 9.877 | 0 | 0.28 | 0.75 | -0.37 | - |
| 2.1 | 517753 | 223069 | 198379 | 24690 | 0 | 0.889317 | 21071 | 0.795169 | 10.337 | 1 | 1.46 | 1.23 | -0.37 | 1 |


## 4. The Omega thresholds

The Omega-census does not begin gradually: each Omega opens at one exact number.

**S17 (THE OMEGA THRESHOLDS; proved; the two-prime lemma extended to every j).** Let q =
nextprime(t'). The smallest t'-rough number with Omega >= j is exactly q^j, and q^j itself is
t'-rough with Omega = j. Hence the core-free members below q^2 are exactly the primes (this is
S15, kernel CoreLeftover.prime_of_rough_lt_sq / twin_of_rough), those below q^3 are the primes and
the products of two primes (kernel primeOrSemiprime_of_rough_lt_cube), those below q^4 add the
products of three, and in depth the j-th threshold sits at u = j ln q / ln t', strictly above j and
falling to j as t' grows. *Proof.* A t'-rough number with j prime factors counted with multiplicity
is a product of j primes each at least q, hence at least q^j; and q^j has exactly j. QED.

Verified exactly, 0 exceptions. The first core-free member of the section at each Omega, against
q^j:

| section | L' | t' | q | q^2 | first Omega = 2 seen | q^3 | first Omega = 3 seen | q^4 | first Omega = 4 seen |
|---|---|---|---|---|---|---|---|---|---|
| base 3 | 50 | 301 | 307 | 94,249 | 94,249 | 28,934,443 | 28,934,443 | 8.88 x 10^9 | above the section |
| base 3 | 75 | 451 | 457 | 208,849 | 208,849 | 95,443,993 | 95,443,993 | 4.36 x 10^10 | above the section |
| base 3 | 100 | 601 | 607 | 368,449 | 368,449 | 223,648,543 | 223,648,543 | 1.36 x 10^11 | above the section |
| base 3 | 150 | 901 | 907 | 822,649 | 822,649 | 7.46 x 10^8 | above the section | - | - |
| base 3 | 200 | 1201 | 1213 | 1,471,369 | 1,471,369 | 1.78 x 10^9 | above the section | - | - |
| base 3 | 300 | 1801 | 1811 | 3,279,721 | 3,279,721 | 5.94 x 10^9 | above the section | - | - |
| base 3 | 400 | 2401 | 2411 | 5,812,921 | 5,812,921 | 1.40 x 10^10 | above the section | - | - |
| base 3 | 579 | 3475 | 3491 | 12,187,081 | 12,187,081 | 4.25 x 10^10 | above the section | - | - |
| base 7 | 50 | 301 | 307 | 94,249 | 94,249 | 28,934,443 | above the section | - | - |
| base 7 | 254 | 1525 | 1531 | 2,343,961 | 2,343,961 | 3.59 x 10^9 | above the section | - | - |
| base 7 | 579 | 3475 | 3491 | 12,187,081 | above the section | - | - | - | - |
| base 3 (diagnostic) | 18 | 109 | 113 | 12,769, below the cut | 16,637 | 1,442,897 | 1,442,897 | 163,047,361 | 163,047,361 |
| base 3 (diagnostic) | 20 | 121 | 127 | 16,129, the cut itself | 16,637 | 2,048,383 | 2,048,383 | 260,144,641 | 260,144,641 |
| base 3 (diagnostic) | 25 | 151 | 157 | 24,649 | 24,649 | 3,869,893 | 3,869,893 | 6.08 x 10^8 | none, as predicted |

Every threshold that lies inside the section is hit exactly: 11 of 11 for Omega = 2, 5 of 5 for
Omega = 3, 2 of 2 for Omega = 4, and 0 core-free members with Omega >= j anywhere below q^j. The
two rows where the first Omega = 2 member seen is not q^2 are the two where q^2 lies below the
section's own floor 16,129: at t' = 109, q^2 = 12,769 is below the cut, and at t' = 121, q^2 =
16,129 IS the cut, whose slot is outside the section. Not exceptions; the section starts above the
threshold. The negative control is L' = 25, where q^4 = 607,573,201 lies above the section top and
exactly 0 Omega = 4 members are found; the positive control is L' = 20, where q^4 = 260,144,641
lies 322,680 below the section top and exactly 1 Omega = 4 member exists, that number itself,
127^4.

The censuses on the diagnostic grid (whole section, both members of every slot):

| L' | t' | core-free members | Omega = 1 | Omega = 2 | Omega = 3 | Omega = 4 | deepest u |
|---|---|---|---|---|---|---|---|
| 18 | 109 | 30,153,861 | 14,218,065 | 14,415,910 | 1,519,851 | 35 | 4.131 |
| 20 | 121 | 29,885,909 | 14,218,065 | 14,246,085 | 1,421,758 | 1 | 4.108 |
| 25 | 151 | 28,608,581 | 14,218,065 | 13,402,066 | 988,450 | 0 | 3.855 |

P2 is confirmed with 0 exceptions in both directions, and the general object the brief asked for,
the Omega-census at depth u, is pinned at every threshold by S17. The reading that matters for the
step: "the leftover members are P or P1 P2" is exactly the statement u < 3 ln q / ln t', and it is
false on 89% of the base-3 section as soon as the core is taken at L' = 50 (q^3 = 2.89 x 10^7
against a section top of 2.60 x 10^8).

## 5. The independence test (the deliverable)

The question is whether the count of twin-free stretches follows the independent-slot model over
the whole (L', u) region, and in particular in the bins that hold Omega = 3 members, where the
P-against-P1P2 dichotomy is false and a sign-sensitive count cannot see the type at all.

### 5.1 Whole sections

Observed twin-free RUNS against the model's prediction and the model's own spread (20 draws):

| section | L' | mean K | min K on twin-free | twin-free starts obs / pred | runs obs | runs pred | model mean +- sd | z |
|---|---|---|---|---|---|---|---|---|
| base 3 | 50 | 3.7699 | 0 | 12,619,848 / 12,621,069 | 307,536 | 307,495.22 | 307,405.80 +- 343.16 | +0.12 |
| base 3 | 75 | 4.9738 | 0 | 6,773,255 / 6,766,580 | 163,647 | 163,809.34 | 163,806.60 +- 258.99 | -0.63 |
| base 3 | 100 | 6.1026 | 0 | 3,649,011 / 3,642,980 | 88,748 | 88,344.56 | 88,318.85 +- 260.86 | +1.55 |
| base 3 | 150 | 8.1239 | 0 | 1,059,017 / 1,056,090 | 25,444 | 25,428.21 | 25,473.95 +- 153.63 | +0.10 |
| base 3 | 200 | 9.9078 | 0 | 310,557 / 306,243 | 7,608 | 7,535.22 | 7,558.40 +- 47.47 | +1.53 |
| base 3 | 300 | 13.0588 | 1 | 28,216 / 26,078 | 619 | 632.47 | 630.20 +- 21.15 | -0.64 |
| base 3 | 400 | 15.8533 | 1 | 2,855 / 2,282 | 83 | 54.23 | 55.35 +- 6.65 | **+4.33** |
| base 3 | 579 | 20.3561 | 12 | 1 / 30.5 | 1 | 0.70 | 0.65 +- 0.81 | +0.36 |
| base 7 | 50 | 3.7833 | 0 | 188,750 / 190,343 | 7,186 | 7,198.79 | 7,199.15 +- 47.48 | -0.27 |
| base 7 | 75 | 4.8607 | 0 | 69,651 / 71,370 | 2,663 | 2,668.09 | 2,664.75 +- 43.66 | -0.12 |
| base 7 | 100 | 5.8038 | 0 | 25,595 / 26,931 | 1,010 | 1,007.51 | 1,003.35 +- 19.20 | +0.13 |
| base 7 | 150 | 7.4371 | 0 | 3,212 / 3,854 | 133 | 144.44 | 145.10 +- 7.35 | -1.56 |
| base 7 | 200 | 8.8794 | 0 | 390 / 532 | 12 | 21.02 | 22.85 +- 4.97 | -1.82 |
| base 7 | 254 | 10.3927 | 1 | 1 / 39.7 | 1 | 1.74 | 2.25 +- 1.65 | -0.45 |
| base 7 | 300 | 11.6614 | - | 0 / 4.4 | 0 | 0.17 | 0.15 +- 0.37 | -0.46 |
| base 7 | 400 | 14.7043 | - | 0 / 0.0 | 0 | 0.00 | 0.00 | - |
| base 7 | 579 | 21.0871 | - | 0 / 0.0 | 0 | 0.00 | 0.00 | - |

Fourteen of the fifteen non-degenerate lengths (base 7 at L' = 400 and 579 predict and observe 0)
lie within 1.9 sd of the independent count. One does not: base 3 at L' = 400, 83 runs observed against 54.23 predicted, +4.33 sd.

The twin-free START count is NOT on the prediction at the long lengths (2,855 against 2,282 at
L' = 400; 1 against 30.5 at L' = 579). That is N16's start-clustering, the artefact of counting one
twin gap once per position it covers, and it is why the run count is the statistic.

### 5.2 By depth, and the bins that hold Omega = 3 members

Aggregating the per-(L', u) rows of section 3 over the two depth regimes on base 3, where the
counts are large:

| region | bins | twin-free runs observed | predicted | combined model sd | z |
|---|---|---|---|---|---|
| 2.0 <= u < 3.0, all eight L' | 62 | 184,398 | 184,223.3 | 315.9 | +0.55 |
| u >= 3.0 (Omega = 3 members present), L' = 50, 75, 100 | 7 | 409,251 | 409,039.7 | 344.1 | +0.61 |

The seven bins above depth 3, in full:

| L' | u | Omega = 3 members in the bin | core-open | PP share | mean K | runs observed | predicted | model sd | z |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 3.0 | 1,055 | 265,136 | 0.3432 | 3.790 | 24,254 | 24,168.95 | 87.23 | +0.98 |
| 50 | 3.1 | 10,847 | 466,753 | 0.3238 | 3.771 | 43,699 | 43,746.94 | 103.96 | -0.46 |
| 50 | 3.2 | 46,327 | 824,506 | 0.3051 | 3.764 | 78,579 | 78,708.79 | 145.09 | -0.89 |
| 50 | 3.3 | 134,691 | 1,370,766 | 0.2873 | 3.760 | 132,748 | 132,458.33 | 180.72 | +1.60 |
| 75 | 3.0 | 3,524 | 853,738 | 0.3427 | 4.971 | 50,461 | 50,451.08 | 137.71 | +0.07 |
| 75 | 3.1 | 19,612 | 1,006,489 | 0.3256 | 4.953 | 62,375 | 62,427.40 | 140.85 | -0.37 |
| 100 | 3.0 | 206 | 439,832 | 0.3491 | 6.083 | 17,135 | 17,078.17 | 86.61 | +0.66 |

409,251 twin-free runs in the region where the leftover's members are no longer P or P1 P2, against
409,039.7 predicted by slots that do not talk to each other: +0.6 sd. Crossing depth 3 changes
nothing in the count.

### 5.3 The one bin that deviates, and its second and third measurements

The deviation is base 3 at L' = 400, and inside it two adjacent depth bins:

| L' | u | core-open | PP share | mean K | runs observed | predicted | model sd | z |
|---|---|---|---|---|---|---|---|---|
| 400 | 2.2 | 205,997 | 0.668505 | 15.350 | 2 | 1.75 | 1.89 | +0.13 |
| 400 | 2.3 | 464,255 | 0.593467 | 15.884 | 21 | 10.42 | 3.22 | **+3.29** |
| 400 | 2.4 | 884,988 | 0.535577 | 16.243 | 60 | 41.86 | 5.67 | **+3.20** |

Neither bin holds an Omega = 3 member: at t' = 2401 the whole section lies below q^3 = 1.4 x 10^10.
So whatever this is, it is not the leftover crossing depth 3.

**Second measurement, a different calibration of the model's p.** A bin 0.1 wide in u at the top of
a section spans a large range of n - the top bin at L' = 400 runs from n = 3.2 x 10^7 to
2.6 x 10^8 - and p varies inside it, which by convexity makes a bin-calibrated prediction of "no
twin here" too small. Recomputed with p taken from bins of width 0.02, and from a local sliding
estimate over W slots (research/stack/r6/leftover_runs.py):

| calibration of p | predicted twin-free starts | obs / pred | predicted runs | obs / pred |
|---|---|---|---|---|
| depth bins of width 0.1 | 2,281.77 | 1.251 | 54.23 | 1.531 |
| depth bins of width 0.02 | 2,337.14 | 1.222 | 55.25 | 1.502 |
| local, W = 200,001 slots | 2,347.97 | 1.216 | 55.48 | 1.496 |
| local, W = 1,000,001 slots | 2,340.88 | 1.220 | 55.32 | 1.500 |
| one global constant | 1,595.04 | 1.790 | 39.53 | 2.100 |

The calibration is not the cause: every local calibration gives 55.3 +- 0.2 against 83 observed,
and simulating the locally calibrated model gives runs 55.00 +- 8.54, z = +3.28. The deviation
survives the second measurement.

**Third measurement, one fixed core so the gap-length bands are consistent.** With the core tied to
L' each length is predicted by a different model and the counts N(>= L') cannot be differenced.
Fixing the core at t' = 2401 and sweeping the stretch length G with that one model
(research/stack/r6/leftover_tail.py):

| G | mean K | observed twin gaps >= G | predicted | model mean +- sd | z | observed var of PP over stretches | model var | ratio |
|---|---|---|---|---|---|---|---|---|
| 250 | 9.908 | 2,169 | 2,150.61 | 2,150.35 +- 34.32 | +0.54 | 5.6791 | 5.6763 | 1.0005 |
| 300 | 11.890 | 619 | 637.29 | 636.60 +- 27.52 | -0.66 | 6.9399 | 6.9314 | 1.0012 |
| 325 | 12.881 | 341 | 341.26 | 340.95 +- 18.09 | -0.01 | 7.5850 | 7.5741 | 1.0014 |
| 350 | 13.872 | 205 | 185.21 | 181.70 +- 12.05 | +1.64 | 8.2388 | 8.2264 | 1.0015 |
| 375 | 14.862 | 133 | 102.76 | 103.10 +- 13.32 | +2.27 | 8.9119 | 8.8981 | 1.0015 |
| 400 | 15.853 | 83 | 55.32 | 55.65 +- 10.27 | +2.70 | 9.5967 | 9.5812 | 1.0016 |
| 425 | 16.844 | 39 | 29.69 | 30.30 +- 5.64 | +1.65 | 10.2850 | 10.2686 | 1.0016 |
| 450 | 17.835 | 17 | 16.72 | 16.30 +- 5.17 | +0.05 | 10.9920 | 10.9747 | 1.0016 |
| 475 | 18.826 | 10 | 9.04 | 9.35 +- 2.92 | +0.33 | 11.7147 | 11.6966 | 1.0016 |
| 500 | 19.817 | 4 | 4.97 | 5.75 +- 2.40 | -0.40 | 12.4463 | 12.4268 | 1.0016 |
| 550 | 21.798 | 1 | 1.50 | 1.95 +- 1.23 | -0.41 | 13.9342 | 13.9108 | 1.0017 |

and the bands, which are now differences within one model:

| band of twin-gap lengths, slots | observed | predicted | Poisson z |
|---|---|---|---|
| [250, 300) | 1,550 | 1,513.32 | +0.94 |
| [300, 325) | 278 | 296.03 | -1.05 |
| [325, 350) | 136 | 156.05 | -1.60 |
| [350, 375) | 72 | 82.45 | -1.15 |
| [375, 400) | 50 | 47.44 | +0.37 |
| [400, 425) | 44 | 25.64 | **+3.63** |
| [425, 450) | 22 | 12.97 | +2.51 |
| [450, 475) | 7 | 7.68 | -0.25 |
| [475, 500) | 6 | 4.07 | +0.96 |
| [500, 550) | 3 | 3.46 | -0.25 |

Three things fall out of the third measurement.

(i) The excess is one bump, confined to twin gaps of 400 to 450 slots: 66 observed against 38.6
predicted. Outside it the bands go the other way - [300, 375) holds 486 against 534.5 - so over the
whole range [300, 450) the section has 602 twin gaps against 620.6 predicted, i.e. BELOW the
independent count. The cumulative statistic N(>= 400) is high only because it sits just above the
bump; by N(>= 450) it is back to 17 against 16.7, and the neighbouring lengths in the L' sweep are
ordinary (L' = 350 at +0.79, L' = 450 at +0.17, L' = 500 at -0.43, all against the locally
calibrated model with its own simulated spread).

(ii) The second moment is exactly on the model. The variance of the twin count over stretches of G
slots matches the variance the independent-slot model itself has - the mean of the within-stretch
variance plus the variance of the stretch's own mean - to between 0.05% and 0.17% at every G from
250 to 550. Local overdispersion of the twins relative to the core-open pattern is the one
mechanism that would manufacture extra long gaps, and at the second moment there is none.

(iii) The other section deviates the other way in the same relative regime. Base 7 at L' = 150 and
200 (0.59 and 0.79 of its record) sits at -1.56 and -1.82 sd; base 3 at L' = 400 (0.69 of its
record) at +4.33.

The honest reading: the deviation is real as a count (83 against 55.3, and the band 44 against
25.6), it survives recalibration, and it has no candidate mechanism - it is a bump in the shape of
one section's twin-gap histogram, compensated by a deficit immediately below it, with the second
moment exactly on the model and the sign reversed on the other section. It is recorded here in
full, and it is where a fourth measurement would go: the same band decomposition on a third
section, or on the base-3 section split into halves.

## 6. The fuels

The composite core-free members that sit in core-open slots inside a twin gap of at least L' slots
- the members that actually finish a twin-free stretch - with their largest prime factor (the
fuel) tested against the brief's P2 > n / p_k:

| section | L' | composite members | Omega = 2 | Omega = 3 | satisfy P2 > n/p_k | Omega = 2 exceptions | Omega >= 3 satisfying | fuel at or above the cut c |
|---|---|---|---|---|---|---|---|---|
| base 3 | 50 | 1,979,641 | 1,944,617 | 35,024 | 1,944,617 | 0 | 0 | 1,678,739 (84.8%) |
| base 3 | 75 | 1,114,057 | 1,111,188 | 2,869 | 1,111,188 | 0 | 0 | 951,367 (85.4%) |
| base 3 | 100 | 644,040 | 644,025 | 15 | 644,025 | 0 | 0 | 549,037 (85.2%) |
| base 3 | 150 | 207,528 | 207,528 | 0 | 207,528 | 0 | 0 | 176,332 (85.0%) |
| base 3 | 200 | 67,057 | 67,057 | 0 | 67,057 | 0 | 0 | 57,077 (85.1%) |
| base 3 | 300 | 6,308 | 6,308 | 0 | 6,308 | 0 | 0 | 5,348 (84.8%) |
| base 3 | 400 | 867 | 867 | 0 | 867 | 0 | 0 | 722 (83.3%) |
| base 3 | 579 | 15 | 15 | 0 | 15 | 0 | 0 | 15 (100%) |
| base 7 | 50 | 29,480 | 29,480 | 0 | 29,480 | 0 | 0 | 22,984 (78.0%) |
| base 7 | 100 | 4,089 | 4,089 | 0 | 4,089 | 0 | 0 | 3,053 (74.7%) |
| base 7 | 200 | 19 | 19 | 0 | 19 | 0 | 0 | 10 (52.6%) |
| base 7 | 254 | 1 | 1 | 0 | 1 | 0 | 0 | 0 (0%) |
| base 3 (diag.) | 18 | 4,627,300 | 4,185,124 | 442,162 (+14 at Omega = 4) | 4,188,606 | 0 | 3,482 | 3,698,519 (79.9%) |
| base 3 (diag.) | 20 | 4,451,751 | 4,046,644 | 405,107 | 4,047,172 | 0 | 528 | 3,572,552 (80.2%) |
| base 3 (diag.) | 25 | 3,772,873 | 3,512,436 | 260,437 | 3,512,436 | 0 | 0 | 3,081,872 (81.7%) |

So P4 holds for 3,981,605 of 3,981,605 Omega = 2 members on base 3 and 44,984 of 44,984 on base 7,
and fails for 37,908 of 37,908 Omega = 3 members on the brief's grid: exactly the prover's P5. The
threshold at which it switches is exact and is verified in both directions by the diagnostic grid.

**S20 (THE FUEL IDENTITY IS THE STATEMENT THAT THE DEPTH IS BELOW 3; proved, threshold verified in
both directions).** On a section [c, c') with c' = p_k^2, core t', q = nextprime(t'), let m be a
composite core-free member of the section and P2 its largest prime factor. If Omega(m) = 2 then
m = P1 P2 with P1 <= sqrt(m) < p_k, so P2 > m / p_k always. If Omega(m) >= 3 then m / P2 is a
product of at least two primes at least q, so m / P2 >= q^2, and P2 > m / p_k fails for every such
m as soon as q^2 >= p_k. *Verified.* On base 3, p_k = 16,139: at L' = 25 and above, q^2 >= 24,649 >
p_k and exactly 0 of the Omega >= 3 members satisfy it (0 of 37,908 on the brief's grid, 0 of
260,437 at L' = 25); at t' = 109 and t' = 121, where q^2 = 12,769 and 16,129 are BELOW p_k, exactly
3,482 of 442,176 and 528 of 405,107 do. 0 Omega = 2 exceptions anywhere, over 4,026,589 members on
base 3 and 44,984 on base 7. So "the composite that kills a core-open slot is (a tail gear) x (a
gear of the section's own machine)" is not an extra structure of the record: it is the two-prime
lemma in disguise, and it expires at depth 3 along with it.

The 85% figure is the other half of the fuel question and is a separate fact: at every L' on base 3
about 85% of these composite members have their largest prime factor at or above the cut c = 16,129
(a gear of the section's own machine), and the fraction rises to 100% at the record stretch itself,
which sits at 255.9 M of 260.5 M, i.e. at the top of the section where m / P1 > m / p_k > c is
forced. On base 7 the same fraction falls from 78.0% at L' = 50 to 0% at the record - the record's
one composite there, 4,871,171 = 2039 x 2389, has its fuel in the tail - so "the fuels are the
section's own primes" is a property of where in the section the stretch sits, not of the finish.

## 7. Verdict

**S22 (THE LEFTOVER'S RUN COUNT IS THE INDEPENDENT COUNT IN (L', u); measured, one deviation
recorded).** Take a section, a stretch length L', the core the primes at most t' = 6L' + 1, and the
section's exact core-open pattern. Model each core-open slot as a twin independently with the
probability observed among the core-open slots of its own depth bin. Then the number of twin-free
stretches of L' slots, counted as runs, is the model's count: over the eight lengths of the brief's
grid on base 3 and the nine on base 7, 14 of the 15 non-degenerate lengths lie within 1.9 sd of the
model's own simulated spread; the 62 bins with 2 <= u < 3 hold 184,398 runs against 184,223.3
predicted (+0.55 sd) and the 7 bins with u >= 3, where core-free members with Omega = 3 exist and
the P-against-P1P2 dichotomy is false, hold 409,251 against 409,039.7 (+0.61 sd). The exception is
base 3 at L' = 400: 83 runs against 54.23, +4.33 sd, concentrated in the depth bins 2.3 (21 against
10.42) and 2.4 (60 against 41.86), neither of which holds an Omega = 3 member; under a locally
calibrated p it is 83 against 55.32 (+3.28 sd), and under a single fixed core it decomposes into
one band of gap lengths, [400, 450) slots, holding 66 against 38.6, with the bands immediately
below it at 486 against 534.5, so that over [300, 450) the section holds 602 twin gaps against
620.6 predicted. The second moment carries no trace of it: the variance of the twin count over
stretches of G slots is the model's own variance to within 0.17% at every G from 250 to 550.

Reading the verdict against the root. The step at the core - "on a stretch of L slots with K > 0
core-open slots, at least one is a twin", equivalently "the products of two survivors of the core
cannot meet every core-open slot of a stretch as long as the section's record" - has now been
measured in its own two intrinsic parameters (L', u) rather than at one point per section, and in
the region where the object is no longer P against P1 P2. Nothing new appears there. The type of a
core-free member is a function of depth alone (S21), its Omega opens at exact powers of
nextprime(t') (S17), the "fuel is a gear of the section's own machine" statement is the two-prime
lemma in disguise and expires at depth 3 with it (S20), and the count of twin-free stretches is the
independent count everywhere the counts are large (S22). The step at the core is therefore **ROOT
in (L, u)**: what remains of it after this branch is a bound on the count of PP among independent
leftovers, i.e. the conjecture restated, with the depth law S15 and the thresholds S17 as its only
structure.

The single deviation is reported, not explained. It is 27 twin gaps too many in one band of one
section, compensated below, with no mechanism candidate surviving three measurements and with the
sign reversed on the other section. It is not filed as a finding; it is filed as the place a fourth
measurement goes.

### Scorecard

| # | prediction | verdict |
|---|---|---|
| P1 | the PP share among core-open slots depends on u alone, the same at fixed u for every L' | **held to 2.7%, refuted as exact.** Across t' from 301 to 3475 the spread at fixed u is 0.75% to 2.74% relative; the residual is a smooth monotone drift in t' of the size of the classical 1/ln t' term, not a section effect (S21) |
| P2 | Omega = 3 from u = 3 and Omega = 4 from u = 4, exactly, 0 exceptions | **held exactly, with the exact threshold: nextprime(t')^j, not t'^j.** First member seen equals q^j in 11 of 11 cases at j = 2, 5 of 5 at j = 3, 2 of 2 at j = 4; 0 members below any threshold; the negative control (q^4 above the section) gives exactly 0 and the positive control (q^4 = 260,144,641 inside it) gives exactly 1 (S17) |
| P3 | the twin-free run count is on the independent-slot prediction in every (L', u) bin above depth 2, including the bins with Omega = 3 | **held, with one recorded deviation.** 14 of the 15 non-degenerate lengths within 1.9 sd; u >= 3 aggregate +0.61 sd over 409,251 runs; worst bin base 3, L' = 400, u = 2.3: 21 against 10.42, +3.29 sd (whole length +4.33 sd), which survives recalibration (+3.28) and decomposes into a single compensated band, with the variance ratio at 1.0016 and the other section at -1.8 sd in the same relative regime (S22) |
| P4 | the fuels satisfy P2 > n / p_k with 0 exceptions | **refuted, with the exception count exactly the Omega >= 3 count**: 37,908 of 37,908 Omega = 3 members on the brief's grid fail it, 3,981,605 of 3,981,605 Omega = 2 members satisfy it |
| P5 (prover's) | P4 is exactly the Omega = 2 statement; it fails for every Omega >= 3 member once nextprime(t')^2 >= p_k | **held, threshold verified in both directions**: 0 of 260,437 Omega = 3 members satisfy it at t' = 151 (q^2 = 24,649 > p_k = 16,139), while 3,482 of 442,176 do at t' = 109 (q^2 = 12,769 < p_k) and 528 of 405,107 at t' = 121 (q^2 = 16,129 < p_k) (S20) |
| P6 (prover's) | the prime share among core-free members follows one curve in u for both sections and every L' | **held**; and trivially so between sections, since at fixed t' the census is a property of the raw line and the two sections coincide on their overlap. Prior art: 1/(1 + ln(u-1)) on 2 <= u <= 3, measured 1-3% below it and converging upward as t' grows |
| P7 (prover's) | min K over twin-free stretches is 0 at short L' and positive at long | **held**: base 3 gives 0, 0, 0, 0, 0, 1, 1, 12 at L' = 50, 75, 100, 150, 200, 300, 400, 579, so the crossing sits between 200 and 300, matching S12's L0 = 278 on the same section |
| P8 (owner's reading) | confirmation of P1-P3 files the step at the core as ROOT in (L, u) | **held**; filed ROOT in (L, u), with the one deviation of P3 recorded rather than absorbed |

## 8. Dead ends

- **"The leftover members are P or P1 P2" as a general statement about a section.** Dead. It is
  exactly the statement that the section lies below nextprime(t')^3, which fails on 89% of the
  base-3 section as soon as the core is taken at L' = 50 and fails on the whole section for
  L' <= 20. Refuting instance: 28,934,443 = 307^3, a core-free member with Omega = 3 at t' = 301.
  What survives: the exact thresholds, S17, which are kernel-ready and generalise the two-prime
  lemma to every j.
- **"The fuels of a twin-free stretch are the section's own primes" as an extra structure.** Dead.
  Refuting instance: any Omega = 3 core-free member on a twin-free stretch at L' = 50, of which
  there are 35,024; and the statement is a theorem, not an observation, for Omega = 2. What
  survives: S20, and the separate measured fact that about 85% of these composite members do have
  their largest factor above the cut on base 3 at every L', a fact about position in the section
  (100% at the record, which sits at the top; 0% at base 7's record, which does not).
- **Depth as the parameter that would show the construction.** Dead as a lever. The census depends
  on depth alone to 2.7%, and the residual is the classical size and sign; the run counts above
  depth 3 are on the independent prediction to +0.6 sd over 409,251 events. Crossing the depth
  where the Liouville sign stops separating the types changes nothing that a count can see.
- **Local overdispersion of the twins as the mechanism behind the long stretches.** Dead. The
  variance of the twin count over stretches equals the independent-slot model's own variance to
  0.05-0.17% at every stretch length from 250 to 550 slots.
- **The by-K reading of a long stretch's twin-free starts.** Dead, and it is N16 again: at
  L' = 400 the 2,855 twin-free starts spread over K classes wildly (K = 13 holds 189 observed
  against 94 predicted, K = 15 holds 10 against 29) because a single twin gap contributes one start
  per position it covers and its K drifts along the gap. The run count is the only event count.

## 9. Remaining open items on this part

- **Closed here.** The Omega-census at depth u and its thresholds (S17). The fuel identity (S20).
  The depth-only form of the type census with its residual (S21). The run count against
  independence in the whole (L', u) region (S22).
- **Measurement with no structural content.** The 85% "fuel above the cut" fraction; the drift of
  the type share with 1/ln t'; the mean K as a function of (L', u).
- **The root question in disguise.** "At least one core-open slot of a stretch as long as the
  section's record is a twin", i.e. a lower bound on the count of PP among K independent leftovers.
  Everything measured in this branch says the count has no excess to exploit.
- **Genuinely open on this part alone.** The band [400, 450) on base 3: 66 twin gaps against 38.6
  predicted, with [300, 375) at 486 against 534.5. An attack is the same band decomposition on a
  third computed section, and on the base-3 section split into two halves, to see whether the bump
  is one region's or the whole section's; and the same decomposition with the core fixed at several
  t', to see whether the bump moves with the core (a property of the model) or stands still (a
  property of the twin gaps).
- **For the Formalist.** S17 is a few lines on top of CoreLeftover (the smallest B-rough number
  with Omega >= j is nextprime(B)^j; primeOrSemiprime_of_rough_lt_cube is its j = 3 case). S20 is
  two lines given S17 and c' = p_k^2. Both are stated in the part's intrinsic parameters (B, j) and
  so hold for every core of every section at once.
