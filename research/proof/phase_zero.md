# Phase zero in the manifold's own terms (R4.c.iv, prover, 2026-09-07)

Branch of R4.c (the valves), spawned by V12 (valve_existence.md): the one-tooth free-phase
adversary obeys every valve law on record and empties the pure charge for 60 turns at
(5, 10^4); the one property it lacks is phase zero, a gear strikes exactly its multiples.
The owner's direction (2026-09-07): the proof space is the valves' domain (Q, Q^2] with the
manifold's and the valves' laws on the raw line; no descent to the engine's coordinate. The
branch asks what phase zero is, as a property of the charge set on the raw line, by taking
every law on record and asking whether it survives when the phases are freed, and whether the
laws that do not survive are satisfied by an adversary defined multiplicatively (the parity
barrier's kind).

Sources read: valves_scratch.md, valves_reconcile.md, turn_ledger.md (V1, V2),
dead_branches_reopened_3.md (c), valve_existence.md (V10-V12), top_machine_lean.md (the
kernel ledger: L1-L8, L10, L12, L13, L17, L19, L22, L30/L31, L34/L35, L44, L45, L46, L67-L69,
the exhaust cap, the zone laws), top_machine_8.md (W95-W102), manifold_census_large.md
(W103). Scripts in research/valves/r3/, outputs in research/valves/r3/results/ (gitignored;
every number used is in this file). Laws numbered V13 onward.

Vocabulary (the owner's, canonical). ENGINE = the primes <= q. MANIFOLD = the primes in (q, Q]
as a machine on the raw line with teeth 0 and -2: gear g strikes the pair (n, n + 2) iff
g | n or g | n + 2, i.e. the domino {x, x + 2} of pair positions with x = -2 mod g; on
[1, Q^2] a number is open iff it has no prime factor in (q, Q] (smooth zone [1, Q]: open iff
q-smooth; quiet zone (Q, Q^2]: open iff s x P with s q-smooth and P = 1 or a prime above Q).
VALVES = the engine acting inside the manifold's open set: a CHARGE s x P with AIR s and FUEL
P; a charge pair (n, n + 2) has a FAMILY (s, s'); the PURE CHARGE (1, 1) = the twins; EMBER =
a q-smooth number above Q; PORT = the class mod 6; IMPRINT = the residues mod q# a family
occupies; ONSET = a family's first member and its TURN m = (mQ, (m + 1)Q]. EXHAUST = the
primes above Q. Forbidden here: window, rung, ladder, the column coordinate.

PHASE. A gear g at phase c strikes the NUMBERS n = c (mod g); on pairs that is the domino
{c - 2, c}. Phase zero is c = 0: the gear strikes its multiples. A FREE-PHASE manifold is the
same gear set with a phase vector (c_g) of its choice; the domino form is kept by
construction (one number class per gear is one pair domino, by the partner law L3).

## Pre-registered (written before any script of this branch was run)

### The theory

1. THE TRANSLATION LEMMA (to be proved, one line). On the manifold's full period W = prod g,
   the open set of a free-phase manifold with phase vector (c_g) is the translate by t of the
   real open set, t the CRT solution of t = c_g (mod g). Hence EVERY period-scale law that is
   invariant under translation (the domino form, no gap 4, the run and chain ceilings, the
   parity law, the loaded record rule, the census law, the mex closed form, the pair
   correlation product, the symmetry group up to conjugation by the translation) holds for
   every free-phase manifold with 0 exceptions, and has NO phase-zero content. Phase zero is
   not visible on the period; it is visible only in WHICH SLICE of the period the zone
   (Q, Q^2] shows. The real manifold shows the slice next to the origin.

2. WHAT THE SLICE AT THE ORIGIN HAS. On the slice next to the origin the open set is the
   integers coprime to the gear product, and that set is SATURATED: closed under
   multiplication by any q-smooth number (up to the Q^2 cut) and closed under division (every
   divisor of an open number is open; below Q that means q-smooth). Saturation is
   equivalent to phase zero for a sieve by a given gear set (the struck set is a union of
   ideals gZ). Every law on record with phase-zero content should be a consequence of
   saturation plus the cut (the gears are all the primes in (q, Q], so a number in (1, Q]
   that is open is q-smooth, and the exhaust cap): the quiet-zone rule in both directions,
   the family decomposition with the fuel a prime above Q, the onset and air cap (n = sP >
   sQ), the ember law, the count identity between burnt and pure charges as numbers, and
   the position of the record (W103). I expect no law to use the ORDER of the multiples
   (g, 2g, 3g, ...) beyond the size inequality P > Q; the order enters only as "the fuel is
   above Q", which is the smooth-zone law, itself saturation plus the cut.

3. THE PARITY ADVERSARY. The set A = {manifold-open n in (Q, Q^2] with an even number of
   prime factors} is multiplicatively closed (Liouville is completely multiplicative) but
   NOT saturated: 3P is in A and P is not (a prime has one factor). So the laws that use
   only "closed under multiplication" hold for A and the laws that use "closed under
   division" fail for A. I predict the divisor-closure laws (the rough part of an open
   number is open; the quiet-zone rule's "only if"; the count identity B = sum over air of
   the pure count) are exactly the phase-zero laws with content beyond multiplicativity,
   and that the size laws (onset, air cap, ember) hold for A because A is a subset of the
   real open set.

4. WHAT SATURATION DOES NOT GIVE. A saturated set on the zone whose open numbers in (1, Q]
   are the q-smooth ones is exactly {s x P : P in F} for a set F of PRIMES above Q (any set
   of primes), i.e. the phase-zero sieve by the gears (q, Q] together with the primes above
   Q not in F. The saturated adversary "add the twin members in (Q, Q^2] as gears" has
   every saturation law and no pure pair. What excludes it is the CUT (no gear above Q),
   which is the definition of the manifold. So the sharpest statement of the branch is
   expected to be: phase zero = saturation separates the real charge set from every
   free-phase and every parity-defined adversary; saturation + the cut is the real manifold;
   the existence consequence is the root. I record this as the predicted ROOT mark before
   computing, so that a finding to the contrary is visible.

### Predictions with numbers, and what would refute each

- Z-P1 (the translation lemma, exact). On the wheel of the gears {7, ..., 29} (q = 5,
  Q = 30, period 215,656,441) three random phase vectors give open sets that are exact
  translates of the real one: 0 mismatches over the full period; their records, gap
  censuses, longest runs and chains, and mex tables agree with the real wheel's exactly.
  Refuted by one mismatch.
- Z-P2 (period laws on the zone, phase-free). For the random-phase manifolds and the
  two-tooth adversary at (5, 10^3), (7, 10^3), (5, 10^4), (7, 10^4) on (Q, Q^2]: 0 gaps of
  4; longest run of consecutive open pairs <= q' - 3 and longest step-2 chain <= q' - 2
  (both attained somewhere in the zone for the random phases at Q = 10^4); the gap census
  of consecutive open pairs matches the period density (the census law's product, computed
  from L22 for d <= 14) within 3 sqrt(N) at every d. Refuted by a gap of 4, a run above
  the ceiling, or a census count off by more than 3 sqrt(N) under free phase.
- Z-P3 (the one-tooth adversary is not a sieve on the raw line). Its struck set is
  {s f : f = c_p (mod p)}, not a residue class of n; I predict it VIOLATES the run ceiling
  at (5, 10^3): a run of 7 or more consecutive open numbers (multiples of 7 stay open when
  gear 7 is used at a nonzero phase). Refuted by every run <= 6 there.
- Z-P4 (the congruence laws are phase-free). Imprint, port and inventory hold with 0
  exceptions for every model, because they are facts about n = s f with f coprime to q#.
- Z-P5 (the size laws are phase-zero via "fuel above Q"). Onset (no member of (s, s')
  before turn max(s, s')), the air cap (air <= m in turn m) and the ember law (every burnt
  pair in turns 1, 2 has an ember member) hold with 0 exceptions for the real set, the
  one-tooth adversary and the parity adversary, and FAIL for the random-phase manifolds and
  the two-tooth adversary, with violations already in turn 1 at every (q, Q): I predict
  more than 100 burnt pairs in turn 1 at (5, 10^4) under random phase against 3 real ones.
- Z-P6 (saturation). SAT-up (every s x f with f pure fuel and sf <= Q^2 is open) and
  SAT-down (the rough part of every open number is 1 or open) have 0 violations for the
  real set. Random phase: both fail at a rate near 1 - (open density) = 0.77 at (5, 10^4).
  Two-tooth adversary: both fail. One-tooth adversary: SAT-up and SAT-down hold by
  construction, and full saturation (every divisor below Q of an open number is q-smooth)
  fails at its composite fuel members (predicted thousands at (5, 10^4)). Parity adversary:
  SAT-up vacuous (no air-1 open number above Q), SAT-down fails at EVERY fuelled charge,
  closure under multiplication by smooth s fails for exactly the s with an odd number of
  prime factors.
- Z-P7 (the count identity as numbers). T - E = sum over smooth s < Q of (N_pure(Q^2/s))
  with N_pure(x) = the pure fuel in (Q, x], exact for the real set at all four (q, Q); it is
  the Legendre identity for the sieve by all primes <= Q read family by family, hence an
  identity (ROOT/FACT), and the pair-level system (T = sum of family counts) is not closed.
  For the other models the identity fails by the saturation violations.
- Z-P8 (the record and the density). The real zone record at (5, 10^4) is 420 after 26,261
  (turn 2, a twin gap). Random-phase records: 250-350, positioned uniformly in the zone
  (turn > 100). Per-turn open density (numbers): real 0.10 at turn 1 rising to the period
  density prod_{7 <= g <= Q} (1 - 1/g) = 0.23 by turn 100; random phase flat at 0.23 +- 0.01
  in every turn. Phase zero acts as a density DEFICIT at the bottom of the zone, which is
  where the record sits (W103's mechanism in density terms).
- Z-P9 (the record rule on the slice). The loaded record rule's condition at L = the zone
  record holds with enormous slack (the tail alone, 2 cells per gear, covers L when
  2 floor(L/4) + min(L mod 4, 2) <= tail(L)); the period record lies above 2,000 at Q = 10^3
  (165 gears) and above 2,300 at Q = 10^4 by the tail-only lower bound, against zone records
  near 150 and 420. The rule says nothing about the slice.

Owner's predictions on the scorecard (from the brief): (O1) the real manifold's
distinguishing property is phase zero, "a gear strikes exactly its multiples" (I predict:
yes, in the form saturation = divisor-closure + smooth-closure, and it is exactly what the
parity adversary lacks); (O2) some phase-zero laws use only multiplicativity and some use the
order of the multiples (I predict: the split is semigroup versus saturated, not
multiplicative versus ordered; nothing uses the order beyond "fuel above Q"); (O3) a property
X with content beyond multiplicativity exists (I predict: X = saturation, count = the number
of fuelled charges, every one of which the parity adversary fails; and X alone does not force
a pure pair, the twin-gear adversary, so X + cut is the root).

### Scorecard (filled after the runs)

| prediction | verdict | where |
|---|---|---|
| Z-P1 translation lemma | EXACT: 0 mismatches in 15 trials on 5 wheels (periods 20,677 to 215,656,441); record, open-pair count, census, runs, chains, gap 4, mex table and conjugated symmetry group identical | V13, table 1 |
| Z-P2 period laws phase-free on the zone | HELD: 0 gaps of 4 in every run; run and chain ceilings met and attained by the random phases and the two-tooth adversary at all four (q, Q); census within 2.5 sqrt(N) at every d <= 14 (56 cells over 4 random runs at Q = 10^4, max abs z = 2.5); the two-tooth adversary within 3.2 sqrt(N), its one -3.2 (d = 2 at (5, 10^4)) being the deficit of the 60 turns it empties (band ratios 0.74 in turns 1-2, 0.89 in 3-10, 0.998 above turn 1000) | table 1, section 4 |
| Z-P3 the one-tooth adversary breaks the run ceiling | HELD: runs of 10 and 13 open numbers at (5, 10^3) and (7, 10^3) against ceilings 6 and 10; 11 and 12 at Q = 10^4 | table 1 |
| Z-P4 imprint, port, inventory phase-free | HELD: 0 exceptions in every model (36 runs) | table 1 |
| Z-P5 size laws phase-zero via fuel above Q | HELD: onset, air cap and ember law have 0 violations for the real set, the one-tooth adversary, the parity adversary and the twin-gear sieve, and fail for the random phases and the two-tooth adversary already in turn 1 (random phase at (5, 10^4): 446 burnt pairs in turn 1 against 3 real, 908 of 917 turn-1-2 burnt pairs without an ember member) | table 1 |
| Z-P6 saturation | HELD in every clause: real 0 violations of 18,140,409 + 18,139,479 + 71,470,323 tests at (5, 10^4); random phase fails 76.5 % of each (predicted 0.77); the one-tooth adversary passes the smooth clauses by construction and has 16,839,460 open numbers with a composite rough part; the parity adversary fails SAT-down at 10,804,164 of 10,804,629 (all but the 465 embers) and module closure at exactly the odd-Omega air (19,312,952 of 19,312,952; 0 of 12,811,341 even-Omega tests) | table 1, section 2 |
| Z-P7 count identity as numbers | HELD: difference 0 at all four real (q, Q) and for the two saturated models (one-tooth, twin sieve); off by 139,093 (random), 236,096 (two-tooth), 10,804,164 (parity) at (5, 10^4); and it is Legendre's identity, ROOT/FACT; the pair system is not closed | section 3 |
| Z-P8 record and density | HELD, with the surplus half unforeseen: random records 314, 322 (q = 5), 265, 233 (q = 7) at turns 4242, 8050, 7135, 9283 (predicted 250-350, uniform); real 420 in turn 2; real density 0.107 in turn 1 (predicted 0.10) but it OVERSHOOTS the period value in the middle turns (0.291 at turn 60 against 0.228) and returns to it at the top (0.226 at turn 9999); random flat 0.225-0.234. The prediction "rising to the period density by turn 100" was wrong: the slice is a Buchstab profile, poor at the bottom, rich in the middle | section 4 |
| Z-P9 the record rule on the slice | HELD: tail-only lower bound on the period record 233 (Q = 10^3) and 1,880 (Q = 10^4) against zone records 120 and 420 / 372; at L = the zone record the core alone at a greedy phase covers the whole window (cost 0 against tail 1,147) | section 4 |
| Owner O1 phase zero is the distinguishing property | HELD in the form saturation: the one property that separates the real charge set from every counterfactual on record, with the counts above | section 2, verdict |
| Owner O2 multiplicativity versus the order of multiples | The split is elsewhere: semigroup versus saturated. Nothing on record uses the order beyond "fuel above Q", and the spacing of a gear's strikes is the same under free phase; what phase zero adds is that the struck set of g is g Z, an ideal | section 2 |
| Owner O3 a property X beyond multiplicativity | HELD for X = saturation (divisor closure), and X alone does not force a pure pair: the twin-gear sieve is saturated with 0 pure pairs and 880,214 extra gears; X + the cut is the manifold's definition: ROOT | verdict |

## Setup

Scripts (research/valves/r3/): phase_zero.py q Q model (the law table for one model on (Q, Q^2];
models real, rand [seed], adv2, adv1, parity, twinsieve; results/pz_<model>_q<q>_Q<Q>.json),
wheel_period.py (full periods of five small wheels: the translation lemma, records, census, mex,
symmetry group, position laws, parity law), record_rule.py q Q L (the period-scale record bounds
against the zone record). Runs: every model at (q, Q) = (5, 10^3), (7, 10^3), (5, 10^4), (7, 10^4);
random phases with two seeds each; wall time 10-30 s per run at Q = 10^4 on one core, memory under
1.5 GB.

The six models on the same gear set G = the primes in (q, Q]:

- REAL: phase zero; n is struck by g iff g | n.
- RAND: every gear at an independent uniform random phase c_g; n struck iff n = c_g (mod g).
  The domino {c_g - 2, c_g} on pairs is kept. This is the free-phase manifold proper.
- ADV2 (the two-tooth free-phase adversary): the phases c_g chosen greedily so that the dominoes
  cover every pure-imprint pair (n, n + 2 both coprime to q#) in the first 60 turns; unused gears
  at phase zero; the sieve acts on n itself over the whole zone. 776 of 1,226 gears used at
  (5, 10^4), 774 at nonzero phase; 165 of 165 at (5, 10^3) with 19 pairs surviving.
- ADV1 (the one-tooth adversary of valve_existence.md): the same phases, applied to the FUEL:
  n = s f is open iff f = 1 or f > Q, f coprime to q#, and f avoids every class c_g. ADV1 and ADV2
  agree on every number with air 1 (so their pure charges coincide, both empty for 60 turns) and
  differ on the burnt charges: ADV1 decides s f by f, ADV2 by s f.
- PARITY: the real open set restricted to the numbers with an even number of prime factors
  (Liouville +1); the charges kept are s x P with Omega(s) odd and the embers with Omega(s) even;
  no prime is in it, so its pure charge is empty in every turn.
- TWINSIEVE: phase zero with the gear set enlarged by every prime P in (Q, Q^2] with P - 2 or P + 2
  prime (880,214 extra gears at Q = 10^4, 16,268 at 10^3); its open set is the real one minus the
  charges whose fuel is a twin member; a saturated set with no pure pair.

For each model the script lists the open numbers of (Q, Q^2], strips the q-smooth part of each
(air s, rough part f, Omega(s)), forms the open pairs (n, n + 2), and tests every law below; the
period densities of the gap census come from L22 (kernel `gap_census`), evaluated for d <= 14 by
the 2^(d - 1) alternating sum with the small gears' offset collisions exact and the gears above
d + 2 in the universal product.

## The law table

### V13 (the translation lemma; PROVED, EXACT).

Let G be pairwise coprime gears, W = prod G, and (c_g) any phase vector. Gear g at phase c_g strikes
the number n iff n = c_g (mod g); at phase 0 iff n = 0 (mod g). Take t with t = c_g (mod g) for
every g (CRT). Then n is struck at phase (c_g) iff n - t is struck at phase 0, for every gear; so the
free-phase open set is the real open set translated by t, on numbers and on pairs. QED.
Consequently every statement about the open set that is invariant under translation holds for
every free-phase manifold exactly when it holds for the real one; the only statements with
phase-zero content are those that name a POSITION, and on the quiet zone (Q, Q^2] that means:
which slice of the period the zone shows. Measured: 15 of 15 free-phase wheels are exact
translates (0 mismatches over periods 20,677; 1,363,783; 7,436,429; 95,041,567; 215,656,441).

### Table 1. Every law on record, under free phase and under the adversaries

Numbers at (5, 10^4) unless stated; (7, 10^4) and the Q = 10^3 runs agree in kind, with the
differences noted. "Period" = tested on full periods (wheel_period.py); "zone" = tested on
(Q, Q^2]. Verdicts: PHASE-FREE (no phase-zero content), POSITION (phase-zero content only in
where, not in what), PHASE-ZERO (fails under free phase), and for the phase-zero laws whether the
PARITY adversary has it.

| law | real | RAND (two-tooth free phase) | ADV2 (two-tooth adversary) | ADV1 (one-tooth adversary) | PARITY | verdict |
|---|---|---|---|---|---|---|
| domino form (L3): one number class per gear = one pair domino | by definition | by construction | by construction | not a sieve on the raw line: its struck set is {s f : f = c_p mod p}, no per-gear residue class | not a sieve (a subset of the real set) | PHASE-FREE (a property of the sieve's shape) |
| no gap 4 (L4) | 0 | 0, 0 | 0 | 0 | 0 | PHASE-FREE; it is a tautology of pairs for ANY set of numbers (n, n + 4 open pairs make n + 2 one) |
| run ceiling q' - 3 open pairs (q' - 1 open numbers, L10) | 6 = 6; 10 = 10 at q = 7 | 6, 6; 10, 10 | 6; 10 | 11 (q = 5), 12 (q = 7); 10 and 13 at Q = 10^3: VIOLATED | 5; 8 | PHASE-FREE for a sieve (q' consecutive numbers meet every class); ADV1 is not a sieve |
| chain ceiling q' - 2 (L10) | 6 = 6; 10 = 10 | 6; 10 | 6; 10 | 11; 12: VIOLATED | 6; 8 | PHASE-FREE, as above |
| parity law F = 2m - (m mod 2) (L17), free wheels m = 3, 4, 5 | 5, 8, 9 | 5, 8, 9 (3 trials each) | period | - | - | PHASE-FREE (translate) |
| loaded record rule (L69) and the period record | records 20, 26 on the loaded wheels | 20, 26 in 6 trials | period | - | - | PHASE-FREE (translate) |
| census law (L22 / W98) | exact on 5 full periods (formula = count at d = 1..12, 60 cells); on the zone +50 to +150 sqrt(N) surplus (see section 4) | zone within 2.5 sqrt(N) at every d <= 14 (4 runs, 56 cells) | within 3.2 sqrt(N) | -101 to +176 sqrt(N) | -290 to -1,106 sqrt(N) | PHASE-FREE on the period; on the zone the REAL slice is not a typical slice (POSITION content, section 4) |
| mex closed form (L30/L31) | lower half 20,000 of 20,000; upper half 20,000 of 20,000 on free wheels, 19,650-19,768 of 20,000 on loaded wheels | identical with the residues shifted by the phases (same samples) | period | - | - | PHASE-FREE (translate); the loaded-regime failures are the regime, not the phase |
| wheel count prod (g - 2) and pair correlation product (L5, L44) | 71,569,575 on {7..29} | 71,569,575 | period | - | - | PHASE-FREE |
| symmetry group (Z/2)^m (L8) | 128 of 128 maps n -> c(n + 1) - 1 preserve the set | the 128 conjugates n -> c(n - t + 1) - 1 + t preserve it, 128 of 128 | period | - | - | PHASE-FREE up to conjugation; the mirror's fixed point moves from -1 to t - 1 |
| position laws (L6: shield -1 open, antipodes 2 and -4 open) | open on every wheel | -1 open in 9 of 15 trials, 2 open in 8 of 15; the translates t - 1, t + 2 open in 15 of 15 | period | - | - | POSITION: phase zero says WHERE the shield is |
| smooth zone (L46): open in [1, Q] iff q-smooth | 175 = 175 (338 at q = 7) | 2,290 open below Q, 2,247 not smooth, 132 smooth ones struck | 2,344 / 2,300 / 131 | defined as the real one | as the real one | PHASE-ZERO |
| quiet zone, "only if" (open => s x P, P = 1 or a prime above Q) | 0 of 23,900,635 | 139,797 rough parts in (1, Q] and 17,264,698 composite rough parts above Q, of 22,830,942 open | 127,671 and 16,914,520 | 0 in (1, Q] by fiat; 16,839,460 composite rough parts of 22,581,678: VIOLATED | 0 (subset) | PHASE-ZERO; ADV1 fails it too; PARITY has it |
| quiet zone, "if" = SAT-up (every s x P with P pure fuel and sP <= Q^2 is open) | 0 of 18,139,479 | 12,705,004 of 16,603,075 (76.5 %) | 12,357,784 of 16,480,454 | 0 (construction) | 0 of 0: vacuous, no air-1 open number above Q | PHASE-ZERO; ADV1 has it by construction; PARITY vacuous |
| SAT-down (the rough part of an open number is 1 or open) | 0 of 18,140,409 | 12,810,573 of 16,742,360 (76.5 %) | 12,560,062 of 16,716,765 | 0 (construction) | 10,804,164 of 10,804,629: every open number but the 465 embers | PHASE-ZERO and BEYOND MULTIPLICATIVITY: PARITY fails it |
| module closure (open n, smooth s, s n <= Q^2 => s n open) | 0 of 71,470,323 | 47,632,812 of 62,250,814 (76.5 %) | 46,527,961 of 62,056,796 | 0 of 58,167,315 | 19,312,952 of 32,124,293, exactly the tests with odd-Omega air (19,312,952 of 19,312,952) and 0 of the 12,811,341 with even-Omega air | PHASE-ZERO; PARITY fails it for odd air only: it is a module over the even-Omega smooth numbers |
| family decomposition, imprint, port, inventory | 0 exceptions | 0 (5,376,052 fuelled pairs real; 4,894,893 RAND) | 0 | 0 | 0 | PHASE-FREE: congruence facts of n = s f with f coprime to q# (V2's proof) |
| onset: no member of (s, s') before turn max(s, s') | 0 of 5,376,052 | 58,675 of 4,894,893 (443 in turn 1, 465 in turn 2) | 54,442 | 0 | 0 | PHASE-ZERO via "fuel above Q"; ADV1 and PARITY have it |
| air cap: a fuelled member of turn m has air <= m | 0 of 23,899,705 | 139,797 | 127,671 | 0 | 0 | PHASE-ZERO, as onset |
| ember law: every burnt pair in turns 1, 2 has an ember member | 15 burnt pairs, 0 violations (3 in turn 1, 12 in turn 2; embers 37 and 21) | 917 burnt pairs, 908 violations (446 burnt in turn 1) | 881 / 874 | 10 / 0 | 7 / 0 | PHASE-ZERO, as onset |
| count identity as numbers, T - E = sum over air s of N_pure(Q^2/s) | 0 (23,899,705 = 23,899,705; also 0 at the other three (q, Q)) | off by 139,093 and 138,049 | off by 236,096 | 0 (construction) | off by 10,804,164 (the right side is 0) | PHASE-ZERO and BEYOND MULTIPLICATIVITY; it is saturation counted (section 3) |
| pure charge = the twins (count identity for pairs) | 440,107 = 440,107 as sets, 0 symmetric difference | pure (air-1) pairs 489,728 / 489,610: not twins | 489,716, none in turns 1-60 | 489,716, none in turns 1-60 | 0 | definitional once the fuel is the primes; the counts say the twins are 0.90 of a random slice's pure pairs (0.94 at Q = 10^3) |
| W103: the record is a twin gap in the bottom turns | 420 after 26,261 (turn 2), a twin gap; 372 after 18,539 (turn 1) at q = 7 | 314 (turn 4,242), 322 (turn 8,050); 265, 233 at q = 7 (turns 7,135, 9,283); not twin gaps | 303 (turn 9,941); 218 (turn 7,682) | 7,288 after 10,935 (turn 1); 3,302 after 12,005 | 5,762 after 31,102 (turn 3); 3,586 | PHASE-ZERO (POSITION): the real record sits in the poor bottom of the slice (section 4) |
| balance of the pure fuel on the engine's teeth (V12) | gear 3: 2,879,906-2,880,320 | 3,044,232-3,044,350 | 3,050,099-3,050,194 | same as ADV2 | 0 (no fuel) | a count; every fuel-set model has it |
| mirror closure of the family set (V12) | 1,451 of 1,510 | 2,621 of 3,030 | 2,625 of 3,008 | 1,363 of 1,434 | 380 of 390 | a set property every model has to the same degree |

The TWINSIEVE column, for the record (5, 10^4): domino, gap 4 (0), run 6, chain 6, smooth zone
175 = 175, quiet only-if 0, SAT-up 0 of 14,895,515, SAT-down 0 of 14,896,445, module closure 0 of
57,300,895, imprint / port / inventory 0, onset 0 of 3,449,447, air cap 0, ember law 0, count
identity difference 0, pure pairs 0 in all 9,999 turns, record 1,458 after 18,223 (turn 1); at
(7, 10^4) record 2,916 after 15,307. It passes every row of the table that the real set passes,
and it has no pure charge.

### The split

Counting the rows: 13 laws are PHASE-FREE (domino form, no gap 4, run ceiling, chain ceiling,
parity law, loaded record rule, census law on the period, mex form, wheel count and correlation
product, symmetry group, imprint, port, inventory; the last three by V2's proof, the first ten by
V13). One row is POSITION (the shield and antipodes: phase zero says where, the translate has
them too). Ten rows are PHASE-ZERO (smooth zone, quiet zone only-if, SAT-up, SAT-down, module
closure, onset, air cap, ember law, the count identity as numbers, W103 and the slice profile).
Of the ten, the PARITY adversary has six (quiet only-if, SAT-up vacuously, onset, air cap, ember,
and module closure over even-Omega air) and fails three (SAT-down, module closure over odd-Omega
air, the count identity) plus the record's position. So THREE laws have content beyond
multiplicativity, and they are one property counted three ways: SATURATION.

## What phase zero gives, exactly

### V14 (phase zero = saturation; PROVED).

For a sieve by pairwise coprime gears G >= 3 on the integers (gear g strikes the class
c_g + g Z), the following are equivalent: (a) every phase is zero; (b) the struck set is closed
under multiplication by every integer (a union of ideals g Z); (c) the open set is SATURATED:
closed under taking divisors, and closed under multiplication by every integer coprime to G.
Proof. (a) => (b): each class is g Z. (b) => (a): if c_g != 0, choose n = c_g (mod g) and, by CRT,
n avoiding the one class c_h g^(-1) (mod h) for every other gear h; then n is struck (by g) and
g n is struck by no gear (g n = 0 != c_g mod g, and g n != c_h mod h), contradicting (b).
(a) => (c): the open set is the integers coprime to prod G, which has both closures. (c) => (a):
if c_g != 0, take an open n with n a unit mod g (CRT: n = 1 mod g and n outside every other
gear's class) and s = c_g n^(-1) (mod g), s = 1 (mod h) for h != g; s is coprime to G and s n is
struck by g, contradicting closure under multiplication. QED. On the finite range [1, Q^2] the
closures are the tested clauses (a divisor of an open number is open; s n open whenever
s n <= Q^2), and the tests in table 1 are exactly these.

The mechanism behind each phase-zero law of table 1, read from the proofs on record:

- smooth zone (L46), the quiet-zone rule (kernel `quiet_zone`), the exhaust cap: their proofs
  use exactly (b): n = p m with p a gear forces n struck (`exhaust_home_or_echo`: n = p m, m >= 2,
  m < Q, so m has a factor at or below the cut). That is the ideal property, not the order of
  the multiples. The first strike of g above Q is at g ceil((Q + 1)/g) <= Q + g under phase zero
  and at the first n > Q with n = c_g (mod g), also <= Q + g, under free phase: the SPACING of a
  gear's strikes is phase-free; what phase zero adds is that the strikes are the multiples, so
  the strikes on the charges are at s x P with the factor structure (a burnt charge is struck by
  the gears dividing its air, and by nothing else). Under RAND 17,264,698 of 22,830,942 open
  numbers have a composite rough part and 139,797 have a rough part in (1, Q]: the charge
  decomposition n = s x P with P prime above Q is a saturation fact.
- onset, air cap, ember law: their proofs use only "fuel above Q" (V2), and "fuel above Q" is
  the smooth-zone law: an open number below Q is smooth. So they are saturation + cut too, but
  through the weakest clause, and any model that puts its fuel above Q by fiat (ADV1, V2's F) or
  takes a subset of the real fuel (PARITY, TWINSIEVE) has them. They are the phase-zero laws
  with the LEAST content: they hold for every adversary on record except the raw-line
  free-phase ones.
- SAT-down, module closure, the count identity: the closure clauses of (c) themselves. These
  are what the parity adversary lacks, and the mechanism is visible in the counts: A = {even
  Omega} is closed under multiplication by even-Omega air (0 of 12,811,341 violations) and by
  nothing else (19,312,952 of 19,312,952 with odd-Omega air fail), and it is not divisor-closed
  at any fuelled charge (10,804,164 of 10,804,629). The parity barrier's set is a sub-SEMIGROUP
  of the open set that is not a sub-SATURATED set. "Multiplicatively defined" gives the
  semigroup; the ideal property gives the saturation; the two are different, and the phase-zero
  laws use the second.

So the answer to the brief's sharper question: the phase-zero laws satisfied by a
multiplicatively-defined adversary are the size laws (onset, air cap, ember) and the
one-directional quiet-zone rule; the phase-zero laws NOT satisfied by it are the divisor-closure
laws, three rows, one property. No law on record uses the order of the multiples beyond the
inequality P > Q.

The Liouville-weighted count, for the record: sum over the real charge pairs of
lambda(n) lambda(n + 2) is -6,635 at (5, 10^4) against 5,376,501 pairs and 440,107 pure (-5,229 at
(7, 10^4); -282 and -82 at Q = 10^3): the signed census cancels to 0.1 % while the pure charge is
8.2 % of the pairs; the sign-weighted count does not see the pure charge. That is the parity
barrier in one number, noted and not pursued (prior art).

## The charge set as a structure

(a) Generators and closure. On (Q, Q^2] the open numbers are the charges s x P and the embers
s; the generators are the q-smooth numbers (air) and the primes above Q (fuel); the closure law
is module closure: a charge times a smooth number that keeps it below Q^2 is a charge (0
violations of 71,470,323 at (5, 10^4); 101,002,981 at (7, 10^4)), and the fuel is the set of
irreducibles vacuously (a product of two fuels exceeds Q^2). Every model except the saturated
ones (REAL, ADV1, TWINSIEVE) breaks the closure at 75-77 % of the tests; the parity adversary
breaks it at exactly the odd-Omega multipliers.

(b) Burn as a map. On NUMBERS the burnt charges are exactly the image of the pure charges under
multiplication by air: burn(s, P) = s P is a bijection from {(s, P) : s >= 2 smooth, P pure,
s P <= Q^2} onto the burnt fuelled charges (unique factorisation; 0 SAT-up and 0 SAT-down
violations). Hence the count identity

    T - E = sum over smooth s < Q of N_pure(Q^2 / s),      N_pure(x) = pure fuel in (Q, x],

exact at all four real (q, Q) (23,899,705; 28,331,388; 325,572; 380,893) and at the two other
saturated models, and false for every non-saturated one by the saturation deficit. Written out:
the pure charges determine the burnt ones (the sum), the burnt ones plus the total determine the
pure ones (N_pure(Q^2) = T - E - sum over s >= 2), and the total T is the manifold's census on
the quiet zone. Is that a closed system? It is an IDENTITY: T is the count of integers in
(Q, Q^2] with no prime factor in (q, Q], which by the quiet-zone rule is sum over smooth s of
#{f <= Q^2/s : f = 1 or f prime > Q}, and the identity is that sum rearranged; Legendre's formula
for T (the alternating sum over the 2^1226 squarefree products of gears) is the same count from
the other side. Solving it for N_pure(Q^2) needs N_pure at Q^2/s for every smooth s >= 2 and T,
and computing T is computing the prime count (it is the Meissel-Lehmer recursion; prior art,
one line). ROOT / FACT: the number-level system is closed and empty of content. On PAIRS there
is no burn map: a burnt pair (s P, s' P') is not the image of a pure pair (P, P + 2) under
anything, and the pair identity T_pairs = sum over families N(s, s') (V1) has the burnt families
as independent prime-pair counts, not determined by the pure one. The pair system is not closed;
V1 already said so, and saturation adds nothing to it. That is the exact place where "phase
zero" stops being a structure on the charges: it is a structure on numbers, and the pure charge
is a property of pairs.

(c) What saturation does force. A saturated open set on the zone whose open numbers below Q are
the q-smooth ones is exactly {s x P : P in F} for a set F of PRIMES above Q, i.e. the phase-zero
sieve by the gears (q, Q] together with the primes above Q not in F (every gear above Q strikes
only itself and its smooth multiples inside (Q, Q^2], by the exhaust cap). Saturation therefore
forces: the fuel is a set of primes above Q, and the pure fuel is non-empty whenever a burnt
fuelled charge exists. It forces nothing about pairs: TWINSIEVE (F = the primes above Q that are
not twin members) is saturated, passes every row of table 1 that the real set passes, and has 0
pure pairs in 9,999 turns at Q = 10^4, with 880,214 gears above Q. What excludes it is the CUT:
the manifold's gears are the primes in (q, Q] and nothing above Q strikes below Q^2 except at
its own multiples of smooth numbers (the exhaust cap). Saturation + the cut is the definition of
the real manifold, so the statement "the minimal saturated fuel has a pure pair in every zone" is
the root.

## The record laws on the charges

The loaded record rule (L69) and the census law (L22 / W98) are period-scale statements; the
quiet zone is a slice of length Q^2 - Q of a period W = prod G of about 10^4300 at Q = 10^4. By
V13 a free-phase manifold shows a different slice of the same period, and the measurements say
what a slice sees.

### The record rule on the slice (record_rule.py)

At L = the zone record the rule asks whether some phase of the core (gears <= L + 1) leaves an
uncovered set of domino cost at most the tail count (gears > L + 1):

| (q, Q) | zone record L | core gears | tail | boundary cost 2 floor(L/4) + min(L mod 4, 2) | slack | core alone at phase zero from the origin: uncovered / cost | core alone at a greedy phase: uncovered / cost | tail-only lower bound on the period record |
|---|---|---|---|---|---|---|---|---|
| (5, 10^3) | 120 | 27 | 138 | 60 | 78 | 12 / 8 | 0 / 0 | 233 |
| (7, 10^3) | 120 | 26 | 138 | 60 | 78 | 22 / 14 | 0 / 0 | 233 |
| (5, 10^4) | 420 | 79 | 1,147 | 210 | 937 | 13 / 9 | 0 / 0 | 1,880 |
| (7, 10^4) | 372 | 70 | 1,155 | 186 | 969 | 26 / 18 | 0 / 0 | 1,880 |

The rule certifies the zone record's length coverable with the core contributing nothing (cost
<= L/2 + 1 against a tail of 1,147), and the core alone at a greedy phase covers the whole window
(uncovered 0). The tail-only sufficiency (the boundary cost of the full window at most the tail,
which is a lower bound on the period record since coverability is downward closed) gives a
period record of at least 233 at Q = 10^3 and 1,880 at Q = 10^4, 1.9x and 4.5x the zone records;
the capacity upper bound (L70 / W102) is vacuous from L = 16 (the core's capacity
sum 2 ceil(L/g) exceeds L: the loaded regime). The period record is unknown and far above the
slice's; the rule is a period statement and says nothing about which slice is shown. The
free-core record the brief asks for is 0 at L = 420: the FREE core covers everything, and the
real core at phase zero from the origin leaves 13 cells (cost 9); the difference between the two
is exactly the position content of phase zero at that length, and it is far inside the slack.

### The census on the slice, by gap length and by turn (V15)

Period densities from L22, zone counts measured; ratio measured / expected by turn band
(Q = 10^4; the random-phase manifold is within 2.5 sqrt(N) of 1.000 at every d overall and within
its noise in every band):

| d | REAL q = 5: overall | turns 1-2 | 3-10 | 11-100 | 101-1000 | 1001-9999 | REAL q = 7 overall | 1-2 | 11-100 | 1001-9999 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.208 | 0.058 | 0.548 | 2.379 | 1.915 | 1.127 | 1.283 | 0.050 | 1.948 | 1.211 |
| 2 | 1.155 | 0.005 | 1.037 | 1.884 | 1.616 | 1.102 | 1.202 | 0.022 | 1.643 | 1.157 |
| 3 | 1.191 | 0.041 | 0.976 | 2.113 | 1.803 | 1.121 | 1.240 | 0.021 | 1.790 | 1.184 |
| 5 | 1.188 | 0.000 | 0.740 | 2.190 | 1.781 | 1.119 | 1.240 | 0.000 | 1.826 | 1.185 |
| 6 | 1.204 | 0.372 | 1.106 | 2.162 | 1.827 | 1.132 | 1.251 | 0.184 | 1.804 | 1.195 |
| 8 | 1.179 | 0.000 | 1.119 | 1.992 | 1.690 | 1.121 | 1.211 | 0.000 | 1.670 | 1.167 |
| 10 | 1.180 | 0.000 | 1.482 | 1.920 | 1.678 | 1.123 | 1.208 | 0.027 | 1.622 | 1.166 |
| 12 | 1.166 | 0.747 | 1.027 | 1.910 | 1.617 | 1.114 | 1.180 | 0.496 | 1.589 | 1.146 |
| 14 | 1.143 | 0.037 | 1.068 | 1.742 | 1.529 | 1.099 | 1.152 | 0.035 | 1.461 | 1.125 |

The per-turn open-number density, real against random phase (q = 5; period density 0.2283):

| turn | 1 | 2 | 3 | 5 | 9 | 15 | 30 | 60 | 100 | 300 | 1000 | 3000 | 9999 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL numbers | 0.107 | 0.153 | 0.184 | 0.222 | 0.254 | 0.268 | 0.286 | 0.291 | 0.289 | 0.280 | 0.255 | 0.242 | 0.226 |
| REAL pairs per turn | 140 | 137 | 328 | 445 | 597 | 667 | 772 | 817 | 787 | 735 | 619 | 550 | 461 |
| REAL pure per turn | 137 | 125 | 124 | 106 | 108 | 100 | 89 | 79 | 84 | 64 | 55 | 33 | 36 |
| RAND numbers | 0.226 | 0.231 | 0.226 | 0.231 | 0.233 | 0.229 | 0.227 | 0.229 | 0.231 | 0.232 | 0.226 | 0.225 | 0.231 |
| RAND pairs per turn | 490 | 513 | 477 | 513 | 493 | 507 | 481 | 468 | 498 | 493 | 449 | 472 | 524 |
| ADV1 numbers | 0.065 | 0.092 | 0.112 | 0.139 | 0.155 | 0.174 | 0.187 | 0.194 | 0.208 | 0.218 | 0.222 | 0.225 | 0.230 |
| PARITY numbers | 0.002 | 0.054 | 0.087 | 0.103 | 0.112 | 0.116 | 0.127 | 0.130 | 0.129 | 0.128 | 0.115 | 0.109 | 0.101 |
| TWINSIEVE numbers | 0.080 | 0.115 | 0.136 | 0.168 | 0.189 | 0.205 | 0.219 | 0.228 | 0.229 | 0.224 | 0.206 | 0.201 | 0.189 |

At q = 7 (period 0.2664): REAL 0.112, 0.156, 0.187, 0.224, 0.269, 0.291, 0.316, 0.327, 0.331,
0.326, 0.302, 0.286, 0.268 at the same turns; RAND flat at 0.258-0.274.

**V15 (the slice profile; MEASURED, mechanism stated).** A random slice of the period is flat at
the period density in every turn and in every gap length (RAND: 56 census cells within
2.5 sqrt(N), per-turn density within 0.005 of 0.2283). The real slice is not typical: it is POOR
at the bottom (turn 1 at 0.107 = 47 % of the period density; the small gaps d <= 3 at 0.5-6 % of
their period count in turns 1-2; gap 5, 8, 9, 10 absent there), RICH in the middle (turn 60 at
0.291 = 127 %; pairs per turn 817 against 468, 175 %; every gap length at 1.7-2.4x its period
count in turns 11-100), and back at the period value at the top (turn 9999: 0.226; gaps at
1.10-1.13x). Over the whole zone the real slice holds 4.7 % more open numbers and 9.9 % more open
pairs than the period average (0.239 / 0.2283, 0.05377 / 0.04894; at q = 7 6.4 % and 13.3 %;
at Q = 10^3 7.5 % and 15.7 %). Mechanism, from the laws on record: in turn m the open numbers are
the charges s x P with s <= m (the air cap), so the bottom holds only the primes and the embers
(density 1/log(1.5 Q) = 0.104 at turn 1, measured 0.107) and the families open in air order
(onset); above the onsets the rough part is a single prime at scale x = Q^2/s with
1 < log x / log Q < 2, where the prime density 1/log x exceeds the period's rough density
e^(-gamma)/log Q by the factor 1/(u e^(-gamma)) with u = log x / log Q (1.78 at u = 1, 0.89 at
u = 2); the slice returns to the period value only where u = 2, at the top. Prior art, one line:
this is Buchstab's function omega(u) = 1/u on 1 <= u <= 2 against Mertens' e^(-gamma), read on
the slice; not pursued. Where phase zero acts on the census is therefore the whole slice, and in
two opposite directions: it EMPTIES the bottom (the families are closed) and it OVERFILLS the
middle (single primes are denser than period-rough numbers). The record sits in the emptied
bottom (turns 1-2 at Q = 10^4, turn 1 at Q = 10^3): W103 in density terms. The one-tooth
adversary and the twin sieve show the same profile shape with the bottom lower still (0.065,
0.080 at turn 1) because they remove fuel from the bottom, and the parity adversary is a
profile of the burnt families alone (0.002 at turn 1: no primes, seven ember pairs).

The pure charge per turn, for the record: 137, 125, 124, 106, 108, 100, 89, 79, 84, 64, 55, 33,
36 (REAL) against 44-63 pairs of air-1 open numbers per turn under RAND: the real bottom has
three times the random slice's pure pairs (twin density 2 C_2 / log^2(1.5 Q) = 0.0137 against
the period's (1/q#) prod (p - 2) prod (1 - 2/g) = 0.0049), and the top has fewer (36 against 47).
Over the zone the twins are 0.90 of the random slice's pure pairs. The pure charge's excess at
the bottom is the same Buchstab factor squared, and it is where the record is: the real slice is
poor in charges and rich in pure ones at the bottom, both because the burnt families have not
opened.

## Laws

- **V13 (the translation lemma; PROVED, EXACT).** A free-phase manifold on its period is the real
  manifold translated by the CRT solution of t = c_g (mod g); every translation-invariant law is
  phase-free. 15 of 15 wheels exact translates; records, censuses, runs, chains, mex tables and
  conjugated symmetry groups identical.
- **V14 (phase zero = saturation; PROVED).** For a sieve by G: all phases zero <=> the struck set
  is a union of ideals <=> the open set is closed under divisors and under multiplication by
  numbers coprime to G. The phase-zero laws on record are consequences of saturation plus the
  cut; three of them (SAT-down, module closure, the count identity as numbers) are the closure
  clauses themselves and fail for the parity adversary; the rest (onset, air cap, ember,
  quiet-zone only-if) are the clause "fuel above Q" and hold for every subset of the real fuel.
  Counts in table 1.
- **V15 (the slice profile; MEASURED).** The real slice of the period is poor at the bottom
  (47 % of the period density in turn 1), rich in the middle (127 % at turn 60, 175 % in pairs),
  and at the period value at the top; a random slice is flat. Mechanism: the air cap and onset
  close the families at the bottom; a single prime's density 1/log x against the period's
  rough density e^(-gamma)/log Q lifts the middle. The record sits in the poor bottom.
- **V16 (the number-level count identity is Legendre; FACT, ROOT as a route).** T - E = sum over
  smooth s of N_pure(Q^2/s), exact for every saturated model; solving it for the pure count is
  computing the prime count; the pair system is not closed.
- **V17 (the saturated adversary; EXACT).** The phase-zero sieve by the primes in (q, Q] together
  with the twin members in (Q, Q^2] passes every row of table 1 that the real manifold passes and
  has 0 pure pairs in every turn (880,214 gears above Q at Q = 10^4). Saturation alone does not
  force a pure pair; saturation plus the cut is the manifold.

## What is new

- The translation lemma as the exact reason the period-scale laws are silent about phase: it
  turns "which laws have phase-zero content" into "which laws name a position", and the answer
  on the zone is the zone laws and their consequences, nothing else. Elementary once written;
  not on record before.
- Phase zero as saturation, and the split it makes among the counterfactuals: F (V2) and the
  one-tooth adversary have composite fuel (not divisor-closed below Q); the raw-line free-phase
  manifolds fail every closure at 76.5 %; the parity adversary is a sub-semigroup and not
  divisor-closed. One property, tested by counts, separates the real charge set from every
  adversary on record. And the saturated adversary that shows the property is not enough.
- The slice profile: the quiet zone is not a typical slice of the manifold's period in either
  direction, and the two directions have named mechanisms (onset at the bottom, the prime
  density in the middle). The record's position (W103) is the bottom of that profile.
- The Liouville-signed charge census cancelling to 0.1 % against a pure share of 8.2 %:
  the parity barrier measured on the charges (prior art in kind).

## Verdict

1. Phase zero has no period-scale content (V13): every manifold law that does not name a
   position, thirteen of them, holds for every free-phase manifold with 0 exceptions. It has
   position content only, and on the quiet zone the position content is the zone laws.
2. Phase zero is saturation (V14): the open set is closed under divisors and under smooth
   multiplication. Every phase-zero law on record is a consequence of saturation plus the cut.
   The order of the multiples is never used beyond P > Q.
3. The property X: the real charge set is SATURATED (0 violations in 107,750,211 closure tests at
   (5, 10^4)); no free-phase adversary has it (76.5 % failure at random phase and for the
   two-tooth adversary; 16,839,460 composite fuel members for the one-tooth adversary; F of V2
   likewise), and no parity-defined adversary has it (10,804,164 divisor-closure failures, and
   closure under exactly the even-Omega air). X is the sharpest such statement, and it is one
   property, not several.
4. X does not force the pure charge (V17): the saturated sieve with the twin members added as
   gears passes every row and has none. What excludes it is the cut, and "saturated with the
   cut" is the manifold's definition. The existence consequence of X + cut is the conjecture.
   ROOT, as pre-registered.
5. Where phase zero acts on the slice (V15): everywhere, in opposite directions; the record sits
   where it empties. The period-scale record laws do not see the slice at all (the rule's slack
   at the zone record is 937 of 1,147; the free core covers the whole window).

Status for the tree: V13 PROVED (FACT: the lemma); V14 PROVED (the characterisation, with the
law split EXACT); V15 MEASURED with mechanism (the slice profile); V16 FACT/ROOT (Legendre);
V17 EXACT (the saturated adversary), ROOT mark on the branch: the property found separates the
real set from every adversary on record and its existence consequence is the root. No
CANDIDATE. Node R4.c.iv: DONE, ROOT honestly, with the split 13 phase-free / 1 position /
10 phase-zero of which 3 beyond multiplicativity (one property).

## Dead ends

- "Some period-scale law has phase-zero content": none does; V13 is the refutation, 15 of 15
  wheels. The census on the zone deviates, but that is the slice, not the law.
- "The order of the multiples is a separate ingredient": the spacing of a gear's strikes is
  the same at every phase; the strikes on the charges at s x P is the ideal property. No law
  uses g, 2g, 3g beyond P > Q.
- "The parity adversary fails the size laws": it has them (0 violations of onset, air cap, ember,
  inventory, imprint, port at both Q), being a subset of the real set.
- "Saturation forces a pure pair": TWINSIEVE, 0 pure pairs at Q = 10^3 and 10^4, both q.
- The prediction "the real density rises to the period value by turn 100": wrong; it overshoots
  to 127 % and comes back at the top (V15). Recorded as the one place the pre-registration was
  wrong in kind.
- The one-tooth adversary's "0 fuel members with a manifold factor" in V12's table: my count of
  its open numbers with a composite rough part is 16,839,460 at (5, 10^4) and 190,100 at
  (5, 10^3), so the "one tooth per manifold prime" row of V12 should read as a property of the
  adversary's STRIKE (one class per gear), not of its fuel's factorisation; its fuel is not
  divisor-closed. Nothing in V12's verdict changes: phase zero was the separating property
  there, and saturation is its name.

## Open items of the part, sorted

- Closed here: which laws have phase-zero content (all of them listed, with counts); what phase
  zero is (saturation); which adversary has the phase-zero laws with content beyond
  multiplicativity (none; the parity adversary fails divisor closure); the charge set as a
  saturated module and its count identity (Legendre); the record rule on the slice (silent).
- Measurement with no structural content: the slice profile's numbers (V15), the random slices'
  records, the twins at 0.90 of a random slice's pure pairs, the signed census.
- Root question in disguise: "the minimal saturated fuel has a pure pair in every zone"; any
  bound on the zone record from the period record.
- Genuinely open on the part alone: none of structural kind. The one object this branch adds
  that is not a count is the profile's shape (poor bottom, rich middle), which is the onset law
  and the prime density read together; whether the pure charge's bottom excess (three times a
  random slice's pure pairs in turn 1) has a structural expression that survives the cut is the
  same question as the root.
