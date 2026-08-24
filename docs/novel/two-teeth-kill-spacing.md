# two-teeth-kill-spacing - the kill spacings of an added gear lie in two alternating classes, and fuel is span-capped in closed form

Status: KERNEL-CHECKED (T1-T5, proofs/TwoTeeth.lean + MergeLaw.lean, round 21
formalist - see the Lean pointers in section 3) + MEASURED (M1, every window of every
full joint period 11->13 .. 29->31) + the operator reading SCRIPT-VERIFIED (exact
matrix identity + operational iteration).  Established round 21 (constructor), from a
law found live with the human on 2026-08-24.  Prior-art check: not yet checked (agent
without web access).

## 1. WHAT IT IS

Plain language.  When gear q' is added to a machine M, the openings it kills inside one
merge window are not free to sit anywhere: consecutive kills are separated by exactly
one of TWO values - 2u' and q'-2u' (u' = round(q'/6)) - or by exactly a multiple of q'
(padding), and the two values must strictly ALTERNATE within a window (padded links are
transparent to the alternation).  The two values are the project's literal letters; the
smaller is ~q'/3.  Consequence, closed form: a window whose interior spans L slots can
contain at most 1 + 3L/(q'-1) kills - the fuel cap needs no census, it is span
arithmetic.  In the round-20 operator frame the same law is the support of the right
tensor factor of the new blocked-walk operator: which kill patterns are even
q'-realizable is decided by this law alone, before the old machine is consulted.

Precise form.  Step M -> M+q', c = 6^{-1} mod q', teeth {+-c}, u' = round(q'/6),
letters a = 2u', b = q'-2u'.  A merge window = maximal run of consecutive M-openings
all killed by q' (k kills, k-1 interior spacings s_i = interior gaps of the merged
window, span = sum s_i).

  T1 (letters identity)   {2c mod q', -2c mod q'} = {a, b}: the tooth-difference
      residues ARE the literal alphabet.  (6u' = q' -+ 1 gives u' = +-c mod q'.)
  T2 (residue law)        s_i mod q' in {0, +2c, -2c}.
  T3 (sign alternation)   a spacing = +2c mod q' moves the kill residue -c -> +c,
      = -2c moves +c -> -c, = 0 mod q' keeps it; hence the nonzero-class spacings
      STRICTLY ALTERNATE in sign (padded spacings transparent), and |#a - #b| <= 1
      within any window.
  T4 (minimum)            every nonzero-class spacing >= a = 2u'; padded >= q'.
      (The q-1 minimum in the adjacent frame is docs/novel/deletion-spacing.md;
      T4 is its k-frame form.)
  T5 (FUEL-SPAN LAW)      k <= 1 + span/(2u') <= 1 + 3*span/(q'-1) - closed form,
      no census: ~3L/q' kills at most in an interior span of L.
  M1 (value law, MEASURED) every realized spacing VALUE is exactly a, b, or q' -
      the classes admit a+q', b+q', 2q', ...; the machine realizes none of them,
      at any of the six full joint periods scanned (29->31, joint period
      3.34e10: spacings 10: 7,815,766 / 21: 205,068 / 31: 4,180 - nothing else;
      the four k=4 windows are exactly Mechanic's four (10,21,10) addresses).

Operator reading (round-20 frame, new SUM SPLITTING).  With E' = I - D' the exposure
projector of q', on the tensor grid Z_P (x) Z_q':

    B_new S_new = (B_M S_M) (x) S'  +  (E_M S_M) (x) (B' S')

- one line of algebra (I (x) I - E (x) E' = B (x) I + E (x) B'), so the NEW blocked
walk = OLD blocked walk (x) shift PLUS old renewal step (x) q'-kill, and
F(M+q') = nilpotency index of this sum of two Kronecker products.  Expanding the m-th
power over binary kill-words, all entries are >= 0 (the operator is a masked
permutation), so the power is nonzero iff SOME word has both tensor factors nonzero,
and the factors live in coprime moduli (CRT): left = an old-machine pattern event,
right = a mod-q' event.  The right factor is nonzero EXACTLY when the kill offsets
obey T2/T3 - the spacing law is the complete support description of the right factor.

## 2. WHY IT MIGHT BE NOVEL

- deletion-spacing.md (round <=19) proved only the MINIMUM (q-1 in the adjacent
  frame).  This entry adds: the exact two-class VALUE structure, the identity of those
  classes with the literal alphabet (T1), the strict alternation with padded
  transparency (T3), the closed-form fuel-span cap (T5), and the measured exact-value
  law (M1).  R21's word list assumed alternation from the corridor censuses; here it
  is three lines of residue arithmetic.
- The one-class shadow (Holt & Rudd, arXiv:1408.6002, Lemma 3.1) makes removals
  trivially multiples of 2p apart; with two teeth the spacing set is two-valued with
  forced alternation - a genuinely different combinatorial structure (runs of kills
  exist at all; their grammar is the alternating word).
- The fuel-span law converts "how many merges can chain" from a census question into
  span arithmetic: k <= 1 + 3L/(q'-1) for any window of interior span L, every gear,
  forever.  No analogous closed form appears in the Jacobsthal computation literature
  the project has read (Hagedorn, Ziller-Morack, Costello-Watts).
- The sum splitting exhibits "adding a gear" as a rank-2 Kronecker RECURSION on the
  nilpotent walk operator (matrix-formulation.md piece 4 stated the flat rank-2 form
  over all gears; the recursive old-machine form and the identification of its right
  factor's support with the spacing law are new).

## 3. PROOF

T1: 6u' = q' -+ 1 (q' = +-1 mod 6), so u' = -+6^{-1} = -+c mod q'; hence
{+-2c} = {+-2u'} = {a, q'-b} = {a, b} as residues, and a + b = q'.  T2: consecutive
kills lie in {+-c} mod q'; differences of two such residues are 0 or +-2c.  T3: the
transition table is forced ( +2c only from -c, -2c only from +c, 0 keeps ); two equal
nonzero signs in a row would need 3c or -3c in {+-c}, i.e. q' | 2c or q' | 4c -
impossible.  |#a - #b| <= 1 follows since a and b are the two signs.  T4: the smallest
positive representatives of the classes +-2c are a and b >= a; class 0 forces >= q'.
T5: span = sum of k-1 spacings, each >= a = 2u' >= (q'-1)/3.  QED (all elementary).

M1 is measured, not proved: nothing in the residue arithmetic forbids a spacing
a + q'; the machine never realizes one (through 29->31).  A spacing a + q' would be a
single old gap of that size with both endpoints killed - its absence at scanned
machines is a size-times-residue coincidence event, unexplained.

KERNEL-CHECKED (round 21, formalist; proofs/TwoTeeth.lean unless noted, all on the
standard axiom footprint or smaller - see proofs/AxiomCheck.lean):

  T1  `TwoTeeth.teeth_letters`      2u' + (q'-2u') = q' and 6u' = -+1 mod q'
  T2  `MergeLaw.interior_gap_mod`   spacings of killed positions are 0, +2u', -2u' mod q'
      (stated for merged-window interiors; teeth abstract as {u, q-u})
  T2+T3 `TwoTeeth.spacing_from_lo` / `spacing_from_hi`  the full transition table,
      padded links transparent: from the low tooth only classes {0, q'-2u'} (landing
      high), from the high tooth only {0, 2u'} (landing low) - the strict sign
      alternation is these two lemmas composed
  T3  `TwoTeeth.next_kill_of_lo` / `next_kill_of_hi` / `kill_spacing` /
      `kill_period`  the consecutive-kill exact forms: spacings in {2u', q'-2u'},
      alternating, two consecutive spacings summing to exactly q'
  T4  `TwoTeeth.kills_gap_ge` (any two kills >= 2u' apart) and
      `TwoTeeth.kill_spacing_min` / `kill_spacing_min_gear` (consecutive form)
  T5  `TwoTeeth.fuel_span_cap` (k kills span >= 2u'(k-1)) and `TwoTeeth.fuel_le`
      (k <= 1 + span/(2u') - the closed-form fuel cap ~3L/q')

The gear side conditions (0 < u', 4u' < q') are discharged from 6u' = q' -+ 1,
q' >= 5 by `TwoTeeth.gear_side`; `kill_spacing_gear` / `kill_spacing_min_gear` are
the pre-discharged forms.  M1 (the exact-VALUE law - no spacing a+q' etc. ever
realized) remains MEASURED only, as stated below.  At the 19->23 step the value law
IS kernel-checked: `Machine23.merge_alphabet` pins every letter to {8, 15, 23}
(using the kernel gap cap F(19) <= 25).

Scripts: research/kill_spacing.py (T1-T5 asserted on every window of the full joint
period, steps 11->13, 13->17, 17->19, 19->23, 23->29, 29->31; M1 census printed; logs
research/data/kill_spacing_23.log, kill_spacing_23_29.log).  Window counts cross-check
R19's chain censuses exactly (with one extra window per period at 13->17 and 23->29 -
the cyclic-seam chain R19's linear scan did not stitch).  Operator part:
research/nilpotency_additivity.py (dense exact identity at {5}+7 and {5,7}+11;
operational index(sum) = F(M+q') at all four steps 11->13 .. 19->23; right-factor 0/1
matrices nonzero on spacing-law patterns and zero on violating ones, q' = 13 and 23).

## 4. IMPLICATIONS

Inside the project:
- (A), the word list, now DERIVES from the algebra: right-factor support = alternating
  {a, b} words with padded links - one derivation replaces the corridor-census route
  at the grammar level (the corridor/litcap CAP on word length remains separate).
- The fuel cap gets a machine-free closed form (T5) complementing the litcap: at the
  measured steps the realized k_max saturates T5's cap at 11->13 and 19->23 and sits
  one below elsewhere.
- THE COUNTING BOUNDARY, sharp (honest negative, nilpotency_additivity.py P3): the
  index of the sum is NOT bounded by any function of the marginal data (index of the
  old walk F, index of the kill factor 2, T1-T5, litcap).  The 2-point relaxation -
  every adjacent kill pair individually realizable - is satisfied by the INFINITE
  alternating word from 19->23 on (pairs (a,b) and (b,a) both occur adjacently), while
  the true chain stops: the growth bound delta <= q' is a >= 3-POINT
  joint-realizability statement, nothing weaker (matches R37's tropical boundary from
  the operator side).  What remains of (D) in this frame: which spacing-compatible
  kill patterns the old machine realizes - the anti-correlation clause.

Outside: for any two-residue-classes-per-prime sieve, the per-prime strike pattern on
the survivor set has a two-letter alternating grammar with a closed-form density cap -
the structural reason paired-Jacobsthal-type functions grow by O(q') per prime.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- The increment law / part (D) (the route's sole open input): this entry proves the
  grammar-and-count shell and locates the remaining content exactly (>= 3-point).
- Ziller-Morack Conjecture 6 / h_2 growth: the per-prime growth mechanism is now
  grammar + span arithmetic; only the flank sizes are open.
- M1 (why no spacing a + q' is ever realized) is a new, sharply falsifiable
  micro-question; first place to look: 31->37, where padding onsets.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access this round).  Suggested searches: Holt Rudd
cycle of gaps closures spacing two residue classes; "Jacobsthal function" increment
adding a prime maximal gap; wheel sieve strike pattern alternation; sum of Kronecker
products nilpotency index recursion sieve.  Nearest known art: Holt & Rudd Lemma 3.1
(one-class closure spacing, trivial there); deletion-spacing.md's check (2026-08-23)
found no two-class statement anywhere - this entry's delta on top of that one is the
class structure/alternation/fuel-span/operator-support content, none of which was
searched separately yet.
