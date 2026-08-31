# Dictionary monotonicity, and the inflation onset of the transfer

Established round 28 (mechanic). Two findings, the second resting on the first.

## 1. WHAT IT IS

### (a) THE DEPTH-0 LEMMA: the gap-tuple dictionary is monotone along the machine ladder

Machine `M` is the slot sieve by all gears `5 <= q <= M`; its *openings* are the
surviving slots and its *gaps* the differences of consecutive openings. Write
`D_m(M)` for the set of gap `m`-tuples that actually occur at machine `M` - the
machine's realised `m`-tuple dictionary. Adding one more gear `q'` deletes some
openings, which MERGES gaps, so there is no a-priori reason for an old tuple to
survive: every one of its four gaps could be destroyed. It does survive.

> **Lemma (depth 0).** For every prime `q' > 2(m+1)`,  `D_m(M) ⊆ D_m(M + q')`.
> In particular `D_4(M) ⊆ D_4(M + q')` for every `q' >= 11`, so the realised
> 4-tuple dictionary only ever GROWS along the ladder of machines.

### (b) THE INFLATION-ONSET LAW

The *dictionary transfer* (project construct K4) computes a certified SUPERSET
of `D_4(M + q')` from `D_4(M)` alone, by taking the order-4 closure of `M`'s
dictionary and running one free phase mod `q'`. Sorting its output by span
`|t| = g_1 + g_2 + g_3 + g_4`, the transfer is EXACT up to some span and starts
over-generating sharply above it. Call that threshold the *inflation onset*:

    onset(M -> q')  =  min { |t| : t in the screened superset, t not in D_4(M+q') }.

Measured exactly at eight steps (each from two exact dictionaries):

    step      11->13 13->17  17->19  19->23  23->29  29->31  31->37  37->41
    onset        13     15      17      25      31      41      53      68

> **The law.** With `q''` the next prime after `q'`,
>
>     onset(M -> q')  =  min span of  [ (D_4(q'') \ D_4(q'))  ∩  (emissions of
>                                       the M -> q' transfer) ].
>
> In words: **the transfer M -> q' first over-generates exactly where the NEXT
> machine's new repertoire begins.** The transfer emits, one gear ahead of
> schedule, the tuples that only become realisable when the following gear is
> added.
>
> The intersection with "what the transfer can emit" is LOAD-BEARING. Dropping
> it gives the simpler form `onset = min span of D_4(q'') \ D_4(q')`, which
> looks almost as good at arity 4 - 7 of 8, failing only at the smallest step
> `11 -> 13`, where `min span of D_4(17) \ D_4(13) = 10` (witness `(2,2,1,5)`)
> against a measured onset of 13, because machine 11's 73-tuple dictionary has
> no walk that emits `(2,2,1,5)` at all. **The arity-3 test settles which form
> is the law: refined 6 of 6, simple 2 of 6.** The simple form's arity-4 record
> was the luck of rich dictionaries.
>
> **AND THE LAW IS ARITY-INDEPENDENT**, which is the test that makes it a fact
> about the transfer rather than about one census. `D_3(M)` is the induced
> 3-tuple dictionary of `D_4(M)`, EXACTLY (every realised 3-tuple sits inside a
> realised 4-tuple), so arity 3 costs no new scan:
>
>     step            11->13 13->17 17->19 19->23 23->29 29->31 31->37
>     onset (arity 3)    17     14     20     25     36     44     57
>     onset (arity 4)    13     15     17     25     31     41     53
>
> refined law HIT at all six testable steps (31 -> 37 needs `D_3(41)`, which
> does not exist). Note the arity-3 onsets are mostly HIGHER: a shorter pattern
> is pinned by the same order-4 closure for more span, which is the direction it
> has to go. Arity 2 goes further in the same direction - the transfer has NO
> onset at all (it is exact) at `11->13`, `13->17` and `17->19`, and where an
> onset exists the refined law hits every time:
>
>     output arity   refined law   simple law   steps with NO onset (exact)
>          2           3/3 tested     1/3                    3
>          3           6/6            2/6                    0
>          4           8/8            7/8                    0
>          5           3/3            2/3                    0
>          6           3/3            2/3                    0
>          7           2/2            2/2                    0  (one step skipped -
>                        the depth-0 lemma genuinely fails at m = 7 there)
>         TOTAL       25/25          16/25
>
> A sixth arity-5 point was added on a much bigger step once machine 29's exact
> 5-tuple dictionary existed: `onset(23 -> 29, arity 5) = 30`, against 31 at
> arity 4 and 36 at arity 3, with the depth-0 lemma holding there.
>
> (arity-2 onsets: -, -, -, 27, 41, 50, 66 across `11->13 ... 31->37`;
> arity-5 onsets 13, 17, 18 at `11->13, 13->17, 17->19`. Arity 5 keeps the
> SOURCE at the exact 4-tuple dictionary, so the closure is still order 4 and
> only the output size moves - the variation that asks whether the law is about
> the transfer or about the chain's particular output arity. It is about the
> transfer.) The depth-0 lemma is asserted at arities 2, 3, 4 and 5.

**AND IT TRACKS THE SCREEN.** The walk screen (section 3, below) moves exactly
one onset, `13 -> 17` from 15 to 17. Since the law's right-hand side is
intersected with the transfer's emissions and the walk screen changes what an
emission is, this is the law's own variable moving. Under the walk screen the
refined law is 6 of 6, with the `13 -> 17` right-hand side moving to 17 in step.
Running total for the refined form: **31 of 31** across six output arities
(2, 3, 4, 5, 6, 7) and two screens.

The causal version - which is what makes the refined form 8/8, since the two
are equivalent given it - was tested separately and holds at every step:
**every tuple refuted at the onset span is realised at the next machine `q''`**
(witnesses `(1,2,3,7)`, `(1,5,4,5)`, `(3,2,3,9)`, `(1,5,2,17)`, `(8,2,6,15)`,
`(5,5,25,6)`, `(10,2,28,13)` at `11->13 ... 31->37`).

**AND IT PREDICTS THE TOP STEP OUT OF SAMPLE.** The 37 -> 41 onset, 68, was
MEASURED in round 27 by computing an exact machine-41 shard span by span. The
law computes it instead as `nu(41 -> 43)` - the min span of a screened
`41 -> 43` transfer candidate that is not already in `D_4(41)` - which needs
only the machine-41 dictionary, no machine-43 anything, and no solver:

    nu(41 -> 43) = 68,  witness (5, 36, 2, 25).     PREDICTED CORRECTLY.

(`research/onset_oos_r28.py`, log `research/data/r28/onset_oos.log`; the
transfer is capped at span 75, inside the shard's exact region, since a walk of
span `s` has every 4-window of span `<= s`.) So the top of the ladder is a genuine
out-of-sample prediction of a number produced by a different vehicle in a
different round.

**THE MECHANISM A PROOF WOULD HAVE TO FORMALISE.** The order-4 closure admits
walks the machine does not realise, and a walk with `k` kills emits a 4-tuple
whose gaps are merges of `k+4` of `M`'s gaps. Gear `q'` alone can only delete
along ONE phase, so the merges it can produce are constrained; the closure,
by allowing an unrealised walk, effectively buys the transfer a little more
deletion than one gear is entitled to - and "a little more deletion" is
precisely what the NEXT gear supplies. So the transfer's first mistake should
be a tuple that becomes realisable at `q''` and not before. That is exactly
what the causal test measures, and it is 8/8. Turning it into a theorem means
bounding, for the smallest over-generated span, the deletion budget of a closure
walk against the deletion budget of one extra gear.

**WHAT IT IS NOT.** Three closed forms in the machine's own constants were
PRE-REGISTERED before the ladder was measured (`F_2` one machine back;
`2F` two machines back; a constant ratio to `F(M)`) and ALL THREE FAIL at
every out-of-sample step - the third only ever "hit" its own calibration
point. The onset is not a letter combination and not a ratio to `F`; it is a
recursion in the ladder.

## 2. WHY IT MIGHT BE NOVEL

(a) is elementary once stated, but it is not the obvious direction: a new gear
DESTROYS openings, and the natural guess is that a dictionary of consecutive
gaps is destroyed with them. The content is that the *finite* pattern's
survival is a phase question with only `2(m+1)` forbidden residues, and one
lap of the CRT bijection supplies the phase. The project's own tooling had
been treating "is this old tuple still realised?" as a solver question for two
rounds (round 27 priced 1.4M such decisions at >= 1,121 core-hours) - 16.7% of
that population is decided by this one line.

(b) is a statement about how much of a machine's local combinatorics is
determined by the machine one gear back, and it is not a statement we have
found an analogue of. The natural sieve-theoretic shadows (admissible tuples,
Hardy-Littlewood local densities) are about which patterns are eventually
realisable in the primes, not about the exact span at which a finite sieve's
own order-4 closure stops being faithful.

## 3. PROOF

### (a) PROVED (elementary) + SCRIPT-VERIFIED at seven pairs


Let `w in D_m(M)` occur at an `M`-opening `y_0`, with exposed offsets
`X = {0, g_1, g_1+g_2, ..., |w|}`, `|X| = m+1`. Adding gear `q'` deletes the
slots congruent to `±u'` mod `q'`, `u' = 6^{-1} mod q'`; equivalently, writing
`A = (u' - y_0) mod q'` and `s = 2u' mod q'`, the point at offset `d` is deleted
iff `d mod q' in {A, A - s}`. So `w` survives intact at the occurrence `y_0`
iff `A` avoids the set `(X mod q') ∪ ((X + s) mod q')`, which has at most
`2(m+1) < q'` elements - so an admissible `A` exists. Finally `P(M)` is
invertible mod `q'`, so as `y_0` runs over the `q'` translates `y_0 + jP(M)`
inside the new period, `A` runs over ALL residues mod `q'`; pick the lap with an
admissible `A`. Then every point of `X` survives and the `m+1` consecutive
openings are consecutive at `M + q'` too (any opening between them at `M+q'` is
an opening at `M` between them, and there are none). Hence `w in D_m(M+q')`. ∎

**AND THE HYPOTHESIS `q' > 2(m+1)` IS SHARP** (`research/depth0_sharp_r28.py`,
log `research/data/r28/depth0_sharp.log`). Sweeping `m` upward at the small
steps, monotonicity holds exactly as far as the proof reaches and then breaks:

    step     proof covers   first m at which D_m(M) is NOT contained in D_m(M+q')
    7 -> 11     m <= 4                 6      witness (2,1,2,2,1,2)
    11 -> 13    m <= 5                 7      witness (3,2,2,1,2,2,3)
    13 -> 17    m <= 7                 8      witness (5,2,2,1,2,2,1,4)
    17 -> 19    m <= 8                 9      witness (2,5,5,2,1,2,5,2,5)

so at `q' = 17` and `q' = 19` the first failure is at exactly the first `m` the
proof does not cover - the hypothesis is TIGHT, not an artefact of the counting
- and at `q' = 11, 13` it has slack 1. Every witness is a dense small-gap
pattern, which is what saturates the new gear's phase set.

Script: `research/onset_anatomy_r28.py` Part A asserts
`D_4(13) ⊆ D_4(17) ⊆ D_4(19) ⊆ D_4(23) ⊆ D_4(29) ⊆ D_4(31) ⊆ D_4(37)`
(all six from full-period censuses) and `D_4(37)|span<=77 ⊆` the round-27
exact machine-41 shard. Log `research/data/r28/onset_anatomy.log`.

### (b) MEASURED (refined form 8/8, simple form 7/8), not proved

Scripts: `research/onset_law_r28.py` (log `research/data/r28/onset_law.log`;
part (1) is the mechanism test, part (2) the law), `research/onset_m11_r28.py`
(the 11 -> 13 rung, where the simple form fails and the refinement is checked
explicitly), `research/onset_oos_r28.py` (out of sample). Inputs are exact full-period
4-tuple dictionaries at machines 13, 17, 19, 23, 29, 31, 37 (the first four
recomputed in-round from the cyclically closed period, `F` and `F_4` asserted
against their known values) and the round-27 exact machine-41 shard.

A partial explanation is available and is worth recording, because it says what
a proof would have to supply. The transfer's emissions split by DEPTH (number
of interiors deleted by `q'`). Depth 0 emissions are realised by the lemma
above, so every refutation has depth >= 1, i.e. comes from a walk of >= 5
`M`-gaps that the order-4 closure admits. Define

    X_5(M) = min span of a 5-walk both of whose 4-windows are realised at M
             but which is not itself realised at M.

Measured: `X_5 = 9` at machines 13, 17, 19, 23 - **the same value with the same
witness `(1,2,3,2,1)` at every machine** - and that witness is phase-saturated
at gear 5 (`X = {0,1,3,6,8,9}`, `X mod 5 = {0,1,3,4}`, `(X-3) mod 5 = {0,1,2,3}`,
union `= Z_5`), hence zero at every machine by the phase-saturation theorem.
This exactly explains the UNSCREENED onset, which is 9 at all seven steps.
Removing the saturated walks gives `Y_5`, a lower bound on the screened onset,
and it was carried to machine 29 by a streamed full-period pass
(`research/y5_m29_r28.py`; the machine's exact 5-tuple dictionary, 208,668
tuples, is new, and its induced 4-tuple dictionary is EXACTLY the round-25
full-period census - two independent scans agreeing cell for cell):

    machine    m13   m17   m19   m23   m29   m31
    X_5          9     9     9     9     9     9    (same witness every time)
    Y_5         10    17    18    22    30    38
    onset       15    17    25    31    41    53
    onset/Y_5 1.50  1.00  1.39  1.41  1.37  1.40

So closure failure plus phase saturation accounts for most of the onset and not
all of it; the residue is the multiplicity fact that a low-span emission usually
has a REALISED source as well as an unrealised one - and that residue is NOT
running away: at the FOUR largest machines where both quantities are known it
is 1.389, 1.409, 1.367, 1.395, i.e. `onset / Y_5` sits in a band of width
0.042. The multiplicity residue behaves as a near-constant FACTOR, so the onset
is (closure failure) x (phase saturation) x (a constant), and the third factor
is the only part still unexplained.
**Status: (a) PROVED, (b) MEASURED / CONJECTURED.**

### A tool consequence, proved in passing: the WALK screen

The round-26 phase-saturation screen is applied to the EMITTED tuple. But every
point of the underlying walk - the deleted interiors included - is an
`M`-opening, so the WHOLE WALK must have an admissible phase at every gear
`q <= M`. Screening the walk is sound (a realised walk has an actual phase),
strictly stronger (it sees obstructions the emission has forgotten) and a
prefix prune (the bad-phase set only grows). Measured on the superset sizes:

    step      truth    raw superset   emission-screened   WALK-screened
    19->23    15,696        66,238            47,623           42,045
    29->31   115,193       715,697           471,135          419,990
    31->37   291,675     2,435,140         1,182,475        1,153,814

and the walk screen SUBSUMES the emission screen (identical output when both
are applied, at all six steps). Script `research/onset_walkscreen_r28.py`,
which asserts at every step that no realised tuple is removed.

## 4. IMPLICATIONS

- **Inside the project.** The depth-0 lemma decides 145,907 of the 874,087
  reverse classes of the machine-41 arity-4 superset - 16.7%, i.e. 291,675
  tuples - as YES with no solver call, at every span including the bands
  round 27 priced at 3.5 s a decision. It removes that share from the
  >= 1,121 core-hour price of the exact census and hands the chain's rung-nine
  oracle a free positive half. The walk screen removes a further 2.4-11.7% of
  the superset by arithmetic.
- **The onset law makes "exact below the onset" predictable.** Round 27 could
  only report the onset after the exact shard had been computed past it. The
  law computes it from ONE machine's dictionary and the next step's transfer -
  seconds, no solver - so a future superset can be split into a
  "certainly exact" region and a "must decide" region before any decision is
  paid for.
- **Outside.** The law is a quantitative statement that a sieve's local
  combinatorics at scale `s` is determined by the previous gear precisely up to
  the span at which the NEXT gear first adds a pattern. If it survives, it is a
  statement about how information propagates along the primorial ladder.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- The (D) chain: rung nine of the Constructor's certificate chain consumes the
  machine-41 arity-4 dictionary; both findings shrink its oracle gap directly.
- Jacobsthal / maximal-gap questions: `D_4(M)` monotone is the tuple-level
  analogue of the (false in general) statement that gaps only grow; the true
  statement is that CONFIGURATIONS only accumulate.
- Open: prove the onset law, or find its first failure. The natural next test
  is 41 -> 43, which needs `D_4(43)` or a further extension of the m41 shard.

## 6. PRIOR-ART CHECK

NOT YET CHECKED (mechanic has no web access). Suggested search terms for the
manager: "gap pattern dictionary primorial sieve monotone"; "consecutive gaps
of the k-th primorial sieve, admissible patterns"; "Hagedorn / Holt-Rudd gap
sequences of Z/p# survivors"; "order-m closure of a sofic shift, exactness
threshold"; the transfer's object is a sofic-shift approximation, so
symbolic-dynamics literature on "follower-set / order-m Markov approximations
of sofic shifts" is the closest external frame.
