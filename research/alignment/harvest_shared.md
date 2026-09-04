# Harvest: what the record already says about ALIGNMENT OF OPENINGS

Sources read in full: `docs/proof-search/agents-shared.md` (all 8,227 lines, every round
block, rounds 25-31 verbatim plus the compacted 1-24 summary and the standing laws list) and
`docs/novel/` (every alignment-bearing entry read in full; the `j2-*`, `jk-*`,
`layered-erdos-rankin`, `paired-jacobsthal-values`, `unit1-submission-memo` entries were
checked and carry no alignment content - they are Jacobsthal bound/citation work).  The
round-29 section-view results named in the brief live in `docs/proof-search/word-tree.md`
sections 7-9 and `docs/proof-search/anchor-235.md` section 7; those two sections are
harvested here as well because the brief names the scripts by name.

VOCABULARY.  Machine `M = {5..y}`, one gear per prime; gears 2 and 3 are the columns
(column `k` = the pair `6k-1, 6k+1`).  Gear `q` STRIKES column `k` iff `k = +-6^{-1} mod q`
(its two TEETH `+-u_q`, `u_q = round(q/6)`); its OPENINGS are the other `q-2` columns.  The
machine's openings are the columns every gear leaves open: `prod(q-2)` per period
`P = prod q`.  WINDOW = the certified range (`6k+1 < q'^2`).  SECTION = the new part,
`p^2 < 6k+1 < q'^2`.  STRETCH = a run of consecutive columns anywhere in the period.
RECORD `F(M)` = the longest stretch with no opening, in the max-gap convention (distance
between consecutive openings).

TRANSLATION NOTE, load-bearing: the docs say "window" for a run of `J` consecutive gaps of
`M` (equivalently a stretch spanning `J+1` consecutive openings).  Every entry below writes
that as a **J-run** and reserves "window" for the certified range.  Second collision, on
record as a hazard (Harvester r30, `agents-shared.md` "THE COLLISION LIST"): `F_2(M)` is the
DEPTH-2 spectrum (longest stretch spanning 3 consecutive openings) while `F(2,y) = 3F(y)` is
the member-unit twin ladder.  Everything here uses `F_j`.

Letters, used throughout: `u' = round(q'/6)`, `c = 6^{-1} mod q'`, `d' = 2u' = 2c`,
`a = d'`, `b = q' - a`, so `a + b = q'` and `3a = q' -+ 1`; `s_min(q') = min(a, b) = a`.
A gap value `v` is a LEGAL LETTER iff `v mod q' in {0, +a, -a}`; `0 mod q'` is PADDED,
`+-a` LITERAL.  `L(M)` = length of the longest realised legal word (T3-alternating run of
legal letters as consecutive gaps of `M`).

---

## A. THE ALIGNMENT MECHANISM ITSELF (copies, phases, teeth)

### Lap/copy structure
- STATEMENT.  The new period `P q'` is `q'` copies ("laps") of the old period.  In copy `j`
  the openings are `o + jP`, and copy `j` deletes exactly the old openings whose residue mod
  `q'` sits on a tooth as seen from that lap; the deleted pair shifts by `-P mod q'` per lap.
  Since `gcd(P, q') = 1` the map `j -> -u' - jP mod q'` is a bijection of `Z_{q'}`: **the
  `q'` copies realise every deletion phase exactly once, and each opening of `M` is deleted
  in exactly 2 of the `q'` copies (one per tooth).**
- CALCULATES: which columns the new gear's openings coincide with the machine's openings at,
  for every phase at once; it turns "does this alignment occur somewhere" into "is there an
  admissible residue", with no search over position.  Algorithm: pick the phase, delete.
- STATUS: kernel-checked (`AnchorChain.copy_phase`, `phase_bijective`, machine-free);
  the "exactly 2 copies" count script-verified at m11..m23 (`hr_twoclass_r30.py` A, 2N hits).
- WHERE: `docs/novel/merge-law.md` 1(a); `docs/novel/anchor-235-layer-laws.md` (L3);
  `agents-shared.md` Harvester r30 follow-on "Holt-Rudd in two classes"; `proofs/AnchorChain.lean`.
- LIMITS: says where alignments CAN occur, not which of them the old machine realises.

### Chain law (when two openings can be struck together)
- STATEMENT.  Two columns lie in a common two-class set `{r, r+d_g} mod g` iff their
  difference is `0`, `+d_g` or `-d_g mod g`.  Hence two consecutive openings of the lower
  machine are both struck by gear `g` at some phase **iff their gap is `0` or `+-d_g mod g`**;
  the gap sizes that can carry a second strike are the classes
  `{d_g, g-d_g, g, g+d_g, 2g-d_g, ...}` cut at `F(M)+1`.
- CALCULATES: from the lower machine's gap value alone, whether the new gear can align its
  teeth on both ends of that gap.  One modular test per gap.
- STATUS: kernel-checked, both directions, every gear (`AnchorChain.chain_law`,
  `teeth_eq_phase`); admissible-gap list script-verified on full periods `{5..23}`.
- WHERE: `docs/novel/anchor-235-layer-laws.md` (L1); `docs/proof-search/anchor-235.md` 9d.
- LIMITS: a condition on ONE gap; says nothing about whether a longer run is realised.

### T1 - the letters are the tooth differences
- STATEMENT.  `{2c mod q', -2c mod q'} = {a, b}` with `a + b = q'` and `a = 2u'`: the
  residues by which two struck openings can differ ARE the literal alphabet, and the smaller
  is `~q'/3`.  Exactly `a = (q'-1)/3` if `q' = 1 mod 3`, `a = (q'+1)/3` if `q' = 2 mod 3`.
- CALCULATES: the whole legal alphabet at a step from `q'` alone: `Lambda(M) = {v <= F(M) :
  v = 0 or +-2c mod q'}`.
- STATUS: kernel-checked (`TwoTeeth.teeth_letters`); `a = 2 round(q'/6)` with `3a = q' -+ 1`
  asserted at all 2,258 primes `11..20000` (`bare_lemma_r31.py` GATE A1).
- WHERE: `docs/novel/two-teeth-kill-spacing.md` T1; `docs/novel/bare-word-uniform-cap.md` 1.1.
- LIMITS: the alphabet, not which letters occur as gaps of `M`.

### T2 - the residue law on a run of aligned strikes
- STATEMENT.  Consecutive struck openings sit on `{+-c} mod q'`, so every interior gap of a
  struck run is `0`, `+2c` or `-2c mod q'`.
- CALCULATES: a necessary test on every interior gap of a candidate merge, machine-free.
- STATUS: kernel-checked (`MergeLaw.interior_gap_mod`).
- WHERE: `docs/novel/two-teeth-kill-spacing.md` T2; `docs/novel/merge-law.md` 1(b).
- LIMITS: necessary only; the cover half (everything else in the stretch blocked) is separate.

### T3 - strict alternation, padded letters transparent
- STATEMENT.  A spacing `= +2c mod q'` moves the struck residue `-c -> +c`, `= -2c` moves
  `+c -> -c`, `= 0` keeps the tooth.  So the NONZERO-class spacings of an aligned run
  STRICTLY ALTERNATE and `|#a - #b| <= 1` within any run; padded spacings are transparent.
  Two equal nonzero classes in a row would need `3c` or `-3c in {+-c}`, i.e. `q' | 2` or
  `q' | 4`.
- CALCULATES: the grammar of alignment: which words of legal letters can describe an aligned
  run at all.  Two consecutive nonzero letters therefore sum to `>= a + b = q'`.
- STATUS: kernel-checked (`TwoTeeth.spacing_from_lo/_hi`, `next_kill_of_lo/_hi`,
  `WordLegal.legal_iff_noRepeat`, `alt_iff_prefixSum`).
- WHERE: `docs/novel/two-teeth-kill-spacing.md` T3; `agents-shared.md` Formalist r30/r31.
- LIMITS: grammar only; the sizes and the cover half are elsewhere.

### Neighbour-of-a-hit (adjacent columns are never both struck)
- STATEMENT.  For every gear `g >= 5`, if column `x` is struck then `x+1` is not, because
  `d_g = 2u = 3^{-1}` and `3^{-1} = +-1 mod g` would force `g | 2` or `g | 4`.
- CALCULATES: the `x+1` restart in the nested next-opening formula; and `P(open | next to a
  g-hit) = P(open) g/(g-2)` exactly (the neighbour-of-a-hit law).
- STATUS: kernel-checked for every gear from `6u = 1` alone (`AnchorChain.neighbour_of_hit`).
- WHERE: `docs/novel/anchor-235-layer-laws.md` (L2); `agents-shared.md` commit
  "anchor 2,3,5 line - ... neighbour-of-a-hit law".
- LIMITS: one gear at a time.

### T4/deletion spacing - the minimum distance between two aligned strikes
- STATEMENT.  Every nonzero-class spacing is `>= a = 2u'`; every padded spacing is `>= q'`.
  In the adjacent frame (gear 3 included, teeth `{o, o+1} mod q`) the same argument gives
  **two consecutive deletions inside one lap are at least `q - 1` apart, and that is tight**
  (attained at `q = 13` and `q = 19`).
- CALCULATES: a stretch of length `G` carries at most `1 + G/(q-1)` deletions; a chain of `k`
  deletions needs span `>= (k-1)(q-1)`.
- STATUS: proved (three-line mod-`q` case analysis); tightness script-verified; k-frame form
  kernel-checked (`TwoTeeth.kills_gap_ge`, `kill_spacing_min`).
- WHERE: `docs/novel/deletion-spacing.md`; `docs/novel/two-teeth-kill-spacing.md` T4.
- LIMITS: a lower bound on spacing, not on how many spacings chain.

### T5 - the fuel-span law
- STATEMENT.  `k <= 1 + span/(2u') <= 1 + 3*span/(q'-1)`: at most `~3L/q'` aligned strikes in
  an interior span of `L`, closed form, every gear, forever.
- CALCULATES: the arity ceiling of a merge from its span alone - span arithmetic, no census.
- STATUS: kernel-checked (`TwoTeeth.fuel_span_cap`, `fuel_le`).
- WHERE: `docs/novel/two-teeth-kill-spacing.md` T5.
- LIMITS: saturated only at m11->13 and m19->23; one below elsewhere.

### The realisability CSP (alignment = feasibility, no period)
- STATEMENT.  A tuple of gap values with prefix-sum points `X` and interior points
  `Y = (0, span) \ X` occurs as consecutive gaps of `M` **iff** there is a phase vector
  `(a_q)` with (open) `a_q not in {+-u_q - x mod q : x in X}` for every gear, and (cover) for
  every `t in Y` some gear has `a_q = +-u_q - t (mod q)`.
- CALCULATES: exactly where alignments occur, from the list of primes alone - the period
  never appears.  Decided by `crt_dict.decide_cover` (bitmask cover, min-remaining-options
  branching, a capacity bound that makes refutations affordable).
- STATUS: proved (CRT, one line); script-verified against R43's independent pruned
  inclusion-exclusion counter on every 1-, 2- and 3-tuple at m11/m13/m17, and set-equal to
  Mechanic's full-period censuses (`D_4(23)`, 15,696 tuples, tuple for tuple).
- WHERE: `docs/novel/scanfree-certificate.md` 1; `research/crt_dict.py`.
- LIMITS: the cover half's cost is `2^{|Y|}` in the worst case; shallow queries are the DEAR
  end and deep queries the cheap end (Constructor r28, "the opposite of the intuition this
  project carried for five rounds").

### Sparing count, and the sharp `s_min` threshold
- STATEMENT.  A run of `j+1` consecutive lower openings with offsets `X` is SPARED in exactly
  `q' - |X u (X+s)| (mod q')` copies; **if its span is `< s_min(q') = min(a,b)` then
  `|X u (X+s)| = 2(j+1)` and all `2(j+1)` hitting copies are distinct** (the two-class form of
  Holt-Rudd Lemma 3.1).  The threshold is SHARP: the smallest span at which two points of one
  run are hit in the same copy is `4, 6, 6, 8, 10` at m11..m23 - exactly the smallest realised
  legal letter each time.
- CALCULATES: how many phases leave a given stretch untouched; and where the one-class
  literature stops (above `s_min` its lemma is silent by construction).
- STATUS: proposition (three-line CRT), script-verified exhaustively at m11/m13/m17
  (945/10,395/155,925 runs, `j <= 7`) and sampled at m19/m23.
- WHERE: `agents-shared.md` Harvester r30, "Follow-on: Holt-Rudd in two classes" (2).
- LIMITS: `F(M) >= s_min` at every machine from m11 on and `F/q'` grows to 2.5 by 53->59, so
  every stretch that matters is above the threshold.

### Multiplicity of a chain (how many copies kill a whole run)
- STATEMENT.  The number of copies in which a run of `k >= 2` consecutive lower openings is
  struck ENTIRELY is `|intersection over the run of {u'-x_t, -u'-x_t} mod q'|`, which is
  **0** if the gap word is illegal, **1** if it is legal and contains a literal letter, and
  **2** if it is legal and every letter is padded (both tooth assignments work).
- CALCULATES: the exact number of aligned phases for a candidate word, with no search.
- STATUS: proposition, script-verified on every maximal run of `>= 2` hits at m11..m23
  (8 / 72 / 1,088 / 11,722 / 243,816 runs; every multiplicity 1 or 2 as predicted).
- WHERE: `agents-shared.md` Harvester r30 follow-on (2).
- LIMITS: gated NEGATIVE attached - the multiplicity does not decrease with `k`, so the count
  alone can never bound `L(M)` (see REFUTED list).

---

## B. WHERE THE NEW MACHINE'S RECORD COMES FROM

### The merge law (the record is computable from the machine below)
- STATEMENT.  Every gap of `M + q'` is either a gap of `M` or the merge of a MAXIMAL run of
  consecutive `M`-openings all struck by `q'`.  With `span(w)` the run's interior span and
  `FS_max(w; M)` the largest sum of the two flanking gaps over occurrences of `w`,
  `F(M+q') = max( F_2(M), max over compatible w of [ span(w) + FS_max(w; M) ] )` -
  an identity, not a ceiling.
- CALCULATES: the new record from the old machine's gap word plus `q'`, at `1/q'` of the cost
  of rebuilding; the word list and compatibility depend on `q' mod 210` alone, only
  occurrences and flanks come from `M`.
- STATUS: proved (elementary) + script-verified (whole gap histogram reproduced at four
  extensions); the BOUND form kernel-checked (`MergeLaw.newgap_le`, `newgap_le_max`) and a
  four-rung hypothesis-free ladder `Ladder.D_ladder` at 11->13 .. 19->23.
- WHERE: `docs/novel/merge-law.md`.
- LIMITS: one-step - it consumes an `F_2` and a qualifying spectrum and produces neither, so
  rungs do not chain without a fresh input.

### The attainment theorem (R68) - a legal word is always aligned SOMEWHERE
- STATEMENT.  If consecutive openings `x_0 < ... < x_J` of `M` have a legal middle-gap word
  then `x_J - x_0 <= F(M + q')`.  Proof: legality gives a tooth assignment, hence a residue
  `r mod q'` putting every interior on a tooth; the joint period is `P(M) q'` with
  `gcd(P(M), q') = 1`, so SOME translate of the stretch sits at that residue and has all its
  interiors struck; the containing gap of `M+q'` is then at least the span.
- CALCULATES: with the converse (`Q*_J <= F_J` and the Kleene identity),
  `max_J Q*_J(M; legal for q') = F(M+q')` EXACTLY - the record of the bigger machine from the
  smaller one, with `M+q'` never built.
- STATUS: proved both ways (r22 Kleene identity; r26 standalone CRT proof); verified EXACTLY
  at eight steps m11..m37 (`qstar.py`), and family-wide at 27,570 counterfactual machines
  with zero exceptions.
- WHERE: `docs/novel/kleene-generator.md` 4c; `docs/novel/old-machine-spectrum.md` 9;
  `agents-shared.md` Constructor r26 Headline 1.
- LIMITS: the NEGATIVE half travels with it - because `Q*_max` EQUALS `F(M+q')`, "the
  word-legal criterion certifies (D)" is the same statement as "(D) holds"; the margins are
  (D)'s true margins, not slack to exploit.

### The Kleene identity (all alignment depths in one equation)
- STATEMENT.  On states `(opening index i, tooth s)`, with `K[(i,s),(i+1,s')] = d_i` when
  `d_i mod q' in {0,a,b}` and `(s -> s')` is the T3 transition, and flanks `L(i) = d_{i-1}`,
  `R(i,s) = d_i`: `F(M+q') = L (x) K* (x) R` in max-plus.  `K` is nilpotent of index
  `k_max`, so `K*` is a finite sum, but **the identity names no truncation depth**; its
  `m`-th layer is exactly `Q*_{m+2}`.
- CALCULATES: every depth's aligned maximum from one algebra; and the arity-free dual
  certificate (C1) `h(i,s) >= d_i`, (C2) `h(i,s) >= d_i + h(i+1,s')`, (C3)
  `d_{i-1} + h(i,s) <= F(M) + q'` - each a ONE-STEP inequality with no depth index.
- STATUS: script-verified exact at m11..m29; the certificate DIRECTION kernel-checked
  (`Potential.IsPotential`, `D_of_potential`, `merged_le_of_potential`), with a concrete
  potential exhibited at 19->23 (`Potential19.lean`).  The identity itself is not
  kernel-checked.
- WHERE: `docs/novel/kleene-generator.md`.
- LIMITS: at a fixed machine it is a finite longest-path computation; the open part is a
  closed form for `h` valid at every machine.

### The word reduction R89 - alignment depth IS a word-length question
- STATEMENT.  `Q*_J(M; q') > -inf` **iff** `L(M) >= J - 2`.  Hence `J_max(M) = L(M) + 2` and
  `A_kill(M -> q') = L(M) + 1`.  Forward half: the `J-2` middles of a word-legal `J`-run are
  a realised legal word.  Converse: an occurrence of a realised legal word plus its two
  flanking gaps IS a word-legal `J`-run (legality constrains only the middles).
- CALCULATES: the depth cap of the whole per-J family from the SHALLOWEST dictionary the
  project has - "what is the longest run of consecutive gaps all of which are legal letters?"
  - decided by a handful of CRT calls with no census.  Every EMPTY cell of the per-J table
  becomes a one-line dictionary fact.
- STATUS: proved; script-verified 16/16 against the recorded `J_max` and `A_kill` rows;
  kernel-checked over an abstract opening enumeration (`WordLegal.chain_iff_word`, `akill`,
  `qstar_iff_word`, `jmax`) with one named hypothesis (periodicity of the gap residues);
  instantiated `L(11) = 1`, `L(13) = 1`, `L(17) = 1`.
- WHERE: `docs/novel/even-j-mechanism.md` 1.1; `docs/novel/per-j-window-analogues.md` 1.7.
- LIMITS: it moves the open question, it does not close it - `L(M)` bounded is still open.

### The same-tooth lemma R90
- STATEMENT.  A padded middle leaves the tooth fixed, a literal middle flips it, so the
  middle span `x_{J-1} - x_1` is `= 0 mod q'` **exactly when the number of non-padded middles
  is even**, and `+-2c mod q'` otherwise.  A LITERAL even-`J` chain therefore starts and ends
  ON THE SAME TOOTH, and its middle span is `>= ((J-2)/2) q'`.
- CALCULATES: the even/odd split of the alignment family by arithmetic (which tooth the chain
  ends on) rather than by counting parity.
- STATUS: proved; script-verified on 38 realised legal words, 0 violations; kernel-checked
  (`WordLegal.same_tooth`, `same_tooth_window`, `literal_even_span`, hypothesis `2c != 0`
  discharged from `6c = 1`).
- WHERE: `docs/novel/even-j-mechanism.md` 1.2.
- LIMITS: literal chains; a padded middle breaks the parity bookkeeping.

### The middle-sum lemma (Theorem A) - the flank envelope must collapse at `q'` per two levels
- STATEMENT.  In a literal word-legal `J`-run the `J-2` middles alternate between class `a`
  and class `b`, so with `k = floor((J-2)/2)`: middle sum `>= k q'` (`J` even),
  `>= k q' + a` (`J` odd).  Hence `Phi_J <= F_2(M) + s_min(q') - m_min(J)`: at `J = 5` the
  two flanks may sum to at most `F_2 - q'`, at `J = 6` to at most `F_2 + a - 2q'`.
- CALCULATES: how much room the two free flanks of a deep alignment have; and why the deep
  layers are the CHEAP ones.
- STATUS: proved from T1-T3.
- WHERE: `docs/novel/per-j-window-analogues.md` 1.1.
- LIMITS: literal middles only.

### The peel bound (Theorem D)
- STATEMENT.  Deleting either flank of a word-legal `J`-run leaves a word-legal `(J-1)`-run,
  so `Q*_J <= Q*_{J-1} + min(g_L, g_R)` at the argmax.  Equivalently `F_2 >= g_L + w` and
  `F_2 >= w + g_R`, so `span <= F_2 + min(g_L, g_R)`.
- CALCULATES: a hypothesis-free reduction of every depth to the one below; and read backwards,
  `Q*_3 > F_2 + s_min` FORCES min flank `> s_min`.
- STATUS: proved, hypothesis-free; the flank consequence asserted at all 27,570
  counterfactual machines.
- WHERE: `docs/novel/per-j-window-analogues.md` 1.4; `agents-shared.md` Lateral r29 (b).
- LIMITS: does NOT reach `J >= 4` - the free reduction gives only `2F_2 - q'`, short by
  exactly `F_2 - a` (R55's 2F wall).

### The par-trading residual `eps`
- STATEMENT.  For a realised legal word `v = u.x`, `eps(v) = Phi(u) - Phi(v) - x`, where
  `Phi` is the flank envelope (max `g_L + g_R` over occurrences).  Then
  `Delta_J = Delta_{J-1} - eps` along the maximising chain, `Delta_2 = 0`.  So
  `Delta_J = O(1)` uniformly in `J` **is exactly** "`eps` is `O(1)` per letter AND `L(M)` is
  bounded".  Decomposition lemma: `eps(v) = d - g_out` with `d = Phi(u) - x - g_kept >= 0` -
  `eps = O(1)` is a CANCELLATION, not a smallness (`d = 27`, `g_out = 28` at m31).
- CALCULATES: exactly what one more aligned strike costs the merged record.
- STATUS: identity proved; the decomposition lemma proved and asserted 30/30;
  `|eps| <= s_min` MEASURED 14/14 at literal cells and REFUTED 10/16 at padded cells;
  `max |eps| = 4` along maximising chains over 12 cells against `s_min` running 4..14.
- WHERE: `docs/novel/even-j-mechanism.md` 1.3, 1.4(b), 1.4(c), 7.1.
- LIMITS: the six failures all carry the padded letter `q'`; unproved in every direction.

### The `F_3` wall - when the padded alignment is worth a record
- STATEMENT.  `Phi(q') + q' <= F_3(M)` trivially.  At m31 it is EQUALITY: the `F_3(31) = 85`
  maximisers are `(18,37,30)`/`(30,37,18)` - **the old machine's depth-3 record has the padded
  letter as its middle**.  At every other machine m11..m37 the `F_3` maximiser's middle is not
  a legal letter of any class.  The excess `F_3 - (F_2 + s_min)` is `+1,+1,-3,-4,+1,0,+5,-7`
  at m11..m37: four machines exceed the increment budget at depth 3, and only m31's exceeding
  run is word-legal.
- CALCULATES: a decidable per-step condition (is the `F_3` maximiser's middle `0 mod q'`?)
  that predicts where the increment law fails, and by how much (`F_3 - F_2 - s_min`).
- STATUS: script-verified exact and gated (`f3_middles_r30.py`); the recurrence is a residue
  event with base rate `3/q'`, so it WILL recur - labelled as arithmetic luck per step, not a
  law.  Prediction on record: `F_3(37)`'s `(37,23,37)`, `F_3(43)`'s `(67,28,30)`,
  `F_3(47)`'s `(28,33,84)` have non-legal middles, so the law holds there.
- WHERE: `docs/novel/even-j-mechanism.md` 7.3; `agents-shared.md` Constructor r30 Headline 3.
- LIMITS: `Phi(12,37) = 39` and `Phi(37) = 48` each rest on ONE occurrence (a mirror pair);
  with that one stretch removed par trading holds at the padded letter too (`eps = +3`).

### The spectrum-plus-depth certificate, and its `A_kill` scope
- STATEMENT.  `F(M+q') <= max_{2 <= J <= J_max(M)} F_J(M)` with `J_max = A_kill + 1`, from
  the attainment theorem plus `Q*_J <= F_J` plus "emptiness is upward closed" (deleting a
  flank of a word-legal `J`-run leaves a word-legal `(J-1)`-run).  Margin at a step is exactly
  `F(M) + q' - F_{A_kill+1}(M)`.
- CALCULATES: (D) at a step from the OLD machine's spectrum over a finite depth range - no
  word list, no flank envelope, no realisability oracle.  Certified 41->43 (margin +16) and
  43->47 (margin +18).
- STATUS: proved; the table script-verified at ten steps.  **Every `A_kill <= 3` step
  certifies (+10 to +24); both failures (29->31 by -11, 47->53 by -6) and the single +3
  squeaker are the `A_kill >= 4` steps.**  Mechanism: one extra unit of `A_kill` admits one
  more level of the `F` ladder, costing 7-16 units, while the budget gains only `q' - q'_prev`
  (4 to 6).
- WHERE: `docs/novel/spectrum-depth-certificate.md` 1, 1.2, 1.4.
- LIMITS: circular below m59 - the `F_J(M)` values are exhaustive only because of
  deletion-ladder caps taken from `F` at machines ABOVE the step (Constructor r29
  self-correction, section 1.3).

### The deletion-ladder cap
- STATEMENT.  `F_j(M) <= F(M + {the next j-1 primes})`.  Proof: take the stretch realising
  `F_j(M)`; it has `j-1` interior openings; by CRT choose the phase tuple that puts interior
  `i` on a tooth of gear `q'_i` - all `j-1` interiors die at once.
- CALCULATES: free exact caps past the scan wall: `F_2(41) <= F(43) = 103`,
  `F_2(53) <= F(59) = 161`, `F_4(43) <= F(59) = 161`.  It is the `r`-gear generalisation of
  `F(M+q') >= F_2(M)`: `r` new gears buy `r` rungs of the `F_j` ladder, one designated strike
  each, because the `r` phases are independent.
- STATUS: proved (three lines, CRT); asserted at all 32 `(M,j)` pairs where both sides are
  known exactly (`deletion_ladder.py`), one equality (`F_2(17) = 25 = F(19)`).
- WHERE: `docs/novel/old-machine-spectrum.md` Corollary B; `docs/novel/spectrum-depth-certificate.md` 1.3.
- LIMITS: circular if used to bound the very quantity it takes as input at `j = 2`.

### The lap-phase transfer (compute a distant machine's alignments from a small one)
- STATEMENT.  `k |-> (k mod P, (k mod q_1, ..., k mod q_r))` is a bijection, and a maximal run
  of `M'`-openings is exactly a pair (run of consecutive `M`-openings, phase tuple) such that
  the endpoints and the chosen survivors avoid every new gear's teeth and every other interior
  `M`-opening is struck by at least one new gear.
- CALCULATES: `Q_J(M')` and `F_J(M')` exactly, on `M`'s period, at `1/(q_1...q_r)` of the
  cost.  It is the vehicle behind `F(59) = 161` (computed on machine 23's period, ratio
  `5.3e11`), `F_2(53) = 159`, `F_4(41) = 118`, `F_6(47) = 177`.
- STATUS: proved (CRT) + script-verified with two-sided anchors (m31 ladder
  `68/85/90/91/90/88` reproduced entrywise; `F_2(37) = 90`, `F_3(37) = 97` from three gears
  below).  Every witness CRT'd to the target machine and re-verified slot by slot.
- WHERE: `docs/novel/old-machine-spectrum.md` 1, Corollary A, 7(a).
- LIMITS: a CERTIFICATION is conditional on the span cap; a FAILURE is not.  A soundness trap
  is on record: with `r >= 2` the survivor-count lower bound is not monotone, so the walk must
  stop on its RUNNING MAXIMUM.

### The survivor identity (the whole low spectrum from one automaton)
- STATEMENT.  With the SKIP letter `SIGMA[(i,s),(i+2,s')] = d_i + d_{i+1}` (legal exactly when
  `d_i` alone is NOT a legal transition out of tooth `s`, so opening `i+1` SURVIVES, while
  `d_i + d_{i+1}` is legal): `F_2(M+q') = L (x) K* (x) SIGMA (x) K* (x) R`, and generally
  `F_j(M+q') = L (x) K* (x) (SIGMA (x) K*)^{j-1} (x) R`.
- CALCULATES: the whole low spectrum of the new machine from the old machine's word, with the
  automaton fixed and only the word growing.  Between two aligned runs there is EXACTLY ONE
  surviving opening (any other would itself be struck, contradicting maximality).
- STATUS: proved for every `j`; script-verified exactly at `j = 2` at six steps (11->13 ..
  29->31), matching Mechanic's independent lag-1 pair census; kernel-checked at 11->13
  (`Gen11Sound.generator_sound`, `F_1..F_4(13) <= 11,16,23,26`, machine 13's period nowhere
  in the derivation, independence gated by `DepAudit.lean`).
- WHERE: `docs/novel/survivor-generator.md`.
- LIMITS: `A_5` (realised 5-tuples) is needed for exactness where `A_4` suffices for the plain
  system - "one more order of history per skip".

### Saturation (a far gear always aligns the same way)
- STATEMENT.  If `q - 1 > F(M)` then `F(M+q) = F_2(M)` EXACTLY.  No chain of two or more
  strikes can exist (its interior gap would need to be `>= q-1 > F(M)`), and every adjacent
  pair of old gaps IS merged somewhere (each opening is struck in some lap).
- CALCULATES: the new record from `F_2(M)` alone, for every prime above the threshold.
  Corollary checked: `{5,7}` plus any of `q = 11,13,17,19,23,29,37,41,53` all give `F = 21`.
- STATUS: proved (elementary) + script-verified over 48 `(M,q)` pairs, zero violations.
- WHERE: `docs/novel/saturation-theorem.md`.
- LIMITS: the compliant regime is PROVABLY DISJOINT from the consecutive chain the route needs
  (`q' < F(M)` always along the ladder).

### The phase-reduction record law (the anchor-2,3,5 form)
- STATEMENT.  On ONE lower period, with the lower opening residues mod `g`:
  `D_g` = longest run of consecutive lower openings whose residues lie in one two-class set
  `{r, r+d_g}`; and `F_bc(M+g) + 1 = max over such runs (all phases r) of (gap before) +
  (run span) + (gap after)`, `F_bc` the blocked count, `F_bc + 1` the corpus max-gap record.
- CALCULATES: the record of the next machine as a maximum over `g` phases of "gap before +
  run span + gap after" on one lower period.  `F = 42` for `{5..29}` from 7,952,175 lower
  openings instead of a `6.5e9`-column period (819x smaller); the record law at 31/37/41
  (58, 88, 91) walked a `1.24e12`-column period with no array beyond machine 29.
- STATUS: script-verified exact at `{5..7}`..`{5..29}` and at 31/37 (full lower periods) and
  41 (36.9% deliberate partial, both answers still exact); KERNEL-CHECKED at machine 17 at
  BOTH ENDS (`AnchorRecord17.record_max` - phase table `16 16 18 18 18 16 18 18 16 15 16 18
  18 16 18 18 18`, max 18; `surv_shift`/`phase_is_machine`; `F17_eq_18`), but not derived one
  end from the other.
- WHERE: `docs/novel/anchor-235-layer-laws.md` (L3); `docs/proof-search/anchor-235.md` 9f;
  `agents-shared.md` Mechanic r29.
- LIMITS: the phase is not looped over in practice - mapping residues by `d^{-1}` turns
  "`{r, r+d}` for some `r`" into "two adjacent values", so one rolling max/min per length
  decides all `g` phases at once and the winning phase is read back.

### The nested next-opening formula
- STATEMENT.  With `M` the lower opening predicate and `H` the hit predicate, the enlarged
  machine's next opening after `x` is `nextM` iterated once past the run of hits:
  `nextG x = nextM^[k+1] x` when the first `k` lower openings after `x` are hits and the
  `(k+1)`-st is not.  Its term cap is `D_g`.
- CALCULATES: the next opening of the bigger machine directly, without materialising it;
  measured lazy cost = `1 + hops above gear 5`, mean 2.4 at `{5..19}` against 162 static terms.
- STATUS: kernel-checked as a theorem abstract in the machine
  (`AnchorChain.hop_zero`/`hop_iter`/`hop_one`); script-verified equal to the walk at every
  column on full periods `{5,7}`..`{5..19}`.
- WHERE: `docs/novel/anchor-235-layer-laws.md` 1; `docs/proof-search/anchor-235.md` 9f/9g.
- LIMITS: exponential in layers as a flat/nested form (`prod(1+D_g)` terms); the scan form is
  quadratic.

### `D_g = A_kill(M -> g)` - two constructs, one object
- STATEMENT.  The anchor line's chain depth and the twin route's kill arity are EQUAL at every
  gear where both exist (`D_17 = D_19 = 2`, `D_23 = 3`, `D_29 = 2`, `D_31 = 4`, `D_37 = 4`,
  `D_41 = 3`): both count co-deletable runs of consecutive `M`-openings, and word legality
  ("prefix-sum range `<= 1`") IS "all in one two-class set".  With R89, `D_g = A_kill = L+1`.
- CALCULATES: a streamed partial pass gives `D_g >= v`, a decided arity level gives
  `D_g <= A_kill`, and **the two halves meet** - `D_41 = 3` is exact from 0.1% coverage.
- STATUS: script-verified 7 for 7 by vehicles built four rounds apart in different languages;
  identity by argument, not kernel-checked.
- WHERE: `docs/novel/anchor-235-layer-laws.md` (L4); `agents-shared.md` Mechanic r29.
- LIMITS: `D_g` bounded is OPEN - Formalist's honest boundary: "a run alternates freely;
  `D_g` is a fact about lower gap SIZES", not an algebraic consequence.

---

## C. WHAT CAPS THE DEPTH OF AN ALIGNMENT

### Phase saturation (a gear with no admissible phase)
- STATEMENT.  Gear `q` strikes the residue pair `{a, a+s_q} mod q` for a free phase `a`,
  `s_q = -2*6^{-1} mod q`.  Define `FREE_q(X) = Z_q \ ((X mod q) u ((X - s_q) mod q))`.
  **If `FREE_q(X)` is empty for some gear `q <= y`, the word has NO occurrence anywhere.**
  `|FREE_q(X)| >= q - 2k`, so only gears `q < 2k` can fire - all the content is at gears
  5, 7, 11.
- CALCULATES: zero-by-arithmetic refutations with no solver; screened Constructor's m41
  arity-4 superset `4,239,676 -> 2,814,574` in seconds (gear 5 kills 780,486, gear 7 644,616).
- STATUS: proved (two lines); gated sound against the project's ENTIRE realised-word record
  (37 words at five steps, none wrongly zeroed; 0 false kills on all 291,675 realised m37
  4-tuples) and it reproduces every structural zero already on record.
- WHERE: `docs/novel/phase-saturation-arity.md`.
- LIMITS: it IS the corridor condition mod 35 by CRT, so it adds NOTHING to a corridor-built
  abstraction - it answered ZERO of the 27,197 superset-YES queries the 41->43 loop asks.

### The alternation ceiling (closed form, no solver)
- STATEMENT.  For the pure alternation `A_k = (s, q'-s, s, ...)`,
  `ceil(M -> q') = min{k : FREE_q(X(A_k)) = {} for some q <= y} - 1`, a closed form in
  `(q' mod q, s mod q)` for the two or three smallest gears.
- CALCULATES: an upper bound on any alternating chain per step: ceiling
  `6, 2, 2, 2, 5, 3, 3, 4` at 31->37 .. 61->67, dead gear 5 (7 at 53->59 and 61->67), against
  measured `A_kill = 4, 3, 3, 3, 5, 4`.
- STATUS: script-verified, gated; `A_kill(47->53) = 5` sat EXACTLY at its ceiling.
- WHERE: `docs/novel/phase-saturation-arity.md` Corollary; `agents-shared.md` Mechanic r26/r27.
- LIMITS: bounds the ALTERNATION only.  At every step whose ceiling is 2 or 3 the answer is
  `ceiling + 1` and the lifting word is PADDED - `(14,41)`, `(43,43)`, `(47,47)`,
  `(20,98,20) = (s, q'+(q'-s), s)`.  Four steps, recorded as a pattern, not a law.

### The literal cap (uniform in `q'`, from `q' mod 210` alone)
- STATEMENT.  A literal chain (built purely from the gear's own two teeth, with every member
  in the corridor `E mod 35`) never has more than 6 members, for every gear forever, and the
  exact cap is a function of `q' mod 210`: cap `2` on 24 classes, `3` on 4, `4` on 14, `6` on
  exactly `{37, 53, 83, 127, 157, 173}`.
- CALCULATES: the fuel cap at any step from one residue, with no census.
- STATUS: KERNEL-CHECKED both ways (`LiteralCap.no_run_seven`, `s_eq`,
  `literal_chain_le_six`, `cap_six_classes_sharp`; `LiteralCapTable.cap_table_maximal`,
  `cap_table_realized`, `cap_spectrum_counts`); script-verified against every prime `<= 5000`,
  0 mismatches.
- WHERE: `docs/novel/literal-cap.md`.
- LIMITS: LITERAL chains only.  Padded runs escape it; "killed runs are bounded by 6" is FALSE.
  Also: the cap is NOT a density fact - over all 1,225 `(t,s)` pairs mod 35 the run spectrum
  reaches 140.

### The Polignac cap (all even gaps)
- STATEMENT.  For gap `d = 2e` the literal-chain cap depends only on `gcd(e, 105)`: 6 for six
  of the eight classes, 10 for `gcd = 15`, 12 for `gcd = 105`.  **12 is the absolute ceiling
  over all Polignac configurations, for every gear, forever.**
- CALCULATES: the cap for any even difference from one gcd.
- STATUS: kernel-checked (`PolignacCap.lean`, `cap_gcd_*`, `capOf_le_twelve`); each cap sharp.
- WHERE: `docs/novel/polignac-cap.md`.
- LIMITS: literal chains; gear 3 FILTERS the candidate list rather than breaking a run (a
  recorded modelling trap the kernel caught).

### The bare-word uniform cap (the first bound on part of `L` that does not grow)
- STATEMENT.  A BARE word (every letter `a` or `b`) is forced by T3 to be one of the two
  alternations, and a realised word's prefix-sum offsets are OPENINGS of `M`, so they fit
  inside the exposed set of every gear - in particular gears 5 and 7, i.e. inside the corridor
  `E_35`.  Hence `L_bare(M) <= PSORD(q' mod 210) <= 5` at EVERY machine, uniformly.
  `PSORD` = 1 on 24 classes, 2 on 4, 3 on 14, **never 4**, 5 on `{37,53,83,127,157,173}`;
  `S = {PSORD <= 2}` has 28 of the 48 classes (density 7/12).
- CALCULATES: the bare half of the alignment depth at any machine, from `q' mod 210` alone.
  At m41 and m43 every bare decision is FREE: `PSORD = 1`, so gear 5 alone refutes
  `(14,29)`, `(29,14)`, `(16,31)`, `(31,16)` and their 3-letter extensions with no search.
- STATUS: PROVED (two lines + a 48-class enumeration); KERNEL-CHECKED
  (`BareAlt.bareAlt_inadmissible_iff`, `S_card = 28`, `psord_le_five`, `psord_ne_four`,
  `psord_eq_one_iff/_two_iff/_five_iff`, `fitsB_of_open`, `open_of_gapWord`, `no_bare_run`),
  instantiated at m23/m37/m41/m43; three vehicles sharing no code agree element for element.
  Also kernel-checked: `c in S <-> LiteralCapTable.capC c <= 3`, so the bare cap and the
  literal cap are ONE object at every class.
- WHERE: `docs/novel/bare-word-uniform-cap.md`; `agents-shared.md` Constructor/Formalist r31.
- LIMITS: bounds `L_bare`, NOT `L`.  At m37/m41/m43 `PSORD = 1` while `L = 2`; at m53
  `PSORD = 2` while `L = 3`.  The deep words there are provably not bare.

### The decomposition `L = max(L_bare, L_pad)`
- STATEMENT.  With `L_pad` the longest realised legal word using at least one NON-BARE letter,
  `L(M) = max(L_bare(M), L_pad(M))`, so requirement (B) is EXACTLY "`L_pad` bounded".
  Measured: `L_pad = 0,0,0,1,1,1,2,2,2,2,3,3` at m11..m53 - it takes every value 0..3 and it
  GROWS.  `L_pad(53) = 3` is DERIVED (theorem plus the recorded `L(53) = 3`) at a machine no
  census reaches.
- CALCULATES: splits the open crux; identifies the four machines where `L > L_bare`
  (m37, m41, m43, m53) as exactly the `S`-machines whose record is carried by a word
  containing the padded letter `q'`.
- STATUS: theorem (trivial) + measured row; `L_pad(47) = 3` measured
  (`(18,35,53)`, `(18,53,35)`, `(35,18,53)` realised; `(35,71,35)` undecided at `6e7` nodes).
- WHERE: `docs/novel/bare-word-uniform-cap.md` 4; `agents-shared.md` Constructor r31.
- LIMITS: nothing on record bounds `L_pad`.

### The spectrum bound on `L` (the theorem that retired (B) as posed)
- STATEMENT.  With `G = F(M+q')`, `T = floor((G-2)/q')`, `p` padded letters:
  `(SIMPLE) L(M) <= 2T + 1`, letter-aware `L <= 2T + 1 - p`;
  `(PARITY) L(M) <= max(2T, 2*floor((G - 2 - a_min)/q') + 1)`; i.e.
  **`L(M) <= 2 F(M+q')/q' + 1`**.  Proof: class minima are `a`, `b`, `q'`; T3 makes two
  consecutive nonzero letters sum to `>= q'`; the attainment theorem gives
  `span <= G - 2` (a `m`-letter word occupies `m+1` openings, plus one on each side).
- CALCULATES: the alignment depth from the record and the gear, metrically.  Corpus row
  (PARITY) `1,1,2,3,3,3,5,4,5,5,5,5` against `L = 1,1,1,2,1,3,3,2,2,2,4,3`; TIGHT at m11,
  m13, m29; beats EXPCAP at m19/m37/m41/m43/m53 (5 vs 18 at m37, 5 vs 21 at m53).
- STATUS: PROVED, unconditional given R68 and T3; script-verified at 12 corpus machines
  (173 gates) and 165,584 counterfactual machines with ZERO violations, including the
  family's `L = 5` member where (PARITY) equals 5 exactly.
- WHERE: `docs/novel/spectrum-bound-on-L.md`; `agents-shared.md` Lateral r31 item 84.
- LIMITS: `L` is `O(F/q')`, not `O(1)`; substituted into the chain it gives
  `G <= (q'(F_2 + c_A) - 4c_A)/(q' - 2c_A)` for `q' > 2c_A`, and (D) whenever
  `8F <= q'^2 - (F_2 - F + 12) q' + 16` - true at 8 of 13 corpus steps, failing only at the
  five small ones.  `c_A = 4` is a LITERAL-letter constant, so the closure is conditional on
  the open (A-pad).

### EXPCAP and the sub-machine lemma (the exposure half's own cap)
- STATEMENT.  A word of length `m` survives phase saturation at `M` **iff** it survives at the
  sub-machine `{g in M : g <= 2m+2}` (a gear `g` has `g` translates and the `m+1` points
  forbid at most `2(m+1)` of them).  `EXPCAP(M) = max{m : S_m > 0}`.
- CALCULATES: the exposure-only cap on alignment depth at any machine, from the small gears
  and the alphabet.  Row `1,1,1,4,2,3,5,18,13,10,5,21` at m11..m53 against
  `L = 1,1,1,2,1,3,3,2,2,2,4,3`.
- STATUS: proved; asserted numerically at every `(M, m)` cell at m11..m53.
- WHERE: `docs/novel/cover-half-counter-ladder.md` 1.2, 1.4.
- LIMITS: `EXPCAP - L` is NOT bounded along the ladder (16, 11, 8, 18 at m37/m41/m43/m53).
  The exposure half over-caps `L` by an arithmetic-selected amount.

### CORRCAP and where the corridor stops constraining
- STATEMENT.  `CORRCAP(q', F)` = the longest T3-legal word with values `<= F` whose prefix-sum
  walk stays inside `E mod 35` - the strongest cap gears 5 and 7 can EVER give.  It is
  `4, 2, 3, 5, 25, 25, 11, 5` at 19->23 .. 47->53 and **INFINITE from 53 -> 59 on**, and at
  every larger `F/q'`.  Mechanism: padded letters step by `j q' mod 35` and `gcd(q',35) = 1`,
  so once `F/q'` is large the steps fill `Z_35` and the corridor acquires a cycle.
- CALCULATES: the exact machine at which a bounded set of small gears stops capping alignment
  depth.  Since `F/q'` grows without bound, NO FIXED SET OF SMALL GEARS CAN EVER CAP THE
  ORDER AGAIN.
- STATUS: script-verified by an explicit automaton on the `35 x 3` corridor states with cycle
  detection (GATE B5, r31; R75's row reproduced 9/9).
- WHERE: `docs/novel/uniform-order-bound.md`; `docs/novel/bare-word-uniform-cap.md` (b).
- LIMITS: the term that makes it infinite is the ALPHABET SIZE `~3F/q'`; the bare alphabet is
  two letters at every machine forever, which is why `PSORD <= 5` is uniform.

### The uniform alternation order `A_relax <= 5`
- STATEMENT.  `A_relax(M) = min{m : one of the two m-letter alternations is NOT realised}`
  satisfies `A_relax(M) <= 5` for every `y >= 7`, and `<= 4` unless
  `q' = 37, 53, 83, 127, 157, 173 (mod 210)` - proved by phase saturation at gears 5 and 7
  alone, so the whole statement is a function of `q' mod 210` with no machine in it.
  **The six exceptional classes are exactly the litcap-6 classes** - phase saturation at
  `{5,7}` and the literal cap are the same arithmetic, differing only in the quantifier
  (litcap MAXIMISES over starting letters, the order MINIMISES).
- CALCULATES: a uniform cap on the candidate alternation cycle from one residue.
- STATUS: script-verified over all 48 classes and cross-checked by a direct sweep of every
  prime `q' < 20000` with all gears to 100; adding gears 11, 13 refutes nothing further
  (60/60 and 720/720 refinements stay at 5).  Kernel-checked as
  `AlternationOrder.ps_min_le_five`, `ps_min_five_iff`, `ps_min_counts 24/16/2/6`,
  `ps_max_eq_capC`.
- WHERE: `docs/novel/uniform-order-bound.md`; `agents-shared.md` Constructor r27, Formalist r29.
- LIMITS: `A_relax` tests ONE candidate cycle.  `N(M)` (the order at which `A_m` is acyclic)
  is `2,2,2,3,2,3,4,3` at m11..m37 and `3` at m41, and the extra order is bought by a PADDED
  cycle - the cycles that push `N` above `A_relax` are padded 2-cycles which die at order 3
  because T3-TRANSPARENCY IS NOT T3-LEGALITY once the run is long enough to see two literal
  letters.

### The killer profile: what actually kills a one-letter extension
- STATEMENT.  For every realised longest legal word and every T3-legal one-letter extension:
  `y*` = the smallest gear set whose OPEN constraint is needed.  The profile is BIMODAL and
  EMPTY IN THE MIDDLE.  Every decided kill is either **cover-only (`y* = 0`): no column of
  `M` blocks the punctured interior** - the pattern does not occur in `M` at all - or a
  CORRIDOR kill: `y* = 7` for the pure alternations `(10,19)` at m23 and `(12,25,12,25)` at
  m31 (whose blocked pattern DOES occur, gate-verified by a period scan), or `y* = 5` from m37
  on for literal and doubly-padded extensions, where five open points leave gear 5 no phase.
  **No extension at any machine was attributed to the open constraint of a gear above 7.**
- CALCULATES: which half of the realisability CSP is doing the work at each machine, and
  therefore where a proof of `L` bounded must go.  The corridor `{5,7}` bounds exactly the
  pure-alternation family; every OTHER extension is refuted by the cover half in its purest
  form (an `F_J`-type statement about `M`'s blocked runs, not a teeth statement about `q'`).
- STATUS: measured, exact CRT decisions at m19..m41; every extension refuted at the full
  machine, 0 undecided, 0 realised; cover-only verdicts at m19/m23 re-derived by a direct
  period scan.
- WHERE: `agents-shared.md` Mechanic r30 (b); `docs/novel/legal-word-length-mechanism.md`.
- LIMITS: m43 and m47 killer profiles NOT delivered (priced, next-round items); 2 at m37 and
  10 at m41 remain unattributed (refuted, but their relaxed instances did not decide).

### The length of the longest legal word is a density statistic; its last unit is not
- STATEMENT.  With the REAL class densities of the legal alphabet in `M`'s exact gap histogram
  and the T3 alternation transfer matrix (growth rate `p0 + sqrt(p+ p-)`), an
  independent-letter model predicts the longest legal run to WITHIN ONE UNIT at every scanned
  machine (3.7/3 at m29, 4.0/3 at m31, 4.0/2 at m37).  But the COUNT of legal runs tracks the
  same model only at short lengths and collapses at the top: 4 against 279 at m29 (`L = 3`),
  216 against 1,610 and 0 against 2.5 at m31, 27 against 10,500 already at length 2 at m37.
- CALCULATES: `L` splits into a histogram statement (the legal letters' class densities) plus
  a one-unit arithmetic collapse; the free screens (alphabet + spectrum + phase saturation)
  are exactly ONE LENGTH too generous at 7 of 8 next-prime cells (`V3 = V4 + 1`).
- STATUS: measured (exact censuses, exact CRT decisions, one model used as an instrument).
- WHERE: `docs/novel/legal-word-length-mechanism.md`; `agents-shared.md` Mechanic r30 (a).
- LIMITS: the next prime is USUALLY but not always the maximising gear (m23: `L_31 = 2 >
  L_29 = 1`; m37: `L_53 = 3 > L_41 = 2`); what sets `L_g` is the alphabet size, which `q'`
  usually maximises.

### The exact null for `L`
- STATEMENT.  Exact finite-automaton expectations of the longest legal run over `N = prod(q-2)`
  gaps.  **The class probability is the largest single suppression and it is NOT `3/q'`**: the
  legal-class probability under the machine's own gap distribution is 0.19-0.39 of `3/q'` at
  m11..m37 (0.105 at m37), a factor 2.5-9.5 below equidistribution, because gap values pile up
  below the smallest legal letter and `p0 = 0` at seven of eight machines.  Alternation costs
  a uniform 12-14%.  Dependence between consecutive gaps is the residue, factor 0.43-1.00,
  NOT monotone in the machine.
- CALCULATES: the right null - `I-actA / L` = 1.14, 1.31, 1.96 at m29, m31, m37 against
  4.3-5.2 for the equidistributed proxy.  The "18 against 4" gap at m47 is mostly the
  estimate, not the machine.
- STATUS: script-verified exact (two independent routes agreeing to three decimals);
  the m41..m47 rows are labelled PROXY.
- WHERE: `agents-shared.md` Harvester r30, "Follow-on: the null model for L, done exactly".
- LIMITS: at m13/m17/m23 the ORDER-1 Markov null already gives `E[L] = 1.00` exactly (the
  alternating literal pair `(+s,-s)` never occurs as adjacent gaps); at m19 the order-1 null
  moves only 3% of the way, so the dependence that limits `L` there is beyond lag 1.

---

## D. MIRROR - THE ONE EXACT SYMMETRY OF THE OPENINGS

### The involution (M0)
- STATEMENT.  Column `k` is struck iff some gear divides `6k-1` or `6k+1`, which is invariant
  under `k -> -k`.  So **the machine's opening set is exactly closed under negation**, `k = 0`
  is always an opening and (P odd) its ONLY fixed column; on indices the map is
  `o_t -> o_{N-t}`.
- CALCULATES: a free exact consistency check on every census, and a factor-2 saving on every
  reversal-invariant search.
- STATUS: kernel-checked for one gear at any period (`Mirror.mirror_gear`), instantiated at
  m11 (3 gears) and m29 (8 gears).
- WHERE: `docs/novel/mirror-parity-laws.md` 1.
- LIMITS: `Z/2` and nothing more (see below).

### The antipodal columns are open at every machine
- STATEMENT.  `P = 0 (mod q)`, so the antipodal column `(P+1)/2` reduces mod every gear to
  `(q+1)/2`; multiply by 6 and it is `3`, while `6(+-u) = +-1`.  It is a tooth only if
  `3 = +-1 (mod q)` - impossible for `q >= 5`.  **So `(P+-1)/2` are OPENINGS at every machine
  and the antipodal gap has length 1**, whence `W_1(g)` is EVEN for every `g >= 2`
  unconditionally, and the record gap NEVER occurs exactly once.
- CALCULATES: an opening exhibited by arithmetic at a machine whose period no kernel will see
  (`antipode_exposed29 : Exposed29 539141103`).
- STATUS: KERNEL-CHECKED (`Mirror.antipode_open`; five lines, no residues:
  `2s = P+1` gives `6s = 3P+3`, so the members are `3P+2` and `3P+4`).
- WHERE: `docs/novel/mirror-parity-laws.md` 7.3; `agents-shared.md` Lateral r26.
- LIMITS: one column (plus its mirror); it is where the machine's gaps are SHORTEST.

### The self-mirror stretch: address, span, and parity
- STATEMENT.  A depth-`j` run is self-mirror iff its endpoints sum to `0 (mod P)`, i.e. it is
  centred on column 0 (`j` even) or on the antipode (`j` odd).  `N` odd gives exactly ONE
  self-mirror run per depth, at index `t_j = -j/2 (mod N)`, with
  `j = 2i` even: span `= 2 o_i`; `j = 2i+1` odd: span `= P - 2 o_{M-i}`, `M = (N-1)/2`.
  COROLLARY `g_j* = j (mod 2)`: `W_j(g)` is EVEN for every `g` of the wrong parity with NO
  computation - half the entire spectrum, free.
- CALCULATES: `g_j*` scan-free at every machine from a few dozen sieved columns (table to m53,
  `j <= 12`); at depth 2 it is `2 d_0`, twice the FIRST gap.
- STATUS: proved; verified against the exact full-period `W_j` census at m11..m29 for every
  `j <= 12` - the odd column is exactly `{g_j*}`, no exceptions.  The counting half
  kernel-checked (`Mirror.even_card_involution`, `window_count_even`, `adjacent_equal_even`,
  `none_of_at_most_one`).
- WHERE: `docs/novel/mirror-parity-laws.md` 1, 7.2, 8.1.
- LIMITS: the LEVER (over an abstract index involution) is kernel-checked; the INSTANTIATION
  at a machine is built only at m11 (`Machine11.opSeq_mirror`, `window2_even`).

### The self-mirror stretch is NEVER word-legal at depth `>= 3`
- STATEMENT.  `J` ODD: its central middle is the antipodal gap, of length 1, and 1 is a legal
  letter only if `3 = +-1 mod q'` - impossible (`2u' = 3^{-1}`).  `J` EVEN `>= 4`: its two
  CENTRAL middles are both `d_0`, and T3 forbids two equal nonzero classes while
  `0 < d_0 < q'` forbids both being padded.  `J = 2`: no middles, so `(d_0, d_0)` IS legal -
  the one depth needing a hypothesis, and there it is exactly `d_0 != F`.
- CALCULATES: `R_J` is FIXED-POINT-FREE on the word-legal family at every `J >= 3`, so **every
  span count is EVEN with no exceptional class, no exception list and no census**; "at most one
  word-legal `J`-run exceeds the budget" proves there are NONE, unconditionally.
- STATUS: proved (elementary) + script-verified at m11..m23, `J = 2..7`, 185 assertion gates.
- WHERE: `docs/novel/mirror-parity-laws.md` 9.2.
- LIMITS: buys ONE UNIT (a factor of two), never four - the full symmetry group is `Z/2`.

### The mirror on records, and in transfer coordinates
- STATEMENT.  For a stretch of machine `y` at address `k`, span `s`, interior offsets `o_i`:
  `k' = (P - k - s) mod P` is an opening, its interior offsets are the reversed `s - o_i`, its
  flanks are the reversed flanks, residues map `r -> (P - r) mod q''`, and `k + k' + s = P`.
  In transfer coordinates `(k, c_q) -> (P0 - k - s, (P0 - c_q) mod q)` with marks reversed.
- CALCULATES: a factor 2 on every transfer sweep, and a PARITY CONSTRAINT - maximisers come in
  pairs, so **a search that has found one maximiser is provably incomplete and the partner's
  address is `P - k - s`**.
- STATUS: proved (one line) + script-verified on all 24 exact record stretches on file (150
  gates), partner always a DIFFERENT column; the two `F_2(59)` maximisers are an exact mirror
  pair INCLUDING their flanks (`y_A + y_B + 173 = P(59)`, kernel-checked as
  `CrtSlots.mirror_59`).
- WHERE: `docs/novel/mirror-parity-laws.md` 10.1, 10.2.
- LIMITS: no inequality on `Q*_J` or `F_J`.

### Word reversal, and the fixed-point criterion
- STATEMENT.  The mirror sends an occurrence of `w` at `k` to an occurrence of `reverse(w)` at
  `-(k + span w)`, bijectively, so `#occ(w) = #occ(reverse w)` exactly and realisability is
  reverse-invariant - kill words included, since both the openings and the teeth are
  negation-symmetric.  For a PALINDROMIC tuple of span `s`, `#occ(w)` is ODD iff `w` occurs at
  the single candidate address `k_w = -s/2 (mod P)` - an `O(#gears)` test.  At `w = (g,g)` it
  forces `g = k_1 = d_0`.
- CALCULATES: decide one word per reverse class and copy the verdict (`w[::-1] in decided`).
  Audit of the project's own logs: 82 word decisions, every reverse pair agreeing, and
  **12,877 s of 27,946 s (46%) had been spent deciding the SECOND member of a reverse pair.**
- STATUS: proved; gated on the exact 4-tuple dictionaries at m23/29/31/37 (reverse-closed,
  15,696/45,854/115,193/291,675) and on the two CRT TRANSFER supersets (2,435,140 and
  4,239,676 tuples), which had no a-priori reason to inherit the symmetry.
- WHERE: `docs/novel/mirror-parity-laws.md` 7.4, 7.6, 8.6.
- LIMITS: word reversal is the SAME involution, not a second one (r28 self-correction).

### `F_2 >= 2 d_0`, and `d_0` in closed form
- STATEMENT.  By the mirror the two gaps around column 0 are `(d_0, d_0)`, so `F_2 >= 2 d_0` at
  every symmetric two-tooth sieve.  And the wrap gap of a period equals the FIRST gap:
  `wrap = P - x_{N-1} = x_1 = d_0`, `= 3,3,5,5,5,7,7,7,10` at m11..m41.
- CALCULATES: a lower bound on `F_2` from one sieved column, and the exact missing gap that a
  linearly-closed census drops (which is how a real defect in `gap_pair_hist.csv` was found and
  repaired without a rescan).
- STATUS: theorem + closed form; gated at 15,217 counterfactual machines.
- WHERE: `docs/novel/mirror-parity-laws.md` 10.3; `agents-shared.md` Mechanic r25 / Lateral r25.
- LIMITS: on the counterfactual family the ONLY depth-2 failure of (D) in 14,616 exhaustively
  enumerated machines is exactly this stretch (`d_0 = 25` against `F = 26`).

### The machine's full symmetry group is `Z/2`, exactly
- STATEMENT.  The affine maps preserving the opening set are exactly the `2^m` multiplications
  by `c = +-1 mod every gear` (`b = 0` forced), of which only `c = +-1 mod P` preserves
  ADJACENCY of openings; dropping affineness, the only rotations/reflections of `Z_P`
  preserving the openings are the identity and the mirror.  Fixed-point counts:
  `#fix(sigma_S) = N / prod_{q in S}(q-2)`, so exactly ONE of the `2^n - 1` sign involutions
  has a single fixed point.
- CALCULATES: the ceiling on any parity lever - it is worth EXACTLY ONE UNIT, a factor of two,
  never four; there is no mod-4 version to hope for from any symmetry of the machine.
- STATUS: proved + brute-forced over all 92,400 affine maps at m11 and all `2P` rotations and
  reflections at m11/m13.
- WHERE: `docs/novel/mirror-parity-laws.md` 7.1 (Theorems A, A2), 8.4.
- LIMITS: a finer parity must come from something that is NOT a symmetry of the opening set.

---

## E. WHERE OPENINGS SIT: EXACT COUNTS, PINNING, AND HISTOGRAM STRUCTURE

### Column 0 is always an opening; the sharing law
- STATEMENT.  Gear `q` strikes `+-6^{-1} mod q`, never 0, so column 0 is an opening at every
  machine.  Survivors per full period `= prod(q-2)` regardless of phases - **sharing moves
  WHERE the waste lands, never HOW MANY survive.**
- CALCULATES: the period identity every census must satisfy (asserted before writing any
  full-period CSV, standing rule 25).
- STATUS: elementary; asserted at every machine.
- WHERE: `agents-shared.md` "Established laws"; `docs/novel/tooth-counterfactual-percentile.md` 1.
- LIMITS: a count, not a position.

### Tooth-sharing pinning (a twin gear pair's double strikes)
- STATEMENT.  For a twin pair `(p, p+2)` both gears carry the SAME tooth value
  `u' = (p+1)/6`, and the columns where BOTH strike are exactly the four CRT classes
  `k = +u', -u', +u'(p+1), -u'(p+1) (mod p(p+2))`.  At `k = u'(p+1)` the lower member is
  `6u'(p+1) - 1 = (p+1)^2 - 1 = p(p+2)` exactly - the twin product.  Consequence: every twin
  gear pair wastes at least two strikes per window on already-dead columns, deterministically,
  at every level.
- CALCULATES: the exact positions of the double-strike coincidences of any twin pair.
- STATUS: proved (elementary); script-verified 60/60 twin pairs to 2000.  General form
  (R6): slot `k` is struck by both `q` and `q'` iff `36k^2 = 1 (mod qq')`.
- WHERE: `docs/novel/tooth-sharing-pinning.md`.
- LIMITS: honest negative already recorded - the wasted strikes land on already-decided
  columns, so tooth-sharing COUNT alone gains only `O(T(y))` per window.

### The corridor and its completeness
- STATEMENT.  `E_35 = {0,2,3,5,7,10,12,17,18,23,25,28,30,32,33}`, `|E_35| = 15 = 3 x 5`, is
  the set of residues mod 35 left open by gears 5 and 7.  A shape with `n` openings can be
  blocked by gear `q` only if `q <= 2n`, so **for `n <= 3` openings the mod-35 test IS the
  entire corridor** - verdicts complete over ALL moduli, not a mod-35 shadow.  By CRT, "fits
  in `E_5` and fits in `E_7`" is equivalent to "fits in `E_35`" (asserted, 4,186 instances).
- CALCULATES: machine-free permanent verdicts for small shapes: state a shape as a step list,
  compute `carrier`, get a verdict.
- STATUS: kernel-checked (`Corridor.exposed_iff_mem`, `endpoint_law`, `adjacency_law`,
  `forbidden_pairs_count` - 294 of the 1,225 gap pairs mod 35 jointly infeasible,
  `no_chain_of_forbidden`); completeness lemma script-verified.
- WHERE: `docs/novel/corridor-law.md`.
- LIMITS: `n <= 3` for completeness (`n <= 5` claimed in one place with the `q <= 2n` bound).

### The corridor law: adjacent equal padded links forbidden for 12 of 24 classes
- STATEMENT.  Two adjacent equal padded links of gear `q'` need openings at `r, r+g, r+2g`
  with `g = q' mod 35`, a 3-term AP inside `E_35`.  This is IMPOSSIBLE for exactly the 12
  classes `{1,4,6,9,11,16,19,24,26,29,31,34}` mod 35, and there is a perfect DICHOTOMY:
  the equal shape `(1,1)` is infeasible iff both unequal shapes `(1,2)`/`(2,1)` are feasible.
- CALCULATES: a permanent residue criterion (the r14 padding lemma's spectrum threshold
  expires at 37->41; this never expires).  Instance: `no_adjacent_padded_41` - at 37->41
  (`g = 6`) there are ZERO solutions.
- STATUS: KERNEL-CHECKED (`TierA.equal_padding_forbidden_classes`, `_card = 12`,
  `padding_shape_dichotomy`, `no_adjacent_equal_padded`).
- WHERE: `docs/novel/corridor-law.md`.
- LIMITS: adjacent EQUAL padded links only; marks 41->43 as the first step with no obstruction
  of any kind.

### The corridor resonance: extreme gaps are phase-locked mod 35
- STATEMENT.  Large gaps RECUR at fixed column distances: multiples of 35.  The
  slot-separation autocorrelation of big-gap left endpoints is `3.2-4.4x` flat at separations
  35, 70, 105 (0.17-1.3 at neighbouring separations, and separation 70 exceeds separation 35 at
  every machine); the left endpoints are PINNED to a few residues mod 35, and the SAME classes
  are rich at every machine (m17/m19: `10,12,17,18` in an exact four-way tie).
- CALCULATES: where the machine's biggest gaps sit relative to one another, mod 35.
- STATUS: measured (full-period exact counts, five machines).
- WHERE: `docs/novel/corridor-resonance.md`; closed form in
  `docs/novel/corridor-eigenvalue-closed-form.md` (the corridor-phase chain's whole spectrum
  is the image of the roots of unity under a Moebius map with one parameter `rho`).
- LIMITS: the closed form has one modelling step (columns independent given their phases).

### The depth-sum identity (how often two openings sit at lag `g`)
- STATEMENT.  With `c_q(g) = #{r mod q : r and r+g both open to gear q}` (closed form:
  `q-2` if `q | g`; `q-3` if `3g = +-1 mod q`; `q-4` otherwise),
  `sum_{j >= 1} W_j(g) = prod_{q in Q} c_q(g)` for every `g >= 1`.
- CALCULATES: an exact CRT count of ordered opening pairs at every lag, hence the
  depth-uniform bound `W_j(g) <= prod_q c_q(g)` with no period scan, uniform in depth, at
  machines far beyond any scan.
- STATUS: PROVED (one line - every ordered pair of openings at lag `g` is the endpoint pair of
  exactly one run) + script-verified exact at m11..m29 for all `g = 1..64`; both halves
  kernel-checked at m13 (`DepthSum.window_depth_unique`, `depth_partition`,
  `depth_sum_at_13`).  **PRIOR ART: this is Holt, arXiv:2502.20470, Corollary 1** specialised
  to one constellation - the novelty label was wrong, the identity and the derivation stand.
- WHERE: `docs/novel/depth-sum-identity.md`.
- LIMITS: the glue (the `Finset` re-indexing between period window starts and residues) is the
  only kernel step still missing.

### The paired Holt recursion: the autocorrelation IS the transfer diagonal
- STATEMENT.  `n_g(M+q') = sum over words w with sum(w) = g of coef(w) n_w(M)`, with
  `coef(w) = #{r in Z_{q'} : r not in T, r + sum(w) not in T, r + sigma_i in T for every
  interior i}` - position-free, so the map is linear with universal coefficients.  At `j = 1`,
  `coef(g) = q'-2` if `q' | g`, `q'-3` if `g = +-2u`, `q'-4` otherwise: **exactly the
  exposed-set autocorrelation `c_{q'}(g)`.**  Generic word survival `q' - 2(j+1)`, against
  Holt's one-class `q' - j - 1`: the paired system contracts TWICE as fast per unit of word
  length.
- CALCULATES: the whole new gap histogram from the old word census, exactly.
- STATUS: script-verified at four rungs including the full WORD-census level (6,714 words at
  5005->85085, 10,489 at 85085->1616615); paper proof elementary.  Correction on record: the
  `q' - 2j - 2` diagonal is a consequence of Holt's point count in his 2025 general dynamics.
- WHERE: `docs/novel/paired-holt-recursion.md`.
- LIMITS: population dynamics, not extremes.

### The record's shape and its genealogy
- STATEMENT.  Of the 132 stretches attaining `F_j` at m19/m23/m29 (full-period census), ZERO
  are literal and ZERO are qualifying: **the attaining shape is always two near-maximal flanks
  with the machine's SMALLEST gaps interior** - which the interior floor `>= 2u'` forbids
  exactly.  And ancestry: at 7 of 8 steps the ancestor stretch is a RUNNER-UP of `M` (deficit
  2-14), at 7 of 8 its largest gap was itself merged one machine down, and the genealogy runs
  2-5 generations deep, every level a runner-up.  For the `F_J` records the largest gap is
  merged one machine down in 12 of 12.  The m31 record's whole tree: `58 <- m29 [18,10,30] <-
  30 = m23 [7,23] <- 23 = m19 [5,15,3] <- 15 = m17 [2,6,7] <- m13 [5,1]`.
- CALCULATES: "records do not recruit records" - FALSE in the spectrum sense (1 of 8), TRUE in
  the depth sense (7 of 8).  Ancestor RANK among `M`'s own `J`-runs by span is 8-219, so
  ancestry is not deterministic by any top-`k` statistic.
- STATUS: script-verified; every record column re-verified at its own machine.
- WHERE: `docs/novel/suppression-law.md` Statement C; `agents-shared.md` Mechanic r30 (c).
- LIMITS: **`F(M+q')` is scan-free exactly when the record is carried at depth `>= 3`** - which
  it is at 31->37, 37->41, 47->53, 53->59; at short words the flank order statistic `Phi(w)` is
  a scan quantity.

### The dictionary-monotonicity (depth-0) lemma
- STATEMENT.  For every prime `q' > 2(m+1)`, `D_m(M) subset D_m(M + q')`: a realised
  `m`-tuple of consecutive gaps SURVIVES adding a gear, because the pattern forbids at most
  `2(m+1) < q'` phases and CRT supplies a lap with an admissible one.  The hypothesis is
  SHARP: at `q' = 17, 19` the first failure is at exactly the first `m` the proof does not
  cover.
- CALCULATES: 145,907 of 874,087 reverse classes of the m41 arity-4 superset are YES BY
  THEOREM (16.7%), at every span, with no solver.
- STATUS: PROVED (elementary) + script-verified at seven pairs and arities 2,3,4,5.  PRIOR ART
  (cite it): Ziller 2020, arXiv:2007.01808, Prop. 2.7 is the one-class arity-1 case, framing
  attributed to de Polignac 1849.
- WHERE: `docs/novel/dictionary-monotonicity-onset.md` (a).
- LIMITS: arity-1 in one class is known; the two-class arity-`m` statement with its sharp
  hypothesis is the delta.

### The inflation-onset law (where a transfer's alignments stop being faithful)
- STATEMENT.  With `q''` the next prime after `q'`,
  `onset(M -> q') = min span of [ (D_4(q'') \ D_4(q')) INTERSECT the transfer's own
  emissions ]` - **the transfer first over-generates exactly where the NEXT machine's new
  repertoire begins.**  Measured 13, 15, 17, 25, 31, 41, 53, 68 at 11->13 .. 37->41.
- CALCULATES: the "certainly exact" region of a superset before any decision is paid for; and
  it PREDICTED `onset(37->41) = 68` out of sample from the m41 shard alone.
- STATUS: measured (refined form 31 of 31 across six output arities and two screens; the
  simple form 16 of 25); causal version 8/8 (every tuple refuted AT the onset span is realised
  at the next machine).
- WHERE: `docs/novel/dictionary-monotonicity-onset.md` (b).
- LIMITS: not proved; the mechanism is closure failure x phase saturation x a near-constant
  factor (`onset/Y_5` in a band of width 0.042 at the four largest machines) and the third
  factor is unexplained.

### The walk screen
- STATEMENT.  Every point of the transfer's WALK - the struck interiors included - is an
  `M`-opening, so the WHOLE WALK must have an admissible phase at every gear `q <= y`.  Sound,
  strictly stronger than the emission screen, and a prefix prune rather than a post-filter.
- CALCULATES: superset sizes `2,435,140 -> 1,182,475 (emission) -> 1,153,814 (walk)` at
  31->37; it SUBSUMES the emission screen at all six steps, and the removals land almost
  entirely above span 100 (the expensive bands).
- STATUS: proved sound; asserted at every step that no realised tuple is removed.
- WHERE: `docs/novel/dictionary-monotonicity-onset.md` 3, "A tool consequence".
- LIMITS: 2.4-11.7% of the superset.

### The two-`n` gap reordering (openings in odometer order)
- STATEMENT.  Sort the openings by CRT phase vector (lex).  The adjacent differences take
  EXACTLY `2n` distinct values at `n` gears, with
  `mult(D(i,d)) = s_i(d) prod_{i'<i}(q_{i'}-2)`, `s_i(2) = 1` if `q_i in {5,7}` else 2.
  The reason is that CRT-lex order IS the mixed-radix odometer and `d_i = 2` at every gear
  **because the teeth are NEVER adjacent** (adjacency needs `3 = +-1 mod q`).  The cyclic
  closure is FREE for the machine's own teeth.
- CALCULATES: an explicit bijection `Phi: [0,N) -> O` (a generalised van der Corput point
  set), so `F` is literally `P` times a digital sequence's dispersion.
- STATUS: PROVED; prior art checked and recorded as KNOWN IN MECHANISM (Langevin's
  lex-successor theorem; Fried-Sos).
- WHERE: `docs/novel/two-n-gap-reordering.md`; `agents-shared.md` Lateral r27 item 1.
- LIMITS: CLOSED LINE.  Over 60 admissible re-choices of the teeth the count stays `2n = 8`
  while `F` ranges over `[10,18]` - **the coordinates discard exactly the arithmetic `F`
  depends on.**

---

## F. WHEN AN OPENING LANDS INSIDE THE WINDOW / THE SECTION

### The horizon theorem and the layer law
- STATEMENT.  Gears `< y` decide the open interior `(y, y^2)` exactly; the top gear's unique
  acts are boundary only.  One layer's novelty is `{y^2} u {y*c : c prime in (y, y'^2/y)}`.
- CALCULATES: which gear can possibly strike a given column of the window.
- STATUS: session-proven (in the "Established laws" list).
- WHERE: `agents-shared.md` "Established laws".
- LIMITS: an existence/attribution statement, not a positional one.

### Inside the section the machine below is exact, and the new gear is silent
- STATEMENT.  Every composite below `q'^2` has a prime factor `<= p`.  So inside the section
  `p -> q'` the gears `5..p` are EXACT - **the periodic word of `m_p` restricted to the section
  IS the twin-prime indicator there** - and the new gear `q'` does nothing in its own section
  (its first strike is `q'^2`, the far edge).  The section attributed to machine `q'` is the
  last stretch where the PREVIOUS machine is still telling the truth.  The previous gear `p`
  enters only through `p*m` with `m` prime in `(p, q'^2/p)`; `p^2` is the excluded near edge,
  so the candidates are `p q'` and at most two more.
- CALCULATES: the section's word from the machine below, with no new arithmetic; and it says
  the newest gear contributes nothing to its own new territory.
- STATUS: forced (proved) + measured over 667 sections to `q' = 5003`.  Measured: gear `p` is
  the death rung of at most 3 columns in its section, and of NONE at 77% of sections with
  `q' >= 500`.
- WHERE: `docs/proof-search/word-tree.md` 7.1, 7.2 (S4).
- LIMITS: as NUMBER-strikes, `p` reaches up to six (`p q', p q'_2, ...`); the "at most three"
  is about DEATH RUNGS.

### The section's blocked word is the divisibility lattice
- STATEMENT.  For every gear `s <= p` the strikes in the section are
  `K_s(p -> q') = s x { m in (p^2/s, q'^2/s) : no prime factor below s }` - the set on the
  right is the OPEN WORD of the sub-machine with gears below `s`, read at NUMBERS.  So
  `blocked(p -> q') = union over s of s * open_{<s}((p^2/s, q'^2/s))`, and **a new twin is a
  column that no such scaled open word reaches on either side.**
- CALCULATES: the section as a stitch of the open words of every smaller machine, each scaled
  by its next gear; "gear `s` consumes the numbers of the section at scale `p/sqrt(s)`", with
  a feeder table naming which lower section each gear eats from.
- STATUS: gated over 666 sections (B1), with the per-gear bands contiguous.
- WHERE: `docs/proof-search/word-tree.md` 9.2 (`section_ab_r29.py`).
- LIMITS: the pre-registered "1 to 3 of them" for gear `p`'s own strikes was REFUTED (up to
  six at wide rungs).  **No section-specific feature of gear interactions was found:** both
  which vectors survive (A) and where the strikes come from (B) reduce to CRT and to the
  smaller machines' open words.

### The residue vectors that enable a new opening in the section are uniform
- STATEMENT.  Over the 122,546 new twins with `q' >= 1000`, the residue classes are UNIFORM
  over the tooth-avoiding classes to total variation 0.0026 (mod 5), 0.0033 (mod 35, 15 open
  classes, least 0.0658 most 0.0675 against 0.0667) and 0.0097 (mod 385, 135 classes).
  **The enabling alignment is the CRT product, nothing finer** - there is no preferred
  combination and no gear whose position in its own word makes a new twin more or less likely.
- CALCULATES: for the proof, "killing twins for ever would need a rung from which no
  tooth-avoiding vector lands in the section, and the vectors that land are the generic ones -
  the kill would have to remove every class at once, not a pattern."
- STATUS: measured over 667 sections (`twin_provenance_r29.py`, V1), 8/10 gates.
- WHERE: `docs/proof-search/word-tree.md` 8.2, 8.3(a); `section_ab_r29.py` A1 (carry-position
  multiset within TV 0.043 of the iid-uniform model of the open vectors).
- LIMITS: a distributional statement, not an existence proof.

### The provenance of a new opening: two independent sides, `ln ln`-slow depth, a big gear on top
- STATEMENT.  (i) The two sides of a new twin are INDEPENDENT (framing-pair joint within TV
  0.024 of the product of its marginals; left marginal `5: 0.665, 7: 0.134, 11: 0.045, ...`,
  `(5,5)` alone 44%).  (ii) The number of gears that touch a new twin's word grows like the
  RECORDS of an iid Mertens sequence: 2.2 at `q' < 100`, 3.7 at `q' ~ 5000` - about one more
  gear per factor 10 in `q'`, model within 6%.  (iii) **The top of a new twin's provenance is
  a gear `> p/2` about HALF the time** (46-48% at `q' >= 1000`), because a new twin lives at
  numbers `~p^2` where near-twins have density of the same order as twins.
- CALCULATES: the provenance is the twin's residue vector plus the records of its flank rungs -
  the first uniform by CRT, the second a Mertens records process.
- STATUS: measured, 130,664 new twins over 667 sections.  (iii) is a REFUTATION of the
  pre-registered 5-25% carried over from the window average (11.7%) - "the section IS the top
  of the window, and the old window is not representative of new twins."
- WHERE: `docs/proof-search/word-tree.md` 8.2 (V2, V3, V4), 8.3.
- LIMITS: `q'` itself appears in NO provenance - the section's own gear strikes nothing in its
  section, so a new twin's provenance ends at gear `p`.

### The real teeth sit on the densest class (the section-view counterfactual)
- STATEMENT.  Pooled over sections `q' >= 1000`, moving gear 13's teeth to ANY other `v`
  leaves 3.6-3.8% MORE survivors than the real teeth; gear 7's 6.4%; and all moved positions
  agree with each other - **the real class is the odd one out.**  Mechanism, parameter-free:
  among the columns no gear but `s*` touches, the tooth class is richer than every other class
  by 1.160 (`s* = 7`), 1.202 (13), 1.273 (31), and the cofactor model `ln n / ln(n/s*)` gives
  1.138, 1.190, 1.271.  A number `s* x m` in the tooth class is clean iff its cofactor `m`,
  smaller by the factor `s*`, is prime - likelier than for a full-size number.
- CALCULATES: **the real teeth of every gear sit exactly on the residue class where the relaxed
  machine's survivors are densest**, so the real machine removes more of them than any
  counterfactual teeth would, in every section, by `ln n / ln(n/s*)`.
- STATUS: measured, pre-registered PREDICTION REFUTED (the section CAN tell real teeth from
  moved ones) - `section_c_r29.py`, log `research/data/r29/section_c.log`.
- WHERE: `docs/proof-search/word-tree.md` 9.3.
- LIMITS: this is the section-view face of the period result (the real machine a low-F outlier
  under moved teeth); it explains the SIGN, not the size.

### Positional existence: the run at `q'^2/6` is short
- STATEMENT.  The worst run of the machine's pattern is longer than the current section
  already at `q = 17` (17 against a section of 12; 144 runs `>= W` covering 2.3% of the
  period), and `F(59) = 161` against a section of 40.  **So existence in the section is
  POSITIONAL**: the run the pattern happens to have AT `q'^2/6` is short - to `q' = 5000` at
  most 0.663 of the section (at `q' = 137`), first twin a MEDIAN 18 columns past `q'^2`, max
  264 at `q' = 4637.`  The worst runs sit deep in the period in MIRROR PAIRS (positions `k` and
  `P - k`, fractions 0.3-0.7 or at the period's ends), **never at the window.**
- CALCULATES: exactly where the machine's record stretch is relative to the window, and why
  the section-only statement is stronger than the conjecture.
- STATUS: measured to `q' = 5000`; the full-period worst-run table (blocked-slot counts
  `4,6,10,17,24,33` at `q = 7..23`, matching the corpus ladder plus one) is exact.
- WHERE: `docs/proof-search/anchor-235.md` 7; `docs/proof-search/word-tree.md` 7.3.
- LIMITS: a dead section would be a twin gap `>= 4 sqrt(x)`; the section framing shows the
  first obstacle is already a `sqrt(x)`-size twin gap.

### Against the WHOLE window the position drops out
- STATEMENT.  `F(q) < W(q) - q/6` forces an opening in the window **whatever the pattern does
  at `q^2`**.  Measured `F/W = 0.25` flat from `q = 5` to 53; `F(59) = 161` against `W = 620`.
  And if (D) holds at every rung then `F(y) <= sum_{q<=y} q ~ y^2/(2 ln y)` against
  `W ~ y^2/6`, so `F/W <= 3/ln y < 1` for `y > 20`: an opening in every window, twins infinite.
- CALCULATES: the reduction of "an opening lands inside the window" to the record law plus (D),
  with the positional question removed entirely.
- STATUS: the ratio is measured; the implication is arithmetic; (D) holds at every computable
  rung through 59 (203 against budget 204 at 53->59).
- WHERE: `docs/proof-search/anchor-235.md` 7; `agents-shared.md` (the kernel route).
- LIMITS: this is the whole open part - (D) at every rung.

### Section aggregates (no dead section on record)
- STATEMENT.  Every one of 667 sections to `q' = 5003` holds a twin; the minimum count rises
  2, 6, 10, 21, 51 across the bands and the minimum is ALWAYS at a gap-2 rung (whose section
  is the shortest, `|S| = (4q'-4)/6`).  `G_S/|S|` (largest gap between twins in the section,
  or edge to nearest twin) is below 1 everywhere, max 0.684 at 29->31, falling
  0.352, 0.221, 0.177, 0.092 by band - like `ln^2 q' / q'`.  Twin counts are
  Hardy-Littlewood: observed/predicted 1.0028 over `1000 <= q' <= 5003`.
- CALCULATES: the section is a Mertens word at every scale, with sections differing from one
  another only by scale.
- STATUS: measured (`section_probe_r29.py`, 8 of 9 gates; the failure S4 is a bookkeeping
  error, `p^2` is the excluded near edge).
- WHERE: `docs/proof-search/word-tree.md` 7.2, 7.3.
- LIMITS: "nothing here is provable by the machine: twin gaps are unbounded in principle and
  the `ln^2` scale is heuristic."

### The blocked run inside a section is the generic Mertens tree
- STATEMENT.  Depth (number of distinct death rungs) 6.6 -> 36.8 across the bands while run
  length grows 15.6 -> 197.8; single-kill levels are 58-63% of the depth and the top
  single-kill chain 46-48% of the depth **in every band from `q' = 5` to 5003**.  Pooled over
  the 502 sections with `q' >= 1000` the top five levels are single-kill in 100% of trees.
  In tuple coordinates: 60% of merges are EXTENSIONS (the strike lands beside an already
  blocked column) and 40% are JOINS, in every band; the median join ratio is exactly 1/2
  through the whole middle of the tree (the `5,7` comb leaves pieces of lengths 1, 2, 4 only);
  the top of the tree is UNBALANCED (last merges join pieces in ratio ~1:3).
- CALCULATES: how a stretch with no openings is assembled - a chain of one-column binary
  merges at the top, the `5,7` comb at the bottom, sealed when a near-twin column closes the
  gap between a large piece and one a third its size.
- STATUS: measured, exploratory (not pre-registered), 667 sections
  (`section_mechanism_r29.py` per the rerun plan; trees in `tuple_tree_r29.py`).
- WHERE: `docs/proof-search/word-tree.md` 7.4, 7.5; `docs/proof-search/section-rerun-plan.md`.
- LIMITS: no section-specific feature from the gear that owns the section; nothing repeats at
  the tuple level (no top 3-tuple pattern reaches 3% of sections) - only the statistics are
  universal.

---

## G. THE INCREMENT LAW AND THE COUNTERFACTUAL FAMILY

### The increment law at literal steps
- STATEMENT.  `F(M+q') - F_2(M) <= s_min(q') = min(2u', q'-2u')`.  Reading, labelled
  hypothesis: `2u'` is the smallest positive LEGAL letter, so the law says "one more aligned
  link buys at most one small letter over the old two-gap maximum, unless the link is padded
  (worth a full `q'`)."
- CALCULATES: an upper bound on the new record from the old machine's depth-2 record plus one
  residue.  Holds at ELEVEN of the twelve testable steps, failing only at the padded 31->37 by
  +8; and it was confirmed OUT OF SAMPLE at 53->59 (predicts `F(59) <= 179`, measured `<= 178`,
  now `= 161`).
- STATUS: KERNEL-CHECKED at all six literal steps, BOTH halves, hypothesis-free
  (`Increment.increment_law_literal_steps`); the lower halves are CRT columns of the real
  machine (e.g. `F_2(29) >= 55` from the single column 858386140 in 35 s).  Sharpness also
  kernel-checked: `f2_19_sharp`, `f2_23_sharp`, `f2_29_sharp`.
- WHERE: `agents-shared.md` Formalist r28 Headline 1; LP thread r27/r28/r29.
- LIMITS: the base cases are kernel facts; the INDUCTION STEP is not, and the LP vehicle
  cannot supply it (cost is a primorial in the number of held gears).  The quantity that
  decides certifiability is `W_inc - F(q')`, negative at EXACTLY ONE corpus step (the padded
  31->37, where the increment width asks for something FALSE since `F(37) = 88 > 80`).

### The triple inequality, the free flank reduction, and `Delta_3`
- STATEMENT.  Both `(g_L + w)` and `(w + g_R)` are 2-runs of `M`, so with NO hypothesis
  `g_L + w + g_R <= F_2(M) + min(g_L, g_R)`: the triple inequality is AUTOMATIC whenever the
  smaller flank is `<= s_min`, which discharges 6 of the 8 steps outright.  `Delta_3` =
  `-3, 2, 0, 2, 4, 3, 2, 0` - BOUNDED BY A CONSTANT while `s_min` grows linearly.
- CALCULATES: the shape to aim at is `Delta_3 = O(1)`, not `Delta_3 <= s_min`.
- STATUS: proved (the reduction) + script-verified (the table), literal middles separated from
  padded ones - the literal triple inequality holds at EVERY step including 31->37 (70 <= 80).
- WHERE: `agents-shared.md` Constructor r27; `docs/novel/per-j-window-analogues.md`.
- LIMITS: the padded half is the tight one (+1 against the literal half's +10 at 41->43).

### `Delta_J` is bounded uniformly in BOTH `M` and `J`
- STATEMENT.  `Delta_J = Q*_J - F_2(M)`; every LITERAL cell lies in `[-3, +4]` at m11..m41,
  and the excess SHRINKS with depth (`Delta_5 = 0` exactly at both machines where `J = 5`
  exists).  `J_max = A_kill + 1` at all eight censused machines.  Confirmed out of sample at
  machine 53 (every `Delta_J` is `+2`).
- CALCULATES: the whole depth-`>=3` half of the increment law at a step is three inequalities
  (`L3, L4, L5`), and `L_J` for `J > A_kill+1` is VACUOUS by a free emptiness certificate.
- STATUS: measured, 13 cells, exact per cell, by three independent vehicles; NOT proved.
- WHERE: `docs/novel/per-j-window-analogues.md` 1.5.
- LIMITS: at m41 two cells are bounds (`Q*_3 <= 116`, `Q*_4 <= 100`), not exact values.

### The palindrome dichotomy of the maximisers
- STATEMENT.  At every measured cell the maximising word is unique up to reversal.  At `J = 3`
  (11 cells) and `J = 4` (4 cells) the maximiser is a reversal PAIR and NEVER a palindrome
  (Theorem B forbids it outright at even `J`); at `J = 5` (2 cells) it is UNIQUE and
  SELF-REVERSE - a PALINDROME at BOTH machines where `J = 5` is non-empty:
  `(7,10,21,10,7)` at m29 and `(3,25,12,25,3)` at m31, each with `Delta_5 = 0` exactly.
- CALCULATES: the palindrome route applies at ODD `J` only, and at odd `J >= 5` it is exactly
  right; since `J = 5` is the deepest non-empty layer at every machine below m47, killing
  self-mirror stretches would close the deepest member of the finite list outright.
- STATUS: measured, exhaustive per cell.
- WHERE: `docs/novel/per-j-window-analogues.md` 1.6.
- LIMITS: at even `J` the mirror lever has nothing to bite on and the route needs a replacement
  (which is the same-tooth lemma plus `eps`).

### The counterfactual family: the record law is structural, the sizes are not
- STATEMENT.  Keep the gears, keep the mirror symmetry, move the teeth: `v_q` ranges over
  `{1..(q-1)/2}`; every member has the SAME period, the SAME `prod(q-2)` openings and the same
  per-gear density - only POSITIONS move.  On this family:
  **(i) the record law `max(F_2, max_{J>=3} Q*_J) = F(M+q')` holds EXACTLY at every one of
  27,570 members, zero exceptions**; (ii) (D) itself fails at only 0.0-0.6% of members;
  (iii) the INCREMENT LAW fails at 13.3/13.9/14.5/21.7/22.3%, GROWING with the machine;
  (iv) pinning `v_{q'} = round(q'/6)` and letting the old teeth range freely drops the
  violations to 0/0/1.1/6.5/5.7% - **the new gear's tooth carries most of the law**;
  (v) `L` reaches 5 at 19->23 (`J_max = 7`, `A_kill = 6`) where the real m19 has 2.
- CALCULATES: the sharp localisation of where arithmetic enters - **the identity that computes
  `F(M+q')` from the machine below is STRUCTURAL; only the SIZE of `Q*_J` is arithmetic.**
  And `L(M)` bounded does NOT follow from CRT, the mirror, T2/T3, R89/R90 or the record law.
- STATUS: script-verified, exhaustive and exact at 7->11 .. 19->23 (the full 142,560-member
  19->23 family); a 601-member SAMPLE at 23->29.
- WHERE: `docs/novel/tooth-counterfactual-percentile.md` 5B.1, 5C; `agents-shared.md`
  Lateral r29/r30.
- LIMITS: the m23 and 23->29 rows are pinned/sampled and labelled so.

### Where the teeth enter `L`: mod-`{5,7}` admissibility of the bare alternation
- STATEMENT.  Call `(a,b,a)` admissible if some residue mod 5 (and mod 7) carries
  `r, r+a, r+a+b, r+2a+b` outside the gear's tooth pair.  Then
  `P(L>=3 | admissible)` = 0.006 / 0.101 / 0.272 / 0.320 and
  `P(L>=3 | NOT admissible)` = 0.0000 / 0.0000 / 0.0001 / 0.0000 at 13->17 .. 23->29, with
  **0 of 4 / 605 / 19,408 / 1,340 bare-letter `L>=3` words inadmissible.**
  The REAL machine's alternation is NOT admissible at 13->17 `(6,11,6)`, 17->19 `(6,13,6)` and
  23->29 `(10,19,10)` - so its `L <= 2` there is decided by gears 5 and 7 alone - and IS
  admissible at 19->23 `(8,15,8)`, where `L = 2` is a fact about the higher gears.
- CALCULATES: gear 5's tooth explains 17.3% of `L`'s variance at 17->19 (more than the
  incoming tooth's 12.5%; all 22 pinned `L = 3` rows have `v_5 = 2`), while every old gear
  above 7 explains under 1% at every step.
- STATUS: measured, exhaustive; an OBSERVATION (near-perfect necessary condition with one
  exception class, the shifted letter `a+q'`), not a theorem - and it IS the round-31 bare-word
  lemma seen on the family.
- WHERE: `docs/novel/tooth-counterfactual-percentile.md` 5C.2; `agents-shared.md` Lateral r30.
- LIMITS: two channels (letter size and `{5,7}` admissibility) are near-orthogonal and together
  explain only 36-42% of `L`'s variance.

### The twin machine is a low-`F` outlier among its own counterfactuals
- STATEMENT.  `F(twin)` sits at the 20.0 / 18.1 / 26.4 / 17.1 / 11.9 percentile of `V(y)` at
  m11..m23, ~10-15% below the median, never the minimum, in a family whose maximum is 1.6-1.9x
  the truth.  And **the placement STRENGTHENS WITH DEPTH at the two largest machines**
  (m19: 17.1 / 12.3 / 6.3 for `F`/`F_2`/`F_3`; m23: `F` 11.9%, `F_2` 3.1%) - and the route
  consumes `F_2`, not `F`.
- CALCULATES: the real phase vector IS distinguished, in the FAVOURABLE direction; and the
  increment law's own margin `s_min - increment` puts the twin at the 66.8-83.3 percentile -
  the twin uses LESS of the law's budget than two thirds to four fifths of its counterfactuals.
- STATUS: script-verified, exhaustive and exact at m7..m23 (pinned at m23).  Independence
  caveat stated: the rows are NESTED, so no p-value is claimed.
- WHERE: `docs/novel/tooth-counterfactual-percentile.md` 1, 5A.
- LIMITS: the honest negative - the BUDGET SLACK `F(M+q') - F(M) - q'` is UNDISTINGUISHED
  (59.0 / 37.2 / 49.3 percentile at the three largest steps).  Mechanism OPEN, three
  candidates dead (see REFUTED).

---

## H. CERTIFICATE-SIDE FACTS ABOUT ALIGNMENT

### The restricted covering vehicle: prescribing OPEN columns deletes branches for free
- STATEMENT.  `RelaxStar(gears, W, held, ws, openpts)` runs the composed covering LP on
  `[0,W)` minus what the HELD gears strike minus the required-OPEN positions, with `dom(q)` =
  the phases of `q` that strike no required-open position.  Prescribing open positions does
  not just shrink the obligation, **it DELETES BRANCHES OF THE CASE SPLIT for free**: at
  machine 23 span 40 with gear 5 held, THREE OF THE FIVE CASES ARE VACUOUS - that phase of
  gear 5 strikes a required-open position, so the configuration is impossible outright.
  At machine 19, of 1,680 windowed cells, 413 are killed outright because **gear 5 has NO
  phase leaving all three required-open positions open.**
- CALCULATES: adjacent-gap-pair realisability by LP duality - scan-free exact `F_2(19) = 31`
  and `F_2(23) = 39`; and with `openpts = {0,W}`, spectrum HOLES ("no gap of size exactly W").
- STATUS: script-verified, exact rational arithmetic on every verdict; every CERTIFIED verdict
  carries an exact dual certificate re-verified from a clean rebuild, every REFUTED verdict an
  exact primal point verified IN the polytope.  Kernel-checked at the 19->23 rung
  (`CaseCert23.D_19_23_case`) and the 29->31 and 31->37 rungs.
- WHERE: `docs/novel/restricted-covering-certificates.md`; LP thread r26/r27/r29/r30.
- LIMITS: NOT exact - nine unrealised cells at m19 spans 28 and 30 are not certified, four with
  exact in-polytope witnesses (genuine integrality gaps); holding gear 5 closes three of the
  nine.  One of the nine is `(15,15)`, the self-mirror split.

### The lowest-blocker identity
- STATEMENT.  If some gear strikes column `x`, then
  `1 + #{(a,b) : a<b, both strike x, no gear below a strikes x} = #{a : a strikes x}` -
  only the LOWEST striking gear can be the `a` of such a pair, and it pairs with each of the
  other strikers exactly once.
- CALCULATES: summed over the position set, `sum_a |A_a| >= |pos| + sum n_ab` - the whole
  recursion row of the covering certificate, as a `decide` over `2^m` Booleans.
- STATUS: KERNEL-CHECKED, NO AXIOMS (`CaseSplit.lowest6`, `lowest7`).
- WHERE: `docs/novel/restricted-covering-certificates.md` r27 addendum; Formalist r27.
- LIMITS: `n_ab = 0` for 96.4% of the gear-index-1 columns at 29->31 - one gear below suffices
  to cover the whole two-gear overlap; the recursion row is numerically almost entirely a
  Kounias row at the smallest free gear.

### The mirror and the boundary-blocked translation are exact symmetries of the LP
- STATEMENT.  `reflect(hits(q, r, W)) = hits(q, (1 - W - r) mod q, W)` with
  `reflect(i) = W - 1 - i` - the case at `ws` and the case at `(1 - W - ws) mod q` have
  reflected position sets, isomorphic relaxations and EQUAL `V*`, `|pos|` and certificate cost.
  And: if `pos(ws + t) = pos(ws) - t` exactly as subsets of `[0,W)` (the held gears strike
  `[0,t)` at `ws` and `[W-t, W)` at `ws+t`) the same five claims hold - the
  BOUNDARY-BLOCKED TRANSLATION.
- CALCULATES: decide one case per orbit and copy the verdict - a factor 2 from the mirror and
  a further 1.8x from the translation (at m53 `k = 4`, 1,391 classes against 2,503 mirror
  orbits).  It EXPLAINS round 29's unnamed "value classes coarser than mirror orbits" to the
  unit (11 = 11 at m37, 14 = 14 at m41).
- STATUS: theorems with scripts and gates; 385/385 mirror transcriptions and 484 translation
  transcriptions re-verified from JSON at the mirrored/translated case.
- WHERE: `docs/novel/restricted-covering-certificates.md` 2C; LP thread r29/r30.
- LIMITS: prior-art check not run.

### Fixed-depth counters cannot bound alignment depth
- STATEMENT.  With `A_m >= S_m = S^(0)_m >= S^(2)_m >= S^(4)_m >= D_m` (abstract T3 words;
  exposure survivors; Bonferroni depth-`s`; realised), **`S^(2)_m = S^(4)_m = S_m` at all 21
  measured cells** (m19..m37) - fixed-depth Bonferroni kills NOTHING - while the exact `N(w)`
  sits far below the depth-0 term: `min E_0/N` is 6..16 at `m = 1`, 845..10,742 at `m = 2`,
  145,158 / 312,151 at `m = 3`, 4,344,055 at m37 `m = 2`, growing in both `m` and `M`.
- CALCULATES: the verdict - `E_0(w) = prod_g c_g(X)` is a `P`-scale count of columns with the
  pattern's points open, and the higher terms are bounded-ratio corrections, so `E_s < 1`
  cannot happen at fixed `s` until the exposure half has already killed the word.  **A uniform
  bound on `L` needs the cover half at FULL depth (`2^{|Y|}` per word) on a candidate set that
  is itself unbounded in `M`.**
- STATUS: proved for fixed-depth truncations given the measured `EXPCAP` growth; "no counter of
  any kind" is labelled JUDGMENT, NOT RESULT.
- WHERE: `docs/novel/cover-half-counter-ladder.md` 1.4, 1.5.
- LIMITS: the closed form `A_m = sum_k C(m,k) p^k T(m-k)` is proved and asserted vs enumeration
  to `m = 6`; the first-moment threshold (`m` at which `N f_legal^m < 1`) is 4,5,6,6,6 at
  m19..m37 against `L = 2,1,3,3,2`.

### The renewal ladder (re-blocking interior columns one at a time)
- STATEMENT.  For ANY subset `Y` of the interior offsets,
  `#{k mod P : X open, Y blocked} = sum over T subseteq Y of (-1)^{|T|} prod_q c_q(X u T)` -
  exact closed-form CRT arithmetic, and every choice gives a VALID upper bound on the true run
  count.  Nesting the chosen points gives a monotone ladder from the exposure bound (no points)
  to the exact count (all points), at cost `2^{|Y|}`.
- CALCULATES: a rigorous bound on how often a prescribed alignment of openings occurs, at any
  budget; three rungs (`s = 3` points per gap) already clear the route's requirement at every
  constrained case, including the two the exposure bound lost.
- STATUS: proved (inclusion-exclusion + CRT) + script-verified (every bound asserted `>=` the
  exact full-period census where one exists).
- WHERE: `docs/novel/renewal-ladder.md`.
- LIMITS: the requirement it is checked against still carries a fitted constant `lambda`.

### The exact-count route (COV-SAT / COV-COUNT)
- STATEMENT.  A phase vector corresponds by CRT to exactly ONE column per period, so
  `#{k in [0,P) : X open, Y blocked at k}` = the number of models of the CNF projected to
  phase variables.  Count 0 = one UNSAT = a zero certificate with no counting at all.
- CALCULATES: exact per-period occurrence counts of an alignment where the count is small -
  and the extreme patterns near the (D) boundary are precisely the rare ones.  Validated exact
  against full-period scans: `(8,15)@19: 31`, `(10,21)@23: 138`, `(10,21,10)@29: 4` with all
  four recorded addresses reproduced, `(21,10,21)@29: 0`.
  **RECORD MULTIPLICITY LADDER: the record gap `F(M)` occurs exactly 4, 2, 4, 2, 4 times per
  period at m23, m29, m31, m37, m41** - `O(1)` per period while the period grows six orders of
  magnitude.
- STATUS: script-verified; every model CRT'd to its column and machine-verified by assert.
  Op-count event: `(10,21,10)@29` exact count 4 at 9,204 solver propagations against
  `2^38 x 9` inclusion-exclusion ops.
- WHERE: `docs/novel/cov-sat-exact-spectra.md` 1b.
- LIMITS: abundant patterns (counts `>~1e5`) stay out of enumeration reach; cost scales with
  the COUNT, not with `2^{|Y|}`.

### Pairwise convexity computes the record through m17 and provably stops at m19
- STATEMENT.  `L*(y) = min{L : the level-2 (Sherali-Adams) covering relaxation proves RUN(L)
  impossible}` equals `F` EXACTLY at m11, m13, m17 (7, 11, 18) - the PAIRWISE LP computes the
  record.  At m19, `L* = 27` against `F(19) = 25`: `V(25) = V(26) = 0`, and PSD does not
  repair it.  **Every certificate of `F(19) <= 26` must use THREE-gear information; no
  pairwise-consistent reasoning, linear or semidefinite, suffices.**  Vacuity ratio `L*/F` =
  1.000, 1.000, 1.000, 1.080, 1.647, `>= 1.721` at m11..m29.
- CALCULATES: the exact arity at which convex certificates stop seeing alignment.
- STATUS: soundness proved; every claimed bound carries an EXACT RATIONAL DUAL CERTIFICATE
  verified in integer arithmetic; the m19 SDP verdicts are numerical and flagged.
- WHERE: `docs/novel/covering-hierarchy-exactness.md`.
- LIMITS: three certificate families (potentials, covering duals, moment hierarchies) now fail
  along the SAME axis - arity, not convexity.

---

## REFUTED ALIGNMENT CLAIMS

One line each, with the pointer.  Do not re-derive these.

- **M1, "every realised legal spacing value is exactly `a`, `b` or `q'`"** - REFUTED: the exact
  legal alphabet is `{v <= F : v = 0 or +-2c mod q', v realised}` and it contains `49 = a+q'`
  at m31, `55, 68` at m37, `57, 72, 86 = 2q'` at m41; a small-machine phenomenon, alphabet
  growing 1,2,2,3,3,3,4,5,6.  (`agents-shared.md` Constructor r28 "FOR MECHANIC - R40's M1 IS
  REFUTED"; `docs/novel/two-teeth-kill-spacing.md` M1.)
- **R49's identity `N = max(2, A_relax)`** (the acyclic order equals the alternation order) -
  REFUTED at m37 (`A_relax = 2`, `N = 3`, bought by the padded cycle
  `14 -> 41 -> 27 -> 41 -> 14`) and again at m41 (`N = 3 > 2`), where the mechanism is a padded
  2-cycle `[43] -> [29] -> [43]` that dies at order 3.  (Constructor r27/r28.)
- **`A_kill(M -> q') <= 3` as a universal fuel cap** - FALSE at 47->53, where `A_kill = 5`
  exactly (the project's only 5-chain).  (Mechanic r25.)
- **Round-25's alternation predictor "`A_kill >= 5` iff the pair `(s, q'-s)` is realised"** -
  REFUTED at 53->59: `(20,39)` IS realised with two machine-verified witnesses, yet
  `(20,39,20)`, `(39,20,39)` and all longer alternations are ZERO with no SAT call.  Pair
  realisability is necessary (overlap lemma) and NOT sufficient.  (Mechanic r26.)
- **"`L <= 3`"** - REFUTED: `L(47) = 4`, decided this round in FOUR CRT calls
  (`(18,35,18,35)` realised - the first realised legal 4-word in the project).  (Constructor r29.)
- **(B) "`L(M)` bounded by an absolute constant"** - RETIRED as probably false in the limit
  and never needed: `L = O(F/q')` is a theorem and `F/q'` is measured 0.54..2.64 and growing.
  (`docs/novel/spectrum-bound-on-L.md`; Lateral r31.)
- **"`L_pad <= 2` persists"** - REFUTED: `L_pad(47) = 3` MEASURED and `L_pad(53) = 3` follows
  from the bare-word theorem plus the recorded `L(53) = 3`.  (Constructor r31, P6 scored.)
- **P7's mechanism "`L_pad` is the cover half because padded letters are invisible mod 35"** -
  the CONCLUSION stands, the MECHANISM is REFUTED: padded letters are FULLY VISIBLE to gears 5
  and 7 (they refute 13 of 26 non-bare 2-words at m47).  What makes `L_pad` the cover half is
  the ALPHABET SIZE `~3F/q'`.  (Constructor r31, corrected mid-round by the manager.)
- **"The corridor caps alignment length at every machine"** - REFUTED: `CORRCAP` is INFINITE
  from 53->59 on, and at every larger `F/q'`.  No fixed set of small gears can ever cap the
  order again.  (`docs/novel/uniform-order-bound.md`.)
- **H1, "`F(M) mod q'` not in `{0, a, b}` is the teeth-sensitive separator"** - KILLED: it
  holds 11/12 against a base rate of `3*sum(1/q') = 1.291`, i.e. one observed against 1.29
  expected, and it HOLDS at m31 while all three of m31's failing rows FAIL there.
  (Constructor r29 Headline 3.)
- **`Pcong := F(M) mod q' in {0,A,B}` as a characterisation of the increment law's residual
  violators** - REFUTED on the counterfactual family: sensitivity 34.0% / 5.6% at the two
  largest steps, 94.4% of residual violators have `F(M)` NOT congruent to a legal letter, and
  the depth-3 attaining middle is the old record in **0.0%** of 19->23 violators.  The best
  predictor of that form reaches 57.9% balanced accuracy.  (Lateral r29 (a).)
- **Lateral's own lemma A0, "a depth-3 violator cannot have middle `s_min` since
  `g_L + g_R > F_2` is impossible"** - FALSE: `g_L` and `g_R` are at LAG 2, not adjacent, so
  `F_2` does not bound their sum; 41-100% of depth-3 violators have middle exactly `s_min`.
  The correct elementary statement is the peel bound.  (Lateral r29 self-correction.)
- **`SPEC_3` and `SPEC_4` (the spectrum-plus-depth certificate truncated below `J_max`)** -
  UNSOUND on the counterfactual family (30 and 5 unsound cells); the depth range genuinely has
  to reach `J_max`.  `SPEC_5` is sound but certifies 0.3-1.2% where word-legality certifies
  96-100%.  (Lateral r29 (a).)
- **Round-28's framing "the spectrum-plus-depth certificate closes rungs with no census of the
  new machine, hence independently"** - CORRECTED: the old machine's `F_J` values are
  exhaustive only because of deletion-ladder caps taken from `F` at machines ABOVE the step,
  and at `j = 2` that cap is the very quantity the rung bounds.  Rungs below m59 are method
  demonstrations, not independent bounds.  (Constructor r29 self-correction 1.)
- **"Extremal implies palindromic"** - FALSE at `J = 3` and `J = 4` (maximisers are reversal
  PAIRS, and Theorem B forbids a literal even-`J` palindrome outright); TRUE only at `J = 5`.
  (`docs/novel/per-j-window-analogues.md` 1.6.)
- **"Even `J` gives an inequality on `F_J` or `Q*_J` via the mirror"** - NO: `R_J` is
  span-preserving, so the only object it adds is the quotient by an involution, the SAME one
  unit the odd-`J` route gives, and the full symmetry group `Z/2` is proved to be the ceiling.
  (`docs/novel/mirror-parity-laws.md` 9.4, 7.1 Theorem A2.)
- **Round-22's "no bounded state certifies at 29->31"** - OVERTURNED by the history ladder:
  `A_3 + phase 385` certifies (72 <= 74) and `A_4` (three gap values, phase-free, 14,368
  states) is EXACT at all seven scannable steps.  The missing object was the machine's
  dictionary of realised 4-tuples, not a finer congruence.
  (`docs/novel/kleene-generator.md` 4b.)
- **Holt-Rudd's counting bounds `L(M)`** - PROVABLY NOT, from the count alone: a `k`-run
  occupies at most 2 copies and exactly 1 unless every letter is padded, WHATEVER `k` is, so
  no inequality "a `k`-run occupies at most `f(k) < 1` copies" exists.  The term that breaks
  the one-class argument is the minimal in-copy hit distance `s_min(q') ~ q'/3`, and every
  stretch that matters has span above it.  (Harvester r30 follow-on (2).)
- **"Gear 5 is the ONLY parity-obstructed gear for `p <= 37`"** - WRONG AS WORDED: the
  GF(2) test decided a narrower question; because `W_1(1)` is the only odd histogram entry and
  `1 = 1 mod p` for every `p`, `alpha_1(p)` is ODD at every gear, so **the pole phase is
  unattainable at EVERY gear.**  (Lateral r26 self-correction; `docs/novel/gear-cell-decomposition.md`.)
- **`arg H_5(1) = 126 deg` as a machine-independent constant** - REFUTED twice: it is
  unattainable exactly (a parity obstruction) and on exact cyclic data it is a monotone
  DOWNWARD ladder `129.776 -> 125.659` at m13..m37 that crossed below 126 and stayed there.
  (Mechanic r26; Lateral r25.)
- **`alpha_1/alpha_2 -> -1/phi` (the golden direction) as an asymptotic law** - REFUTED: it is
  a CROSSING, not a limit (`-0.5778` at m37, `+0.0403` past `-1/phi` and still rising).
  (Lateral r27 item 3.)
- **Lateral item 29(a)'s machine-DFT formula `prod_q hat_q(m mod q)`** - WRONG AS WRITTEN;
  the correct form is `prod_q hat_q(m c_q)` with `c_q = (P/q)^{-1} mod q`.  Everything item 29
  concludes survives.  (Lateral r29 self-correction.)
- **The 2n-gap reordering as a route to `F`** - REFUTED BY ITS OWN PROOF: over 60 admissible
  re-choices of the teeth the count stays `2n = 8` while `F` ranges over `[10,18]`, and `F` is
  not a statistic of the order permutation at all (a permutation records order; `F` needs the
  metric).  Marked a CLOSED LINE.  (`docs/novel/two-n-gap-reordering.md`.)
- **Round-22's "#distinct eigenvalues = |Farey(F+1)| - 2"** - assumed every gap length `1..F`
  is realised; HOLES break it (true counts 21/41/113/183/363/549/981/1813/2467 at m11..m41
  against the published values), with the exact loss rule `loss = sum phi(hole+1)`.
  (Lateral r26 self-correction 1.)
- **"Peak qualifying depth is non-decreasing in `M`"** - REFUTED by Mechanic's own table: the
  peak is terminal at `M <= 23` and INTERIOR at m31 (5 of 7) and m37 (7 of 8).  (Mechanic r28.)
- **"The transfer superset is exact at span `<= 80"`** - REFUTED: the inflation onset is at
  span 68, sharply (every reverse class of span `<= 67` realised, first refutation at 68).
  (Mechanic r27, C1 scored.)
- **Three pre-registered closed forms for the onset** (`F_2` one machine back; `2F` two
  machines back; a constant ratio to `F`) - ALL THREE FAIL out of sample; each had fitted the
  single round-27 data point.  (Mechanic r28.)
- **`R45`'s `A_relax(37) = 3`** - WRONG, it is 2; `arity_ladder.py` HARDCODED the `m=1` and
  `m=2` entries at m29/31/37 as "realised" instead of looking them up, and gear 5 refutes
  `(14,27)` by phase saturation.  (Constructor r27 self-correction.)
- **`R61`'s scan-free `D_2 = 1,254` / `D_3 = 15,020` at m31** - should read 1,253 / 15,019;
  the run predated the `decide_cover` fix and counted the phantom `(1,1)` and `(1,1,1)`.
  (Constructor r26.)
- **The three "mechanism" candidates for the low-`F` outlier** - ALL REFUTED: angular
  coherence (refuted IN THE SIGN - the twin sits in the LOWEST-dispersion quartile, which has
  the HIGHEST mean `F`); "the teeth are the reciprocal of a small integer" (the `m = 1..60`
  sweep's median is exactly the family median, argmin at `m = 12` not the twin's `m = 6`); and
  localisation in `(v_5, v_7)` (the top-variance gear is 7 or 11, never 5; no gear explains
  more than 9%).  **The twin is a low-`F` outlier INSIDE the high-`F` class on every axis
  proposed.**  (`docs/novel/tooth-counterfactual-percentile.md` 5, 5A.3.)
- **"The twin machine's advantage shows in the budget slack `F(M+q') - F(M) - q'`"** - NO:
  59.0 / 37.2 / 49.3 percentile at the three largest steps, undistinguished.  The favourable
  quantity is the INCREMENT LAW'S OWN MARGIN (66.8-83.3 percentile).  (Lateral r28 (iv).)
- **Section pre-registration S4, "gear `p` is the death rung of 0.35-0.70 of its section"** -
  REFUTED as stated and EXPLAINED by the boundary: `p^2` is the excluded near edge, so the
  candidates are 1-3 columns each credited with probability `~1.7/ln p`, giving 75-80% zero.
  (`docs/proof-search/word-tree.md` 7.2.)
- **Section pre-registration V4, "the largest gear interacting with a new twin exceeds `p/2`
  for 5-25%"** - REFUTED: 46-48% at `q' >= 1000`; the 11.7% was a WINDOW AVERAGE dominated by
  the low part of the window where near-twin columns cannot exist.  Also refuted: "the fraction
  of new twins whose word is final by level 13" is 0.7-2%, not 5-25%.
  (`docs/proof-search/word-tree.md` 8.2.)
- **Section pre-registration C, "the section cannot tell the real teeth from moved ones"** -
  REFUTED, and the refutation is the finding: moving gear 13's teeth to ANY `v` leaves 3.6-3.8%
  more survivors than the real teeth (gear 7's 6.4%), with the cofactor model
  `ln n / ln(n/s*)` matching.  (`docs/proof-search/word-tree.md` 9.3.)
- **Round-27's "second frontier: a convergence frontier with no closed form"** - the object
  DOES NOT EXIST at the named cell: the lifted LP's limit polytope is EMPTY there, so the
  decelerating loop was converging to a certificate.  "I had built a frontier out of a
  symptom."  (LP thread r28 self-correction.)
- **E10, "the `k = 3` case split is tight on `F` at m41 (certifies `F(41) <= 91`)"** -
  REFUTED: 32 of 92 decided cases carry EXACT in-polytope refutations.  The error was reading
  "case 0 is already empty at 92" as a statement about the SPLIT when the sentence beside it
  said "a per-case reading only".  (LP thread r29.)
- **"k_max <= 4 at 47->53" and the fuel-cap route to repairing the word-free criterion there**
  - DEAD, not deferred: `A_kill = 5` forces depth `>= 6` and `Q_6(47;18) = 174 > 171`.
  (Mechanic r25.)
- **The plain (size-floor) criterion at 43->47 and 47->53** - FAILS (152 vs 150, 177 vs 171),
  and the failure is in the CRITERION, not the machine: the m47 witness's four middles
  `[22,28,30,67]` all clear the floor `a = 18` but NOT ONE is congruent mod 53 to a legal
  letter, so the stretch can never be merged.  Word legality repairs both.
  (`docs/novel/old-machine-spectrum.md` 8.)

---

## GAPS: what the record itself names as unstated about alignment

Only items the record explicitly flags as open.

- **Is `L_pad(M)` bounded?**  This is now the WHOLE of (B).  Nothing on record bounds it; the
  row is `0,0,0,1,1,1,2,2,2,2,3,3` at m11..m53 and it grows.  (Constructor r31, U22.)
- **Is `L(M) = A_kill - 1 = D_g - 1` bounded at all?**  Open; `L(47) = 4` is the current
  maximum and the row is non-monotone.  Formalist's honest boundary: "the chain DEPTH `D_g` is
  not an algebraic consequence - a run alternates freely, and `D_g` is a fact about lower gap
  SIZES."  (`docs/novel/anchor-235-layer-laws.md` 3, 5.)
- **(A-pad): the `F_3`-wall event.**  `|eps| <= 4` is measured on LITERAL letters; the padded
  letter at m31 has `eps = -17`, and the closure of the whole chain rests on exactly this.
  Open question named: does the `F_3`-wall event recur, and when it does, does the increment
  law fail by exactly `F_3 - F_2 - s_min`?  (Constructor r30/r31; Lateral r31 U22.)
- **(D2), the depth-2 slack `S_2 = F + q' - F_2`,** measured 9..49 and growing roughly with
  `q'` - "no instrument on record supplies it".  It is R55's 2F wall.  (Constructor r30 R99.)
- **Is the spectrum bound on `L` tight infinitely often?**  Tight at m11, m13, m29 and at
  13-87% of counterfactual rows.  If `L = 2F/q' + 1 - o(1)` then (B) is definitively false; if
  `L` stalls, a better bound exists.  Cheap test named: the full 23->29 family.  (Lateral r31 U23.)
- **`spearman(L, n_0) = +0.311` beats both `a_min` and `min(n_a,n_b)`,** and `n_0` does not
  depend on `v_{q'}` at all - so this is an unstated statement about the OLD machine's gap
  histogram at multiples of `q'`.  UNCLAIMED.  (Lateral r31 U24.)
- **Why is `Phi(q')` twice as large relative to `F_2` at m31 as anywhere else?**  The counted
  padded-gap census would say; the flank order-statistic law is a LITERAL-letter law and
  inverting it at m31 gives an eight-orders-wide interval.  (Constructor r29/r30.)
- **Is there a statement "the padded letter's flank envelope obeys a DIFFERENT law", or is the
  padded letter simply where the literal analysis stops applying?**  Named, unanswered.
  (Constructor r29 open questions.)
- **Is `Ghat` computable below a scan?**  The whole walk - and therefore the anchor-235 floor -
  reduces to that ONE object; its mean-field part is closed form and carries 69-77% of the
  energy, the residual is depth-1 adjacency, which is the same open term as the depth-sum
  identity's.  (Lateral r29, U17.)
- **Does the family-wide record law survive the ASYMMETRIC family** (teeth at arbitrary
  `{t_q, t'_q}`, not `+-v_q`)?  The attainment proof does not obviously use the mirror; the
  answer says which hypothesis the derivation may assume.  Cheap at m11/m13, NOT RUN.
  (Lateral r29, U18.)
- **Is `d_0` the whole depth-2 story?**  The non-wrap depth-2 slack is positive (min 4-9) at
  every step; is `F_2 <= max(2 d_0, F + c)` for a small `c` on the family?  (Lateral r30, U20.)
- **The `{5,7}` admissibility cap at the corpus rungs** - a one-page table of the rungs at
  which the twin alternation is `{5,7}`-admissible to length `k`, against
  `L = 1,1,1,2,1,3,3,2,2,2,4,3`.  Named as the next step, UNCLAIMED.  (Lateral r30, U21.)
- **Prove the onset law, or find its first failure.**  The natural next test is 41->43, which
  needs `D_4(43)` or a further extension of the m41 shard.  The multiplicity residue
  (`onset/Y_5`, a near-constant factor 1.37-1.41) is the only part still unexplained.
  (`docs/novel/dictionary-monotonicity-onset.md` 5.)
- **No teeth-arithmetic separator of the three open `Phi` rows was found:** none of `q'/F`,
  `q' mod 210`, litcap, `F mod q'`, `a/q'` orders the machines so that m31 is extreme.  The
  construct that would decide it is named and NOT delivered (the counted padded-gap census
  `occ(q'; M)` at m29/m31/m37 - since delivered at m37 in r30, still open above).
  (Constructor r29 Headline 3.)
- **The mechanism of the low-`F` outlier is OPEN** with three candidates dead; the effect is
  an INTERACTION spread over the whole tooth vector, not a main effect of any gear.
  (`docs/novel/tooth-counterfactual-percentile.md` 5A.3.)
- **Whether the phase-reduction record law can be derived in the kernel from the walk** (a
  correctness proof of `chain_depth.py`'s walk against `Machine17.nextOp`), which would make
  the record law a single kernel theorem at 17 rather than two verified ends.
  (`docs/novel/anchor-235-layer-laws.md` 5.)
- **Does partial level 3 (triple moments on selected gear triples) certify `L = 25` at m19, and
  WHICH triple is the obstruction?**  Named construct, not built.
  (`docs/novel/covering-hierarchy-exactness.md` 7.)
- **The named construct behind the padded arity ceiling:** the phase-saturation ceiling of the
  PADDED alternation family `(s, q'+(q'-s), s, ...)` - the same pigeonhole on a different
  exposed set, therefore another closed form in the small gears, with no solver.  NAMED, NOT
  BUILT.  (Mechanic r27.)
- **The counted occurrence list of every realised legal word by CRT enumeration at m41..m47,
  with the flank sum per column** (Constructor's `occ(q'; M)` as a list of addresses) -
  NAMED, NOT BUILT; it is what makes `F(M+q')` scan-free at the short-word steps.
  (Mechanic r30 (c).)
