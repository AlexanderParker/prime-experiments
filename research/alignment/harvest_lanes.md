# Alignment of openings — everything on record

READER harvest, 2026-09-04. Sources read in full: `docs/proof-search/anchor-235.md`,
`constructor.md`, `mechanic.md`, `lateral.md`, `formalist.md`, `harvester.md`,
`lp-duality.md`, `human.md`, and the header comments of `proofs/*.lean`.
Nothing here is new; nothing is rederived. Every entry is a statement the record
already makes about WHERE and WHEN the openings of the gears coincide.

VOCABULARY (the human's). The machine `{5..y}` has one gear per prime; gears 2 and 3
are built into the columns (column `k` = the pair `6k-1, 6k+1`). A gear's OPENINGS are
the `g-2` columns of every `g` it does not strike. The machine's openings are the
columns every gear leaves open: `prod (g-2)` per period `P = prod g`. The WINDOW is the
certified range (columns with `6k+1 < q'^2`); the SECTION is its new part
`p^2 < 6k+1 < q'^2`. A STRETCH is a run of consecutive columns. The RECORD `F(M)` is
the longest stretch with no opening, max-gap convention.

TRANSLATION NOTE used throughout: the lanes' word "gap" means the distance between
two consecutive openings, i.e. the blocked stretch plus one; `F(M)` = max gap;
`F_j(M)` = max sum of `j` consecutive gaps (the longest stretch holding `j-1`
interior openings). anchor-235 section 9 uses a BLOCKED-COUNT convention `F_bc = F - 1`
and says so; entries below are in max-gap units unless flagged.

---

## A. WHERE A SINGLE GEAR'S OPENINGS ARE

### tooth rule
- STATEMENT: gear `q` strikes column `k` iff `k = +u_q` or `k = -u_q (mod q)`, with
  `u_q = 6^{-1} mod q = round(q/6)` and `6u_q = q -+ 1`. The two struck residues are the
  gear's TEETH; the other `q-2` residues are its openings. Every gap between two teeth
  of one gear is `2u_q` or `q - 2u_q`, and the two sum to `q`.
- CALCULATES: the exact residue set a gear leaves open, from `q` alone; the two literal
  letters `a = 2u'`, `b = q' - 2u'` of every merge word; `s_min(q') = min(a,b) = a`.
- STATUS: kernel-checked (`TwoTeeth.kill_spacing`, `kill_spacing_min`, `kill_period`,
  `teeth_letters`, `gear_side`; `LiteralCapTable.tripled_teeth_antipode`); numerically
  asserted for every prime gear 5..199 (`research/check_two_teeth.py`) and to `q = 100000`.
- WHERE: mechanic.md Definitions; constructor.md R40 (T1, T4); formalist.md 2.18;
  lateral.md item 29(b) (T3 law `3u = (q+1)/2`).
- LIMITS: says nothing about which columns SURVIVE all gears jointly.

### one gear over the 2,3,5 anchor
- STATEMENT: with anchor 30 (6 anchor-open numbers per cycle: `1,11,13,17,19,29 mod 30`),
  a gear `q >= 7` hits exactly six anchor-open numbers per run of `30q`, one per
  anchor-open residue class, so it hits each twin-slot type (`11|13`, `17|19`, `29|31`)
  exactly twice per run — once on the lower number, once on the upper — and leaves
  `q - 6` cycles wholly untouched. The class of `q mod 30` fixes only WHERE in the run
  the six hits sit, not how many.
- CALCULATES: the six `m`-values (four classes: `q=+-1: m=1,11,13,17,19,29`;
  `+-7: 7,11,13,17,19,23`; `+-11: 1,7,11,19,23,29`; `+-13: 1,7,13,17,23,29`), hence the
  untouched-run lengths `q x (gap in m)` and the cycle index `((q m - 11) div 30) mod q`
  of each hit.
- STATUS: exact, script-verified for every prime `11 <= q <= 5000`
  (`research/anchor235/anchor30b.py`, `recheck_cycles.py`).
- WHERE: anchor-235.md sections 1-2.
- LIMITS: single gear only; clean ends (first and last cycle of the run untouched) hold
  from `q = 37` on, with listed exceptions at 11,13,17,19,29,31.

### anchor-open columns
- STATEMENT: the anchor 2,3,5 leaves open exactly the slots `k mod 5 in {0,2,3}` (numbers
  `1,11,13,17,19,29 mod 30`); every machine's openings are a subset of these.
- CALCULATES: the base density `6/30` and the residue frame in which every gear's hits
  are counted; the AP lemma and the adjacent-gap exclusion law are consequences.
- STATUS: exact, definitional; kernel-adjacent through `Machine19.expT` / `Corridor.Exposed`.
- WHERE: anchor-235.md conventions; lateral.md Definitions (gear 5 exposes `{0,2,3} mod 5`).
- LIMITS: gear 5's frame only.

### corridor mod 35 (the exposed set E)
- STATEMENT: gears 5 and 7 jointly leave open exactly 15 residues mod 35,
  `E = {0,2,3,5,7,10,12,17,18,23,25,28,30,32,33}`. Every opening of every machine
  containing 5 and 7 lies in `E`; a stretch of openings with gaps `g_1..g_l` can sit only
  at base residues whose partial sums all stay in `E` (`carrier`).
- CALCULATES: for any prescribed word of gaps, the set of admissible base residues mod 35;
  an EMPTY carrier forbids that configuration at every machine, forever, with no scan.
- STATUS: kernel-checked (`Corridor.exposed_iff_mem`, `endpoint_law`, `adjacency_law`,
  `forbidden_pairs_count = 294`; `TierA.carrier`, `mem_carrier_of_chain`,
  `no_chain_of_carrier_empty`).
- WHERE: formalist.md 2.8, 2.14; constructor.md R16, R22; lateral.md items 18-20.
- LIMITS: corridors constrain WHERE, never HOW BIG (X11): every `(G1,G2)` pair is within
  L1 distance 1 of a corridor-allowed pair at any bounded modulus.

### completeness lemma (only small gears can block a short shape)
- STATEMENT: a shape with `n` prescribed open columns can be blocked by gear `q` only if
  `q <= 2n`, because two teeth forbid at most `2n` of `q` phases and CRT makes the gears
  independent. For `n <= 5` the mod-35 test IS the entire corridor; gear 11 first enters
  at `n = 6`.
- CALCULATES: which gears need to be consulted at all for a given number of prescribed
  openings; makes every `<= 5`-point alignment question a finite mod-35 computation.
- STATUS: proved (lateral r17); reused as the phase-saturation bound `|FREE_q(X)| >= q - 2|X|`.
- WHERE: lateral.md item 20; mechanic.md K9/C29.
- LIMITS: gives a NECESSARY condition only (the exposed half of realisability).

### the 32-cap (horizon of prime-adjacent stretches)
- STATEMENT: gears 5 and 7 have both-composite classes at `k = 1` and `k = 34 (mod 35)`,
  whose largest cyclic gap is 33; so ANY 33 consecutive columns contain a column with
  both members composite, and a stretch of columns each carrying a prime member is at
  most 32 long — unconditionally, at every scale, from two gears.
- CALCULATES: an absolute ceiling on saturated runs; `n2_packing`: `W/33 <= n2` on any
  window of width `W`.
- STATUS: kernel-checked (`Corridor.exists_class_in_run`, `both_composite_in_run`,
  `double_slot_in_run`, `prime_adjacent_run_le`, `n2_packing`).
- WHERE: formalist.md 2.8; lateral.md item 10.
- LIMITS: about prime-carrying columns, not about machine openings; `L0 = 32` survives
  gears through 23, and whether `lim L0 = 32` is a finite check nobody has run (B2).

### the window (kernel route)
- STATEMENT: inside `(y, y^2]` the gears below `y` decide primality exactly — a composite
  there has a prime factor strictly below `y` — so a column of the machine `{5..y}` that
  is open and lies in the window IS a twin pair. The kernel's window is
  `k in (q/6, W]`, `W = (q'^2 - 1)/6`, and it is the OPENING STRETCH of the machine's
  periodic pattern, containing every lower section.
- CALCULATES: the certified range of any machine; converts "an opening lands here" into
  "a twin exists".
- STATUS: kernel-checked (`Horizon.exists_prime_factor_lt`, `prime_of_no_prime_factor_lt`,
  `twin_of_no_prime_factor_lt`; `BlockedSlots.survivor_iff_twin`,
  `twins_infinite_iff_survivor_in_window`).
- WHERE: formalist.md 2.1; proofs/Horizon.lean, proofs/BlockedSlots.lean; anchor-235.md 7.
- LIMITS: the kernel needs one survivor anywhere in `(q, q'^2)`; the SECTION-only
  statement is strictly stronger than the twin conjecture (a dead section is a twin gap
  `>= 4 sqrt(x)`).

---

## B. THESE COLUMNS ARE ALWAYS OPEN

### column 0 and the wrap gap
- STATEMENT: slot 0 is an opening at every machine (gear `q` blocks `k = +-u_q`, never 0),
  and because the opening set is mirror-closed the largest opening is `P - x_1`; hence the
  wrap gap equals the FIRST gap `d_0`.
- CALCULATES: the missing cyclic-seam gap in closed form, `d_0 = 2,3,3,5,5,5,7,7,7,10` at
  m7..m41; every full-period census total `prod(q-2)`.
- STATUS: exact, asserted at nine machines (`research/cyclic_close_r25.py`).
- WHERE: mechanic.md C26; lateral.md item 46(d).
- LIMITS: `d_0` is a per-machine number; nothing predicts it beyond the sieve.

### the antipodal columns are open (`g_1* = 1`)
- STATEMENT: the columns `(P +- 1)/2` are openings at EVERY machine: `6 * (P+1)/2 = 3P+3`,
  so the antipodal column's members are `3P+2` and `3P+4`, and a gear striking either
  would divide 2 or 4. Equivalently `6s = 3 mod q` against teeth at `6u = +-1`, and
  `3 = +-1 mod q` is impossible for `q >= 5`.
- CALCULATES: the antipodal gap is 1 at every machine, with no scan; e.g.
  `Machine29.Exposed29 539141103` is exhibited by arithmetic alone.
- STATUS: kernel-checked (`Mirror.antipode_open`, `antipode_exposed11`, `antipode_exposed29`).
- WHERE: formalist.md R26.4; lateral.md item 53.
- LIMITS: two columns per period; buys the parity corollary below, nothing about size.

### maximal gaps occur an even number of times
- STATEMENT: the mirror `k -> -k` fixes only column 0, so `W_1(g)` (the number of gaps of
  length `g` per period) is EVEN for every `g >= 2`; only the count of gaps of size 1 is
  odd. In particular the record gap never occurs exactly once, and record gaps come in
  mirror pairs summing to `P - F`.
- CALCULATES: parity of every gap-length count; a counting argument that caps a
  configuration at ONE proves there are NONE.
- STATUS: proved + measured (lateral item 53, 46(a)); record multiplicities 12,20,20,4,2,4,[2],[4]
  at m13..m41, all even; kernel halves in `Mirror.self_mirror_unique`,
  `Mirror.even_card_involution`, `window_count_even`, `none_of_at_most_one`, instantiated at
  machine 11 (`MirrorM11.opSeq_mirror`, `g11_mirror`, `window2_even`).
- WHERE: lateral.md items 46, 51, 53; mechanic.md C18; formalist.md R27.5, R28.0.
- LIMITS: worth exactly ONE unit (a factor two) — the full symmetry group of the opening
  set inside `Z_P` is `{id, mirror} = Z/2`, proved, so no mod-4 lever exists from any
  symmetry.

### the self-mirror stretch, located
- STATEMENT: a depth-`j` window of openings is self-mirror iff it is centred on a mirror
  centre — on column 0 for even `j`, on the antipode for odd `j` — and there is exactly
  one such window per depth (`N = prod(q-2)` is odd). Its span is `2 o_{j/2}` (`j` even) or
  `P - 2 b_{(j+1)/2}` (`j` odd), computable by sieving a few dozen columns.
- CALCULATES: the exceptional window's ADDRESS and SPAN scan-free at any machine; the
  parity of every span count.
- STATUS: proved + verified against exact `W_j` censuses at m11..m29 for every `j <= 12`;
  kernel `Mirror.self_mirror_unique`.
- WHERE: lateral.md items 52, 65; formalist.md R27.5/R28.0.
- LIMITS: at `j = 2` the self-mirror window `(d_0, d_0)` IS legal, so the mirror lever
  needs the hypothesis `d_0 != F` there (discharged by a one-line inequality at every
  machine); at `J >= 3` it is never word-legal (below).

### the self-mirror window is never word-legal at depth >= 3
- STATEMENT: at odd `J` the self-mirror window's central middle is the antipodal gap,
  length 1, and 1 is a legal letter only if `3 = +-1 mod q'` — impossible. At even
  `J >= 4` its two central middles are both `d_0`, which T3 forbids (equal nonzero classes
  in a row) and which cannot both be padded since `0 < d_0 < q'`. So the reversal map is
  fixed-point-free on the word-legal family and every span count there is even, with no
  exception list and no census.
- CALCULATES: removes the mirror lever's hypothesis at every depth `>= 3`; every maximiser
  of `Q*_J` at `J >= 3` comes in a pair whose partner's address is `P - k - s`.
- STATUS: proved, gated at m11..m23 for `J = 2..7` (185 assertions).
- WHERE: lateral.md item 73, 74, 83.
- LIMITS: no inequality on `F_J` or `Q*_J` follows — the map is span-preserving, so the
  lever is worth one unit and no more.

---

## C. NO ALIGNMENT HERE: FORBIDDEN CONFIGURATIONS

### AP lemma (four openings in arithmetic progression)
- STATEMENT: openings have `k mod 5 in {0,2,3}`; four terms of an AP with difference
  coprime to 5 occupy four distinct residues mod 5, and three residues cannot hold four.
  So there are NO four openings in arithmetic progression with common difference `q'`,
  for every prime `q' > 5`.
- CALCULATES: kills `j = 2` and `j = 4` literal links between two padded links, and
  `p = 3` all-adjacent padding, for every gear; generalised form: four openings at pure
  multiples `i q'` with the four `i` distinct mod 5 are impossible.
- STATUS: proved, scale-free (lateral r16/r17).
- WHERE: lateral.md item 19, 20.
- LIMITS: about APs of difference `q'` only.

### openings AP theorem (equal-gap runs)
- STATEMENT: an AP of `L` openings has common difference divisible by every gear
  `q < L+2`: 3 consecutive equal gaps require `5 | g`, 5 require `35 | g`, 9 require
  `385 | g`, and `L >= y+2` needs the full primorial.
- CALCULATES: the longest run of equal gaps at any machine (measured 3-4 everywhere,
  always with `g = 5`).
- STATUS: proved; verified m13..m29, zero violations.
- WHERE: lateral.md item 22.
- LIMITS: equal gaps only.

### adjacent-gap exclusion law (mod 5)
- STATEMENT: three consecutive openings with gaps `(g1, g2)` are IMPOSSIBLE whenever
  `(g1 mod 5, g2 mod 5)` lies in `{(1,1),(1,3),(2,4),(3,1),(4,2),(4,4)}` — 6 of 25 classes
  — at every scale in every machine containing gear 5, and the law is COMPLETE (by the
  completeness lemma only gear 5 can block a 3-point shape).
- CALCULATES: a free prune on any adjacent gap pair.
- STATUS: proved; cross-checked against full-period censuses m11..m31 (1,589 populated
  lag-1 cells, zero forbidden; 6.2e9 adjacent pairs at m31).
- WHERE: lateral.md item 24.
- LIMITS: ADJACENT gaps only — at separation `j >= 2` the same classes carry up to 35.8M
  counts, so nothing follows.

### phase saturation (a gear with no free phase)
- STATEMENT: a prescribed pattern with exposed offsets `X` occurs somewhere only if every
  gear has a phase avoiding `X`, i.e. `FREE_q(X) = Z_q \ ((X mod q) u ((X - s_q) mod q))`
  is non-empty, `s_q = -2 * 6^{-1} mod q`. Since `|FREE_q(X)| >= q - 2|X|`, only gears
  `q < 2|X|` can ever fire — the whole content is at gears 5, 7, 11.
- CALCULATES: a solver-free ZERO verdict for any word/tuple; a closed-form ceiling on the
  pure alternation per step (6,2,2,2,5,3,3,4 at 31->37 .. 61->67); level prunes
  (3/11, 6/15, 6/15, 4/19, 7/36 of the k=3 words at 37->41 .. 53->59; it closes the k=6
  level at 41->43 and the k=7 level at 43->47 outright).
- STATUS: theorem, no solver; sound (never zeroes any of 37 words known realised);
  reproduces every structural zero on record; reverse-invariant.
- WHERE: mechanic.md C29, K9, C31, C40; docs/novel/phase-saturation-arity.md.
- LIMITS: bounds the ALTERNATION family only — padded letters give words the obstruction
  does not kill; it answers 0 of the 27,197 superset-YES arity-4 queries at 41->43,
  because MF_4 edges are already corridor-admissible.

### the walk screen (screen what the search walks)
- STATEMENT: every point of the underlying WALK — the deleted interiors included — is an
  opening of `M`, so the whole walk must have an admissible phase at every gear, not just
  the emitted tuple. Screening the walk is sound, strictly stronger, and a prefix prune.
- CALCULATES: tighter certified supersets of a machine's realised tuples (at 31->37:
  2,435,140 -> 1,153,814 against a truth of 291,675).
- STATUS: exact; walk-screened == walk+emission at all six steps, so the walk screen
  SUBSUMES the emission screen; soundness asserted (no realised tuple removed).
- WHERE: mechanic.md C40, standing rule 35.
- LIMITS: a screen, not a decision; the universal witness `(1,2,3,2,1)` it catches is
  phase-saturated at gear 5.

---

## D. WHAT ADDING A GEAR DOES TO THE OPENINGS

### merge law
- STATEMENT: when gear `q'` is added, every gap of `M + q'` is a MERGED WINDOW of `M`: a
  run of `j` consecutive old gaps whose `j-1` interior openings are all struck by one
  phase of `q'`. `F(M + q') = max over maximal legal killed runs of `o[i+k] - o[i-1]`,
  computed from the OLD machine alone.
- CALCULATES: the new machine's record from the old machine's opening sequence — no new
  period is built.
- STATUS: exact, verified at six steps 13->17 .. 31->37 (F = 18,25,34,43,58,88);
  kernel-checked as `MergeLaw.newgap_le`, `newgap_le_step`, `Spectrum.merged_eq`.
- WHERE: lateral.md item 16; constructor.md R39; formalist.md 2.16-2.17, 2.20.
- LIMITS: one-step and structural, not budgetary — it bounds SINGLE new gaps, so it does
  not by itself supply the next rung's `F_2` (formalist verdict 9).

### residue necessity (T2) — where consecutive kills can sit
- STATEMENT: any two openings struck by one phase of `q'` differ by `0`, `2u'` or
  `q' - 2u'` mod `q'`. So every interior gap of a merged window is in
  `V = {0, +-2u' mod q'}`, and a positive gap in one of those classes is at least `2u'`.
- CALCULATES: the qualifying floor `a = 2u'` for free; the legal alphabet
  `Lambda(M) = {v <= F(M) : v = 0 or +-2u' mod q'}` (about `3F/q'` letters).
- STATUS: kernel-checked (`MergeLaw.interior_gap_mod`, `floor_of_mod`,
  `TwoTeeth.spacing_from_lo/hi`, `Machine23/29/31/37.merge_alphabet`).
- WHERE: constructor.md R40 (T2); formalist.md 2.17, 2.18, 2.20.
- LIMITS: necessary, not sufficient — realisation is a fact about the old machine.

### T3 alternation
- STATEMENT: within a merged window the nonzero classes STRICTLY ALTERNATE (`+2u'`,
  `-2u'`, `+2u'`, ...), padded letters (`0 mod q'`) being transparent, so
  `|#a - #b| <= 1` per window. Equivalently, read with the current tooth: `pad` keeps the
  tooth, `up` needs `-` and moves to `+`, `down` the reverse.
- CALCULATES: the legal word grammar — exactly two alternating words per length; two
  consecutive nonzero letters sum to `>= a + b = q'`.
- STATUS: kernel-checked (`WordLegal.Alt`, `legal_iff_noRepeat`, `alt_iff_prefixSum`;
  `AnchorChain.no_two_up`, `no_two_down`); asserted on every window of every full joint
  period at steps 11->13 .. 29->31.
- WHERE: constructor.md R40 (T3); formalist.md R30.1; mechanic.md definitions.
- LIMITS: the residue arithmetic alone bounds no length — a run alternates freely.

### fuel-span cap (T5)
- STATEMENT: consecutive kills are at least `2u'` apart, so a chain of `k` kills spans at
  least `2u'(k-1)`, i.e. `k <= 1 + span/(2u') <= 1 + 3 span/(q'-1)`.
- CALCULATES: an a-priori cap on chain length from the span alone, no census.
- STATUS: kernel-checked (`TwoTeeth.kills_gap_ge`, `fuel_span_cap`, `fuel_le`).
- WHERE: constructor.md R40 (T4/T5); formalist.md 2.18.
- LIMITS: `span/q'` grows without bound along the ladder, so the cap grows.

### attainment theorem (the record law)
- STATEMENT: if consecutive openings `x_0 < ... < x_J` of `M` have a legal middle-gap word
  then `x_J - x_0 <= F(M + q')`; and conversely every gap of `M+q'` is such a window. So
  `max(F_2(M), max_{J>=3} Q*_J(M; legal for q')) = F(M + q')` EXACTLY, where `Q*_J` is the
  maximal span of a `J`-gap window of `M` whose `J-2` middles form a legal kill word.
  PROOF: legality is the existence of a tooth assignment `t_1..t_{J-1}` with
  `x_{i+1} - x_i = (t_{i+1} - t_i) c (mod q')`, `c = 6^{-1} mod q'`; fix one, set
  `r = t_1 c - x_1`; the joint period is `P(M) q'` with `gcd(P(M), q') = 1`, so some
  translate `x + jP(M)` with `jP(M) = r (mod q')` is a window of `M` with the same gaps
  in which `q'` strikes EVERY interior.
- CALCULATES: the new machine's record on the OLD machine's period — machine `M+q'` is
  never built. `J* = k_win + 1` at every step.
- STATUS: proved both ways (constructor R68 / R46); computed exactly at eight steps
  11->13 .. 37->41; two out-of-scan confirmations `Q*_max(43;47) = 118 = F(47)` and
  `Q*_max(47;53) = 145 = F(53)`; the vehicle then computed `F(59) = 161` on machine 23's
  period (ratio 5.3e11). Verified family-wide at 27,570 counterfactual machines.
- WHERE: constructor.md R68, R46; mechanic.md C24, C27, C35, C51; lateral.md item 69.
- LIMITS: `Q*_max` EQUALS `F(M+q')`, so the criterion is not a relaxation — there is no
  slack in it to exploit; its value is entirely that it is computed on the old machine.

### word reduction (R89) — when a legal window exists at all
- STATEMENT: with `L(M)` the length of the longest REALISED legal word (a run of
  consecutive gaps all in `Lambda(M)` whose nonzero T3 classes alternate),
  `Q*_J(M;q') > -inf  <=>  L(M) >= J-2`, hence `J_max(M) = L(M) + 2` and
  `A_kill(M -> q') = L(M) + 1`. So every EMPTY cell of the per-J table is a one-line
  dictionary fact.
- CALCULATES: the exact depth range a rung must cover, from the word dictionary alone;
  `L = 1,1,1,2,1,3,3,2,2,2,4,3` at m11..m53.
- STATUS: proved; kernel-checked over an abstract machine (`WordLegal.word_of_window`,
  `window_of_word`, `qstar_iff_word`, `jmax`, `akill`), instantiated at
  m11 (`WordLegal11.L11/jmax11/akill11`), m13, m17 (`WordLegal13/17`).
- WHERE: constructor.md R89; formalist.md R30.1, R31.3; mechanic.md standing rule 34.
- LIMITS: `hper` (periodicity of the gap residues) is needed for the `J_max` half;
  `L(M)` itself is a dictionary fact the reduction takes as input.

### same-tooth lemma (R90)
- STATEMENT: a padded middle leaves the tooth fixed, a literal middle flips it, so the
  middle span `x_{J-1} - x_1 = (t_{J-1} - t_1)c` is `0 mod q'` iff the number of
  NON-PADDED middles is even. For a LITERAL even-`J` window all `J-2` middles are
  non-padded: the first and last struck opening sit on the SAME TOOTH and the middle span
  is `>= ((J-2)/2) q'`.
- CALCULATES: the even/odd separator as an arithmetic fact rather than a count; the
  minimum middle sum `k q'` (`J` even) or `k q' + a` (`J` odd), `k = floor((J-2)/2)`.
- STATUS: kernel-checked (`WordLegal.same_tooth`, `same_tooth_window`,
  `literal_even_span`, needing only `2c != 0`, discharged from `6c = 1`); checked on every
  realised legal word at every machine with an exact source (38 words, 0 violations).
- WHERE: constructor.md R90, R82 Theorem A; formalist.md R30.1.
- LIMITS: literal windows only; the two padded even-`J` maximisers `(12,37)` at m31 and
  `(41,14)` at m37 have middle sums `12 mod 37` and `14 mod 41`.

### the flank envelope must collapse (par trading, exact form)
- STATEMENT: a literal `J`-window's span exceeds its flank sum by an amount growing by
  `q'` every two levels, so the per-J analogue forces `Phi_J <= F_2 + s_min - m_min(J)` —
  the flank envelope collapses at rate `q'` per two levels. That is why the DEEP layers
  are the cheap ones.
- CALCULATES: an upper bound on the flank sum available at each depth.
- STATUS: theorem from T1-T3 (R82 Theorem A); measured `Phi_J <= F_2 - b` at every
  non-empty literal even-J cell, margins +5 (m19), +10 (m29), +9 (m31).
- WHERE: constructor.md R82, R92, R30.
- LIMITS: says nothing at the padded letter.

### peel bound / free reduction
- STATEMENT: deleting either flank of a legal `J`-window leaves a legal `(J-1)`-window,
  so `Q*_J <= Q*_{J-1} + min(g_L, g_R)` at the argmax; in particular
  `g_L + w + g_R <= F_2(M) + min(g_L, g_R)` with no hypothesis at all. Hence a violating
  depth-3 window must have MIN FLANK `> s_min`.
- CALCULATES: discharges the triple inequality automatically at any triple whose smaller
  flank is `<= s_min`; the whole depth-3 obligation reduces to triples with both flanks
  above `s_min` (measured: 0, 0, 16, 4, 24, 131, 205, 317 such triples at m11..m37).
- STATUS: proved (R82 Theorem D, R78); the min-flank necessity asserted at all 27,570
  counterfactual members.
- WHERE: constructor.md R78, R82; lateral.md item 71.
- LIMITS: short by exactly `F_2 - a` at `J = 4` — the free reduction does NOT discharge
  `J >= 4`, which is a flank-PAIR statement about one specific 2-letter word.

### increment law
- STATEMENT: `F(M + q') - F_2(M) <= s_min(q') = min(2u', q' - 2u') = 2u'`. Reading: one
  more link buys at most one small letter over the old two-gap maximum, unless the link is
  padded (worth a full `q'`).
- CALCULATES: the new record from the old machine's adjacent-pair record plus one number.
- STATUS: holds at 11 of 12 corpus steps (differences 0,2,0,3,4,3,20,1,0,...,+2 at
  53->59 against caps 4,6,6,8,10,10,12,14,14,16,18,20); fails at the padded 31->37 by +8.
  Kernel-checked at all six literal steps (`Increment.increment_law_literal_steps`,
  1749 jobs), with LOWER halves as exhibited CRT slots.
- WHERE: constructor.md R76, R68; mechanic.md C38, C43; formalist.md R28.1; lp-duality.md.
- LIMITS: NOT generic — violated by 13-22% of the tooth-counterfactual family (0-6.5% once
  the incoming gear's tooth is pinned to `round(q'/6)`), so no proof from "same gears, same
  density, symmetric teeth" can exist.

### deletion ladder
- STATEMENT: `F_j(M) <= F(M + the next j-1 primes)`. PROOF: take the window realising
  `F_j(M)`; it has `j-1` interior openings; `P(M)` is invertible mod each of the next
  `j-1` primes, so CRT gives a translate in which the `i`-th interior opening is congruent
  to gear `q_i`'s own tooth, for every `i` at once — every interior dies.
- CALCULATES: free span caps: `F_2(43) <= F(47) = 118`, `F_3(43) <= F(53) = 145`,
  `F_4(43) <= F(59) = 161`, `F_2(53) <= F(59) = 161`, `F_2(41) <= F(43) = 103`.
- STATUS: proved; asserted at all 32 `(M,j)` pairs where both sides are known exactly, one
  attained with equality (`F_2(17) = 25 = F(19)`), tightest non-equality
  `F_2(37) = 90` vs `F(41) = 91`.
- WHERE: mechanic.md K3; constructor.md R93.
- LIMITS: LOGICALLY CIRCULAR as an induction step (it prices `F_2(M)` by the very `F` the
  rung is certifying), and its slack thins — `F(M+q') - F_2(M)` is 3 at 29->31, 1 at
  37->41, 0 at 41->43.

### depth-0 lemma (openings only ever get merged, never split)
- STATEMENT: `D_m(M) subset D_m(M + q')` for every prime `q' > 2(m+1)`: a realised
  `m`-tuple of consecutive gaps stays realised when a gear is added. PROOF: the tuple's
  `m+1` exposed offsets forbid at most `2(m+1) < q'` phases of the new gear; `P(M)` is
  invertible mod `q'`, so some lap has an admissible phase, every point survives and the
  `m+1` openings are still CONSECUTIVE (a new opening between them would be an old one).
- CALCULATES: 16.7% of a 1.4M-decision census answered YES with no solver at m41.
- STATUS: proved, three lines; checked at arities 2,3,4 at all six exact pairs 13->17 ..
  31->37 and at arities 5,6,7 at the small steps; the hypothesis is SHARP (first failure
  at `m = 6,7,8,9` for `q' = 11,13,17,19`).
- WHERE: mechanic.md C40; harvester.md 14b (Ziller 2020 Prop 2.7 is the one-class ancestor).
- LIMITS: monotonicity of the dictionary only — says nothing about which NEW tuples appear.

### the two-class copy picture (Holt-Rudd in two classes)
- STATEMENT: the new machine's period is `q'` copies of the lower period; copy `i` deletes
  the openings whose class lies in the two-class set at phase `r_i = i P_M mod q'`.
  (a) EACH LOWER OPENING IS HIT IN EXACTLY TWO COPIES, one per tooth. (b) A window of
  `j+1` consecutive lower openings with offsets `X` is SPARED in exactly
  `q' - |X u (X+s)|` copies; if its span is `< s_min(q')` all `2(j+1)` hitting copies are
  distinct. (c) MULTIPLICITY: a run of `k >= 2` consecutive lower openings is hit ENTIRELY
  in 0 copies if its gap word is illegal, 1 if legal with a literal letter, 2 if legal and
  every letter is padded.
- CALCULATES: the survival factor `q' - 2(j+1)` below the threshold (= the paired-Holt
  recursion); the threshold is SHARP — the smallest span at which two points of one window
  are hit in one copy is 4,6,6,8,10 at m11..m23, exactly the smallest realised legal letter.
- STATUS: proposition, three-line CRT arguments, script-verified exactly at m11..m23
  (`research/hr_twoclass_r30.py`, all assertions green).
- WHERE: harvester.md 14 "Follow-on: Holt-Rudd in two classes".
- LIMITS: gives NO inequality on `L(M)` — the multiplicity does not decrease with `k`, so
  the count converts "M realises this word" into "it dies in 1 or 2 copies" and never
  decides realisation.

### firing law
- STATEMENT: inside a chain of gear `q'` the spacing word's first entry fixes the
  orientation and hence a SINGLE firing residue (word starting with `s` fires iff
  `p = -u mod q'`, starting with `q'-s` iff `p = +u`), density `1/q'` per window; across
  the new machine's full period every fuel site fires EXACTLY ONCE, at
  `j = (fire - p) P_old^{-1} (mod q')`.
- CALCULATES: realised `k`-chains per new period `= N_k` exactly.
- STATUS: exact, zero violations over 13,062 sites.
- WHERE: lateral.md item 15.
- LIMITS: alignment is a DENSITY factor, never a COUNT factor — there is no suppression
  multiplier to find.

### onset gate for padded links
- STATEMENT: a padded link's interior gap is a positive multiple of `q'` and is one of
  `M`'s own gaps, so `q' <= F(M)` is NECESSARY for any padded link to exist. It is NOT
  sufficient (`supply(29,41) = 0` despite `F(29) = 43 >= 41` — a spectrum hole).
- CALCULATES: the exact step at which padding can first appear; `supply(M,q') = hist_M[q']`
  is ONE LOOKUP.
- STATUS: kernel-checked (`TierA.onset_gate`, `padding_at_most_one_below_onset`);
  sufficiency refuted by census.
- WHERE: formalist.md 2.14; mechanic.md C12, R9; constructor.md R23.
- LIMITS: padding count is bounded only by budget arithmetic `p <= F/q' + 5/6`, which GROWS.

---

## E. HOW DEEP THE ALIGNMENT CAN GO

### literal cap theorem
- STATEMENT: a literal chain's walk must stay in the corridor `E mod 35`, and the maximal
  run is a function of `q' mod 210` ALONE: cap 2 at 24 classes, 3 at 4, 4 at 14, 6 at 6
  (`q' = 37,53,83,127,157,173 mod 210`); there is no class of cap 5. LITERAL CHAINS HAVE
  AT MOST 6 MEMBERS, AT EVERY GEAR, FOREVER.
- CALCULATES: the finite word list of clause (A) — alternating words over `{2u', q'-2u'}`
  of length `1 .. capC-1`, two per length — from `q' mod 210` alone.
- STATUS: kernel-checked BOTH WAYS (`LiteralCap.literal_chain_le_six`,
  `cap_six_classes_sharp`; `LiteralCapTable.cap_table_maximal`, `cap_table_realized`,
  `no_cap_five`, `cap_spectrum_counts`); verified against every prime to 5000.
- WHERE: constructor.md R20, R26; formalist.md 2.11, 2.12.
- LIMITS: caps only the LITERAL part and never predicts realised arity (X29) — at m41
  litcap = 4 while the literal 2-word count is exactly 0, and at m37 litcap = 2 while
  `A_kill = 3` (forced padded).

### bare-alternation admissibility lemma and the set S
- STATEMENT: if neither `X_A = {0, a, q', q'+a}` nor `X_B = {0, b, q', q'+b}` admits a
  translate inside `E_5` and `E_7` (equivalently inside `E_35`), then `M` has no realised
  bare legal word of length 3, so `L_bare(M) <= 2`. Generally
  `L_bare(M) <= PSORD(q' mod 210) <= 5`, with `PSORD = 1` at 24 classes, 2 at 4, 3 at 14,
  4 at NONE, 5 at exactly `{37,53,83,127,157,173}`; `S = {c : PSORD(c) <= 2}` has 28
  classes (density 7/12).
- CALCULATES: `L_bare` at any machine from `q' mod 210` alone — at m41/m43 the four bare
  words are refuted with NO search at all.
- STATUS: kernel-checked (`BareAlt.no_gapWord`, `no_bare_run`, `no_bare_run_ge`,
  `bareAlt_inadmissible_iff`, `S_card = 28`, `psord_le_five`, `psord_ne_four`,
  `inadmissible_iff_capC`); instantiated at m23/m37/m41/m43 (`BareAltInst`); three
  independent vehicles agree element for element.
- WHERE: constructor.md R103; formalist.md R31.0-R31.2; lateral.md item 79.
- LIMITS: bounds `L_bare`, NOT `L` — at m37, m41, m43, m53 the machine's realised `L`
  strictly exceeds the bare cap, and those excesses are exactly the padded letter `q'`.

### A_relax / PSORD (uniform order)
- STATEMENT: `A_relax(M) <= 5` for every machine `{5..y}` with `y >= 7`, and `<= 4` unless
  `q' = 37, 53, 83, 127, 157, 173 (mod 210)`. Proof is arithmetic: for the `m`-point
  alternation `X = {0, a, q', q'+a, 2q', ...}`, `X mod g` is determined by
  `(a mod g, q' mod g)` and `3a = q' -+ 1`, so everything at gears 5 and 7 is a function
  of `q' mod 210`; enumerate the 48 invertible classes. Adding gears 11 and 13 refutes
  NOTHING further (60/60 and 720/720 refinements stay at order 5).
- CALCULATES: a machine-free cap on the alternation order from one residue.
- STATUS: kernel-checked as 48 classes mod 210 (`AlternationOrder.ps_min_le_five`,
  `ps_min_five_iff`, `ps_min_four_iff`, `ps_min_counts`), with `A_relax <= psMin` carried
  as an explicit hypothesis `hred`.
- WHERE: constructor.md R74; formalist.md R29.2.
- LIMITS: caps a PROXY — `A_relax` tests one candidate cycle, while `A_m` is nilpotent only
  when EVERY legal cycle is broken, and padded letters are T3-transparent, so
  `N(37) = 3 > A_relax(37) = 2` and `N(41) = 3 > 2`.

### psMax = capC — two invariants are one object
- STATEMENT: the phase-saturation order (translates at gears 5 and 7 separately, the
  MAXIMISING convention) equals the corridor literal cap (a walk in `E mod 35`) at ALL 48
  classes: `psMax c = capC c`. Two invariants found five rounds apart by different vehicles
  are one object; by CRT a translate fits inside `E_5` and `E_7` separately iff it fits
  inside `E_35`.
- CALCULATES: either object from the other; the distribution `{2:24, 3:4, 4:14, 6:6}`.
- STATUS: kernel identity at all 48 classes (`AlternationOrder.ps_max_eq_capC`).
- WHERE: formalist.md R29.2; constructor.md R74.
- LIMITS: the two differ only in the QUANTIFIER over start letters — `litcap` maximises
  (chain existence), the order minimises (one broken window kills a cycle) — so `S` (28
  classes) is not R74's order-2 set (24 classes).

### CORRCAP — where gears 5 and 7 stop capping length
- STATEMENT: `CORRCAP(q', F)` = the longest T3-legal word with values `<= F` whose
  prefix-sum walk stays inside `E mod 35`, computed by an automaton on `35 x 3` states.
  Values 4, 2, 3, 5, 25, 25, 11, 5, INFINITE at 19->23 .. 53->59. The mechanism: padded
  letters contribute steps `j q' mod 35` and `gcd(q',35) = 1`, so as `F/q'` grows those
  fill `Z_35` and the corridor stops constraining.
- CALCULATES: the exact step at which no fixed set of small gears can cap the order again
  — 53 -> 59.
- STATUS: exact automaton computation, reproduced at all nine cells by two lanes.
- WHERE: constructor.md R75, R100, R105.
- LIMITS: `F/q'` grows without bound (1.1, 1.2, 1.4, 1.6, 2.1, 2.1, 2.2, 2.2, 2.5 at
  19->23 .. 53->59), so this is a statement about small machines only.

### spectrum bound on L
- STATEMENT: a realised legal word of `m` letters is the middle of a window of consecutive
  openings whose span is `<= max_J Q*_J = F(M+q') =: G`; T3 makes two consecutive nonzero
  letters sum to `>= q'`; with `p` padded letters and `n = m - p` nonzero ones
  `span >= p q' + floor(n/2) q' + [n odd] a_min`. Hence with `T = floor((G-2)/q')`:
  `L(M) <= 2T + 1` (SIMPLE), `L(M) <= 2T + 1 - p` (letter-aware), and
  `L(M) <= max(2T, 2 floor((G-2-a_min)/q') + 1)` (PARITY) — i.e. `L <= 2F(M+q')/q' + 1`.
- CALCULATES: a cap on `L` from the spectrum alone; TIGHT at m11, m13, m29; beats EXPCAP
  at m19, m37, m41, m43, m53; caps the padded half, `L_pad(M) <= 2T`.
- STATUS: theorem given R68 and T3, unconditional; verified at all 12 corpus steps and at
  every one of 165,584 counterfactual rows, 0 violations.
- WHERE: lateral.md item 84; docs/novel/spectrum-bound-on-L.md.
- LIMITS: `L` bounded by an absolute constant is probably FALSE — the bound grows with
  `F/q'`; it closes the chain only under a Jacobsthal-square condition
  `8F <= q'^2 - (eps+12)q' + 16` (true at 8 of 13 corpus steps).

### L = max(L_bare, L_pad)
- STATEMENT: every realised legal word is bare or it is not, so `L(M) = max(L_bare, L_pad)`;
  with `L_bare <= PSORD <= 5` PROVED, requirement (B) is exactly `L_pad(M) <= c_pad`.
  Measured `L_pad = 0,0,0,1,1,1,2,2,2,2,3,3` at m11..m53 — it takes every value from 0 to 3
  and it grows; `L > L_bare` at four corpus machines (m37, m41, m43, m53), each excess the
  single padded letter `q'`.
- CALCULATES: splits the depth question into a closed residue condition plus one open half.
- STATUS: decomposition trivial; the `L_pad(53) = 3` entry is a consequence of the theorem
  plus a recorded `A_kill`, at a machine no census reaches; `L_pad(47) = 3` measured.
- WHERE: constructor.md R104, R105.
- LIMITS: the term that makes `L_pad` the cover half is the ALPHABET SIZE `~3F/q'`, and
  nothing bounds it.

### cover-half order N(M)
- STATEMENT: `N(M)` = the smallest `m` at which the history abstraction `A_m` is acyclic,
  `max(2, A_relax) <= N <= A_res`. Measured `N = 2,2,2,3,2,3,4,3,3` at m11..m41. The cycles
  that push `N` above `A_relax` are PADDED 2-cycles (at m41 `[43] -> [29] -> [43]`), and
  they die at order 3 because T3-transparency is not T3-legality once the window is long
  enough to see two literal letters.
- CALCULATES: the abstraction order a certificate needs at a given machine, scan-free by CRT.
- STATUS: exact, decided by CRT at m11..m41 (`research/cover_order.py`), reproducing R75's
  hand-computed row by a different vehicle.
- WHERE: constructor.md R75, R85.
- LIMITS: the vehicle stops at m43 (arity-1 and arity-2 CRT decisions are the expensive
  end); whether `N` is bounded in general is open.

---

## F. THE LAYERED WALK: HIT LAW, CHAIN LAW, NESTED FORM, PHASE REDUCTION

### hit law
- STATEMENT: gear `g` hops at the lower landing `x` iff `x = +-u_g (mod g)` — the walk to
  the next opening of `{5..g}` moves past `x` exactly when `g` strikes `x`.
- CALCULATES: the layer's contribution to the walk as a single residue test; the fraction
  of landings with no hit is exactly `1 - 2/g` (0.8182, 0.8462, 0.8824, 0.8947, 0.9130 at
  `g = 11,13,17,19,23`).
- STATUS: exact, holds without exception over the full period of every machine `{5..23}`;
  kernel form `AnchorChain.teeth_eq_phase`, `hop_zero`.
- WHERE: anchor-235.md 9d, 9b; formalist.md R29.1.
- LIMITS: one layer at a time; the residual the search cannot express is the lower walk.

### chain law
- STATEMENT: two consecutive lower openings `x < y` are BOTH hopped by `g` iff
  `y - x = 0` or `+-d_g (mod g)`, `d_g = 2u_g mod g`. Equivalently: a set of openings lies
  in one two-class set `{r, r+d}` iff every pairwise difference is `0` or `+-d`. So the
  lower gap sizes that can carry a second hop at layer `g` are the short list
  `{d_g, g-d_g, g, g+d_g, 2g-d_g, ...}` cut at `F_{g-} + 1`, and the hop chain of `g` from
  `x` is exactly the maximal run of consecutive lower gaps after `x` lying in those classes.
- CALCULATES: the chain depth per layer from the admissible gap list. Measured
  (layer, `d_g`, lower `F`, admissible gaps `<= F+1`, realised, depth):
  `7, 5, 1, {2}, {2}, 2`; `11, 4, 4, {4}, none, 1`; `13, 9, 6, {4}, {4}, 2`;
  `17, 6, 10, {6,11}, {6,11}, 2`; `19, 13, 17, {6,13}, {6,13}, 2`;
  `23, 8, 24, {8,15,23}, {8,15,23}, 3`.
- STATUS: exact over the full period of every machine `{5..23}`, no exception; kernel-checked
  in both directions and for every gear at once (`AnchorChain.chain_law`).
- WHERE: anchor-235.md 9d; formalist.md R29.1 (i).
- LIMITS: the DEPTH is not an algebraic consequence — a run in a two-class set alternates
  freely, so `D_g` is a fact about the lower gap SIZES, a per-machine measurement (stated
  as the honest boundary in `AnchorChain.lean`'s own header).

### neighbour-of-a-hit law
- STATEMENT: the neighbour of a `g`-hit is `g`-free — the two teeth are `d_g` apart and
  `d_g = 3^{-1} mod g` is never `+-1` (that would need `g | 2` or `g | 4`) — and the other
  gears do not see `g` at all, so EXACTLY
  `P(x+1 open | x is a g-hit) = P(open) * g/(g-2)`.
- CALCULATES: the conditional opening rate next to a known hit: measured 0.2342 =
  0.2139 x 23/21 at `{5..23}` for `g = 23`, 0.2390 for `g = 19`, 0.2994 for `g = 7`, all
  exact to four places.
- STATUS: exact (identity plus full-period measurement); kernel-checked for every gear at
  once from `6u = 1` alone (`AnchorChain.neighbour_of_hit`).
- WHERE: anchor-235.md 9e; formalist.md R29.1 (ii).
- LIMITS: knowing one gear's hit buys the factor `g/(g-2)` and NOTHING else — the neighbour
  of a hit is open LESS often than the neighbour of a random blocked column (0.2481) and
  much more often than the neighbour of an opening (0.0881: openings repel). "One side of a
  hit is open" is not a way to find openings.

### nested residue formula
- STATEMENT: with `x = s + W_M(s)` the lower landing,
  `W_g(s) = W_M(s) + h1 (1 + W_M(x+1)) + h1 h2 (1 + W_M(x1+1)) + ...` (`D_g` terms),
  `h1 = [x on g's teeth]`, `x1 = x + 1 + W_M(x+1)`, `h2 = [x1 on g's teeth]`, ... — after a
  hit the walk re-enters the whole lower machine at the new landing.
- CALCULATES: the distance to the next opening exactly. With `D = 2,1,2,2,2,3` for
  `g = 7,11,13,17,19,23` the capped formula equals the true walk at EVERY column of the
  full period for `{5,7}` through `{5..19}` (periods 35 to 1,616,615).
- STATUS: exact at every column of the full period to `{5..19}`; the recursion step is
  kernel-checked over an abstract machine (`AnchorChain.hop_iter`, `hop_one`, `hop_zero`).
- WHERE: anchor-235.md 9f, 9b; formalist.md R29.1 (ii).
- LIMITS: size is `prod (1 + D_g)` bottom evaluations = 3, 6, 18, 54, 162, 648, 1944 to
  `{5..29}` — exponential in the number of layers; no cross-layer cancellation was found.
  (Cheap to EVALUATE lazily: exactly `1 + (crossed columns not on gear 5's teeth)` bottom
  evaluations, mean 1.37..2.36 at `{5,7}`..`{5..19}`.)

### phase reduction (the g copies realise every deletion phase once)
- STATEMENT: the full period is `g` copies of the lower period, copy `j` shifted by
  `j P_M`; since `P_M` is invertible mod `g`, `j -> -u - j P_M` is a BIJECTION onto `Z_g`,
  so the copies realise every deletion phase exactly once, copy `j` deleting the lower
  openings whose class lies in the two-class set `{r, r + d_g}`.
- CALCULATES: the layer needs the lower opening residues mod `g` ONCE, not the full period:
  `D_g` = longest run of consecutive lower openings with residues in one set `{r, r+d_g}`,
  and `F_g + 1` = max over such runs of (gap before) + (run span) + (gap after).
- STATUS: kernel-checked machine-free (`AnchorChain.copy_phase`, `phase_bijective`);
  script-verified exact at `{5..7}` .. `{5..29}` (F = 4,6,10,17,24,33,42, all equal to the
  corpus, the last from 7,952,175 lower openings instead of a 6.5e9-column period — 819x
  smaller) and carried to `g = 31, 37, 41` (records 58, 88, 91, all matching).
- WHERE: anchor-235.md 9f; mechanic.md C50; formalist.md R29.1 (iii).
- LIMITS: the reduction is CONCEPTUAL, not economic, in a slot-walk kernel encoding
  (86,173 kernel column tests against the direct scan's 85,085 — 1.01x); to collect the
  Python saving the kernel needs the opening LIST as an object.

### the record law at 31, 37, 41 (record from one lower period)
- STATEMENT: `max over two-class runs of (gap before + run span + gap after)` on ONE lower
  period equals the new machine's record: 58 = F(31), 88 = F(37), 91 = F(41). The attaining
  run length is `k_win`; the `L = 1` row is `F_2` of the lower machine every time
  (55 = F_2(29), 68 = F_2(31), 90 = F_2(37)), because a run of one deleted opening merges
  exactly two lower gaps.
- CALCULATES: `F(M+g)` and `k_win` from the lower opening sequence; the per-`L` rows are
  `Q*_{L+1}(M; word-legal for g)` WITH PADDING INCLUDED (m37 pass reads off
  `Q*_2(31)=68, Q*_3(31)=85, Q*_4(31)=88, Q*_5(31)=68`, maximum 88 = F(37)).
- STATUS: exact full lower period at 31 and 37; at 41 a deliberate 36.9% sweep whose two
  headline answers are still exact (sample gives `D_41 >= 3`, census gives `<= 3`; sample
  exhibits 91, `F(41) = 91` caps it). All three survivors re-derived at the TARGET machine
  column by column.
- WHERE: mechanic.md C50; anchor-235.md 9f.
- LIMITS: what the remaining coverage would buy is only the exact per-`L` rows.

### D_g = A_kill, an identity
- STATEMENT: `D_g` (the chain depth of the layered walk) EQUALS `A_kill(M -> g)` (the fuel
  census's largest co-deletable tuple): `D_31 = 4 = A_kill(29->31)`, `D_37 = 4`,
  `D_41 = 3`, and `D_17 = D_19 = 2, D_23 = 3, D_29 = 2` reproduce the small steps. Both
  count maximal runs of consecutive `M`-openings that ONE phase of `g` deletes; the fuel
  legality condition "prefix-sum range `<= 1`" IS "all in one two-class set".
- CALCULATES: either quantity from the other; with R89, `D_g = L(M) + 1`.
- STATUS: identity, not a measurement; two vehicles built four rounds apart in two
  languages agree 7 for 7.
- WHERE: mechanic.md C50 (1); harvester.md 14 (L4).
- LIMITS: `D_g` bounded is OPEN.

### phase-reduction record law in the kernel at machine 17
- STATEMENT: the 17 phases of the lower machine `{5,7,11,13}` (period 5005, 1485 openings)
  each give a maximum merged gap `mg r`, with `max_r mg r = 18 = F(17)`, attained at
  `r = 2`; and phase `r` IS machine 17 shifted by `tOf r = ((31-r)*5) mod 17` whole lower
  periods (`5005 = 7 mod 17`, `7^{-1} = 5 mod 17`).
- CALCULATES: `F(17) = 18` EXACTLY (a new corpus fact — the corpus carried only the upper
  half), with the witness `openings 117 and 135, nothing open between`.
- STATUS: kernel-checked at BOTH ends (`AnchorRecord17.record_max`, `surv_shift`,
  `phase_is_machine`, `gap18_realized`, `F17_eq_18`; 17 per-phase `decide +kernel` in
  `AnchorRecord17Core`).
- WHERE: formalist.md R29.1 (iii).
- LIMITS: the identity "max over phases = F + 1" is verified at both ends rather than
  DERIVED from one to the other — that needs a correctness proof of the walk against
  `Machine17.nextOp`, not written.

---

## G. WHEN IN THE WINDOW AN OPENING LANDS

### the window is the opening stretch of the periodic pattern
- STATEMENT: at gear `q` the gears in play are `7..q`, their untouched columns are a known
  periodic pattern, and the kernel's window (columns `(q/6, W]`, `W = (q'^2-1)/6`) is the
  OPENING STRETCH of that pattern — it includes every lower section.
- CALCULATES: the window against the record. Full-period figures (blocked-column counts):
  `q=7: period 35, worst run 4, W=12, 4 open in window, run entering the window 2`;
  `11: 385, 6, 8, 2, 4`; `13: 5005, 10, 20, 7, 4`; `17: 85085, 17, 12, 2, 4`;
  `19: 1616615, 24, 28, 4, 11`; `23: 37182145, 33, 52, 8, 7`.
- STATUS: exact full-period computation (`research/anchor235/period_vs_window.py`).
- WHERE: anchor-235.md 7.
- LIMITS: the worst run of the pattern already exceeds the current SECTION at `q = 17`
  (17 against 12) — so existence in the section is POSITIONAL.

### where the worst stretches sit (never at the window)
- STATEMENT: the worst runs sit DEEP in the period, in mirror pairs at positions `k` and
  `P - k` (fractions 0.3-0.7 of the period, or at the period's ends), never at the window.
  The run the pattern has AT `q^2/6` is short: at most 0.663 of the section (worst at
  `q = 137`) to `q = 5000`.
- CALCULATES: the provenance of the section's survivor — it is not that the record is small,
  it is that the record is somewhere else.
- STATUS: measured to `q = 5000`; mirror pairing exact (record gaps mirror-paired at every
  machine).
- WHERE: anchor-235.md 7; word-tree.md 78 (T2'); lateral.md item 12, 46.
- LIMITS: positional, so it is a statement about the section and not about `F`.

### against the WHOLE window the position drops out
- STATEMENT: `F(q) < W(q) - q/6` FORCES a survivor in the window whatever the pattern does
  at `q^2`. Measured `F/W = 0.25` flat from `q = 5` to 53; `F(59) = 161` against `W = 620`.
  If (D) holds at every rung then `F(y) <= sum_{q<=y} q ~ y^2/(2 ln y)` against
  `W ~ y^2/6`, so `F/W <= 3/ln y < 1` for `y > 20` — a survivor in every window, twins
  infinite.
- CALCULATES: the whole tolerance route's arithmetic; `F(2,y) <= 354 + alpha(S(y) - 328)`
  checked at every prime in `[53, 10^6]`, zero failures, worst ratio 0.6557 at `y = 113`.
- STATUS: exact arithmetic given the ladder; the ladder itself is (D), open.
- WHERE: anchor-235.md 7; constructor.md R14; killer-spec.md 40.
- LIMITS: rests on (D) at every rung, which is measured true at every computable step and
  is not a law.

### section view: every section holds an aligned column
- STATEMENT: per section `(q^2, q'^2)`, taking nothing from lower windows, EVERY section
  to `q = 5000` holds an aligned column (a twin). The minimum per section rises
  2, 3, 6, 7, 19, 21, 42, 51, 68 across the bins 5-50 .. 4000-5000; the longest blocked
  run of anchor-open columns inside a section grows like `q^0.51` while the section grows
  like `2q ln q`, so run/section FALLS: median 0.235 -> 0.020, worst 0.544 (29 -> 31) ->
  0.085. Aligned count = anchor-open x `prod_{7<=g<=q}(1-2/g)` x 0.66-1.0.
- CALCULATES: how much of the new part of the window is filled and how thin the worst case
  gets.
- STATUS: exact to `q = 5000` (`research/anchor235/section_trend.py`).
- WHERE: anchor-235.md 4.
- LIMITS: "every section holds an aligned column" is STRONGER than the twin conjecture (a
  dead section is a twin gap `>= 4 sqrt(x)`) — recorded as an overstatement corrected.

### the walk from q^2 (how a twin is surfaced)
- STATEMENT: starting at the column holding `q^2` and stepping by residue tests only — no
  primality test anywhere — the walk lands on an opening of `{5..q}`, and that opening IS a
  twin pair. From `q = 37, 97, 499, 997, 4999, 10007, 100003` it lands at
  `1427|1429` (10 columns), `9419|9421` (2), `249131|249133` (22), `994067|994069` (10),
  `24990239|24990241` (40), `100140119|100140121` (12), `10000600481|10000600483` (79
  columns, in a section of 533,392).
- CALCULATES: the address of the first twin above `q^2` by residue arithmetic alone.
- STATUS: exact (`research/anchor235/slot_walk.py`); all landings verified twins.
- WHERE: anchor-235.md 6.
- LIMITS: existence for a FIXED gear set is CRT (`prod (q-2) x 3/5` open columns per
  period, never zero); for the GROWING gear set it is the conjecture.

### the layered walk from q^2, for every prime to 5000
- STATEMENT: running the layered closure `W_g = W_{g-} + hits of g` recursively from the
  column holding `q^2` under gears `5..q` for every prime `q <= 5000` (667 walks), EVERY
  landing is a twin prime pair; walk length median 19, maximum 265 at `q = 4637` (second
  187 at 2593 and 4003); between 1 and 44 layers hop per walk; total hops equal the walk
  length in every walk (an identity — each traversed column is counted once, at the layer
  of its smallest blocking gear).
- CALCULATES: the distance from `q^2` to the first twin, layer by layer.
- STATUS: exact for 667 primes (`research/anchor235/layered_walk.py`).
- WHERE: anchor-235.md 9c.
- LIMITS: the walk is closed by a recursion of depth `pi(q)`; no formula collapsing the
  recursion has been found.

### the first opening past q^2 (positional statistics)
- STATEMENT: the first twin sits a median 18 columns past `q^2`, maximum 264 at
  `q = 4637`, to `q = 5000`; the position of an open whole cycle inside a section is
  uniform (quartiles 0.24, 0.48, 0.74).
- CALCULATES: the expected provenance depth of a section's survivor.
- STATUS: measured to `q = 5000`.
- WHERE: anchor-235.md 4, 5.
- LIMITS: a measurement, not a law; the maximum is a record on a curve.

### whole anchor cycles inside sections
- STATEMENT: a cycle `j` (three whole twin slots) is untouched by gear `q` iff `j mod q`
  avoids six residues `((q m - 11) div 30) mod q`; under gears `7..Q` the open cycles are a
  fixed pattern of period `prod q` with `prod (q-6)` open cycles per period. Against the
  window sections to `10^8`: 1226 sections, 1088 with no open cycle, 121 with one, 16 with
  two, 1 with three; the share holding one rises 0% (`q < 100`) to 13% (3000-10000);
  longest dry stretch 50 sections (`q = 7079..7549`).
- CALCULATES: where a whole prime sextuplet lands relative to the section boundaries.
- STATUS: exact below `10^8` (156 such cycles, all on the rule and none off it).
- WHERE: anchor-235.md 5.
- LIMITS: the section is NOT the natural unit for whole cycles; existence for the growing
  gear set is the Hardy-Littlewood sextuplet conjecture, stronger than twin primes.

### onset law (when the first double column appears)
- STATEMENT: `L0(y)`, the lag from the window's start to the first column with BOTH members
  composite, satisfies `L0(y) <= L* = 27129` for every `y`, unconditionally (via
  Montgomery-Vaughan `pi(x+H) - pi(x) < 2H/ln H`, since `6L*+2 > e^12`). Under Condition X
  the onset prefix is perfectly fragile (`n2 = 0` there).
- CALCULATES: an absolute bound on how deep into a window the first double column can be.
  Measured over 442 windows `13 <= y <= 3163`: max `L0 = 17` (at `y = 13`), `L0 = 0` in
  153/442, a twin precedes the first double in 132/442; the first double sits at column
  ~2-4 with no growth in `y`.
- STATUS: unconditional theorem of the programme (R7); measurements exact.
- WHERE: constructor.md R7; mechanic.md C3.
- LIMITS: 310 of 442 real windows have NO twin in the onset prefix (X2), so the onset scale
  is not itself a contradiction.

### the first double column is k = 20
- STATEMENT: the first column in all of `W` with both members composite is `k = 20`
  (`119, 121`); every column `k <= 19` has a prime member. So under Condition X the demand
  `n2(t)` has no supply before `t = 20`.
- CALCULATES: an exact floor on the prefix.
- STATUS: exact.
- WHERE: constructor.md R5; lateral.md item 5.
- LIMITS: a prefix statement; the prefix-pigeonhole refutation reaches only `t <= 4`.

---

## H. RECORDS: WHAT A RECORD STRETCH IS MADE OF

### what a new gear must do to lengthen the record
- STATEMENT: the record of `M + q'` is an old stretch of consecutive openings
  `x_0 < ... < x_J` with every interior opening struck by `q'` and both END openings
  surviving; kills `= J-1`. Consecutive kills differ by `0` (same tooth) or `+-2u'`
  (opposite teeth) mod `q'`. "Both sides of a stretch" is the two-kill case: the two
  openings bordering a big old gap both sit on teeth — the SAME tooth when the gap is a
  multiple of `q'` (118 = 2 x 59 at 53->59), OPPOSITE teeth when it is `+-2u' mod q'`.
- CALCULATES: the decomposition of every record. `{5,7,11}+13`: `7 -> 11`, old gaps
  `[6,5]`, 1 kill; `{..13}+17`: `11 -> 18`, `[5,11,2]`, 2 kills, residues `[3,14]`;
  `{..17}+19`: `18 -> 25`, `[7,18]`, 1 kill; `{..19}+23`: `25 -> 34`, `[4,8,15,7]`,
  3 kills; `53->59`: `145 -> 161`, `[10,118,33]`, 2 kills, same tooth.
- STATUS: exact at every computable rung (`research/anchor235/extension.py`).
- WHERE: anchor-235.md 8.
- LIMITS: PARITY IS NOT A CONSTRAINT — old gaps are mixed parity and `q'` is odd, so the
  two opposite-tooth differences have opposite parity and any gap parity has a tooth
  arrangement. The obstacle is arithmetic mod `q'`, never parity.

### the lower side of the record is forced
- STATEMENT: `F'(M+q') >= F_2(M)` with no computation: the middle opening of any two
  consecutive old gaps dies in exactly 2 of its `q'` lifts; the ends survive (run = both
  gaps) or die too (run longer). The best one-kill run equals `F_2` at every rung.
- CALCULATES: an unconditional floor on the new record.
- STATUS: exact at every rung `{5}+7` .. `{5..23}+29` (`record_decomp.py`).
- WHERE: anchor-235.md 9; mechanic.md K3 (`r = 1` case).
- LIMITS: floor only; the upper side `F' - F_2` is the whole content.

### interior gaps of the record chain are minimal-stride
- STATEMENT: the interior gaps of the record chain are exactly `+-2u' mod q'`, NEVER a
  multiple of `q'` (kills alternate teeth at the minimum stride). Since `3 x 2u' = 1 mod q'`,
  `s_min = (q' +- 1)/3`, so a chain of `m` kills spends at least `(m-1)(q'-1)/3` columns
  on its interior.
- CALCULATES: `F' - F_2 = 1,0,0,2,0,3,4` against `s_min = 2,4,4,6,6,8,10` at the eight
  rungs — the increment law over every kill chain, not only the record.
- STATUS: exact over the full period at all eight rungs (`both.py`: rung 23 has 733,670
  one-kill chains max 30, 11,746 two-kill max 32, 62 three-kill max 33; rung 29 has 15.4M
  one-kill max 38, 243,822 two-kill max 42, no three-kill chain).
- WHERE: anchor-235.md 9.
- LIMITS: about the REAL teeth; the delta-uniform form fails at small `delta`.

### record gaps are isolated
- STATEMENT: the neighbours of every record gap are small: `(1,2)` at `{5,7}`; `(1,3)` at
  `{5..11}`; `(2,2),(2,5)` at `{5..13}`; up to 7 at `{5..17}`; `<= 5` at `{5..19}`; `<= 7`
  next to any gap `>= 0.8F` at `{5..23}`; `<= 7` at `{5..29}`. `F_2 - F = 2,2,4,5,7,6,5`
  against `q' - s_min = 5,7,9,11,13,15,19`.
- CALCULATES: the depth-2 slack directly.
- STATUS: exact over the full period at seven machines.
- WHERE: anchor-235.md 9; also `F_2(47) = 134` maximiser `[54,80]` contains NEITHER a
  maximal gap, so both neighbours of every maximal gap of m47 are `<= 16` (mechanic C25).
- LIMITS: no explanation on record for why a record-size gap of a CRT word has only small
  neighbours — this is named as "the part with no teeth in it at all".

### the record stretch is ordinary at the bottom and made at the top
- STATEMENT: survivors of a record stretch under `{5..g}` track a random stretch of the
  same length at the bottom layers (`S_5/mean = 0.96-1.0`, `S_7/mean = 0.92` at `{5..23}`)
  and the record is MADE at the top three or four layers, where each gear removes 2-3
  survivors a random stretch would keep. The three longest gaps of every machine share the
  same profile to within one survivor.
- CALCULATES: the layer at which a record becomes a record. Example `{5..23}`, record 33:
  survivors 19, 13, 10, 7, 5, 3, 0 against random-stretch means 19.8, 14.2, 11.6, 9.8, 8.7,
  7.7, 7.0.
- STATUS: exact over the full period at `{5..11}` .. `{5..23}` (`layer_law.py`).
- WHERE: anchor-235.md 9d.
- LIMITS: "an ordinary lower stretch whose last few survivors sit exactly on the teeth of
  the last few gears" is a description of the alignment, not a criterion for it.

### residues at the top of a record
- STATEMENT: the top layer of each record stretch carries 1 to 3 survivors, ALL necessarily
  on the top gear's teeth, with differences in the chain classes: `{5..11}` 1 survivor;
  `{5..13}` 1; `{5..17}` 2 at 4, 15 (teeth `+,-`, difference `11 = -d`); `{5..19}` 1;
  `{5..23}` 3 at 3, 11, 26 (teeth `-,+,-`, differences `8 = +d`, `15 = -d`). One layer down
  the survivors are 3,3,3,3,5 and only some sit on that gear's teeth.
- CALCULATES: the exact top-layer alignment of a record.
- STATUS: exact (`record_residues.py`).
- WHERE: anchor-235.md 9e.
- LIMITS: the pattern does NOT repeat from rung to rung in any form found; what repeats is
  the shape (ordinary lower stretch, 1-3 survivors at the top on the teeth).

### where the record first shows
- STATEMENT: the records of `{5..13}` through `{5..19}` start at columns 123, 118, 111
  (numbers about 670-740) — i.e. right after `19^2` the periodic word already shows its
  record.
- CALCULATES: the earliest address at which a machine's record is realised.
- STATUS: exact.
- WHERE: anchor-235.md 9e.
- LIMITS: three machines.

### record genealogy — records recruit runner-ups
- STATEMENT: for a record window of `y = M + q'` at column `k`, the `M`-openings inside
  `(k, k+F)` are the deleted chain and the ancestor is the `(L+1)`-gap window of `M` they
  cut. THE ANCESTOR IS ALMOST NEVER THE `F_J(M)` MAXIMISER (1 of 8) — it is a RUNNER-UP,
  by 2 to 14 — but its LARGEST GAP is itself a merged window one level down (7 of 8), and
  that continues for 1-5 generations.
- CALCULATES: the whole tree of a record. `29->31`: record 58 <- m29 `[18,10,30]` phase 8,
  teeth `'+-'` <- 30 = m23 `[7,23]` <- 23 = m19 `[5,15,3]` <- 15 = m17 `[2,6,7]` <-
  6 = m13 `[5,1]`, 7 = m13 `[5,2]` — five generations, each a runner-up (deficits 7, 9, 12,
  13, 10, 9). Full table at 23->29 .. 53->59 with phases and tooth words.
- STATUS: exact, computed by residue arithmetic on the column (no scan); the four highest
  record columns lifted and verified at the target machine.
- WHERE: mechanic.md C54.
- LIMITS: the ancestor's RANK among `M`'s own `J`-windows by span is 8 to 219 — so
  `F(M+q')` cannot be computed from the top-`k` `J`-windows of `M`, nor from `M`'s spectrum
  records.

### what F(M+q') could be computed from
- STATEMENT: by the attainment theorem it is `max over the realised legal words w` (1-4 per
  machine) `of max over the OCCURRENCES of w of (gap before + span + gap after)`. For long
  words the occurrences are few (4 for `(10,21,10)` at m29, two mirror pairs) and each is a
  CRT solution enumerable scan-free; for short words (the `L=1` and `L=2` rows, 8e6 and
  1.3e4 occurrences at m29) the flank order statistic `Phi(w)` is exactly what a scan
  supplies and no enumeration reaches.
- CALCULATES: the record is scan-free precisely when it is carried at depth `L >= 3`.
- STATUS: consequence of R68 plus the counted census; named, not built.
- WHERE: mechanic.md C54.
- LIMITS: the ladder is entering that regime (`k_win = 3` at 31->37 and 37->41; 47->53 and
  53->59 carried at `J = 4` and `J = 3`).

---

## I. HOW MANY ALIGNMENTS: EXACT COUNTING IDENTITIES

### depth-sum identity
- STATEMENT: `sum_{j >= 1} W_j(g) = prod_q c_q(g) = N2(g)` — every ordered pair of openings
  at lag `g` is the endpoint pair of EXACTLY ONE window, and CRT counts the pairs.
  COROLLARY: `W_j(g) <= N2(g)` for EVERY depth `j`, a depth-uniform closed-form upper bound
  on every window-sum count, with no period scan, at arbitrarily large machines.
- CALCULATES: the number of stretches of any span, at any machine, in closed form.
- STATUS: proved, one line; integer-exact m11-29 for `g = 1..64`; kernel-checked in both
  halves (`DepthSum.window_depth_unique`, `depth_partition`, `local_factor_{5,7,11,13}`,
  `depth_sum_at_13`, `depth_sum_hl_form`).
- WHERE: lateral.md item 27; formalist.md 2.23.
- LIMITS: PRIOR ART — this is Holt arXiv:2502.20470 Corollary 1 specialised to the
  constellation `(2, 6g-2, 2)`; the identity is true and was derived independently, only
  the novelty label was wrong. The kernel GLUE (index range vs residue range) is not built.

### the local factor c_q(g)
- STATEMENT: `c_q(g) = q-2` if `q | g`; `q-3` if `g = +-2u_q mod q` (the literal-link lag);
  `q-4` otherwise. Equivalently `c_q(g) = q - nu_q({0, 2, 6g, 6g+2})` — the machine's
  transfer diagonal IS the Hardy-Littlewood prime-quadruplet local factor. The `n`-point
  form: `c_q(d_1..d_n) = q - 2n + O`, `O = #{pairs with d_i - d_j = 0 or +-2u mod q}`,
  exact whenever `q >= 2n`.
- CALCULATES: the CRT count of any prescribed alignment; explains the notorious absences
  (gap 24 at m19/m23 and gap 29 both carry the MINIMUM endpoint phase count 3 mod 35).
- STATUS: proved; verified gears 5..31 all lags, 0 mismatches; 16,500 brute-force checks
  for the `n`-point form (gears 5..43, `n = 1..5`, 0 mismatches); kernel-checked at four
  gears (`DepthSum.local_factor_*`).
- WHERE: lateral.md items 21, 23; harvester.md 5f; formalist.md 2.23.
- LIMITS: endpoint arithmetic only — dividing out `N2` removes 11-30% of the histogram's
  post-trend residual; the rest is INTERIOR arithmetic.

### slot-level sharing between two gears
- STATEMENT: gear `q` strikes `k = +-u_q mod q`, so two gears share a column in exactly 4
  residue classes mod `q q'` — two where both sit on the same member (multiples of `q q'`),
  two where each takes one member of the pair (double kill); `12/5` of these per `q q'` are
  anchor-open; three gears share a column only mod `q q' q''`. Hit points of two gears
  coincide only at multiples of `q q'` (first 1517 = 37 x 41).
- CALCULATES: the exact overlap of two gears' strikes; the waste. Tooth density
  `sum 2/q` over 7..47 is 1.257 against a blocked fraction 0.745: 0.512 of the teeth land
  on already-blocked columns, and that waste is ENTIRELY this rigid sharing.
- STATUS: exact (`slot_interact.py`); nine gears 37..71 below 300000 give untouched 50,141
  against `prod(1-1/q) x anchor-open = 50,165`.
- WHERE: anchor-235.md 4; constructor.md R6 (roots-of-unity law: a column is double iff
  `36k^2 = 1 mod qq'`, so the double set is one fixed pinned subset of the integers,
  computable by semiprime arithmetic with no primality tests).
- LIMITS: sharing moves WHERE, never HOW MANY (the sharing law: survivors per period are
  `prod(q-2)` regardless of phases).

### the twin pin
- STATEMENT: a twin gear pair `(p, p+2)` SHARES a tooth, so its four within-pair double-kill
  CRT classes mod `p(p+2)` are pinned at `{+u', -u', +u'(p+1), -u'(p+1)}`, and the mixed
  class is the twin-product column: `6u'(p+1) - 1 = (p+1)^2 - 1 = p(p+2)` exactly. Every
  twin gear pair donates `>= 2` deterministic wasted strikes per window.
- CALCULATES: an exact address per twin gear pair; `slotOf(p(p+2)) = u(p+1) = 6u^2`.
- STATUS: exact (60/60 pairs to 2000); kernel-checked (`Polignac.twin_product_slot`,
  `Corridor.product_slotOf`, `twin_product_pin`, `twin_pin_self_block`).
- WHERE: lateral.md item 1; formalist.md 2.8, 5a.
- LIMITS: the net gain from tooth-sharing counts is `O(T(y))` per window against the needed
  `~K/log^2` (refuted angle 2).

### gap-graded split law
- STATEMENT: for gears `q < q' = q+g`, the nontrivial square root of 1 mod `qq'` has the
  closed form `m0 = (-2 q^{-1}) mod g`, `b0 = (2 + m0 q)/g`, `i = (q'-b0) q^{-1} mod 6`,
  `x = (q'(b0+iq)-1)/6`, with the mirror at `P - x`. `g = 2` is the UNIQUE gap with
  `b0 = 1`: its split representative `x = u' <= K` is IN-WINDOW at every scale
  unconditionally.
- CALCULATES: the exact column at which any two gears cross-kill, in closed form.
- STATUS: exact for ALL 2850 pairs `5 <= q < q' <= 400` and 753,378 pairs at `y = 10007`,
  zero failures.
- WHERE: lateral.md item 4.
- LIMITS: twin gear pairs are the only gap class whose split-double contribution is
  guaranteed in-window; all others are alignment-rated (twin hit rate 100%, non-twin
  decaying to ~50.8%).

### end-zone alignment of all the gears (a negative)
- STATEMENT: gear `q`'s clean end zone is the window `+-h_q` around every multiple of
  `30q`, `h_q = q` for the classes `+-1, +-11, +-13` and `7q` for the class `+-7`. Two zones
  drift by `30(q'-q)` per period and realign exactly at `30 q q'`; a set of zones stacks
  exactly at `30 x prod q`; the joint zone density is `prod (2h_q/30q)`.
- CALCULATES: where all gears' end zones would coincide. EXACT SEARCH over
  `n in [1369, 10^7]` for all gears `37 <= q <= sqrt(n)`: ZERO SOLUTIONS. Closest calls:
  missing one gear's zone never past `n = 1680`, two never past 2478, three never past 3550.
- STATUS: exact search; expected fraction 3.1e-2 with 2 gears, 3.8e-14 with 15.
- WHERE: anchor-235.md 3.
- LIMITS: the end zones alone are too narrow ever to line up — this is a NEGATIVE about
  one particular alignment, not about openings generally.

### the counted word census
- STATEMENT: the exact number of occurrences `occ(w)` of every run of legal letters of
  length `<= 4`, with its flank envelope `Phi(w)` and (length `<= 3`) its whole flank-sum
  distribution, over the full cyclic period at m11..m37.
- CALCULATES: the padded supply — `occ(23;m19) = 86`, `occ(29;m23) = 6`,
  `occ(31;m29) = 2090`, `occ(37;m31) = 26,366`, `occ(41;m37) = 61,460`; the two-letter
  words at m31 (`(12,25)` 35,314 each, `(12,37)` 150, `(25,37)` 18) and three-letter
  (`(12,25,12)` 188, `(25,12,25)` 28, reproducing the chain inventory by a different
  vehicle).
- STATUS: exact full period, gated five ways (count `= prod(q-2)`, weighted sum `= P`,
  max `= F`, max pair `= F_2`, every table mirror-symmetric).
- WHERE: constructor.md R102; mechanic.md C12.
- LIMITS: m41 is ~40x m37 and is not a one-round job.

### how the flank envelope depends on occurrences, not span
- STATEMENT: the flank envelope follows OCCURRENCE COUNT, not span:
  `maxflank(w) ~ 2.05 ln(occ(w))` (sd 0.27), `FS_max(w) ~ 2.77 ln(occ(w))` (sd 0.24).
  Counts fall 2-5 orders across a step's compatible spans (29->31: 7,815,766 / 205,068 /
  6,500 / 4 at spans 10/21/31/41).
- CALCULATES: the expected flank sum of a word from its count.
- STATUS: measured, exact counts; the monotone-envelope reading is REFUTED as a law.
- WHERE: constructor.md R33; mechanic.md C13, R11.
- LIMITS: it is a LITERAL-letter law — the two padded letters at m19/m23 are the two
  extremes of the band (1.80 and 6.14 against 2.39-2.96), so inverting it at m31 gives
  `occ(37;m31)` in `[2.5e3, 4.0e11]`, eight orders wide.

---

## J. THE ALIGNMENT AS A CERTIFICATE OBJECT

### word-free (qualifying) criterion
- STATEMENT: `F(M+q') <= max(F_2(M), max_{j>=3} qualmax_j(M; q'))`, where `qualmax_j` is
  the maximum span of a `j`-gap window of `M` whose `j-2` middles are all `>= 2u'` and
  T3-legal. `(D)` at `alpha = 3` follows whenever that is `<= F(M) + q'`.
- CALCULATES: the rung from three exact census quantities of the OLD machine.
- STATUS: exact; verified at all eight steps 11->13 .. 37->41 with equality at 7 of 8;
  kernel-checked as `MergeLaw.newgap_le_max`, `D_of_qualmax`,
  `Spectrum.merged_le_of_qual_flat_all`, and instantiated at five hypothesis-free rungs
  (`Ladder.D_at_11_13 .. Machine23.D_23_29`) plus two dictionary rungs
  (`Machine31.D_29_31`, `Machine37.D_31_37` under `Census29/31`).
- WHERE: constructor.md R39; formalist.md 2.16, 2.17, 2.22, R25.1, R25.6; mechanic.md C20.
- LIMITS: the ALL-DEPTHS form FAILS from 43->47 on (152 > 150; 177 > 171), with exhibited
  witnesses — because the floor grows with the gear while the mean gap does not, so
  depth-7 runs of qualifying gaps first occur at m43. It kills the hypothesis-free
  all-depths form, not (D).

### word-legal criterion Q*_J
- STATEMENT: the same object with the SHARP predicate — each middle gap in
  `V = {0, +s, -s} mod q'` AND the letter word's prefix sums of range `<= 1` — instead of
  the size shadow `>= 2u'`. `Q*_J <= Q_J` pointwise and `max_J Q*_J = F(M+q')` exactly.
- CALCULATES: the rung at every step the plain criterion loses: 43->47 certifies at
  `<= 149 <= 150`, 47->53 at `<= 170 <= 171`, at EVERY depth `J = 2..7`, consuming NO
  arity hypothesis. The failing 47->53 window's middles `[22,28,30,67]` contain not one
  legal letter — the plain criterion was failing on a relaxation the merge law never needed.
- STATUS: exact; two-sided anchors `88 = F(37)` at 31->37 and `58 = F(31)` at 29->31; the
  vehicle then delivered `F(59) = 161`.
- WHERE: mechanic.md C24, K7, C27, C35, C43, C51.
- LIMITS: a CERTIFICATION (never a failure) is conditional on the span cap; the caps used
  sit 30-90 columns above the observed maxima, and every step with an independent value has
  agreed exactly.

### spectrum-plus-depth certificate
- STATEMENT: `Q*_J <= F_J(M)` by definition, and EMPTINESS IS UPWARD CLOSED (deleting a
  flank of a legal `J`-window leaves a legal `(J-1)`-window). Hence with
  `J_max = A_kill + 1`, `F(M+q') <= max_{2 <= J <= J_max} F_J(M)` — the OLD machine's
  spectrum plus ONE emptiness certificate, no word list, no flank envelope, no oracle.
- CALCULATES: rung nine `F(43) <= max(103,117,118) = 118 < 134` and rung ten
  `F(47) <= max(116,125,132) = 132 < 150`, margin 18.
- STATUS: proved; certifies 8 of 9 steps whose spectrum is complete.
- WHERE: constructor.md R84, R93; mechanic.md C44, C49.
- LIMITS: it is a FINITE-DEPTH criterion in the literal sense — the margin is
  `F(M) + q' - F_{A_kill+1}(M)`, so every step with `A_kill <= 3` certifies and it FAILS at
  29->31 (`F_5(29) = 85` vs budget 74) and at 47->53 (`F_6(47) = 177` vs 171). Each extra
  unit of `A_kill` admits one more level of the `F` ladder, costing 7-16 units, while the
  budget gains only `q' - q`.

### the potential (depth-quantifier-free (D))
- STATEMENT: (D) holds IFF a POTENTIAL `h` exists with three ONE-STEP, ONE-OPENING
  inequalities: `(C1) h(i,s) >= d_i`; `(C2) h(i,s) >= d_i + h(i+1,s')` for every legal
  qualifying transition; `(C3) d_{i-1} + h(i,s) <= F(M) + q'`. Necessity `h = K* (x) R`;
  sufficiency because any super-solution dominates the star.
- CALCULATES: the whole ladder from one function per machine; exhibited `h11, h13, h17,
  h19`, whose tail depths do NOT grow with the machine (4, 3, 5, 4).
- STATUS: the certificate direction is kernel-checked (`Potential.IsPotential`,
  `chain_le_potential`, `D_of_potential`, `merged_le_of_potential`; `Potential19`,
  `PotentialLadder`); `(C2)` holds with EQUALITY in every branch at every machine and its
  deepest branch is always that machine's own `no_big_run`.
- WHERE: constructor.md R46; formalist.md 2.24.
- LIMITS: the CONVERSE (a potential always exists) is not formalised; a potential valid at
  every machine at once is not known — the generator is arity-free but NOT machine-free.

### the survivor generator (F_2 of the new machine, one gear down)
- STATEMENT: a window of two consecutive NEW gaps is a window of old openings ALL struck at
  one phase EXCEPT ONE SURVIVOR, and the spacing straddling the survivor is `d_i + d_{i+1}`;
  the survivor lives iff `cls(d_i)` is ILLEGAL out of the current tooth. Adding that one
  SKIP transition SIGMA: `F_2(M+q') = L (x) K* (x) SIGMA (x) K* (x) R`, and generally
  `F_j(M+q') = L (x) K* (x) (SIGMA (x) K*)^(j-1) (x) R`.
- CALCULATES: `F_2(M+q')` from the OLD machine — the two-gap statement at `M+q'` is LAYER 0
  of the same algebra one gear down, so it DESCENDS rather than being an extra hypothesis.
- STATUS: proved both directions; verified exact, full period, seam-stitched at every
  scannable step (`F_2(M+q') = 16, 25, 31, 39, 55, 68, 90`), against the independent pair
  census and against CRT+SAT at 31->37; kernel first statement at 11->13
  (`Gen11.gen_zero = 11`, `gen_one = 16`, `Gen11Sound.generator_sound` giving
  `F_1..F_4(13) <= 11,16,23,26` from machine 11's 135-letter word with machine 13's period
  nowhere in the derivation).
- WHERE: constructor.md R56, R57, R59; formalist.md R25.3, R26.2.
- LIMITS: the survivor system needs ONE MORE order of history than the plain system
  (`A_4` exact 7/7 plain, `A_5` exact on the survivor side).

### realisability CSP (the two halves of an alignment question)
- STATEMENT: column `k` is struck by gear `q` iff `k = +-u_q mod q`, so by CRT a column IS
  a phase vector `(a_q)`. A gap tuple with prefix-sum points `X` and interiors
  `Y = (0, span) \ X` occurs as `m` consecutive gaps of `M` iff the system
  `(open) a_q not in {+-u_q - x : x in X}` for every gear, and
  `(cover) for every t in Y some gear q has a_q = +-u_q - t`
  is feasible. The period never appears — `pi(y)-2` variables, domains `<= q`.
- CALCULATES: any realisability question by exact CSP with no scan: the corpus ladder
  `F = 7,11,18,25,34,43,58,88` and `F_2 = 11,16,25,31,39,55,68` recovered with no period.
- STATUS: exact decider (`research/crt_dict.py`), gated on 2,013 tuples of arity 1,2,3 at
  m11/13/17 against the pruned-IE counts, plus nine published anchors.
- WHERE: constructor.md R60, R61, R64.
- LIMITS: the two halves behave oppositely — SHALLOW queries are dear (few open points,
  large gear domains: an arity-1 refutation costs 13.2 s at m31, 10-20 s at m37, > 250 s
  undecided at m41) and DEEP queries are cheap (3 ms at m29 for arity 4).

### the killer profile of a word extension
- STATEMENT: for every realised legal word and every T3-legal one-letter extension, the
  kill is attributed to one of two halves: SAT (some gear has no admissible phase — the
  screen) or `y*` = the smallest gear prefix whose OPEN constraints make the CSP infeasible.
  `y* = 0` means NO COLUMN OF `M` BLOCKS THE PUNCTURED INTERIOR — the word dies of the
  COVER half alone, with no tooth position of any open point needed.
- CALCULATES: which half kills each extension. Measured: at 19->23 all 4 extension classes
  die at `y* = 0`; at 23->29 3 at 0 and 1 at 7; at 29->31 both at 0; at 31->37 3 at 0 and 1
  at 7; at 37->41 8 at 0, 5 at gear 5 alone; at 41->43 9 at gear 5. NO extension anywhere
  was attributed to the open constraint of a gear above 7.
- STATUS: exact, 0 realised and 0 undecided at the full machine at every machine m19..m41.
- WHERE: mechanic.md C53.
- LIMITS: 10 of 19 classes at m41 are refuted but unattributed (the relaxed instance did
  not decide at 10M nodes); m43/m47 not attempted.

### the case-split covering certificate (alignment as an LP dual)
- STATEMENT: "machine `y` has a fully blocked window of width `W`" is a covering polytope
  with one 0/1 variable per phase tuple of every gear and pair; an infeasibility certificate
  is an exact rational DUAL proving `F(y) <= W` from the primes alone — no census, no
  period, no word list. Fixing the phases of the `k` smallest gears (the CASE SPLIT) shrinks
  the position set; a certificate in EVERY case is a certificate of the rung.
- CALCULATES: ten (D) rungs 7->11 .. 41->43, the eleventh rung 47->53 at `W = 171` by 8,077
  certificates, tight `F` at five machines (`F(19) <= 25`, `F(23) <= 34`, `F(29) <= 43`,
  `F(31) <= 58`, `F(37) <= 88`), and the increment-law upper half at seven steps.
- STATUS: exact rational duals, re-checked from their own integers; kernel-transcribed at
  19->23 (`CaseCert23`), 29->31 (`CaseCert31`), 31->37 (`CaseCert37`, 385 cases in 35
  tiers) and the three increment rungs (`IncCert23/29/31`).
- WHERE: lp-duality.md rounds 29-30; formalist.md R27.1, R28.1, R29.0, R30.0.
- LIMITS: cost is a PRIMORIAL in `k` (1, 5, 35, 385, 5005, 85085 cases); `W_inc - F(q')` is
  negative at exactly one corpus step (31->37, by 8), where no sound method can certify at
  any `k`.

### the lowest-blocker identity (what the recursion row counts)
- STATEMENT: if some gear strikes column `x`, then
  `1 + #{(a,b) : a < b, both strike x, no gear below a strikes x} = #{a : a strikes x}` —
  only the LOWEST blocker can be the `a` of such a pair, and it pairs with each other
  blocker exactly once. Summed over the position set: `sum_a |A_a| >= |pos| + sum_{a<b} n_ab`.
- CALCULATES: the certificate's recursion row without evaluating an 8.2-million-term
  max-cover; and 96.4% of the gear-index-1 coefficients are ZERO, sound with no evaluation.
- STATUS: kernel-checked, NO AXIOMS (`CaseSplit.lowest6`, `lowest7`, `CaseSplit5.lowest5`,
  `degpos5/6/7`).
- WHERE: formalist.md R27.1, R28.2; proofs/CaseSplit.lean.
- LIMITS: it is a pointwise counting identity — it says nothing about which phases occur.

### the mirror as a symmetry of the certificate
- STATEMENT: `reflect(hits(q, r, W)) = hits(q, (1 - W - r) mod q, W)`, `reflect(i) = W-1-i`.
  So the case at held phases `ws` and the case at `(1 - W - ws) mod q` have position sets
  that are REFLECTIONS of each other, isomorphic relaxations, and equal `V*`, `|pos|` and
  certificate cost — decide one case per mirror orbit and TRANSCRIBE the other. Every level
  has exactly ONE self-mirror case (each `q` is odd), so the mirror halves every level to
  `(cases+1)/2` orbits.
- CALCULATES: a free factor 2 on every sweep of this species; the transcription
  `rows' = [(rho(i), lam)], y' = y, yff' = yff, nu'[pi(t)] = nu[t]` gives the same
  `lhs, rhs, margin` and op count.
- STATUS: theorem with proof; gated at every gear of m11..m47 at `W = 74, 95, 104, 132` and
  on three non-vacuous cell families; 385/385 transcribed certificates re-verified from
  JSON alone at 31->37.
- WHERE: lp-duality.md round 29 section 5, round 30 section 3; lateral.md item 83.
- LIMITS: the transcription is a genuine SECOND certificate, not the same one — the float
  solver found a different dual 124 times of 385.

### the translation transcription
- STATEMENT: if `pos(ws + t) = pos(ws) - t` as subsets of `[0, W)` — i.e. the held gears
  block `[0,t)` at `ws` and `[W-t, W)` at `ws + t` — then with `rho(i) = i - t` and
  `m_q(r) = (r+t) mod q` the five claims of the mirror theorem hold verbatim, and
  `(rows - t, y, nu o pi_t^{-1}, yff)` is an exact dual certificate of `ws + t` with the same
  margin and op count.
- CALCULATES: the VALUE CLASSES of a case split are exactly the orbits of
  `{mirror, boundary-blocked translation}` — matching the measured class counts at both
  sweeps where they were measured (11 of 35 at m37 `W=95 k=2`; 14 of 35 at m41 `W=104 k=2`),
  a further 1.8x saving at m53 `W=171 k=4` (1,391 classes against 2,503 orbits).
- STATUS: theorem; gated by 484 translation transcriptions from 330 of 385 certificates,
  every one re-verified from JSON.
- WHERE: lp-duality.md round 30 section 3b.
- LIMITS: the boundary condition (held gears blocking a whole end segment) is invisible to a
  test of "every case", which is why round 29 wrongly recorded "it is not a translation".

---

## K. WHAT THE COUNTERFACTUAL FAMILY SAYS ABOUT ALIGNMENT

### the tooth-counterfactual family
- STATEMENT: the machine has two inputs — WHICH gears, and WHERE the teeth are. Move the
  teeth: keep the mirror symmetry (teeth `+-v_q`) and let `v_q` range over `{1..(q-1)/2}`.
  Every member has the SAME period, the SAME opening count `prod(q-2)` and the same per-gear
  density; only the positions move. `|V(y)| = 30 / 180 / 1440 / 12960` at m11/13/17/19.
- CALCULATES: a clean null model for every alignment statistic. `F(twin)` sits at the
  20.0 / 18.1 / 26.4 / 17.1 percentile (and 11.9% at m23) — the twin machine's record is in
  the bottom fifth to quarter of its own family, never the minimum, in a family whose
  maximum is 1.6-1.9x the twin value.
- STATUS: exhaustive and exact at m11..m19; m23 exhaustive in the pinned family (12,960).
- WHERE: lateral.md items 61, 62, 63; docs/novel/tooth-counterfactual-percentile.md.
- LIMITS: the rows are NESTED, so they are not independent draws and no p-value is claimed.

### the record law is family-wide (the identity is structural)
- STATEMENT: `max(F_2(M), max_{J>=3} Q*_J(M; q')) = F(M+q')` EXACTLY at every member of the
  tooth-counterfactual family, at every step: 30 + 180 + 1440 + 12960 + 12960 = 27,570
  counterfactual machines, zero exceptions.
- CALCULATES: the identity that computes `F(M+q')` from the old machine is STRUCTURAL; only
  the SIZE of `Q*_J` is arithmetic. So the counterfactual obstruction is an obstruction to
  BOUNDING `Q*_J`, not to the record law — a strictly smaller target.
- STATUS: asserted at all 27,570 members.
- WHERE: lateral.md item 69.
- LIMITS: the family is mirror-SYMMETRIC (teeth at `+-v_q`); whether the record law needs
  the mirror at all is untested (U18).

### the increment law and (D) are not generic
- STATEMENT: over the full counterfactual family the increment law is VIOLATED by
  13.3 / 13.9 / 14.5 / 21.7 / 22.3 percent of members at 7->11 .. 19->23 — and the rate
  GROWS with the machine. Pinning the incoming gear's tooth to `round(q'/6)` drops it to
  0 / 0 / 1.1 / 6.5 / 5.7 percent. (D) itself fails at only 0.00-0.56% of the family.
- CALCULATES: the decomposition of where the law's difficulty lives — THE NEW GEAR'S TOOTH
  POSITION carries most of it and the old machine's arithmetic the rest.
- STATUS: exhaustive at m11..m19 (and the full 142,560-member 19->23 family).
- WHERE: lateral.md items 62, 82; constructor.md R76 note.
- LIMITS: no argument using only "same gears, same density, symmetric teeth" can prove the
  increment law.

### the residual violators are not a congruence on F(M)
- STATEMENT: the predicate "F(M) mod q' in {0, a, b}" (the record being congruent to a tooth
  difference) has sensitivity 34.0% at 17->19 and 5.6% at 19->23 over the residual violators;
  the best predictor of the form "F(M) mod q' in S" reaches 57.9% balanced accuracy, barely
  above chance. What DOES describe the residual set is a DEPTH-4 word-legal window (70% of
  19->23 violators are invisible at depth 3) plus the flank condition min flank `> s_min`.
- CALCULATES: what a teeth-sensitive separator would have to be.
- STATUS: exact over the pinned family at three steps.
- WHERE: lateral.md items 70, 71; constructor.md R95, R96.
- LIMITS: `H1` HOLDS at m31 while all three of the corpus's failing rows fail there — so it
  is not the separator; killed as an explanation, kept as a per-step condition.

### L is not capped on the family by the real machine's constant
- STATEMENT: max `L` over the FULL family is 1, 3, 3, 3, 5 at 7->11 .. 19->23 against the
  real machine's 0, 1, 1, 1, 2; the `L = 5` member (`J_max = 7`, `A_kill = 6`, beyond
  anything the corpus shows below m47) is `V(19)`'s `(1,2,5,2,1,5)` with `v_23 = 9`, word
  `[5,18,5,18,5]`, residues mod 23 alternating 16, 21. EVERY deepest word at every step is
  LITERAL. So "(B) L bounded" does NOT follow from the structural theorems alone — CRT, the
  mirror, T2/T3, R89/R90 and the record law hold at every member.
- CALCULATES: any proof of (B) must use the teeth.
- STATUS: exhaustive at five steps, 165,584 rows.
- WHERE: lateral.md items 78, 86.
- LIMITS: the real machine is at or below the family median at every step.

### where the teeth enter L
- STATEMENT: on the family, `P(L >= 3 | bare alternation {5,7}-admissible)` =
  0.006 / 0.101 / 0.272 / 0.320 at 13->17 / 17->19 / 19->23 / 23->29, and
  `P(L >= 3 | not admissible)` = 0.0000 / 0.0000 / 0.0001 / 0.0000; every `L >= 3` word
  built from the bare letters is admissible (0 exceptions in 21,357 rows). The real
  machine's alternation is NOT admissible at 13->17 `(6,11,6)`, 17->19 `(6,13,6)` and
  23->29 `(10,19,10)` — its `L <= 2` there is decided by gears 5 and 7 ALONE — and IS
  admissible at 19->23 `(8,15,8)`.
- CALCULATES: which corpus steps the corridor decides and which it does not.
- STATUS: exact on the family; the necessity direction kernel-checked (`BareAlt.no_gapWord`).
- WHERE: lateral.md item 79; constructor.md R103.
- LIMITS: size and corridor admissibility are ORTHOGONAL channels explaining together only
  36-42% of `L`'s variance; `L` is NOT monotone in the letter size, and the SMALLEST letter
  gives the SHORTEST words.

### the depth-2 slack and its one family failure
- STATEMENT: `F_2 >= 2 d_0` at every symmetric two-tooth sieve, because the two gaps around
  column 0 are `(d_0, d_0)` by the mirror. The depth-2 half `F_2 <= F + q'` can therefore
  fail by THAT WINDOW ALONE whenever `2 d_0 > F + q'` — and over 14,616 exhaustively
  enumerated old machines it fails at exactly ONE, `V(19)`'s `(1,1,4,3,5,2)` with `F = 26`,
  `F_2 = 50`, `d_0 = 25`. Excluding wrap-pair members the minimum slack is 8/6/6/5/4/9 —
  POSITIVE at every step.
- CALCULATES: the only depth-2 failure mode found is the self-mirror 2-window, which is
  exactly the one depth at which the mirror lever needs a hypothesis (`d_0 != F`), and
  `d_0` is a closed form on the real machine.
- STATUS: exact, gated at 15,217 old machines.
- WHERE: lateral.md items 80, 81.
- LIMITS: the real machine's depth-2 slack is ORDINARY (23.7-86.5 percentile), not favourable.

---

## REFUTED ALIGNMENT CLAIMS

- **Spectrum flatness `F_{k_max+1} - F <= q'`** — FALSE at 29->31 (`F_5 - F = 42` against
  `q' = 31`); raw flatness fails 5 of 15 machine-depth pairs. constructor.md X17; mechanic R13.
- **The all-depths word-free (hypothesis-free) criterion** — FAILS from 43->47 on, with
  machine-verified witnesses (`Q_7(43;16) >= 152` at `k = 110,350,776,715,218`,
  `Q_7(47;18) >= 177` at `k = 41,120,916,229,562,503`). mechanic.md C20.
- **"k_max <= 4 restores 47->53"** — DECIDED NEGATIVE: `A_kill(47->53) = 5` EXACT.
  mechanic.md R24, C23.
- **"Nothing seen contradicts k_max = 3" (at 47->53)** — FALSE; the first 5-chain
  `(18,35,18,35)` is realised. mechanic.md R23.
- **The alternation-pair predictor `A_kill >= 5 iff the pair (s, q'-s) is realised`** —
  REFUTED by its own pre-registered test at 53->59: the pair `(20,39)` IS realised and every
  longer alternation is ZERO by theorem. mechanic.md R25.
- **M1, "every realised legal spacing value is exactly `a`, `b` or `q'`"** — FAILS at m31
  (49 = a+q'), m37 (55, 68) and m41 (57, 72, 86 = 2q'). constructor.md R86; mechanic.md C43b.
- **Litcap as a predictor of realised arity** — a proved cap on the LITERAL part only:
  litcap 4 at m41 where the literal 2-word count is 0; litcap 2 at m37 where `A_kill = 3`.
  constructor.md X29.
- **`A_relax` as the order the chain needs** — `N(M) = max(2, A_relax)` is REFUTED at m37
  (`A_relax = 2`, `N = 3`) and at m41; the extra order is bought by a PADDED cycle.
  constructor.md R75, R85.
- **"The truncation arity grows"** — self-corrected: the growing sequence was the RESIDUE
  arity, and a residue-qualifying run is not a kill chain (T3 forbids two same-class letters
  in a row). constructor.md X28.
- **Bounded-modulus residue laws capping SIZES** — every `(G1,G2)` pair is within L1
  distance 1 of a corridor-allowed pair at ANY bounded modulus: corridors constrain WHERE,
  never HOW BIG. constructor.md X11; formalist.md verdict 4.
- **Tier B (moduli 385 .. 1616615)** — adds EXACTLY ZERO exclusions anywhere tier A did not.
  constructor.md X13; lateral X11.
- **Congruence-class potentials at any modulus** — a potential that is a function of
  `k mod m` for a PROPER divisor `m` of the period certifies nothing: every class mod `m`
  contains a blocked column, so `h(k) >= h(k-1)+1` forces `0 >= m`. constructor.md X32;
  lateral.md item 40 (T1).
- **The machine-free corridor certificate** — `MF_3 mod 35`, `MF_3 mod 385` and `MF_4 mod 35`
  are IDENTICAL at every step; neither a finer modulus nor more history buys one unit.
  constructor.md X34, R52.
- **The histogram / any unitary invariant as a supplier of the two-gap fact** — the tight
  rearrangement bound is `F + G_2 = 2F` (maximal gaps mirror-paired), over budget from
  19->23 on; since every unitary invariant of `N = BS` is a function of the gap histogram,
  this is a THEOREM that the invariant route dies. constructor.md X35; lateral.md item 39.
- **"(D) might be corridor-forced at n = 4"** — 0 of the 1225 `(span, flank-sum)` classes
  mod 35 are blocked; every flank-sum value above the (D) requirement is corridor-feasible
  for every span. lateral.md item 31.
- **Both-flanks-maximal exclusion as a route to (D)** — correct and kernel-worthy, but the
  binding flank pairs are MID-SIZE, never maximal, so it excludes a configuration that never
  binds. constructor.md X16; formalist.md verdict 2.
- **Monotone flank envelope as a machine law** — FALSE, six violations with addresses
  (m29 span 25 -> flank 30 beats span 21 -> 27); the envelope follows OCCURRENCE COUNT.
  mechanic.md R11; constructor.md X19.
- **`FS <= F`** — FALSE: `FS/F = 1.09` at 13->17 and 1.12 at 29->31. constructor.md X18.
- **Drift recursion "new max address = f(old top-stratum address)"** — 0/4 and 1/2
  reachability at 19->23 and 23->29; the honest law is LOCAL, `address = pin(word)`.
  lateral.md refuted angle 5.
- **A-priori stabilisation of the near-top word-SHAPE family** — cross-machine full-shape
  recurrence is ZERO; observed halves are 3.2% of admissible and disjoint per machine.
  lateral.md refuted angle 6.
- **"1 of 4 fuel sites fire" / a fuel x alignment "double rarity" multiplier** — a
  one-window artefact; every site fires exactly once per new-machine period, and alignment
  is a density factor, never a count factor. lateral.md refuted angles 7, 8.
- **Smooth `supply^2/gaps` prediction of padding events** — padding switches on/off with
  `q' mod 35`; it predicted ~5 double-padded runs at 37->41 where the corridor forbids the
  adjacent shape outright. lateral.md refuted angle 12.
- **Covering/capacity explanation of absent gaps** — residual interior demand has positive
  slack (8-16 spare strikes) at every `g`; gap 24's absence is arithmetic selection plus
  rarity, NOT impossibility. lateral.md refuted angle 11.
- **`#distinct eigenvalues = |Farey(F+1)| - 2`** — true only at the holeless machine 11;
  the true level count is a divisor-closure statistic and the HOLE LIST is the defect.
  lateral.md refuted angle 35.
- **"Gear 5 is the only parity-obstructed gear"** — every gear `p >= 5` is parity-obstructed
  (`alpha_1(p)` is odd at every machine and modulus), so the pole phase is never attained
  anywhere. lateral.md item 56, refuted angle 34.
- **"alpha_1/alpha_2 -> -1/phi"** — a CROSSING, not a limit: the ratio crosses `-1/phi`
  between m29 and m31 and is +0.0403 past it at m37, still rising. lateral.md refuted angle 37.
- **The 2n-gap reordering as a route** — the distinct-difference count is `2n` under EVERY
  admissible re-choice of teeth while `F` moves by 1.8x, so the coordinates discard exactly
  what `F` depends on; and `F` is not a function of the order permutation at all.
  lateral.md item 58, refuted angle 41.
- **The L1 character bound** — provably BLIND to the teeth: `sum_m |Shat(m)|/P = prod_q S_q/q`
  does not depend on `v_q`, identical at all 30/180/1440 counterfactual tooth vectors while
  `F` spreads 1.8-2.5x. lateral.md item 76.
- **Moment / PSD / Chebyshev routes** — margins 67.6 .. 4.3e10 and GROWING; the L2
  instrument's spread is real but essentially uncorrelated with `F` (spearman -0.038, +0.023,
  -0.186), and NEGATIVELY correlated at the largest machine. lateral.md items 34, 77, refuted 19.
- **`A_4` (4-tuple abstraction) as the carrier of the qualifying-tail potential** — the
  qualifying sub-digraph has a CYCLE at m23, m29, m31, so the longest qualifying path is
  INFINITE; `A_5` would not fix m29 either. formalist.md verdict 19.
- **The deletion ladder as the induction's supplier of `F_2`** — numerically sufficient but
  LOGICALLY CIRCULAR, and its slack thins to 0 at 41->43. constructor.md X36.
- **"`F_2` needs slack below the budget"** — REFUTED by the slack sweep: every `U <= 74`
  certifies, `U in [75,85]` stalls at `U`, `U >= 86` stalls at 86 — the obligation is EXACTLY
  the two-gap statement itself, zero further slack. constructor.md X37.
- **The marked spectrum `Q^[J]` and the "J=5 object"** — an implementation artefact (the DP
  returned success as soon as the mark quota was filled, never checking interiors after the
  last mark); the corrected marked spectrum is EXACT in all 30 entries and the 29->31 verdict
  REVERSES. mechanic.md R15; constructor.md X33; formalist.md verdict 12c.
- **The C13 qualifying-spectrum table (4 of 7 rows)** — wrong, built before the vacuity fix;
  the criterion column was always right but any earlier use of an individual `Q_j` from that
  table must be re-checked. mechanic.md R14.
- **The self-mirror window's span `<= 0.8 F_j`** — REFUTED: `span_self(j) = F_j` exactly at
  m7 (`j = 3,7,9,11,14`) and m11 (`j = 11`); the correction is an explicit exception list,
  empty from m13 up. lateral.md refuted angle 42.
- **"Every hole lies in the top half of the gap range" as `> 0.71 F`** — the tightest is
  0.7059 at m23, so `> 0.70 F` holds and `> 0.71 F` would FAIL. lateral.md item 67.
- **Angular coherence / "m is small" / the `(v_5,v_7)` class as the mechanism of the twin
  machine's low record** — all three REFUTED, all three in the same shape: the twin is a
  low-`F` outlier INSIDE the high-`F` class of the proposed variable.
  lateral.md items 61, 64, refuted angles 39, 40, 45.
- **"The real machine grows, untouched happens exactly once"** — WITHDRAWN as overstepping
  the line of enquiry. anchor-235.md 10.
- **"Every section holds an aligned column IS the twin prime conjecture"** — overstated: it
  is STRONGER. anchor-235.md 10.
- **`G_2 <= F + q'` (the 3-sparse sufficient criterion)** — fails from rung 47 (152 > 149,
  174 > 170); the stretches that beat it are not two-progression patterns, so (D) is
  untouched and the relaxation is simply too loose past 43. anchor-235.md 9a.
- **The onset formulas `onset = F_2(M_prev)`, `2F` two machines back, `onset/F(M)` constant**
  — all three fail at every out-of-sample step; only the recursion form (intersected with the
  transfer's own emissions) survives. mechanic.md R27 (D1-D3).
- **`peak depth of Q_J` non-decreasing in the machine** — REFUTED: 5,4,6,5,6 at m11..m23,
  then 5 at m31 and 7 at m37; the peak is terminal below m31 and INTERIOR from m31 on.
  mechanic.md R28 E1, C42.
- **The linear-close defect** (a full-period census over a CIRCLE taken linearly) — every
  full-period table was short by its seam structures; found by the mirror parity law on
  first use. mechanic.md C26; lateral.md item 46(d).

---

## GAPS THE RECORD ITSELF NAMES

1. **`L_pad(M) <= c_pad` uniformly** — the bare half is closed (`L_bare <= PSORD <= 5`,
   kernel); gears 5 and 7 cannot supply the padded half past 53->59 (CORRCAP infinite); no
   fixed-depth counter can supply it; the exposure half over-caps by 16-18. What is left is
   the COVER half at full depth on the NON-BARE words only. constructor.md R105.
2. **`|eps| = O(1)` per literal letter (A-lit) and the padded residual (A-pad)** — measured
   `|eps| <= 4` along every literal maximising chain, but the padded letter at m31 has
   `eps = -17`, and that failure is the F_3-wall event "the old machine's `F_3` maximiser has
   a padded middle", base rate `3/q'`. constructor.md R91, R101/C6.
3. **The depth-2 half `S_2 = F + q' - F_2 >= c_A c_B`** — measured 9 to 49, growing roughly
   with `q'`, but UNPROVED, and it is R55's 2F wall: no rearrangement invariant and no
   congruence potential supplies it. constructor.md R99.
4. **The chain depth `D_g` is not an algebraic consequence** — a run inside a two-class set
   alternates freely, so nothing in the residue arithmetic bounds its length; `D_g` is a fact
   about the lower gap SIZES and stays a per-machine measurement. Written into
   `proofs/AnchorChain.lean`'s own header.
5. **No formula collapsing the layered recursion** — the walk closes layer by layer at depth
   `pi(q)`; the nested form is exponential (`prod (1+D_g)`), the flat form is
   `prod(g-2)`, the scan form is quadratic but needs `F+1` as its term bound, which is the
   unknown. Below the scan, a form would have to compute the first integer outside a union of
   `2 pi(q)` arithmetic progressions from the `pi(q)` residues of `s` alone; NONE IS KNOWN
   HERE AND NONE WAS FOUND. anchor-235.md 9b, 9g.
6. **Is `Ghat` computable below a scan?** — the whole walk reduces to one object, the
   gap-weighted opening transform `Ghat(m) = sum_o g(o) e(-mo/P)`; its mean-field part is
   closed form and carries 69-77% of the energy, the residual is depth-1 adjacency. Same
   question as the depth-1 term of the depth-sum identity. lateral.md U17, item 75.
7. **Why a record-size gap of a CRT word has only small neighbours** — named as "the part
   with no teeth in it at all". anchor-235.md 9.
8. **The uniform order** — nothing says which `m` makes `A_m` nilpotent and tight; `A_relax`
   is non-monotone (1,2,2,3,2,3,4,2,2 at m11..m41) and is a lossy proxy for `N(M)`, whose
   boundedness is open; the cover half is the only supplier. constructor.md R67(i), R75, R100.
9. **The query count / decision cost** — the CEGAR query count is bounded by a machine-free
   `T_4 + T_2 <= F^4 + F^2`, but the RATIO to that cap is unbounded and the count is
   strategy-dependent; a single arity-1 realisability refutation grows ~x15 per added gear.
   constructor.md R67(ii)(iii), R69.
10. **The first-moment transfer** — the independence model already gets the two-gap law right
    with a polylog-versus-linear margin, so what is missing is not a sharper inequality but
    an unconditional TRANSFER of that first moment (a large-deviation bound on the machine's
    own covering system), which no rearrangement invariant and no congruence potential can
    perform. That is Wall V with its scale named. constructor.md R64, R67.
11. **`Census29P` / `Census31P` are one-period claims and will not be kernel-checked** —
    what round 26 shrank is only their finiteness; removing them needs either the dictionary
    transfer from machine 23 or the sandwich lemma. formalist.md verdicts 21, 25.
12. **The generator's soundness bridge above 11->13, and the depth-sum glue** — the same
    lemma at two machines; done at m11/m13 by `Periodic.lean`, not at m17+ (machine 17's
    `ow` base case is a 19,305-step `decide`). formalist.md verdicts 11, 20; R31.3.
13. **The exact `L(M)` above the corpus, and whether the spectrum bound is tight infinitely
    often** — tight at m11, m13, m29 and at 13-88% of family rows; if `L = 2F/q' + 1 - o(1)`
    along the corpus then (B) is definitively false. lateral.md U23.
14. **The counted occurrence census of realised legal words by CRT enumeration at m41..m47**
    — named, not built; it is what would decide the three failing rows at m31.
    mechanic.md C54; constructor.md R96.
15. **The exact m41 4-tuple census** — complete at every span `<= 80`; the remaining paid
    population is 711,279 reverse classes, span 81-90 alone being 23 h at five workers.
    mechanic.md C41.
16. **`F_2(59) <= 173` carries a span condition** — unconditional only as `>= 173`; the upper
    half needs `F(61)`, which the project does not own, and is deliberately NOT stated in the
    kernel. formalist.md verdict 51; mechanic.md C48.
17. **The mirror lever where it pays** — instantiated at machine 11, where the direct bound
    is cheaper anyway; at machine 29 the base case `opSeq(N-1)` is not reachable by a walk
    and the lever needs a different route to it. formalist.md R28.8 item 0.
18. **A mod-4 lever** — proved impossible from any SYMMETRY of the opening set (the group is
    exactly `Z/2`); the two remaining candidates (a free `Z/4` action on a subset of
    configurations, or a pairing not induced by a map of `Z_P`) are untouched. lateral.md U10.
19. **The 41->43 and 43->47 rungs by the realisability chain** — pinned to the ORACLE's
    information content, not to cost or strategy: the transfer superset stalls at 222 under
    three settings. constructor.md R79.
20. **The residue law of the gap histogram and its machine-independent phase** — the richest
    classes are `+-s` of the SMALL gears (the small gears' letters are visible in the whole
    machine's gap histogram), and this is not the naive endpoint-survival prediction.
    UNEXPLAINED. mechanic.md C14, C17.5.
