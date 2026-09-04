# The pair statement: F_2(M) <= F(M) + q'

Prover A, round 32.  Everything below is either PROVED (proof given), REDUCED TO a named
statement, or MEASURED (script + machines named).  Vocabulary as in
`docs/proof-search/alignment-rules.md` section 0; max-gap convention throughout.

## 1. The statement and the reduction

**Pair statement (PS).**  For every machine `M = {5..p}`, with `q'` the next prime after `p`:
`F_2(M) <= F(M) + q'`, where `F_2(M)` is the largest sum of two consecutive gaps of `M`
(the longest stretch containing exactly one opening, endpoints open) and `F(M)` the largest gap.

**Reduction (two lines).**  The attainment identity (R68, alignment-rules 3.2, proved both ways
and structural on the family) is `F(M + q') = max( F_2(M), max_{J >= 3} Q*_J(M; q') )`.  Hence
the budget inequality `F(M+q') <= F(M) + q'` holds iff BOTH `F_2(M) <= F(M) + q'` (the pair
statement, the `J = 2` layer) AND `max_{J>=3} Q*_J(M; q') <= F(M) + q'` (the chain statement,
the `J >= 3` layers).  Conversely the deletion ladder (`F_2(M) <= F(M+q')`, alignment-rules 3.4)
shows the pair statement is IMPLIED by the budget inequality at the same rung.

## Pre-registration

Written before any computation.  Expected status of each lemma:

- **L1 (reduction)** -- PROVED (on record; restated above).
- **L2 (trivial discharge)** -- PROVED.  `g_L + g_R <= F + min(g_L, g_R)`, so PS holds at every
  opening whose smaller flank is `<= q'`; in particular PS is FREE whenever `F_2(M) <= 2q' + 1`.
  Expect: by the recorded spectra this covers m11..m31 (F_2 = 11,16,25,31,39,55,68 against
  2q'+1 = 27,35,39,47,59,63,75), and m37 (F_2 = 90 > 83) is the first machine with content.
  Expect ZERO openings with both flanks `> q'` at m11..m23 (this is forced by L2, so the gate is a
  consistency check, not a measurement).
- **L3 (the column-0 instance)** -- PROVED that PS at column 0 is exactly `F(M) >= 2 d_0 - q'`
  (mirror: the pair at 0 is `(d_0, d_0)`).  Expect the real corpus to satisfy it trivially
  (`d_0 <= 10 <= q'` at m7..m41), and expect the family member `V(19) (1,1,4,3,5,2)` on record to
  refute the teeth-free form `"columns 1..d_0-1 blocked => a gap >= 2d_0 - q'"` (F = 26,
  d_0 = 25, q' = 23).  Expect d_0 <= q' for every prime p <= 10^6 (real machine) with the
  ratio 6 d_0 / q' never above ~1.3 -- MEASURED, and the general statement is a twin-Bertrand
  postulate (a twin prime pair in (p, 6q'+1]) which is OPEN in the literature.
- **L4 (re-phasing / sole-coverer lemma, from the manager's one-class argument)** -- PROVED,
  teeth-free, both worlds: re-phasing one gear `q0` (translate by a multiple of `P/q0`) to strike
  `x` leaves every column struck by another gear struck; so `F(M) >= ` the blocked run around `x`
  in the re-phased machine, which is bounded by the nearest `q0`-SOLE columns that the new teeth
  do not re-cover.  Corollary: if `g_L + g_R > F` then every gear is the sole striker of some
  column of the stretch.  Expect the single-gear certificate (`cert(x) >= g_L + g_R - q'` for
  some `q0`) to certify EVERY pair with `g_L + g_R > F` at the real machines m11..m23 (they are
  all free by L2 anyway, so this is a mechanism check) and to FAIL at some family members
  (starting with the wrap failure, where it cannot succeed because F < 2d_0 - q').  Expect the
  two-gear version to certify more but not all.
- **L5 (one-class)** -- MEASURED: at P_5..P_9 the one-hole record equals j(P_{k+1}); sole-coverer
  counts per prime in the one-hole record stretches are >= 1 for every prime (forced by L4's
  corollary since one-hole record > j) -- expect the smallest primes to have the fewest sole
  positions (1-2) and the top prime the most.
- **L6 (negative correlation, the manager's branch 5b)** -- MEASURED on the family: expect the
  effect (gap after a long gap shorter than mean; F_2 below the shuffle null) to be PRESENT on
  most tooth-counterfactual members too, i.e. structural rather than a real-teeth fact; expect
  it to be absent exactly at column 0 (correlation +1 there).  Expect NOT to prove it, and record
  that a conditional-mean statement cannot supply an extremal one.
- **L7 (non-wrap family pairs)** -- MEASURED: reconfirm the record's "non-wrap slack positive at
  every member m11..m19"; test the hypothesis that the TIGHTEST non-wrap pairs sit at columns
  `x` congruent to 0 modulo all gears but one (one-gear-shifted copies of the wrap structure).
  Status uncertain; pre-registered as a hypothesis, expect it to hold for a majority but not all.
- **Overall** -- expect NO proof of PS.  Expected obstruction: the column-0 instance is a
  twin-existence statement (first twin-rough column above p at most (F+q')/2), which no
  Jacobsthal-type upper bound can supply and which is false teeth-free; the generic instances are
  one-hole Jacobsthal statements with no mechanism on record beyond L4.

## 2. The attack: lemmas

Gate script: `research/proof/pair_statement_r32.py` (entry points `real`, `family`, `famfail`,
`exhibit`, `oneclass`, `d0`); logs and JSON beside it (`pair_*_r32.{log,json}`).  All sieves are
full periods in exact integer arithmetic; "cert" values are gaps of the machine exhibited by
re-phasing (L4), never estimates.

### L1 -- reduction.  PROVED (on record).
See section 1.  The pair statement is the `J = 2` layer of the attainment identity and is implied
by the budget inequality at the same rung via the deletion ladder.

### L2 -- trivial discharge.  PROVED.
For consecutive gaps `g_L, g_R` at an opening `x`: `g_L + g_R <= F + min(g_L, g_R)` because the
larger gap is at most `F`.  Hence PS holds at every opening whose smaller flank is `<= q'`, and
a VIOLATING pair must have both flanks in `[q'+1, F]`, which forces `F_2 > 2q' + 1`.
Corollary: PS is free at every machine with `F_2(M) <= 2q' + 1`.  With the recorded spectra
(`F_2 = 11, 16, 25, 31, 39, 55, 68` against `2q'+1 = 27, 35, 39, 47, 59, 63, 75`) PS is free
through m31 -- not only through m17 as the brief says -- and m37 (`F_2 = 90 > 83`) is the first
machine where it has content.  At m47 the recorded maximiser `[54, 80]` has both flanks above
`q' = 53`, so from there L2 no longer discharges even the argmax (the slack is still 37).
Gate: `real` -- `bothflanks>q' = 0` at m11..m23, `L2free = True` at all five.

### L3 -- the column-0 instance.  PROVED (the equivalence); its routes REDUCED TO open statements.
The mirror `k -> -k` preserves the opening set and column 0 is open (kernel: `Mirror.mirror_gear`,
the shield).  So the openings adjacent to 0 are `+-d_0` with `d_0` the least positive opening,
the pair at 0 is `(d_0, d_0)`, and

    PS at column 0  <=>  2 d_0 <= F(M) + q'  <=>  F(M) >= 2 d_0 - q'.

On the real machine `6 d_0 -+ 1` are both `p`-rough (no gear `5..p` strikes `d_0`, and `6k +- 1`
is coprime to 6); if `6 d_0 + 1 < q'^2` both are prime, so `d_0` is the column of the FIRST TWIN
PRIME PAIR ABOVE `p`; and every `k <= (p-1)/6` is blocked (`6k+1 <= p`, so `6k-1 >= 5` has a
prime factor `<= p`), whence `d_0 > (p-1)/6`.
Gate `d0`: for all 78,496 primes `5 <= p <= 10^6`, `d_0 <= q'` (max `d_0/q' = 0.2857` at `p = 5`;
`d_0 = 2,3,3,5,5,5,7,7,7,10` at m7..m41, matching the record), so the instance is true by L2 at
every real machine in range.  Gate `exhibit`: the family member `V(19) (1,1,4,3,5,2)` has
`F = 26, F_2 = 50 = 2 d_0, d_0 = 25, q' = 23`: `2d_0 - q' = 27 > F`; the teeth-free form is
FALSE.

What a proof for ALL `p` would need -- three routes, each reduced to a named statement:
- (a) via L2: `d_0 <= q'`, i.e. a twin prime pair in `(p, 6q'+1]`, a subset of `(p, 12p+1]`, for
  every prime `p` -- a twin-prime Bertrand postulate.  OPEN.  (Not even a pair of `p`-rough
  numbers at distance 2 below `p^{2+eps}` is a theorem as far as I can cite: the dimension-2
  lower-bound sieve needs the range to exceed `p^{beta_2}`, `beta_2 ~ 4.27`, the
  Diamond--Halberstam--Richert sieving limit.  Literature claim, not verified here.)
- (b) structurally: "columns `1..d_0-1` all blocked forces a gap `>= 2d_0 - q'` somewhere" --
  FALSE for symmetric two-tooth sieves (the exhibit), so any proof must use the real teeth; but
  the only teeth-specific information about `(0, d_0)` is the factorisation of `6t +- 1` for
  `t < d_0`, i.e. "no `p`-rough twin pair below `6d_0 - 1`", which is route (a) again.
- (c) via a provable lower bound on `F`: Rankin / Ford--Green--Konyagin--Maynard--Tao coverings
  transfer to the machine by translation (choosing the start column chooses every gear's phase by
  CRT, and each gear offers two classes at a fixed separation, at least the one class those
  constructions use), giving `F >> p log p loglog p / (logloglog p)^2` [literature, not verified
  here]; then one needs `d_0` below that -- a twin-Cramer statement.

So the column-0 instance of PS is a twin-EXISTENCE statement dressed as a gap inequality.  It is
where the real teeth enter, and they enter as the conclusion of the whole programme, not as a
tool: combined with ANY bound `F(M) <= B(p)`, PS at column 0 places a `p`-rough twin pair
`6k +- 1` at `k <= (B(p) + q')/2`; with the measured `F ~ q'^2/24` that is a twin prime pair
below `q'^2/8`.

### L4 -- the re-phasing (sole-striker) lemma.  PROVED, teeth-free, both worlds.
Let `S` be a set of gears and for each `q in S` let `T_q` be a translate of `q`'s tooth set.  The
machine `M'` with the gears of `S` moved to `T_q` is a TRANSLATE of `M` (CRT: `s = 0 mod P/prod S`,
`s = t_q mod q`), so every gap of `M'` is a gap of `M` and `F(M) >= F(M')`.  Let `x` be an opening
of `M` with flanks `g_L, g_R` and choose `S`, `T` so that some `q0 in S` strikes `x` (`x in T_q0`).
Then the blocked run of `M'` through `x` contains every column of `(x-g_L, x+g_R)` struck by a
gear outside `S` (old teeth) or by a gear of `S` at its new teeth; the only columns that can open
are those struck ONLY by gears of `S` and not re-struck.  Two corollaries:
- **Sole-striker corollary.**  If `g_L + g_R > F(M)` then EVERY gear of `M` is the sole striker
  of at least one column of `(x-g_L, x+g_R)` other than `x`.  (Else re-phase that gear alone onto
  `x`: the whole stretch is blocked in `M'`, a gap `>= g_L + g_R > F`.)  This is the manager's
  one-class argument, verbatim in two-class.  Gate: every sole count is `>= 1` at every `F_2`
  maximiser at m11..m23 and P_5..P_9 (`real`, `oneclass` logs).
- **Single-gear certificate.**  With `S = {q0}` and `T` one of `{x, x+a}`, `{x-a, x}` (`a` the
  tooth separation), the columns that open are the `q0`-sole columns whose residue is not in `T`;
  for real teeth `{u, -u}` the class `+u` is re-covered iff `x = 3u = 2^{-1} (mod q0)` and the
  class `-u` iff `x = -3u`, otherwise no sole column of `q0` survives.  Walking outward from `x`
  in `M'` gives an exact gap `cert(x, q0, T)` of `M`, and PS at `x` follows whenever
  `max_{q0, T} cert >= g_L + g_R - q'`.

MEASURED (exact gaps, at every pair with `g_L + g_R > F`, i.e. every pair with content):
- real m11..m23: certified 20/20, 88/88, 124/124, 400/400, 130/130 pairs; the top gear alone is
  NOT always the certifier (best gear at the `F_2` maximisers: 7, 13, 13, 17/19, 7).
- one-class P_5..P_9 (`oneclass`): 22/22, 22/22, 94/94, 70/70, 286/286 pairs above `j`; the top
  prime alone certifies 22, 22, 90, 66, 282 of them and its loss reaches 10, 14, 22, 24, 30
  against `q' = 13, 17, 19, 23, 29` (exceeding `q'` at P_7..P_9), while the best prime's loss is
  at most 8, 10, 12, 16, 18 -- so the certificate is not the small-`k` artefact `j < 2p`; the best
  prime is usually not the top one (P_9: 17 at 104 pairs, 23 at 86, 19 at 84, 7 at 12).
- family: V(11) 446/446; V(13) 6362/6364 (the two misses are `(2,25)`/`(25,2)` at member
  `(1,1,5,1)`, both free by L2, both certified by two gears); V(17) 97,672/97,682 (ten misses,
  all with a flank `<= 7`, all L2-free, all two-gear certified); V(19): 345,330/346,090 (the first 30 above-record pairs per member); the 760 misses sit at 332 members and ALL but one have their smaller flank `<= 23` (L2-free); two gears certify 756 of the 760, and the four two-gear failures are the wrap pair `(25,25)` of the failing member and three L2-free record-plus-small-flank pairs `(33,11)`, `(39,6)`, `(6,39)`.  So on the whole family m11..m19 (14,610 members, `famfail` logs) every pair with content is certified by L2 or by one-gear re-phasing EXCEPT the single wrap pair, where no certificate can exist.
- at the wrap-failure member the certificate at `x = 0` reaches 18 (per gear 18, 16, 14, 6, 10,
  4) with one gear and 26 = F with two, against the 27 needed -- it cannot succeed, and the reason
  is visible: every gear is a sole striker within nine columns of 0 (`2:[19] 3:[13] 5:[17]
  7:[11] 8:[7] 9:[5]`).

Why L4 is a certificate and not a proof: `cert` is bounded by the positions of the nearest
surviving sole columns of one gear, which lie on two residue classes mod `q0` and can sit next to
`x`; a bound `min-loss <= q'` needs the JOINT placement of every gear's sole columns, i.e. the
covering structure itself.  No residue-only bound follows (the exhibit is the counterexample).

### L5 -- the one-class comparison.  MEASURED, plus one structural fact PROVED.
Confirmed (`oneclass`): one-hole record `= 22, 26, 34, 40, 46 = j(P_6..P_10)` at P_5..P_9; the
pair at `n = 1` is `(2, q'-1)`.  PROVED: the one-class world has NO counterfactual family --
one residue per prime is always a CRT translate of `0 mod q` -- so "teeth-free" and "the true
Jacobsthal sieve" coincide there.  Consequences: (i) a teeth-free proof of the two-class generic
instances would prove `onehole(P_k) <= j(P_k) + p_{k+1}`, which equals the increment
`j(P_{k+1}) - j(P_k) <= p_{k+1}` while `j(P_k) < 2p_{k+1}` (through `k = 18`) -- a major open
result; (ii) the two structures the two-class world has and one-class lacks are exactly the
mirror fixed point being OPEN (the shield forces the pair `(d_0, d_0)`; in one-class `0` is
struck by every prime and the pair at 1 costs `q'+1`) and the letters `a, b ~ q'/3` (two kills in
one stretch need a gap `>= 2p` one-class but only `>= a` two-class, which is why the two-class
pair statement decouples from the increment at m19 while one-class stays coupled to `k = 18`).

### L6 -- the lag-1 negative correlation (manager branch 5b).  MEASURED; NOT a route.
Real m11..m23 reproduce the manager's numbers exactly (`E[gap after >= 0.7F] = 2.77, 3.07, 3.44,
3.52, 3.09` vs mean `2.85, 3.37, 3.82, 4.27, 4.68`; `E[after record] = 2.00, 3.00, 3.70, 2.60,
3.00`; shuffled `F_2` ranges `(12,14), (17,21), (27,34), (39,44), (48,55)`).  On the family the
effect (`E[after >= 0.7F] < mean`) is present at 27/30, 165/180, 1301/1440, 12,501/12,960
members, and `F_2` lies below one shuffle at 19/30, 134/180, 1257/1440, 12,320/12,960 -- so it
is a property of symmetric two-tooth sieves in ~90% of members, not of the real teeth, and it is
absent in the rest.  Residue form, PROVED: at an opening `x` the right offset set of gear `g` is
`R_g = {u_g - x, -u_g - x}` and the left one is `L_g = -R_g`; `R_g = L_g` iff `x = 0 (mod g)`,
and `R_g`, `L_g` are DISJOINT otherwise (`u - x = x + u` or `-u - x = x - u` force `x = 0`; the
cross cases force `x = +-u`, blocked).  So the left tiling is the right tiling reflected gear by
gear, identical exactly on the gears dividing `x`, and at `x = 0` on all of them
(`W^- = W^+ = d_0`).  I did not find an inequality on `E[W^- | W^+ >= L]` from this, and the
route is closed for a structural reason: PS is EXTREMAL and the extremal pair of the failing
member is at `x = 0`, where the correlation is `+1`; no conditional-mean statement can supply an
extremal bound at the one point the mirror pins.

### L7 -- non-wrap pairs on the family.  MEASURED.
Minimum non-wrap slack `F + q' - (g_L + g_R)`: 6 at V(11) (member `(1,1,3)`), 6 at V(13)
(`(1,1,1,1)`), 5 at V(17) (`(1,2,2,4,7)`), 4 (member `(2,2,5,3,7,3)`, `F = 20`, pair sum 39 at `x = 90584`) at V(19) -- consistent with the
record's `6/6/5/4`.  No non-wrap pair with slack `<= 3` exists at V(11)..V(17), so the
pre-registered residue-structure hypothesis has no test cases there; at V(19): none either (0 of 12,960 members has a non-wrap pair with slack `<= 3`), so the hypothesis is untested on the whole family and is NOT scored.
`F_2 = 2d_0` at 4/30, 5/180, 7/1440, 11/12,960 members: the wrap pair is the `F_2`
maximiser only rarely, and never at a real machine m11..m23.

## 3. No proof landed -- the obstruction, exactly

**The smallest statement I could not prove** is the column-0 instance

    F(M) >= 2 d_0(M) - q'      (equivalently  d_0 <= (F(M) + q')/2),

with `d_0` the column of the first twin prime pair above `p`.  It is implied by PS, it is
numerically trivial at every computed machine (`d_0 <= q'`), and every route to it for all `p`
is one of: a twin-Bertrand postulate (a twin pair in `(p, 12p+1]`), a twin-Cramer statement, or
a teeth-free structural bound that the exhibit refutes.  The real teeth enter PS here, and they
enter as twin existence.

**The generic instances** (`x` not 0, both flanks `> q'`) are one-hole Jacobsthal statements:
"the best one-hole stretch beats the best zero-hole stretch by at most `q'`".  Every tool on
record and every one I built gives the relation only as a certificate: the re-phasing lemma (L4)
exhibits a gap `>= g_L + g_R - loss` with `loss` the distance from the stretch ends to one gear's
nearest surviving sole columns, and `loss <= q'` held at every pair with content on every real
machine and one-class primorial checked but is a fact about the joint placement of sole columns,
not about residues.  The heuristic margin is polylog-versus-linear (`F_2 - F ~ log^3 p` against
`q' ~ p` under independent gaps), consistent with the record's `S_2 = 9..49` growing, and the
negative lag-1 correlation only widens it; none of that is a proof.

**Classification, argued.**  PS is not "a one-hole Jacobsthal bound needing a constant-factor
improvement on Iwaniec 1978":
- Iwaniec-type bounds are ABSOLUTE upper bounds on covered lengths (`F, F_2 << p^2 log^2 p`); PS
  is RELATIVE (`F_2 - F <= q'`), and no absolute bound on `F_2` combines with a provable lower
  bound on `F` (Rankin-type, `~ p log p loglog p`) to give it -- the two sides differ by a factor
  `p/(log p)^{O(1)}`.  A constant-factor improvement would not change this.
- Worse, at column 0 PS needs a LOWER bound on `F` relative to `d_0`, the opposite direction from
  the entire Jacobsthal upper-bound literature; and joined to any Iwaniec-type upper bound it
  yields a `p`-rough twin pair below `p^{2+o(1)}`, which sits below the dimension-2 sieving limit.
- The structures the two-class world has that the literature does not: the mirror (it CREATES the
  hard instance rather than helping), the shield (column 0 unkillable, which is why the wrap
  stretch can never be merged in place), the survivor generator (`F_2(M) = F(M+Q)` for far `Q`,
  an exact identity that restates PS as "a far gear costs at most `q'`", with no slack), the
  letters (they decouple PS from the increment at m19).  None of them yields a lower bound on `F`
  or an upper bound on `d_0`.  The only structure that does work -- re-phasing (L4) -- is
  one-class as well, so if it were ever made into a proof it would prove the one-class increment
  through `k = 18` too.

Two honest riders.  (i) "Prove either the two-class or the one-class statement": neither; the
one-class pair statement is the Jacobsthal increment while `j < 2p`, and the two-class one
contains it (L5) plus a twin-existence instance.  (ii) Nothing here says PS is false: every
computed rung has slack `>= 9`, the family fails only at the wrap pair, and the heuristic margin
is enormous.  It says PS cannot be settled before the conclusion it serves.

## 4. Pre-registration scored

- L1 PROVED as expected.  L2 PROVED; free through m31 (expected); 0 both-flank openings (expected).
- L3 PROVED/REDUCED as expected; `d_0 <= q'` to `10^6` (expected; max ratio 0.29, below the 1.3
  guessed).
- L4 PROVED; real m11..m23 certified 100% (expected); one-class 100% at P_5..P_9 (not
  pre-registered -- new); family single-gear misses exist (expected) but ALL are L2-free and
  two-gear certified at V(13)/V(17) (stronger than expected); V(19): single-gear 345,330/346,090 and every miss but the wrap pair L2-free -- the combined certificate (L2 or one-gear re-phasing) covers the entire family except the one pair that is genuinely false.
- L5 as expected (the top prime has the fewest sole positions, 1-2; the smallest primes the most,
  6-10).
- L6 present at ~90% of family members: "structural, not real-teeth" confirmed; absent at ~10%
  (not anticipated).  No proof, as expected.
- L7: no tight non-wrap pairs below V(19), hypothesis untested there; V(19): no test cases at V(19) either; untested, not scored.
- Overall: no proof; obstruction as pre-registered.

## 5. Kernel-shaped items (cheap; none load-bearing for a proof)

- L2 as a one-line `Nat` inequality on three naturals.
- L3's equivalence from `Mirror.mirror_gear` plus "least positive opening".
- L4's translation lemma: re-phasing a set of gears is a translate of the machine (CRT), hence
  `F(M') = F(M)`; the sole-striker corollary follows by `decide` at m11/m13 for every above-record
  pair.
- The residue fact of L6 (`L_g = -R_g`; equal iff `g | x`) from `6u = 1`.
