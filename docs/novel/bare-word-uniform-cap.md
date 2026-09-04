# A uniform cap on the bare half of the longest legal word: L_bare(M) <= PSORD(q' mod 210) <= 5

Constructor, round 31.  Status per statement in section 3; nothing is announced as new
until section 6 carries a dated prior-art verdict.

---

## 1. WHAT IT IS

**Plain language.**  A sieve machine `M = {5, 7, ..., y}` acts on the slot line (slot `k`
stands for the pair `6k-1, 6k+1`; gear `q` blocks `k` exactly when `k = +-6^{-1} mod q`).
For the next prime `q'`, a gap of `M` is a LEGAL LETTER if its size is `0` or `+-d'` mod
`q'`, `d' = 2 * 6^{-1} mod q'`, and a run of consecutive gaps is a LEGAL WORD if every
letter is legal and the two nonzero residue classes strictly alternate along it (letters
that are `0 mod q'` are transparent to the alternation).  `L(M)` is the length of the
longest legal word that actually occurs, and the project's open crux is whether `L(M)` is
bounded as the machine grows: `L(M) + 1` is the largest number of consecutive openings of
`M` that one phase of `q'` deletes, and `L(M) + 2` is the depth at which the machine's
word-legal spectrum runs out.

The two SMALLEST legal letters are special.  Write `a` and `b` for the smallest positive
values in the two nonzero classes; they satisfy `a + b = q'` and `3a = q' -+ 1`, so they
are exact integer functions of `q'`.  Call a legal word BARE if every letter is `a` or `b`.
By the alternation rule a bare word is forced to be the alternation `a b a b ...` or
`b a b a ...` - there is no other bare word of any length.  The finding is that gears 5 and
7 alone cap the length of a realised bare word, **uniformly in the machine**, at 5; that
the cap is a function of `q' mod 210` only; and that for 28 of the 48 invertible classes
mod 210 the cap is `<= 2`.  That is the first bound on any part of `L(M)` that does not
grow with the machine.

### 1.1 DEFINITIONS

* Teeth of gear `g`: `{+-6^{-1} mod g}`.  Exposed set `E_g = Z_g \ teeth`; `|E_5| = 3`,
  `|E_7| = 5`, and the CORRIDOR `E_35 = {r mod 35 : r mod 5 in E_5, r mod 7 in E_7}`,
  `|E_35| = 15`.
* `u' = round(q'/6)` (the smaller tooth of `q'`, `6u' = q' -+ 1`), `d' = 2u'`;
  BARE letters `a = d' = 2u'`, `b = q' - a`.  Exactly:
  `a = (q'-1)/3` if `q' = 1 mod 3`, `a = (q'+1)/3` if `q' = 2 mod 3`.
* PREFIX-SUM OFFSET SET of a word `w = (w_1..w_m)`:
  `X(w) = {0, w_1, w_1+w_2, ..., w_1+...+w_m}`, `|X| = m+1`.
* `X` is ADMISSIBLE AT `{5,7}` if some translate `t + X` lies inside `E_5` mod 5 AND some
  translate lies inside `E_7` mod 7 - equivalently, by CRT, some translate lies inside
  `E_35` mod 35.
* `PSORD(c)`, for `c` coprime to 210: the largest `m` such that SOME bare alternation of
  length `m` (either phase) is admissible at `{5,7}`, when `q' = c mod 210`.
* `S = { c mod 210, gcd(c,210) = 1 : PSORD(c) <= 2 }`.
* `L_bare(M)` = the length of the longest REALISED bare legal word of `M`.

### 1.2 THE LEMMA (the case PSORD <= 2)

> **LEMMA.**  Let `M` be a machine containing the gears 5 and 7 (i.e. `y >= 7`), with next
> prime `q'` and bare letters `a`, `b`.  If neither `X_A = {0, a, a+b, 2a+b} = {0, a, q',
> q'+a}` nor `X_B = {0, b, a+b, 2b+a} = {0, b, q', q'+b}` is admissible at `{5,7}`, then
> `M` has no realised bare legal word of length 3, and hence `L_bare(M) <= 2`.

**PROOF.**  Two steps.

*(i) The word list is two words.*  A bare word has every letter in `{a, b}`; `a` and `b`
lie in the two different nonzero classes mod `q'`, and neither is `0 mod q'`, so no letter
of a bare word is transparent.  T3 therefore forbids two consecutive equal letters, and a
bare word of length 3 is exactly `(a,b,a)` or `(b,a,b)`.  Their prefix-sum offset sets are
`X_A` and `X_B`.

*(ii) A realised word's offsets are admissible.*  Suppose `(a,b,a)` occurs as three
consecutive gaps of `M` starting at the opening `k`.  Then `k, k+a, k+a+b, k+2a+b` are all
OPENINGS of `M`, so for every gear `g` of `M` none of them is at a tooth of `g`:
`k + X_A subset E_g (mod g)`.  Applying this to `g = 5` and `g = 7` (both in `M`) exhibits
the translate: `X_A` is admissible at `{5,7}`.  Same for `(b,a,b)` and `X_B`.  Contrapositive
of (ii) plus (i) is the lemma.  []

### 1.3 THE GENERAL FORM, AND THE UNIFORM CAP

The same two lines with a length-`m` alternation give the general statement, which is the
one worth naming:

> **THEOREM.**  For every machine `M` containing 5 and 7,
> `L_bare(M) <= PSORD(q' mod 210) <= 5`,
> and `PSORD` takes only the values 1, 2, 3, 5 over the 48 invertible classes mod 210:
>
> | PSORD | classes mod 210 | count |
> |---|---|---|
> | 1 | 11, 13, 17, 19, 41, 43, 47, 71, 73, 79, 101, 103, 107, 109, 131, 137, 139, 163, 167, 169, 191, 193, 197, 199 | 24 |
> | 2 | 29, 59, 151, 181 | 4 |
> | 3 | 1, 23, 31, 61, 67, 89, 97, 113, 121, 143, 149, 179, 187, 209 | 14 |
> | 4 | (none) | 0 |
> | 5 | 37, 53, 83, 127, 157, 173 | 6 |
>
> `S` is the union of the first two rows, `|S| = 28`; its complement has 20 classes.

That `PSORD` is a function of `q' mod 210` alone is the content of `3a = q' -+ 1`: `a mod
5` and `a mod 7` are determined by `q' mod 3, 5, 7`, and so are the offset sets `X` mod 5
and mod 7.  That `PSORD <= 5` is a finite enumeration over 48 classes.  Note `PSORD = 4`
is EMPTY: a class that admits a 4-letter bare alternation admits a 5-letter one.

**S, explicitly** (28 classes):
`11, 13, 17, 19, 29, 41, 43, 47, 59, 71, 73, 79, 101, 103, 107, 109, 131, 137, 139, 151,
163, 167, 169, 181, 191, 193, 197, 199`.
**Complement** (20 classes):
`1, 23, 31, 37, 53, 61, 67, 83, 89, 97, 113, 121, 127, 143, 149, 157, 173, 179, 187, 209`.

By Dirichlet the primes `q'` with `q' mod 210 in S` have density `28/48 = 7/12`.

### 1.4 THE QUANTIFIER, AND WHY THIS IS NOT R74

R74 (`docs/novel/uniform-order-bound.md`) proved `A_relax(M) <= 5` from the same phase-
saturation step.  It is a DIFFERENT invariant of the same walk, in two ways, and the two
must not be conflated:

* R74 asks for a CYCLE (an infinite alternating word every window of which survives), so it
  MINIMISES over the two starting letters - "one broken window kills the cycle".  The word
  question asks whether SOME bare word of length `m` is realised, so it MAXIMISES.
* R74 counts POINTS (deleted openings, the arity), this counts LETTERS.

In R74's own convention (`min` over phases, points) the distribution over the 48 classes is
`24 / 16 / 2 / 6` at orders `2 / 3 / 4 / 5`, order 5 exactly on `{37,53,83,127,157,173}`
and order 4 exactly on `{23,187}` - reproduced exactly as a gate (GATE A4).  The `max`/
letters convention gives the different table of 1.3, and `S` (28 classes) is not R74's
order-2 set (24 classes).  R74 caps a proxy that need not be realised anywhere; this caps
`L_bare`, a quantity in the derivation.

### 1.5 THE CORPUS GATE

`research/bare_lemma_r31.py`, log `research/data/r31/bare_lemma_r31.log`.  `L` from the
counted word census (`research/data/r30/occ_*_words.json`, exact cyclic, r30) at m11..m37,
from R97/R98 at m43/m47, from Mechanic's round-30 killer table at m41; `L_bare` from the
same census at m11..m37 and from `crt_dict.realised` at m41..m47.

    M     q'   a   b  q' mod 210  in S?  PSORD  L   L_bare   (a,b,a)?  (b,a,b)?
    m11   13   4   9      13      IN S     1    1     1        no        no
    m13   17   6  11      17      IN S     1    1     1        no        no
    m17   19   6  13      19      IN S     1    1     1        no        no
    m19   23   8  15      23      no       3    2     2        no        no
    m23   29  10  19      29      IN S     2    1     1        no        no
    m29   31  10  21      31      no       3    3     3       YES        no
    m31   37  12  25      37      no       5    3     3       YES       YES
    m37   41  14  27      41      IN S     1    2     1        no        no
    m41   43  14  29      43      IN S     1    2     1        no        no
    m43   47  16  31      47      IN S     1    2     1        no        no
    m47   53  18  35      53      no       5    4     4       YES       YES

`L_bare <= PSORD` at all 11 machines, tight at m29 (3 = 3) and at m37/m41/m43 (1 = 1);
`L_bare <= 2` at all 7 machines with `q'` in `S`, and in fact `<= 1` at six of them.
Every one of the 40 realised legal words on record at m11..m37 is admissible at `{5,7}`
(the proof step, checked on the data).

---

## 2. WHY IT MIGHT BE NOVEL

* It is a bound on a longest-run statistic of a sieve that does NOT grow with the sieve.
  Every other instrument the project has on the same quantity grows: the exposure cap
  `EXPCAP` is 16-18 above `L` at m37/m53 (`cover-half-counter-ladder.md`), the corridor cap
  over the full alphabet (`CORRCAP`, R75) is INFINITE from `53 -> 59` on, and the
  independent-letter density model grows like `log N / log(q'/3)`.  What makes the bare
  half different is that its alphabet has exactly TWO letters at every machine, forever,
  while the full legal alphabet grows like `3F(M)/q'`.
* The bound is a residue condition on the incoming prime only (`q' mod 210`), computed from
  two gears, and it decides a question about arbitrarily large machines.
* The `PSORD = 4` gap (no class admits a 4-letter bare alternation without admitting a
  5-letter one) is a small, checkable rigidity of the 35-state corridor walk.

The honest boundary: this bounds `L_bare`, not `L`.  At m37, m41 and m43 the machine's `L`
is 2 while `L_bare` is 1, carried by words containing the letter `q'` itself.  So the
theorem is one half of a decomposition, not a bound on the crux.

---

## 3. PROOF / STATUS

| statement | status | pointer |
|---|---|---|
| a bare word of length `m` is one of two alternations | **PROVED** (two lines, T3) | 1.2(i) |
| a realised word's offsets are admissible at every gear of `M` | **PROVED** (definition of an opening) | 1.2(ii) |
| `L_bare(M) <= PSORD(q' mod 210)` | **PROVED** | 1.2, 1.3 |
| `a = 2*round(q'/6)`, `3a = q' -+ 1` | **PROVED**, asserted at 2,258 primes | GATE A1 |
| `PSORD` depends only on `q' mod 210` | **PROVED** (CRT) + asserted: constant on each class over 2,258 primes, and equal to the pure mod-210 vehicle | GATE A2 |
| `{5}`-fit AND `{7}`-fit == corridor-mod-35 fit | **PROVED** (CRT), asserted on 4,186 instances | GATE A3 |
| the PSORD table of 1.3, and `PSORD <= 5` | **SCRIPT-VERIFIED**, exhaustive over the 48 classes, exact integer arithmetic | GATE A5 |
| R74's own distribution 24/16/2/6 reproduced in R74's convention | **SCRIPT-VERIFIED** | GATE A4 |
| the corpus table of 1.5 | **SCRIPT-VERIFIED** (census + exact CRT decisions) | GATES B1-B4 |

**KERNEL CONFIRMATION, same round (Formalist round 31 - this paragraph corrected and
extended by Formalist at round close).**  `proofs/BareAlternation.lean` defines the same
`S` - 28 classes, element for element - and upgrades three rows of the table above from
SCRIPT-VERIFIED to KERNEL-CHECKED (`lake env lean AxiomCheck.lean`: 508 declarations,
sorryAx 0, native_decide 0):

* `bareAlt_inadmissible_iff` (`S` itself), `S_card = 28`, `S_mirror`, `S_half_mirror`;
* the whole PSORD table of 1.3, not only its bound: `bareAdm_downward` (so the count is a
  maximum), `psord_le_five`, `psord_ne_four` (**PSORD = 4 is empty**),
  `psord_eq_one_iff` (the 24 classes), `psord_eq_two_iff` (`{29,59,151,181}`),
  `psord_eq_five_iff` (`{37,53,83,127,157,173}`), and `S_iff_psord` (`c in S <-> PSORD c <= 2`);
* the necessary condition of 1.2(ii) as a theorem over an ABSTRACT machine -
  `fitsB_of_open`, `open_of_gapWord`, `no_gapWord`, `no_bare_run`, `no_bare_run_ge` -
  stated on the OPENING PREDICATE rather than on consecutive gaps, so it forbids the
  offsets being open at all;
* the class-to-machine bridge (`aOfClass_mod_five/_seven`, `bOfClass_mod_five/_seven`,
  `bareAdmAB_congr`) and the assembled `no_bare3_of_class_mem`.

Instantiations in `proofs/BareAltInst.lean` are at m23 (`L_bare(23) <= 2`), m37
(`L_bare(37) <= 1`: no bare PAIR at all) and m41, m43 (both rotations, on the opening
predicate) - NOT at m19, whose `q' = 23` is not in `S`.  `proofs/WordLegal13.lean` and
`proofs/WordLegal17.lean` turn the cap into `L(13) = 1` and `L(17) = 1` (and
`A_kill(13->17) = A_kill(17->19) = 2`, `J_max(13) = 3`) because `F(M) < q'` there makes
every legal letter bare.

One identity the kernel adds that section 1.4 does not claim: `c in S` is equivalent to
`LiteralCapTable.capC c <= 3` (`inadmissible_iff_capC`, through R74's `ps_max_eq_capC`),
so in the MAXIMISING convention the bare cap and round 29's literal cap are the same
object at every one of the 48 classes - which is consistent with 1.4, since what differs
from R74 is the min/max convention and the points/letters count, not the underlying walk.

Two lanes reached the same 28-element set by different vehicles in the same round.  The
remaining non-kernel statements are listed for Formalist in the round-31 append to
`agents-shared.md`.

---

## 4. IMPLICATIONS

* Inside the project: `L(M) = max(L_bare(M), L_pad(M))` where `L_pad` is the longest
  realised legal word using at least one non-bare letter.  With `L_bare <= 5` PROVED
  uniformly, requirement (B) - `L(M)` bounded - is now EXACTLY "`L_pad(M)` bounded".  The
  crux has shrunk from all legal words to the words that use a letter of size `>= q'`.
* The theorem immediately yields a NEW value, at a machine no census reaches.  At `m53`,
  `q' = 59`, `59 mod 210 = 59` and `PSORD(59) = 2`, so `L_bare(53) <= 2`; the recorded
  `L(53) = 3` (from `A_kill(53 -> 59) = 4`) then forces `L_pad(53) = 3` exactly.  With the
  measured `L_pad(47) = 3` (`research/lpad47_r31.py`: `(18,35,53)`, `(18,53,35)`,
  `(35,18,53)` and their mirrors realised, every non-bare length-4 word refuted by R98)
  the non-bare half is 0,0,0,1,1,1,2,2,2,2,3,3 at m11..m53 - it takes every value from 0
  to 3, and `L > L_bare` at four machines (m37, m41, m43, m53).  So the theorem caps one
  half of `L` by a constant and leaves the other half visibly growing.
* It explains a measured fact that had no explanation: `L_bare` is 1 at m37, m41 and m43 -
  three consecutive machines - because `PSORD(41) = PSORD(43) = PSORD(47) = 1`, i.e. gear 5
  alone refutes the two-letter bare alternation there.  R74 recorded the m37 instance
  (`gear 5 refutes (14,27)`) as a correction to a hardcoded assumption; it is a residue
  class, and it recurs.
* Outside: a two-modulus local obstruction that caps the length of an alternating pattern
  in a sieved set uniformly in the sieve, with the cap read off the incoming modulus.

---

## 5. UNSOLVED QUESTIONS IT TOUCHES

Boundedness of `A_kill = L + 1` (the project's crux, now reduced to `L_pad`); requirement
(D) and the tolerance route; Jacobsthal-type extremal problems for the two-teeth sieve;
`docs/novel/uniform-order-bound.md` (R74/R75) whose quantifier this corrects.

---

## 6. PRIOR-ART CHECK

**Not yet checked** (this lane has no web access).  Search terms for the manager:
"longest alternating run residue classes sieve uniform bound"; "local obstruction bounded
independent of modulus covering system"; "arithmetic progression avoiding two residues mod
5 and 7"; "Jacobsthal function consecutive deleted residues bound"; "phase saturation
covering congruences".  Nearest relatives inside the project: `uniform-order-bound.md`
(R74/R75, the same walk with the other quantifier), `literal-cap.md` (the `q' mod 210`
classification of literal chains - the same arithmetic, maximised, and kernel-checked),
`cover-half-counter-ladder.md` (R100, the instrument that does NOT bound `L`),
`legal-word-length-mechanism.md` (Mechanic r30, the density half).
