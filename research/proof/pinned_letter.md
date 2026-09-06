# Node 4.i.a.i.a.1 - THE PINNED LETTER: why `a_L + r(a_L)` sits at the record

Parent: node **4.i.a.i.a, the short-letter row** (`research/proof/short_letter_row.md`, STRONG,
2026-09-06), whose single hand-forward was a measured law with no mechanism:

> **THE PINNED LETTER (measured; 8 of 8 rungs, one of them out of sample).**
> `F(M) <= a_L + r(a_L) <= F(M) + 3`, i.e. `r(a_L) <= F(M) - a_L + 3`.
> *"It is measured, not proved, and proving it is the single thing this branch hands forward."*

What spawned this branch exactly: the parent's observation that the law is **not** a property of a
general gap size (`v + r(v) > F + 3` at 23 of 41 sizes at m29, reaching `F_2` at `v = 20, 25, 30,
35`), together with the closer law (`Leg(a_L) = {q', q' -+ 2}`) which says an `a_L`-gap is the one
size the machine `M` cannot close at both ends. The brief's candidate mechanism: an **uncoupled**
gap is struck at its two ends by disjoint gear sets, so the gap and a neighbour can be re-phased
into one blocked run (the glue idea) with no shared gear to obstruct, and the pair's span is then
capped by `F` rather than by `F_2`.

Scripts in `research/anchor235/r62/`; result outputs in `research/anchor235/r62/results/`
(untracked). Every number this document relies on is written into the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, defined exactly

Machines `M_0 = {5}`, ..., `M_8 = {5..31}`; a **rung** is `(M, q')` with `q'` the incoming gear.
`u_g = 6^{-1} mod g`, teeth `T_g = {u_g, -u_g} mod g`, `d_g = 2 u_g`; letters
`a_L = min(2u_{q'}, q' - 2u_{q'}) mod q'`, `b_L = q' - a_L`, `3 a_L = q' -+ 1` (file 05 T1).
`F(M)` is the largest gap over a full period, `F_2(M)` the largest sum of two adjacent gaps,
`D[a][v]` the adjacent-pair dictionary, `r(v) = max { a : (a,v) or (v,a) in Dict_2 }` the row
maximum (the largest SINGLE neighbour of a `v`-gap), and

    G(v) := v + r(v)        the longest 2-run one of whose gaps has size v
    E(v) := G(v) - F(M)     the EXCESS of that 2-run over the record

so the pinned letter reads `0 <= E(a_L) <= 3`.

**Coupling.** The chain law (file 05 (C)) says a gear `p` can strike two columns at distance `v`
iff `v = 0, +d_p, -d_p (mod p)`, i.e. iff `p | v` or `p | 3v - 1` or `p | 3v + 1`. Write

    Pad(v) = { p prime >= 5 : p | v }
    Leg(v) = { p prime >= 5 : p | 3v - 1 or p | 3v + 1 }
    Coup(v) = Pad(v) u Leg(v)

Three predicates are tested separately, because the brief's phrase ("no gear of the machine can
strike both ends of a `v`-gap, i.e. `Leg(v)` has no member `<= y`") names the chain-law condition
but the divisor set of only half of it:

    U_leg(M)  : Leg(v) n M = {}          (the brief's literal predicate)
    U_full(M) : Coup(v) n M = {}         (the chain law's actual predicate)
    U_pad(M)  : Pad(v) n M = {}          (the complement half)

### 0.2 The theory

**T. The pinned letter is the glue lemma run on a 2-run instead of a 3-run, with the middle
opening closed by ONE re-phased gear. The gap `v` and its largest neighbour `a` already sit
adjacent in the machine; the only thing between them and a single gap of size `v + a` is the
middle opening `x_1`. Re-phase exactly one gear `h` onto `x_1`. By CRT the resulting configuration
occurs somewhere in the period, so `F(M) >= v + a - (whatever the re-phasing costs at the ends)`.
The gear `h` must strike `x_1` and miss `x_0` and `x_2`; a gear that divides `v` or `a` (a PAD
gear) strikes the two ends of that flank together and is therefore disqualified, so the sizes that
break the law should be the ones whose pad is inside the machine. The constant `3` is the cost of
the ends, not of the middle.**

Mechanism, stated before measuring. Let the 2-run be `x_0 < x_1 < x_2`, `a = x_1 - x_0`,
`v = x_2 - x_1`, `S = a + v`. Fix all gears at their observed phase except `h`, whose teeth are
moved to `{u_h + s, -u_h + s}` for some shift `s`. By CRT (the moduli are coprime) the resulting
window pattern occurs at some column of the period, so any blocked run it exhibits is a genuine
gap of `M` and is `<= F(M)`. Three conditions decide the construction:

1. `h` strikes `x_1` (the middle opening is closed) -- one of the two teeth is placed on `x_1`,
   which fixes `s` up to the choice of tooth: two choices per `h`.
2. `h` misses `x_0` and `x_2` (the ends stay open). Given (1), `h` also strikes `x_0` iff
   `a = 0` or `+-d_h (mod h)`, and `x_2` iff `v = 0` or `+-d_h (mod h)`; the `= 0` case
   (`h | a`, `h | v`) kills BOTH tooth choices at once, the `+-d_h` cases kill one.
3. `h`'s own strikes inside the two flanks move with `s`; any interior column that `h` alone
   blocked re-opens, which shortens the run. So `h` must be **spare** on the flanks.

Where the naive glue fails is known and cited: the parent-of-parent's glue lemma
(`neighbour_profile.md` 2.5, part (i)) proves that a two-colouring re-phasing leaves the middle
column OPEN for every colouring, so it can only ever produce an adjacent PAIR (`F_2`), never a
single run (`F`). The present construction is the missing move: the middle column is not left to
the colouring, it is closed by a single deliberately re-phased gear, at the price of that gear's
sole-striker columns.

### 0.3 Predictions, each with the number that would refute it

**Disclosure of what had already been read** when these predictions were written: the `r(v)`
profile of `M = {5..29}` (`short_letter_row.md` 2.2) and of `M = {5..31}`
(`r60/results/slr_m31.txt`), so any prediction about those two machines is post-hoc and is
labelled `[read]`. The profiles of `M = {5..11}` ... `{5..23}` had NOT been read and are blind.

- **A1 (instrument).** Rebuilt from scratch: `F = 2, 5, 7, 11, 18, 25, 34, 43` and
  `F_2 = 4, 7, 11, 16, 25, 31, 39, 55` at `M_0..M_7`; `E(a_L) = 2, -, 0, 2, 0, 3, 1, 2` at rungs
  5->7 .. 29->31 and `E(a_L) = 0` at 31->37. Any mismatch is an instrument failure.
- **A2 (the brief's law, literal predicate `U_leg`).** `E(v) <= 3` for every `U_leg`-uncoupled
  size. Pre-registered **REFUTED**, and the refuting instance is visible in the parent's printed
  m29 profile without any computation of this branch: `v = 20` at `M = {5..29}` has
  `3v -+ 1 = 59, 61`, both prime and above 29, so `Leg(20) n M = {}`, while `r(20) = 35` gives
  `E = 55 - 43 = +12` `[read]`. The blind content is the COUNT: predict that at every machine
  m11..m23, `U_leg`-uncoupled sizes with `E > 3` exist, at least 2 per machine from m17 up.
- **A3 (the chain law's predicate `U_full`).** Predict the `U_full`-uncoupled set is nearly EMPTY
  and the law is vacuous on it: at every machine from m19 up, the only `U_full`-uncoupled realised
  size is `v = 1`. REFUTED by any machine m19..m31 with 3 or more `U_full`-uncoupled realised
  sizes `>= 2`. Corollary predicted: `a_L` itself is `U_full`-COUPLED at 6 or more of the 8 rungs,
  so the pinned letter is NOT an instance of "uncoupled sizes are pinned".
- **A4 (the pad predicate `U_pad`, the one the mechanism actually names).** Predict `E(v) <= 3`
  fails for `U_pad`-uncoupled sizes too, but far less often than for coupled ones: predict the
  violation RATE among `U_pad`-uncoupled sizes is below half the rate among `U_pad`-coupled sizes
  at every machine m17..m31. REFUTED if the rate is higher among the uncoupled at any 2 machines.
- **A5 (the single-gear glue, the branch's real object).** `Glue1(run)` = the largest gap
  obtainable at the run's window by re-phasing exactly ONE gear, requiring the middle opening
  blocked; `loss = S - Glue1`. Soundness: `Glue1 <= F(M)` always. Predict:
  - **A5a.** At every machine m11..m23 the attaining 2-run of `v = a_L` admits a single-gear glue
    with `loss <= 3`, which *constructively certifies* `E(a_L) <= 3` there. REFUTED by one machine
    where every occurrence has `loss >= 4`.
  - **A5b.** The closing gear `h` never divides `a` and never divides `v`, 100%.
  - **A5c (tightness).** For a size with `E(v) > 3` the best single-gear glue is exactly `F(M)`,
    i.e. `loss = E(v)`, at 80% or more of the sizes at each machine m17..m23.
  - **A5d (the discriminator).** Among the sizes where the glue reaches `loss <= 3`, the fraction
    that are `U_pad`-uncoupled is at least twice the fraction among the sizes where it does not.
- **A6 (the family; teeth-free or real-teeth).** 20 counterfactual tooth-shifted members (rng seed
  20260906, the same members as r57/r58) at rungs 13->17, 17->19, 19->23. Predict
  `0 <= E(a_L) <= 3` at 17 or more of 20 members at each rung (a glue lemma is teeth-free).
  REFUTED if fewer than 15 of 20 obey at any rung.
- **A7 (the gate band, item 4a).** With `r(a_L) <= F + 3 - a_L` the gate closes at `F + 3 - a_L`.
  Predict the band `[deep-chain cap, F + 3 - a_L]` at rung 29->31 is `[15, 36]` with 22 realised
  sizes (an instrument check against `availability_gate.md` 2.5), and report it at all 8 rungs.
- **A8 (the twin rungs, item 4b).** At 11->13, 17->19, 29->31 the closer `q' - 2` is a gear of
  `M`; measured `E(a_L) = 0, 0, 2`. Predict the glue still succeeds there, and that the closing
  gear is NOT `q' - 2` at 2 or more of the 3.
- **A9 (the record at depth 2, item 4c).** The attainment identity `F(M + q') = max_J Q*_J`
  (file 08) is cited, not re-derived. Predict `F(M + q') = a_L + N(a_L)` (the 3-run form) at 6 or
  more of the 8 rungs, and that the 2-run form `max over letters of (letter + r(letter))` FAILS at
  4 or more rungs (at 29->31 it gives 48 against `F(m31) = 58` `[read]`). Out of sample: predict
  `12 + N(12) = 88 = F({5..37})` at m31.

**Stop rules.** Anything that reduces to the merge/chain law and the letters (file 05), the
attainment identity (08), the peel bound / triple inequality (16), `N(v) <= F_2` and the glue
lemma (2g.i), the pair filter, the closer law and the row profile (4.i.a.i.a), the divisor law
(L20), the gate ladder and the pair cap (4.i.a.i), or the LP windowed dictionary vehicle is
stopped in one line and cited.

### 0.4 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| A1 | instrument: `F`, `F_2`, `E(a_L)` ladders | CONFIRMED exactly, all three, plus `E(a_L) = 0` at 31->37 | 1, 2.1 |
| A2 | the brief's `U_leg` law is refuted, systematically | **the law REFUTED** (14 uncoupled sizes with `E > 3` over 9 machines, worst `v = 20` at m29 with `E = +12`); the COUNT clause refuted (1, 3, 1, 3, 4 from m17 up, not "2 or more everywhere") | 2.2 |
| A3 | `U_full` uncoupled nearly empty; `a_L` mostly coupled | CONFIRMED on both clauses (at most 2 uncoupled sizes `>= 2` at any machine; `a_L` is `U_full`-coupled at 7 of 8 rungs) -- **and the chain-law predicate does not imply the bound either**: `v = 6` at `{5..11}` and `v = 24` at `{5..29}` are uncoupled with `E = +4` | 2.2 |
| A4 | `U_pad` uncoupled violates at less than half the coupled rate | REFUTED at 2 machines (m17: 0.50 vs 0.71; m31: 0.35 vs 0.60), CONFIRMED at 3. The direction is right at 5 of 5 but the factor is not 2 | 2.3 |
| A5a | single-gear glue certifies `E(a_L) <= 3` at m11..m23 | **REFUTED at 2 of 5**: `loss = 0, 2, 0, 5, 7` at m11, m13, m17, m19, m23 | 3.2 |
| A5b | the closing gear never divides `a` or `v` | CONFIRMED, 0 of 90 attaining runs | 3.2 |
| A5c | the glue is tight (`loss = E`) at 80% of sizes | REFUTED: tight at 5/7, 7/10, 7/17, 8/23, 3/33 = 30 of 90 (33%) | 3.2 |
| A5d | the glue's successes are pad-uncoupled twice as often | REFUTED as stated (0.71 against 0.42, a factor 1.7 not 2); the direction is confirmed | 3.2 |
| A6 | family: 17+ of 20 members pinned at each of 3 rungs | **REFUTED, decisively**: 14, 12, 14 of 20; 43 of 63 members over the three rungs, range `-6 .. +7` | 2.5 |
| A7 | the band at 29->31 is `[15, 36]`, 22 sizes | CONFIRMED exactly | 4.1 |
| A8 | twin rungs glue, and not with `q' - 2` | CONFIRMED at the 2 testable twin rungs (depth 1, closing gears 5 and 11, neither is `q'-2 = 11, 17`); the law holds at 4 of 4 twin rungs (`E = +2, 0, 0, +2`) | 4.2 |
| A9 | `F(M+q') = a_L + N(a_L)` at 6+ rungs; 2-run form fails | the 2-run clause CONFIRMED (fails at 8 of 8, by 1 to 10); the 3-run clause REFUTED by one (5 of 8, under at 11->13, 19->23 and, out of sample, at 31->37 where the best depth-2 form is **85 against `F({5..37}) = 88`**) | 4.3 |

---

## 1. Setup (exact ranges)

Everything exact: full periods, integer arithmetic, no sampling except where a sample is named.

| object | range | script |
|---|---|---|
| `r(v)`, `G(v) = v + r(v)`, `E(v) = G(v) - F`, and the three coupling predicates for every realised size | full periods `{5}`, `{5,7}`, ..., `{5..23}` (7,952,175 gaps) and `{5..29}` (214,708,725 gaps, built as 29 copies of the m23 period) | `pl_profile.py` |
| the same at `M = {5..31}` | read from the parent branch's cached exact streamed run `r60/results/slr_m31.json` (period 33,426,748,355), not recomputed | `pl_profile.py` |
| the one-gear glue `Glue1` and the two-gear glue, at every attaining 2-run | `{5..11}` .. `{5..23}`, all occurrences of each attaining pair (4 to 24 per period) | `pl_glue.py`, `pl_letter_glue.py` |
| the exact re-phasing depth (minimum number of gears moved) | the same five machines, targets within 3 columns of the run's ends, depth cap 4 | `pl_depth.py` |
| the spare-gear lemma and its census | the same five machines; plus 13,616 ORDINARY 2-runs sampled with a fixed seed | `pl_spare.py`, `pl_spare_check.py` |
| the tooth-counterfactual family, 20 members + the real machine | rungs 13->17, 17->19, 19->23, full periods each | `pl_family.py` |
| `N(v)` (largest neighbour SUM) and the depth-2 record formulas | `{5}` .. `{5..29}` on full periods; `{5..31}` streamed in 3 processes, 141.3 s | `pl_depth2.py` |
| the cross-rung tables | -- | `pl_summary.py` |

**Instrument gates, all passed.** `F = 2, 5, 7, 11, 18, 25, 34, 43, 58` and
`F_2 = 4, 7, 11, 16, 25, 31, 39, 55, 68` at `M_0..M_8`, the recorded ladders. `E(a_L) = +2, -, 0,
+2, 0, +3, +1, +2, 0` at the nine rungs, reproducing `short_letter_row.md` 2.7 exactly. The
streamed m31 pass returns 6,226,553,025 openings, `F = 58`, `r(12) = 46`. The glue never returns a
gap longer than `F(M)` (a soundness assertion inside `pl_glue.py`, 0 failures). **A1 CONFIRMED.**

**The tool used throughout.** A *configuration* assigns each gear `g` a shift `s_g in Z_g`, its
teeth being `{u_g + s_g, -u_g + s_g}`. The moduli are coprime, so by CRT **every configuration
occurs at some column of the period**; hence any blocked run exhibited by any configuration, with
open ends, is a genuine gap of `M` and is at most `F(M)`. That is the only device this branch uses
to produce lower bounds on `F`, and it is what makes the glue a proof technique rather than an
analogy.

## 2. Results: the generalisation (item 1)

### 2.1 The excess of the letter, and of every size

| rung `q'` | `F` | `F_2` | `a_L` | `r(a_L)` | `E(a_L)` | max `E` over sizes | `#E > 3` | `#sizes` | `a_L` in `U_leg` / `U_full` / `U_pad` |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 2 | 4 | 2 | 2 | **+2** | +2 | 0 | 2 | no / no / yes |
| 11 | 5 | 7 | 4 | unrealised | - | +2 | 0 | 4 | - |
| 13 | 7 | 11 | 4 | 3 | **+0** | +4 | 2 | 7 | no / no / yes |
| 17 | 11 | 16 | 6 | 7 | **+2** | +5 | 4 | 10 | yes / yes / yes |
| 19 | 18 | 25 | 6 | 12 | **+0** | +7 | 10 | 17 | no / no / yes |
| 23 | 25 | 31 | 8 | 20 | **+3** | +6 | 5 | 23 | no / no / yes |
| 29 | 34 | 39 | 10 | 25 | **+1** | +5 | 2 | 33 | yes / no / no |
| 31 | 43 | 55 | 10 | 35 | **+2** | +12 | 23 | 41 | no / no / no |
| 37 | 58 | 68 | 12 | 46 | **+0** | +10 | 28 | 55 | no / no / yes |

The letter's excess stays in `[0, 3]` at 8 of 8 realised rungs while the maximum excess over sizes
grows `2, 2, 4, 5, 7, 6, 5, 12, 10` and the number of sizes breaking the bound grows to 28 of 55.
So the object to explain is real, and it is about the letter and not about the size distribution.

### 2.2 The brief's generalisation is false, under all three readings (item 1)

Pre-registered "yes"; the answer is **NO**, and it fails at the smallest non-trivial machine.

| predicate | uncoupled realised sizes, by machine `{5..11}` .. `{5..31}` | uncoupled sizes with `E > 3` |
|---|---|---|
| `U_leg` (the brief's) | `{1,6}`, `{1,6,10,11}`, `{1,10,14}`, `{1,10,14,20,21}`, `{1,10,14,20,21,34}`, `{1,14,20,21,24,34,36}`, `{1,14,20,24,34,36,46,49,50,55}` | **14 in all**: `6:+4` (m11), `11:+5` (m13), `14:+4` (m17), `10:+6, 20:+5, 21:+6` (m19), `34:+5` (m23), `20:+12, 21:+5, 24:+4` (m29), `14:+4, 20:+5, 34:+7, 55:+7` (m31) |
| `U_full` (the chain law) | `{1,6}`, `{1,6}`, `{1}`, `{1}`, `{1}`, `{1,24,36}`, `{1,24,36}` | **2 in all**: `v = 6` at `{5..11}` (`E = +4`, and `G(6) = 11 = F_2`), `v = 24` at `{5..29}` (`E = +4`) |
| `U_pad` | 5, 6, 10, 11, 14, 17, 20 sizes at the seven machines | 1, 1, 5, 0, 0, 6, 7 = **20 in all** |

Read the three rows in order. The brief's literal predicate is refuted 14 times and the worst
instance is the one the parent had already printed: at `M = {5..29}`, `v = 20` has
`3v - 1 = 59` and `3v + 1 = 61`, both prime and above the machine, so no gear of `M` can close a
20-gap by the leg condition -- and yet `r(20) = 35` and `G(20) = 55 = F_2`, twelve above the
record. **The uncoupled sizes are not pinned; the pad is doing the work the brief attributed to
the leg** (5 divides 20, and `5 in M`). The chain law's own predicate, which includes the pad, is
much better -- 2 exceptions in 15 uncoupled instances -- but it is *nearly vacuous*: from
`M = {5..17}` on, the only uncoupled sizes are `1`, `24` and `36`, because a size below `F` almost
always has a prime factor `<= y` or a member of column `v/2` with one. And the two exceptions kill
it as a law: `v = 6` at `{5..11}` (`Coup(6) = {17, 19}`, neither in `M`) has
`G(6) = 11 = F_2 = F + 4`.

**And the letter itself is not an uncoupled size.** `a_L` is `U_full`-uncoupled at exactly 1 of
the 8 realised rungs (13->17, `a_L = 6`, `Coup(6) = {17, 19}`), `U_leg`-uncoupled at 2 of 8, and
`U_pad`-uncoupled at 6 of 8. So even if "uncoupled implies pinned" were true, **it would not
contain the pinned letter**: at the two rungs that carry the budget's tightness (23->29 and
29->31, where `5 | a_L = 10`) the letter is coupled by gear 5 in the pad, exactly like the sizes
`20, 25, 30, 35` that break the bound by up to 12.

> **The item-1 verdict, in one line.** The pinned letter is NOT the instance at `v = a_L` of a law
> about uncoupled sizes. There is no divisibility predicate on `v` on file that separates the
> pinned sizes from the rest: the best of the three (`U_full`) covers 13 of the 186 realised sizes
> of the seven machines (7%) and is wrong at 2 of those 13.

### 2.3 Which sizes are pinned, then

The rates behind A4, per machine (violations `E > 3` among `U_pad`-uncoupled against coupled):

| machine | `{5..11}` | `{5..13}` | `{5..17}` | `{5..19}` | `{5..23}` | `{5..29}` | `{5..31}` |
|---|---|---|---|---|---|---|---|
| uncoupled | 1/5 = .20 | 1/6 = .17 | 5/10 = .50 | 0/11 = .00 | 0/14 = .00 | 6/17 = .35 | 7/20 = .35 |
| coupled | 1/2 = .50 | 3/4 = .75 | 5/7 = .71 | 5/12 = .42 | 2/19 = .11 | 17/24 = .71 | 21/35 = .60 |

The direction is right at 7 of 7 machines -- a size no gear of `M` divides is safer -- but the
factor runs 1.4, 2.0, 2.5, 4.4 and twice infinite, and the pre-registered factor 2 fails twice. A pad gear is a *tendency*, not
a gate. The mechanism section says why: dividing `v` disqualifies one gear from closing the middle
opening, and the machine has six or seven others.

### 2.4 What the record-attaining pairs all have in common (new, exceptionless)

At the attaining 2-run of EVERY realised size at every machine `{5..11}` .. `{5..23}` -- 90 sizes,
each checked at up to 16 occurrences --

> **every gear of `M` is a sole striker of some column strictly inside the run.**

90 of 90, with no exception, and with 0 sizes having a "free" gear in the sense of 3.1. This is
the L4 property of the record (`tiling`, node 5: every gear a sole coverer somewhere) extended
from the record gap to the extremal 2-run of every size. It is the reason the glue is hard: there
is no idle gear to spend on the middle opening.

### 2.5 The family: the law is real-teeth, not teeth-free (item 3)

20 counterfactual members per rung (teeth at `+-v_g`, `v_g` uniform in `1..(g-1)/2`, the same
members as r57/r58), each on its full period, with the member's own letters `a_L = 2 v_{q'}`:

| rung | `0 <= E(a_L) <= 3` | `E(a_L) <= 3` | `E(a_L) >= 0` | REAL | family range |
|---|---|---|---|---|---|
| 13->17 | 14 of 20 | 17 of 20 | 17 of 20 | +2 | `-6 .. +7` |
| 17->19 | 12 of 20 | 15 of 20 | 17 of 20 | +0 | `-2 .. +5` |
| 19->23 | 14 of 20 | 16 of 20 | 18 of 20 | +3 | `-3 .. +7` |
| all 63 members | **43 of 63** | 51 of 63 (81%) | 55 of 63 (87%) | 3 of 3 | `-6 .. +7` |

**A6 REFUTED, and this is the branch's most consequential negative.** A glue lemma is a statement
about periodic 2-tooth sets and must therefore hold for every tooth assignment; the pinned letter
does not. 12 of 63 members break the upper half (up to `E = +7`, member m3 at 13->17 with
`a_L = 7`, `r = 15`, `F = 15`), and 8 break the lower half (down to `E = -6`). So **no proof of
the pinned letter can use only the combinatorics of two-tooth gears; it must use
`u_g = 6^{-1} mod g`.** The one visible arithmetic handle: with gear 5 at its real tooth
(`v_5 = 1`) the upper half holds at 35 of 39 members (90%), and with `v_5 = 2` at 16 of 24 (67%).
That is the same "nearly structural, not family-combinatorial" signature `neighbour_profile.md`
2.4 records for the `F_2` form, and the same which-residues signal `availability_gate.md` G7
records for the gate.

## 3. The proof attempt (item 2)

### 3.1 What IS proved: the spare-gear lemma

> **THE SPARE-GEAR LEMMA (proved here; new).** Let `x_0 < x_1 < x_2` be three consecutive openings
> of `M`, `a = x_1 - x_0`, `v = x_2 - x_1`, `S = a + v`. Call a gear `h`
>
>   * **obstructed** at the run if `h | a`, or `h | v`, or
>     (`a = +d_h` or `v = -d_h`) and (`a = -d_h` or `v = +d_h`) mod `h`;
>   * **busy** in the run if some column strictly between `x_0` and `x_2` is struck by `h` alone.
>
> If some gear is neither obstructed nor busy, then `F(M) >= S`, i.e. `E(v) <= 0` at that run.

*Proof.* Let `h` be neither obstructed nor busy. Move `h`'s teeth to `{x_1, x_1 - d_h}` or to
`{x_1, x_1 + d_h}`; both place a tooth on `x_1`. The first fails to keep an end open only if
`a = 0` or `a = d_h` (then it strikes `x_0`) or `v = 0` or `v = -d_h` (then `x_2`); the second
only if `a = 0`, `a = -d_h`, `v = 0` or `v = d_h`. "Not obstructed" is exactly the statement that
at least one of the two choices keeps both ends free. Take that one. Every other gear keeps its
phase, so it still misses `x_0` and `x_2` (they are openings of `M`) and still blocks every column
it blocked. Every interior column other than `x_1` was blocked by some gear; if its only blocker
was `h` the run would be busy, which it is not; so it is still blocked. And `x_1` is now blocked by
`h`. So `(x_0, x_2)` is a gap of the new configuration, and by CRT that configuration occurs in the
period, hence `F(M) >= S`. `[]`

**Verified, and it bites.** On 13,616 ordinary 2-runs sampled with a fixed seed across the five
machines: 0 counterexamples; the largest span carrying a free gear is exactly `F` at
`{5..11}`, `{5..13}` and `{5..17}` (7, 11, 18) and below `F` at the other two. Every run of span
`> F` had no free gear, 133 of 133 such runs.

**Its contrapositive is the meaning of the excess.** `E(v) > 0` -- the pair beating the record --
is *equivalent to no free gear at any occurrence*: every gear is either obstructed at the middle
opening (an arithmetic condition on `a`, `v` mod `h`) or is carrying a column of the run alone.
Section 2.4 measures the second half at the attaining runs: it is always the case, at every gear,
at every size. **So the excess is not slack; it is the statement that the run uses the whole
machine.**

### 3.2 What is NOT proved: the one-gear glue does not reach the constant 3

The lemma gives `E <= 0` when a gear is free, and 2.4 says no gear is ever free at an attaining
run. The next construction is to move a busy gear anyway and pay for its sole columns. Measured on
every occurrence of every attaining pair (`Glue1` = the best gap containing the closed middle
opening over all one-gear re-phasings):

| machine | `a_L` | `S` | `E(a_L)` | best `Glue1` | `loss = S - Glue1` | closing gear | certificate |
|---|---|---|---|---|---|---|---|
| `{5..11}` | 4 | 7 | +0 | 7 = `F` | **0** | 5 | `E(a_L) <= 0` |
| `{5..13}` | 6 | 13 | +2 | 11 = `F` | **2** | 13 | `E(a_L) <= 2` |
| `{5..17}` | 6 | 18 | +0 | 18 = `F` | **0** | 11 | `E(a_L) <= 0` |
| `{5..19}` | 8 | 28 | +3 | 23 < `F = 25` | **5** | 17 | `E(a_L) <= 5` -- too weak |
| `{5..23}` | 10 | 35 | +1 | 28 < `F = 34` | **7** | 13 | `E(a_L) <= 7` -- too weak |

**A5a REFUTED.** The one-gear glue proves the pinned bound at the three small machines and stops
working exactly where the machine gets big enough for the statement to have content. Over all 90
attaining runs it reaches `S - 3` at 28 and reaches `F` exactly at 30. Two facts about it survive:
the closing gear divides neither `a` nor `v` at 0 of 90 runs (**A5b confirmed**, and it is forced
-- a pad gear is obstructed by the lemma's own condition), and the runs it can glue are
`U_pad`-uncoupled at 0.71 against 0.42 for the runs it cannot (**A5d's direction confirmed,
its factor refuted**).

### 3.3 The exact obstruction, and how far the record really is from the pair

Deepening the construction: `depth` = the minimum number of gears whose phase must be moved,
starting from an occurrence of the attaining pair, to produce a gap of length `>= S - 3` covering
the middle opening (targets allowed to slide up to 3 columns beyond either end; cap 4 gears). This
is an exact weighted set cover, solved by branch and bound.

| machine | depth 1 | 2 | 3 | 4 | no glue within 4 | sizes with `E > 3` | **depth at `a_L`** |
|---|---|---|---|---|---|---|---|
| `{5..11}` | 3 | 2 | 0 | 0 | 2 | 2 | **1** (gear 5) |
| `{5..13}` | 3 | 3 | 0 | 0 | 4 | 4 | **1** (gear 13) |
| `{5..17}` | 5 | 1 | 1 | 0 | 10 | 10 | **1** (gear 11) |
| `{5..19}` | 11 | 4 | 1 | 1 | 6 | 5 | **2** (gears 11, 17) |
| `{5..23}` | 7 | 10 | 8 | 4 | 4 | 2 | **4** (gears 17, 7, 23, 19) |

Two readings, and they are the branch's answer to item 2.

1. **The construction is complete as a certificate.** Every size with `E <= 3` is glued within
   depth 4 except three (`v = 3` at `{5..19}`, `v = 3` and `v = 28` at `{5..23}`): 64 of 67. And
   every size with `E > 3` fails, necessarily -- a glue of length `S - 3 > F` cannot exist. So
   "bounded re-phasing of the attaining pair" and "the pinned bound" agree at 87 of 90 sizes. The
   glue is the right object.
2. **The depth is not bounded.** At the letter it runs `1, 1, 1, 2, 4` over five machines. A
   depth-`k` glue costs `k` sole-striker repairs, and the number of gears that must be repaired
   grows with the machine because (2.4) every gear is busy. There is no constant-depth lemma in
   sight, and a depth that grows with `|M|` is not a proof of a statement about all `M`.

**Where exactly the parent's glue lemma is superseded, and where it is not.** The parent's lemma
(`neighbour_profile.md` 2.5 (i)) proves that a two-colouring re-phasing of a 3-run leaves the
middle column open, so that construction can only ever produce an adjacent pair and the `F_2`
bound. This branch's construction closes the middle deliberately with one named gear, which
removes that obstruction -- and hits a second one in its place: **the gear that closes the middle
is the gear that was holding a column of the flank**. In the parent's language: the middle column
of a 2-run is not left open by the colouring, it is *bought*, and the price is exactly the excess
`E`. That is why `E > 0` at 5 of the 8 rungs rather than `E = 0` everywhere, and it is the mechanism the parent
branch asked for. What is *not* explained is why the price is never more than 3.

**And the family says no proof of this shape can exist.** The whole construction -- CRT, teeth as
translates, sole strikers, obstruction mod `h` -- is invariant under changing the teeth. It
therefore proves, at most, statements true of every family member; and 12 of 63 members have
`E(a_L) > 3`. So the constant 3 is **not** available from this construction, at any depth. Any
proof of the pinned letter must use the real teeth, i.e. `u_g = 6^{-1} mod g`, `3 a_L = q' -+ 1`.

## 4. Consequences, toward the root (item 4)

### 4.1 The gate closes at `F + 3 - a_L`, and the band (item 4a)

With `r(a_L) <= F + 3 - a_L` and the parent's identity `a_hasM = r(a_L)` (7 of 8 rungs), the
availability gate closes at `F + 3 - a_L`, about `F - q'/3` since `3 a_L = q' -+ 1`:

| rung `q'` | `F` | `a_L` | pinned top `F + 3 - a_L` | measured `a_hasM` | `F - q'/3` | deep-chain cap | residual band | realised sizes in it |
|---|---|---|---|---|---|---|---|---|
| 7 | 2 | 2 | 3 | 2 | -0.3 | 3 | empty | 0 |
| 11 | 5 | 4 | 4 | 0 | 1.3 | 8 | empty | 0 |
| 13 | 7 | 4 | 6 | 0 | 2.7 | 6 | empty | 0 |
| 17 | 11 | 6 | 8 | 7 | 5.3 | 9 | empty | 0 |
| 19 | 18 | 6 | 15 | 12 | 11.7 | 12 | `[13, 15]` | 3 |
| 23 | 25 | 8 | 20 | 20 | 17.3 | 12 | `[13, 20]` | 7 |
| 29 | 34 | 10 | 27 | 25 | 24.3 | 21 | `[22, 27]` | 5 |
| **31** | 43 | 10 | **36** | 35 | 32.7 | 14 | **`[15, 36]`** | **22** |

(the deep-chain cap is `availability_gate.md` 2.5's, `floor((F + q')/J_max)` at the measured
`J_max`, cited). **A7 CONFIRMED**: at the rung that matters the pinned letter closes the gate at
36, one above the certified truth 35, five below the vacuous proved cap `F_2 - a_L = 45`; above
36 only the pair statement remains (slack 19), and the band `[15, 36]` with 22 realised sizes is
what the chain statement must cover. The band grows with the rung -- 3, 7, 5, 22 sizes -- so the
pinned letter buys the top of the spectrum and nothing about the band.

### 4.2 The twin rungs (item 4b)

| rung `q'` | `q' - 2` a gear of `M`? | `a_L` | `Leg(a_L) n M` | `Pad(a_L) n M` | `E(a_L)` | glue depth | gears moved |
|---|---|---|---|---|---|---|---|
| 7 | TWIN (5) | 2 | 5 | - | +2 | - | - |
| 13 | TWIN (11) | 4 | 11 | - | **+0** | 1 | 5 |
| 17 | no | 6 | - | - | +2 | 1 | 13 |
| 19 | TWIN (17) | 6 | 17 | - | **+0** | 1 | 11 |
| 23 | no | 8 | 5 | - | +3 | 2 | 11, 17 |
| 29 | no | 10 | - | 5 | +1 | 4 | 17, 7, 23, 19 |
| 31 | TWIN (29) | 10 | 29 | 5 | **+2** | - | - |
| 37 | no | 12 | 5, 7 | - | **+0** | - | - |

The law holds at 4 of 4 twin rungs, and the excess there (`+2, 0, 0, +2`) is if anything smaller
than at the non-twin rungs (`+2, +3, +1, 0`). **What replaces the disjointness is that the coupled
gear is not the gear the construction uses**: at the two twin rungs inside the glue's range the
closing gear is 5 and 11, while the coupling gear is `q' - 2 = 11` and `17`. This is the same
reading the parent reached from the other side (`short_letter_row.md` 2.5): the twin structure
decides the CLOSERS of an `a_L`-gap and has no effect on the row. The spare-gear lemma explains
why in one line: what disqualifies a gear from closing the middle is `h | a`, `h | v` or the
`+-d_h` pattern -- a condition on `h` against the two gaps, not on `Leg(a_L)`, which is a
condition on gears *outside* `M`.

### 4.3 The record of `M + q'` from `M`'s dictionary at depth 2 (item 4c)

| rung | `M` | `F(M)` | `F(M + q')` | 2-run form `max_l (l + r(l))` | 3-run form `max_l (l + N(l))` | equal? |
|---|---|---|---|---|---|---|
| 5->7 | `{5}` | 2 | 5 | 4 | 5 | YES |
| 7->11 | `{5,7}` | 5 | 7 | 0 (letters unrealised) | 0 | under |
| 11->13 | `{5..11}` | 7 | 11 | 7 | 8 | under by 3 |
| 13->17 | `{5..13}` | 11 | 18 | 16 | 18 | YES |
| 17->19 | `{5..17}` | 18 | 25 | 20 | 25 | YES |
| 19->23 | `{5..19}` | 25 | 34 | 28 | 33 | under by 1 |
| 23->29 | `{5..23}` | 34 | 43 | 37 | 43 | YES |
| 29->31 | `{5..29}` | 43 | 58 | 48 | 58 | YES |
| **31->37** | `{5..31}` | 58 | **88** (recorded) | 67 | **85** | **under by 3** |

`l` ranges over `{a_L, b_L, q'}`; `N` is the largest neighbour SUM (`neighbour_profile.md`, the
`J = 2` fusion of the merge law, cited). The 2-run form -- the object the pinned letter is about --
**never** reaches the record: it is short by 1, 4, 2, 5, 6, 6, 10, 21 at the eight rungs where the
letters are realised. So the pinned letter is a statement about one term of the record's formula
and not about the record. The 3-run form reaches it at 5 of 8 rungs (**A9's 3-run clause REFUTED
by one**; it is under at 11->13 and 19->23), and the out-of-sample rung is a clean failure: at
`M = {5..31}` the full-period streamed pass gives `N(12) = 56`, `N(25) = 45`, `N(37) = 48`, so the
best depth-2 value is `37 + 48 = 85`, three short of the recorded `F({5..37}) = 88`, whose
attaining word `(28, 37, 12, 11)` is a `J = 4` fusion (`neighbour_profile.md` 7, cited).
**The prediction `12 + N(12) = 88` is REFUTED: `12 + 56 = 68`.**

For rung 37->41 the same computation needs the period of `{5..37}` (1.24e12 columns) and is out of
reach in this lane, so the brief's "91?" is not tested here. What the pinned letter does predict
there is one row of that computation: `a_L(41) = 14` (`3 * 14 = 42 = q' + 1`), so
`r(14) <= F({5..37}) + 3 - 14 = 77` and the 2-run term is at most 91 -- a bound on a term, not on
the record, and the table above says the record will exceed it (at 31->37 the record is 21 above
the 2-run term).

## 5. What is new

1. **The item-1 answer, negative and complete.** "Uncoupled sizes are pinned to `F`" is false
   under all three readings of uncoupled, with 14, 2 and 20 violating sizes; the worst is
   `v = 20` at `{5..29}` (`E = +12`, `G = F_2`), and the smallest is `v = 6` at `{5..11}`
   (`E = +4`), which is uncoupled even for the chain law's own predicate. And the letter is not an
   uncoupled size: it is `U_full`-coupled at 7 of 8 rungs.
2. **THE SPARE-GEAR LEMMA**, proved and verified (0 counterexamples in 13,616 runs): a 2-run with
   a gear that is neither obstructed at the middle opening nor a sole striker inside the run has
   span at most `F(M)`. The obstruction condition `h | a`, `h | v`, or the `+-d_h` pattern, is the
   chain law's condition read at one opening instead of two.
3. **The meaning of the excess.** `E(v) > 0` is *equivalent* to "no free gear at any occurrence",
   so the amount by which a 2-run beats the record is the price of buying its middle opening from
   a gear that was already carrying a column. This is the mechanism the parent branch asked for,
   and it explains the sign of `E`, not its bound.
4. **Every gear is a sole striker inside the attaining 2-run of every size**, 90 of 90 over five
   machines -- the L4 tiling property of the record extended to the whole `r(v)` profile.
5. **The re-phasing depth** of the attaining pair: `1, 1, 1, 2, 4` at the letter over
   `{5..11}` .. `{5..23}`, and a full depth census (29 sizes at depth 1, 20 at 2, 10 at 3, 5 at 4,
   3 unreachable within 4 among the 67 sizes with `E <= 3`). The glue and the pinned bound agree at
   87 of 90 sizes; the depth is not bounded.
6. **The family verdict: the pinned letter is a real-teeth law.** 43 of 63 members obey
   `0 <= E(a_L) <= 3`, 51 of 63 obey the upper half, range `-6 .. +7`. Since every step of the
   glue construction is tooth-invariant, **no argument of that shape can prove the constant 3**.
   With gear 5 at its real tooth the upper half rises to 90%.
7. **The gate's top under the pinned letter at every rung** (`F + 3 - a_L` = 3, 4, 6, 8, 15, 20,
   27, 36) with the residual bands and their realised sizes (0, 0, 0, 0, 3, 7, 5, 22).
8. **The depth-2 record formula, settled.** The 2-run form never reaches `F(M + q')` (short by 1 to 21 at 8 of 8 rungs); the 3-run form reaches it at 5 of 8 and falls 3 short at the ninth,
   where `N(12) = 56`, `N(25) = 45`, `N(37) = 48` at `M = {5..31}` are computed here for the first
   time on the full 33,426,748,355-column period.

Prior art inside the project: the glue lemma and `N(v) <= F_2` are 2g.i (`neighbour_profile.md`
2.5), and part (i) of it is what this branch's construction is built to evade; the chain and merge
laws and the letters are file 05; the attainment identity is file 08; the row `r(v)`, the pair
filter, the closer law and the `E(a_L)` ladder are the parent (`short_letter_row.md`); the gate
ladder, the band and the counterfactual family are 4.i.a.i (`availability_gate.md`); the divisor
form of `Leg` is `half_column.md` / separability X8; L4 (every gear a sole coverer in the record)
is node 5. The spare-gear lemma, the excess-as-price reading, the depth census and the family
verdict on the pinned letter are not on record. Outside the project: not checked (no web access).

## 6. Verdict

**The pinned letter is NOT a law about uncoupled sizes, its glue mechanism is proved only in the
`E <= 0` case, and the family says the constant 3 cannot come from a glue argument at all. Node
status: the law survives as a measured CANDIDATE (now 8 of 8 real rungs and 43 of 63 family
members), the route by glue is DEAD, and what the branch adds is one proved lemma and a sharp
statement of what a proof must use.**

- **Item 1 (generalise before proving): refuted.** `v + r(v) <= F + 3` fails for uncoupled sizes
  under the brief's predicate (14 instances), under the chain law's predicate (2 instances,
  including `v = 6` at the smallest interesting machine), and under the pad predicate (20). No
  divisibility predicate on file separates the pinned sizes. The letter is coupled at 7 of 8
  rungs, so the pinned letter is not the `v = a_L` instance of any of them.
- **Item 2 (the mechanism): half proved.** The spare-gear lemma is exact and new, and its
  contrapositive identifies the excess as the price of closing the middle opening. The exact
  obstruction to pushing it to `E <= 3` is stated and measured: at an attaining 2-run every gear
  is busy (90 of 90), so the closing gear must be bought from the flanks, and the number of gears
  that must then be repaired grows with the machine (depth `1, 1, 1, 2, 4` at the letter).
- **Item 3 (the constant): real-teeth.** 12 of 63 family members exceed `E = 3`, up to `+7`. The
  constant is not teeth-free, so it is not a glue constant. Every step of the construction is
  tooth-invariant, so the construction cannot produce it. A proof must use `3 a_L = q' -+ 1`.
- **Item 4 (consequences): delivered.** The gate closes at `F + 3 - a_L` (36 at rung 29->31,
  against the certified 35 and the vacuous 45), leaving the band `[15, 36]` with 22 realised sizes
  for the chain statement and the pair statement above it with slack 19. The twin rungs obey the
  law with the smallest excesses on record, and the closing gear is never the twin. The record of
  `M + q'` is not a depth-2 function of `M`'s dictionary: the 2-run form is short at 8 of 8 rungs
  and the 3-run form falls 3 short at 31->37 (85 against 88).

**CANDIDATE, unchanged in status and better understood.** `F(M) <= a_L + r(a_L) <= F(M) + 3`.
What would have to break it: a machine and an incoming gear with an `a_L`-gap adjacent to a gap of
size more than `F + 3 - a_L`. Why the system may not be able to do that: **still not shown, and
now known not to be shown by any tooth-invariant argument.** The next child branch is therefore
not another covering construction but the arithmetic one: `c_5(a_L)` is decided by `q' mod 5` and
`q' mod 3` (parent, 9 of 9), gear 5 at its real tooth lifts the family rate from 67% to 90%, and
the letters of the real machine are always even -- the place to look is what
`3 a_L = q' -+ 1` does to the covering capacity, not what two-tooth periodicity does to it.

## 7. Dead ends, each with its refuting instance

- **"Uncoupled sizes are pinned to `F`" (the brief's item-1 hypothesis).** Refuted at
  `v = 20`, `M = {5..29}`: `Leg(20) = {59, 61}`, disjoint from `M`, `G(20) = 55 = F_2 = F + 12`.
- **The same with the chain law's full predicate.** Refuted at `v = 6`, `M = {5..11}`:
  `Coup(6) = {17, 19}`, disjoint from `M`, `G(6) = 11 = F + 4`.
- **The same with the pad predicate.** Refuted 20 times; worst `v = 37` at `{5..31}`, `E = +9`.
- **The one-gear glue as a proof of `E(a_L) <= 3`.** Refuted at `M = {5..19}` (best `loss = 5`)
  and `{5..23}` (best `loss = 7`), over every occurrence of the attaining pair.
- **A bounded-depth glue.** Refuted by the depth ladder at the letter: `1, 1, 1, 2, 4`.
- **Any tooth-invariant proof of the constant 3.** Refuted by the family: 12 of 63 members have
  `E(a_L) > 3`, up to `+7` (member m3 at rung 13->17, teeth `[1,1,3,6,5]`, `a_L = 7`, `r = 15`,
  `F = 15`).
- **The glue lemma's own two-colouring (cited, closed by the parent-of-parent).** Its middle
  column is provably open, so it produces `F_2` and never `F`; this branch's construction is the
  repair, and it runs into the busy-gear obstruction instead.
- **`F(M + q') = a_L + r(a_L)` (a depth-2 record formula from the 2-run dictionary).** Refuted at
  8 of 8 rungs where the letters are realised, short by 1, 4, 2, 5, 6, 6, 10, 21.
- **`F(M + q') = max_l (l + N(l))` (the depth-2 record formula from the 3-run dictionary).**
  Refuted at 11->13 (8 against 11), 19->23 (33 against 34) and, out of sample, 31->37 (85 against
  88).

## 8. What holds without exception (item 5)

| statement | count | status |
|---|---|---|
| the spare-gear lemma: a 2-run with a gear neither obstructed nor busy has span `<= F(M)` | proved; 0 counterexamples in 13,616 ordinary runs over 5 machines, and 133 of 133 runs of span `> F` have no free gear | **proved here** + verified |
| at the attaining 2-run of every realised size, EVERY gear of `M` is a sole striker of some interior column | 90 of 90 sizes, 5 machines, up to 16 occurrences each | measured, exceptionless |
| no free gear at any attaining 2-run of any size | 90 of 90 | measured, exceptionless |
| the closing gear of a successful one-gear glue divides neither `a` nor `v` | 90 of 90 | forced by the lemma + verified |
| `0 <= E(a_L) <= 3` (the pinned letter) | 8 of 8 real rungs; 43 of 63 family members | measured; **not** teeth-free |
| `E(a_L) <= 3` at every twin rung (`q' - 2` a gear of `M`) | 4 of 4 | measured |
| the 2-run form `max_l (l + r(l))` is strictly below `F(M + q')` | 8 of 8 rungs with realised letters | measured |
| every size with `E(v) <= 3` is glued by a re-phasing of depth `<= 4`, and every size with `E(v) > 3` by none | 64 of 67 and 23 of 23, 5 machines | measured (3 exceptions: `v = 3` at `{5..19}`, `v = 3, 28` at `{5..23}`) |
| `Glue1 <= F(M)` (soundness of the CRT realisation) | every run computed, 0 failures | proved (CRT) + asserted in code |
