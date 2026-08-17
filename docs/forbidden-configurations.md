# Forbidden configurations, and the step form of the remaining gap

Companion to `covering-bound-route.md`. Scripts: `research/minimal_forbidden.py`,
`research/covering_decomposition.py`, `research/gap_automaton.py`.

Working frame throughout is the halved one: each odd prime `q <= y` blocks the adjacent pair
`{o, o+1} mod q`, where `o` is its offset. For a set `S` of positions write

    W_q(S) = { s-1, s : s in S }  mod q.

Gear `q` is **forced** to block one of `S` exactly when `W_q(S) = Z_q`, since then no offset avoids
all of `S`. If `|W_q(S)| < q` there is an offset leaving every position of `S` exposed at once. This is
the exposure criterion, and everything below is a consequence of it.

## 1. The minimal size law

Each position contributes a domino `{s-1, s}` of at most 2 residues, so covering `Z_q` needs at least
`ceil(q/2) = (q+1)/2` positions for odd `q`. That many suffice: take `s` at residues
`0, 2, 4, ..., q-1`, whose dominoes are `{q-1,0}, {1,2}, {3,4}, ..., {q-2,q-1}` and cover everything
with a single overlap. Since `3` is invertible mod `q`, integer positions with those residues and all
`= 0 mod 3` exist by CRT.

> **Gear `q` can be forced to block one of `S` only if `|S| >= (q+1)/2`, and that bound is attained.**

Read as exposure, which is the direction that matters: **any `(q-1)/2` positions can be simultaneously
exposed to gear `q`**, whatever their spacing. A gear only starts constraining once the configuration
is half its circumference.

Verified exhaustively for `q = 3` to `19`, and the explicit construction checked for all 45 odd primes
below 200 - zero failures.

    q      3   5   7  11  13  17  19
    min|S| 2   3   4   6   7   9  10
    (q+1)/2 2  3   4   6   7   9  10

This subsumes the two known laws as its first two cases: `q = 3` gives `|S| >= 2`, which is "gear 3
blocks one of any two adjacent positions"; `q = 5` gives `|S| >= 3`, the three-positions-spaced-3 law.

## 2. Minimal span grows linearly

Restricting positions to multiples of 3 - which is what the pattern actually offers, by the gear-3 law
- and minimising the span rather than the count:

    q        5    7   11   13   17   19   23   29   31
    span     6   12   18   24   30   36   42   54   60
    span/q 1.20 1.71 1.64 1.85 1.76 1.89 1.83 1.86 1.94

Computed by a bitmask dynamic programme over covered residues (`min_span`), so these are exact minima,
not search artefacts. The span grows like `1.9 q`, and the number of positions like `q/2`.

Gear 3 has **no** forbidden configuration inside a single class mod 3, which is right: gear 3 is what
forces the pattern into one class mod 3 in the first place, and inside that class it forces nothing
further.

**Correction to `ideas-from-the-session.md`.** That note recorded "`q >= 13`: none within span 27".
That is false - gear 13 has a forbidden configuration of span 24, inside 27. The earlier search had
been enumerating subsets in an order that missed it. The corrected table is the one above.

**Second correction.** The same note recorded gear 5 as forbidding the single gap-word factor `11` and
gear 7 as forbidding `121`. Both are genuinely forbidden, but neither list is complete. The minimal
forbidden words of each gear alone are

    q = 5   10 words: 11, 13, 16, 24, 31, 42, 61, 121, 151, 222
    q = 7   17 words: 121, 131, 213, 312, 314, 333, 413, 1111, 1113, 1233,
                      2112, 2114, 3111, 3113, 3321, 4112, 12321
    q = 11  170 words, lengths 5 to 9

## 3. The factorisation law

Let `w(S) = |{s-1, s : s in S}|` counted over the **integers**, so `w(S) = 2|S| - adj(S)` where
`adj(S)` counts `s in S` with `s+1 in S`. Then

> **`|W_q(S)| = w(S)` for every gear `q > span(S) + 1`.**

The threshold is `span + 1`, not `span`: the extreme members of `{s-1, s}` are `min(S) - 1` and
`max(S)`, which differ by `span(S) + 1` and so collide mod `q` exactly at `q = span(S) + 1`. Getting
this wrong is easy and was caught by the check rather than by inspection - for `S = {0, 12}` and
`q = 13`, `W_13 = {0, 11, 12}` has size 3 while `2|S| = 4`. Verified with zero exceptions for
`L = 9, 12, 15, 18` against gear sets to 31 once the threshold was corrected.

Consequence: in any product `prod_q (q - |W_q(S)|)`, the gears above `span(S) + 1` contribute
`prod (q - w(S))`, which sees only the size and adjacency of `S`, never its placement. All placement
dependence lives in the gears at or below `span(S) + 1`.

## 4. The decomposition of the covering counts

`N(L)`, the number of shifts at which `L` consecutive positions are all blocked, equals
`sum_g max(0, g - L)` over gaps `g`, and by inclusion-exclusion over exposed subsets of `[0, L)`

    N(L) = sum_{T subset of [0,L)} (-1)^{|T|} prod_q ( q - |W_q(T)| ),

with the empty set contributing `P`. Applying section 3, every gear `q > L` sees only `w(T)`, so

    N(L) = sum_j c_j(L) * prod_{L < q <= y} (q - j),
    c_j(L) = sum_{T : w(T) = j} (-1)^{|T|} prod_{q <= L} ( q - |W_q(T)| ).

**The `c_j(L)` do not depend on `y`.** Verified by computing them once and reassembling `N(L)` at
`y = 13, 19, 23, 31`, matching direct inclusion-exclusion in every case:

    L      shape gears        c_j
    2      -                  c_0=1, c_2=-2, c_3=1
    3      3                  c_0=3, c_2=-3
    4      3                  c_0=3, c_2=-4, c_4=1
    6      3,5                c_0=15, c_2=-18, c_4=3
    9      3,5,7              c_0=105, c_2=-135, c_4=42
    12     3,5,7,11           c_0=1155, c_2=-1620, c_4=651, c_6=-60

with `c_0(L) = prod_{q <= L} q` throughout, since `T` empty contributes exactly that.

This explains why the block starts `L = 3, 6, 9` came out in closed form (sections 20b, 20c, 24c of
`covering-bound-route.md`): the shape-carrying gears there are just `{3}`, `{3,5}`, `{3,5,7}` and
everything else is a clean product. It also marks the method's limit - once `L > y` there is no tail
left and every gear carries shape, and the `L` that matter run up to `F_h ~ 0.055 y^2`, far past `y`.

## 5. The automaton question, settled two ways

Admissibility is **factor-closed**: if `S' ⊆ S` then `W_q(S') ⊆ W_q(S)`, so a configuration that fails
to force has no sub-configuration that forces. Hence every factor of an admissible gap word is
admissible, and a word is *minimally* forbidden exactly when it is forbidden while both of its
length-`(n-1)` factors are admissible. That makes a level-by-level search possible: extend admissible
words by one letter, keep the admissible ones (`research/gap_automaton.py`).

**Result 1 - the forbidden set saturates in the gear direction.** Counting minimal forbidden words
inside a box of length 16 and letters up to 6:

    gears to    19      23      29      31      37      41      47
    minimal  25060   25270   25270   25270   25270   25270   25270
    new          -     260       0       0       0       0       0

Zero new words from gear 29 onward, reproduced at a smaller box (length 12, letters 5), where the count
settles at 1908 from `y = 23`. This is not a cap artefact for gears 29 and 31: their own minimal
configurations need `(q-1)/2 = 14` and `15` letters, both inside the box, and they still add nothing -
every configuration those gears could force already contains a shorter word forbidden by a smaller
gear. Gears from 37 up need 18+ letters and are untested at this width.

> **Large gears force nothing new.** Gear `q` needs `(q+1)/2` exposed positions spread over a span of
> about `1.9 q` before it can force anything, and a stretch of exposed positions that long has already
> violated a small gear.

**Result 2 - the antidictionary is not finite.** At both box widths the longest minimal forbidden word
equals the box length and the per-length count is still rising at the boundary (273 new at length 12,
and 25270 total at length 16 with no sign of decay). So minimal forbidden words keep appearing at every
length, and idea 1's route through a finite antidictionary is not available.

**Why this closes idea 1 regardless.** Even granting an automaton - and one does exist, with state
"accumulated `W_q` per gear plus phase", though its size is `prod q 2^q` - its letter statistics are
the frequencies of letters across *admissible words*, a set whose size does not depend on `y`. The
`n_j` are counts of shifts of one specific word, the pattern's own cyclic gap sequence, and they scale
with `P`. Recovering `n_j` needs the automaton *weighted* by the CRT measure, and those weights are the
`n_j`. So the transfer-matrix route is circular in the same way section 22b was.

Measured, to leave no doubt - letter frequencies of the pattern's own gap word against unweighted
letter frequencies over admissible words:

    letter (gap/3)          1       2       3       4       5
    pattern, y = 7      0.2000  0.5333  0.1333    -     0.1333
    pattern, y = 11     0.1556  0.4148  0.1630  0.0444  0.1630
    pattern, y = 13     0.1273  0.3394  0.1603  0.0646  0.1818
    automaton, y = 7    0.0489  0.1539  0.1118  0.0492  0.2400
    automaton, y = 11   0.0447  0.1497  0.1084  0.0484  0.2384
    automaton, y = 13   0.0439  0.1492  0.1045  0.0500  0.2344

They are not close, and they move in opposite directions: the pattern's distribution shifts
substantially with `y` while the automaton's is nearly static. The automaton counts which words *could*
occur; `n_j` counts how often they *do*.

What survives is Result 1, which is about mechanics rather than counting, and is new.

## 6. The step form of the remaining gap

A gap of length `g` contributes to `N(L)` once for each of its `g - L` starting positions, so it
contributes to `N(L) - N(L+1)` exactly once when `g > L`. Hence

> **`G(L) = N(L) - N(L+1)`**, and therefore `h(L) = 1 - rho(L)` with `rho(L) = N(L+1)/N(L)`.

Verified for gear sets to 7, 11, 13. So the remaining gap of the covering route -
`min_L h(L) = h(1)`, section 26c - is exactly

> **`rho(L) <= rho(1)` for every `L`: the step ratio of the covering counts is largest at `L = 1`.**

### 6a. `rho(1)` in closed form, and why gear 3 is what makes it free

The offsets `o_q` are independent and uniform, and gear `q` blocks position `i` exactly when
`o_q in {i-1, i} mod q`. So `P(position i exposed) = prod (1 - 2/q) = d`, and for two adjacent
positions gear `q` must avoid three offsets, giving `P(0 and 1 both exposed) = prod (1 - 3/q) =: e`.
Inclusion-exclusion on the complement:

    P(0 blocked) = 1 - d
    P(0 and 1 both blocked) = 1 - 2d + e
    rho(1) = P(1 blocked | 0 blocked) = (1 - 2d + e)/(1 - d)

**`e = 0` always, because `3` is in the gear set** and contributes the factor `1 - 3/3 = 0`. Gear 3
cannot avoid two adjacent positions at all - which is its blocking law restated. Hence

    rho(1) = (1 - 2d)/(1 - d)   and   h(1) = d/(1 - d),   exactly.

That is what "`h(1)` is free" in section 26c means mechanically: gear 3 annihilates the third-order
term, so the `L = 1` hazard is a two-term expression with no correction. Matches measurement to 8
digits at `y = 7, 11, 13, 17, 19`.

### 6b. The mechanism, and where the one-gear version of it fails

`rho(L) <= rho(1)` says the *first* step of a blocked run is the cheap one. The reason is the domino:
gear `q` blocks the adjacent pair `{o, o+1}`, so given position 0 blocked by gear `q`, position 1 comes
free whenever that gear's domino points right - half of the offsets that block 0. No later position can
share a domino with position 0, so no later step is cheap in the same way.

Turning that into a proof needs `P(L blocked | [0,L) blocked) <= 1 - d`, and the natural one-gear
version of it - "the offsets that block `L` are below-average at covering `[0,L)`, so conditioning
disfavours them" - is **false in general**, with an exact threshold. Write `L = aq + b` with
`0 <= b < q`, and let `n(r) = #{i in [0,L) : i = r mod q}`, so `n(r) = a+1` for `r < b` and `a`
otherwise. The usefulness of offset `o` is `u(o) = n(o) + n(o+1)`, averaging `2L/q = 2a + 2b/q`. The two
offsets that block position `L` are `o = L-1` and `o = L`, and for `1 <= b < q-1`

    u(L-1) = 2a + 1,    u(L) = 2a,    sum = 4a + 1   against   2 * average = 4a + 4b/q.

So those offsets are jointly below average only when `b >= q/4`. For `b < q/4` they are *above*
average, and the conditioning pushes the wrong way for that gear. Any proof therefore has to use the
gears jointly rather than one at a time - a per-gear usefulness argument cannot work, and the
`b >= q/4` threshold says exactly which gears it fails for at a given `L`.

This is *weaker* than log-concavity of `N`, which would make `rho` decreasing and is false at `L = 3`
(section 27a). It asks only that the maximum sits at the left end.

Checked exhaustively over **every** `L` up to `F_h`, not merely the block starts:

    y     P          F_h   rho(1)       peak rho     at L   h(1)/d
    7     105        15    0.83333333   0.83333333   1      1.166667
    11    1155       21    0.86764706   0.86764706   1      1.132353
    13    15015      33    0.89024390   0.89024390   1      1.109756
    17    255255     54    0.90439093   0.90439093   1      1.095609
    19    4849845    75    0.91530740   0.91530740   1      1.084693

The peak is at `L = 1` every time, with no near-ties elsewhere. `h(1)/d = 1/(1-d)` shrinks towards 1
as `y` grows, so the margin the bound has over the threshold is thin but exactly known, and positive
for every `y`.

Mechanically `rho(L) <= rho(1)` says: **conditioning on a long blocked run makes the next position
less likely to be blocked than it is unconditionally.** It is tempting to read that as gear exhaustion
- a gear whose one block per rotation has been spent inside the run is not available to block just past
it - but section 7 measures exactly that and finds the opposite happens gear by gear. The statement is
true jointly and false per gear.

## 7. The hazard is not a per-gear effect: two refutations

Script: `research/negative_association.py`. Everything here is exact enumeration over `[0, P)`, so the
numbers are integers rather than estimates.

Gear `q` blocks the absolute positions `= 0, 1 mod q`, so a window starting at `m` sees gear `q` at
effective offset `-m mod q`, and ranging `m` over `[0, P)` ranges over all offset vectors bijectively by
CRT. With `A(L) = {m : m, ..., m+L-1 all blocked}` we have `|A(L)| = N(L)`, and the `m in A(L)` whose
next position is exposed number one per maximal blocked run of length at least `L`, so

    h(L) = P( position m+L exposed | m in A(L) ),

the hazard as a conditional probability. Unconditionally the per-gear events "gear `q` blocks the next
position" are independent with probability `2/q`, and `d` is the product of their complements. So
`h(L) >= d` would follow from two claims, both of which were tested:

    (M)  the per-gear marginals do not rise:  P( q blocks m+L | m in A(L) ) <= 2/q
    (N)  weak negative association:  h(L) >= prod_q ( 1 - P( q blocks m+L | m in A(L) ) )

**(M) is false, and badly.** The conditional marginals routinely *exceed* `2/q` - at `y = 17` it fails
at 36 of the 53 values of `L`, with the worst ratio `marg / (2/q) = 1.4875` at `L = 50, q = 7`, and at
`y = 13` reaching `1.625` at `L = 27, q = 13`. So conditioning on a blocked run makes each individual
gear **more** likely to block the next position, not less. Gear exhaustion is not the mechanism, and
the intuitive reading of section 6 is wrong at the per-gear level.

**(N) is false, but only just.** It fails once at `y = 17` (at `L = 2`, `h = 0.105717` against a product
bound of `0.108535`, short by 2.6%) and twice each at `y = 11` and `y = 13`, always at small `L`, and
holds at every other `L`.

Together these say the bound is a genuinely **joint** property. The gears become strongly correlated
under the conditioning, and the union of their blocking events is smaller than independence would
predict even though each marginal has risen. No argument that treats gears one at a time, or that
multiplies per-gear conditional probabilities, can reach it.

## 8. The tight cases are a short fixed list of small L

Because `G` is constant on the blocks `{1,2}, {3,4,5}, {6,7,8}, ...` while `N` decreases, `h` rises
within a block, so the minima of `h/d` sit at the block starts `L = 1, 3, 6, 9, ...`. Ranking those:

    y     F_h    tightest block starts, as (h/d, L)
    7     15     (1.1667, 1) (1.1667, 6) (1.1667, 9) (1.4000, 3) (2.3333, 12)
    11    21     (1.1324, 1) (1.2162, 6) (1.3004, 3) (1.3162, 9) (1.9012, 15)
    13    33     (1.1098, 1) (1.2002, 6) (1.2409, 3) (1.3039, 9) (1.3481, 24) (1.4259, 21)
    17    54     (1.0956, 1) (1.1787, 6) (1.2052, 3) (1.2207, 24) (1.2481, 21) (1.2711, 9)
    19    75     (1.0847, 1) (1.1602, 6) (1.1788, 3) (1.1885, 21) (1.1896, 24) (1.2420, 9)
    23    102    (1.0768, 1) (1.1455, 6) (1.1600, 3) (1.1833, 21) (1.1845, 24) (1.2186, 9)
    29    129    (1.0711, 1) (1.1344, 6) (1.1468, 3) (1.1742, 21) (1.1834, 24) (1.2007, 9)

`y = 23` was computed twice, once from the pattern in Python and once by enumerating all 111546435
offset vectors in `rust2/src/bin/coverbound.rs`, with identical results to four decimals. `y = 29` is
the Rust enumeration only - 3.2 billion vectors - and adds no new tight members, the full top ten being
`1, 6, 3, 21, 24, 9, 15, 39, 54, 33`.

Two things stand out.

**The tight `L` are fixed small numbers, not a fixed fraction of `F_h`.** The list is
`1, 3, 6, 9, 21, 24` with `39` and `54` entering at larger `y`, while `L/F_h` for `L = 24` falls from
`0.727` at `y = 13` to `0.32` at `y = 19`. So the tight set does not spread out as the gear set grows.

**The four tightest are `1, 6, 3, 9` - exactly the four cases already proved** (sections 18b, 18c, 18d
and 24c of `covering-bound-route.md`). The next tier is `21` and `24`, then `39` and `54`, with margins
stable around `1.16` to `1.25` rather than drifting towards 1.

That reopens the per-`j` recipe of section 24b, which was abandoned because it needs a condition per
value of `j` and the number of those grows like `0.055 y^2`. If only a fixed finite list of `j` is
actually tight, the rest need nothing better than a crude bound with a factor of `1.13` of slack, and
the recipe handles the short list. The falsification test is whether the tight list keeps growing with
`y`; the measurement above says it has not through `y = 29`.

### 8a. A pattern in the tight list, stated as a prediction and not a law

The tight block starts through `y = 29` are `3, 6, 9, 15, 21, 24, 33, 39, 54` together with `L = 1`. Six
of the nine are `3q` for `q` prime - `6, 9, 15, 21, 33, 39` at `q = 2, 3, 5, 7, 11, 13` - and the two
that are not, `24` and `54`, are `6 * 4` and `6 * 9`, that is `6q^2` at `q = 2, 3`.

If that reading is right the next member is `6 * 25 = 150`, and the next `3q` members are `51` and `57`.
`F_h(29) = 129`, so `150` cannot appear yet; `F_h(31)` should be around 160, which makes the prediction
decidable at `y = 31`. This is recorded as a prediction with a named test, **not** as a law - five
patterns in this programme have been recorded from small data and then refuted, and this one rests on
nine numbers with two families fitted to them.

## 9. The per-`j` recipe does scale, because almost every term vanishes

Script: `research/closed_hazard.py`. Section 25b judged the per-`j` recipe unusable because `c_j(L)` is
a sum over `2^L` subsets. That estimate ignores how aggressively the head gears annihilate terms:
`prod (q - |W_q(T)|)` is zero the moment **one** gear has `W_q(T) = Z_q`, and a depth-first scan that
prunes on the first fully covered gear never visits the rest.

    L        1    3    6    9   15    21     24     25     30       39
    visited  2    4   10   19   61   181    289    353    721     2548
    2^L      2    8   64  512  32768  2.1e6  1.7e7  3.4e7  1.1e9   5.5e11

Every visited subset contributes; nothing is wasted. At `L = 39` that is 2548 terms instead of
`5.5 * 10^11`. So the recipe reaches every tight block start comfortably, and the exact coefficients are

    L =  1   c_0=1,        c_2=-1
    L =  3   c_0=3,        c_2=-3
    L =  6   c_0=15,       c_2=-18,       c_4=3
    L =  9   c_0=105,      c_2=-135,      c_4=42
    L = 15   c_0=15015,    c_2=-22275,    c_4=9792,      c_6=-1272
    L = 21   c_0=4849845,  c_2=-7952175,  c_4=4454838,   c_6=-1038864, c_8=94284,   c_10=-1920
    L = 24   c_0=111546435, c_2=-190852200, c_4=118032579, c_6=-33045960, c_8=4138290, c_10=-167856

**Validity condition.** `c_j(L)` is built from the gears `q <= L`, so it describes the machine only when
the gear set contains all of them. The hazard at `L` needs `N(L)` and `N(L+1)`, so its condition is
`y >= L + 1`. Both halves of this were caught by the checks rather than by inspection: at `y = 13,
L = 21` the formula returns `406008` against a true `N(21) = 312`, and at `y = 29, L = 30` the `N(31)`
term invents a gear 31 the machine does not have and the hazard comes out at `h/d = -536`. With the
guard on `L + 1` the closed forms reproduce `N(L)` exactly at every tight `L` in range, checked against
direct construction of the pattern at `y = 13, 17, 19, 23`.

This is what makes the tight list of section 8 actionable: each of those cases is a finite, explicit
inequality between products over the gear set, with no enumeration of the pattern required.

## 10. Status

New and verified here:

* the minimal size law `(q+1)/2`, exact, with the two known blocking laws as its first two cases;
* the exposure form: any `(q-1)/2` positions are simultaneously exposable to gear `q`;
* minimal span growing like `1.9 q`, by exact dynamic programme;
* the factorisation law with the correct `span + 1` threshold;
* the `c_j(L)` decomposition, with the gear set entering only through `prod (q - j)`;
* `G(L) = N(L) - N(L+1)`, hence the step form `rho(L) <= rho(1)`;
* `rho` peaking at `L = 1` for all `L`, all gear sets to `y = 19`;
* `rho(1) = (1-2d)/(1-d)` proved, from `prod (1 - 3/q) = 0` via gear 3;
* the tight block starts being a short fixed list of small `L`, stable through `y = 23`;
* the per-`j` recipe scaling after all, 2548 contributing terms at `L = 39` rather than `5.5 * 10^11`;
* every tight case verified from closed forms alone for `y = 23` through `199`, worst case always the
  proved `L = 1`.

Refuted here: the finite-automaton route in both its forms; the per-gear marginal claim `(M)`, which
fails by up to 63%; weak negative association `(N)`, which fails narrowly at small `L`.

Corrected here: the "`q >= 13`: none within span 27" claim; the single-word forbidden lists for gears 5
and 7; the `span` versus `span + 1` factorisation threshold; the gear-exhaustion reading of the step
form; and the validity range of the closed forms, which needs `y >= L + 1`, not `y >= L`.

Still open: `rho(L) <= rho(1)`, the whole of the remaining gap in one inequality. The route this
section suggests is now concrete rather than open-ended - prove the short tight list case by case from
the closed forms, which are in hand, and cover every other `L` with a crude bound, which needs only the
`1.14` of slack the measurements show at `y = 23`. What is not yet established is that the tight list
stays finite as `y` grows, and the margins at its members do drift slowly downward
(`h(6)/h(1)` runs `1.0740, 1.0815, 1.0759, 1.0696, 1.0638` over `y = 11` to `23`), so the slack cannot
be assumed constant.
