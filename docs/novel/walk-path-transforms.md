# The path of the walk from q^2: what builds its shape

Branch W.t (round 38), `research/proof/walk_transforms.md`. Scripts
`research/anchor235/r38/pt_path.py`, `pt_qr.py`, `pt_spectrum.py`, `pt_levels.py`.

**Overlap note.** Branch W.a (`docs/novel/walk-path-parts.md`, the arithmetic decomposition of
the same path, run in parallel and without contact) reached the offset-progression form, the
gear-5 forced phase, `L >= 2` and the quadratic-residue admissibility independently; those items
are corroborated twice by different code and should be credited to both. What is only here: the
`q mod 30` start-slot law in the anchor-30 coordinate and its forward/backward asymmetry, the
exact `+-1 (mod 8)` identification of the first column's admissible set, the square phase vector
as a joint restriction together with its null effect on `L`, the unique sole-striker tooth, the
two-sided one-tooth-per-run rule, the dip-plateau-spike depth identities, the forced square
columns behind `q^2`, and the section-level and across-level nulls.

## 1. WHAT IT IS

The walk from `q^2` under the machine `{5..q}` starts at column `k_0 = (q^2-1)/6` and crosses a
run of blocked columns before it lands on a twin. This entry is about that run - the **path** -
and about which parts of the machine put each of its columns there. Five statements, all exact
and all with zero exceptions over every prime gear `q = 5..19,997` (2,260 walks).

**(A) The start slot is pinned by `q mod 30`.** Since `q` is coprime to 30, `q^2 = 1` or
`19 (mod 30)`, so `k_0 = 0` or `3 (mod 5)`: the walk begins on the twin slot `29|31` when
`q = +-1, +-11 (mod 30)` and on `17|19` when `q = +-7, +-13 (mod 30)`, and **never** on `11|13`.
Consequences: gear 5 never strikes the walk's first column; `k_0 + 1` is a tooth of gear 5 in
both classes, so **gear 5 strikes offset 1 at every prime `q > 5` and the walk always has length
at least 2**; gear 5's whole contribution to the path is fixed in advance - it takes the offsets
`i = 1, 4 (mod 5)` in the first class and `i = 1, 3 (mod 5)` in the second, exactly `2/5` of the
path (measured 0.4025 of 88,677 path columns); and the two directions are inequivalent at
distance 1 (`k_0 - 1` is a gear-5 tooth only in the first class). In the smallest-striker word
this forces `P(next letter = 5 | this letter is not 5) = 2/3` against the letter's own share
`2/5`.

**(B) The offset transform: two arithmetic progressions per gear, phase a square.** The members
of column `k_0 + i` are `q^2 + 6i - 2` and `q^2 + 6i`, so

```
   gear g strikes offset i   iff   i = i_lo  or  i = i_hi  (mod g),
   i_lo = (2 - q^2) 6^{-1},   i_hi = -q^2 6^{-1},   i_lo - i_hi = d_g = 2 * 6^{-1}.
```

The path is therefore exactly the covering of `[0, L)` by two progressions per gear, of
difference `g` and separation `d_g`, whose common phase is a function of `q^2 mod g` alone. Two
consequences:

*The quadratic-residue bar.* Gear `g` can strike offset `i`, for any `q` whatever, only if
`2 - 6i` or `-6i` is a nonzero quadratic residue mod `g`. Generic offsets admit `3/4` of the
machine (median 0.7473 of 2,260 gears over the 192 nonzero offsets checked, min 0.7323); the
offsets `i = -6t^2` admit all of it (`-6i` is then the perfect square `(6t)^2`); and **offset 0
admits exactly the gears `g = +-1 (mod 8)`** (0.4960 of the machine).

*The square phase vector.* Both phases are functions of `q^2 mod g`, a square in every
coordinate, so the walk's phase vector lies in the image of the squaring map - one part in
`2^{pi(q)}` of the phase space (2,260 coordinates at `q = 20,000`).

**(C) The `q^2` tooth is the unique sole-striker tooth of the top gear.** Every other tooth of
`q` inside its own window `(q, q^2]` carries the member `q m` with `1 < m < q`, and every prime
factor of such an `m` is a gear of the machine; so `k_0` is the only column of the whole window at
which `q` is the sole striker of its member. Hence `dep(k_0) = 1 +` (the gears dividing `q^2-2`),
which is 1 exactly when `q^2 - 2` is prime (the square gate), and the walk begins at the
shallowest tooth in the window: mean depth 2.4212 at `k_0` against 3.2692 at a random blocked
column of the same section, depth-1 share 0.1996 against 0.1381, and the path's minimum depth
sits at offset 0 in 666 of 2,212 paths (30% against 4.8% for a uniform position).

**(D) One tooth per run, two-sided.** The maximal blocked run containing the column of `q^2`
holds exactly one strike of the top gear, i.e. `L < d` and `L^- < q - d` where `L`, `L^-` are the
forward and backward walk lengths and `d = 2*6^{-1} mod q` is the forward tooth arc. The arcs are
`(q+1)/3` forward and `(2q-1)/3` backward for `q = 5 (mod 6)`, and `(2q+1)/3` and `(q-1)/3` for
`q = 1 (mod 6)`; so the short arc is forward in one class and backward in the other. **Exactly
two exceptions in 2,260 walks - `q = 53` forward and `q = 31` backward - and each is in the short
direction of its own class.** Maximum of (path)/(arc) over all walks: 1.5000 in the short
direction, 0.7692 in the long one.

**(E) The depth profile: dip, plateau, spike, and a per-offset law.** Mean depth along the path
by normalised position (21 bins, 2,260 paths): 3.05, then 3.24-3.39 across the interior, then
3.75 at the last blocked column. The three levels are the machine's own independent-gear values:
the plateau is `sum_g 2/g = 3.1805` (measured 3.2692 on a random blocked column), the end spike is
`sum_g 2/(g-2) = 3.7007` (measured 3.7668 at the last column, 3.6934 just above the landing) -
the neighbour-of-an-opening law applied gear by gear - and the dip at the start is (C). In the raw
offset coordinate the mean depth is a function of `i` alone, given by the root counts of `-6i` and
`2-6i`: correlation 0.9694 across 193 offsets, from 2.0465 at `i = 87` to 5.8209 at `i = -54`
(the largest values sitting exactly at the offsets `-6t^2`, where `q^2 - 36t^2` splits).
Additionally, column `k_0 - 6t^2` is blocked at every `q > 6t+1` (its member is `(q-6t)(q+6t)`),
while forward the first such column is at offset `>= (2q+2)/3`, beyond the tooth arc: the
forced-composite square columns lie entirely behind the walk.

## 2. WHY IT MIGHT BE NOVEL

The arithmetic behind each piece is elementary; what is new is that it constitutes a **complete
account of the local shape of this particular object** in the machine's own terms, with the parts
separated by how many gears they need. (A) is the anchor alone; (B), (C), (D), (E)'s dip and
plateau are single-gear statements; (E)'s spike needs one proven pairwise law (neighbour-of-hit)
and nothing else. Nothing in the project's record says that the walk's start slot is pinned, that
gear 5's contribution to the path is deterministic up to one bit of `q mod 30`, that the walk's
first column is barred to half the machine by a quadratic-residue condition, that the phase vector
of the walk is a global square, or that the `q^2` tooth is the unique sole-striker tooth.

Honest shadows: the prime divisors of `n^2 - 2` being `+-1 (mod 8)` is Gauss's second supplement;
the per-offset opening density is a Hardy-Littlewood singular series for the polynomial pair
`(x^2+6i-2, x^2+6i)`; the length `L` follows the Hardy-Littlewood twin-gap null
`(ln q^2)^2/(12 C_2)` to within 2% from `q = 200`. Those are named and not claimed. The claim is
the decomposition and the exception counts.

## 3. PROOF

* (A), (B), (C), (D)'s frame, (E)'s two independent-gear values and the `-6t^2` family: **PROVED
  (elementary)**, each one line of congruence arithmetic, written out in
  `research/proof/walk_transforms.md` section 4.
* **SCRIPT-VERIFIED** at the counts above: `pt_path.py` (2,260 walks, all six representations,
  0 exceptions to (A), (C) at 337,011 teeth, the depth identities, the transition matrix with 12
  zero diagonal cells and 132 non-zero off-diagonal cells); `pt_qr.py` (**493,101,490** (gear,
  offset) checks with 0 disagreements between the offset-progression form and divisibility, 0
  strikes by a barred gear, 0 of 3,212 first-column strikers outside `+-1 mod 8`, 9,021 forced
  square columns all blocked); `pt_levels.py` (49 chain levels: 48/49 with `L < d`, 0/49 starting
  on `11|13`, 49/49 with the square-gate depth identity).
* (D) as an inequality (`L + L^- < q`) is **MEASURED**, not proved: 2 exceptions in 2,260, none
  above `q = 53`. Proving it is a twin-Bertrand-strength statement at scale `q/3`, as R2.a already
  recorded for the one-sided form.

## 4. IMPLICATIONS

Inside the project: it answers "what builds the path" by order of interaction. Orders 0 and 1
account for everything local; exactly one proven pairwise law is needed (for the far end); nothing
measured here needs three gears. The only feature left unexplained is the length `L` itself - the
first offset every progression misses - which is the root question. It also gives the walk frame
a two-sided form (D) and identifies the walk's first column as the unique thinnest tooth of the
top gear (C), which is why the square gate decides the deepest hop layer.

The sharp new question it leaves: the walk's phase vector is confined by (B) to a set of density
`2^{-pi(q)}`, and the length statistic does not notice - `L` sits at percentile 0.5270 among
1,000 random blocked-column walks of the same section, 47.34% of the walks from the other teeth of
`q` are longer, and `L` is at percentile 0.505 among the section's blocked runs. Either the
square-phase restriction is the reason the walk is short (and then it is a route to the root) or
it is inert, and that is directly testable on the tooth-counterfactual family.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Twin-Bertrand at scale `q/3` (statement (D)); the root question `F(y) < y^2/6` (Ziller-Morack
Conjecture 6 at the real teeth) through the length `L`; nothing else. The square gate `q^2 - 2`
prime is now readable as a half-machine event (the gears `+-1 mod 8` only), which is a small
sharpening of the record's square-gate statement.

## 6. PRIOR-ART CHECK

Not yet checked (no web access in this lane). Screened against the project's own record
(`docs/novel/README.md` index, `docs/proof-search/anchor-235.md` 9c-9g,
`research/proof/self_feeding.md`): the walk's tooth frame, the square gate, the level-free
transfer congruence, the hit/chain/neighbour-of-hit laws, the two-teeth kill-spacing law, the
gear-5 lock and the machine's Fourier structure are all on record and are cited, not claimed.
Statement (B)'s congruence is the walk-coordinate form of R2.a's transfer rule N3 (prior art); its
quadratic-residue consequence, the `+-1 mod 8` identification at offset 0 and the square phase
vector are not on record.
