# 3. Columns that are always open, the mirror, and the parity of gap counts

## In plain words

Some columns are open no matter how many gears there are. The very first column is one; so are
the two columns at the exact halfway point of the machine's full cycle. The whole pattern of
openings is symmetric: read the cycle backwards from the start and it looks the same, and this
mirror is the only symmetry the pattern has. Because of the mirror, every gap size between
openings shows up an even number of times in a cycle, except the size-one gap at the halfway
point. So the longest gap, the record, can never appear just once.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; a gear `q >= 5` strikes `k` iff `k = +-u_q (mod q)`
(file 02).  Let `M` be a finite set of gears, `P = prod_{q in M} q` its period (odd), and
`O subset Z_P` its set of openings (residues struck by no gear); `N := |O| = prod (q - 2)` is
odd.  The **gaps** of `M` are the distances between consecutive openings; the **record** `F(M)`
is the largest gap; `W(g)` is the number of gaps of length `g` in one period.

Classical translation: an opening is a column neither of whose members is divisible by a prime
of `M`; the mirror `k -> -k` exchanges the two members of a column up to sign.

## Statement

**(a) Column 0.**  `0` is an opening of every machine.

**(b) The antipode.**  The columns `(P-1)/2` and `(P+1)/2` are both openings of `M`; the gap
between them has length 1.

**(c) The mirror.**  `k in O` iff `-k in O`.  Column `0` is the only fixed point of `k -> -k`
on `Z_P`.

**(d) The symmetry group.**  The rotations `k -> k + b` and reflections `k -> b - k` of `Z_P`
that map `O` onto itself are exactly the identity and the mirror `k -> -k`.  More generally, the
affine bijections `k -> ck + b` (`c` a unit mod `P`) preserving `O` are exactly those with
`b = 0` and `c = +-1 (mod q)` for every gear `q` -- `2^{|M|}` maps -- and among them only
`c = +-1 (mod P)`, the identity and the mirror, send adjacent columns to adjacent columns.

**(e) Gap counts are even.**  For every `g >= 2`, `W(g)` is even.  In particular the record
gap `F(M)` occurs at least twice per period, never exactly once.

## Proof

**(a)**  The members of column `0` are `-1` and `1`; no prime divides them.

**(b)**

1. `P` is odd, so `s := (P+1)/2` is an integer and `6s = 3P + 3`.  The members of column `s`
   are `3P + 2` and `3P + 4`.
2. For a gear `q` of `M`, `q | P`, so `q | 3P`; if `q` divided `3P + 2` or `3P + 4` it would
   divide `2` or `4`, impossible for `q >= 5`.  So `s` is an opening.
3. `(P-1)/2 = P - s = -s (mod P)` is an opening by (c).  The two are adjacent columns, so the
   gap between them is 1.

**(c)**

4. For `1 <= k < P` and `q | P`, write `P = qc`.  Then
   `(6(P-k) - 1) + (6k + 1) = 6P = q(6c)` and `(6(P-k) + 1) + (6k - 1) = 6P`.  Hence
   `q | 6(P-k) - 1` iff `q | 6k + 1`, and `q | 6(P-k) + 1` iff `q | 6k - 1`.  A gear strikes a
   column iff it divides one of the two members, a condition symmetric in the two, so `q` strikes
   `P - k` iff it strikes `k`.  This holds for every gear, so `k in O` iff `-k in O`.
5. `k = -k (mod P)` iff `2k = 0 (mod P)` iff `k = 0`, since `P` is odd.

**(d)**

6. *Rotations.*  Suppose `O + b = O`.  Since `0 in O`, every multiple `jb` lies in `O`.  If some
   gear `q` does not divide `b`, then `b` is invertible mod `q` and some multiple `jb` is
   `= u_q (mod q)`, a struck column -- contradicting `jb in O`.  So every gear divides `b`,
   i.e. `b = 0 (mod P)`.
7. *Reflections.*  If `k -> b - k` preserves `O`, composing it with the mirror (which preserves
   `O` by (c)) gives the rotation `k -> b + k` preserving `O`; by step 6, `b = 0`: the
   reflection is the mirror.
8. *Affine maps.*  Let `f(k) = ck + b` with `c` a unit mod `P`, and suppose `f(O) = O` (so also
   `f^{-1}(O) = O`).  Fix a gear `q` and put `rho := c u_q + b (mod q)`.  Suppose
   `rho not in {u_q, -u_q}`.  Choose, by the Chinese remainder theorem, a column `k` with
   `k = u_q (mod q)` and, for every other gear `q'`, a residue `k mod q'` such that
   `ck + b` is not a tooth of `q'` (possible: as `k mod q'` runs over `Z_{q'}` so does
   `ck + b mod q'`, and `q' - 2 >= 3` residues are not teeth).  Then `f(k)` is open for `q`
   (its residue is `rho`) and for every other gear, so `f(k) in O`; but `k` is struck by `q`,
   so `k not in O = f^{-1}(O)`.  Contradiction.  Hence `c u_q + b = e_1 u_q` and, by the same
   argument at the other tooth, `-c u_q + b = e_2 u_q`, with `e_1, e_2 in {+1, -1}`.
   Subtracting: `2c u_q = (e_1 - e_2) u_q`; the left side is nonzero mod `q` (`2`, `c`, `u_q`
   are units), so `e_1 = -e_2` and `c = e_1 (mod q)`.  Adding: `2b = 0`, so `b = 0 (mod q)`.
   This holds for every gear, so `b = 0 (mod P)` and `c = +-1 (mod q)` for each `q`.
   Conversely any such `c` maps each gear's tooth pair `{+-u_q}` to itself, hence preserves
   `O`.  By CRT there are `2^{|M|}` such `c`.
9. `f(k+1) - f(k) = c`, so `f` sends adjacent columns to adjacent columns iff `c = +-1 (mod P)`.

**(e)**

10. List the openings of one period `0 = o_0 < o_1 < ... < o_{N-1} < P` and put `o_N := P`.
    Gap `t` (for `0 <= t <= N-1`) is `L(t) := o_{t+1} - o_t`; gap `N-1` is the wrap gap.
11. By (c) the map `k -> P - k` is an order-reversing bijection of `O cap (0, P)` onto itself,
    so `o_{N-t} = P - o_t` for `1 <= t <= N-1`.  Hence for `1 <= t <= N-2`,
    `L(N-1-t) = o_{N-t} - o_{N-t-1} = (P - o_t) - (P - o_{t+1}) = L(t)`, and
    `L(N-1) = P - o_{N-1} = o_1 = L(0)`.  So `m(t) := N - 1 - t` is an involution on gap
    indices preserving length.
12. Its fixed points solve `2t + 1 = N`, i.e. `t* = (N-1)/2`, unique since `N` is odd.  For
    that gap `o_{t*+1} = o_{N-t*} = P - o_{t*}`, so `o_{t*} + o_{t*+1} = P`: the gap is
    symmetric about `P/2`.  Both `(P-1)/2` and `(P+1)/2` are openings by (b), so
    `o_{t*} = (P-1)/2`, `o_{t*+1} = (P+1)/2`, and `L(t*) = 1`.
13. For `g >= 2`, `m` restricts to a fixed-point-free involution of the finite set
    `{t : L(t) = g}`, which therefore has even cardinality (remove a point and its image,
    repeat).  So `W(g)` is even.  Since `F(M) >= 2` for any nonempty `M` (a gear strikes some
    column, so not every gap is 1), `W(F(M))` is even and positive, hence `>= 2`.

## Status

Kernel: `Mirror.mirror_gear` (step 4), `Mirror.mirror_exposed11`, `Mirror.mirror_exposed29`;
`Mirror.antipode_open` (steps 1-2), `Mirror.antipode_exposed11`, `Mirror.antipode_exposed29`;
`Mirror.self_mirror_unique` (step 12, uniqueness), `Mirror.periods_odd`;
`Mirror.even_card_involution` (step 13, the pairing), `Mirror.window_count_even`,
`Mirror.adjacent_equal_even`, `Mirror.none_of_at_most_one` (the counting half in the abstract
form: an involution on indices preserving length, whose only fixed point does not carry the
counted length); machine-level instantiation only at m11 (`MirrorM11`).  Steps 6-9 (the
symmetry group) and step 11 (identification of the index involution with the mirror) are
written proofs only.

Verified computationally: all `92,400` affine maps at m11 and all `2,880` units at m13; all
`2P` rotations and reflections at m11 and m13; `W_j` parities against full-period censuses
m11..m29 for every `j <= 12`.

## Relationship to the conjecture

Bookkeeping plus one structural fact (the record occurs at least twice per period).  It
supplies a parity lever worth exactly a factor two in counting arguments; it bounds no gap; it
depends on nothing measured.

## Where it is used

The endpoint lever "a configuration counted at most once, and not carried by the self-mirror
window, does not occur"; the fact that any parity argument from a symmetry of `O` is worth
exactly a factor 2; column 0 as the anchor of the pair `(d_0, d_0)` in file 19.

## Source

Lateral rounds 25-27 (`docs/novel/mirror-parity-laws.md` sections 1-2 and 7.1);
`docs/proof-search/alignment-rules.md` 1.4 and 3.12; `proofs/Mirror.lean`.
