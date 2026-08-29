# The non-tensor sector's spectrum is Farey-Chebyshev, and it is more rigid than GUE

Status: PROVED (path decomposition; distinct-level count) + SCRIPT-VERIFIED
(`research/nontensor_spec.py`: exact dense diagonalisation at machine 11,
exact combinatorial verification at 13/17/19/23, exact distinct-level counts
to machine 37; spacing statistics are floats, labeled). Established round 22
(Lateral). Prior-art check: NOT YET CHECKED (section 6).

## 1. WHAT IT IS

Plain language. Round 21 (`eigenvalue-statistics.md`) tested the human's
Riemann/GUE hunch on the machine's TENSOR operators and found Poisson - but
that test could never have found anything else, because a CRT-product
spectrum is Berry-Tabor by construction. It also localised the one place a
GUE-bearing operator could live: the NON-TENSOR sector, everything built from
B = I - (x)_q E_q (blocking is the complement of a product, which is Wall V
in operator form). This entry runs the test there. The answer is a theorem,
not a statistic: the non-tensor sector's Hermitian operators are disjoint
unions of PATH GRAPHS, one per gap of the machine, so their spectra are
Chebyshev values 2cos(pi a/b) with b bounded by F(M)+1 - only O(F^2) DISTINCT
levels out of a period of size P, each with P/F^2-fold multiplicity. Those
distinct levels are the image of a FAREY SEQUENCE, whose spacings obey Hall's
distribution with a HARD GAP at 3/pi^2 of the mean. So this spectrum is not
GUE either - it is MORE RIGID than GUE.

Precise form.

PATH-DECOMPOSITION THEOREM. Let B = diag(blocked), S the slot shift, and
A = BS + (BS)^T, the Hermitian part of the blocked walk whose nilpotency
index is F(M). A is the adjacency matrix of the graph on Z_P with an edge
{k, k+1} exactly when k+1 is blocked. Between consecutive openings at slots
m and m+g the vertices m..m+g-1 are chained and the edge {m+g-1, m+g} is
absent, so A is the disjoint union over the machine's gaps of PATH graphs, a
gap of g slots contributing P_g. Hence exactly

    spec(A) = union over gaps g, with multiplicity W_1(g), of
              { 2 cos(pi j/(g+1)) : j = 1..g }.

COROLLARY 1 (distinct levels). The distinct eigenvalues are 2cos(pi a/b) over
reduced fractions a/b in (0,1) with b <= F+1, so

    #distinct = |Farey(F+1)| - 2 = sum_{b=2}^{F+1} phi(b)  ~  3 (F+1)^2/pi^2,

against a period of size P. Measured (exact):

    y      F      P                 distinct   ties per level
    11     7      385                     21             18
    13    11      5,005                   45            111
    17    18      85,085                 119            715
    19    25      1,616,615              211          7,662
    23    34      37,182,145             383         97,081
    29    43      1,078,282,205          603      1,788,196
    31    58      33,426,748,355       1,085     30,808,063
    37    88      1,236,789,689,135    2,455    503,783,987

COROLLARY 2 (rigidity). The distinct spectrum is a smooth image of the Farey
set, whose consecutive spacings satisfy s_min/s_mean > 3/pi^2 = 0.30396 - a
HARD GAP, no small spacings at all. Measured (floats):

    y    F   #levels   <r~> (Farey coord)   s_min/s_mean   P(s < 0.1 mean)
    11    7       21        0.72055            0.47619          0
    17   18      119        0.69280            0.38562          0
    23   34      383        0.70458            0.34046          0
    29   43      603        0.70260            0.33333          0
    31   58    1,085        0.70304            0.32789          0
    37   88    2,455        0.70460            0.32053          0

against <r~> = 0.38629 (Poisson), 0.53590 (GOE), 0.60266 (GUE), 1 (clock).
The value sits at 0.703, ABOVE GUE, and s_min/s_mean descends monotonically
to Hall's constant 3/pi^2.

COROLLARY 3 (the whole sector). Every operator of the form diag(w) S^t + h.c.
with w a 0/1 vector has maximum degree 2, hence is a union of paths and
cycles, hence has spectrum inside {2cos(pi a/b)} u {2cos(2 pi a/b)}: always
degenerate, never repulsive. And the operators that carry the sector's
GROWING Kronecker/Schmidt rank - the deep window operators
(BS)^n = diag(v_n) S^n (see `nontensor-sector.md`) - are NILPOTENT, spectrum
{0} with multiplicity P at every depth. The cross-machine word-level transfer
matrix H is block-triangular with an INTEGER diagonal
(q' - #distinct residues), so it too has a bounded, hugely degenerate
spectrum.

THE DICHOTOMY. In this machine, where the spectrum is RICH the operator
FACTORISES (Poisson by Berry-Tabor); where the operator does NOT factorise
the spectrum is either DEGENERATE (Farey-Chebyshev, or integer) or EMPTY
(nilpotent). The growth of the non-tensor sector happens exactly in the
nilpotent direction, which has no spectrum at all. So NO operator of this
machine can be GUE, and GUE is now bracketed three times over:

    clock 1.000 > Farey-Chebyshev 0.703 > GUE 0.603 > GOE 0.536 >
    Poisson 0.386 = the machine's tensor sector.

## 2. WHY IT MIGHT BE NOVEL

The path spectrum 2cos(pi j/(L+1)) is textbook; Farey-spacing statistics
(Hall's distribution, the 3/pi^2 hard gap) are classical analytic number
theory. What appears unrecorded:

- the identification: the Hermitian part of a SIEVE's blocked-walk operator
  is exactly a disjoint union of paths indexed by the sieve's GAPS, so its
  spectral measure IS the gap histogram read through Chebyshev polynomials -
  a dictionary between a Jacobsthal-type gap spectrum and an operator
  spectrum;
- the consequence that a sieve operator's distinct-level count is
  |Farey(F+1)| - 2, i.e. O(Jacobsthal^2), which is the exact obstruction to
  any random-matrix behaviour;
- the resulting completeness argument: combining this with the round-21
  tensor result, NO natural operator of the machine is GUE, for the
  structural reason that spectral richness and non-factorisation are
  mutually exclusive here. That is a negative answer to a Hilbert-Polya-style
  question with a proof rather than a statistic.

## 3. PROOF / STATUS

PROVED: the path decomposition (two lines, above); the distinct-level count
(reduced fractions with denominator <= F+1); Corollary 3's degree-2 argument.
SCRIPT-VERIFIED (`research/nontensor_spec.py`): dense `eigvalsh` at machine 11
(385 levels) agrees with the path prediction to 1.3e-15; the path/gap
bookkeeping (number of paths = number of openings, sum of lengths = P,
longest path = F) is asserted at machines 13/17/19/23; the distinct-level
count is asserted equal to sum phi(b) by direct construction at every
machine; the hard gap s_min/s_mean > 3/pi^2 is asserted at every machine.
Spacing statistics are floats and labeled as such.

## 4. IMPLICATIONS

Inside the project: closes the Riemann-bridge question at finite machines
completely - round 21 refuted it on the tensor side and named the non-tensor
sector as the only remaining location; this entry shows that location cannot
carry it either, for a reason that is structural rather than numerical. It
also gives a clean statement of what the non-tensor sector IS spectrally: its
Hermitian content is exactly the gap histogram, so any spectral attack on F
is circular unless it goes through the nilpotent (spectrum-free) direction -
which is precisely the arity-free/nilpotency route the round-22 spine is
about.

Outside: a family of sieve-derived operators whose level statistics are
neither Poisson nor RMT but Hall/Farey, with a hard spacing gap - a concrete
counterexample to the folklore trichotomy "integrable -> Poisson, chaotic ->
RMT" in an arithmetic setting.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Hilbert-Polya / Montgomery-Odlyzko (a second, sharper negative finite-machine
data point); Berry-Tabor (this is a non-Poisson exactly-solvable case, so it
sits outside the usual dichotomy); Jacobsthal / Ziller-Morack h_2 (the
distinct-level count is an exact function of the paired Jacobsthal value, so
a spectral statistic of the operator determines F and vice versa).

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"path graph spectrum union level statistics Farey fractions Hall
distribution"; "spectral statistics of sparse 0/1 band matrices paths and
cycles"; "adjacency matrix of coprime residues consecutive integers
spectrum"; "Chebyshev spectrum sieve operator Jacobsthal"; "level spacing
distribution Farey sequence three-distance theorem hard gap 3/pi^2".
Expected nearest art: Hall (1970) on Farey spacings and the
Boca-Cobeli-Zaharescu work on Farey statistics; the delta to check is the
identification of a sieve/Jacobsthal operator's spectrum with a Farey set and
the resulting GUE-exclusion argument.

---

## 7. ROUND-26 EXTENSION: THE MULTIPLICITY THEOREM, AND A CORRECTION TO
##    COROLLARIES 1 AND 2

Lateral, round 26 (2026-08-29), backlog item U4. Script
`research/mirror_lever2.py` part F (log `research/data/mirror_lever2.log`).

### 7.1 The multiplicity theorem

Write an eigenvalue `2 cos(pi j/(g+1))` as `2 cos(pi a/b)` with `a/b` in lowest
terms. Then `b | g+1`, and for a FIXED `b` the values of `a` coprime to `b`
arise from exactly the gaps `g = -1 (mod b)`, each contributing once. Hence

>   **THEOREM.  mult(2 cos(pi a/b)) = Sigma(b) := sum_{g = -1 (mod b)} W_1(g),
>   independent of `a`.**

So the eigenvalue multiplicities of `A` **are the gap histogram's residue-class
counts**, one class (`-1`) per modulus. The map is invertible: by Mobius
inversion over multiples,

>   `W_1(b-1) = sum_{t >= 1} mu(t) Sigma(tb)`,

so `spec(A)` (with multiplicity) and the gap histogram determine each other
exactly, and `F + 1 = max{ b : Sigma(b) > 0 }`. This is the constructive form of
`nilpotent-invariants.md`'s negative ("every unitary invariant of `BS` is the
gap histogram"): here is the inversion.

Two corollaries.

**(i) PARITY.** By the mirror parity law (`mirror-parity-laws.md` 7.3) the only
gap length with an odd count is `1`, so `Sigma(b)` is odd iff `b | 2`:

>   **every eigenvalue multiplicity of `A` is EVEN, except that of the
>   eigenvalue `0` (`b = 2`).**

Asserted exactly at m11..m29 for every `b <= F+1`. In particular the top
eigenvalue `2cos(pi/(F+1))` has even multiplicity: `A` never has a simple
Perron eigenvalue, and the maximal gap never occurs exactly once.

**(ii) THE LEVEL COUNT IS A DIVISOR-CLOSURE STATISTIC.** Corollary 1's
`|Farey(F+1)| - 2` silently assumes **every** gap length `1..F` is realised.
The true statement is

>   `#distinct = sum { phi(b) : b >= 2 and b | g+1 for some REALISED gap g }`,

and machines with HOLES in their gap spectrum break the Farey form.

### 7.2 CORRECTION to Corollary 1's table

Recomputed on the true supports (m11-m29 from the full-period `W_j` census,
m31/m37 from the exact realised 4-tuple dictionaries, m41 from the COV-SAT
support `{1..91} \ {84,87,89}`; the tuple route is validated at m23 and m29
where both sources exist, and the `phi`-sum is cross-checked against direct
construction of the level set at every machine):

    y      F   holes                 distinct (TRUE)   as published (naive)
    11     7   -                                  21                    21
    13    11   {9}                                41                    45
    17    18   {17}                              113                   119
    19    25   {19,24}                           183                   211
    23    34   {24}                              363                   383
    29    43   {41,42}                           549                   603
    31    58   {54,56,57}                        981                 1,085
    37    88   13 holes (73..87)               1,813                 2,455
    41    91   {84,87,89}                      2,467                 2,595

Only machine 11 (whose spectrum has no holes) was right. **THE LOSS RULE**, and
it is exact at all nine machines: `b` is absent iff every multiple of `b` in
`[2, F+1]` is `hole+1`; for `b > (F+1)/2` that is just "`b-1` is a hole", and
since every observed hole lies in the top half of the range (the spectrum fills
monotonically from below),

>   `loss = sum over holes h of phi(h+1)`,

e.g. `phi(85) + phi(88) + phi(90) = 64 + 40 + 24 = 128` at m41 - predicted
before the run and matched exactly. **The hole list is a spectral observable**:
an arithmetic-selection object nobody can predict is exactly the defect between
the true level count and the Farey count.

### 7.3 CORRECTION to Corollary 2's rigidity numbers

Corollary 2's spacing statistics were also computed on the full Farey set.
On the true level set (same unfolding):

    y    #levels   <r~> true   <r~> as published   s_min/s_mean true
    11        21      0.7206              0.7206              0.4762
    13        41      0.6306              0.6987              0.3636
    17       113      0.6785              0.6928              0.3875
    19       183      0.6507              0.7002              0.3297
    23       363      0.6876              0.7046              0.3226
    29       549      0.6982              0.7026              0.3182
    31       981      0.6897              0.7030              0.3070
    37     1,813      0.6788              0.7046              0.2422
    41     2,467      0.6938              0.7049              0.3011

WHAT SURVIVES: `P(s < 0.1 mean) = 0` exactly at every machine - and now for a
reason rather than a measurement, since deleting levels only lengthens spacings,
so a subset of a set with a hard gap keeps it. `<r~>` stays above GUE's 0.6027
at every machine, so the round-22 conclusion (this spectrum is MORE RIGID than
GUE, the Riemann bridge is closed at finite machines) is unchanged.

WHAT DOES NOT: the **normalised** hard gap. "Descending to `3/pi^2 = 0.30396`"
was a property of the full Farey set; on the true level set `s_min/s_mean` is
not monotone and **dips below the Hall constant at m37 (0.2422)**, because
deleting levels raises the mean spacing while the surviving minimal spacing is
unchanged. The ABSOLUTE hard gap is intact; the ratio statement is not a law.

(Floats in 7.3 are labeled as such; all integers in 7.1-7.2 are exact.)

### 7.4 Status and prior art

The multiplicity theorem and the Mobius inversion are proved (one paragraph
each) and assertion-gated. The corrected level counts are exact integers, each
cross-checked by direct construction of the level set. Prior-art check for the
round-26 material: **not yet checked**; suggested terms - "eigenvalue
multiplicities of a disjoint union of paths", "Chebyshev spectrum path forest
divisor", "Farey subset spacing statistics", plus section 6's terms.
