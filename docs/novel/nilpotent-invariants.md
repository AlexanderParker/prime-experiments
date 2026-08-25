# What replaces the spectrum in a nilpotent sector: Jordan structure IS the gap histogram, and the only escape is a potential

Status: PROVED (the Jordan/permutation theorem, the invariant-completeness
corollary, the numerical-radius identity, the pseudospectral exponent) +
SCRIPT-VERIFIED with exact integers and labelled floats
(`research/nilpotent_invariants.py`, machines 11/13/17/19; log
`research/data/nilpotent_invariants.log`). Established round 23 (Lateral,
answering the round's brief "when the spectrum is empty, what carries the
information?"). Prior-art check: NOT YET CHECKED (section 6).

## 1. WHAT IT IS

Plain language. Round 22 found that the sieve machine's difficulty lives in
the NILPOTENT direction: with `S` the slot shift and `B = diag(blocked)`, the
blocked walk `N = BS` satisfies `N^n = diag(v_n) S^n` and `N^F = 0`, where
`F = F(M)` is the machine's maximal gap. Spectrum `{0}`: no eigenvalues, no
spectral radius, nothing for a spectral method to grip. The natural next
question - which the round-23 brief asked - is what the OTHER operator
invariants see there: singular values, Jordan blocks, the kernel filtration,
the pseudospectrum, the numerical range, the Schatten norms.

The answer is a single theorem that settles all of them at once, plus one
frame that escapes it.

**THEOREM (JORDAN = GAP HISTOGRAM).** `N` is PERMUTATION-similar (hence
orthogonally, hence unitarily equivalent) to a direct sum of nilpotent Jordan
blocks, one block of size `g` for each gap of `g` slots:

        N  ≅  ⊕_g  J_g^{⊕ W_1(g)}.

Proof, two lines: `N e_k = b(k+1) e_{k+1}`, so the directed graph of `N` is
the disjoint union of the chains of consecutive blocked slots; between
openings at `m` and `m+g` the chain is `m → m+1 → … → m+g-1 → 0`, a single
Jordan block of size `g`. Equivalently, by counting,

        rank(N^n) = Σ_g W_1(g) (g-n)_+        (the gap histogram's TAIL SUM)
        #{Jordan blocks of size exactly L} = W_1(L),   largest block = F.

**COROLLARY (INVARIANT COMPLETENESS - the negative).** Every UNITARY
INVARIANT of `N` is a function of the gap histogram `W_1` alone: the singular
values, all Schatten norms, the Jordan type, the kernel filtration
dimensions, the numerical range, the resolvent norms, the pseudospectra.
Nothing in the operator's invariant world carries information the histogram
does not already carry - so no operator invariant can bound `F` except
circularly. This is the sharpest available statement of why the spectral
frame stalls here, and it is a theorem rather than a report of failure.

It also UPGRADES round 22's path-decomposition theorem
(`farey-chebyshev-spectrum.md`): the Hermitian `A = N + N^T` being a disjoint
union of PATH graphs `P_g` is exactly the symmetrised shadow of this Jordan
decomposition - blocks and paths carry the same index set.

Three invariants are nevertheless worth having, because each turns `F` into a
different KIND of quantity.

**(a) THE NORM CLIFF.** `N^n = diag(v_n) S^n` is a PARTIAL ISOMETRY: its
singular values are `0/1`, with exactly `rank(N^n)` ones. Hence

        ‖N^n‖_op = 1 for every n < F,   = 0 for n ≥ F,
        ‖N^n‖_{S_p} = rank(N^n)^{1/p} for every p.

A step function: there is NO decay rate to estimate, and Gelfand's formula
sees `F` only at the discontinuity. With teeth: any envelope
`‖N^n‖ ≤ C λ^n` with `λ < 1` forces `C ≥ λ^{1-F}`, i.e. THE WHOLE OF `F` SITS
IN THE CONSTANT. That is a precise reason why every analytic decay-rate frame
the project has tried has stalled.

**(b) THE NUMERICAL RADIUS - `F` becomes VARIATIONAL.** The numerical range
of a nilpotent Jordan block `J_L` is the disk of radius `cos(π/(L+1))`, so

        w(N) = max_{‖x‖=1} |⟨Nx, x⟩| = cos( π / (F+1) )   EXACTLY,
        F = π / arccos( w(N) ) - 1,

and the numerical range of `N` is exactly that disk. `w` is the optimum of a
concave maximisation (equivalently `λ_max(A)/2` for the Hermitian part), so
it is SDP-representable and every upper bound on it has a dual certificate.
This is the one classical invariant that converts the maximal gap from a
combinatorial extremum into an optimisation with a proof direction.

**(c) THE PSEUDOSPECTRUM - the empty spectrum still encodes `F`.** The
spectrum is `{0}` but the resolvent blows up at the rate of the nilpotency
index:

        ‖(zI - N)^{-1}‖ = |z|^{-F} (1 + O(|z|)),
        r_ε(N) = ε^{1/F} (1 + o(1)),
        F = lim_{ε→0} log(1/ε) / log(1/r_ε).

Setting `z = e^{-1/t}` this is a MASLOV DEQUANTISATION statement:
`t · log ‖(zI-N)^{-1}‖ → F`. So THE (+,×) RESOLVENT COMPUTES THE (max,+)
LONGEST PATH, and the project's three vehicles for `F` - Constructor's
max-plus Kleene star, the Boolean filtration/window indicator, and the
analytic resolvent - are ONE computation carried out in three semirings.

**(d) WHAT ESCAPES THE THEOREM: THE POTENTIAL.** A certificate is not a
unitary invariant. Let `h : Z_P → R` with

        (*)   h(k) - h(k-1) ≥ 1   for every BLOCKED slot k.

Then `h` increases by at least `L` along any run of `L` blocked slots, so
`F ≤ 1 + osc(h)`, and it is TIGHT: `h(k) =` (distance back to the previous
opening) achieves `osc = F - 1` exactly. In multiplicative form
`w = exp(h/t)` this is a SCHUR TEST on `A` (equivalently a bound on the
weighted-shift similarity `D_w^{-1} N D_w`), with
`F ≤ 1 + log κ(w) / log(1/β(w))`, `β = max_{k blocked} w_{k-1}/w_k`,
`κ = max w / min w`; and its tropical limit `t → 0` is exactly Constructor's
max-plus potential inequality. The certificate frame loses NOTHING - only its
ARITY can fail, which is the subject of `potential-arity-ladder.md`.

**(e) WHERE THE NON-INVARIANT CONTENT LIVES.** `ker N^n` is a COORDINATE
subspace (spanned by the `e_k` with `v_n(k) = 0`), so the kernel flag
`ker N ⊂ ker N^2 ⊂ …` is a nested family of SUBSETS of `Z_P`. Its DIMENSIONS
are histogram tail sums - unitary data, circular - while its POSITION
relative to the CRT gear tensor basis is not a unitary invariant at all. That
position is precisely what round 22 measured as the Schmidt-rank profile of
`v_n`, and it is the part that GROWS. So the round-22 and round-23 findings
fit together exactly: the invariants are the histogram; the growth is the
ALIGNMENT OF THE KERNEL FLAG WITH THE GEAR TENSOR BASIS.

**(f) TWO CHECKED NON-GAINS** (recorded so they are not rebuilt).
* MOMENTS REDUCE. `tr(A^{2t}) = Σ_L m_t(L) r_L` where `m_t(L)` counts closed
  `2t`-walks on `Z` of range `L` and `r_L = rank(N^L)` (a closed walk's
  support is an interval, so it demands exactly an `L`-run of blocked slots).
  Verified `t = 1..6` at machine 11. So every trace/moment - equivalently
  every exponential-sum - attack on `λ_max(A)` is a POSITIVE combination of
  the `r_L` ladder that round 21 already computes exactly and scan-free. No
  new information.
* WEYL ON THE MERGE STEP IS VACUOUS. `A_new = A_old + Δ` with `Δ` the edges
  whose right endpoint is newly blocked by `q'`. Measured: the longest run of
  consecutive newly-blocked slots is 1 at every step 11→13, 13→17, 17→19,
  19→23, so `λ_max(Δ) = 1` and the Weyl bound is
  `2cos(π/(F_old+1)) + 1 = 2.85, 2.93, 2.97, 2.99 > 2` - vacuous at every
  step. The merge step's content is WHICH edges are added, never how many.

## 2. WHY IT MIGHT BE NOVEL

Each ingredient has a classical shadow: `diag(w)S` being a weighted shift and
its Jordan form being read off the zero pattern is standard; `W(J_L)` a disk
of radius `cos(π/(L+1))` is Haagerup-de la Harpe / classical; `r_ε ~ ε^{1/F}`
for a nilpotent is Trefethen-Embree textbook. What appears unrecorded is the
IDENTIFICATION and what it decides:

- the Jordan form of a SIEVE's blocked-walk operator is exactly the sieve's
  GAP HISTOGRAM, block-for-gap, so the histogram is a complete similarity
  invariant of the operator;
- consequently a COMPLETENESS statement - every unitary invariant of the
  non-tensor sector's growing part is a function of the gap histogram - which
  turns "spectral methods do not see Jacobsthal" from an observation into a
  theorem;
- the numerical radius identity `w = cos(π/(F+1))` reading a
  JACOBSTHAL-TYPE maximal gap as an SDP-representable variational quantity;
- the dequantisation statement tying an analytic resolvent exponent to a
  tropical longest path IN THE SAME OBJECT, which is what makes three
  independently-built project vehicles literally the same computation.

## 3. PROOF / STATUS

PROVED: the Jordan/permutation theorem and its rank formula; invariant
completeness (a corollary of unitary equivalence); the partial-isometry
statement and the norm cliff; `w(N) = cos(π/(F+1))`; the resolvent exponent;
the potential bound and its tightness; the moment reduction identity.

SCRIPT-VERIFIED (`research/nilpotent_invariants.py`, assertion-gated, 33 s):
* part 1 - `rank(N^n) == Σ_g W_1(g)(g-n)_+` and Jordan multiplicities `==
  W_1(L)` as EXACT INTEGERS at machines 11/13/17/19; the explicit permutation
  is built at machines 11 and 13 and the permuted matrix asserted EQUAL to
  `⊕_g J_g` entry by entry;
* part 2 - the singular values of `N^n` asserted to be `0/1` and
  `‖N^n‖_F^2 == rank` at machine 11, all `n`;
* part 3 - direction-independence of `max λ_max(Re e^{iθ}N)` to 6.7e-16 at
  machine 11 (so the numerical range is a disk), and at machines 11-19 a
  two-sided check with no eigensolver: the path Perron weight gives a SCHUR
  test with `θ = 2cos(π/(F+1))` to 1e-9 at every machine;
* part 4 - the full 385x385 resolvent norm asserted equal to the
  largest-block formula, and the recovered exponent
  `log(1/ε)/log(1/r_ε) → F` monotonically from above (25.782, 25.107, 25.005
  at machine 19 for `ε = 1e-6, 1e-12, 1e-24`);
* part 5 - the additive and multiplicative certificates both asserted EXACTLY
  equal to `F` at machines 11-19;
* part 6 - `dim ker N^n` asserted equal to `P - Σ_g W_1(g)(g-n)_+`, and the
  kernel asserted to be a coordinate subspace at machine 11;
* parts 7-8 - the moment identity asserted for `t = 1..6`, and the Weyl
  numbers printed with the vacuity verdict.

Floats are labelled as such throughout; every claim marked EXACTLY above is
an integer or closed-form statement.

## 4. IMPLICATIONS

Inside the project:
- it closes the round-23 brief with a theorem: the spectrum is empty, and the
  invariants that "replace" it all collapse to the gap histogram, so the only
  non-circular frame is a CERTIFICATE (the potential) - which is where the
  round's second document goes;
- it explains, quantitatively, why the analytic and moment frames stalled:
  the norm sequence is a step function whose only content is the position of
  the cliff, and moments are positive combinations of the run ladder;
- it identifies Constructor's max-plus generator, the Boolean window
  filtration and the analytic resolvent as ONE object in three semirings,
  which means a bound proved in any one of them transfers by dequantisation;
- it kills two obvious next attempts (moment/exponential-sum bounds on
  `λ_max`; Weyl perturbation across the merge step) with numbers.

Outside: an exactly-solvable family where a combinatorial extremal quantity
(a Jacobsthal-type maximal gap) equals a numerical radius and a pseudospectral
exponent, and where the complete unitary-invariant content of a natural
operator is one classical statistic (the gap histogram).

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Jacobsthal / Ziller-Morack `h_2` (`F` is the numerical-radius/pseudospectral
exponent of an explicit operator); requirement (D) and the twin route (this
document bounds nothing - it says where a bound can and cannot come from);
Wall V (its operator-invariant form: invariants are histogram-complete);
Hilbert-Polya (a third negative finite-machine data point, now structural).
Open here: does the numerical-radius SDP admit any tensor-structured dual
certificate at all (the arity ladder measures exactly this); and does a bound
proved in the max-plus semiring dequantise to a usable resolvent bound.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"Jordan form of a weighted shift with 0/1 weights gap structure"; "nilpotency
index sieve maximal gap operator"; "numerical radius of a nilpotent Jordan
block cos(pi/(n+1)) Haagerup de la Harpe"; "pseudospectral radius epsilon^{1/n}
nilpotent Trefethen Embree"; "Maslov dequantisation resolvent longest path
tropical"; "unitary invariants of a partial isometry determined by orbit
lengths"; "prime gap histogram as spectral measure of a sieve operator".
Expected nearest art: Haagerup-de la Harpe (numerical radius of nilpotents),
Trefethen-Embree (pseudospectra of nilpotents), the standard theory of
weighted shifts (Shields). The delta to check is the SIEVE identification -
Jordan blocks indexed by the sieve's gaps, the completeness corollary, and
the maximal gap as a numerical radius.
