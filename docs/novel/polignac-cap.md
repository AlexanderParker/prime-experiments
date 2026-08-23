# polignac-cap - 12 is the absolute ceiling on literal chains over ALL even gaps

## 1. WHAT IT IS

Plain language: the literal cap (docs/novel/literal-cap.md) says that for TWIN pairs, a run of
consecutive same-gear kills built from exact tooth spacings never exceeds 6 members. This
finding generalises that to EVERY even gap d (the Polignac family): for the pair (n, n + d),
the analogous literal-chain cap depends only on gcd(e, 105) where d = 2e - eight possible
values - and is 6 for six of the eight classes, 10 for gcd = 15, and 12 for gcd = 105. So 12
is the absolute ceiling over all Polignac configurations, for every gear, forever.

Definitions (harvester's halved-coordinate frame). For gap d = 2e, position n denotes the pair
(2n+1, 2n+1+2e). Gear q blocks n = 0 and n = -e (mod q). A LITERAL CHAIN is a maximal run of
consecutive frame-admissible q'-kills - kills that survive gear 3 - all of which are exposed
to gears 5 and 7. Gear 3 FILTERS the candidate list rather than breaking a run: a
3-inadmissible kill is skipped and the run continues across it. (Modelling gear 3 like gears
5/7 gives wrong caps 2/4 - a recorded modelling trap the kernel caught.) The exposed set E_e
for gears {3,5,7} satisfies |E_e| = prod over q in {3,5,7} of (q - r_q), r_q = 1 if q | e
else 2 - the standard Hardy-Littlewood local admissibility count, which is why e's
interaction with {3,5,7} is summarised by gcd(e, 105).

Precise form (kernel-checked): with `capOK e L B` the Boolean scan asserting that no literal
chain exceeds L over every invertible gear class t mod 105, every exposed start, and both
parities, the eight class theorems are

    cap_gcd_1   : capOK   1  6 26 = true        gcd(e,105)   1  5  7  3  21  35  15  105
    ...                                          cap          6  6  6  6   6   6  10   12
    cap_gcd_105 : capOK 105 12 26 = true

and the roll-up (`proofs/PolignacCap.lean`):

    def capOf : ℕ → ℕ
      | 1 => 6 | 3 => 6 | 5 => 6 | 7 => 6 | 15 => 10 | 21 => 6 | 35 => 6
      | 105 => 12 | _ => 0

    theorem capOf_le_twelve : ∀ g ∈ [1, 3, 5, 7, 15, 21, 35, 105], capOf g ≤ 12

Every even gap d falls into one of the eight classes, so the cap statement covers all of
Polignac's family. Each cap is sharp - the scan fails at cap - 1 (checked numerically). The
two rows exceeding 6 are exactly where e absorbs the small gears (5 and 7 both divide e),
enlarging the exposed set. gcd = 3 - the d = 0 (mod 6) case, the densest Polignac gaps -
still caps at 6. The twin row (e = 1) reproduces the mod-35 literal-cap table, cross-
validating the frame change.

## 2. WHY IT MIGHT BE NOVEL

- It is, to the corpus's knowledge, the first UNIFORM structural statement over the entire
  Polignac family in this ledger or in any searched literature: one bounded quantity (12)
  controlling every even gap at once, with the exact dependence on d reduced to gcd(e, 105)
  and computed for all eight classes.
- The published Polignac literature (Zhang, Maynard, Polymath8) is of a different KIND: those
  results prove INFINITUDE of prime pairs at some bounded gap - existence statements about
  primes, obtained by sieve weights, and to date not for any specific d. This finding is a
  finite, unconditional, structural CAP on run configurations in the small-prime wheel,
  holding for EVERY d and every gear, and it says nothing about infinitude. The delta is
  total: neither statement implies the other; they are about different objects.
- The reduction "the cap depends on d only through gcd(e, 105)" is itself a clean structural
  fact - the analogue for run-caps of the classical fact that the HL singular series' small-
  prime factor depends on d the same way.
- Formal strength: all eight `cap_gcd_*` theorems AND `capOf_le_twelve` depend on NO AXIOMS
  AT ALL (empty axiom footprint - pure kernel computation, no propext, no choice, no
  Quot.sound, no native_decide). Kernel-checked classification results at this level of
  hygiene are rare in number theory.

Classical shadow, stated honestly: |E_e| = prod (q - r_q) IS the standard Hardy-Littlewood
local count and is claimed as known, not novel. The cap values and their classification are
the candidate-novel content.

## 3. PROOF

Status: KERNEL-CHECKED (all eight class caps and the ceiling); SCRIPT-VERIFIED (sharpness at
cap - 1 for each class; the independent reproduction of all eight spectra row-for-row before
formalising).

- Lean (round 17; ledger at 1252 jobs then, 1254 now; zero sorries, zero warnings):
  - `proofs/PolignacCapCore.lean` - frame definitions (`inE`, `scan`, `capOK`) and the
    reusable coprime-multiplier lemma `exists_mul_mod_eq` (surjectivity of coprime multiples
    mod n; standard three axioms; the prerequisite of the single-cycle reduction, proved for
    reuse though the cap did not need it).
  - `proofs/PolignacCap{1,3,5,7,15,21,35,105}.lean` - one gcd class each, e.g.
    `theorem cap_gcd_1 : capOK 1 6 26 = true := by decide +kernel` and
    `theorem cap_gcd_105 : capOK 105 12 26 = true := by decide +kernel`. One module per
    class because eight heavy `decide +kernel` calls in one file exhaust one process
    (measured >20 min stalled vs 17-60 s each when separated).
  - `proofs/PolignacCap.lean` - `capOf` and `capOf_le_twelve` (by `decide`).
  - Axiom audit (`proofs/AxiomCheck.lean`): all eight `cap_gcd_*` and `capOf_le_twelve`
    depend on NO axioms at all.
- Verification discipline: harvester's table reproduced independently (all 8 spectra) before
  formalising; each cap checked numerically sharp (scan fails at cap - 1); the twin row
  cross-checked against `research/literal_cap_gap_d.py` / the LiteralCap mod-35 table.

Scope caveats: as with the twin case, the cap covers LITERAL chains only - padded structure
is bounded separately (count/onset arithmetic, docs/novel/corridor-law.md). "cap" bounds runs
of admissible kills in the {3,5,7}-wheel; it is not a statement about primes.

## 4. IMPLICATIONS

Inside the project:
- The route is a theorem schema over all even gaps: part (B) of the factorisation is now
  universal, not twin-specific. Combined with harvester's measurements (twins at the 13.3rd
  percentile of difficulty in their own family; F ranging 30..75 across the 2,880 coprime
  differences at gears <= 13), "prove it for twins" is quantifiably the easy end of "prove it
  for every even difference".
- The gcd(e,105)-reduction plus |E_e| = prod (q - r_q) ties the machine's frame directly to
  the Hardy-Littlewood singular series at the small primes, cross-validating the halved frame.
- The eight-module + empty-axiom-footprint pattern is the template for future large kernel
  scans (recorded in the formalist infrastructure notes).

Outside: a kernel-checked, axiom-free classification theorem attached to the Polignac family
- usable as a benchmark example of large `decide +kernel` verification in Lean 4, and as a
structural datum (the 6/10/12 spectrum) for anyone modelling gap constellations across even
differences.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Polignac's conjecture: gives structural (not existential) data for every even d;
  emphatically does NOT prove or approach infinitude for any d.
- Zhang / Maynard / Polymath8 bounded gaps: disjoint statement type (see section 2); this cap
  neither uses nor strengthens their sieve results.
- Hardy-Littlewood k-tuple conjecture: the |E_e| identity is the HL local factor; the cap is
  a new kind of local datum (max run length rather than density) the HL framework does not
  compute.
- Project Conjecture 6 / paired-Jacobsthal h_2 (Ziller-Morack): the per-d machinery here is
  what made harvester's exact h_2 values and the "why is 13 extremal?" question reachable.

## 6. PRIOR-ART CHECK

Date: 2026-08-23. Engine: Claude WebSearch (web). Searches actually run:

1. "Polignac conjecture partial results even gaps Zhang Maynard bounded gaps "de Polignac
   numbers" density" - the live literature: Zhang (2013, gap < 7x10^7), Maynard (600),
   Polymath8b (246; 12 and 6 under EH/GEH), Pintz on de Polignac numbers having positive
   density (arXiv:1305.6289), Granville-Kane-Koukoulopoulos-Lemke Oliver on densities of
   Dickson m-tuples (arXiv:1410.8198). ALL are infinitude/density-of-d results via sieve
   weights - none states any structural cap on wheel configurations per d. The delta is
   the statement type, not the strength.
2. "maximal run consecutive integers coprime residue classes admissible tuples mod 210
   classification" - admissible-tuple size sequences (OEIS A023193) and coprime-gap papers
   (arXiv:2007.01808); nothing per-d, nothing on run caps.
3. "Jacobsthal function maximal gap reduced residues primorial g(210)" - Jacobsthal/Hagedorn:
   the classical wheel-gap function; grows with the modulus, no per-d classification, no
   analogue of a universal finite ceiling.
4. "Montgomery Vaughan distribution of reduced residues gaps Erdos conjecture" - Montgomery &
   Vaughan (Annals 1986) on moments of totative gaps; statistical, not configurational.
5. "Holt Rudd "Eratosthenes sieve" cycle of gaps primorial constellations" - Holt & Rudd
   (arXiv:1408.6002, arXiv:1510.00743): nearest frame (cycle-of-gaps recursion, constellation
   populations across all gap sizes, evidence toward k-tuple/Erdos-Turan questions). They
   track POPULATIONS of constellations; no run-cap theorem, no gcd(e,105) classification, no
   formal verification.
6. ""mod 35" ... kernel Lean formalized" and ""37, 53, 83, 127, 157, 173" ... "mod 210"" - no
   published formalisation of any comparable wheel classification found; no occurrence of the
   cap spectrum {6,10,12} attached to gcd(e,105) anywhere searched.

Nearest published results overall: Zhang/Maynard/Polymath8 (different statement type -
infinitude, not caps); Hardy-Littlewood local admissibility counts (the |E_e| identity here
is exactly this, and is KNOWN); Jacobsthal's function (wheel gaps, growing); Holt-Rudd
constellation dynamics (nearest frame, no cap).

VERDICT: NOVEL AS FAR AS SEARCHED - for the universal cap capOf_le_twelve, the 6/10/12
spectrum, and the reduction of the cap's d-dependence to gcd(e, 105). PARTIAL OVERLAP,
explicitly conceded as KNOWN: |E_e| = prod_{q in {3,5,7}} (q - r_q) is the standard
Hardy-Littlewood local factor and is used here as a cross-check, not claimed. No overlap with
Zhang-Maynard-Polymath beyond the shared word "Polignac": those results concern infinitude of
prime pairs, this one a finite structural cap - the delta matters and is recorded.
