# The four killer concepts not yet ruled out: the plan of attack (2026-09-19)

Everything else in the anatomy is proved bounded, proved to act on one column, or proved to act
only through one of these four. Each entry below gives the concept as a statement, what is proved
about it, the attack - a structural experiment paired with the lemma it aims at - the proof route
if the attack lands, and the criterion for stopping. Order of work: 3, 2, 1, 4, because 3 has a
lemma within reach today, 2 depends on it, 1 is where the exponent lives, and 4 absorbs the rest.

## 1. Killer residue vectors

**Statement.** At the stretch (p^2, q^2] the residue vector of p - the shifts (p mod h)^2 at every
gear h up to p - covers every column except the square's.

**Proved.** Killers exist in the residue space from p = 17 and are prime-compatible from p = 29
(entries 100, 101); every stretch to 20000 is alive (entry 100); and, found today, small lifts DO
kill short runs: for the gears up to 17 the square 55^2 starts a 12-column fully struck run, for
the gears up to 59 the square of the PRIME 26987 starts a 40-column run struck by those gears
alone (research/stack/r8/killer_lift_scan.py). So the vector of a real prime does land on killer
sets of fixed length; the realisability barrier is not absolute.

**Attack.** Separate length from position. Let R(p) be the number of columns above p^2 struck by
the gears up to p before the first open one - by the stretch rule, the offset of the first twin
above p^2. A kill needs R(p) at least the stretch length, about p x gap / 3. The position question
is settled (a square can start a covered run); the whole concept is the growth of R(p) against p.
Experiment: R(p) for every prime to 10^6, against p x gap / 3, and the record R(p) against ln^2 p.
Lemma aimed at: none - a bound on R(p) of the form R(p) < stretch(p) IS the window statement for
that stretch. What can be proved is the reduction: `kill needs R(p) >= stretch(p)`, one line.

**Stop criterion.** If R(p) stays polylogarithmic against a linear need, record the exponent gap
(entry 80's exponent 2, now exponent 1 against polylog at the stretch) and close the concept as
"position permits, length forbids, and the length is the conjecture".

## 2. Two-prime products (the plugs)

**Statement.** A run that the base gears (up to p) cannot finish is continued only by squares of
later primes and products of two later primes (`rough_member_form`, proofs/StretchRule.lean).

**Proved.** The form theorem above; squares are one column per prime and upper members
(`square_is_upper_member`, `power_member_side`); a large gear strikes a survivor only through a
prime cofactor (`top_gear_cofactor`).

**Attack.** Pin the plug positions. In the stretch (p_2^2, p_3^2] with base up to p_1 < p_2, the
only members open to the base and struck by p_2 are p_2 p_3 and p_2 p_4 (when p_2 p_4 fits below
p_3^2): the plug law, a corollary of the form theorem and size. Then in a run across the stretches
of p_2, ..., p_k the plugs are exactly the products p_i p_j of near-consecutive primes and the
squares - a sparse set fixed in advance by the primes themselves. Experiment: for real machines,
list the columns of each stretch that the base leaves open, and mark which are plugged (by which
product) and which are twins; confirm every plug is a product of the stretch's own gear with one
of the next two primes.
Lemma aimed at: `plug_law` (provable now). Proof route beyond it: none structural - whether the
plugs cover the base's open set is concept 3.

**Stop criterion.** Plug law proved and confirmed on real machines; the concept then reduces to 3.

**Status (round 99): plug law proved (`plug_law`); reduced to concept 3, which is closed.**

## 3. Multiplicative interleave

**Statement.** For a plug a b to close the base's open column at m, a b = 6m -+ 1, so modulo the
base product P the product of the two primes' residues must hit the residue of that member; a dead
run needs this at every open column of the base in every period of the span, using primes that
exist at the right sizes.

**Proved.** The base's open set is a union of classes modulo P recurring at every common multiple
(`open_at_multiple_of_product`, `open_columns_for_any_gears`); the plugs' form (concept 2).

**Attack.** With the plug law the interleave is not a general covering question but a specific
one: the products p_i p_j of near-consecutive primes must land on the base's open columns. Since
the base pattern above p_1^2 is fixed by p_1 and the plugs are fixed by the primes above p_1, the
structural question is whether a consecutive-prime product can ever sit on an open column of the
base at all, and how the residues of consecutive primes against the base combine. Experiment: for
each base p_1 and the stretches above, the residues of p_i p_j against P at the open columns they
plug - do they show any relation to the base's own residues (p_1 mod h), or are they free?
Lemma aimed at: a constraint on p_i p_j modulo the base from the consecutiveness of the primes -
if none exists (likely: consecutive primes' residues are free, CRT), the concept is analytic
(distribution of products of primes in classes) and is closed as such.

**Stop criterion.** Either a proved relation between consecutive primes' residues and the base
(new mechanic), or the explicit statement that the interleave is Dirichlet-type distribution,
outside the machine's rules.

**Status (round 100): CLOSED.** The plug's position is fixed by the gaps; its partner's residues
show the consecutive-prime correlation (0.124 base-open against 0.155 to 0.166 for non-consecutive
products, entry 106), which is the Lemke Oliver / Soundararajan bias, a Hardy-Littlewood-type law
outside the machine's rules. No machine rule forces or forbids the covering.

## 4. Truncation strays

**Statement.** Open columns in a window = the positive main term W x product of (1 - 2/h) plus the
signed sum over all tuples of their strays; a kill needs the strays to sum to minus the main term.

**Proved.** The identity is exact (inclusion-exclusion); each stray is at most 2^|T| and is fixed by
the window's edges modulo the tuple's period; joint strikes are one class (`joint_strike_class`).

**Attack.** None independent. For a stretch both edges are squares, so every stray is fixed by
(p^2 mod product of T, q^2 mod product of T) - square residues, the location law again. Any
structural progress on 1 to 3 is a statement about which strays can align; the identity itself
adds nothing that is not already in the configuration. Mark as the sieve identity and route all
work through 1 to 3.

**Stop criterion.** Already met: concept 4 is the accounting of concepts 1 to 3, not a separate
mechanism.

## The proofs

Each round on this plan adds its lemma to the kernel before its measurement is recorded, with
the name in the entry, zero sorries, standard axioms. Lemmas targeted: `kill_needs_run` (1),
`plug_law` (2). Concept 3's lemma exists only if a new mechanic is found; concept 4 has none. The
plan closes when each concept is either reduced to the exponent of concept 1 or shown to be
outside the machine's rules.
