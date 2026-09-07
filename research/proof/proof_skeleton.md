# The proof by construction, as it stands (2026-09-07)

Every line is labelled: PROVED (kernel or written proof on record), MEASURED (zero exceptions
over a stated range, no proof), or THEORISED (not proved; if true, it closes the conjecture).
No step uses the Chinese remainder theorem as a mechanism; where a step is a known fact it is
stated in the machine's terms with its name in brackets.

## The objects

- The **anchor** 2, 3 folds the line: every prime above 3 lies in a slot (n, n + 2) with
  n = 5 mod 6, and a twin prime pair is a slot whose two members are both prime. [PROVED]
- A **gear** g strikes exactly its multiples (**phase zero**). A machine is a set of gears; a
  slot is open under the machine if neither member is struck. [definition]
- The **cuts** are c_1 = 3 and c_{k+1} = (the first prime above c_k)^2: 3, 9, 121, 16129,
  2.6 x 10^8, ... The **section** k is [c_k, c_{k+1}). **Machine** k is the set of primes in
  section k. [definition; S1: every cut lands on a twin slot, PROVED]

## The truths, in order

1. **The cap.** On section k, every strike by a prime of section k or above is either the
   prime itself (a home strike) or a number already struck by a smaller prime (an echo). So
   inside section k only machines 1 .. k - 1 strike genuinely, and a slot open under them is a
   twin prime pair. [PROVED: OneStepE.new_iff, MachineStack.exhaust_home_or_echo; verified with
   0 exceptions at 16 bands to q = 23]

2. **The hand-up.** The twin prime pairs inside section k are exactly the twin-gear pairs of
   machine k (its gears at distance 2), and by 1 they are exactly the slots that machines
   1 .. k - 1 leave open in section k. Nothing is created between sections: open slots below
   become twin gears above. [PROVED as an identity; S7, 0 mismatches at 16 bands]

3. **Saturation.** Phase zero is exactly the statement that the open set of any machine is
   closed under taking divisors and under multiplying by numbers free of the machine's gears.
   No free-phase machine has this; it is the whole content of "gears strike multiples".
   [PROVED, V14; 0 violations in 107,750,211 tests]

4. **Twin gears are cheap.** Two gears at distance 2 collide (strike the same slot) earlier
   than any other pair, at (g + 4)/3 in the column coordinate, and their joint deficit grows by
   exactly 4 per common period. A machine's twin gears block fewer slots than the same number
   of unrelated gears. [PROVED: ArcFloor.collision_add; file 21]

5. **The section machines are equal-weight.** Every machine k removes exactly half of the
   numbers that reach it (its Mertens product is 1/2 to within 0.03 from k = 2), and about a
   quarter of the slots. The construction cuts the primes into machines of equal strength.
   [PROVED as Mertens; S5 measured to q = 23]

6. **The start of a section is rich.** Just above c_k the slots open under machines
   1 .. k - 1 are twins at up to five times the rate the machines' individual densities would
   give, falling to about 0.85 of it by height c_k^4. Mechanism: at height x only the gears of
   machine k - 1 below x / c_{k-1} have struck anything yet. [MEASURED, S6; 16 bands]

7. **The base.** Section 1 = [3, 9) contains the twin-gear pairs (3, 5) and (5, 7). Every
   section ever computed (to c_5 = 2.6 x 10^8 along the chain from 3, and every chain from
   every base to q = 23) contains at least one twin pair. [PROVED for the base; MEASURED for
   every computed section]

## The step

8. **THEORISED (the step).** For every k, machines 1 .. k - 1 leave at least one open slot in
   section k. Equivalently, by 2: every machine k has a twin-gear pair. Equivalently, in plain
   words: between every cut and its square along the chain there is a twin prime pair.

## The conclusion

9. If 8 holds, induction from 7 through 2 gives a twin-gear pair in every machine k, hence a
   twin prime pair in every section, and the sections tile [3, infinity). Therefore there are
   infinitely many twin primes. [PROVED given 8]

## What 8 needs, and what it cannot be

- The step cannot be a density statement about machines 1 .. k - 1 on section k: any argument
   that uses only how many slots each gear strikes is a two-dimensional sieve and stops at
   about four sections' worth, never one (face A). [PROVED in the sense of the sieve limit;
   confirmed inside the construction by the counterfactuals below]
- The step cannot follow from the valve laws (imprint, onset, port, inventory, ember) alone:
   a free-phase machine obeys all of them with no open slot in 60 turns. [PROVED, V2, V12]
- The step cannot follow from saturation alone: a saturated machine with the section's own
   twin primes added as gears obeys everything on record and has no open slot. What excludes
   that machine is only the cut, and saturation plus the cut is the definition of machine k.
   [PROVED, V17]
- So 8 needs a property of machine k that is (a) inherited by machine k + 1 through the
   squaring of the cut, (b) not a density, and (c) sufficient for a twin-gear pair. The twin
   gears themselves satisfy (a) only if 8 already holds. Every invariant proved to survive the
   squaring so far (5, and the quarter law for slots) is a density. [the exact shape of the gap]

10. **THEORISED (the mechanism candidate).** The inherited property is the cheapness of 4
    compounded with the richness of 6: a machine whose gears include twin pairs strikes the
    start of the next section less than its size predicts, by an amount that is structural
    (forced by the collision law at every twin gear) rather than statistical, and that amount
    is enough to leave one slot open before the machine's gears have all engaged (height
    c_k x g for its smallest gear g). Not measured. If true it is 8.

11. **THEORISED (the alternative).** Saturation gives the open set of machines 1 .. k - 1 a
    multiplicative structure on section k (closed under divisors; closed under multiplying by
    rough numbers below the cut). A slot (n, n + 2) both of whose members are rough is a twin.
    The property sought is one that this structure carries and the parity-defined adversary
    (even number of prime factors) does not, since that adversary is also multiplicative and
    also kills the count: the Liouville-signed slot census is 0.1% against an 8% pure share
    [MEASURED]. Whatever separates the real open set from the parity adversary on a section is
    the invariant of 9(b). Not identified.

## Where to think

The gap is one sentence, 8, and its shape is fixed by 9: a non-density property of a
section's primes that the next section inherits when the cut is squared, and that forces two
of them to sit at distance 2. Everything above 8 is proved; everything below it follows.
