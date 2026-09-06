# human.md - the state of the hunt, in plain language

(Harvester-rewritten 2026-09-07 as a current-state snapshot, in the canonical words of
2026-09-06. No round appends: this file is replaced, not extended. The tree of every branch and
verdict is research/proof/theory_tree.md, the map of every blocker is research/proof/the_wall.md,
the parts ledger is research/proof/objects_ledger.md, the method is the theory-tree skill.)

## The five-minute version

Twin primes are two primes two apart, like 41 and 43. Nobody has proved there are infinitely
many. We model the question as a machine: one gear per prime, each gear stamping the number
line on a fixed schedule; a twin prime pair is a slot every gear misses. The Lean kernel has
checked that the conjecture is exactly this: for every machine, an unstamped slot lands inside
its window, the range where an unstamped slot is certified to be a twin (below the square of
the next prime). The proof we want has one shape: a known object we can point at and say "this
is always in the window, because the machine works this way, and nothing the machine does can
prevent it".

The machine has four parts, and since 6 September they have one name each, the owner's line
engine -> valves -> manifold -> exhaust:

- The ENGINE is the primes up to q. It is the part we understand in depth: its pattern repeats
  every q# columns, and inside the window the gaps are the engine's alone.
- The MANIFOLD is the primes between q and q#, studied as a machine of its own on the raw line
  with its own laws. It has two regions: the smooth zone [1, Q], where a pair is open only if
  both members are made of the engine's primes, and the quiet zone (Q, Q^2], where every open
  number is a smooth cofactor times at most one manifold prime.
- The VALVES are the engine acting inside the manifold's open set. In the quiet zone every open
  number is s x P, air (a smooth cofactor s) times fuel (a manifold prime P); each family
  (s, s') of cofactors is one valve, and a charge s x P burns when the engine strikes its air.
  The pure charge, family (1, 1), has no air and never burns: it is the twin primes.
- The EXHAUST is every tier above the manifold. On the window above any cut it does nothing
  new: every strike of an exhaust gear there is a home strike (the gear striking its own prime)
  or an echo of a lower gear's strike.

Older documents say motor for the engine, wheels or top machine for the manifold, clutch for
the valves, and tower or third machine for the exhaust. The Lean namespace stays TopMachine.

## What was found, in order of strength

1. THE ENGINE HAS SAID WHERE THE TWIN SLOT IS. Theorem (E), proved: a column above q is blocked
   by the engine iff it is blocked by the gears up to the square root of its own number, so the
   effective engine at every column is exact. Consequence, measured with no exception: no
   blocked run of length L begins before 3.25 L (1.25 L proved from the certified ladder), and
   from q = 1427 the longest blocked run of the whole prefix is the run from column 1, the gap
   up to the first twin above q, about q/6 long against a window of q^2/6. So the window can
   only be emptied from the bottom, and the whole conjecture inside the window is one
   statement: d_0 <= W, the first twin above q lies below q'^2. Measured to level 33,317, d_0
   sits inside the window by a factor of 10 to 58.

2. THE ENGINE'S LADDER REACHES PAST THE SCAN WALL. The records F(M) = 5, 7, 11, 18, 25, 34, 43,
   58, 88, 91, 103, 118, 145, 161 at q = 7..59 are exact; eleven rungs of the budget inequality
   F(M+q') <= F(M) + q' are certified, the 31 -> 37 rung in the kernel (385 case modules). The
   way one machine's gap spectrum becomes the next machine's is a closed-form recursion (proved),
   and run as an instrument it reproduces F(37) = 88 and F(41) = 91 from machine 23's period
   alone, every check exact; m41 has 8.5 trillion gaps and its record is one of 3,052 fourfold
   fusions among them. Budget slack along the extended ladder: 14, 20, 16, 7, 38.

3. THE MANIFOLD IS A DOMINO MACHINE, AND ITS LAWS ARE PROVED AND IN THE KERNEL. On the raw line
   a manifold gear's teeth sit at 0 and -2, so a strike never comes alone: its partner is exactly
   2 away (kernel, no hypothesis). From that: no gap of exactly 4 between open pairs, ever;
   the parity law, that when every gear exceeds 2m + 1 the record is 2m - (m mod 2), decided by
   the gear COUNT and not the gears' sizes (kernel, as an equality); the loaded record rule,
   F_top = max{L : the cheapest phasing of the gears <= L + 1 costs at most as many dominoes as
   there are gears above L + 1}, proved both ways, 0 mismatches on 6,659 gear sets, in the
   kernel with coprimality as the only hypothesis; the gap census law, the exact count of
   consecutive open pairs at each distance as an inclusion-exclusion product, in the kernel with
   gears positive and coprime only; the cover polynomial, which gives every universal census
   coefficient, the moment vanishing, the eight published record multiplicities (8 of 8) and the
   gap-3 = gap-5 identity in a line each; and the next-opening closed form: the next open pair
   after x is x + mex{(-x) mod g, (-x - 2) mod g}, exact whenever F_top < q' (kernel for gears
   above 2m). The wall's face D, transfer, was never about the manifold: its laws hold for every
   gear set.

4. THE MANIFOLD'S RECORD IS A TWIN GAP AT SCALE. The quiet-zone rule (open iff q-smooth times at
   most one prime above Q) holds on all of [1, Q^2] with 0 exceptions in 45 machines; the zone
   decomposes into families (s, s') whose counts do not depend on q (1,510 families at q = 5,
   Q = 10^4; family (1, 1) = pi_2(10^8) - pi_2(10^4) = 440,107 exactly). The overnight census at
   Q = 10^5 on [1, 10^10] found the same quiet-zone record at the same place for the engines
   q = 5, 7, 11, 13: 924 after 187,907, and 187,907 and 188,831 are consecutive lower twins. The
   manifold's own record is a twin-prime gap in its bottom stratum, and the engine cannot touch
   it. Family (1, 1) is identical across engines (27,411,455 at Q = 10^5; 203,707,420 at
   Q = 3 x 10^5). At Q = 3 x 10^5 the record differs between q = 5 and q = 7 by one pure-air
   pair: 850,500 = 2^2 3^5 5^3 7 is open for the engine {2..7} and splits the twin gap 1,452 into
   151 + 1,301. That is the valves picture in one number: air shortens the manifold's gap, and
   the engine burns it.

5. THE EXHAUST IS CAPPED, MEASURED, AND ITS ARITHMETIC IS A THEOREM. The stack: tier 1 the
   engine, tier 2 the manifold, tier k + 1 the primes in (cut_{k-1}, cut_k] with cut_k tier k's
   period. In the kernel: every gear of tier k + 2 spans tier k and strikes at most two positions
   per stride while the lower pattern repeats in full; the cap, that on (C, C^2] every strike by
   a gear above C is a home strike or an echo, so a pair left open by the primes up to C is a
   twin prime, on every window; and CutMono, that the cuts increase, unconditional for every
   prime q >= 5 (cut_{k+1} > 4 cut_k by a halving induction on Bertrand; false at q = 2, 3, 4,
   and the kernel says exactly why). Measured once (exhaust_1.md): tier 3 obeys the manifold's
   laws in its own parameters on 18,095,756 residues with 0 exceptions; on the windows (30, 900]
   and (210, 44,100] all 57,344 exhaust incidences are home or echo, and every open pair
   receives exactly 2.000 exhaust strikes, both home; the exhaust's first strike that is neither
   comes at exactly p_1^2 (961 = 31^2 at 30^2 + 61; 44,521 = 211^2 at 210^2 + 421), and one
   decade above the window the exhaust is the majority partner (its share 0.0%, then 36.9, 62.8,
   75.0, 82.0, 86.6% by decade at q = 5). Back pressure, the manifold's strikes, is a separate
   object from these non-echo exhaust strikes. The exhaust's own record is the root: its bottom
   stratum is family (1, 1), the twin primes above its top gear.

6. WHAT THE WALL IS, EXACTLY. Every route from the machine's structure ends at one of three
   statements, each now precise. (A) Counting: a proof that uses only how many positions each
   prime blocks is a dimension-two sieve, which cannot reach the window (limit 4.27, window at
   s = 2); the covering-systems method applies to the machine but collapses on an interval for
   the same reason. (D) Transfer: the island witness (an open offset 12 mod 35 past q^2, 0
   exceptions to 200,000) is rare among all phase vectors, but the real vectors are typical
   (failure rate 0.9984 +- 0.0033 for real, locally-square and random), and turning rare into
   never needs equidistribution beyond any theorem: 10^54 covers against a class density of
   10^-30. (O) Order: the interaction between gears needed to cut coverage below the window
   grows with the number of gears (block size 1, 2, 2, 3, 4, 5, 6, 7, 8, 8 at K = 3..12), so no
   fixed-order law reaches all machines. Position facts (corridors, locks, pinning) never see
   length. The first proofs on the adversarial side stand: no ten or fewer primes with
   fixed-separation pairs can block the next prime's window (docs/proofs/20), and the exact
   adversarial ladder A(K) = 2, 5, 7, 16, 22, 28 at K = 1..6 by reasoning, exact to K = 12.

7. THE VALVES ARE NOT YET OPENED, BY RULE. The owner's rule: no valve work until the engine, the
   manifold and the exhaust are fully understood, and the objects ledger is the gate. The first
   step when it opens is fixed (the hybrid): a scratch lane with clean context defines interface
   objects from the three definitions alone; a review lane sorts the 36 refiled interaction facts,
   the wall, theorem (E), the conjugacy and the family decomposition into interface objects,
   flagging every fact measured in the engine's coordinate; then reconcile. Each interface object
   gets a definition, a law with proof, and a closed form where one exists; partial interfaces
   are kept. What is already visible: the split cannot touch family (1, 1). Any valve law must
   explain why (1, 1) is never empty in (Q, Q^2], and the count of that family is the twin-prime
   count, unchanged by q.

## Honest ledger

- The conjecture is accepted as true and every measurement agrees with slack: the record is a
  quarter of the window at every computed engine (F/W = 0.25, flat to q = 53); the walk from
  q^2 lands on a twin within 2 to 79 columns at every prime to 100,003; d_0 sits inside the
  window by 10 to 58 times; the island witness has growing room.
- Proved and in the kernel this fortnight: the manifold's structure, metric, walk, record and
  census laws; the stack, the cap and CutMono. The manifold-and-exhaust library is 310
  declarations across seven files (TopMachine, TopMachineWheel, TopMachineCrt, TopMachineWalk,
  MachineStack, TopMachineRecord, TopMachineCensus), zero sorries, no native_decide, standard
  axioms; build green at 2,248 jobs. The engine's own kernel corpus (the route, the tooth rule,
  the mirror, the merge grammar, the caps, the corridor, the certified rungs and the 385 case
  modules) is separate and larger; no single total is on record for it.
- Written proofs: 23 (docs/proofs/01-23), each with a plain-words opening, a status line and a
  prior-art section. Laws registered with permanent numbers: W1-W85 and X1-X8, with
  W86-W93 landing (research/proof/law_register.md).
- The gate (objects_ledger.md): ENGINE not yet, three structural items (L(M) bounded; the chain
  statement at depths 3 and 4 on the band [15, 36]; a monotone functional of the merge closure,
  one lane running on it now). MANIFOLD yes on paper: no open structural item; what remains is
  non-structural (the moment vanishing into the kernel, W-numbers for document 7, prior art
  for W86-W93). EXHAUST: CutMono in the kernel, measured once, no open structural item; what
  remains is prior art for X10 and X13, the crossover height of the regime law (needs an upper
  bound on a prime gap, a known theorem), and the range form of the redundancy lemma into the
  kernel.
- Refuted and worth knowing: the real teeth are typical in every symmetry, spacing and
  squareness measure; coherence of spacings is a liability; twin gears are the cheapest small
  gears, so de-twinning LOWERS the record; the flank brick is the pair statement itself; no
  bounded-order interaction law exists; the sum rule cannot force depletion; the one-block
  covering inequality is trivial; the pinned letter's lower half fails at 37 -> 41; the manifold
  never saturates (its smooth-zone record is linear in the largest gear).
- Knocking: the one unexplained symptom is the gluability anomaly, the real teeth gluing about
  2.4 times more than the counterfactual family at matched cells. No timing cause on record.
- Open: the one statement, in any of its three faces. Nothing measured says which face gives
  way first.

## What is worth sleeping on

- Family (1, 1) is the same set for every engine, and it alone makes the manifold's record. The
  valves' first law has to be about why the pure charge never runs out over (Q, Q^2]; every
  other family is vented by the engine, and the census shows the venting shortening gaps.
- The engine's last structural item that has never been run: a monotone or contracting
  functional of the merge closure, whose extremes are the records. A lane is on it.
- The crossover height of the exhaust's regime law is the one place on the whole ledger where
  the missing instrument is a known theorem, not the conjecture.

## The map

- research/proof/theory_tree.md: every branch, nested by what spawned it, with verdicts; the
  R4 subtree is the manifold, the exhaust and the closed valves node.
- research/proof/objects_ledger.md: the parts ledger and the gate, one section per object.
- research/proof/the_wall.md: every blocker stated precisely, the faces, the weak points
  tested, and section 5l mapping the faces onto the four parts.
- research/proof/law_register.md: the manifold's and the exhaust's laws with permanent
  numbers and prior-art verdicts.
- research/proof/manifold_census_large.md: the overnight census at Q = 10^5 and 3 x 10^5.
- research/proof/top_machine_lean.md: the kernel ledger for the manifold and the exhaust,
  rounds 32-37.
- docs/proofs/: 23 written proofs with plain-words intros, status, prior art and their
  relationship to the conjecture.
- docs/novel/README.md: the register of results with prior-art status.
- README.md: the glossary, in the canonical words.
