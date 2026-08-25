# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 23, 2026-08-25. Current-state snapshot; history in git and
docs/proof-search/archive/.)

## The five-minute version

We model the twin prime question as a machine: one gear per prime, each blocking positions on a
fixed schedule; twin primes are the positions every gear misses. The conjecture fails only if
some set of gears could block an entire window, so the strategy is mechanism exhaustion - show
no combination can.

ROUND 23 IS THE ROUND THE WALL TURNED OUT NOT TO EXIST. Last round I reported a specific
obstruction - two independent methods failing at the same step and depth - as the project's
sharpest open object. It was three things, none of them mathematics: a NAMING COLLISION (two
different steps had been given the same label), A BUG (a search declared success before checking
the end of its own window), and A CERTIFICATE BUILT ON THE WRONG VARIABLE. All three are now
resolved, and the failing step passes comfortably.

What replaced it is better. The target inequality (D) now HOLDS AT EVERY STEP WE CAN COMPUTE,
through the 47->53 step, by arithmetic alone. A certificate keyed on the last THREE GAP SIZES -
no residues, 14,368 states - is EXACT at all seven steps we can check, including both that had
defeated every previous method. The insight was one sentence: they had been refining the wrong
axis. And a companion result explains why: any certificate that FORGETS EVEN ONE GEAR proves
nothing at all, which is why every modulus tried had failed and why no modulus could have worked.

## Where the proof stands

- (D) is machine-verified with ZERO assumptions at four consecutive gear steps.
- The 17->19 step now has TWO INDEPENDENT KERNEL PROOFS - one from the merge law, one from a
  37-number certificate that verifies in seconds where the direct check needs a 1.6-million-slot
  scan. Formalising it revealed three facts nobody had noticed: the certificate is a palindrome,
  it is supported entirely on one distinguished gear, and it signs by 17.
- The depth-quantifier-free form of (D) now exists at every scanned rung, and its complexity does
  NOT grow with the machine.
- A new transfer trick computes properties of machines SIX GEARS BEYOND what can be scanned.
- First computation: the record gap for the machine of primes up to 47 is exactly 118.

## The wall, honestly

Everything above is PER-MACHINE. We can now certify every step we can compute, and we still
cannot certify all steps at once. The generator found last round is arity-free but not
machine-free, and the machine-free system was measured SATURATING - it stops improving no matter
how much detail is added. The remaining obligation has been narrowed to a single object: A
TWO-GAP STATEMENT. Concretely, a certificate search that stalls on its own reaches the answer
the moment it is handed ONE INTEGER about the machine. Round 24's question is what
machine-independent fact could replace that integer.

## A second result, nearly published

Separately from the twin route, the lane working adjacent problems has taken a function that had
NO PROVED UPPER BOUNDS OF ANY KIND - an empty ladder, zero published attempts in nine years - and
produced explicit ones with every constant stated. That unit is publication-ready pending three
verification items. The lane also found where the tools were hiding: not in the sieve textbooks
(every fundamental lemma there carries unspecified constants) but in the explicit Goldbach
literature, because sifting n and N-n has exactly the same structure as our problem. And it
opened a new question - the LOWER ladder is emptier still, and unlike everything else here, the
parity barrier provably does not obstruct it.

## Honest ledger - every lane corrected itself this round

- The census bug that created last round's "wall" was found by the lane that wrote it, proved by
  a second lane, and diagnosed to the exact line by a third.
- One lane retracted its own "most interesting paragraph" from last round: a claim stated as an
  impossibility theorem is actually an open problem, and false as stated.
- One lane's referee pass on its own paper found five defects - and one CORRECTION IS SHARPER
  THAN THE ERROR: a published hypothesis we had assumed conservative turns out to be exactly
  sharp.
- A citation that passed through my own hands was a chimera - a theorem number that does not
  exist, conflated between two books. Nine citation fixes in total. New standing rule: theorem
  numbers are the fastest-decaying citation of all, so a numbering sweep is now part of every
  referee pass.
- A data-integrity flag was raised against one of our own earlier scans (an opening count that
  disagrees with a closed form matching everywhere else) - flagged, not buried, and scheduled.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
(A), (B), (C): closed. (D): holds at every computable step, kernel-proved at four, and reduced
to one machine-free obligation - the two-gap statement.
