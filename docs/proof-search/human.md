# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 25, 2026-08-30. Current-state snapshot; history in git and
docs/proof-search/archive/.)

## The five-minute version

We model the twin prime question as a machine: one gear per prime, each blocking positions on
a fixed schedule; twin primes are the positions every gear misses. The strategy is mechanism
exhaustion: prove no combination of gears can ever block an entire window.

Round 25 was the first run with the new setup - worker agents on the cheaper model, the
manager on the stronger one, every claim forced through verification gates. Verdict: it
worked. Every lane corrected itself at least once, the gates caught everything, and the round
produced more per token than any before it.

THE LADDER: seven steps of the key inequality are now computer-verified. Two were added this
round in FIVE MINUTES EACH by a new method - down from a 170-hour estimate - because the lane
noticed the proof only needs a small, slowly-growing dictionary of window shapes, not the
astronomically growing full pattern.

THE CHAIN: the certificate machinery now runs with NO big computations at all - it asks
small yes/no questions it generates itself, and one step's answers produce the ingredients
for the next step's questions. It certified a new step this way, self-contained. Where it
stalls, the reason is measured precisely: not mathematics, but the cost of answering single
questions at bigger machines.

A BELIEF DIED AND SOMETHING BETTER REPLACED IT: a computation left running from last round
quietly finished and disproved our own standing claim about how gap-merging chains are capped
(a 5-link chain exists where we believed 3 was the max). The repair found the real criterion -
and at both machines where it was computed exactly, the new criterion doesn't just bound the
answer, IT EQUALS IT. If that pattern holds, we may have found the exact law of the machine's
records - now a registered conjecture under test.

THE TARGET SHARPENED: the "two-gap law" framing was corrected (the binding window can be
deeper than two gaps at bigger machines - an assumption died there too), and the missing
argument is now stated in one sentence: transfer a first-moment bound that already has a
huge measured margin (polylog versus linear) into an unconditional statement. That is the
manager-level derivation target for round 26. A new symmetry lever helps: gap patterns come
in mirror pairs, so the worst-case configuration occurs an EVEN number of times - meaning
proving "at most one" proves "none". The worst case is already verified absent at all six
computable machines.

THE PAPER: the headline exponent fell again, 15 to 8.04 - last round's miss had a named
cause (we priced the book's theorems but never its propositions, and the propositions were
the explicit ones). A thesis lead was obtained and closed negatively first-hand; two
independent leads turned out to be literally the same equation in two books; and we caught
an arithmetic slip in the published book itself (in our favour).

## Honest ledger

Every lane on the cheaper model made mistakes - and every one was caught by the safety net,
none by luck: a bug caught by another lane's census, a silent no-op caught by a shadow gate,
a never-measured claim exposed, a wrong prediction falsified by the lane's own script, an
open cell closed by exhibiting an exact witness rather than by a failed search. Two false
beliefs of the project itself (the fuel cap, the fixed two-gap depth) were destroyed by the
project's own machinery. This is the system working exactly as designed.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
(A), (B), (C): closed. (D): SEVEN kernel rungs; every computable step verified; the
obligation now precisely stated as a first-moment transfer over the qualifying-window
family, with a candidate exact law (Q* = F) under test and a parity lever on the endpoint.
Round 26 is briefed: the derivation attempt moves to the manager.
