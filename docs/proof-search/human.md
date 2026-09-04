# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 31, 2026-09-04. Current-state snapshot; history in git and
docs/proof-search/archive/.)

## The five-minute version

We model twin primes as a machine: one gear per prime, each blocking positions on a fixed
schedule; twins are the positions every gear misses. The target: prove that adding a gear
never stretches the record blocked-run past the budget (old record plus the new prime), for
ALL machines at once. That inequality is the target, measured true at every computable step;
it is not a law.

ROUND 31 WAS ONE LEMMA, AND IT LANDED IN THREE PARTS:

1. HALF OF THE CRUX IS NOW A THEOREM. The open question was whether the longest "legal word"
   L stays bounded. Legal words come in two kinds: bare (built from the two basic letters the
   new prime allows) and padded (using the prime itself as a letter). For bare words there is
   now a proof, checked by the kernel: a bare word must alternate its two letters, and any
   realised word has to thread the teeth of gears 5 and 7, so its length is capped by a number
   that depends only on the new prime's residue mod 210, and that number is never more than 5.
   On 28 of the 48 possible residues the cap is 2. At machines 13 and 17 this alone decides L
   = 1 with no other gear consulted.

2. THE OTHER HALF WAS THE WRONG QUESTION. The lateral lane proved a second theorem: every
   legal word sits inside a window no wider than the next record, and every letter is at
   least a third of the new prime, so L is at most about twice the ratio (next record) / (new
   prime), plus one. That ratio grows slowly across the corpus, so L is allowed to grow with
   it. "L is bounded by a constant" is therefore probably false in the limit, and it was never
   what the argument needed. What the argument needs is a single inequality between the record
   and the prime, of the form 8F ≤ q′² minus lower-order terms. The corpus satisfies it from
   machine 23 onward with a margin that widens (the record is a factor 2.4 to 3.3 inside the
   bound at every step). This closure rests on one still-open piece: the residual for padded
   letters, which misbehaves at exactly one place in the corpus (machine 31).

3. WHAT IS LEFT IS SHARP. The padded half of L grows (0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3
   across the corpus). Padded letters are not invisible to any gear; what makes them hard is
   that the alphabet grows with the record, roughly 3F/q′ letters, and beyond machine 53 the
   small gears can no longer refute every long word. Bounding that half means the cover half
   of the sieve at full depth on the non-bare alphabet only, or proving the record-versus-prime
   inequality directly.

Also: the manager's framing "letter size governs L" was tested on the counterfactual family and
refuted; the two lanes' independent computations of the 28-class set agree element for element
with the kernel's; one lane forgot to pre-register and said so.

## Honest ledger

Three lanes on Opus 5, one hour, no outage, no compute drama. One lane refuted three of its own
pre-registered predictions including its count of the class set (24, actually 28), one was
redirected mid-round by the human's call and recorded its original pre-registration as
superseded rather than scored, one skipped pre-registration and filed that as a verdict against
itself. The manager's own contribution this round (the letter-size framing) was wrong and is
recorded as such; the spectrum bound, which the manager suggested in weaker form, was sharpened
by the lane to a parity-refined bound that is tight at three machines.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
(A), (B), (C): closed. (D): true at every computable step and beyond the corpus; eleven
certified rungs. The uniform obligation: a per-letter residual bounded on literal letters
(measured, at most 4), one padded exception at machine 31 (open), the depth-2 slack (measured
9 to 49, unproved), and the padded half of L, equivalently the record-versus-prime inequality
8F ≤ q′² minus lower-order terms, which the corpus satisfies with room from machine 23 on. The
bare half of L is proved and in the kernel.
