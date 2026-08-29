# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 27, 2026-09-01. Current-state snapshot; history in git and
docs/proof-search/archive/.)

## The five-minute version

We model twin primes as a machine: one gear per prime, each blocking positions on a fixed
schedule; twins are the positions every gear misses. Strategy: prove no set of gears can
ever block an entire window. Round 26 found the machine's exact record law. Round 27 used it
to do something new in kind:

WE LEFT THE MAP. Every previous verification lived inside the "corpus" - the ladder of
machines small enough that someone, somewhere, had computed their records. Round 27 decided
the target inequality at 53->59, A STEP BEYOND THE END OF THAT LADDER, where no record was
known at all - by computing on a machine 500 BILLION TIMES SMALLER than the one the answer
is about. The record law made the question small. The answer: true, with room (203 against a
budget of 204... and the new machine's record is now bracketed [161,178] where nothing was
known before).

THE DERIVATION IS DOWN TO A FINITE LIST. The remaining obligation (the uniform statement)
now decomposes into: a depth cap - PROVED this round, machine-free (order at most 5,
everywhere, from gears 5 and 7 alone; and its six exceptional classes turned out to be an
object discovered five rounds ago wearing different clothes); a "triple inequality" at depth
3 - verified at nine steps, certified by exact certificates at six, holding even at the one
bad step once padding is separated; and per-depth analogues for depths 4-6 only, because the
cap closes the list. The residual quantity the derivation must control was measured to be
BOUNDED BY A CONSTANT (between -3 and 4, no trend, while the naive bound grows linearly).
One constant. That is what remains.

THE KERNEL CAUGHT UP: the optimisation certificates from round 26 are now inside the proof
kernel - one ladder step exists in two independent kernel forms, one resting on nothing but
arithmetic. The mirror lever's counting core ("if it happens at most once, it happens zero
times") is kernel-checked too.

YOUR SORT-STEP IDEA, closed honestly: proved as a theorem (the phase order really is an
odometer with exactly 2n gap sizes), then found to be known in mechanism in the literature,
and - the decisive test - shown insensitive to the teeth: the pattern is universal, so those
coordinates discard exactly what the record depends on. A clean, gated closure. But the
closing experiment found something better: among all counterfactual machines (same gears,
teeth moved), THE REAL TWIN MACHINE IS AN OUTLIER WITH UNUSUALLY SMALL RECORDS - the first
measurable way the true machine is special, and special in the direction the conjecture
needs. Whether that advantage extends to the budget slack is now a named question.

THE PAPER: every last page verified first-hand; the submission memo is written and ON YOUR
DESK (docs/novel/unit1-submission-memo.md) - it honestly does NOT recommend submitting
(audience reality: the anchor paper has one citation in nine years), lays out venue options,
suggests writing to the original authors directly, and leaves the AI-disclosure decision
where it belongs - with you. A companion note contains a first-ever evaluation of the next
function up the family.

## Honest ledger

Every lane self-corrected again - hardcoded values found in a lane's own script, an
extrapolated prediction refuted by its own computation ("I extrapolated a pattern instead of
computing it"), a possibly-corrupted build discarded and redone on principle ("a kernel
claim on a possibly-raced olean is not a kernel claim"), and the manager's own pre-registered
bound superseded by a sharper one the data insisted on. Ten gates green under the manager's
check. Nothing unverified reached the record.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
(A), (B), (C): closed. (D): true at every computable step AND one step beyond the corpus;
ten LP-certified rungs; hypothesis-free kernel rungs through 29->31; the uniform obligation
reduced to a FINITE lemma list with one constant (Delta_3) at its centre. Round 28 is
briefed: derive the constant.
