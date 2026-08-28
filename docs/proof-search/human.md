# human.md - the state of the hunt, in plain language

(Manager-rewritten after round 24, 2026-08-29. Current-state snapshot; history in git and
docs/proof-search/archive/.)

## The five-minute version

We model the twin prime question as a machine: one gear per prime, each blocking positions on
a fixed schedule; twin primes are the positions every gear misses. The conjecture fails only
if some set of gears could block an entire window. Everything is proven per machine; the one
missing piece is a single machine-independent fact - WHY THE MACHINE NEVER PUTS ITS TWO
BIGGEST GAPS NEXT TO EACH OTHER (the "two-gap law").

Round 24 was run across a three-day API outage and a memory crisis that froze the computer -
and still delivered on both fronts:

THE MATHEMATICS. The two-gap question was answered in both directions. Negative: we PROVED
that no amount of clever optimisation can extract the two-gap law from the machine-free
facts we currently have - the constraint system is already being read perfectly; it simply
does not contain the law. (This closes the "try a fancier solver" direction permanently, by
theorem.) Positive: the certificate machinery, upgraded, now proves the target at three steps
NEEDING NO MACHINE-SPECIFIC FACT AT ALL - and when weakened it stalls at exactly the two-gap
statement, confirming that is the one and only missing fact. The two-gap object also turned
out to have its own generator (a second exact algebra, verified 6 for 6), and a chain now
exists on paper: each machine's certificate can generate the INPUT for the next machine's -
if the chain runs, the ladder feeds itself. Round 25's job is to run it.

A FIFTH LADDER RUNG entered the computer-verified record with zero assumptions (the 23->29
step - 37 million cases checked inside the proof kernel, resting on one standard axiom). The
side-project paper got stronger: its headline exponent fell from 19 to 15 (and 15 is proved
to be that method's floor), a constants dispute was settled from first-hand sources, and the
lane proved the first-ever LOWER bound using the paired structure - while retracting two of
its own earlier claims that its own data did not support.

THE ENGINEERING. The frozen computer was a livelock: six proof workers, each needing ~5 GB,
launched on a 16 GB box - because the cost model had no memory column. Ten and three-quarter
hours produced literally zero finished work. Serialised to two workers by a small guard
script (suspend, don't kill; trim pages; scale by free RAM), the same work finished in 3.6
hours. The fix that made each worker 22x faster was measured properly this time - including
the finisher catching its own draft being wrong by 13x on the memory figure. Cost models
carry a memory column now; the parallelism budget for these scans is 2.

THE PROCESS. The new gate-check rule (manager re-runs every lane's headline verification
from a clean process before writing the round up) ran for the first time and immediately
earned its keep: it caught one lane having corrected a claim but not the gate that checks it
- and the rerun settled an open value as a side effect. The mandate audit found one lane
(Lateral) had drifted into live-route support over three rounds - the briefs came from the
manager, so the drift is the manager's, and round 25 restores the lane to its own territory.

## Where the proof stands

- (D) - "adding a gear never stretches the record gap by more than the gear" - holds at
  every computable step through 47->53, is kernel-proven with zero hypotheses at FIVE
  consecutive steps (11->13 through 23->29), and one step (17->19) has two independent
  kernel proofs by unrelated methods.
- The obligation is one fact: the two-gap law. Both its negative boundary (what cannot prove
  it) and its positive machinery (what could) are now mapped precisely.
- The next vehicle is ready: a dictionary-based certificate that turns the next rung into a
  finite graph check, plus the self-feeding chain above.

## Honest ledger

- Two agents were lost mid-round (API limits); both rounds were completed from disk state -
  one by a successor that independently re-verified everything, one by the manager filing
  the lane's own completed drafts. Every gate was re-run clean afterwards.
- The round's corrections: a claim/gate desync (caught by the gate-check), a 13x-wrong
  memory estimate (caught by measurement), a timing artifact from a killed run (caught by
  the lane's own standing rule), two retracted claims in the paper lane (caught by its own
  referee pass), and a label that lied about a scan's range (the scan itself was right).
  Nothing reached the record unverified.

## The map

Route: twins infinite <=> no machine ever covers a window (kernel-checked iff).
(A), (B), (C): closed. (D): five kernel rungs, every computable step verified, one named
machine-independent obligation - the two-gap law - with a chain strategy and a derivation
target (mirror law + kill-spacing + survivor identity) queued for round 25.
