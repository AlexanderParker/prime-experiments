# human.md - proof-search snapshot

## ELI5 SUMMARY (rewritten each round)

We are trying to prove twin primes never run out, using the gear machine. The plan: assume some
stretch of numbers has NO twins (call it condition X) and show the machine cannot actually do
that. Round 1 established, with exact bookkeeping: if X ever happened, the machine would have no
slack at all - every prime in that stretch must sit next to a composite (a fake twin), and
double-composite slots must show up at least as fast as primes, starting from the very first slot.
Real stretches break that rule immediately: below about 400 there literally are not enough
composites to go around, and even past that, the bottom of every window starts with too many
primes packed together. So if X can ever happen, it must be arranged in the lowest band of the
window - that is now the single place the proof fight happens.

We also learned what does NOT work, saving future effort: twin gears sharing teeth wastes some of
the machine's kills, but the waste lands on slots that were already decided, and it never changes
the machine's largest twin-free stretch. And the count of "fake twins" (composite-next-to-prime)
turns out to follow the same simple probability law as real twins - no hidden structure there.

Two solid bricks are now machine-verified in Lean (kernel-checked, strongest possible standard):
the reduction of the whole conjecture to the window statement, and the horizon theorem (only gears
below y matter strictly inside y's window). One genuine mystery surfaced: the machine's actual
gear phases are extremely special by some measures (they maximise wasted kills) yet totally
ordinary by the measure that matters (stride). Round 2 digs at the bottom band, formalises the
next brick, and tries to turn the mystery into a theorem.

## Round 1 (2026-08-18)

**Formalist** - proofs/Horizon.lean kernel-checked, zero sorries, standard axioms: the horizon
theorem (strict p < y), its contrapositive, and the twin-pair corollary. Lake lib registered;
BlockedSlots untouched.

**Mechanic** - fragile census exact to y = 50021 (members 2.5e9). Fragile/twins grows ~ lnln
(Mertens): 1.11 -> 5.06. Bottom-decile gears own 88% of fragile load at scale. Candidate
zero-parameter identity: fragile*pi_win/(twins*W1) -> 2 (measured 1.95-1.99, 0.43% at top end) -
the fragile census is the lone-composite population times the same partner-prime structure that
makes twins. Tool: research/fragile_census.py.

**Lateral** - new closed form: a twin gear-pair (p, p+2) marks the higher window at exactly two
deterministic slots (own slot + product slot 6u'(p+1)-1 = p(p+2)); coincidence classes mod p(p+2)
are {+-u', +-u'(p+1)}. Refuted: tooth-sharing cannot close the recursion by counting (O(T(y)) vs
K/log^2; stride dead flat under sharing). Anomaly found: the machine's real phase vector is wildly
non-generic in overcount (z = +6.1) and fragile count (z = -5.9) yet stride-generic. Tool:
research/tooth_sharing.py.

**Constructor** - Condition X (a twin-free window) formalized with an exact ledger: kills per slot
are 0/1/2 from distinct gears (slot-cap lemma); lpf attribution makes supply exactly 2N - P;
zero-slack theorem: X forces the full prefix census (fragile = primes count, doubles = the rest,
no waste). Sharpest condition C2: under X the doubles must outpace in-window primes from the first
slot. Real windows: X is impossible OUTRIGHT below y ~ 403 (too few composites exist - P - N > 0);
at y = 47 the global ledger passes but C2 fails in the bottom band by +7. The violation zone is
always the window's bottom band. Tool: research/constructor_ledger.py.

**Manager synthesis.** The constructor's descent idea (X at y forces a half-window stride event at
scale sqrt(2y)) re-derives Reduction A at constant 1/2 rather than reducing its difficulty - noted
honestly. The genuinely new leverage: (a) X-impossible-below-403 is a finite counting theorem
candidate (formalizable); (b) C2 localizes any hypothetical X-failure to the bottom band, where
the layer law and horizon machinery are strongest; (c) the mechanic's constant-2 identity and the
lateral's phase-anomaly both say the window censuses carry structure beyond density - possible
variational handle.

**Round 2 tasks queued**: Mechanic - per-gear fragile closed form incl. the top-gear rare-event
tail. Constructor - make X-impossible-below-403 rigorous + the bottom-band double-onset law from
deletion spacing. Formalist - the layer law's arithmetic core in Lean. Lateral - test whether the
real phase vector is extremal for any window observable.
