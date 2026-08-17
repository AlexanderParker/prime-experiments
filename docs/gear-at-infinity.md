# The gear at infinity

A conceptual frame for the machine, recorded because it is the source of several results in this
programme and because parts of it are now proved. It is not a proof of the conjecture, and the
sections below separate what is established from what is not.

## The argument

1. **The primes are gears joined sequentially.** Each prime `q` is a wheel of circumference `q`,
   turning at the same rate as every other, carrying its blocking teeth.

2. **The machine is fully constructed up to infinity.** Every gear exists simultaneously; nothing is
   added over time. The state of the machine at a position is the vector of all gear phases at that
   position.

3. **Being integers, the gears return to their starting point.** They diverge as they turn, but the
   phases are integers modulo integer circumferences, so after the least common multiple - the
   primorial - every gear is simultaneously back where it began. Infinite rotation resets the machine
   to its state at 0.

4. **At 0 the machine is completely aligned.** Every gear divides 0, so every gear shields rather than
   threatens. Slot 0 is exposed, in every gear set, always.

5. **No gear can turn faster than the base cycle.** Gears 2 and 3 make a 6-cycle leaving slots 1 and
   5. A later gear cannot outpace it, because a larger circumference means a slower walk across those
   slots. So the 1 and 5 slots keep being presented, forever, at the fastest rate in the machine, and
   the later gears can only ever sample them.

6. **Therefore the structure near 0 recurs.** Twins sit next to 0 - `(5,7)`, `(11,13)`, `(17,19)` at
   slots 1, 2, 3. If the machine's state recurs, so does its behaviour, and twins recur with it. To
   deny infinitely many twins is to claim the machine stops presenting a configuration it presents
   near 0, which contradicts its own periodicity.

## What of this is now proved

Four of the six steps are theorems in this programme, and two of them were found *because* of this
frame rather than independently of it.

**Step 5 is exactly right, and provable.** Every gear and every combination of gears walks the 6-cycle
at exactly `+/-1` per rotation - never faster, never slower. Successive multiples of `q` land in
successive slots stepping by `q mod 6`, and every prime gear is `1` or `5 mod 6`, so the step is `+1`
or `-1`. The same holds for composite sub-machines, since the units mod 6 are closed under
multiplication. Measured for every sub-machine of up to four gears from 5 to 59: periods mod 6 are
always 1 or 5, never anything else. (`twin-prime-program.md` section 26a, confirmed at section 32a;
the `+/-1` walk was first noticed in section 18a of the same document.)

**Step 4 is right.** Slot 0 is exposed for every gear set, since a gear's teeth sit at `+/- 6^{-1} mod
q` and `6^{-1}` is never `0`. It is the complete-shield position: every gear divides the midpoint, so
none can reach either member.

**Step 3 is right.** The pattern is exactly periodic with period `P = prod q`, and the exposed set is
a union of complete residue classes mod `P`. The machine does return to its initial state, and the
threat set is symmetric under `m -> -m`, so the pattern is symmetric about 0 as well as periodic.

**Step 1 and 2 are the correct model.** Everything in the programme is built on them, and the
closed-form next-twin method (`research/jump_distance.py`, verified to `k = 10^16`) is a direct
consequence: each gear's distance to its next tooth is `min((u_q - m) mod q, (-u_q - m) mod q)`, which
is exactly "read the gear's phase".

Two further results came from taking this frame seriously:

* **gear 3 blocks one of any two adjacent positions**, so every gap in the admissible pattern is a
  multiple of 3 and `F_h(y) = 0 mod 3` - checked against all thirteen known values;
* **gear 5 blocks one of any three positions spaced 3 apart**, so exposed runs have length at most 2
  and the pattern is exactly isolated points and dominoes.

Both are proved in `covering-bound-route.md` section 18a, and both are statements about what the
fastest gears do to the 1 and 5 slots - step 5's question asked of individual gears.

## Where the argument does not close

Step 6 is the gap, and the difficulty is **localisation, not existence**.

The recurrence in step 3 is real, but its period is the primorial - about `e^y` for gears up to `y` -
while the region in which those gears can decide primality is only `(y, y^2]`. The exposed set is
never empty in a period, `prod (q-2) > 0` always, so the machine does present the configuration
forever. What is not established is that it presents it *inside the window where the gears in play are
enough to certify it*. Beyond `y^2` a slot may be exposed to every gear up to `y` and still not be a
twin, because a prime larger than `y` divides one of its members.

So the frame gives: the configuration exists, recurs, and recurs at the fastest rate the machine
allows. It does not give: it recurs within `y^2` of where the gear set was assembled. That last step is
the whole content of `F_h(y) < y^2/6` - the maximum gap of the admissible pattern, in `k`-units, fitting
inside the validity window. The covering route reduces exactly that to a single minimisation claim,
`min_L h(L) = h(1)` (`covering-bound-route.md` section 26c), of which three cases are proved.

A second point of care. Step 6 says denying infinitely many twins contradicts periodicity. That is
true of the *admissible pattern* - the slots left open by gears up to `y` - and the pattern's
recurrence is proved. It is not automatically true of the twins themselves, because membership in the
pattern is certified only inside the window. The two coincide exactly on `(y, y^2]`, which is the
window identity `survivors(y, K) = T(6K+1) - T(y)`, verified exactly at `y = 11` through `1009`
(section 17d). Outside it they diverge, and that divergence is where the argument needs the missing
bound.

## Why it is worth keeping

Recorded not as decoration but because it earned its place: the `+/-1` walk law, the two blocking
laws, the mod-3 law for `F_h`, and the closed-form next-twin method all came from taking the machine
literally as a set of gears turning together forever. The frame has been more productive than any of
the formal routes tried against the remaining gap, five of which are now closed
(`docs/ideas-from-the-session.md`, section 5).

Its central claim - that the machine cannot stop presenting a configuration it presents near 0 - is
correct about the pattern. Making it correct about twins is the same thing as bounding the maximum gap
inside the validity window, and that remains open.
