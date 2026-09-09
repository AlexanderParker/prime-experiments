# The next prime gap as a closed form, from what the machine has established (2026-09-10)

Assembled from results on record; each piece carries its status. "Closed form" here means an
explicit finite expression in the machine's residues that evaluates to the gap with a
certificate, not a formula in p alone (no such formula exists on the record, and the last
section says what one would require).

## 1. The pieces on record

- **The mex form of the next opening** (manifold, KERNEL `TopMachine.mex_form`, round 34): for
  a gear set G of m gears all larger than 2m, the next open pair after x is
  x + mex{ (-x) mod g, (-x - 2) mod g : g in G }. Each gear contributes one residue per tooth
  and the least excluded value is the walk. Sharp: it fails as soon as a gear is at most 2m
  (a gear that small strikes twice within the walk). The single-tooth version (next open
  number) is the same with one residue per gear, `mexT`-type, KERNEL `triple_mex_form` for the
  three-tooth case.
- **The true hypothesis** (top_machine_5.md, L50): the mex form is exact iff the machine's record
  is below its smallest gear, F_top < q', not "gears > 2m".
- **The in-use form with a certificate** (top_machine_5.md, L57; exact on 890,501 walks): when
  small gears are present, replace each gear's residue by its arithmetic progression truncated
  at a bound B: M_B(x) = mex of the union over gears of { r_g + k g <= B }, with r_g the gear's
  residues after x; if M_B(x) < B the answer is exact (every strike below B is listed).
- **The core / tail split** (the loaded record rule, KERNEL `loaded_record_rule`, round 36): on a
  stretch of L slots the gears at most 6L + 1 (the core) decide, the gears above (the tail)
  contribute at most one domino each.
- **The fold** (proof skeleton, step 2): gears 2 and 3 leave the slots (6j - 1, 6j + 1); gears
  2, 3, 5 leave the anchor's cycle of 30 with its three slots per cycle (README glossary).
- **The square-root rule** (proof skeleton, step 3; KERNEL `OneStepE.blocked_iff_sqrt` inside the
  next prime's square): a number below P^2 with no prime factor below P is prime.

## 2. The assembled function

Let p be a prime and B a bound on the gap to be certified. Let C = the primes at most B (the
core) and T = the primes in (B, sqrt(p + B)] (the tail). Then

    g(p)  =  mex(  U_core(p, B)  union  { (-p) mod q : q in T }  )

where U_core(p, B) = the union over q in C of { ((-p) mod q) + k q : k >= 0, <= B }, and the
answer is certified exact whenever g(p) < B (then every strike of every gear below sqrt(p + B)
on (p, p + B] is in the set, and by the square-root rule the least missing position is prime).

Why the tail needs one residue per gear: a gear q > B strikes at most one position in (p, p + B],
namely its first multiple after p, at offset (-p) mod q. Why the core needs its progression: a
gear q <= B strikes about B/q positions in the stretch. Both facts are the core / tail split of
the record rule read on one stretch.

Cost: |T| + sum over q in C of B/q evaluations; for B of the order of the expected gap (about
ln p) the core is the primes below ln p and the tail is about sqrt(p)/ln p residues. The tail's
residues (-p) mod q for all q <= sqrt(p + B) are the whole content: the mex is over them.

## 3. The folded form (the anchor as a lookup)

Fold the smallest core gears into a fixed pattern: with the anchor 2, 3, 5 the candidates are the
three slots per cycle of 30, so the gap is measured in slots and U_core loses its three smallest
progressions, replaced by "the next anchor-open position after p", a lookup in one cycle of 30.
The manifold's proved laws apply to the rest: with the remaining core (7 .. B) and the tail, the
same mex, now over slots. Nothing changes in the certificate.

## 4. What it is, and what it is not

- It is exact, self-certifying, and an expression of the gap purely in the gears' residues at
  p: the machine's "next gap algorithm" (README, rust2) is this expression evaluated by walking.
- It is not a formula in p alone. The residues (-p) mod q for q up to sqrt(p) are the
  information; no expression on the record compresses them. The only situation where the mex
  collapses to one residue per gear with no truncation is the free-manifold regime (every gear
  above the record), which the primes never satisfy because 2, 3, 5, 7 are gears.
- A bound on g(p) from the expression would need a bound on the mex over the residues, i.e.
  on the longest run of consecutive positions each hit by some residue progression: that is
  the record of the machine {primes <= sqrt(p)} on a stretch, the object of research/proof/
  step_evidence.md, and for the prime gap it is Jacobsthal's function j(P_sqrt(p)), bounded in
  print only by counts (the largest gap between integers coprime to a primorial). So the closed
  form gives the gap with a certificate but no bound; the bound is the same wall as the step.

## 5. The twin version, for the record

Replace one residue per gear by two ((-p) mod q and (-p - 2) mod q) and the certificate by
"both members prime below sqrt(p + B)^2": g_2(p) = mex(U_core^2 union { (-p) mod q, (-p-2) mod q :
q in T }), exact when g_2 < B (top_machine_5.md L57 measured this on {5..31}; the free-regime
half is `mex_form` in the kernel). The next twin after p is p + g_2(p).
