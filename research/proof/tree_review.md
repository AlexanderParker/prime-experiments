# Whole-tree review (Reviewer lane, 2026-09-11)

Mandate: audit every node's verdict against its document, reconcile results across branches, list
the promising leads not fully tested, and state whether anything already on record implies the
step at any link of the stack. Reading, cross-referencing and two spot checks only (Python, one
core, seconds). Nothing here is committed; the tree and the novel index are not edited.

Read: the theory-tree skill; theory_tree.md whole (profile, ~90 nodes, log to 2026-09-10);
proof_skeleton.md, step_evidence.md (sections 1-7), core_leftover.md, base_and_step.md (Parts I
and II), objects_ledger.md, the_wall.md, law_register.md, docs/novel/README.md, the three unstick
passes, human.md; branch documents lengthen_never_precede.md and rich_half.md in full, and by
targeted reads monotone_functional.md, pad_cap.md, top_machine_8.md (W101), top_machine_lean.md
(round 40, proofs/CoreLeftover.lean), stacked_squares.md (S1-S7), turn_ledger.md, valves_scratch.md.
Canonical words used below (engine, valves, manifold, exhaust; stack, cut, section, machine, band;
core, tail, leftover); older node names quoted verbatim where a mismatch is being quoted.

Vocabulary for the stack used throughout: cuts c_1 = base, c_{k+1} = p_k^2 with p_k the first
prime at or above c_k; section k+1 = [p_k^2, p_{k+1}^2); the composite machine acting on section
k+1 is the set of primes below p_{k+1}; since no prime lies in [p_k^2, p_{k+1}), that set is the
engine {5..q} with q = prevprime(p_{k+1}), and p_{k+1} = q' is its next prime. Write Q = p_{k+1}
for the cut's root, L for the section's composite record in slots, B = 6L + 1 for the core's top,
so core = primes in [5, B] and tail = primes in (B, Q).

---

## 1. Node-by-node audit

### 1.1 Verdict or statement mismatches (quoted)

M1. **Theorem (E) as stated on the node and in the ledger is false; the correction reached the
wall and the log but not the node or the ledger row.** Node R2.e.i: "for every column with
6k - 1 > q, blocked under {5..q} iff blocked under {5..floor(sqrt(6k + 1))}". Ledger M6: "for
`6k - 1 > q`: blocked under `{5..q}` iff blocked under `{5..floor(sqrt(6k+1))}`". Log 2026-09-07
(Formalist round 39): "theorem (E) as phrased in position_frontier.md is false outside the prefix
(refuting instance q = 5, k = 8 in the kernel, E_needs_prefix); the true form needs 6k + 1 < q'^2".
The refutation is elementary (column 8 = (47, 49): open under {5}, blocked under {5, 7} since
49 = 7^2). The wall's 5n carries the fix; node R2.e.i, ledger M6 and node R4 ("theorem (E) already
says the effective machine at a column is exact") should carry the hypothesis 6k + 1 < q'^2.

M2. **Node R2.c.i keeps a number the manager withdrew.** Node: "the trivial bound misses by a
factor 2 to 3.7, the smallest miss of any unconditional bound on the tree ... Child named: the
shape of the rho_g profile". Log 2026-09-06: "manager's second check on R2.c.i: the one-block
inequality is trivially valid (each term is alpha_g^2) and its content is the survivor lower
bound itself; the 'factor 3.7' assumed the fair share in the denominator and is withdrawn. Wall 5g
rewritten." The node text and its named child rest on the withdrawn reading; the wall's 5g is the
correct state. The child was, rightly, never opened.

M3. **Node 4 still reads WEAK; its child refuted it.** Node 4: "Genealogy (records recruit
runner-ups). WEAK: exact at 8 steps ... the theory 'bounded branching bounds growth' is untested."
Node 4.i: "NODE 4 REFUTED: branching is bounded and bounds nothing (J_max x max piece fraction =
2.4, 2.0, 3.5 at m23, m29, m31)." Node 4 should read DEAD (refuted), survived by the frontier
object of 4.i.

M4. **Node 4.i.a.i.a (STRONG) states the pinned letter as a law whose lower half is refuted two
nodes down.** Node: "NEW MEASURED LAW, THE PINNED LETTER: F(M) <= a_L + r(a_L) <= F(M) + 3 at 8 of 8
rungs ... confirmed OUT OF SAMPLE at 31 -> 37". Node 4.i.a.i.a.1.a: "THE LAW IS HALF REFUTED out of
sample: at 37 -> 41 ... r(14) = 63, so a_L + r(a_L) = 77 against F = 88; the lower half fails by
11". The ledger's engine-measured row 3 has the corrected form; the node does not.

M5. **The CANDIDATE OBJECT marks on R2.a.i.a and R2.a.i.a.1 were never withdrawn.** R2.a.i.a:
"STRONG, exact, and it names a CANDIDATE OBJECT"; R2.a.i.a.1: "STRONG. THE OBJECT SHARPENED". Later
verdicts on the same object: C3 (square_vector.md) "real vectors are typical"; R4.d.i.a "The
witness = blind classes (proved) + ordinary density (a count). ROOT for existence; the blind-class
structure kept as FACT"; the wall's 5m still calls it "the only candidate on record that is
neither a count nor a period-scale construction". By the skill's rule a candidate carries "what
would have to break it and why the system cannot"; both nodes now have the answer ("a count") and
should read FACT (blind classes) / ROOT (existence), with the 5m sentence amended.

M6. **The objects ledger contradicts itself on the gate.** Section "THE GATE": "THE GATE IS OPEN
(2026-09-07): engine, manifold and exhaust each have no open structural item that is not the
conjecture in disguise" and, further down the same section, "**Gate verdict: the gate is NOT open,
on the engine alone.** ... the engine has three, all instrumented, one with a lane running." The
second sentence is the 2026-09-06 verdict left in place. Tree node R4.c ("the ledger's gate
cleared") agrees with the first.

M7. **Stale register numbers in the ledger.** Ledger MANIFOLD gate: "the non-cancellation of the
top moment (W86, PROVED, the cover polynomial); the parity-refined capacity bound (W93, PROVED,
exact below core density 0.376)". The register renumbered top_machine_8.md's laws to W95-W102 and
says so ("Read 'W86' and 'W93' in those two files as W95 and W102"); the tree's R4.b.xi was
corrected, the ledger was not. W86 and W93 now name the piece law and r(d) = D(d - 1).

M8. **Two "pending" notes are stale.** Node R4.c.iii.a: "the m37-m43 period table pending its
scan" - lengthen_never_precede.md section 2 has the m37, m41, m43 rows (prefixes of 2^35 columns),
filled. Node R4.c.ii: "the 37 -> 41 positions addendum pending its scan" - rich_half.md section 12:
"Not filled: the 33,263-copy scan of m37 was killed by the session limit ... The 37 -> 41 record's
translate rank stays unmeasured." The first note should be deleted, the second should say
"unmeasured", not "pending".

M9. **F(59) is "exact" in one document and an interval in another.** human.md item 2: "The
records F(M) = 5, 7, 11, 18, 25, 34, 43, 58, 88, 91, 103, 118, 145, 161 at q = 7..59 are exact".
docs/novel/README.md (old-machine-spectrum, round 27): "161 <= F(59) <= 178 with the lower bound an
exhibited machine-53 window". The tree (R4.c.iii), valve_existence.md and block_moment.md use
"F(59) = 161" as certified. I could not find the record that closed 161 <= F(59) <= 178 to
equality; either it exists and the novel index is stale, or human.md and the three uses overstate.
The manager should decide which; V8's certified range "turn 1 for every Q in [30, 1859]" depends on
it (the pre-registration used 1739 with 59^2 as the cap).

M10. **R4.b.iv's open items were closed elsewhere and the node was not annotated.** Node: "OPEN: a
bound above the zone (A(q, N)); the first-hit model as a law". R4.b.vii: "this object IS R4.b.iv's
A(q, N) ... NO UPPER BOUND, AND THE REASON IS EXACT (L65)" - the first item is ROOT. The second (the
first-hit exactness W32) is still open and is in the untested list below.

M11. **Which prime starts machine 1 differs across the stack documents.** proof_skeleton.md
section 4: "machine 1 = {3, 5, 7}"; step_evidence.md header: "machine 1 taking every prime from 5";
base_and_step.md Part II: "machine 1 = the primes from 7 up to c_2". Harmless for every count
(2, 3, 5 fold into the slot coordinate) but one sentence in the skeleton should fix it.

M12. **base_and_step.md Part II has an unfilled scorecard (Q1-Q7, O3) although its result files
exist.** research/stack/r2/results/ holds generated.json, record_scan.json, record_scan_rows.npz,
chains_all.json, twins_1e9.npy; the document ends at the scorecard with every verdict
"(filled below)" and nothing below. The lane died at the weekly limit; the manager's step_evidence
sections 5-7 measured a subset locally (the composite record on 11 sections). The full scan over
every section with q'^2 <= 10^9 (about 3,400 sections, Q5) and the chain pairs (Q6) are on disk and
unread. See lead U1.

### 1.2 Every OPEN, WEAK, STRONG or CANDIDATE node, and where it stands

| node | verdict on node | current state | closed by / what would close it |
|---|---|---|---|
| R1 per-step | OPEN, "at least as hard as twin-Bertrand" | ROOT in substance (face E1) | nothing short of twin-Bertrand at every rung; the owner retired the ladder as a proof framing (log 2026-09-07) |
| 1 pair statement | OPEN | ROOT (1e: at column 0 it is 2 d_0 <= F + q') | the ledger already lists it under "the conjecture in disguise"; the node should say ROOT |
| 2 chain statement | OPEN | genuinely open as a finite statement per rung (band [15, 36] at 29 -> 31 passes by complete enumeration, monotone_functional.md); as a law it needs the real higher gears (2f refuted) and over-asks (E1) | no route on the tree; 4.i.b.ii's one lemma is its shape |
| 2b literal case | OPEN | stale: reduced to the flank envelope whose per-J form assumes Delta_J <= s_min, measured in [-3, +4] to m41 (per-j-window-analogues) | a proof of Delta_J = O(1), which the family refutes (eps in [-21, +15]); O-M6 |
| 2g three-gap repulsion | STRONG as a pattern, unproved | superseded by 2g.i's N(v) <= F_2 (v >= 6), exceptionless to m31, mechanism proved (glue), law unproved; the J-run outer law likewise to m23 | out-of-sample test at m37 from the closure dictionary (lead U4); no proof route on the tree (the glue as a covering statement is dead) |
| 4 genealogy | WEAK | refuted by 4.i (M3) | should read DEAD |
| 4.i.b.ii monotone functional | STRONG, item 1 open | item 1 (L bounded) resolved by 4.i.b.ii.a: E1 caps the measured half, the skip half is ROOT; the theorem B_{L+1} <= F + q' is the single named lemma whose proof gives the budget at every rung | the level-(J_max - 1) relaxed deepest-fusion lemma; out-of-sample decision at 37 -> 41 (lead U2) |
| 4.i.a.i.a short-letter row | STRONG | stands (LP-duality certificates); the pinned-letter half is refuted (M4) | node text to be amended |
| 5 made at the top | STRONG as an observation | closed (R3 line verdict: position objects, no length lever) | - |
| 5b repulsion | STRONG pattern, closed | rediscovery of the suppression law; the exact profile became 2g.i | - |
| 7b anchor rigidity | CANDIDATE OBJECT, DEAD as a route | its proved rigidity was later named as an ingredient twice (R2.c.i: "the unproven part is exactly the in-window equidistribution ... 7b's curve"; R4 zones: "the rigidity generalisation" as "first things to formalise") and never formalised | the moving-anchor generalisation (lead U8) |
| R2 whole window | OPEN | the root's least-demanding formulation; stands | - |
| R2.a.i path taken apart | STRONG as a description | closed; its CANDIDATE (the landscape) went to R2.a.i.a | - |
| R2.a.i.a, R2.a.i.a.1 island witness | STRONG, CANDIDATE OBJECT | FACT (blind classes) + ROOT (existence), see M5 | verdict to be changed on the nodes |
| R2.e.i frontier | STRONG | stands as a proved reduction, with (E)'s hypothesis corrected (M1); E6-E8 sharpen it | - |
| R4 period scale | OPEN | the manifold, exhaust and valves lines all hang here; the owner's construction moved to R4.d | - |
| R4.b, R4.b.ii, R4.b.iii, R4.b.ix, R4.b.x, R4.b.xi | STRONG | closed: the manifold has no open structural item; the census, record rule, walk and exhaust laws are in the kernel | - |
| R4.b.iv in-use bound | STRONG, two OPEN items | one ROOT (L65), one open (W32 exactness) | lead U5 |
| R4.c valves | (no verdict word; "THE VALVES HAVE A MECHANISM") | every child FACT or ROOT; the node should carry FACT with the mechanism (imprint, port, onset, ember) and ROOT for the pure charge | - |
| R4.d.i base case and step | (no verdict word; the live node) | children R4.d.i.a FACT/ROOT, R4.d.i.b FACT/ROOT; Part II of its document unfilled (M12); round 40 kernel landed | leads U1, U3, U6 |

Genuinely open and not the root in disguise, on the whole tree: the chain statement at a rung as
a finite object (2, O-M2); the monotone functional's one lemma (4.i.b.ii); N(v) <= F_2 and the
J-run outer law as laws (2g.i, ledger engine-measured 1-2); the pinned letter's upper half (O-M4);
the first-hit exactness W32; the complexity of the core minimisation (W101, class (b)); the
crossover height O-X6 (a known theorem is the missing instrument); the moving-anchor rigidity
(7b generalised, named twice, never done). Everything else open is a count or the root.

### 1.3 DEAD or FACT nodes whose statement a later result changed

- **7b (DEAD as a route)** - its proved in-window rigidity of {5..13} is the exact ingredient two
  later nodes name and leave unproved (R2.c.i's rho_g = 1 head; R4's "the bottom is rigid inside
  the window against every larger gear (the anchor rigidity of 7b generalised)"). Not revived as a
  route, but a proved part of a DEAD node is load-bearing in two OPEN sentences and the
  generalisation is unformalised. Listed as lead U8.
- **7d (DEAD: the region past zero is thinner than the period mean, 0.79)** - S14 (core_leftover)
  finds the same fact from the stack's side: the real section's core leftover density is 6% below
  the CRT product at L*, 14% at L = 2000, rising with u. The two measurements agree in sign and
  mechanism (Buchstab next to the origin); the tree files them under different objects with no
  cross-reference. Not a revival; a duplicate that should be linked (S14 is 7d one coordinate up).
- **R2.c (the claimed positive, withdrawn) and R2.c.i (M2)** - correctly recorded as withdrawn in
  the log and wall; the node text of R2.c.i is the only place the withdrawn reading survives.
- **6 (coherent spacings DEAD)** - unstick pass 3 named the pullback's coherent separations as
  "the only untested place where the real one-third separation could matter (never measured for
  Omega)"; rich_half's E4 then proved the arc floor for ANY separations with arcs >= 2, so
  coherence is irrelevant there too. DEAD stands; the one named test was run and confirmed it.
- No DEAD node carries a "not a route" filing that a later proved law makes a route. The one
  correction of that kind on record (4.i.b, "not a route is a bold claim") was made by the owner
  on 2026-09-06 and is on the node.

### 1.4 The "closest thing / strongest lead / child named" sentences and their children

| where | sentence | child opened? |
|---|---|---|
| 2g.i | "CHILD NAMED: for every 3-run with v >= 6 there is a two-colouring ..." | yes, 2g.i.a (DEAD) |
| R2.c.i | "Child named: the shape of the rho_g profile" | no; motivation withdrawn (M2) |
| 4.i | "CHILD NAMED: why Rest(a) collapses as a -> F(M)" | yes, 4.i.a |
| 4.i.a | "CHILD NAMED: the availability gate" | yes, 4.i.a.i |
| 4.i.b.i | "Child named: the Q*_J peak sits at J <= 4 at all five rungs, including both where J_max = 5" | **no**; and 4.i.b.ii's order law reports "the binding term is always the deepest fusion J = J_max (7 of 7)". The two sentences are about different objects (realised Q*_J against the level-k bound B_k) but read as a tension; one line on 4.i.b.ii should say why both hold |
| 4.i.a.i.a.1.a | "NEXT NAMED: the record gap itself as a 2-run" | yes, 4.i.a.i.a.1.a.i |
| 4.i.a.i.a.1.a.i | "Child named: the letter's N(l) as a 3-run and what the padded middle buys" | **no** (the window line was closed by the owner the same day; recorded as not opened, not as dead) |
| R4.c.i / unstick 3 | "THE ONE OBJECT THAT IS NOT THE RECORD RESTATED: its dual, the rich-interval function Omega ... Recommended and opened" | yes, R4.c.ii (ROOT) |
| R4.c.iii | "Next child named: a new gear never creates a run of length >= d_0 before the runs already there" | yes, R4.c.iii.a |
| log 2026-09-07 | "follow the hot lead" (the island witness is the stack's step at a section start) | yes, R4.d.i.a (closed the same day) |
| unstick 1, reading as a whole | thin places 1-5 | all five opened and closed (R2.b, 2g.i.a, thin place 3 scan, 2f.i, R2.a.i.a.1.c) |
| unstick 2, reading as a whole | items 1-4 | 1, 2, 3 opened (R2.d, R3.i.a, R2.c.ii); **4 ("words, not columns": the glue at the level of the gap word; separability at the junctions) never opened** |
| unstick 3 | (b)(i) burnt-run law on prime-led charges; (c)(ii) the free-phase manifold adversary's reach R(q, Q); (d)(i) the charge-count walk; W103 idea 2 (record per turn against A(m)); L65 idea 1 (the disjunction's covering form) | **none run**; only the rich half was |
| the wall 5m | "The island witness ... is the only candidate on record" | overtaken by R4.d.i.a (M5) |

---

## 2. Cross-branch reconciliation

### 2.1 The step, the window statement, and the straddling condition (exact correspondence)

**(a) The step at link k is the window statement at rung p_{k+1}, up to the home column.** Section
k+1 = [p_k^2, p_{k+1}^2). No prime lies in [p_k^2, p_{k+1}), so the twin pairs with lower member in
the section are exactly those with lower member in [p_{k+1}, p_{k+1}^2). An opening of the engine
{5..p_{k+1}} in its window (p_{k+1}, p_{k+1}^2] is a twin with lower member in (p_{k+1}, p_{k+1}^2)
(the kernel's open_iff_twin). Hence

    step at link k  <=>  [window statement at rung y = p_{k+1}]  or  [(p_{k+1}, p_{k+1} + 2) is twin].

Equivalently, with q = prevprime(p_{k+1}) and W(q) = (p_{k+1}^2 - 1)/6: the section is the top of
the prefix [1, W(q)] of the engine {5..q}, the sliver (q, p_k^2) holds no twin, so the step at
link k is the window statement at rung q in its prefix form (an opening of {5..q} in (q/6, W(q)]).
This is what the tree's R4.d ("twin-Bertrand between consecutive squares of the chain") and
skeleton section 8 say, now with the rung named. Two consequences that were not written down:

- The composite record on section k+1 is at most F({5..q}), the engine's period record at the rung
  q = prevprime(nextprime(p_k^2)). So wherever F is certified the step is proved (section 4).
- The stack visits the engine's ladder at the sparse rungs q_k with q_k ~ p_k^2, one rung per
  link; the budget summed to q (R1's content) implies the step at every link whose rung it
  reaches. R1 over-asks (E1) and the stack does not: the stack's statement per link is exactly the
  root at one rung, no more.

**(b) The straddling condition is strictly stronger than the step at the next link, and it lives
at the stack's cut.** At the ladder step q -> q' = p_{k+1}, the square column W(q) = (q'^2 - 1)/6 is
the column of the cut c_{k+2} = p_{k+1}^2: the boundary between section k+1 and section k+2 IS the
square column of lengthen_never_precede.md. The straddling run of E8 (the run of {5..q'} through
W(q)) is the twin-free run across the cut, joining the last twin of section k+1 to the first twin
of section k+2; its upper half is exactly the object of step_evidence section 1 (the first twin
above the cut, offset below the top gear's long arc, 0 exceptions to q = 10^7). The straddling
condition (E8's proviso, in the twin coordinate: t_1 - t_0 >= q' + 6 implies
t_1 - t_0 <= (t_0 + 7)/4.625 + 6) gives, in either branch of its case split, a twin in
(q'^2, max(1.216 q'^2, q'^2 + q' + 6)], which lies inside section k+2 because p_{k+2} >= q'^2.
So

    straddling condition at rung p_{k+1}  =>  step at link k+1,

and not conversely (the step is existence; the straddling condition is a two-sided gap bound at
the square, face E). The tree marks the straddling condition ROOT "in face E's sense"; the exact
statement is: it is the step at the next link with a 21.6% margin condition added.

**(c) What the frontier's proved parts give about the section above a cut: nothing about its
interior, one thing about its record.** E6 says the new gear q' newly blocks at most two columns of
[1, W(q)] (the twin column d_0(M), the square column); above W(q) theorem (E) (in its correct form)
makes the effective machine the whole {5..q'} at every column of section k+2, so no sub-machine
decides any column there. E7 (prefix inheritance) says every maximal run of {5..q'} strictly inside
[1, W(q) - 1] is a run of {5..q}; applied up the ladder it says the composite record of section
k+1 (a twin gap) is a maximal run of every later engine's prefix, unchanged, until the initial run
absorbs it. That is all: the record of a section is a permanent feature of the line (it is a twin
gap), and E7 is the machine-side proof of the obvious. No inequality on the section above a cut
follows from E6-E8.

**(d) A measured corollary worth one cheap computation.** By E8, H_4.625(q') follows from
H_4.625(q) plus the straddling condition at q'. The straddling run's length is L_0 + L_1 + 1 with
L_0 the last prefix run of {5..q} (the twin gap up to q'^2; by H_c(q), L_0 <= W(q)/(c + 1), measured
top-run share tau <= 0.0833) and L_1 the offset of the first twin above q'^2 (measured
< (2q' + 1)/3 columns to 10^7). The condition x_s >= 4.625 L_s then reads
(1 - tau) W >= 4.625 (tau W + L_1 + 1), which holds whenever tau <= 0.0833 and
L_1 <= (2q' + 1)/3 for every q' >= 41. So the first-twin scan already extends the frontier floor
4.625 to every rung to 10^7 PROVIDED tau stays below 0.0833 there; tau is measured only to
19,997. Lead U3.

### 2.2 One window function under three names (exact identity)

For a gear set G, a step s coprime to every gear, a translate x and a length n, let

    c^{(s)}_n(x) = #{ 0 <= j < n : x + j s is open under G }.

Then, exactly:

- **the core's leftover** (core_leftover.md): K_L(x) = c^{(1)}_L(x) (slots, s = 1 in slot units);
- **W101's two windows** (top_machine_8.md): in the manifold's pair coordinate,
  K_L(x) = c^{(2)}_{ceil(L/2)}(x) + c^{(2)}_{floor(L/2)}(x + 1), the even and odd cells of [x, x + L);
  W101's content is that multiplying by 2^{-1} mod W_core turns the second window into the first
  translated by the half-turn H = (W_core + 1)/2 of one adjacent-teeth wheel;
- **the rich-interval function** (rich_half.md): Omega_{M^{(q')}}(n) = max_x c^{(q')}_n(x), since the
  pullback M^{(q')} (separations 2 u_g q'^{-1} mod g) is M read along an arithmetic progression of
  step q' - the same construction as the manifold's "coherent twisted copy at separation 2 g^{-1}"
  of R4.a, at the gear q'.

So the free-phase record of a core is the largest L with min_x c^{(1)}_L(x) = 0 (S12 with the
loaded rule's t = 0), W101's record formula is the coupled min of two s = 2 windows, and Omega is
the max of the s = q' window: one function, its minimum giving the record and its maximum the
word-depth cap E2. In the same coordinate the attainment identity reads: for any stretch of S
columns and any phase of q', c^{(1)}_S(x) >= c^{(q')}_{T_1}(x_1) + c^{(q')}_{T_2}(x_2) (the two tooth
classes of q' inside the stretch), with equality iff q' strikes every opening of M in the stretch,
and F(M + q') is the largest S admitting equality at some phase. This is why E2's cap is "linear in
T = F/q'" (rich_half's ROOT reading) and why no bound flows between the min and the max: they are
the two extremes of the same function over translates, and a bound on one extreme says nothing
about the other. Filed as the deliverable identity; it closes the manager's question "are these
the same object in two coordinates" with yes, and says what the coordinates are (s = 1, 2, q').

### 2.3 The valves' onset law inside the core's leftover types (exact, spot-checked)

Take section k+1 with cut root Q = p_{k+1}, record L, core B = 6L + 1, tail (B, Q). A core-leftover
slot has both members B-rough and below Q^2 < B^3 (B^3/Q^2 = 1.6 x 10^2 on base 3, 4.4 x 10^2 on
base 7), so by the two-prime lemma (round 40, primeOrSemiprime_of_rough_lt_cube) each member is a
prime or P_1 P_2 with P_1 <= P_2 primes above B. A tail gear g strikes such a member iff the member
is g x P_2 with P_2 a prime above B; two kinds:

- **ember-type**: P_2 < Q, the member is Q-smooth (an ember of the section's own split, engine =
  primes below Q); exists only above B^2;
- **fuel-type**: P_2 > Q, the member is a charge s P with air s = P_1 (a single tail prime) and
  fuel P_2; the slot belongs to the family (1, P_1), (P_1, 1) or (P_1, P_1') of the split at Q, and
  by the onset law (a family fires at turn max(s, s'), i.e. at height above s Q) the tail gear P_1
  can finish a core leftover by a fuel-type strike only at heights above P_1 Q >= B Q.

Below B^2 the tail is silent on B-rough numbers, so every core-leftover slot is a twin: this is
the round-40 kernel's section identity (leftover_eq_card_twins) and core_leftover's "quiet part".
Spot check (scratch onset_transport.py, base-7 section [2809, 7,946,761), L = 254, B = 1525,
Q = 2819, sieve 0.1 s): 54,183 core-leftover slots, types PP / P+C / C+C = 48,249 / 5,612 / 322;
first ember-type finish at height 2,362,333 against the floor B^2 = 2,325,625; first fuel-type
finish at 4,315,889 against the floor B Q = 4,298,975; 0 violations of "fuel-type least factor
P_1 <= height / Q" in 3,592 fuel-type finishes; min K_254 by stratum 1 / 2 / 1 (quiet, ember-only,
fuel). So the section splits into three strata by which finishes the tail can make: none on
[c, B^2), embers only on [B^2, B Q), embers and charges above B Q; and within the last, gear P_1
engages from height P_1 Q, in order of size. This is step_evidence section 2's "the newest machine
engages only from the next cut" one level down (the tail's gears engage on the core's leftovers
one at a time), stated exactly.

Against step_evidence section 7: the record stretch of base 3 (height 2.56 x 10^8, turn 15,858)
has leftover types 0 PP / 9 P+C / 3 C+C against 13.9 / 5.6 / 0.7 on random stretches, and every
composite member listed there (14173 x 18059, 11903 x 21503, 5987 x 42751) has P_2 > Q = 16139:
all twelve finishes at the record are fuel-type, i.e. charges of families with a single tail prime
as air, all of which have fired (their onsets P_1 Q <= 14173 x 16139 = 2.3 x 10^8 lie below the
record). The record sits above both floors by a factor 20 (B^2 = 1.2 x 10^7) and 4.6
(B Q = 5.6 x 10^7), so the stratification does not bind at the record; what it gives is the exact
list of who can finish where. It is a which-residues fact (allowed by face A) and it is a
mechanism, not a count; it does not bound anything. Reading for the tree: FACT, to be attached to
R4.d.i.b with the round-40 kernel names.

### 2.4 The loaded record rule at fixed phases

No fixed-phase version exists in the kernel or on paper, and none can have content: the rule's
only non-trivial direction at a fixed phase vector is cost_le_tail_of_coverable (no hypothesis),
which at the real phases says "if the real machine covers [x, x + L) then D(U_x) <= t", and D(U_x)
is at least ceil(K_L(x)/2) while t is the whole tail (1,390 on base 3 against K = 12 at the
record). With the phases fixed the condition "coverable" is literally "every leftover slot is
struck by a tail gear", which is the definition. core_leftover's finding stands unchanged: the
rule bounds only the free-phase record (25,267 slots against the real 3-leftover minimum at
L* = 579), and its slack at L* is the whole tail. The min-over-phases is the rule's content; at one
phase there is nothing to minimise. (S10, one phasing = one position, is the exact reason: the real
phases are the period's origin, and the rule's minimum is over the period.)

### 2.5 The island witness against the section starts

Reconciled by the manager (R4.d.i.a, step_evidence sections 1 and 4): the witness's four classes
are the offsets gears 5 and 7 cannot strike relative to a square, the square fixes classes by
exact availability fractions and adds no richness (322,186 against 321,052), existence in the arc
is a count. Two facts from island_witness.md that the stack side has not used: (i) the witness
holds for every INTEGER coprime to 30 above 2849, so the section start being a prime square is
not what makes it work - a section could be cut at any square coprime to 30 and the same blind
classes would carry the first twin; (ii) the absolute offset of the free island never exceeds
2,392 columns to q = 200,000, against the first-twin scan's "median offset below 0.005% of the arc"
at 10^7 - both are the Cramer-type scale ln^2 q for twins at squares, measured from two sides. No
identity beyond what is filed; no bound.

### 2.6 The order law and Phi = B_{L+1} against the composite record per section

No direct contact. Phi bounds the PERIOD record F(M + q') of the next engine from the depth-(L+1)
table of M; the composite record per section is the largest twin gap in a slice of length about
p_k^4/10 slots of a period of length exp(p_k^2), and along the chain the engine jumps from rung
q_k to rung q_{k+1} ~ q_k^2 in one link, across about q_k^2 / ln q_k ladder steps. What the two
share is 2.1(a): the section record is at most F(q_k), so any theorem giving the budget at every
rung (Phi's theorem, B_{L+1} <= F + q' for all M, with E1 capping L's measured half) would give the
step at every link. The undecided instance of the order law is 37 -> 41 (m37 sticks at depth 2,
B_2 = 161 > 129, k* predicted 3); lead U2.

### 2.7 The pad cap E1 against the record runs' covering structure

E1 is a per-gear statement: the legal word of ONE added gear over its small alphabet has length
at most 8, so one gear fuses at most nine old runs in a legal word. The section's record run on
base 3 is fused by 181 tail gears each striking one slot (12 of them on core leftovers, 169 on
core-struck slots), i.e. a fusion of depth 181 across 1,390 added gears, far outside any word cap;
iterating E1 gear by gear gives nothing because the alphabet resets at every gear. The right
object on the covering side is the tail's supply per unit length, which step_evidence section 6
showed never binds (T >= K on 100% of stretches). The connection that is exact: the tail's 181
strikes on the record are 181 singletons, and by the loaded rule's piece law (W86) a gear above
L + 1 contributes at most one domino, one slot in slot units - the covering structure of the
record is the piece law with real phases. No bound.

### 2.8 The arc floor and the collision laws against the exact finish

The arc floor (ArcFloor.lean, any separations with arcs >= 2) and the collision laws bound the
MINIMUM over phases of two gears' overlap in a window; at the real phases the overlap is what it
is, and the tail's finish at the record is 12 leftovers covered out of 181 strikes (169 wasted on
core-struck slots, waste 93%, against an expectation of 3.75 leftovers covered by 181 uniform
strikes on 12 of 579 slots). The twin collision law (shared arc, onset (g + 4)/3) was already
priced by the skeleton's 11a as a density correction of a few per cent (step_evidence section 3).
Nothing in the collision laws speaks to a fixed phase vector; no bound, and no identity beyond
"the finish is a coincidence of K placements" already on record.

### 2.9 W103 against S7 and the composite record: not one object

W103 (manifold_census_large.md): the manifold's quiet-zone record at Q is the same twin gap for
every engine, in the bottom stratum (1.35-2.63 Q), because there only primes and q-smooth numbers
are open and the record gap holds no smooth number with a prime neighbour. The composite record
of a section (step_evidence section 5) is the longest TWIN-FREE run in [p_k^2, p_{k+1}^2): charges
do not split it, only twins do, so it grows with height like ln^2 x and sits near the TOP of the
section (base 3: at 2.56 x 10^8 = 0.98 of the section, 15,858 Q; base 23: at 187,913 = 0.64 of
the section, 347 Q). W103's record is the longest CHARGE-FREE run, which is shortened by every
family that has fired and therefore sits where no family has: the bottom two turns. The manager's
match (the base-23 section record at 187,913 is the W103 gap 187,907 -> 188,831) is a coincidence
of two censuses at different cuts (Q = 541 for the section, Q = 10^5 for the census), not an
identity of objects. The exact relation: on the bottom stratum (Q, 3Q] of any cut the two records
agree unless an ember splits the twin gap; above it the composite record is the larger. S7 (the
twins of band k are the double-home slots of machine k+1) is the cut-side reading of the exhaust
cap and does not enter. Verdict: two ROOT objects, correctly filed twice, different in position and
in what splits them; no cross-branch bound.

### 2.10 Round 40 in the kernel, read against the tree

proofs/CoreLeftover.lean (green at 1036 jobs, standard axioms, zero sorries, not in defaultTargets)
makes theorems of: S11 rigidity (a pairwise-coprime replacement core does not exist; the least
prime factor map is a bijection onto the primes in [5, N] and every member is a prime power); the
two-prime lemma (1 < n < B^3, B-rough gives prime or p q with p, q >= B, no hypothesis on B); the
depth lemma in slot and column forms (both members B-rough and n + 2 < B^2 gives a twin); S12 the
crossing (min leftover 0 iff the record run reaches L), with the hypothesis "some slot beyond every
x is open" discharged for the engine by an explicit open column; and the section identity (below
(q + 1)^2 the leftover of {5..q} on a stretch is exactly its twin count). Consequences for the
tree: the stratification of 2.3 rests on kernel theorems at every step but the onset law itself
(which is one line: n = P_1 P_2 > P_1 Q); the FACT "the record stretch's leftover members are P or
P_1 P_2" is a theorem for every section with Q^2 < B^3, which holds at every computed link (it
needs L > Q^{2/3}/6, true since the record grows at least like the first twin gap above the cut);
and S12's ROOT mark is now exact in the kernel: min K_L > 0 for L above L_0 IS R(6L + 1) < L on the
section, and on the quiet part that is the longest twin gap. Nothing in round 40 is a lever; it is
the construction's bookkeeping made exact, which is what the skill asks for.

---

## 3. The untested list, ranked

"Prove" = a positive result would prove the step at a link or the budget at a rung; "sharpen" = it
would move a verdict or fill a measured law's range. Cost is wall time on one or two cores.

**U1. Fill base_and_step.md Part II from the files on disk (node R4.d.i).** Known: the lane died
three times; generated.json, record_scan.json (about 3,400 sections with q'^2 <= 10^9),
chains_all.json and twins_1e9.npy exist in research/stack/r2/results/ and no verdict was ever
written. Test: fill Q1-Q7 and O3 exactly as pre-registered - in particular Q5 (the section record
against the section at every prime q to 31,622: the ratio, the band [0.5, 4] (ln q^2)^2) and Q6
(the chain-pair ratio record(k+1)/record(k), median in [2, 8], no determination by the previous
record). Cost: minutes (reading JSON). Rank 1 because it is the only step-shaped table the project
has and it is already computed; it decides whether the composite record along the chain has any
law beyond the twin-gap growth. Sharpen.

**U2. The order law out of sample at 37 -> 41 (node 4.i.b.ii).** Known: k* = L + 1 at 9 of 9
computable rungs, B_{L+1} <= F + q' at 9 of 9, B_L > F + q' at 7 of 7; 37 -> 41 undecided (B_2 = 161
> 129, k* predicted 3); Phi = B_{L+1} is the only functional of eleven that bounds the next record
and is budget-monotone; its theorem implies the budget at every rung, hence (2.1a) the step at every
link. Test: build D_3(m37) by the closure (4.i.b.i built D_4 at m41 in 1,992 s and 2.4 GB) and
compute B_3(m37; 41): the law predicts B_3 <= 129 = F(37) + 41 and B_2 > 129. Cost: about an hour,
3 GB. Rank 2 because it is the one named lemma on the engine whose truth is decidable out of
sample and whose theorem would be a proof shape; it stays face E (the budget over-asks), so a
positive result proves rungs, not the conjecture. Prove (rung by rung).

**U3. The frontier floor to 10^7 without a frontier scan (nodes R4.c.iii.a, R4.d.i.a).** Known: E8
reduces H_4.625(q') to H_4.625(q) plus the straddling condition; the first-twin scan bounds the
straddling run's upper half (offset below (2q' + 1)/3, 0 exceptions to 10^7); the last twin below
each square is in the same sieve; the condition follows from tau <= 0.0833 and the arc bound for
q' >= 41 (2.1d). Test: from the existing sieve to 10^7 compute, at every rung, x_s and L_s of the
straddling run and tau = L_0/W; report min x_s/L_s over rungs > 19,997 and max tau. Cost: minutes
(the sieve exists). Rank 3 because it converts two measured facts into a third at 500 times the
range with no new scan, and if x_s/L_s ever falls below 4.625 that rung is a new brick. Sharpen.

**U4. N(v) <= F_2 and the J-run outer law at m37 (node 2g.i).** Known: exceptionless to m31 (6.4
billion gaps) and m23 (3.3 million runs), tight once, mechanism proved (glue), no proof of the law;
the closure dictionary gives the m37 adjacent-pair and window rows scan-free, F_2(37) = 90 exact.
Test: max over realised v >= 6 of N(v) at m37 against 90; the J-run outer law at J = 3..5 on the
m37 dictionary. Cost: minutes with the r61/r66 dictionaries. Rank 4: the two laws are the
engine's only exceptionless measured laws with a proved mechanism and no proof; one more machine
either keeps them or names the first failure. Sharpen.

**U5. W32's first-hit exactness on the engine's own window (node R4.b.iv).** Known: for a fixed
gear set the range record is the first hit on the census, F_range(N) = max{d : W/c(d) <= N} - 1,
within one unit at 19 of 21 checkpoints on three manifold wheels; "the exactness is the finding
and the thing to try to break"; never tested on the engine, whose window is the one phase-zero
translate. Test: at y = 7..53 (full periods and gap censuses exist) compare F_W(y), the window's
longest twin gap, with the first-hit prediction from the engine's own gap census over W(y)
columns; and the same for the section records of U1. Cost: minutes. Rank 5: S13 found the real
minimum an extreme value of the count "like random phases"; W32 is the same claim for the record
and it is the one place a real-teeth exception would be a which-residues fact. Sharpen.

**U6. The finish census across sections (node R4.d.i.b).** Known: at two records every finish is
fuel-type with a single tail prime as air (2.3). Test: for every section record of U1's scan
(about 3,400), the type census of the record's leftover slots (PP / P+C / C+C), the kind of each
finish (ember / fuel) and the air P_1 against the tail's size distribution; the prediction from the
count is that airs are distributed as the tail primes weighted by 1/P_1. Cost: minutes after U1.
Rank 6: it tests whether "who finishes" has any structure beyond the count; a preference for
particular airs would be new. Sharpen.

**U7. The one-tooth free-phase manifold adversary's full reach R(q, Q) (unstick 3, (c)(ii)).**
Known: 776 of 1,226 gears empty 60 turns at Q = 10^4; R(q, Q) < Q for every Q is the root at rung Q
in the exhaust's coordinate; it is the missing third column (real <= fixed-domino free-phase <=
free-class A072753) of the three-record table. Test: greedy with all gears at Q = 10^4, 3 x 10^4,
10^5, and an ILP certificate at small m the way K(d) was certified; R/Q against Q. Cost: an hour.
Rank 7: it measures the slack in turn units and completes the adversary hierarchy; no route.
Sharpen.

**U8. The moving anchor: rigidity of the effective machine inside the window (7b generalised;
named at R2.c.i and R4, never done).** Known: the anchor {5..13} is rigid in every window (proved
from the interval discrepancy of 180 re-toothed anchors); R2.c.i's head rho_g = 1 holds exactly
for gears whose lower period times g fits the window; R4 named "the rigidity generalisation" as
the first thing to formalise. Test: prove, for the machine {5..sqrt(6k)} at every column k of the
window, that its openings sorted modulo any larger gear miss their fair share by at most the
number of runs of the pattern (a finite discrepancy bound per machine), and measure the constant
against 7b's 30. Cost: a day of proof-writing, minutes of measurement. Rank 8: it would turn two
OPEN sentences into a theorem and formalise the one proved part of a DEAD node that is still in
use; it is a count-type bound and cannot by itself reach the window (face A4). Sharpen.

**U9. Words, not columns (unstick 2 item 4, never opened).** Known: the glue and separability
asked column questions and died; the engine's grammar is in gap words; the concatenation of the
two flanks' words at a junction is a word of M iff the junction letters are legal. Test: at the
resistant m29 run (18, 10, 30) and the 106 hard runs of separability.md, whether the concatenated
flank word is realised in the period (exact word search at m11..m23), which bounds L + R by F_2
whenever it is. Cost: minutes. Rank 9: cheap, never run, and the only untried coordinate on the
gluability knock (the one face-C exception). Sharpen.

**U10. The antipode's neighbourhood (unstick 1, 7d idea 1, never run).** Known: the antipode
(P +- 1)/2 is open at every machine (kernel, antipode_open) with every gear at phase (g +- 1)/2, the
opposite of zero; the region past zero is thinner than the mean (0.79). Test: opening density and
longest run in (antipode, antipode + W] against the period mean and (0, W] at m11..m23. Cost:
minutes. Rank 10: a rich computable region would be a new brick, but a period-scale one (face 5m
says such constructions never reach the window). Sharpen.

Not listed because they are counts or the root: the two-dimensional Buchstab closed form for
S14's deficit (a formula for a count); the section-length minimum of K (the root); Omega with
gears >= 13 to n = 26 (rich_half's own open item, low value after E4); the crossover height O-X6
(a citation, not a test).

---

## 4. The staring-at-us check

**Does any combination of proved results already imply the step at any link?** Yes, at exactly
three links, by 2.1(a) and the certified ladder: the step at link k is the window statement at
rung q = prevprime(nextprime(p_k^2)) (prefix form), so F({5..q}) < W(q) proves it. F is certified
at q <= 59 (with the F(59) caveat of M9). The links whose rung is at most 59 are those with
nextprime(base^2) <= 61, i.e. bases 3, 5, 7, link 1 only:

| link | section | q | F({5..q}) | W(q) = (q'^2 - 1)/6 | composite record measured (slots) |
|---|---|---|---|---|---|
| base 3, link 1 | [9, 121) | 7 | 5 | 20 | 4 |
| base 5, link 1 | [25, 841) | 23 | 34 | 140 | 24 |
| base 7, link 1 | [49, 2809) | 47 | 118 | 468 | 27 |

base_and_step.md's Q7 reached the same three links through the free-class table h_2 (30 < 112,
366 < 816, 1284 < 2760 in numbers); the real-teeth ladder is the tighter and is the project's own.
No fourth link is reachable: base 11's link 1 needs F({5..113}) < 2688 and base 3's link 2 the same
rung; base 13's link 1 needs F({5..167}) < 4988; the corpus stops at 59, the SAT instrument gives
lower bounds only past m41, and the only proved upper bounds on the paired record are Iwaniec-type
with exponent 4.27 against the section's exponent 2. Every link from the second on, of every
chain, is the window statement at a rung above the ladder: measured true (the composite record is
1.3 x 10^-5 of the section on base 3), not proved.

**Does anything on record bound the composite record on a section?** Only the trivial bound by
the engine's period record at the section's rung, F({5..q}), which is itself unbounded beyond the
ladder; and the free-class record h_2 at the same gears, exact to p_n = 73, conjectured
(Ziller-Morack Conjecture 6) to stay below p_n^2 - p_n, which would give every link (on record
since 2026-09-06). The core-side results (the loaded record rule, S12, the section identity) bound
the FREE-phase record (25,267 slots at L* on base 3) and identify the real-phase minimum, but
neither bounds the real-phase record from above: S12 says min K_L = 0 iff R(B) >= L, which is the
statement to be bounded, not a bound.

**The smallest missing lemma, in the construction's terms.** For every cut c = p_k^2 and its root
Q = p_{k+1}: the primes below Q, striking their multiples, leave no run of (Q^2 - c)/6 consecutive
struck slots beginning in [c, Q^2). By round 40 this is exactly "min over x in the section of
K_{L_sec}(x) > 0" for the core B >= Q (the whole machine), i.e. R(Q) < (Q^2 - c)/6 on the section,
and since every slot of the section is below Q^2 the section identity makes it "the section holds
a twin". There is no proper sub-lemma of it on the tree that is not itself the root at another
rung or a count: the stratification of 2.3 shows the tail is silent below B^2, so the lemma's
bottom stratum is "the twin gaps in [c, B^2) are shorter than (B^2 - c)/6" (the root on the quiet
part), and above B^2 it is a count of coincidences (the tail's finishes against the core's
leftovers, T >= K everywhere yet one stretch finished). The recursion (the gears are the survivors
of the sections below) enters only through the mean (S14, the origin's density deficit, which
lowers the leftover) and through rigidity (S11, the core is the primes up to prime powers, so no
counterfactual coprime core exists): both are now kernel theorems, and neither is an inequality
on the record. The honest name for what is missing is the same as the wall's: a lower bound on
twin density in a short interval above the cut (face A at s = 2), or an invariant of the
survivor set that forces two survivors at distance 2 without counting (skeleton 11b), which no
node has identified. The one new sentence this review can add is 2.1(b): the step at link k+1
would follow from a bound on the twin gap across the cut p_{k+1}^2 of 21.6% of its position (the
straddling condition), a strictly stronger local statement whose upper half is measured to 10^7
with room by a factor of about q'/(ln q')^2.

---

## 5. Small housekeeping items found on the way

- The tree's dead-ends list and the wall's section 6 agree; the wall's 5m needs the M5 amendment.
- law_register.md's map line "future documents number from W103" is respected by W103 (the
  manifold record, manifold_census_large.md), but W103 is registered only in the log and the
  wall, not in the register's tables; add the row.
- S-numbers: S8, S9 (stacked_squares.md), S10-S14 (core_leftover.md); base_and_step.md Part II
  says "laws numbered S10 onward" and issues none; when it is filled it must start at S15.
- core_leftover.md line 12 says "outputs research/stack/r5/results/ (untracked)" while the folder
  is present locally with sections.json, lscan.json, followup.json, free.json; fine as long as the
  1 MB guard holds (lscan.log and sections.log should be checked before any commit).
- research/stack/r5/leftover_depth.py and leftover_shadow.py (untracked, git status) belong to
  the unstick-4 lane and have no document yet; the tree's log entry of 2026-09-10 names the lane.
