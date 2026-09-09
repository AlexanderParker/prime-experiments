# Dead branches reopened, fourth pass: the step at the core (lateral lane, 2026-09-10)

The unstick protocol with the shadow step (SKILL.md, "Parts, interfaces, shadows") run over the
newest blocker: the step at the core (research/proof/step_evidence.md section 7,
research/proof/core_leftover.md). Format as in dead_branches_reopened_3.md; the faces A-E of
the_wall.md and the bricks N1-N11 of the earlier passes apply to every idea. Thinking and small
spot checks only (two cores, seconds to two minutes each); every number used is in this file. No
tree or index edits; nothing committed.

Vocabulary (the stack by squares). A section [c, c') of a chain, c' the square of the first prime
at or above c; the machines below the cut c strike the section; a SLOT is (n, n + 2), n = 5 mod 6;
a STRETCH is L consecutive slots; L is the section's record (the longest twin-free stretch), and
t = 6L + 1. The CORE is the primes in [5, t], the TAIL the primes in (t, c); a tail gear has at
most one multiple among the 6L numbers a stretch spans. A member is CORE-FREE if no core gear
divides it; a slot is CORE-OPEN if both members are core-free; K is the number of core-open slots
of a stretch (the core's leftover). The DEPTH of a number n in the core is u = ln n / ln t. The
core's SURVIVORS are its core-free numbers; below t^2 they are the primes (the exhaust cap, kernel
CoreLeftover.twin_of_rough); S = the primes in (t, t^2). With the tail as an engine acting on the
core's open set: a composite core-free member P1 P2 is a CHARGE with AIR P1 (the smaller factor, a
tail gear) and FUEL P2 (the larger); its family is (P1, 1) when the partner is prime, (P1, P1') when
both members are composite; the pure charge (1, 1) is a twin.

## The blocker, exact (the object of this pass)

"The step at the core is that among a stretch's leftover slots, whose members are P or P1 P2 with
all primes above t, at least one is PP; the size K of the leftover set is an extreme value of a
count with no structural excess (S13); the TYPE of the leftover members (P against P1 P2) is what
decides, and a sign-sensitive count cannot separate them (parity)."

## Scripts and spot checks

research/stack/r5/leftover_shadow.py (per section: the record's depth, its composite members
factored, the sliding census of core-free members by primality, PP given K, the cover at the
record, the tail's strikes on core-free numbers); leftover_depth.py (prime share and PP share by
depth bin; the independent-slot prediction of twin-free stretches per bin; PP share by K);
leftover_indep.py (PP among K leftovers against the binomial, by K; twin-free starts and RUNS
against the independent-slot prediction at shorter lengths). Outputs in the lane's scratch
directory; the numbers are below.

### Table 1. Every computed record stretch: depth, tail, leftover

| base | section | L | t | core | tail | record at n | depth of record | depth of whole section | K at record | PP / P+C / C+C | K mean | composite core-free members per stretch, mean / max / record | composites in core-open slots, mean / max / record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | [9, 121) | 4 | 25 | 2 | 0 | 77 | 1.35-1.42 | 0.68-1.49 | 0 | 0/0/0 | 1.6 | 0 / 0 / 0 | 0 |
| 3 | [121, 16129) | 46 | 277 | 28 | 0 | 13,403 | 1.69 | 0.85-1.72 | 0 | 0/0/0 | 4.7 | 0 / 0 / 0 | 0 |
| 3 | [16129, 260,467,321) | 579 | 3475 | 485 | 1,390 | 255,949,955 | 2.37 | 1.19-2.38 | 12 | 0/9/3 | 20.4 | 41.8 / 87 / 59 | 7.39 / 28 / 15 |
| 5 | [25, 841) | 24 | 145 | 7 | 0 | 665 | 1.31-1.34 | 0.65-1.35 | 0 | 0/0/0 | 4.7 | 0 | 0 |
| 5 | [841, 727,609) | 167 | 1003 | 144 | 0 | 688,457 | 1.95 | 0.97-1.95 | 0 | 0/0/0 | 8.6 | 0 | 0 |
| 7 | [49, 2809) | 27 | 163 | 13 | 0 | 2,387 | 1.53-1.54 | 0.76-1.56 | 0 | 0/0/0 | 4.2 | 0 | 0 |
| 7 | [2809, 7,946,761) | 254 | 1525 | 239 | 168 | 4,869,917 | 2.10 | 1.08-2.17 | 1 | 0/1/0 | 10.4 | 6.4 / 28 / 8 | 1.20 / 10 / 1 |
| 11 | [121, 16129) | 46 | 277 | 28 | 0 | 13,403 | 1.69 | 0.85-1.72 | 0 | 0/0/0 | 4.7 | 0 | 0 |
| 13 | [169, 29,929) | 82 | 493 | 37 | 0 | 24,425 | 1.63 | 0.83-1.66 | 0 | 0/0/0 | 7.4 | 0 | 0 |
| 17 | [289, 85,849) | 104 | 625 | 59 | 0 | 62,303 | 1.71-1.72 | 0.88-1.76 | 0 | 0/0/0 | 7.7 | 0 | 0 |
| 19 | [361, 134,689) | 104 | 625 | 70 | 0 | 62,303 | 1.71-1.72 | 0.91-1.83 | 0 | 0/0/0 | 7.1 | 0 | 0 |
| 23 | [529, 292,681) | 153 | 919 | 97 | 0 | 187,913 | 1.78 | 0.92-1.84 | 0 | 0/0/0 | 9.2 | 0 | 0 |
| 31 | [961, 935,089) | 241 | 1447 | 160 | 0 | 850,355 | 1.88 | 0.94-1.89 | 0 | 0/0/0 | 11.9 | 0 | 0 |

The slot holding the next cut is dropped (core_leftover.py's convention), so slot counts are one
below supply_demand.py's. In every section the top c' is below t^3 (members are P or P1 P2, the
two-prime lemma, kernel CoreLeftover.primeOrSemiprime_of_rough_lt_cube).

### Table 2. The record's composite members (base 3, section 4; t = 3475, cut 16129)

| n | air P1 | fuel P2 | fuel in tail (< 16129)? | partner prime? |
|---|---|---|---|---|
| 255,950,207 | 14,173 | 18,059 | no | no (C+C with the next) |
| 255,950,209 | 11,903 | 21,503 | no | no |
| 255,950,237 | 5,987 | 42,751 | no | no (C+C) |
| 255,950,239 | 4,481 | 57,119 | no | no |
| 255,950,687 | 13,001 | 19,687 | no | no (C+C) |
| 255,950,689 | 6,037 | 42,397 | no | no |
| 255,950,803 | 12,583 | 20,341 | no | yes |
| 255,951,041 | 11,923 | 21,467 | no | yes |
| 255,951,341 | 3,931 | 65,111 | no | yes |
| 255,951,461 | 8,291 | 30,871 | no | yes |
| 255,951,587 | 5,693 | 44,959 | no | yes |
| 255,952,093 | 4,027 | 63,559 | no | yes |
| 255,952,139 | 4,457 | 57,427 | no | yes |
| 255,952,243 | 14,159 | 18,077 | no | yes |
| 255,953,149 | 15,289 | 16,741 | no | yes |

All 15 airs distinct, all 15 fuels distinct, all 30 prime factors distinct; every fuel in
(16129, 65,111), i.e. a prime of the section itself (a gear of the next machine), and every fuel
below t^2 = 12,075,625 (a core survivor at depth < 2). Base 7's single composite 4,871,171 = 2039 x
2389 has its fuel in the tail. The cover at the base-3 record: 12 core-open slots, 15 composites,
12 of 12 slots hold a composite, 3 hold two (the C+C slots): loose by 3.

### Table 3. The census by depth (base 3, section 4; twelve equal bins in u)

| bin | depth | core-free members | prime share among them | core-open slots | PP share among them | starts of length 579 | twin-free stretches predicted by independent slots | actual |
|---|---|---|---|---|---|---|---|---|
| 0-7 | 1.19-1.98 | 681,641 | 1.0000 | 60,294 | 1.0000 | 1,714,914 | 0.00 | 0 |
| 8 | 1.98-2.08 | 799,222 | 0.9647 | 65,738 | 0.9310 | 2,133,566 | 0.00 | 0 |
| 9 | 2.08-2.18 | 1,854,957 | 0.8887 | 158,075 | 0.7890 | 4,783,831 | 0.15 | 0 |
| 10 | 2.18-2.28 | 4,267,801 | 0.8277 | 373,654 | 0.6860 | 10,726,195 | 5.52 | 0 |
| 11 | 2.28-2.38 | 9,745,368 | 0.7782 | 868,383 | 0.6050 | 24,049,447 | 24.58 | 1 |

Base 7 (eight bins): prime share 1.0000 to depth 1.90, then 0.9945, 0.9081; PP share 1.0000,
0.9874, 0.8276; predicted twin-free stretches 1.07 + 34.01, actual 1 (in the top bin, depth
2.03-2.17, where the record at 2.10 sits).

### Table 4. Independence of the types (the coordinator's lead, tested)

Test 1, PP among the K leftovers against Binomial(K, p), p = 0.6050 the PP share of the record's
bin, base 3 (observed / binomial x starts):

| K | starts | PP = 0 | 1 | 2 | 3 | 4 | 5 | 6 | PP = K |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 5,145 | 0 / 3.0 | 0 / 37.3 | 269 / 200.2 | 787 / 613.3 | 916 / 1,174.4 | 1,338 / 1,439.3 | 1,355 / 1,102.5 | 111 / 92.4 |
| 10 | 38,809 | 0 / 3.6 | 11 / 54.9 | 435 / 378.5 | 1,601 / 1,546.4 | 4,423 / 4,145.7 | 6,507 / 7,621.0 | 9,494 / 9,729.0 | 280 / 255.1 |
| 12 | 164,698 | 1 / 2.4 | 37 / 43.6 | 315 / 367.5 | 1,750 / 1,876.8 | 6,639 / 6,469.0 | 16,827 / 15,856.0 | 28,377 / 28,338.5 | 318 / 396.4 |
| 14 | 525,155 | 0 / 1.2 | 0 / 25.3 | 328 / 252.1 | 1,722 / 1,544.5 | 7,541 / 6,506.8 | 20,890 / 19,935.7 | 46,153 / 45,809.9 | 582 / 462.7 |
| 16 | 1,169,822 | 0 / 0.4 | 0 / 10.1 | 136 / 115.5 | 870 / 825.7 | 3,814 / 4,110.9 | 16,329 / 15,114.1 | 45,874 / 42,448.3 | 334 / 377.3 |

Base 7 (p = 0.8276): K = 3, 2,240 starts: 0 / 11.5, 204 / 165.4, 844 / 793.6, 1,192 / 1,269.6;
K = 5, 18,465 starts: 0 / 2.8, 58 / 67.6, 466 / 648.4, 2,698 / 3,111.8, 7,148 / 7,467.1,
8,095 / 7,167.4.

Test 2, shorter lengths L' with the core fixed at t: predicted twin-free starts (sum over starts
of (1 - p_bin)^K) against observed, and twin-free RUNS (twin gaps >= L') against the prediction
(starts x twin density per slot):

| base 3, L' | mean K | starts predicted | starts observed | runs observed | runs predicted |
|---|---|---|---|---|---|
| 144 | 5.1 | 1,224,778 | 1,227,106 | 30,237 | 29,004 |
| 193 | 6.8 | 365,757 | 369,132 | 8,951 | 8,661 |
| 289 | 10.2 | 34,167 | 36,095 | 806 | 809 |
| 386 | 13.6 | 3,211 | 4,175 | 102 | 76 |
| 434 | 15.3 | 987 | 1,021 | 29 | 23 |
| 479 | 16.8 | 330 | 264 | 8 | 7.8 |
| 529 | 18.6 | 98 | 52 | 2 | 2.3 |
| 559 | 19.7 | 48 | 21 | 1 | 1.1 |
| 579 = L | 20.4 | 30.3 | 1 | 1 | 0.72 |

Base 7: L' = 63: 112,271 / 112,323 starts, 4,414 / 4,091 runs; 127: 8,927 / 8,612, 349 / 325;
190: 707 / 586, 25 / 26; 234: 96 / 73, 6 / 3.5; 254 = L: 35.1 / 1, 1 / 1.28.

### Other counts used below

Base 3, section 4: composite core-free members per stretch 41.8 +- 16.2 (record 59, +1.1 sd);
composites in core-open slots 7.39 +- 3.95 (record 15, +1.9 sd); K 20.4 +- 4.0 (record 12,
-2.1 sd); PP (twins per stretch) 13.7 +- 3.85 (record 0, -3.56 sd); core-free members 231 +- 11
(record 221), of which primes 190 +- 15 (record 162). Of all 3,130,924 composite core-free members
of the section, 553,795 (17.7%) sit in core-open slots (base 7: 6,256 of 33,456, 18.7%). Tail
strikes on numbers per stretch 181 +- 13 (record 174), of which on core-free numbers 41.8 (record
59): the tail's strikes on core-free numbers coincide with the composite core-free members at every
one of the 43,407,953 starts (identity, both sections). P(PP = 0 | K = 12) over the whole section:
1 of 456,049 starts against 0.67 by independent slots at the section's PP share 0.674.

## New bricks (constraints every idea below must respect)

- N12 (Table 1; kernel CoreLeftover.twin_of_rough, twin_of_not_blocked). Below t^2 every
  core-open slot is a twin: the prime share among core-free members and the PP share among
  core-open slots are exactly 1.0000 in every bin below depth 2 (8 of 12 bins on base 3, 6 of 8
  on base 7), and fall above it (0.965, 0.889, 0.828, 0.778; 0.931, 0.789, 0.686, 0.605).
- N13 (Table 1). The tail is empty in 11 of 13 computed sections, and in each the whole section
  lies below depth 2 (top depth 1.35 to 1.95) and the record has K = 0. The two sections with a
  tail have their records at depth 2.10 and 2.37 with K = 1 and 12. The depth of the base-3 chain's
  record grows along the chain: 1.35-1.42, 1.69, 2.37.
- N14 (Table 2). In a stretch of 6L numbers every prime above 6L has at most one multiple, so the
  airs of a stretch's charges are distinct, the fuels are distinct, and no factor is shared: 30 of
  30 at the record. No shared-air structure can exist on a stretch.
- N15 (Table 4). The types of a stretch's leftover slots are independent to the binomial: PP
  among K against Binomial(K, p_bin) at K = 8 .. 16 (Test 1); twin-free runs against the
  independent-slot prediction at every length from L/4 to L (806 / 809, 102 / 76, 29 / 23,
  8 / 7.8, 2 / 2.3, 1 / 1.1, 1 / 0.72 on base 3; 1 / 1.28 at L on base 7). The exact finish is
  not rarer than independence predicts.
- N16 (Table 3). The independent-slot model's 30.3 twin-free STARTS against 1 at L is a start
  count, not an event count: a twin gap of L + r slots contributes r + 1 starts, the record gap
  contributes exactly one because L is its length by definition, and the run count (0.72
  predicted, 1 observed) is the event count. Starts at PP <= 1 come from a handful of gaps and
  cluster by K (37 of the 48 observed PP = 1 starts at K = 8 .. 16 sit at K = 12, the record's
  neighbourhood); that is the same start-clustering, not a structure.

---

## The step at the core (step_evidence.md section 7; core_leftover.md; face A)

- **Object.** The exact finish: on a stretch of L slots with K > 0 core-open slots, at least one
  is a twin; equivalently, the tail cannot strike every core-open slot of any stretch as long as
  the section's record. The object that decides is the type of each core-free member (P against
  P1 P2), since the size K is an extreme value of a count (S13) with no excess.
- **Vectors.** Supply against demand (step_evidence.md section 6: supply never binds, 100% and
  99.95% of stretches have T >= K); the real-phase minimum of K (core_leftover.md: an extreme
  value, z in [4.2, 4.7], the same for real phases, random phases and integer sets; min K_L = 0 iff
  R(6L + 1) >= L); the covering structure (withdrawn: strikes were counted, not leftovers); the
  type census (section 7: 0 / 9 / 3 against 13.9 / 5.6 / 0.7). This pass adds: the depth
  coordinate (Tables 1, 3), the factor structure of the charges (Table 2), the independence of
  the types (Table 4), the identity tail-strikes-on-core-free = composite core-free (43.4 M
  starts).
- **Failure, precisely.** Every measured quantity of the leftover is a count that behaves as an
  independent count: K is an extreme value of its own distribution (S13); the number of composite
  core-free members per stretch is at +1.1 sd at the record; the number of composites in core-open
  slots is at +1.9 sd; the types are binomial given K (N15); the twin-free run count is the
  independent-slot prediction at every length (N15, N16). The record is a stretch with K at the
  2nd percentile whose 12 core-open slots each happen to hold one of 15 charges, and 0.72 such
  events were expected: a coincidence of independent placements, in the count's own terms, with
  no rarity beyond independence. What the measurements left untouched: (i) the depth coordinate
  as the parameter of the type census (N12: below depth 2 no decision exists at all); (ii) the
  fuel's identity (Table 2: every fuel of the record is a prime of the section itself); (iii) the
  general object above depth 3, which no computed section reaches.
- **The shadow.** The blocker is the outline of THE SECOND-ORDER HAND-UP. The hand-up (skeleton
  section 6) says a section's survivors become the next machine's gears; below depth 2 of a core
  its survivors are exactly the primes and their only strikes on the core's open set are home
  strikes (the exhaust cap; that is why no decision exists there, N12); between depth 2 and 3 the
  core's survivors S strike the core's open set at the products of two of them, S.S, and nowhere
  else (the identity of Table 4's footnote: a tail strike on a core-free number is P1 P2 with P2
  prime, since P1 P2 P3 > t^3 > c'). So the P-against-P1 P2 decision is "is this core-free
  number a survivor or a product of two lower survivors", and the step at the core reads:

  **the products of two of the core's survivors cannot meet every core-open slot of a stretch as
  long as the section's record.**

  The object is S.S restricted to the stretch, with S the primes in (t, t^2), and the recursion
  appears in it three times exactly: the airs are core survivors in (t, c) (the tail; c < t^2 in
  every computed section, so the tail is the core's survivor set there by the exhaust cap); the
  fuels are core survivors in (t, c'/t) (all 15 below t^2 at the record); and near the top of a
  section, where every record so far sits (255.9 M of 260.5 M; 4.87 M of 7.95 M), the fuel is above
  n / p_k > c and so is a gear of the section's OWN machine (15 of 15). The composite that kills a
  core-open slot at the top of section k + 1 is (a gear of machine k) x (a gear of machine k + 1
  from the bottom of the section). Candidates for the object the blocker outlines, each developed
  below: (a) the tail's charge set on the core's open set (S.S as a machine); (b) the cover of the
  core-open slots by S.S (a matching, not a graph); (c) the Omega-census of core-free members at
  depth u (the sign's true object; P against P1 P2 is its slice at 2 < u < 3); (d) the depth of
  the stretch in its own core (the parameter the type census depends on, exactly).

### (a) The two-prime members as charges of the tail

Definition. The tail T = primes in (t, c) as an engine acting on the core's open set on the
stretch; a composite core-free member is a charge P1 P2 with air P1 in T (P1 <= sqrt(n) < p_k, so
the smaller factor is always a tail gear) and fuel P2 in S = primes in (t, t^2); families (P1, 1)
(partner prime, P+C), (P1, P1') (both composite, C+C), (1, 1) (PP). The valves' proved laws read
here as: IMPRINT, the family (P1, .) lives in the class 0 mod P1 (trivial); ONSET, family (P1, .)
opens at P1 x nextprime(t) > t^2, so no family is open below depth 2 (this IS the depth law, N12);
PORT, P1 P2 is a lower member iff P1 P2 = 5 mod 6, a residue fact; INVENTORY, which (P1, P1')
occur as C+C (P1 P2 + 2 = P1' P2'); EMBER, the charge whose fuel is also in the tail (base 7's
2039 x 2389; 0 of 15 on base 3). The turn ledger's counterfactual (N8, V17) transfers verbatim:
none of these laws uses that S is the prime set; replace the fuels by any set of t-rough numbers
above t with no two at distance 2 from a core-free number and every law holds with PP = 0.
Nothing forces the pure charge from the laws.

The new thing, tested. The airs must be core survivors: they are (all 15 are primes above t), and
the test of clustering is decided by N14: each prime above 6L has at most one multiple in the
stretch, so airs, fuels and all 30 factors are distinct. There is no shared-air structure to find;
the exact finish is 15 charges with 30 distinct prime factors landing on 12 slots. What (a)
leaves as an object: the fuel set of the record is {18,059, ..., 65,111}, primes of the section's
own first 0.03%, and P2 > n / p_k is exact for every charge at the top of a section.

Realisations.
- (a)(i) The fuel map. For every twin-free stretch of length L' (L' = L/2 .. L, hundreds to
  thousands of them on base 3, Table 4) list the fuels of its charges and their position in the
  section (bottom primes of machine k + 1 against tail primes): if the records of a section are
  always finished by the section's own bottom primes, state it as the exact consequence of
  P2 > n / p_k and measure how far below the top it holds (the fraction of charges with fuel above
  c as a function of n / c'). Cheap; the arrays of leftover_shadow.py. Reopens: the section's
  bottom (where the first twins sit within cycles of the cut, step_evidence.md section 1) as the
  supplier of the top's fuels; a connection between the two ends of one section, exact, whose
  content is a count unless the bottom primes' residues mod the tail gears carry a law.
- (a)(ii) The counterfactual fuel one tier down: replace S by the one-tooth free-phase adversary's
  survivors (N11) or by V17's saturated set and measure the twin-free run count at L' against
  Table 4's: it will match the independent prediction too (N15 says the real S already does).
  Value: closes (a) as a route with one number; not opened unless someone claims S's primality
  matters to the finish.

### (b) The pairing of survivors at distance 2 as a graph

Definition. Vertices the core-free numbers of the stretch, an edge between two at distance 2.
Fact (two lines): a number coprime to 6 lies in exactly one slot (5 mod 6 as lower, 1 mod 6 as
upper), so the graph is a matching of K disjoint edges and the minimum vertex cover by composites
is K, one per slot; a stretch is twin-free iff every edge has a composite endpoint. At the record:
12 edges, 15 composites, 12 of 12 covered, 3 spare (the C+C slots): loose by 3 (Table 2). There is
no graph structure to use; the cover question is K independent one-slot questions, which is N15.

Realisations.
- (b)(i) The incidence tail gear -> core-open slot (a tail gear is adjacent to the core-open slot
  its unique multiple in the stretch falls in, if that multiple is core-free). Twin-free iff the
  incidence is onto the K slots. The positions of the tail's multiples are x mod P1 for each P1,
  and "which land on core-free numbers with prime cofactor" is exactly S.S again; the bipartite
  degree sequence (how many tail gears hit each core-open slot: 1 for the P+C slots, 2 for C+C)
  is measurable on every twin-free stretch of Table 4; a law in it (e.g. C+C slots forced at a
  rate) would be new; the expected result is the independent rate 0.74 / 20.4 per slot.
- (b)(ii) The chain of open numbers at distances 2 and 4 (5 mod 6 to 1 mod 6 to the next 5 mod 6):
  the twin-free condition uses only the distance-2 edges; the distance-4 and distance-6 edges pose
  the same question for other constellations (S.S covering the "sexy" pairs), with the same
  independent answer expected. Noted; not a lever.

### (c) The sign

Definition. The census of core-free members by number of prime factors Omega. At 2 < u < 3 it is
P (Omega = 1) against P1 P2 (Omega = 2), opposite Liouville signs; the signed census cancels
(phase_zero.md: 0.1% against a pure share of 8.2%). The exact upper bound on the P1 P2 members of a
stretch: each tail gear has at most one multiple in the stretch, so the count is at most the
number of tail gears whose multiple is coprime to 6 with prime cofactor, at most min(#tail, 2L) =
1,158 on base 3; measured max 87 (mean 41.8, record 59), and in core-open slots max 28 (mean 7.39,
record 15). The bound is a factor 13 above the maximum ever attained: the composites are not the
constraint, and the record is at +1.1 sd in composites and +1.9 sd in composites-in-open-slots.
What constrains the P1 P2 census beyond its density, exactly: (1) onset, none below t^2 (N12);
(2) each prime factor at most once per stretch (N14); (3) fuel above n / p_k, so at the top of a
section the fuels are the section's own primes (Table 2); (4) the prime share among core-free
members falls with depth, 1.000, 0.965, 0.889, 0.828, 0.778 across the base-3 bins (the Buchstab
profile of rough numbers at 2 < u < 3; prior art, one line); (5) nothing else: given K and the
bin's share the types are binomial (N15). The deficit at the record is the twin count's own
extreme value: PP per stretch 13.7 +- 3.85, z = -3.56 at 0, and the run count 0.72 against 1.

The general object. The reading "P against P1 P2" is a slice of one link. The record's depth grows
along the chain (1.35-1.42, 1.69, 2.37; N13), and at the next link of the base-3 chain, [2.6 x 10^8,
6.8 x 10^16), with L of order 2,000 slots by the ln^2 scaling of twin gaps (579 slots at 2.6 x 10^8
is 9.3 x ln^2 c'), t is of order 1.3 x 10^4 and u = ln(6.8 x 10^16) / ln t is about 4: core-free
members then have Omega up to 4, and P1 P2 P3 has the SAME Liouville sign as P. The sign-count
does not separate the types because it cannot see Omega at all above depth 3, not because P and
P1 P2 have opposite signs. The object is the Omega-census of t-rough members at depth u, and the
step at the core in general is "among the core-open slots of a stretch, whose members are t-rough
with Omega <= floor(u), at least one has both members at Omega = 1".

Realisations.
- (c)(i) Measure the general object now, without the chain: on the base-3 section take L' small
  enough that t'^3 lies inside the section (L' = 50: t' = 301, t'^2 = 90,601, t'^3 = 2.7 x 10^7), so
  stretches near 10^8 sit at depth 3.2 of their own core and members with Omega = 3 appear; the
  Omega-census of core-open members by (L', u) over the reachable region 1 < u < 3.5, the PP share
  among core-open slots as a function of u alone (prediction: it depends on u only, not on L' or
  the section, to the accuracy of the Buchstab profile), and the twin-free run count at (L', u)
  against the independent prediction with that share. Two cores, minutes. Result that counts: a
  dependence on L' or on the section at fixed u (a trace of the construction beyond depth), or a
  run count off the independent prediction at depth > 3.
- (c)(ii) The exact upper bound as a law: the P1 P2 count in a stretch of 6L numbers is at most
  the number of tail gears with a multiple in it, and its expectation is 6L x sum over the tail of
  1/P1 x (prime-cofactor rate): 2L x (ln ln c - ln ln t) x (1 / ln(n / P1)) is about 42 on base 3
  (measured 41.8). A formula for a count; filed with the bound. Not a route.

### (d) The toolbox: mirror, set algebra, depth

The mirror n -> -n - 2 mod P_core swaps the two teeth of every core gear, so the core-open slot
pattern is mirror-symmetric on its period and P+C slots map to C+P slots of the mirrored stretch;
but the mirror image of the section [c, c') is the stretch [-c', -c) mod P_core, outside the
section, and the tail's period is not P_core: the mirror says nothing about which members are
prime. Not forcing.

Set algebra (exact, verified at every start of both sections): leftover twin-free slots = core-open
slots minus tail-struck slots; tail strikes on core-free numbers = composite core-free numbers =
S.S on the stretch (the identity of Table 4's footnote, 43,407,953 + 1,323,738 starts, 0
exceptions); so twin-free iff S.S meets every core-open slot.

**S15 (THE DEPTH LAW, slot form; proved; kernel CoreLeftover.twin_of_rough / twin_of_not_blocked,
round 40).** A core-open slot (n, n + 2) with n + 2 < t^2 is a twin. Hence a twin-free stretch of L
slots with K > 0 core-open slots has a composite core-free member m, m >= nextprime(t)^2 > t^2, so
its top exceeds t^2: it lies at depth > 2 of its own core. *Proof.* A t-rough composite has two
prime factors > t, so exceeds t^2. QED. *Verified.* Table 3: PP share 1.0000 in every bin below
depth 2 on both sections; the two records with K > 0 at depth 2.10 and 2.37; every K = 0 record at
depth <= 1.95 (Table 1).

**S16 (THE TAIL-EMPTY DICHOTOMY; proved).** On a section [c, c') with record L and t = 6L + 1: the
tail is empty (no prime in (t, c)) iff every core-free member of the section is prime; then every
core-open slot of the section is a twin, every twin-free stretch has K = 0, and the record is the
core's own composite record (S12: R(t) >= L). *Proof (forward).* A composite core-free member is P1 P2 with
t < P1 <= sqrt(n) < p_k (the first prime >= c); if there is no prime in (t, c) there is none in
(t, p_k) either (p_k is the next prime after the last prime below c), so P1 cannot exist. QED.
*Converse, with its exact hypothesis.* A tail gear P1 in [c / nextprime(t), c) gives the composite
core-free member P1 x nextprime(t) inside [c, c') (it is below c x t < c' whenever t < c, which
holds as soon as the tail is nonempty), so "tail nonempty" gives a composite core-free member in
the section exactly when the tail reaches above c / nextprime(t); on base 3 (c / nextprime(t) =
16129 / 3479 = 4.6) and base 7 (2809 / 1531 = 1.8) every tail gear qualifies. *Verified.* 11 of 11 tail-empty sections have K = 0 at the record and no
composite core-free member anywhere (Table 1, the column "composite core-free members" is 0 / 0 / 0
throughout); the 2 sections with a tail have 3,130,924 and 33,456 composite core-free members and
K = 12, 1 at the record.

Consequence (the location of the step). For a section whose record has 6L + 1 above the last
prime below c (S16), the step at that link is exactly the composite record statement of the core
at the lower cut: R(6L + 1) < L is false there by definition of L, and the step reads "the core's
own record R(t) is shorter than the section", the record law at the core's scale. For a section
whose record has 6L + 1 < c (a tail exists; base 3 from section 4 on, base 7 from section 3 on,
and every larger section, since L grows like ln^2 c' while c grows like c'^(1/2)), the whole
section lies above depth 2 of the core (c = p_{k-1}^2 > t^2), K > 0 is possible, and twin-free needs
S.S: the step needs the products of survivors. The first kind is where every chain starts, the
second kind is where every chain ends up, and the record's depth grows without bound along the
chain (N13; about 4 at the next base-3 link).

Realisations.
- (d)(i) The depth law along the chain as a law in the part's intrinsic parameters (L, u): the
  minimum K of a twin-free stretch of length L' at depth u, and the type census there, over the
  (L', u) region reachable on the base-3 section (as (c)(i)); the exact form of "K = 0 iff
  R(t') >= L'" per depth (the core's own record as a function of depth). Reopens the core's record
  R(t) as a function of position, which S12 stated at the section only.
- (d)(ii) The formal side: S16's converse with its exact hypothesis (a tail gear above
  c / nextprime(t)) and S15's slot form are kernel-ready one-liners on top of CoreLeftover;
  S16 forward is a Formalist item of a few lines. Value: fixes the search space (tail-empty
  sections are the composite record statement; the rest need S.S), and stops "what about the
  small sections" from reopening.

---

## Scorecard (pre-registered in the brief, filled here)

| # | prediction (brief or coordinator) | verdict |
|---|---|---|
| (a) airs are core survivors | **held**, trivially (primes above t); clustering **impossible** (N14: 30 of 30 factors distinct) |
| (a) the valves' laws force nothing here | **held** (N8 / V17 transfer verbatim; onset = the depth law) |
| (b) the cover is tight | **refuted**: loose by 3 (12 of 12 slots covered, 15 composites); the graph is a matching, no structure |
| (c) an exact bound on P1 P2 members far above the record's 15 | **held**: bound 1,158, measured max 87 (open slots 28), record 59 (15) |
| (c) the PP deficit is a fluctuation of size z about -4 | **held**: z = -3.56 unconditional; P(PP = 0 | K = 12) 1 of 456,049 against 0.67 |
| (d) a twin-free stretch with K > 0 lies at depth > 2 | **held**, proved (S15), 13 of 13 records (Table 1) |
| (d) base 3 depth 2.38, base 7 2.11, base 23 K = 0 | **held**: 2.37, 2.10, K = 0 at depth 1.78 |
| coordinator: the exact finish is about 40 times rarer than independence predicts | **refuted** (N15, N16): the 30.3 against 1 is starts against one run; runs 0.72 against 1; PP given K binomial at every K tested; twin-free runs at every L' from L/4 to L on the independent prediction (Table 4) |

## Reading the file as a whole

1. **Every candidate returns to one object, S.S on the core's open set: the products of two of
   the core's survivors.** The tail's charge set (a), the cover (b), the composite census (c) and
   the set algebra (d) are the same set in four coordinates, and its recursion content is exact
   and threefold: airs are survivors in (t, c), fuels are survivors in (t, c'/t), and at the top
   of a section the fuels are the section's own gears (15 of 15). The sharpest reformulation of
   the step at the core: **the products of two survivors of the core cannot meet every core-open
   slot of a stretch as long as the section's record.** Its count side is fully independent
   (N15): given K and the depth, the types are binomial, and the twin-free run count is the
   independent prediction at every length (Table 4). ROOT in the count; what is not a count in
   it is the depth (S15, S16) and the fuel's identity (Table 2).
2. **The depth is the exact parameter the type decision depends on, and the P-against-P1 P2
   reading is one link's slice of it.** Below depth 2 there is no decision (kernel); between 2
   and 3 it is P against P1 P2; above 3 the Liouville sign no longer even separates the types
   (P1 P2 P3 has the sign of P). The record's depth grows along the chain (1.35, 1.69, 2.37,
   about 4 next). The general object is the Omega-census of core-free members at depth u, which
   the base-3 section can measure now to depth 3.5 with shorter L' ((c)(i), (d)(i)), in the part's
   intrinsic parameters (L, u) rather than at the section's record.
3. **What died here and what it fixes.** The coordinator's "40 times rarer" lead: dead, by the
   run count (N16); its brick is that twin-free STARTS are never the event count at the record
   length. The cover as a graph: dead, a matching. Shared-air structure: impossible (N14). The
   composite count as the constraint: dead (factor 13 of slack). What these fix: the exact
   finish has no rarity, no clustering, no cover structure and no composite shortage beyond
   independence; whatever is not a count in the step at the core is in the depth and in the
   identity of the fuels, or it is nowhere in this coordinate.

**Brief to launch (one paragraph).** Open "the leftover at depth u" on the base-3 section
[16129, 260,467,321) with shorter stretch lengths L' (50, 100, 200, 300, 400, 579), core t' = 6L' + 1
per L', depth u = ln n / ln t' in bins from 1 to the maximum reachable (3.5 at L' = 50): for every
(L', u) bin, the census of core-open members by Omega (1, 2, 3), the PP share among core-open
slots, the twin-free run count against the independent-slot prediction with that share, the
minimum K over twin-free stretches, and the identity of the fuels (fraction above the cut, i.e.
gears of machine 4) as a function of n / c'. Pre-register: (P1) the PP share depends on u alone
(the same at fixed u for every L' to within the bin's sampling), with the Omega = 3 members
appearing exactly from u = 3 (kernel-shaped: a t'-rough number below t'^3 has Omega <= 2); (P2)
the twin-free run count follows the independent prediction at every (L', u) including u > 3;
(P3) at u > 3 the PP share continues the profile 1 / 0.965 / 0.889 / 0.828 / 0.778 smoothly
(Buchstab), with no step at u = 3. A refutation of P1 (a dependence on L' at fixed u) or of P2
(runs off the independent count at some depth) is the finding: the first trace of the
construction in the leftover beyond depth; confirmation files the step at the core as ROOT in
the (L, u) coordinate with the depth law as its only structure, and sends the Formalist S15's
slot form and S16 with the exact converse hypothesis. Two cores; leftover_shadow.py and
leftover_depth.py adapted to a per-L' core; results in research/stack/r5/results/ (untracked).
