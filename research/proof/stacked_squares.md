# The stack by squares: machines built to the square of their first gear (branch S)

Builder, round 1 of the stack line, 2026-09-07. Script research/stack/r1/stacked_squares.py
(numpy only, `uv run python research/stack/r1/stacked_squares.py`, 11 s at q = 23, about 1.2 GB)
and research/stack/r1/tables.py (the tables below are generated from the result files, not
typed). Outputs in research/stack/r1/results/ (untracked; every number the text uses is in the
text). Parent: the owner's construction of 2026-09-07 (README glossary, "Machine k, square part,
band"), spawned by the observation that the manifold as one machine is a mess of interactions.
Nothing here is committed by the builder.

Prior results checked before opening (docs/novel/README.md): the exhaust-stack entry records the
square-root cap X6 (every strike of a gear above the cut on (C, C^2] is a home strike or an echo)
as KNOWN (Legendre's sieve; Holt, "Surviving Eratosthenes sieve I", arXiv:2603.25915); the README
glossary and research/proof/anchor_cycles.md record the six kill residues of a gear on the cycle
index, the one-number-per-cycle rule for gears >= 11 and the gear-7 double; anchor-235.md
section 5 lists the cycles whose six numbers are all prime (j = 601, 3261, 5523, ...). Those are
cited, not rederived. What this branch can find that is not known: the machines of the owner's
construction as objects (their gear ranges, their cycle classifications, the rules of their
bands, the relation between consecutive machines and to the engine), none of which has a
measurement on record.

The one-paragraph answer. The construction works exactly as the manager claimed at the boundary
of each machine (claim A, 0 exceptions once the square itself, which always lands on a twin
slot, is put on the band side), and the machines reach q# in 1 + floor(log2(theta(q) / ln q'))
banded machines plus one home-and-echo machine (2, 2, 2, 3, 3, 3 banded at q = 7 .. 23). Inside a
band the machine is not a CRT object: its open numbers are exactly the g_k-smooth numbers and
the g_k-smooth multiples of one prime above the machine (law S4, exact), so the open density is
the prime density (30/8)/ln x plus a smooth-times-prime term, both modelled to 0.01, and it sits
above the CRT value for the small machine 2 and below it for machine 3. Every machine k >= 2 has
Mertens weight 1/2 (0.502-0.524 at machine 2, 0.5016-0.5026 at machine 3), so the CRT
classification of every machine is the same (open : closed : mixed) = (1 : 27 : 36)/64. The
connection to the engine is exact and simple: on band k the slots the lower machines leave open
are the pairs of g_k-rough numbers, and machine k's whole action there is to strike the rough
composites; the "independence ratio" is therefore P(prime | rough) / P(open under k), which is
1.9-5.0 at the start of a band (a rough number below g_k g_k' is prime) and 0.81-0.87 at its end.
The cycle at q# is open under no machine k >= 2 at any q (0 of 16) and under the engine only for
q <= 7 (claim C refuted). The twins of band k are exactly the slots on which machine k + 1 makes
two home strikes, and machine k + 1's gears are exactly the primes of band k.

## 1. Pre-registered (written before the script ran; verdicts filled in afterwards)

### Claims put to the scorecard first (the manager's)

(A) THE BAND STRUCTURE. Below the square of its first gear g_k, every strike of machine k on a
slot number is a home strike (the number is a gear of k) or an echo (the number is p x c with
p a gear of k and c < g_k, so c has a prime factor below g_k, in a lower machine or the anchor).
Hence machine k acts genuinely only from g_k^2 on, and on the band [g_k^2, g_{k+1}^2) a slot
open under the anchor and machines 1..k is a twin prime pair. Prediction: 0 exceptions at
every q, once the boundary convention is fixed (see P1).

(B) THE COUNT. The first gears are q', then about q'^2, q'^4, ..., and the number of machines
needed to reach q# is about log2(q / log q).

(C) THE LAST CYCLE (the manager's expectation). The cycle at q# (numbers q# + 11 .. q# + 31) is
open for every machine, as the engine's is, its numbers being coprime to everything below q.

(D) THE DENSITIES. On a band the open cycles of machine k follow prod over its gears of
(1 - 6/g), the closed cycles follow the CRT expectation, and the jointly open slots of machines
1..k follow the product of the individual open fractions (independence ratio 1).

### The builder's predictions

P1 (claim A, with its one boundary). Claim A holds with 0 exceptions when the square part is
the half-open range [1, g_k^2). The number g_k^2 itself is a genuine strike (p x c with c = g_k)
and it is ALWAYS a slot number: a prime g >= 7 has g^2 = 1 or 19 (mod 30), and 30j + 1 is the
upper member of slot (29, 31) of cycle j - 1, 30j + 19 the upper member of slot (17, 19) of
cycle j. So with the inclusive range [1, g_k^2] claim A fails at exactly one number per
machine, the square. Home strikes on the square part are exactly the gears of the machine in
the classes +-1, +-11, +-13 (mod 30); the gears in the class +-7 sit next to a multiple of 5 and
never strike a slot at home.

P2 (claim B, the exact table). With g_{k+1} = the next prime above g_k^2: q = 5 has machine 2 =
7..47 whose band is empty (49 > 30), so one machine covers q#; q = 7, 11, 13 have machines 1 and
2 with bands and a third machine that is home-and-echo only; q = 17, 19, 23 have machines 1, 2, 3
with bands and a fourth home-and-echo machine. The count of machines with a band is
K = 1 + floor(log2(theta(q) / (2 ln q'))) + [correction], with theta(q) = ln q#; the manager's
log2(q / log q) gives 1.2, 1.9, 2.2, 2.3, 2.6, 2.7, 2.9 at q = 5 .. 23 and is the right order.

P3 (claim C, refuted for the engine from q = 11). The cycle at q# is the mirror of cycle 0:
q# + e is struck by a prime p <= q iff p divides e. So under the engine the cycle at q# is open
for q <= 7 and mixed for q >= 11 (q# + 11 struck by 11 from q = 11, q# + 13 by 13 from q = 13,
q# + 17, q# + 19 from q = 17, 19; the slot (q# + 29, q# + 31) is open at every q). The top cycle
inside the period, j = q#/30 - 1 (numbers q# - 19 .. q# + 1), is likewise open for q <= 7 and
mixed from q = 11. For machines k >= 2 the cycle at q# is struck iff a gear of k divides one of
the six numbers; no closed form; predicted: struck at least once among the machines k >= 2 over
the seven q, never closed.

P4 (the single-gear cycle law; known, restated with its proof). Gear g strikes cycle j's six
numbers at the residues j = -e x 30^{-1} (mod g), e in {11, 13, 17, 19, 29, 31}: six distinct
residues for g >= 11 (two offsets share a residue iff g divides their difference, at most 20),
five for g = 7 (offsets 17 and 31 share j = 2 mod 7). A gear closes a cycle alone iff one
residue carries all three slots, which needs g to divide a difference from each of {4, 6, 8}
and {10, 12, 14}: only 2 and 3, both in the anchor. So NO gear >= 7 closes a cycle alone; a gear
>= 11 strikes at most one slot per cycle; a closed cycle of a machine all of whose gears are
>= 11 (every machine k >= 2) needs exactly three distinct gears, one per slot; the engine can
close with two gears at j = 2 (mod 7). Prediction: 0 exceptions at every q and every machine.

P5 (the half law and the shape of the band). Because machine k is built to the square of its
first gear, its Mertens product prod_{g_k <= p <= g_k^2} (1 - 1/p) is ln g_k / ln g_k^2 = 1/2 up
to O(1 / ln g_k). So every machine k >= 2 has the same CRT densities: slot open prod(1 - 2/p)
within 3 % of 1/4, cycle open prod(1 - 6/p) within 5 % of 1/64, cycle closed within 5 % of
27/64, at every q. Measured on the band the open fractions are BELOW the CRT value at the start
of the band (at x = g_k^2 only g_k-smooth numbers escape the machine) and rise toward it along
the band; in the coordinate u = ln x / ln g_k (u from 2 to 4 on the band) the ratio
measured / CRT rises from about 0.35 at u = 2 to about 0.9 at u = 4, and the curve is the same
for machine 2 and machine 3 at every q within the sampling noise (the construction is
self-similar in u).

P6 (the independence ratio). The ratio (jointly open slots) / (product of the individual open
fractions x slots) on a band lies in [0.8, 1.3] and drifts along the band; it is not 1.

P7 (the cross-machine relation, exact). The gears of machine k + 1 are exactly the primes of
band k (the primes in [g_k^2, g_{k+1}^2)), so the twin prime pairs of band k are exactly the
slots of machine k + 1's square part on which machine k + 1 makes two home strikes: 0
exceptions. The closed-cycle positions of machine k + 1 bear no relation to those of machine k
through the squares (the count of closed cycles j' of k + 1 with 30 j' within one cycle of
(30 j)^2 for a closed j of k is at the chance rate).

P8 (runs). The longest run of closed cycles of machine k on its band grows with k at fixed q and
with q at fixed k; the positions of the record runs carry no structure in j.

### Scorecard

| # | owner | claim | verdict |
|---|---|---|---|
| A | manager | band structure: home-and-echo below g_k^2, open under 1..k on the band = twin | CONFIRMED, 0 exceptions at all 23 machine instances with the half-open square part (T2); it is the known cap X6 in the stack's coordinate |
| B | manager | first gears q', ~q'^2, ~q'^4; about log2(q / log q) machines | CONFIRMED: g_3 = 127, 173, 293, 367, 541, 853 and g_4 = 134699, 292693, 727613; banded machines 2, 2, 2, 3, 3, 3 at q = 7..23 = 1 + floor(log2(theta(q) / ln q')) exactly, and round(log2(q / ln q)) at all six (T1) |
| C | manager | the cycle at q# is open for every machine, as the engine's is | REFUTED: under the engine open only at q = 5, 7 (mixed from q = 11: q# + 11 by 11, q# + 13 by 13, ...); under machines k >= 2 open at 0 of 16 instances (7 mixed, 9 closed) (T7). "Coprime to everything below q" is false: q# + e is divisible by the prime factors of e |
| D | manager | densities follow prod(1 - 6/g), CRT, independence ratio 1 | REFUTED in all three parts: machine 2's open cycles are 1.2-2.0 x prod(1 - 6/g), machine 3's 0.27-0.62 x; closed 0.70-0.92 x CRT (machine 2) and 1.16-1.40 x (machine 3); the ratio runs 0.83-5.0 (T3, T4, T5). The mechanism is law S4: the band is not a period sample |
| P1 | builder | claim A exact on [1, g_k^2); g_k^2 is always a slot number; home = classes +-1, +-11, +-13 | CONFIRMED at every machine: g_k^2 mod 30 in {1, 19} at all 23 instances, struck as a genuine strike at every banded machine, new strikes below the square = home strikes = gears in the slot classes, not-home 0 (T2) |
| P2 | builder | exact machine table as predicted | CONFIRMED (T1) |
| P3 | builder | cycle at q# mixed under the engine from q = 11; struck by some k >= 2; never closed under k >= 2 | first two clauses CONFIRMED, "never closed" REFUTED (9 of 16 closed: machine 3 at q = 7 .. 23, machine 2 at q = 17, 19, machine 4 at q = 23) (T7) |
| P4 | builder | single-gear law, three gears per closed cycle for k >= 2 | CONFIRMED: minimum distinct gears 3 in every sample of closed band cycles of machines 2 and 3 (T8), 2 for the engine (the gear-7 double at j = 16 = 2 mod 7) |
| P5 | builder | the half law within 3 % / 5 %; band profile 0.35 -> 0.9, same curve for k = 2, 3 | PARTLY REFUTED: prod(1 - 1/p) within 5 % of 1/2 at every machine, the 3 %/5 % clauses hold for machine 3 (0.5 %, 0.6 %, 2.6 %, 1.1 %) and fail for machine 2 (slot 8.3 % at q = 17, cycle-open 46 % at q = 7) (T9); the profile is ABOVE CRT at the start of machine 2's band (1.0-1.2 x per slot) and below at the start of machine 3's (0.57-0.66 x), the curves differ, and the mechanism is the prime density (30/8)/ln x, which depends on g (T4, law S4) |
| P6 | builder | independence ratio in [0.8, 1.3], drifting | REFUTED at the band starts (1.9-5.0), CONFIRMED at the ends (0.81-0.87 at u = 3.75-4.01); the drift is monotone and has a closed mechanism (law S6) |
| P7 | builder | gears of k + 1 = primes of band k; twins of band k = double-home slots of k + 1 | CONFIRMED for k >= 2 (gear sets equal at every q; the engine is the one exception since it stops at q < 49); twins = double-home slots with 0 mismatches at all 16 bands; the square map lands at the chance rate 0.50, 0.66, 0.52 against densities 0.58, 0.59, 0.49 (T6) |
| P8 | builder | closed runs grow with k and q; positions unstructured | PARTLY: machine 2's longest closed run 4, 5, 8, 8, 9 at q = 11..23 and machine 3's 15, 30, 25 at q = 17, 19, 23 (grows with k; not monotone in q for machine 3); the positions repeat across q (3573 at q = 17 and 19) because the numbers are the same, which is structure of the trivial kind |

## 2. Setup

Anchor = the primes 2, 3, 5 as one object, period 30. Cycle j = the integers 30j .. 30j + 29
with its three twin slots (30j + 11, 30j + 13), (30j + 17, 30j + 19), (30j + 29, 30j + 31); the
six slot numbers are exactly the numbers of the cycle coprime to 30 other than 30j + 1, which is
the upper member of the previous cycle's third slot. Machine 1 = the engine = the primes 7 .. q
(g_1 = 7). Machine 2 = the primes in [q', q'^2], q' the next prime after q (g_2 = q'). Machine
k + 1 = the primes from the next prime after machine k's largest gear (which is the first prime
above g_k^2, since g_k^2 is composite) to the square of that prime (g_{k+1} = nextprime(g_k^2),
gears [g_{k+1}, g_{k+1}^2]). The last machine is the first one whose square exceeds q#; its
gears are the primes from g_K to q#. Each machine strikes the multiples of its gears (phase
zero); a slot (a, a + 2) is struck if a or a + 2 is; a cycle is OPEN for a machine if none of
its three slots is struck, CLOSED if all three are, MIXED otherwise; "two open" and "one open"
split the mixed class. The machine's OWN strikes include its echoes; its NEW strikes are the
numbers whose smallest prime factor is one of its gears.

Square part and band. The square part of machine k is the half-open range [1, g_k^2) and its
band is [g_k^2, g_{k+1}^2), capped at q# + 1 so that the slot (q# - 1, q# + 1) of the top cycle
belongs to the last band. The brief's inclusive square part [1, g_k^2] differs at one number,
g_k^2, and law S1 says why it must be moved: the square is the machine's first genuine strike
and it is always a slot number. In cycle terms the square part is the cycles j with
30j + 31 < g_k^2, the band is the cycles from the first with 30j + 11 >= g_k^2 to the last with
30j + 31 < g_{k+1}^2, and the one cycle that straddles g_k^2 (the boundary cycle) is reported
on its own in T2. The cycle at q# is j = q#/30 (numbers q# + 11 .. q# + 31); the top cycle inside
the period is j = q#/30 - 1 (numbers q# - 19 .. q# + 1).

Computation. One sieve of [1, q# + 31]; per machine a boolean strike array over the six slot
numbers of every cycle 0 .. q#/30 (residues j = -e x 30^{-1} mod p for every gear p and offset
e); the last machine through the cofactor of every slot number after every prime below g_K is
divided out (the cofactor is asserted to be 1 or a prime >= g_K, which is the content of the
cap). A cumulative cofactor R_k (every prime below g_k divided out) is carried through the
machines and asserted against the strike arrays (law S4). Everything is exact over the whole
range at every q.

### T1. The machines (claim B)

| q | q# | cycles | k | g_k | g_k^2 | gears | count | below q# | square cycles | boundary | band cycles | band numbers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 30 | 1 | 1 | 7 | 49 | none | 0 | 0 | [0, 1) | - | empty | empty |
| 5 | 30 | 1 | 2 | 7 | 49 | 7..(q#) | 12 | 7 | [0, 1) | - | empty | empty |
| 7 | 210 | 7 | 1 | 7 | 49 | 7..7 | 1 | 1 | [0, 1) | 1 | [2, 3) | [49, 121) |
| 7 | 210 | 7 | 2 | 11 | 121 | 11..113 | 26 | 26 | [0, 3) | 3 | [4, 7) | [121, 211) |
| 7 | 210 | 7 | 3 | 127 | 16129 | 127..(q#) | 23 | 16 | [0, 7) | - | empty | empty |
| 11 | 2310 | 77 | 1 | 7 | 49 | 7..11 | 2 | 2 | [0, 1) | 1 | [2, 5) | [49, 169) |
| 11 | 2310 | 77 | 2 | 13 | 169 | 13..167 | 34 | 34 | [0, 5) | 5 | [6, 77) | [169, 2311) |
| 11 | 2310 | 77 | 3 | 173 | 29929 | 173..(q#) | 308 | 304 | [0, 77) | - | empty | empty |
| 13 | 30030 | 1001 | 1 | 7 | 49 | 7..13 | 3 | 3 | [0, 1) | 1 | [2, 9) | [49, 289) |
| 13 | 30030 | 1001 | 2 | 17 | 289 | 17..283 | 55 | 55 | [0, 9) | 9 | [10, 1001) | [289, 30031) |
| 13 | 30030 | 1001 | 3 | 293 | 85849 | 293..(q#) | 3189 | 3187 | [0, 1001) | - | empty | empty |
| 17 | 510510 | 17017 | 1 | 7 | 49 | 7..17 | 4 | 4 | [0, 1) | 1 | [2, 11) | [49, 361) |
| 17 | 510510 | 17017 | 2 | 19 | 361 | 19..359 | 65 | 65 | [0, 11) | 11 | [12, 4489) | [361, 134689) |
| 17 | 510510 | 17017 | 3 | 367 | 134689 | 367..134683 | 12481 | 12481 | [0, 4489) | 4489 | [4490, 17017) | [134689, 510511) |
| 17 | 510510 | 17017 | 4 | 134699 | 18143820601 | 134699..(q#) | 29779 | 29778 | [0, 17017) | - | empty | empty |
| 19 | 9699690 | 323323 | 1 | 7 | 49 | 7..19 | 5 | 5 | [0, 1) | 1 | [2, 17) | [49, 529) |
| 19 | 9699690 | 323323 | 2 | 23 | 529 | 23..523 | 91 | 91 | [0, 17) | 17 | [18, 9755) | [529, 292681) |
| 19 | 9699690 | 323323 | 3 | 541 | 292681 | 541..292679 | 25339 | 25339 | [0, 9755) | 9755 | [9756, 323323) | [292681, 9699691) |
| 19 | 9699690 | 323323 | 4 | 292693 | 85669192249 | 292693..(q#) | 620592 | 620591 | [0, 323323) | - | empty | empty |
| 23 | 223092870 | 7436429 | 1 | 7 | 49 | 7..23 | 6 | 6 | [0, 1) | 1 | [2, 27) | [49, 841) |
| 23 | 223092870 | 7436429 | 2 | 29 | 841 | 29..839 | 137 | 137 | [0, 27) | 27 | [28, 24253) | [841, 727609) |
| 23 | 223092870 | 7436429 | 3 | 853 | 727609 | 853..727589 | 58462 | 58462 | [0, 24253) | 24253 | [24254, 7436429) | [727609, 223092871) |
| 23 | 223092870 | 7436429 | 4 | 727613 | 529420677769 | 727613..(q#) | 12224923 | 12224923 | [0, 7436429) | - | empty | empty |

The count. Machines with a non-empty band: 0 at q = 5 (the engine is empty and machine 2's
square already covers 30), 2 at q = 7, 11, 13 and 3 at q = 17, 19, 23; one more machine, home
and echo only, sits on top in every case. Since g_{k+1} = nextprime(g_k^2) makes ln g_k double
at each step, the last banded machine is the largest k with g_k^2 <= q#, i.e. with
2^{k-1} ln q' <= theta(q) = ln q#, so K_band = 1 + floor(log2(theta(q) / ln q')) (S9): 2, 2, 2,
3, 3, 3 at q = 7 .. 23, exact at all six; the manager's log2(q / ln q) = 1.85, 2.20, 2.34, 2.59,
2.69, 2.87 rounds to the same values. A fourth banded machine first appears when theta(q) >=
8 ln q', at q = 41 (theta = 33.35 against 8 ln 43 = 30.09; at q = 37 the ratio is 7.98, just
short). The last machine's defined range runs to g_K^2, far above q#, so T1 counts its gears only
to q# + 32: 7 of 12, 16 of 23, 304 of 308, 3187 of 3189, 29778 of 29779, 620591 of 620592 and all
12224923 lie below q#, and every gear above q# is idle.

## 3. The band structure (claim A): proved and verified

Proof (two lines). Let n < g_k^2 be a slot number struck by machine k, so p | n for a gear p
of k, p >= g_k. Write n = p c. If c = 1 the strike is a home strike. Otherwise c = n/p < g_k^2 /
g_k = g_k, so every prime factor of c is below g_k, i.e. in the anchor or a lower machine, and the
strike is an echo. For the band: a slot number n < g_{k+1}^2 with no prime factor below g_{k+1}
is prime (its smallest prime factor would be >= g_{k+1} and its cofactor >= g_{k+1} too, giving
n >= g_{k+1}^2); a slot open under the anchor and machines 1..k has both members free of every
prime below g_{k+1}, so both are prime. Conversely a twin pair on the band is struck by no
machine <= k, because its members exceed every gear of k (the gears are <= g_k^2 <= n and n is
prime, so n is not a gear, and n = p c with p <= g_k^2 < n would make n composite). Prior art:
this is the sieve to the square root (X6 of the exhaust-stack entry, Legendre; Holt's interval
of survival); it is stated here in the stack's coordinate because every later section rests on
it. It is not new.

Verification (T2). At every one of the 23 machine instances the new strikes below g_k^2 are
exactly the home strikes: 0, 9, 0, 19, 19, 1, 25, 232, 2, 42, 2388, 3, 48, 9338, 22327, 4, 67,
19007, 465371, 4, 102, 43811, 9168289 new strikes, all home, 0 not home; in every case the count
equals the number of the machine's gears in the classes +-1, +-11, +-13 (mod 30) (three quarters
of the gears for the large machines: 9338 of 12481, 43811 of 58462, 9168289 of 12224923). On
every band the slots open under machines 1..k coincide with the twin prime pairs slot by slot:
0 mismatches at all 16 bands, 24,226 + 24,225 + 24,225 slots and 2092 + 2085 + 2047 twins on band
2 at q = 23, 7,412,175 slots per slot type and 296,672 + 296,783 + 296,350 twins on band 3.

### T2. Claim A verified

| q | k | new strikes below g_k^2 | home | not home | gears in slot classes | g_k^2 mod 30 | g_k^2 in cycle | g_k^2 struck as new | band slots (3 slots) | band twins (3 slots) | mismatches | boundary cycle numbers | struck by k | new by k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 0 | 0 | 0 | 0 | 19 | 1 | False | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | None | - | - |
| 5 | 2 | 9 | 9 | 0 | 9 | 19 | 1 | True | None | None | None | None | - | - |
| 7 | 1 | 0 | 0 | 0 | 0 | 19 | 1 | True | [2, 2, 2] | [2, 1, 1] | [0, 0, 0] | [41, 43, 47, 49, 59, 61] | ...x.. | ...x.. |
| 7 | 2 | 19 | 19 | 0 | 19 | 1 | 3 | True | [3, 3, 2] | [1, 2, 2] | [0, 0, 0] | [101, 103, 107, 109, 119, 121] | xxxxxx | xxxx.x |
| 7 | 3 | 19 | 19 | 0 | 19 | 19 | - | False | None | None | None | None | - | - |
| 11 | 1 | 1 | 1 | 0 | 1 | 19 | 1 | True | [4, 3, 4] | [2, 2, 2] | [0, 0, 0] | [41, 43, 47, 49, 59, 61] | ...x.. | ...x.. |
| 11 | 2 | 25 | 25 | 0 | 25 | 19 | 5 | True | [71, 71, 71] | [22, 19, 16] | [0, 0, 0] | [161, 163, 167, 169, 179, 181] | xxxx.. | .xxx.. |
| 11 | 3 | 232 | 232 | 0 | 232 | 19 | - | False | None | None | None | None | - | - |
| 13 | 1 | 2 | 2 | 0 | 2 | 19 | 1 | True | [8, 7, 8] | [4, 4, 5] | [0, 0, 0] | [41, 43, 47, 49, 59, 61] | ...x.. | ...x.. |
| 13 | 2 | 42 | 42 | 0 | 42 | 19 | 9 | True | [991, 991, 991] | [153, 150, 146] | [0, 0, 0] | [281, 283, 287, 289, 299, 301] | xxxxxx | xx.x.. |
| 13 | 3 | 2388 | 2388 | 0 | 2388 | 19 | - | False | None | None | None | None | - | - |
| 17 | 1 | 3 | 3 | 0 | 3 | 19 | 1 | True | [10, 10, 10] | [5, 5, 5] | [0, 0, 0] | [41, 43, 47, 49, 59, 61] | ...x.. | ...x.. |
| 17 | 2 | 48 | 48 | 0 | 48 | 1 | 11 | True | [4478, 4477, 4477] | [509, 519, 511] | [0, 0, 0] | [341, 343, 347, 349, 359, 361] | x.xxxx | ..xxxx |
| 17 | 3 | 9338 | 9338 | 0 | 9338 | 19 | 4489 | True | [12527, 12527, 12527] | [1043, 1033, 1000] | [0, 0, 0] | [134681, 134683, 134687, 134689, 134699, 134701] | xx.x.x | xx.x.. |
| 17 | 4 | 22327 | 22327 | 0 | 22327 | 1 | - | False | None | None | None | None | - | - |
| 19 | 1 | 4 | 4 | 0 | 4 | 19 | 1 | True | [16, 15, 16] | [8, 5, 6] | [0, 0, 0] | [41, 43, 47, 49, 59, 61] | ...x.. | ...x.. |
| 19 | 2 | 67 | 67 | 0 | 67 | 19 | 17 | True | [9738, 9738, 9738] | [970, 986, 961] | [0, 0, 0] | [521, 523, 527, 529, 539, 541] | xxxx.. | xx.x.. |
| 19 | 3 | 19007 | 19007 | 0 | 19007 | 1 | 9755 | True | [313567, 313567, 313566] | [18275, 18187, 18049] | [0, 0, 0] | [292661, 292663, 292667, 292669, 292679, 292681] | xxx.xx | x.x.xx |
| 19 | 4 | 465371 | 465371 | 0 | 465371 | 19 | - | False | None | None | None | None | - | - |
| 23 | 1 | 4 | 4 | 0 | 4 | 19 | 1 | True | [26, 26, 26] | [10, 7, 10] | [0, 0, 0] | [41, 43, 47, 49, 59, 61] | ...x.. | ...x.. |
| 23 | 2 | 102 | 102 | 0 | 102 | 1 | 27 | True | [24226, 24225, 24225] | [2092, 2085, 2047] | [0, 0, 0] | [821, 823, 827, 829, 839, 841] | xxxxxx | xxxxxx |
| 23 | 3 | 43811 | 43811 | 0 | 43811 | 19 | 24253 | True | [7412175, 7412175, 7412175] | [296672, 296783, 296350] | [0, 0, 0] | [727601, 727603, 727607, 727609, 727619, 727621] | .x.xx. | ...x.. |
| 23 | 4 | 9168289 | 9168289 | 0 | 9168289 | 19 | - | False | None | None | None | None | - | - |

Law S1 (the square lands on a twin slot; the square part is half-open). For every prime
g >= 7, g^2 = 1 or 19 (mod 30): g = +-1, +-11 give 1 and g = +-7, +-13 give 19. Hence g^2 is
always the upper member of a twin slot, (g^2 - 2, g^2) of type (29, 31) or of type (17, 19), and
the first genuine strike of machine k is on a twin slot of the boundary cycle. Verified at all
23 instances (the field "g_k^2 mod 30" in T2: 19, 19, 19, 1, 19, ..., and "struck as new" true at
every banded machine). This is the square gate of research/proof/anchor_runs_zero.md (the first
exclusive kill of a gear is its square) read in the anchor's coordinate; the slot-type dichotomy
by class is the addition. Consequence for the definitions: the inclusive square part [1, g_k^2]
would make claim A false at exactly one number per machine.

The boundary cycle. The cycle containing g_k^2 holds gears of machine k below the square and the
square itself: at q = 13 the boundary cycle of machine 2 is 281, 283, 287, 289, 299, 301, struck
at every number (281, 283 home; 287 = 7 x 41, 289 = 17^2, 299 = 13 x 23, 301 = 7 x 43 echoes and
the square), new strikes at 281, 283 and 289. The pattern is the same at every q: the boundary
cycle is closed or mixed by home strikes and the square, never open.

## 4. The cycle classification per machine and q

### T3. The machine's own strikes (echoes included), per part

| q | k | part | cycles | open | closed | mixed (2 open / 1 open) | slot strikes 11-13 / 17-19 / 29-31 | first closed | last open | first open | last closed | longest closed run (len @ j, ties) | longest open run | longest non-open run | CRT open | CRT closed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | square | 1 | 1 | 0 | 0 (0 / 0) | 0 / 0 / 0 | - | 0 | 0 | - | 0 @ - (0) | 1 @ 0 | 0 @ - | - | - |
| 5 | 2 | square | 1 | 0 | 1 | 0 (0 / 0) | 1 / 1 / 1 | 0 | - | - | 0 | 1 @ 0 (1) | 0 @ - | 1 @ 0 | - | - |
| 5 | 2 | all | 1 | 0 | 1 | 0 (0 / 0) | 1 / 1 / 1 | 0 | - | - | 0 | 1 @ 0 (1) | 0 @ - | 1 @ 0 | - | - |
| 7 | 1 | square | 1 | 1 | 0 | 0 (0 / 0) | 0 / 0 / 0 | - | 0 | 0 | - | 0 @ - (0) | 1 @ 0 | 0 @ - | 0.3 | 0.0 |
| 7 | 1 | band | 1 | 0 | 0 | 1 (0 / 1) | 0 / 1 / 1 | - | - | - | - | 0 @ - (0) | 0 @ - | 1 @ 2 | 0.3 | 0.0 |
| 7 | 2 | square | 3 | 0 | 3 | 0 (0 / 0) | 3 / 3 / 3 | 0 | - | - | 2 | 3 @ 0 (1) | 0 @ - | 3 @ 0 | 0.0 | 1.2 |
| 7 | 2 | band | 3 | 0 | 0 | 3 (2 / 1) | 2 / 1 / 1 | - | - | - | - | 0 @ - (0) | 0 @ - | 3 @ 4 | 0.0 | 1.2 |
| 7 | 3 | square | 7 | 4 | 3 | 0 (0 / 0) | 3 / 3 / 3 | 4 | 3 | 0 | 6 | 3 @ 4 (1) | 4 @ 0 | 3 @ 4 | - | - |
| 7 | 3 | all | 7 | 4 | 3 | 0 (0 / 0) | 3 / 3 / 3 | 4 | 3 | 0 | 6 | 3 @ 4 (1) | 4 @ 0 | 3 @ 4 | - | - |
| 11 | 1 | square | 1 | 0 | 0 | 1 (1 / 0) | 1 / 0 / 0 | - | - | - | - | 0 @ - (0) | 0 @ - | 1 @ 0 | 0.1 | 0.0 |
| 11 | 1 | band | 3 | 0 | 0 | 3 (2 / 1) | 1 / 1 / 2 | - | - | - | - | 0 @ - (0) | 0 @ - | 3 @ 2 | 0.4 | 0.1 |
| 11 | 2 | square | 5 | 0 | 5 | 0 (0 / 0) | 5 / 5 / 5 | 0 | - | - | 4 | 5 @ 0 (1) | 0 @ - | 5 @ 0 | 0.1 | 1.9 |
| 11 | 2 | band | 71 | 2 | 19 | 50 (17 / 33) | 44 / 49 / 47 | 12 | 69 | 53 | 61 | 4 @ 22 (1) | 1 @ 53 | 47 @ 6 | 0.9 | 27.3 |
| 11 | 3 | square | 77 | 5 | 40 | 32 (2 / 30) | 58 / 63 / 61 | 6 | 4 | 0 | 76 | 5 @ 12 (3) | 5 @ 0 | 72 @ 5 | - | - |
| 11 | 3 | all | 77 | 5 | 40 | 32 (2 / 30) | 58 / 63 / 61 | 6 | 4 | 0 | 76 | 5 @ 12 (3) | 5 @ 0 | 72 @ 5 | - | - |
| 13 | 1 | square | 1 | 0 | 0 | 1 (1 / 0) | 1 / 0 / 0 | - | - | - | - | 0 @ - (0) | 0 @ - | 1 @ 0 | 0.1 | 0.1 |
| 13 | 1 | band | 7 | 0 | 0 | 7 (4 / 3) | 4 / 3 / 3 | - | - | - | - | 0 @ - (0) | 0 @ - | 7 @ 2 | 0.5 | 0.5 |
| 13 | 2 | square | 9 | 0 | 8 | 1 (0 / 1) | 8 / 9 / 9 | 1 | - | - | 8 | 8 @ 1 (1) | 0 @ - | 9 @ 0 | 0.1 | 3.6 |
| 13 | 2 | band | 991 | 25 | 327 | 639 (184 / 455) | 681 / 693 / 701 | 12 | 979 | 108 | 998 | 5 @ 815 (2) | 2 @ 889 | 172 @ 315 | 13.0 | 398.4 |
| 13 | 3 | square | 1001 | 11 | 769 | 221 (20 / 201) | 902 / 912 / 915 | 10 | 44 | 0 | 1000 | 23 @ 900 (1) | 10 @ 0 | 956 @ 45 | - | - |
| 13 | 3 | all | 1001 | 11 | 769 | 221 (20 / 201) | 902 / 912 / 915 | 10 | 44 | 0 | 1000 | 23 @ 900 (1) | 10 @ 0 | 956 @ 45 | - | - |
| 17 | 1 | square | 1 | 0 | 0 | 1 (0 / 1) | 1 / 1 / 0 | - | - | - | - | 0 @ - (0) | 0 @ - | 1 @ 0 | 0.0 | 0.1 |
| 17 | 1 | band | 9 | 0 | 0 | 9 (4 / 5) | 4 / 5 / 5 | - | - | - | - | 0 @ - (0) | 0 @ - | 9 @ 2 | 0.4 | 1.1 |
| 17 | 2 | square | 11 | 0 | 9 | 2 (0 / 2) | 10 / 11 / 10 | 1 | - | - | 10 | 7 @ 4 (1) | 0 @ - | 11 @ 0 | 0.2 | 4.2 |
| 17 | 2 | band | 4477 | 95 | 1565 | 2817 (750 / 2067) | 3192 / 3187 / 3200 | 12 | 4354 | 43 | 4488 | 8 @ 3573 (2) | 2 @ 1168 | 369 @ 2735 | 73.8 | 1696.2 |
| 17 | 3 | square | 4489 | 13 | 3900 | 576 (31 / 545) | 4269 / 4278 / 4274 | 12 | 44 | 0 | 4488 | 108 @ 4173 (1) | 12 @ 0 | 4444 @ 45 | 72.0 | 1873.5 |
| 17 | 3 | band | 12527 | 54 | 7241 | 5232 (858 / 4374) | 10432 / 10416 / 10481 | 4491 | 16886 | 4783 | 17016 | 15 @ 14939 (2) | 2 @ 9633 | 935 @ 12064 | 200.8 | 5228.2 |
| 17 | 4 | square | 17017 | 5815 | 1526 | 9676 (4680 / 4996) | 6411 / 6386 / 6453 | 4501 | 17016 | 0 | 17010 | 4 @ 5240 (2) | 4489 @ 0 | 59 @ 9467 | - | - |
| 17 | 4 | all | 17017 | 5815 | 1526 | 9676 (4680 / 4996) | 6411 / 6386 / 6453 | 4501 | 17016 | 0 | 17010 | 4 @ 5240 (2) | 4489 @ 0 | 59 @ 9467 | - | - |
| 19 | 1 | square | 1 | 0 | 0 | 1 (0 / 1) | 1 / 1 / 0 | - | - | - | - | 0 @ - (0) | 0 @ - | 1 @ 0 | 0.0 | 0.2 |
| 19 | 1 | band | 15 | 0 | 2 | 13 (4 / 9) | 8 / 10 / 10 | 12 | - | - | 16 | 1 @ 12 (2) | 0 @ - | 15 @ 2 | 0.5 | 2.7 |
| 19 | 2 | square | 17 | 0 | 15 | 2 (1 / 1) | 16 / 16 / 16 | 1 | - | - | 16 | 13 @ 4 (1) | 0 @ - | 17 @ 0 | 0.3 | 6.6 |
| 19 | 2 | band | 9737 | 203 | 3435 | 6099 (1650 / 4449) | 6951 / 6943 / 6959 | 22 | 9682 | 43 | 9754 | 8 @ 3573 (1) | 2 @ 486 | 303 @ 9220 | 161.9 | 3760.6 |
| 19 | 3 | square | 9755 | 18 | 8448 | 1289 (86 / 1203) | 9289 / 9275 / 9272 | 20 | 44 | 0 | 9754 | 82 @ 7054 (1) | 17 @ 0 | 9710 @ 45 | 154.8 | 4088.3 |
| 19 | 3 | band | 313567 | 1384 | 183443 | 128740 (20730 / 108010) | 262273 / 262313 / 262493 | 9758 | 323268 | 9938 | 323322 | 30 @ 80011 (1) | 2 @ 33429 | 1976 @ 157205 | 4976.3 | 131415.7 |
| 19 | 4 | square | 323323 | 36743 | 52985 | 233595 (104144 / 129451) | 173974 / 173918 / 174109 | 9769 | 323319 | 0 | 323316 | 7 @ 132494 (3) | 9756 @ 0 | 139 @ 322854 | - | - |
| 19 | 4 | all | 323323 | 36743 | 52985 | 233595 (104144 / 129451) | 173974 / 173918 / 174109 | 9769 | 323319 | 0 | 323316 | 7 @ 132494 (3) | 9756 @ 0 | 139 @ 322854 | - | - |
| 23 | 1 | square | 1 | 0 | 0 | 1 (0 / 1) | 1 / 1 / 0 | - | - | - | - | 0 @ - (0) | 0 @ - | 1 @ 0 | 0.0 | 0.2 |
| 23 | 1 | band | 25 | 0 | 6 | 19 (5 / 14) | 16 / 19 / 16 | 12 | - | - | 25 | 4 @ 22 (1) | 0 @ - | 25 @ 2 | 0.6 | 5.5 |
| 23 | 2 | square | 27 | 0 | 25 | 2 (1 / 1) | 26 / 26 / 26 | 1 | - | - | 26 | 23 @ 4 (1) | 0 @ - | 27 @ 0 | 0.4 | 11.0 |
| 23 | 2 | band | 24225 | 432 | 9082 | 14711 (3897 / 10814) | 17587 / 17617 / 17567 | 31 | 24103 | 43 | 24250 | 9 @ 22686 (1) | 2 @ 1168 | 587 @ 22730 | 358.1 | 9901.7 |
| 23 | 3 | square | 24253 | 36 | 20845 | 3372 (210 / 3162) | 23028 / 23016 / 23025 | 30 | 4986 | 0 | 24252 | 80 @ 19524 (1) | 28 @ 0 | 19266 @ 4987 | 385.3 | 10166.7 |
| 23 | 3 | band | 7412175 | 73124 | 3618540 | 3720511 (797480 / 2923031) | 5833391 / 5832641 / 5833130 | 24254 | 7436403 | 24296 | 7436426 | 25 @ 144307 (2) | 3 @ 2315639 | 3177 @ 161105 | 117752.1 | 3107139.0 |
| 23 | 4 | square | 7436429 | 325934 | 2167536 | 4942959 (1679667 / 3263292) | 4902968 / 4902855 / 4903036 | 24264 | 7436423 | 0 | 7436426 | 13 @ 4768419 (2) | 24253 @ 0 | 435 @ 7146725 | - | - |
| 23 | 4 | all | 7436429 | 325934 | 2167536 | 4942959 (1679667 / 3263292) | 4902968 / 4902855 / 4903036 | 24264 | 7436423 | 0 | 7436426 | 13 @ 4768419 (2) | 24253 @ 0 | 435 @ 7146725 | - | - |

### T3b. The machine's new strikes and the joint strikes of machines 1..k on the band

| q | k | strikes | cycles | open | closed | mixed | slot strikes | first open | last open | first closed | longest closed run | longest open run | longest non-open run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 2 | new (all) | 1 | 0 | 1 | 0 | [1, 1, 1] | - | - | 0 | 1 @ 0 (1) | 0 @ - | 1 @ 0 |
| 7 | 1 | new | 1 | 0 | 0 | 1 | [0, 1, 1] | - | - | - | 0 @ - (0) | 0 @ - | 1 @ 2 |
| 7 | 1 | joint 1..k | 1 | 0 | 0 | 1 | [0, 1, 1] | - | - | - | 0 @ - (0) | 0 @ - | 1 @ 2 |
| 7 | 2 | new | 3 | 1 | 0 | 2 | [0, 1, 1] | 4 | 4 | - | 0 @ - (0) | 1 @ 4 | 2 @ 5 |
| 7 | 2 | joint 1..k | 3 | 0 | 0 | 3 | [2, 1, 1] | - | - | - | 0 @ - (0) | 0 @ - | 3 @ 4 |
| 7 | 3 | new (all) | 7 | 4 | 3 | 0 | [3, 3, 3] | 0 | 3 | 4 | 3 @ 4 (1) | 4 @ 0 | 3 @ 4 |
| 11 | 1 | new | 3 | 0 | 0 | 3 | [1, 1, 2] | - | - | - | 0 @ - (0) | 0 @ - | 3 @ 2 |
| 11 | 1 | joint 1..k | 3 | 0 | 0 | 3 | [1, 1, 2] | - | - | - | 0 @ - (0) | 0 @ - | 3 @ 2 |
| 11 | 2 | new | 71 | 11 | 3 | 57 | [31, 28, 33] | 6 | 69 | 31 | 1 @ 31 (3) | 1 @ 6 | 12 @ 54 |
| 11 | 2 | joint 1..k | 71 | 0 | 27 | 44 | [49, 52, 55] | - | - | 12 | 4 @ 22 (1) | 0 @ - | 71 @ 6 |
| 11 | 3 | new (all) | 77 | 6 | 35 | 36 | [55, 59, 58] | 0 | 44 | 6 | 5 @ 12 (2) | 5 @ 0 | 39 @ 5 |
| 13 | 1 | new | 7 | 0 | 0 | 7 | [4, 3, 3] | - | - | - | 0 @ - (0) | 0 @ - | 7 @ 2 |
| 13 | 1 | joint 1..k | 7 | 0 | 0 | 7 | [4, 3, 3] | - | - | - | 0 @ - (0) | 0 @ - | 7 @ 2 |
| 13 | 2 | new | 991 | 76 | 145 | 770 | [535, 540, 537] | 10 | 995 | 87 | 3 @ 451 (2) | 2 @ 21 | 61 @ 250 |
| 13 | 2 | joint 1..k | 991 | 1 | 599 | 391 | [838, 841, 845] | 601 | 601 | 12 | 16 @ 814 (1) | 1 @ 601 | 591 @ 10 |
| 13 | 3 | new (all) | 1001 | 39 | 256 | 706 | [645, 642, 650] | 0 | 996 | 10 | 5 @ 12 (2) | 10 @ 0 | 130 @ 689 |
| 17 | 1 | new | 9 | 0 | 0 | 9 | [4, 5, 5] | - | - | - | 0 @ - (0) | 0 @ - | 9 @ 2 |
| 17 | 1 | joint 1..k | 9 | 0 | 0 | 9 | [4, 5, 5] | - | - | - | 0 @ - (0) | 0 @ - | 9 @ 2 |
| 17 | 2 | new | 4477 | 360 | 688 | 3429 | [2483, 2503, 2486] | 12 | 4477 | 87 | 4 @ 3526 (1) | 3 @ 20 | 100 @ 2124 |
| 17 | 2 | joint 1..k | 4477 | 2 | 3074 | 1401 | [3969, 3958, 3966] | 601 | 3261 | 12 | 20 @ 2077 (1) | 1 @ 601 | 2659 @ 602 |
| 17 | 3 | new | 12527 | 9990 | 6 | 2531 | [904, 923, 918] | 4490 | 17016 | 11192 | 1 @ 11192 (6) | 75 @ 4686 | 6 @ 14333 |
| 17 | 3 | joint 1..k | 12527 | 2 | 9652 | 2873 | [11484, 11494, 11527] | 5523 | 13075 | 4490 | 30 @ 6264 (1) | 1 @ 5523 | 7551 @ 5524 |
| 17 | 4 | new (all) | 17017 | 5815 | 1526 | 9676 | [6411, 6386, 6453] | 0 | 17016 | 4501 | 4 @ 5240 (2) | 4489 @ 0 | 59 @ 9467 |
| 19 | 1 | new | 15 | 0 | 2 | 13 | [8, 10, 10] | - | - | 12 | 1 @ 12 (2) | 0 @ - | 15 @ 2 |
| 19 | 1 | joint 1..k | 15 | 0 | 2 | 13 | [8, 10, 10] | - | - | 12 | 1 @ 12 (2) | 0 @ - | 15 @ 2 |
| 19 | 2 | new | 9737 | 866 | 1376 | 7495 | [5212, 5241, 5228] | 18 | 9748 | 104 | 4 @ 7184 (3) | 9 @ 18 | 80 @ 8657 |
| 19 | 2 | joint 1..k | 9737 | 3 | 7035 | 2699 | [8767, 8751, 8776] | 601 | 5523 | 22 | 30 @ 6264 (1) | 1 @ 601 | 4231 @ 5524 |
| 19 | 3 | new | 313567 | 187157 | 1190 | 125220 | [49628, 49639, 49717] | 9756 | 323322 | 17339 | 2 @ 197775 (5) | 217 @ 9756 | 13 @ 195102 |
| 19 | 3 | joint 1..k | 313567 | 31 | 261543 | 51993 | [295292, 295380, 295518] | 13075 | 320599 | 9757 | 50 @ 95632 (2) | 1 @ 13075 | 39976 @ 173516 |
| 19 | 4 | new (all) | 323323 | 62670 | 23755 | 236898 | [136868, 136973, 137019] | 0 | 323322 | 9769 | 4 @ 12996 (3) | 9756 @ 0 | 65 @ 79639 |
| 23 | 1 | new | 25 | 0 | 6 | 19 | [16, 19, 16] | - | - | 12 | 4 @ 22 (1) | 0 @ - | 25 @ 2 |
| 23 | 1 | joint 1..k | 25 | 0 | 6 | 19 | [16, 19, 16] | - | - | 12 | 4 @ 22 (1) | 0 @ - | 25 @ 2 |
| 23 | 2 | new | 24225 | 2221 | 3383 | 18621 | [12870, 12910, 12883] | 28 | 24249 | 104 | 4 @ 7498 (5) | 7 @ 32 | 80 @ 8657 |
| 23 | 2 | joint 1..k | 24225 | 5 | 18423 | 5797 | [22133, 22140, 22178] | 601 | 22119 | 30 | 32 @ 22949 (1) | 1 @ 601 | 9043 @ 13076 |
| 23 | 3 | new | 7412175 | 3739652 | 56612 | 3615911 | [1501993, 1502333, 1502729] | 24254 | 7436428 | 36479 | 3 @ 3224568 (4) | 126 @ 27097 | 21 @ 3233682 |
| 23 | 3 | joint 1..k | 7412175 | 243 | 6550887 | 861045 | [7115503, 7115392, 7115825] | 33411 | 7370678 | 24254 | 100 @ 4542781 (1) | 1 @ 33411 | 342531 @ 6737010 |
| 23 | 4 | new (all) | 7436429 | 1795691 | 344893 | 5295845 | [2760007, 2759069, 2759408] | 0 | 7436428 | 24264 | 5 @ 3278489 (1) | 24253 @ 0 | 65 @ 79639 |

What the tables say, part by part.

The square part of machine 2 (numbers below q'^2) is closed at every cycle but zero to two:
3 of 3, 5 of 5, 8 of 9, 9 of 11, 15 of 17, 25 of 27 closed at q = 7 .. 23. The exceptions are
cycle 0 from q = 13 on (its slot (11, 13) is two engine gears, unstruck by machine 2) and cycle
3 from q = 17 on, whose slot (119, 121) = (7 x 17, 11^2) has both members q-smooth once 17 is in
the engine. Open cycles: none at any q. Mechanism: below q'^2 machine 2's strikes are its gears and the
echoes p x c with c <= q, and the gears are so dense there (three quarters of the primes in
[q', q'^2] sit on slots) that every slot carries one or an echo. The square part of machine 3 is
the same picture at a larger scale: 40 of 77, 769 of 1001, 3900 of 4489, 8448 of 9755, 20845
of 24253 closed; open cycles 5, 11, 13, 18, 36, all of them cycles whose six numbers are
g_3-smooth (the cofactor law S4 makes "open on the square part" mean "every number g_k-smooth"),
the last at j = 4 (g_3 = 173), 44 (g_3 = 293, 367, 541: cycle 44 is 11^3, 31 x 43, 7 x 191,
13 x 103, 19 x 71, 7 x 193) and 4986 (g_3 = 853). The initial open run is 5, 10, 12, 17, 28
cycles: the cycles below g_3 and the smooth ones just after.

The band of machine 2 (u from 2 to about 4 at q >= 17): open 2, 25, 95, 203, 432 of 71, 991,
4477, 9737, 24225 cycles, against the CRT expectation 0.9, 13.0, 73.8, 161.9, 358.1 (ratios 2.2,
1.9, 1.3, 1.25, 1.2); closed 19, 327, 1565, 3435, 9082 against 27.3, 398.4, 1696.2, 3760.6,
9901.7 (0.70, 0.82, 0.92, 0.91, 0.92). The band of machine 3 (u from 2 to 2.2-2.85): open 54,
1384, 73124 of 12527, 313567, 7412175 against 200.8, 4976, 117752 (0.27, 0.28, 0.62); closed
7241, 183443, 3618540 against 5228, 131416, 3107139 (1.38, 1.40, 1.16). Both deviations have
the same cause (section 5, law S4) and opposite signs because the prime density at the start of
the band, (30/8)/ln x per slot number, is 0.52-0.62 for machine 2 and 0.26-0.30 for machine 3.

The first closed cycles of machine 2's band are the same cycles at q = 11, 13, 17 (12, 16, 22,
23, 24, 25, 30, 31, 32, 37, 41, ...) and at q = 19 from 22 on: these are cycles whose three
composite members are semiprimes with one factor in the engine and one in machine 2 (cycle 12:
371 = 7 x 53, 377 = 13 x 29, 391 = 17 x 23; cycle 16: 493 = 17 x 29, 497 = 7 x 71, 511 = 7 x 73),
and they stay closed while a factor of each stays inside the machine; cycle 39 (1183 = 7 x 13 x
13) is closed at q = 11 and open at q = 13 when 13 enters the engine; at q = 23 the gear 23 moves
into the engine, 391 = 17 x 23 has no factor in [29, 839], and cycle 12 reopens. The band's
start is made of such straddling semiprimes; nothing in it is a property of j.

The new strikes (T3b) show the other face of the same fact: on band 2 the cycles open under
machine 2's new strikes alone are 11, 76, 360, 866, 2221 (against 2, 25, 95, 203, 432 for the
own strikes), i.e. most of machine 2's strikes at the start of its band are echoes; on band 3
the new strikes leave 9990 of 12527, 187157 of 313567 and 3739652 of 7412175 cycles open and
close only 6, 1190, 56612. The joint strikes of machines 1..k close 599 of 991, 3074 of 4477,
7035 of 9737, 18423 of 24225 cycles on band 2 and 9652 of 12527, 261543 of 313567, 6550887 of
7412175 on band 3; the jointly open cycles (three twin pairs in one cycle) are 0, 0, 1, 2, 3, 5
on band 2 at q = 7 .. 23 and 2, 31, 243 on band 3, at j = 601, 3261, 5523, 13075, 22119, 33411, ... - the known
all-prime cycles of anchor-235.md section 5 (a fact, noted and not pursued). The longest
twin-free run of cycles on band 3 is 30 cycles at j = 6264 (q = 17), 50 at j = 95632 (q = 19)
and 100 at j = 4542781 (q = 23; numbers 136,283,430 .. 136,286,460).

## 5. The rules of the band

Law S3 (the single-gear cycle law; known, README glossary and anchor_cycles.md item 2; restated
with the closed-cycle clause). Gear g strikes the six numbers of cycle j at j = -e x 30^{-1}
(mod g), e in {11, 13, 17, 19, 29, 31}: six distinct residues for g >= 11, five for g = 7
(offsets 17 and 31 coincide at j = 2 mod 7). One residue carries a whole slot only if g divides
2; one residue carries two slots only if g divides a difference in {4, 6, 8, 10, 12, 14, 16, 18,
20} - for g >= 7 only 7 | 14 (offsets 17 and 31). So a gear >= 11 strikes at most one number per
cycle, gear 7 at most two, and no gear >= 7 closes a cycle alone. A machine all of whose gears
are >= 11 (every machine k >= 2) closes a cycle only with three distinct gears, one per slot.
Verified: the minimum number of distinct gears over every sampled closed band cycle is 3 for
machines 2 and 3 at every q (T8; 500 cycles sampled where there are more), with histograms
peaking at 5 gears and reaching 11; the engine's minimum is 2, at the gear-7 double (cycle 16 =
2 mod 7: 497 = 7 x 71, 511 = 7 x 73, 493 = 17 x 29).

### T8. Distinct gears per closed band cycle

| q | k | closed cycles sampled | minimum gears | histogram (gears: cycles) | max numbers struck in one cycle |
|---|---|---|---|---|---|
| 11 | 2 | 19 | 3 | {'3': 1, '4': 7, '5': 3, '6': 5, '7': 3} | 4 |
| 13 | 2 | 327 | 3 | {'3': 24, '4': 53, '5': 90, '6': 69, '7': 54, '8': 29, '9': 4, '10': 4} | 6 |
| 17 | 2 | 500 | 3 | {'3': 54, '4': 87, '5': 149, '6': 99, '7': 72, '8': 31, '9': 8} | 6 |
| 17 | 3 | 500 | 3 | {'3': 87, '4': 177, '5': 158, '6': 62, '7': 15, '8': 1} | 6 |
| 19 | 1 | 2 | 2 | {'2': 1, '3': 1} | 4 |
| 19 | 2 | 500 | 3 | {'3': 53, '4': 100, '5': 139, '6': 95, '7': 64, '8': 29, '9': 16, '10': 4} | 6 |
| 19 | 3 | 500 | 3 | {'3': 59, '4': 161, '5': 151, '6': 91, '7': 27, '8': 10, '10': 1} | 6 |
| 23 | 1 | 6 | 2 | {'2': 2, '3': 2, '4': 2} | 4 |
| 23 | 2 | 500 | 3 | {'3': 48, '4': 98, '5': 124, '6': 104, '7': 75, '8': 33, '9': 12, '10': 4, '11': 2} | 6 |
| 23 | 3 | 500 | 3 | {'3': 42, '4': 118, '5': 159, '6': 93, '7': 56, '8': 26, '9': 6} | 6 |

Law S4 (the band composition law; exact). On the band of machine k >= 2, a slot number n is
open under machine k iff n is g_k-smooth, or n = m r with m g_k-smooth and r a prime >= g_{k+1}.
Proof: n < g_{k+1}^2; n free of primes in [g_k, g_k^2] factors as m times primes above g_k^2;
there is no prime in (g_k^2, g_{k+1}) by the definition of g_{k+1}, so those primes are
>= g_{k+1}, and two of them would exceed n. Verified as an assertion of the script at every
banded machine k >= 2 and every q (the cofactor R_k of every slot number below the next square,
after every prime below g_k is divided out, is 1 or a prime above g_k^2 exactly when the strike
array says open; 0 failures). Corollary: on its square part a machine's open numbers are exactly
the g_k-smooth ones, and its open cycles are the cycles of six g_k-smooth numbers (section 4).

What S4 makes of claim D. The open density of machine k on its band is not prod(1 - 6/g) and
cannot be: it is (per slot number) the density of primes, (30/8)/ln x among numbers coprime to
30, plus the density of g_k-smooth multiples of a single large prime, plus the density of
g_k-smooth numbers. T4 measures the three parts exactly per bin of u = ln x / ln g_k and puts
the model next to them: the prime part (30/8)/ln x matches within 0.01 from u = 2.5 on at every
band and within 0.05 in the first bins of machine 2 (0.617 against 0.567 at q = 13 is the
worst; 0.261 against 0.258, 0.234 against 0.231, 0.212 against 0.209, 0.199 against 0.198 on
band 3 at q = 23); the smooth-times-prime part, modelled as the sum over g_k-smooth m coprime to
30 of (30/8)/(m ln(x/m)), matches within 0.025 (0.182 against 0.196, 0.251 against 0.254 on band
3 at q = 23; 0.218 against 0.194 at u = 2.75-3 on band 2), and it switches on at u = 2 + ln 7 /
ln g_k where 7 x g_{k+1} enters (u = 2.69 at g = 17, 2.29 at g = 853). The smooth part is
0.00-0.05 on machine 2's band and 0.12 -> 0.03 on machine 3's; no model is offered for it
(Dickman's rho does not describe g-smooth numbers coprime to 30 at these sizes). The per-number
open density is then 0.49-0.67 on machine 2's band at q >= 11 (CRT 0.507-0.524: above, because
primes are that dense below 10^6 among
numbers coprime to 30) and 0.38-0.48 on machine 3's (CRT 0.502: below, because at x = g_3^2
only primes and smooth numbers escape and the primes are 0.26-0.30 per number). Per slot the
own open fraction runs 1.20 -> 0.94 x CRT along band 2 at q = 23 and 0.57 -> 0.93 x CRT along
band 3. The curves are not the same function of u for k = 2 and k = 3 (P5 refuted on that
clause): the prime term carries 1/ln x = 1/(u ln g_k) and the machines differ in g_k by a factor
30.

### T4. The band profile

| q | k | u | cycles | number open | prime | smooth x prime | smooth | model prime | model mixed | CRT per number | slot open | CRT slot | own/CRT | cycle open | cycle closed | twins | joint/product |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 1 | 2.19-2.25 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.857 | 0.333 | 0.714 | 0.47 | 0.0000 | 0.000 | 1 | 1.00 |
| 7 | 2 | 2.03-2.23 | 3 | 0.778 | 0.778 | 0.000 | 0.000 | 0.730 | 0.000 | 0.502 | 0.556 | 0.244 | 2.28 | 0.0000 | 0.000 | 5 | 1.29 |
| 11 | 1 | 2.19-2.25 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.779 | 0.333 | 0.584 | 0.57 | 0.0000 | 0.000 | 1 | 1.00 |
| 11 | 1 | 2.25-2.50 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.779 | 0.667 | 0.584 | 1.14 | 0.0000 | 0.000 | 2 | 1.00 |
| 11 | 1 | 2.50-2.58 | 1 | 0.833 | 0.833 | 0.000 | 0.000 | - | - | 0.779 | 0.667 | 0.584 | 1.14 | 0.0000 | 0.000 | 2 | 1.00 |
| 11 | 2 | 2.05-2.25 | 5 | 0.667 | 0.667 | 0.000 | 0.000 | 0.677 | 0.000 | 0.519 | 0.467 | 0.263 | 1.77 | 0.0000 | 0.000 | 7 | 1.88 |
| 11 | 2 | 2.25-2.50 | 9 | 0.593 | 0.556 | 0.000 | 0.037 | 0.612 | 0.000 | 0.519 | 0.296 | 0.263 | 1.13 | 0.0000 | 0.222 | 7 | 1.39 |
| 11 | 2 | 2.50-2.75 | 19 | 0.544 | 0.544 | 0.000 | 0.000 | 0.556 | 0.000 | 0.519 | 0.246 | 0.263 | 0.93 | 0.0000 | 0.421 | 14 | 1.78 |
| 11 | 2 | 2.75-3.00 | 34 | 0.627 | 0.505 | 0.118 | 0.005 | 0.508 | 0.098 | 0.519 | 0.392 | 0.263 | 1.49 | 0.0588 | 0.265 | 26 | 1.12 |
| 11 | 2 | 3.00-3.02 | 4 | 0.583 | 0.542 | 0.042 | 0.000 | 0.486 | 0.157 | 0.519 | 0.333 | 0.263 | 1.27 | 0.0000 | 0.000 | 3 | 1.12 |
| 13 | 1 | 2.19-2.25 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.719 | 0.333 | 0.495 | 0.67 | 0.0000 | 0.000 | 1 | 1.00 |
| 13 | 1 | 2.25-2.50 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.719 | 0.667 | 0.495 | 1.35 | 0.0000 | 0.000 | 2 | 1.00 |
| 13 | 1 | 2.50-2.75 | 3 | 0.778 | 0.778 | 0.000 | 0.000 | - | - | 0.719 | 0.556 | 0.495 | 1.12 | 0.0000 | 0.000 | 5 | 1.00 |
| 13 | 1 | 2.75-2.88 | 2 | 0.750 | 0.750 | 0.000 | 0.000 | - | - | 0.719 | 0.500 | 0.495 | 1.01 | 0.0000 | 0.000 | 3 | 1.00 |
| 13 | 2 | 2.03-2.25 | 10 | 0.600 | 0.567 | 0.000 | 0.033 | 0.617 | 0.000 | 0.510 | 0.300 | 0.255 | 1.17 | 0.0000 | 0.200 | 8 | 1.90 |
| 13 | 2 | 2.25-2.50 | 20 | 0.558 | 0.542 | 0.000 | 0.017 | 0.555 | 0.000 | 0.510 | 0.250 | 0.255 | 0.98 | 0.0000 | 0.400 | 14 | 2.07 |
| 13 | 2 | 2.50-2.75 | 41 | 0.549 | 0.504 | 0.028 | 0.016 | 0.503 | 0.000 | 0.510 | 0.309 | 0.255 | 1.21 | 0.0000 | 0.293 | 31 | 1.62 |
| 13 | 2 | 2.75-3.00 | 83 | 0.594 | 0.448 | 0.145 | 0.002 | 0.460 | 0.146 | 0.510 | 0.345 | 0.255 | 1.35 | 0.0361 | 0.289 | 52 | 1.21 |
| 13 | 2 | 3.00-3.25 | 169 | 0.594 | 0.420 | 0.170 | 0.004 | 0.423 | 0.176 | 0.510 | 0.353 | 0.255 | 1.38 | 0.0355 | 0.272 | 81 | 0.91 |
| 13 | 2 | 3.25-3.50 | 342 | 0.553 | 0.387 | 0.164 | 0.001 | 0.392 | 0.173 | 0.510 | 0.292 | 0.255 | 1.14 | 0.0234 | 0.351 | 140 | 0.95 |
| 13 | 2 | 3.50-3.64 | 326 | 0.536 | 0.368 | 0.166 | 0.002 | 0.371 | 0.169 | 0.510 | 0.277 | 0.255 | 1.08 | 0.0245 | 0.353 | 123 | 0.92 |
| 17 | 1 | 2.19-2.25 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.677 | 0.333 | 0.436 | 0.76 | 0.0000 | 0.000 | 1 | 1.00 |
| 17 | 1 | 2.25-2.50 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.677 | 0.667 | 0.436 | 1.53 | 0.0000 | 0.000 | 2 | 1.00 |
| 17 | 1 | 2.50-2.75 | 3 | 0.778 | 0.778 | 0.000 | 0.000 | - | - | 0.677 | 0.556 | 0.436 | 1.27 | 0.0000 | 0.000 | 5 | 1.00 |
| 17 | 1 | 2.75-2.98 | 4 | 0.625 | 0.625 | 0.000 | 0.000 | - | - | 0.677 | 0.417 | 0.436 | 0.95 | 0.0000 | 0.000 | 5 | 1.00 |
| 17 | 2 | 2.01-2.25 | 13 | 0.590 | 0.577 | 0.000 | 0.013 | 0.598 | 0.000 | 0.524 | 0.256 | 0.271 | 0.95 | 0.0000 | 0.385 | 9 | 2.93 |
| 17 | 2 | 2.25-2.50 | 28 | 0.554 | 0.518 | 0.000 | 0.036 | 0.535 | 0.000 | 0.524 | 0.286 | 0.271 | 1.06 | 0.0357 | 0.357 | 20 | 1.94 |
| 17 | 2 | 2.50-2.75 | 57 | 0.532 | 0.471 | 0.044 | 0.018 | 0.484 | 0.000 | 0.524 | 0.275 | 0.271 | 1.02 | 0.0175 | 0.404 | 37 | 1.82 |
| 17 | 2 | 2.75-3.00 | 119 | 0.604 | 0.441 | 0.158 | 0.004 | 0.443 | 0.187 | 0.524 | 0.364 | 0.271 | 1.35 | 0.0504 | 0.277 | 73 | 1.25 |
| 17 | 2 | 3.00-3.25 | 248 | 0.602 | 0.397 | 0.200 | 0.005 | 0.407 | 0.202 | 0.524 | 0.348 | 0.271 | 1.29 | 0.0282 | 0.246 | 104 | 0.92 |
| 17 | 2 | 3.25-3.50 | 520 | 0.567 | 0.376 | 0.188 | 0.003 | 0.377 | 0.195 | 0.524 | 0.315 | 0.271 | 1.16 | 0.0385 | 0.310 | 203 | 0.95 |
| 17 | 2 | 3.50-3.75 | 1084 | 0.546 | 0.347 | 0.196 | 0.002 | 0.351 | 0.197 | 0.524 | 0.296 | 0.271 | 1.09 | 0.0258 | 0.339 | 364 | 0.87 |
| 17 | 2 | 3.75-4.00 | 2263 | 0.523 | 0.326 | 0.197 | 0.001 | 0.329 | 0.199 | 0.524 | 0.268 | 0.271 | 0.99 | 0.0137 | 0.375 | 688 | 0.87 |
| 17 | 2 | 4.00-4.01 | 145 | 0.509 | 0.321 | 0.189 | 0.000 | 0.318 | 0.194 | 0.524 | 0.253 | 0.271 | 0.93 | 0.0069 | 0.386 | 40 | 0.84 |
| 17 | 3 | 2.00-2.23 | 12527 | 0.411 | 0.297 | 0.000 | 0.114 | 0.301 | 0.000 | 0.503 | 0.166 | 0.253 | 0.66 | 0.0043 | 0.578 | 3076 | 4.32 |
| 19 | 1 | 2.19-2.25 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.641 | 0.333 | 0.390 | 0.85 | 0.0000 | 0.000 | 1 | 1.00 |
| 19 | 1 | 2.25-2.50 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.641 | 0.667 | 0.390 | 1.71 | 0.0000 | 0.000 | 2 | 1.00 |
| 19 | 1 | 2.50-2.75 | 3 | 0.778 | 0.778 | 0.000 | 0.000 | - | - | 0.641 | 0.556 | 0.390 | 1.42 | 0.0000 | 0.000 | 5 | 1.00 |
| 19 | 1 | 2.75-3.00 | 5 | 0.600 | 0.600 | 0.000 | 0.000 | - | - | 0.641 | 0.400 | 0.390 | 1.02 | 0.0000 | 0.000 | 6 | 1.00 |
| 19 | 1 | 3.00-3.21 | 5 | 0.600 | 0.600 | 0.000 | 0.000 | - | - | 0.641 | 0.200 | 0.390 | 0.51 | 0.0000 | 0.400 | 3 | 1.00 |
| 19 | 2 | 2.01-2.25 | 20 | 0.558 | 0.542 | 0.000 | 0.017 | 0.558 | 0.000 | 0.520 | 0.267 | 0.267 | 1.00 | 0.0000 | 0.350 | 15 | 2.96 |
| 19 | 2 | 2.25-2.50 | 46 | 0.547 | 0.500 | 0.000 | 0.047 | 0.503 | 0.000 | 0.520 | 0.283 | 0.267 | 1.06 | 0.0217 | 0.391 | 32 | 1.99 |
| 19 | 2 | 2.50-2.75 | 100 | 0.507 | 0.445 | 0.047 | 0.015 | 0.455 | 0.085 | 0.520 | 0.270 | 0.267 | 1.01 | 0.0300 | 0.400 | 63 | 2.01 |
| 19 | 2 | 2.75-3.00 | 221 | 0.604 | 0.406 | 0.189 | 0.009 | 0.416 | 0.172 | 0.520 | 0.360 | 0.267 | 1.35 | 0.0226 | 0.244 | 103 | 1.08 |
| 19 | 2 | 3.00-3.25 | 482 | 0.593 | 0.382 | 0.206 | 0.005 | 0.383 | 0.215 | 0.520 | 0.344 | 0.267 | 1.29 | 0.0477 | 0.270 | 184 | 0.95 |
| 19 | 2 | 3.25-3.50 | 1057 | 0.565 | 0.351 | 0.210 | 0.004 | 0.354 | 0.205 | 0.520 | 0.315 | 0.267 | 1.18 | 0.0312 | 0.307 | 370 | 0.95 |
| 19 | 2 | 3.50-3.75 | 2315 | 0.544 | 0.326 | 0.216 | 0.002 | 0.330 | 0.218 | 0.520 | 0.292 | 0.267 | 1.09 | 0.0186 | 0.342 | 700 | 0.88 |
| 19 | 2 | 3.75-4.00 | 5068 | 0.524 | 0.308 | 0.215 | 0.001 | 0.309 | 0.219 | 0.520 | 0.271 | 0.267 | 1.01 | 0.0182 | 0.380 | 1342 | 0.84 |
| 19 | 2 | 4.00-4.01 | 427 | 0.509 | 0.296 | 0.211 | 0.002 | 0.298 | 0.212 | 0.520 | 0.268 | 0.267 | 1.00 | 0.0070 | 0.342 | 107 | 0.81 |
| 19 | 3 | 2.00-2.25 | 37296 | 0.393 | 0.276 | 0.000 | 0.117 | 0.280 | 0.000 | 0.502 | 0.153 | 0.252 | 0.61 | 0.0034 | 0.606 | 8088 | 4.66 |
| 19 | 3 | 2.25-2.50 | 179868 | 0.395 | 0.248 | 0.075 | 0.072 | 0.251 | 0.041 | 0.502 | 0.156 | 0.252 | 0.62 | 0.0039 | 0.601 | 31193 | 3.68 |
| 19 | 3 | 2.50-2.56 | 96403 | 0.427 | 0.236 | 0.136 | 0.055 | 0.236 | 0.130 | 0.502 | 0.182 | 0.252 | 0.72 | 0.0058 | 0.547 | 15230 | 2.83 |
| 23 | 1 | 2.19-2.25 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.613 | 0.333 | 0.356 | 0.94 | 0.0000 | 0.000 | 1 | 1.00 |
| 23 | 1 | 2.25-2.50 | 1 | 0.667 | 0.667 | 0.000 | 0.000 | - | - | 0.613 | 0.667 | 0.356 | 1.87 | 0.0000 | 0.000 | 2 | 1.00 |
| 23 | 1 | 2.50-2.75 | 3 | 0.778 | 0.778 | 0.000 | 0.000 | - | - | 0.613 | 0.556 | 0.356 | 1.56 | 0.0000 | 0.000 | 5 | 1.00 |
| 23 | 1 | 2.75-3.00 | 5 | 0.600 | 0.600 | 0.000 | 0.000 | - | - | 0.613 | 0.400 | 0.356 | 1.12 | 0.0000 | 0.000 | 6 | 1.00 |
| 23 | 1 | 3.00-3.25 | 7 | 0.571 | 0.571 | 0.000 | 0.000 | - | - | 0.613 | 0.238 | 0.356 | 0.67 | 0.0000 | 0.286 | 5 | 1.00 |
| 23 | 1 | 3.25-3.44 | 8 | 0.542 | 0.542 | 0.000 | 0.000 | - | - | 0.613 | 0.208 | 0.356 | 0.58 | 0.0000 | 0.500 | 5 | 1.00 |
| 23 | 2 | 2.00-2.25 | 36 | 0.565 | 0.514 | 0.000 | 0.051 | 0.522 | 0.000 | 0.507 | 0.306 | 0.255 | 1.20 | 0.0278 | 0.306 | 26 | 2.36 |
| 23 | 2 | 2.25-2.50 | 86 | 0.494 | 0.459 | 0.000 | 0.035 | 0.469 | 0.000 | 0.507 | 0.264 | 0.255 | 1.03 | 0.0116 | 0.384 | 58 | 2.42 |
| 23 | 2 | 2.50-2.75 | 199 | 0.504 | 0.420 | 0.070 | 0.015 | 0.424 | 0.078 | 0.507 | 0.251 | 0.255 | 0.99 | 0.0050 | 0.392 | 98 | 1.80 |
| 23 | 2 | 2.75-3.00 | 463 | 0.589 | 0.388 | 0.194 | 0.007 | 0.387 | 0.218 | 0.507 | 0.341 | 0.255 | 1.34 | 0.0432 | 0.274 | 189 | 1.11 |
| 23 | 2 | 3.00-3.25 | 1074 | 0.578 | 0.351 | 0.221 | 0.006 | 0.356 | 0.217 | 0.507 | 0.331 | 0.255 | 1.30 | 0.0354 | 0.282 | 370 | 0.98 |
| 23 | 2 | 3.25-3.50 | 2491 | 0.549 | 0.326 | 0.220 | 0.003 | 0.330 | 0.220 | 0.507 | 0.298 | 0.255 | 1.17 | 0.0213 | 0.337 | 753 | 0.95 |
| 23 | 2 | 3.50-3.75 | 5782 | 0.536 | 0.306 | 0.230 | 0.001 | 0.307 | 0.231 | 0.507 | 0.285 | 0.255 | 1.12 | 0.0202 | 0.358 | 1507 | 0.86 |
| 23 | 2 | 3.75-4.00 | 13416 | 0.511 | 0.286 | 0.225 | 0.001 | 0.287 | 0.228 | 0.507 | 0.260 | 0.255 | 1.02 | 0.0147 | 0.398 | 3078 | 0.83 |
| 23 | 2 | 4.00-4.01 | 677 | 0.501 | 0.280 | 0.220 | 0.001 | 0.278 | 0.220 | 0.507 | 0.239 | 0.255 | 0.94 | 0.0059 | 0.421 | 144 | 0.83 |
| 23 | 3 | 2.00-2.25 | 106818 | 0.381 | 0.258 | 0.000 | 0.123 | 0.261 | 0.000 | 0.502 | 0.145 | 0.252 | 0.57 | 0.0030 | 0.625 | 20226 | 4.95 |
| 23 | 3 | 2.25-2.50 | 577283 | 0.392 | 0.231 | 0.085 | 0.076 | 0.234 | 0.063 | 0.502 | 0.153 | 0.252 | 0.61 | 0.0035 | 0.606 | 86781 | 3.72 |
| 23 | 3 | 2.50-2.75 | 3119791 | 0.450 | 0.209 | 0.196 | 0.045 | 0.212 | 0.182 | 0.502 | 0.202 | 0.252 | 0.80 | 0.0081 | 0.508 | 384402 | 2.26 |
| 23 | 3 | 2.75-2.85 | 3608282 | 0.484 | 0.198 | 0.254 | 0.032 | 0.199 | 0.251 | 0.502 | 0.234 | 0.252 | 0.93 | 0.0126 | 0.448 | 398396 | 1.73 |

Law S5 (the half law; Mertens, known, applied to the construction). For a machine with gears
exactly the primes of [g, g^2], prod (1 - 1/p) = ln g / ln g^2 x (1 + O(1 / ln g)) = 1/2 + O(1 /
ln g); hence prod(1 - 2/p) = 1/4 and prod(1 - 6/p) = 1/64 in the same sense, and the CRT
classification of every machine k >= 2 is (open : closed : mixed) = (1 : 27 : 36)/64 in the
limit. Measured (T9): prod(1 - 1/p) = 0.5021, 0.5191, 0.5098, 0.5238, 0.5199, 0.5069 at machine
2 (26 to 137 gears) and 0.5026, 0.5016, 0.5016 at machine 3 (12481 to 58462 gears); slot open
0.2438-0.2707 and 0.2515-0.2526; cycle open 0.0085-0.0166 and 0.01587-0.01603 against 1/64 =
0.015625; cycle closed 0.379-0.413 and 0.4174-0.4192 against 27/64 = 0.4219. The construction
therefore makes every machine an equal-weight object: each halves the density of numbers it
leaves open, and each leaves the same CRT share of cycles open, closed and mixed. The engine is
the one machine of different weight (0.857 -> 0.613 at q = 7 .. 23).

### T9. CRT densities against the half law

| q | k | gears | prod(1 - 1/p) | prod(1 - 2/p) (slot open) | cycle open prod(1 - 6/p) | cycle closed | cycle mixed | 1/2 | 1/4 | 1/64 | 27/64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 1 | 1 | 0.8571 | 0.7143 | 0.28571 | 0.0000 | 0.7143 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 7 | 2 | 26 | 0.5021 | 0.2438 | 0.00848 | 0.4126 | 0.5789 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 11 | 1 | 2 | 0.7792 | 0.5844 | 0.12987 | 0.0260 | 0.8442 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 11 | 2 | 34 | 0.5191 | 0.2631 | 0.01277 | 0.3844 | 0.6028 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 13 | 1 | 3 | 0.7193 | 0.4945 | 0.06993 | 0.0759 | 0.8541 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 13 | 2 | 55 | 0.5098 | 0.2555 | 0.01308 | 0.4020 | 0.5849 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 17 | 1 | 4 | 0.6770 | 0.4363 | 0.04525 | 0.1270 | 0.8277 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 17 | 2 | 65 | 0.5238 | 0.2707 | 0.01649 | 0.3789 | 0.6046 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 17 | 3 | 12481 | 0.5026 | 0.2526 | 0.01603 | 0.4174 | 0.5666 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 19 | 1 | 5 | 0.6413 | 0.3904 | 0.03096 | 0.1778 | 0.7912 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 19 | 2 | 91 | 0.5199 | 0.2674 | 0.01663 | 0.3862 | 0.5972 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 19 | 3 | 25339 | 0.5016 | 0.2516 | 0.01587 | 0.4191 | 0.5650 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 23 | 1 | 6 | 0.6135 | 0.3565 | 0.02288 | 0.2216 | 0.7555 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 23 | 2 | 137 | 0.5069 | 0.2548 | 0.01478 | 0.4087 | 0.5765 | 0.5 | 0.25 | 0.01562 | 0.4219 |
| 23 | 3 | 58462 | 0.5016 | 0.2515 | 0.01589 | 0.4192 | 0.5649 | 0.5 | 0.25 | 0.01562 | 0.4219 |

The closed cycles against CRT. The closed count of machine 2 on its band is 0.70-0.92 x CRT and
of machine 3 1.16-1.40 x (section 4): the same S4 mechanism as the open count, since a cycle is
closed when each of its three slots has a struck member and the struck members at the start of
a band are the rough composites and echoes, fewer than CRT for machine 2 (primes dense) and more
for machine 3.

Law S6 (the rough-to-prime law; exact; the "connection to machine 1"). On band k a slot number
is open under machines 1..k-1 iff all its prime factors are >= g_k (g_k-rough), and open under
machines 1..k iff it is prime (claim A). So on the slots the lower machines leave open, machine
k's entire action is to strike the g_k-rough composites, and every one of its strikes there is
new (a rough composite has no factor below g_k, and its smallest prime factor is at most
sqrt n < g_{k+1}, so it lies in machine k). The independence ratio of T5, joint /
(product of individual open fractions), is therefore not a free quantity: per number it is
P(open under k | rough) / P(open under k) = P(prime | rough) / P(open under k). At the start of
the band every rough number is prime except g_k^2 and g_k g_k' (a rough composite is at least
g_k^2), so the ratio there is 1 / P(open under k) per number and about its square per slot: with
P(open under k) = 0.38-0.41 per number at the start of band 3 that is 6-7 per slot for
independent members, measured 4.3-5.0 in the first bin; with 0.56-0.67 at the start of band 2
(q >= 11), 2.2-3.2, measured 1.9-3.0. Along the band P(prime | rough) falls as products of two,
three and four primes >= g_k fill in, while P(open under k) rises toward 1/2; at u = 4 the
per-number ratio is (e^gamma ln g_k / ln x) / (1/2) = e^gamma / 2 = 0.89 by Mertens, and the
per-slot value measured at u = 3.75-4.01 is 0.81-0.87 at q = 17, 19, 23. The ratio is 1 only
where the falling curve crosses it, between u = 2.75 and 3.25 on band 2 (1.08-1.25 in the bin
below, 0.91-0.98 in the bin above).

### T5. The connection to machine 1

| q | k | band slots | open fraction by machine 1..k | product | joint (twins) | joint fraction | ratio | cycles open by machine 1..k | jointly open cycles |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 1 | 3 | 0.3333 | 0.33333 | 1 | 0.33333 | 1.000 | [0] | 0 |
| 7 | 2 | 9 | 0.7778, 0.5556 | 0.43210 | 5 | 0.55556 | 1.286 | [1, 0] | 0 |
| 11 | 1 | 9 | 0.5556 | 0.55556 | 5 | 0.55556 | 1.000 | [0] | 0 |
| 11 | 2 | 213 | 0.5822, 0.3427 | 0.19952 | 57 | 0.26761 | 1.341 | [10, 2] | 0 |
| 13 | 1 | 21 | 0.5238 | 0.52381 | 11 | 0.52381 | 1.000 | [0] | 0 |
| 13 | 2 | 2973 | 0.4941, 0.3021 | 0.14925 | 449 | 0.15103 | 1.012 | [70, 25] | 1 |
| 17 | 1 | 27 | 0.4815 | 0.48148 | 13 | 0.48148 | 1.000 | [0] | 0 |
| 17 | 2 | 13431 | 0.4364, 0.2868 | 0.12515 | 1538 | 0.11451 | 0.915 | [203, 95] | 2 |
| 17 | 3 | 37581 | 0.4363, 0.2613, 0.1664 | 0.01896 | 3076 | 0.08185 | 4.316 | [567, 192, 54] | 2 |
| 19 | 1 | 45 | 0.3778 | 0.37778 | 17 | 0.37778 | 1.000 | [0] | 0 |
| 19 | 2 | 29211 | 0.3901, 0.2861 | 0.11161 | 2917 | 0.09986 | 0.895 | [296, 203] | 3 |
| 19 | 3 | 940701 | 0.3904, 0.2600, 0.1633 | 0.01658 | 54511 | 0.05795 | 3.496 | [9714, 4893, 1384] | 31 |
| 23 | 1 | 75 | 0.3200 | 0.32000 | 24 | 0.32000 | 1.000 | [0] | 0 |
| 23 | 2 | 72675 | 0.3564, 0.2739 | 0.09761 | 6224 | 0.08564 | 0.877 | [552, 432] | 5 |
| 23 | 3 | 22236525 | 0.3565, 0.2531, 0.2130 | 0.01922 | 889805 | 0.04002 | 2.082 | [169617, 107169, 73124] | 243 |

The whole-band ratios (T5) are the u-average of the falling curve: 1.29, 1.34, 1.01, 0.92,
0.90, 0.88 on band 2 at q = 7 .. 23 (the band lengthens in u as q grows, so more of the low end
of the curve is included) and 4.32, 3.50, 2.08 on band 3 at q = 17, 19, 23 (the band covers only
u <= 2.23, 2.56, 2.85, the top of the curve). The per-machine open fractions on band 3 at
q = 23 are 0.3565 (engine), 0.2531 (machine 2), 0.2130 (machine 3): the two square-built
machines both sit near 1/4 (S5) and the engine at its own prod(1 - 2/g) = 0.3565 (the band is
99.7 % of the engine's period, so the engine is at its CRT value to seven digits: 0.3564514
measured against 0.3564513).

## 6. The pattern across machines and across q

Law S7 (the cross-machine identity; exact by construction, verified). For k >= 2, the gears of
machine k + 1 are exactly the primes of band k: band k = [g_k^2, g_{k+1}^2), there is no prime in
(g_k^2, g_{k+1}), and g_{k+1}^2 is composite. Verified equal at every q (T6: 16, 304, 3187,
12481 / 29778, 25339 / 620591, 58462 / 12224923 primes = gears). The engine is the exception
(its gears stop at q < 49, and machine 2's gears begin at q' below 49). Consequently the twin
prime pairs of band k are exactly the slots on which machine k + 1 makes two home strikes: 4, 5,
6, 57, 13, 449, 15, 1539, 3076, 19, 2917, 54511, 27, 6224, 889805 twins = double-home slots at
all 16 bands, 0 mismatches. And the start of band k + 1 is the square of the first prime above
the start of band k: 121 -> 127^2, 169 -> 173^2, 289 -> 293^2, 361 -> 367^2 -> 134699^2, 529 ->
541^2 -> 292693^2, 841 -> 853^2 -> 727613^2. In log scale band k spans [2 ln g_k, 4 ln g_k] up
to the gap to the next prime, so each band is twice as long in ln x as the one below, which is
the whole content of the count S9.

So the chain reads: machine k's band is where machines 1..k are exact and machine k + 1 is
silent (home and echo); the twins of that band become machine k + 1's paired gears; machine
k + 1's square part is the union of all lower bands; on it machine k + 1 is home and echo, and
its open cycles are the cycles of six g_{k+1}-smooth numbers, whose last one is finite (j = 44
for g = 293, 367, 541; j = 4986 for g = 853; the initial run of open cycles is the cycles below
g_{k+1}, 4489, 9756, 24253 cycles for the last machine at q = 17, 19, 23).

### T6. Cross-machine relations

| q | k | band k | primes in band k | gears of k+1 (below q#) | equal | twins in band k | double-home slots of k+1 | mismatch | closed j of k mapped by the square | landing on closed j' of k+1 | rate | closed density of k+1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | [49, 31) | 0 | 7 | False | 0 | 0 | 0 | - | - | - | - |
| 7 | 1 | [49, 121) | 15 | 26 | False | 4 | 4 | 0 | - | - | - | - |
| 7 | 2 | [121, 211) | 16 | 16 | True | 5 | 5 | 0 | - | - | - | - |
| 11 | 1 | [49, 169) | 24 | 34 | False | 6 | 6 | 0 | - | - | - | - |
| 11 | 2 | [169, 2311) | 304 | 304 | True | 57 | 57 | 0 | - | - | - | - |
| 13 | 1 | [49, 289) | 46 | 55 | False | 13 | 13 | 0 | - | - | - | - |
| 13 | 2 | [289, 30031) | 3187 | 3187 | True | 449 | 449 | 0 | - | - | - | - |
| 17 | 1 | [49, 361) | 57 | 65 | False | 15 | 15 | 0 | - | - | - | - |
| 17 | 2 | [361, 134689) | 12481 | 12481 | True | 1539 | 1539 | 0 | 4 | 2 | 0.500 | 0.578 |
| 17 | 3 | [134689, 510511) | 29778 | 29778 | True | 3076 | 3076 | 0 | - | - | - | - |
| 19 | 1 | [49, 529) | 84 | 91 | False | 19 | 19 | 0 | 2 | 0 | 0.000 | 0.353 |
| 19 | 2 | [529, 292681) | 25339 | 25339 | True | 2917 | 2917 | 0 | 35 | 23 | 0.657 | 0.585 |
| 19 | 3 | [292681, 9699691) | 620591 | 620591 | True | 54511 | 54511 | 0 | - | - | - | - |
| 23 | 1 | [49, 841) | 131 | 137 | False | 27 | 27 | 0 | 6 | 3 | 0.500 | 0.375 |
| 23 | 2 | [841, 727609) | 58462 | 58462 | True | 6224 | 6224 | 0 | 174 | 90 | 0.517 | 0.488 |
| 23 | 3 | [727609, 223092871) | 12224923 | 12224923 | True | 889805 | 889805 | 0 | - | - | - | - |

The closed-cycle positions of machine k + 1 against the squares of machine k's: the map j ->
floor((30j + 11)^2 / 30) sends 4, 35, 174 closed cycles of machine 2 onto band 3 at q = 17, 19,
23, of which 2, 23, 90 land on closed cycles of machine 3, rates 0.50, 0.66, 0.52 against
machine 3's closed density 0.58, 0.59, 0.49 on its band: chance. There is no relation through
the squares (P7's second clause confirmed); the relation between consecutive machines is S7,
through the primes, not through positions.

The same statistic across q at fixed k (T3, T9): machine 2's band-open fraction 0.028, 0.025,
0.021, 0.021, 0.018 (q = 11 .. 23) falls toward its CRT 0.013-0.017 as the band lengthens in u;
machine 3's 0.0043, 0.0044, 0.0099 rises toward its CRT 0.016 as its band reaches further
(u to 2.23, 2.56, 2.85). Machine 3's CRT values are the same to three digits at the three q
(0.01603, 0.01587, 0.01589 open; 0.4174, 0.4191, 0.4192 closed): the half law S5 makes the
machines' full-period statistics independent of q and k, and everything that varies with q and
k in the tables is the band's position u and the prime density 1/(u ln g_k), i.e. law S4.

Law S8 (the cycle at q#). The cycle j = q#/30 is the mirror of cycle 0 under the engine:
q# + e is struck by a prime p <= q iff p | e. So under the engine it is open for q <= 7 and
mixed from q = 11, the slot (q# + 11, q# + 13) struck from q = 11 (by 11) and the slot (q# + 17,
q# + 19) from q = 17 (by 17), the slot (q# + 29, q# + 31) open at every q < 29; the top cycle
inside the period, q# - 19 .. q# + 1, reads the same by the mirror e -> -e (T7: struck patterns
x..... at q = 11, xx.... at q = 13, xxx... at q = 17, xxxx.. at q = 19, 23 for the cycle at q#).
Under the machines k >= 2 the cycle at q# is open at 0 of 16 instances: mixed at 7 (machine 2
at q = 5, 7, 11, 13, 23 and machine 4 at q = 17, 19; machine 3 at none) and closed at 9 (machine
3 at q = 7 .. 23, machine 2 at q = 17, 19, machine 4 at q = 23). The chance of an open cycle
under a square-built machine is about 1/64 per instance, so 0 of 16 is what chance gives; claim
C is refuted as stated and the numbers q# + e are not coprime to the primes below q. One more
exact reading of T7: machine 3's new strikes on the cycle at q# are empty at q = 17, 19, 23 (all
its strikes there are echoes of the engine and machine 2), and the top cycle is open under the
last machine at q = 17 (510491 .. 510511 are all 134683-smooth) and under no other.

### T7. The top cycle and the cycle at q#

| q | k | top cycle j | numbers | struck by k | class | cycle at q# j | numbers | struck by k | class | new by k |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 0 | 11..31 | ...... | open | 1 | 41..61 | ...... | open | ...... |
| 5 | 2 | 0 | 11..31 | xxxxxx | closed | 1 | 41..61 | xxxx.. | mixed | xxxx.. |
| 7 | 1 | 6 | 191..211 | ...... | open | 7 | 221..241 | ...... | open | ...... |
| 7 | 2 | 6 | 191..211 | ....x. | mixed | 7 | 221..241 | x..... | mixed | x..... |
| 7 | 3 | 6 | 191..211 | xxxx.x | closed | 7 | 221..241 | .xxxxx | closed | .xxxxx |
| 11 | 1 | 76 | 2291..2311 | ...x.. | mixed | 77 | 2321..2341 | x..... | mixed | x..... |
| 11 | 2 | 76 | 2291..2311 | x..x.. | mixed | 77 | 2321..2341 | .xxx.. | mixed | .xxx.. |
| 11 | 3 | 76 | 2291..2311 | .xx.xx | closed | 77 | 2321..2341 | x.x.xx | closed | ....xx |
| 13 | 1 | 1000 | 30011..30031 | ..xx.. | mixed | 1001 | 30041..30061 | xx.... | mixed | xx.... |
| 13 | 2 | 1000 | 30011..30031 | .....x | mixed | 1001 | 30041..30061 | ...x.x | mixed | ...x.x |
| 13 | 3 | 1000 | 30011..30031 | xxxxxx | closed | 1001 | 30041..30061 | xxx.xx | closed | ..x.x. |
| 17 | 1 | 17016 | 510491..510511 | .xxx.. | mixed | 17017 | 510521..510541 | xxx... | mixed | xxx... |
| 17 | 2 | 17016 | 510491..510511 | x.x.xx | closed | 17017 | 510521..510541 | .xx.xx | closed | ....xx |
| 17 | 3 | 17016 | 510491..510511 | xxxxx. | closed | 17017 | 510521..510541 | x.x.xx | closed | ...... |
| 17 | 4 | 17016 | 510491..510511 | ...... | open | 17017 | 510521..510541 | ...x.. | mixed | ...x.. |
| 19 | 1 | 323322 | 9699671..9699691 | xxxx.. | mixed | 323323 | 9699701..9699721 | xxxx.. | mixed | xxxx.. |
| 19 | 2 | 323322 | 9699671..9699691 | x...xx | mixed | 323323 | 9699701..9699721 | .x.xxx | closed | ....xx |
| 19 | 3 | 323322 | 9699671..9699691 | x..xxx | closed | 323323 | 9699701..9699721 | xxx.x. | closed | ...... |
| 19 | 4 | 323322 | 9699671..9699691 | .xx... | mixed | 323323 | 9699701..9699721 | .....x | mixed | ...... |
| 23 | 1 | 7436428 | 223092851..223092871 | xxxx.. | mixed | 7436429 | 223092881..223092901 | xxxx.. | mixed | xxxx.. |
| 23 | 2 | 7436428 | 223092851..223092871 | .x..xx | mixed | 7436429 | 223092881..223092901 | ....xx | mixed | ....xx |
| 23 | 3 | 7436428 | 223092851..223092871 | .x..xx | mixed | 7436429 | 223092881..223092901 | x..xx. | closed | ...... |
| 23 | 4 | 7436428 | 223092851..223092871 | x.xx.. | mixed | 7436429 | 223092881..223092901 | .xx..x | closed | ...... |

## 7. What is new (checked against docs/novel/README.md)

- S1: every prime's square is a slot number (g^2 = 1 or 19 mod 30 by class), so a machine's
  first genuine strike lands on a twin slot; the square part must be half-open. Not in the index
  (the square gate is on record in its own coordinate for the exclusive kill; the slot-type
  dichotomy and the boundary consequence are not). Elementary; a definition fact, not a route.
- S4, the band composition law and its measured decomposition: the open numbers of a
  square-built machine on its band are the smooth numbers and the smooth multiples of one prime
  above the machine, exactly, with the prime part (30/8)/ln x and the smooth-times-prime part
  matched to 0.01-0.02 at every bin. Prior art line: the factorisation is the Buchstab identity
  for numbers free of primes in an interval (standard); the exactness (at most one large prime)
  is forced by building to the square; the measurement on the machines is new to the project.
  Use toward the root: it says exactly why prod(1 - 6/g) is the wrong expectation on a band and
  what the right one is; it is a description of a density, not an existence statement.
- S5 applied: every square-built machine has Mertens weight 1/2 and the same CRT cycle
  classification (1 : 27 : 36)/64, measured to 0.5 % at machine 3. Known (Mertens); the
  observation that the owner's construction is exactly the equal-weight decomposition of the
  manifold is the new reading, and it is what makes claim B a count (each machine doubles the
  log-length).
- S6, the rough-to-prime law: on band k the lower machines leave the g_k-rough pairs and machine
  k strikes exactly the rough composites, all as new strikes; the independence ratio is
  P(prime | rough) / P(open under k), 1 / P(open under k) at the band's start, e^gamma / 2 per
  number at u = 4. Not in the index. It closes claim D's third part with a mechanism and turns
  "the deviation from independence" into a curve with a closed form at both ends. Use toward the
  root: none beyond description - it is the twin count on the band written as a conditional
  density, and the twin count is the root.
- S7, gears of k + 1 = primes of band k and twins of band k = double-home slots of k + 1, with
  the squares chain of band starts. Exact by construction; not in the index; the one structural
  relation between consecutive machines, and it runs through the primes, not through positions.
- S8 and S9: the cycle at q# (refuting claim C, with the exact engine pattern) and the machine
  count K_band = 1 + floor(log2(theta(q) / ln q')). Facts.

## 8. Verdict

Claim A holds exactly (0 exceptions at 23 machine instances and 16 bands, one of them empty) and
is the known cap;
claim B holds with the exact count 1 + floor(log2(theta(q) / ln q')) (2, 2, 2, 3, 3, 3 banded
machines at q = 7 .. 23 plus one on top); claim C is refuted (the cycle at q# is open under the
engine only for q <= 7 and under no machine k >= 2 at any q); claim D is refuted in all three
parts and replaced by laws S4, S5, S6. The single-gear law S3 is known and verified (three
distinct gears per closed cycle of every machine k >= 2, minimum 3 in every sample; the engine
closes with two at the gear-7 double). The machines of the construction are equal-weight
objects (S5), their bands are not period samples (S4), their connection to the engine is the
rough-to-prime law (S6), and consecutive machines are tied by S7: the primes of one band are the
gears of the next machine, and the twins of one band are the paired gears of the next. Nothing
in the branch is a route to the root: every exact law here is a description of what the machines
are, and the one statement that would be a route - that band k always holds a twin, i.e. that
the joint open count is positive on every band - is twin-Bertrand between consecutive squares of
the chain (ROOT).

For the tree: node S (the stack by squares) FACT for S1, S3, S7, S8, S9; STRONG as a description
for S4, S5, S6 (tested, mechanism visible, no route); claim C DEAD; claim D DEAD as an
expectation, replaced by S4-S6.

## 9. Dead ends (with the refuting instance)

- D1. "The cycle at q# is open for every machine": DEAD - under the engine q# + 11 = 11 x 211 at
  q = 11 (mixed); under machine 3 closed at every q from 7 to 23 (T7).
- D2. "The open cycles of a machine on its band follow prod(1 - 6/g)": DEAD - 25 against 13.0
  (machine 2, q = 13), 54 against 200.8 (machine 3, q = 17); the band's open numbers are primes
  and smooth-times-prime numbers (S4), not a residue sample.
- D3. "Independence ratio 1": DEAD - 4.95 in the first bin of band 3 at q = 23, 0.83 at the end
  of band 2; it is P(prime | rough) / P(open under k) (S6).
- D4. "The band profile is the same function of u for every machine": DEAD - own/CRT per slot
  1.20 at u = 2-2.25 for machine 2 and 0.57 for machine 3 at q = 23; the prime term is
  (30/8)/(u ln g_k) and g_k differs by a factor 30 between the two.
- D5. "Closed positions of machine k + 1 relate to those of k through the squares": DEAD - 90 of
  174 mapped cycles land on closed cycles against a closed density 0.49 (q = 23).
- D6. Pre-registration misses for the record: P3's "never closed" (7 of 15 closed); P5's numbers
  (0.35 -> 0.9, and "below CRT at the start" for machine 2); P6's interval at the band starts;
  P8's monotonicity in q for machine 3 (15, 30, 25).

## 10. The part's remaining open items

Closed here: claims A, B, C, D; the composition of a machine's open set on its band (S4); the
weight of every machine (S5); the joint law (S6); the cross-machine identity (S7); the cycle at
q# (S8); the count (S9).

Measurement with no structural content: the longest closed runs and their positions (4, 5, 8, 8,
9 for machine 2; 15, 30, 25 for machine 3; twin-free runs 30, 50, 100 cycles on band 3), the
gear-count histograms of closed cycles, the per-slot strike counts (equal across the three slot
types to within 3 % on band 2 and 0.05 % on band 3, as the class-free residue set predicts).

Root question in disguise: "every band holds a twin" (joint open > 0 on every band k) - it is
twin-Bertrand between g_k^2 and nextprime(g_k^2)^2; and any bound on the longest twin-free run
of cycles on a band.

Genuinely open on the part alone: (i) the last open cycle of a machine's square part (the last
cycle of six g-smooth numbers: j = 4, 44, 44, 44, 4986 for g = 173, 293, 367, 541, 853) - finite
for every g by Stormer's theorem, with no rule visible in five values; an attack would tabulate it
for every prime g to 10^4 and compare with the largest pair of consecutive g-smooth numbers,
which is on record for the pair (Stormer, Lehmer). (ii) A model for the smooth part of S4 among
numbers coprime to 30 at small g (measured 0.12 -> 0.03 on band 3), where Dickman's rho does not
apply; the attack is the saddle-point density with the local factors of 2, 3, 5 removed.
