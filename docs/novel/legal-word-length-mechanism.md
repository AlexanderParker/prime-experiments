# The length of the longest legal word is a histogram statistic; its last unit is not

Round 30 (mechanic), 2026-09-03.  Status: MEASURED (exact censuses, exact CRT decisions,
one independent-letter model used as an instrument, no proof).

## 1. What it is

For a sieve machine M = {gears 5..M} acting on the slot line (slot k = the pair 6k-1,
6k+1; gear q blocks k iff k = +-6^{-1} mod q) and a further prime g, a gap value v is a
LEGAL LETTER for g if v mod g is 0, +d or -d with d = 2*6^{-1} mod g, and a word of
consecutive gaps of M is LEGAL if every letter is legal and the nonzero classes strictly
alternate (padded letters, class 0, are transparent).  L_g(M) is the length of the
longest legal word that occurs as consecutive gaps of M; it equals A_kill(M -> g) - 1
(the largest number of consecutive M-openings one phase of g deletes, minus one) and
J_max(M) - 2 (the depth beyond which the word-legal spectrum Q*_J is empty).  Its
growth is the one open quantity in this project's derivation of F(M+q') <= F(M) + q'.

The finding, in plain words.  Counting naively (N = prod(q-2) gaps, each in the three
classes with probability 3/g, independent) predicts longest runs of order ln N / ln(g/3)
- 8 at m29, 9 at m31, 10 at m37, 13-18 at m47 - where the truth is 3, 3, 2, 4.  Replacing
3/g by the REAL class densities of the legal alphabet in M's exact gap histogram (the
legal values are 3-6 specific gap sizes whose frequencies are far below 3/g) and adding
the alternation transfer matrix (growth rate p0 + sqrt(p+ p-)) brings the independent
model to 3.7, 4.0, 4.0 at m29, m31, m37 - within one unit of the truth at every scanned
machine.  So the LENGTH of the longest legal word is a density statistic of the gap
histogram.  But the NUMBER of legal windows tracks that same model only at the short
lengths and collapses at the top: for gear 31 on m29 the counts are 8.02e6 / 8.02e6,
13,000 / 15,100, 4 / 279 at lengths 1, 2, 3; for gear 37 on m31 (full lower period,
6.2e9 gaps) 1.148e8 / 1.15e8, 70,964 / 175,000, 216 / 1,610, 0 / 2.5; for gear 41 on m37
the model predicts 3,900 legal 3-windows and there are none, and already the 2-windows
are 27 against 10,500 on a 1% sweep (every realised 2-word carries the padded letter 41;
the pure alternation (14,27) is unrealised).  The last unit or two of L are decided by
an arithmetic collapse of occurrence counts (a factor 8-400 to infinity) that no density
sees.

The mechanism behind the collapse, measured on the same objects.  Every one-letter
extension of a longest legal word is refuted, and the refutation can be attributed to the
two halves of the realisability CSP (open constraints: the teeth of the open points;
cover constraints: every interior slot blocked).  Relaxing ALL open constraints - asking
only whether any slot of M blocks the punctured interior - the extensions are still
refuted at every machine 19..37 examined, except the pure alternations, which survive
the cover half and are killed by gears 5 and 7 jointly through their open constraints
(the corridor mod 35).  So the words that would lengthen L die because their blocked
pattern does not occur in M at all (an F_J-type statement about M's blocked runs), or,
for the alternation only, because of the corridor.

## 2. Why it might be novel

Longest-run statistics of residue classes along a sieved sequence, with the
alternation constraint that a two-tooth gear imposes, do not appear in the Jacobsthal /
covering literature the Harvester has surveyed (Ziller-Morack work in class-assignment
space and never look at runs of the lower machine's gaps).  The decomposition "length =
density, top count = arithmetic" is a statement about how a sieve's kill chains are
capped, which the project needs and which we have not seen stated.  The classical
shadow is the theory of longest success runs and of pattern occurrence counts in
Markov sequences; the content here is the measured departure from it at a specific
length, and the attribution of that departure to the cover half.

## 3. Proof / verification

MEASURED.  research/resrun_r30.py (exact cyclic scans at m11..m29, the machine-31 full
lower period streamed from the machine-29 memory map, a 12-of-1147-chunk deliberate
partial at m37; V2 = D_g - 1 asserted against anchor235/chain_depth.py at g = 7..29 and
against the recorded A_kill values at every next-prime cell by research/gate_mechanic_r30.py
section C); research/wordkill_r30.py (CRT decisions, cover-only verdicts re-derived by a
direct period scan at m19 and m23 in gate section B); models research/data/r30/
models_31_37.log from the exact cyclic histograms of research/data/r26/ghist_31.csv and
ghist_37.csv.  No proof of either half.

## 4. Implications

Inside the project: the target "(B) L(M) bounded" splits into a histogram statement (the
class densities of the legal alphabet - three to six gap sizes - fall with the machine
like the letter frequencies do) and a one-unit arithmetic collapse; a bound on L needs
the first and then the cover half for the last unit.  The word vehicle's free screens
(alphabet, spectrum caps, phase saturation at every gear) are exactly one length too
generous at 7 of 8 machines.  Outside: a small, concrete instance of how much of a
sieve's extremal structure a first-moment / independence model captures (the scale) and
where it fails (the top count), on an object where both sides are exact.

## 5. Unsolved questions it touches

Whether L(M) = A_kill(M -> q') - 1 is bounded (the project's crux); the size of the
counted padded-gap census occ(q'; M) that Constructor named; the deletion-ladder /
increment-law slack.  It reframes "bound L" as "bound the legal letters' class densities
and then close one unit by the cover half".

## 6. Prior-art check

Not yet checked (no web access in the lane).  Terms to search: longest run of a pattern
in a sieved sequence; Jacobsthal covering chains; Markov-chain longest-run / pattern
occurrence deviations for sieve gaps; "kill chain" / "co-deletable run" in Hagedorn,
Ziller, Ziller-Morack.
