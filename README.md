# Primes: gaps, next primes, and the gear machine

A repository about prime gaps and prime structure. It holds several lines of work:

- **Next prime by blocked gaps** (the origin, kept at the bottom of this file): an algorithm that
  finds the next prime after any n from the residues of the small primes, with a written proof;
  notebooks in notebooks/ (test11.ipynb is the working version), a Python package in src/, Rust
  ports in rust/, rust2/, rust3/, and a prime-gaps worksheet. Its proofs were corrected in place on
  5 September 2026 (divisor bound and search range).
- **Mersenne primes**: main.py and notebooks/mersenne_analysis.ipynb, a toolkit for exploring
  Mersenne numbers, with data under data/mersenne/.
- **Evolutionary search** (ga/ .. ga5/): genetic-algorithm runs (evolve.py, hyper.py, seeds and
  hyperparameter logs) evolving expressions and parameters against the prime data.
- **Maximal gaps in the twin sieve**: exact Jacobsthal-type ladders for the two-class sieve (the
  project's F ladder to 59 by scan, lower bounds to 71 by SAT, the j_2 and h_2 campaigns), with the
  literature adjacency in docs/novel/ and docs/proof-search/harvester.md.
- **The gear machine and the twin prime conjecture** (since August 2026, the main current line):
  the primes modelled as gears, the conjecture reduced to one statement about openings inside a
  window, a Lean kernel corpus, and a multi-lane proof search. The rest of this section is about it.

## The machine, in one paragraph

Column k stands for the pair (6k-1, 6k+1), which is what survives the primes 2 and 3. Every prime g
from 5 up is a gear: it turns once every g columns and its one tooth strikes the column holding a
multiple of g, which through the columns lands on two residues, k = +-6^-1 (mod g). A column no
gear strikes is an opening. The window of the machine {5..y} is the certified range, the columns
whose numbers lie below the square of the next prime; an opening inside the window is a twin prime
pair, and that equivalence is checked by the Lean kernel (proofs/BlockedSlots.lean,
`twins_infinite_iff_survivor_in_window`). So the conjecture is: openings always land inside the
window, for every machine. The record F(M) is the longest stretch of consecutive columns with no
opening; the target inequality is that the record stays below the window's growth, and its per-step
form, F(M+q') <= F(M) + q' (the budget inequality), is what the search has been certifying rung by
rung. It is a target, measured true at every computable step, never a law.

## Why gears, and not just modular arithmetic

Everything here could be written in the standard language: residues, congruences, covering
systems, the Jacobsthal function. The gear picture is deliberately kept instead, for three reasons.

**It puts the mechanics in view.** The primes are not a static list; they are generated. Each prime,
once found, starts turning: from then on it strikes every g-th column forever, and nothing else
about it ever changes. A gear is exactly that: one fixed tooth, one fixed period, no memory, no
choice. What the machine does at any column is the sum of what its gears are doing there, and a
new prime is simply a column that every gear happened to miss. Thinking in gears keeps the question
"how does the next prime get made" in front of "what congruence classes are excluded", which is
the same fact looked at from the answer's side.

**It separates what turns from what aligns.** In the modular language a residue system is a set of
conditions and CRT says every combination of them occurs somewhere. In the gear language the same
statement is: the gears turn at fixed rates, so every relative position of their teeth comes round
once per period. That framing makes the two halves of the problem visible as different things.
The turning is fixed and fully understood (how many openings there are, where each gear strikes,
what merges when a gear is added: all of that is proved). The alignment is the open half: where,
inside the window, the teeth of all the gears line up to leave a column open. The whole search is
about the second half, and the picture names it directly.

**It resists a habit of the formal language.** Modular arithmetic invites density arguments, and
density arguments always end in the same logarithmic estimates that say nothing about where a
particular opening lands. Gears invite the opposite question: what is this tooth doing at this
column, next to that tooth. The project's useful findings came from that question (the chain law,
the record law, the merge grammar, the mirror, the caps at gears 5 and 7), and its dead ends were
the density-shaped ones. The vocabulary is a discipline as much as an analogy.

The translation is one line and is used whenever a formal statement is needed: gear g strikes
column k if and only if k = +-6^-1 (mod g), i.e. g divides 6k-1 or 6k+1; the machine {5..y} is the
sieve by all primes up to y on the two linear forms 6k +- 1; its record is the two-class Jacobsthal
function of that sieve. Every proof in docs/proofs/ and every kernel theorem in proofs/ is stated in
that translation; the gears are how the statements were found.

## Where things stand (5 September 2026)

- The route is kernel-checked and loses nothing: twins infinite if and only if the window always
  holds an opening.
- Eleven rungs of the budget inequality are certified exactly (machines up to 59, records
  5, 7, 11, 18, 25, 34, 43, 58, 88, 91, 103, 118, 145, 161 at q = 7..59), the last two by tools that
  never see the quantity that broke the cheap certificate; the 31 -> 37 case-split proof is in the
  kernel (385 case modules and a tiered root).
- The exact machinery is complete: the record of the bigger machine is computed from the smaller one
  three ways that agree (record law, attainment identity, exact record algorithm), all structural,
  and the grammar of what can merge (hit law, chain law, alternation) is kernel-checked for every
  gear at once. The bare half of the alignment depth is capped at 5 forever (kernel).
- What remains is one size statement, and it has been traced to its barriers: the per-step form
  contains a twin-Bertrand postulate at column 0; the direct window form sits below the
  dimension-2 sieve limit; every class-count-only method is provably insufficient; the real teeth
  must enter, and no instrument on record turns them into a size bound. See
  research/proof/theory_tree.md for the live branch tree.

## Where mathematics stands on small prime gaps

The twin prime conjecture is the statement that the gap 2 between consecutive primes occurs
infinitely often: the smallest gap that keeps recurring all the way to infinity is 2. What is
proved, in order:

| year | result | smallest gap known to recur infinitely often |
|---|---|---|
| 1919 | Brun: the sum of reciprocals of the twin primes converges (twins are sparse; finitely or infinitely many is left open) | - |
| 1973 | Chen: infinitely many primes p with p + 2 prime or a product of two primes | - |
| 2005 | Goldston, Pintz, Yildirim: gaps arbitrarily small relative to the average gap log p (liminf of gap / log p is 0) | - |
| 2013 | Zhang: some gap below 70,000,000 recurs infinitely often | 70,000,000 |
| 2013 | Polymath 8a: the same with Zhang's method sharpened | 4,680 |
| 2013 | Maynard (and Tao): a simpler sieve, also giving m primes in bounded intervals | 600 |
| 2014 | Polymath 8b: Maynard's method optimised | 246 |
| 2014 | conditional on the Elliott-Halberstam conjecture: 12; on its generalised form: 6 | 12 / 6 |
| 2026 | two claims below 246 (240; 186) are on record as unverified and not peer reviewed at the time of writing | 246 stands |

So the established unconditional bound is 246: infinitely many pairs of primes differ by at most
246. Nobody has proved any gap below that recurs infinitely often, and no method known reaches 2,
or even 6 unconditionally: the sieve methods behind every row from 2005 on run into the parity
barrier (a sieve cannot distinguish numbers with an even number of prime factors from those with an
odd number), which is why the conditional bound stalls at 6 rather than 2. The largest known twin
pair has 388,342 digits (2016); there are 808,675,888,577,436 twin pairs below 10^18.

Where this project sits. The table is a benchmark, not a direction: the project does not follow
the Maynard-Tao line and does not chase the number 246. Its route is a covering bound:
show that the two-class sieve never blocks a whole window, which by the kernel-checked equivalence
gives infinitely many twins directly, with gap exactly 2 and no intermediate bound. The same parity
barrier appears on this side as the dimension-2 sieve limit (docs/novel/j2-upper-bound.md, and
research/proof/iwaniec_two_class.md): any class-count-only bound sharp enough to close the window
would itself be the conjecture, so a proof has to use the specific residues the primes strike, not
just how many. Eleven steps of the ladder are certified exactly, the record is a quarter of the
window at every computed machine, and the uniform statement is open.

## Map of the repository

- docs/proofs/: the proved theorems as written proofs, one file each, with a plain-words opening,
  the classical translation, the kernel theorem names where the Lean corpus has them, and an
  honest line on what each contributes to the conjecture; docs/proofs/README.md is the index.
- docs/proof-search/human.md: the state of the hunt in plain language, rewritten every round.
- docs/proof-search/alignment-rules.md (+ alignment-rules-index.md): every alignment rule on record
  (300 merged entries), each with its status (kernel / exact / measured / conjectured) and pointer;
  refuted claims and named gaps. Read this before proposing anything.
- research/proof/theory_tree.md: the branch tree of the proof hunt (theory, test, verdict, log).
- docs/proof-search/agents-shared.md: the lanes' findings exchange, round by round, with the
  manager's SUMMARY at the top and the standing rules.
- docs/proof-search/{constructor,mechanic,lateral,formalist,harvester,lp-duality,anchor-235}.md:
  the lane documents (cumulative). docs/proof-search/archive/: rounds 1-19 verbatim.
- docs/novel/: one document per finding that may be new to mathematics, with prior-art status.
- proofs/: the Lean 4 corpus (729 files; `lake build` from that directory; the axiom audit is
  `lake env lean AxiomCheck.lean` and must report no sorryAx).
- research/: the scripts (486) and their persisted results; research/data/rNN/ holds each round's
  emissions and logs (logs and bulk certificates are gitignored).
- notebooks/, src/, rust*/, ga*/: the origin algorithm, its ports and experiments.

## Running things

Python runs through uv (`uv run python research/<script>.py`); the SAT instrument uses the
`.venv-sat` environment. Lean: `cd proofs; lake build` (never `lake -d` from the root; a root module
that imports hundreds of case modules must be tiered, see proofs/lakefile.toml). The box is 20 cores
and 16 GB; commit charge, not core count, is the binding limit.

## Glossary

The project's terms, in the order you meet them. The classical translation is given where there
is one; the status of a term's main fact is marked kernel (checked by the Lean kernel), exact
(computed on full periods or by an exact certificate), measured, or open.

- **Column k.** The pair of numbers (6k-1, 6k+1). Every prime above 3 lives in some column, so the
  columns are what survives the primes 2 and 3. A twin prime pair is a column whose two numbers
  are both prime.
- **Gear g.** A prime g >= 5, turning once every g columns.
- **Tooth.** The single point of gear g that lands on a multiple of g. Seen through the columns it
  strikes two residues, k = +-6^-1 (mod g); their distance apart is d_g = 3^-1 (mod g), the
  "tooth spacing", the same rational one third for every gear (kernel).
- **Real teeth.** The two residues the prime g actually strikes, u_g = round(g/6) and g - u_g. A
  counterfactual gear keeps g and moves the two residues elsewhere.
- **Strike / blocked.** Gear g strikes column k when g divides 6k-1 or 6k+1. A struck column is
  blocked.
- **Opening.** A column no gear of the machine strikes: both numbers escape every prime in the
  machine. Openings are 3 of 5 columns for gear 5 alone, and prod(g-2) per period overall.
- **Machine {5..y}.** The gears 5 up to the prime y, together. Its period is the product of its
  gears; its opening pattern repeats with that period.
- **Anchor.** The three primes 2, 3, 5 as one object: 2 and 3 are built into the columns, 5 is the
  first gear.
- **Window.** The certified range of the machine {5..y}: the columns whose numbers lie below the
  square of the next prime. Inside it the machine's openings are exactly the twin prime pairs
  (kernel: BlockedSlots.twins_infinite_iff_survivor_in_window).
- **Section.** The new part of the window when the machine grows from p to q: the columns with
  p^2 < 6k+1 < q^2. Inside a section the previous machine is exact and the new gear is silent
  (kernel: the layer law).
- **Stretch.** Any run of consecutive columns anywhere in the period. (Older documents call this
  a "window"; the picture document translates.)
- **Gap.** The distance between two consecutive openings; a gap of w means w-1 blocked columns.
- **Record F(M).** The widest gap of the machine over its whole period: the longest empty stretch.
  Corpus values 5, 7, 11, 18, 25, 34, 43, 58, 88, 91, 103, 118, 145, 161 at y = 7..59 (exact). The
  blocked-count convention is F - 1.
- **Spectrum F_J(M).** The widest stretch containing exactly J-1 openings, i.e. the largest sum of
  J consecutive gaps; F_1 = F, F_2 the one-hole record.
- **Budget, budget inequality, target.** At the step from machine M to M + q', the budget is
  F(M) + q'. The budget inequality F(M+q') <= F(M) + q' is the project's target: measured true at
  every computed step (eleven certified rungs), never a law. Summed along the ladder it keeps the
  record below the window's growth, which forces an opening in every window and hence infinitely
  many twins.
- **Rung.** One step of the ladder, certified when the budget inequality is proved exactly at that
  step (by scan, by LP certificates, or by the kernel).
- **Copies / phases.** Adding gear q' lays q' copies of the old pattern end to end, one per phase
  of the new gear; each old opening dies in exactly two copies (kernel).
- **Merge.** Every gap of the bigger machine is an old gap or a merge of consecutive old gaps
  whose interior openings the new gear strikes at one phase (kernel: the merge law).
- **Chain / hit / hop.** A run of consecutive old openings all struck by the new gear is a chain
  (or kill chain); each struck opening is a hit; the walk to the next opening hops over it. Two
  consecutive openings can both be hit only if their gap is 0 or +-d (mod q') (kernel: the chain
  law).
- **Letters a, b; padded letter.** The two bare letters are a = 2 round(q'/6) and b = q' - a
  (a + b = q', 3a = q' -+ 1); a gap of value q' (or a multiple) is a padded letter. A gap is legal
  for q' when it is 0 or +-a (mod q').
- **Legal word.** Consecutive gaps that are all legal letters with the nonzero classes strictly
  alternating (T3). A realised legal word of m letters is exactly a chain of m+1 hits.
- **L(M), A_kill, J_max.** L is the longest realised legal word; A_kill = L + 1 is the deepest chain
  the new gear can make; J_max = L + 2 (kernel: the word reduction). L is 1, 1, 1, 2, 1, 3, 3, 2, 2,
  2, 4, 3 at machines 11 to 53 (exact). L <= 2F(M+q')/q' + 1 (proved); its bare part is at most 5
  forever (kernel); its padded part is open.
- **Q*_J, word-legal J-run, attainment.** Q*_J is the widest stretch of J consecutive gaps whose
  J-2 middles form a legal word. The attainment identity max(F_2, max_J Q*_J) = F(M+q') computes the
  next record from the old machine (exact, structural).
- **Record law / phase reduction.** F(M+q') is the widest run of old openings lying in one two-class
  set {r, r+d} mod q', plus the gap before and after, maximised over phases, on one old period
  (kernel at 17, exact to 41).
- **Corridor.** What gears 5 and 7 forbid forever: the 15 open residues mod 35 that every opening
  must occupy (kernel). Corridor arithmetic fixes where configurations sit, never how big they are
  (escape distance 1).
- **Literal cap, bare-word cap, PSORD.** Chains built from the two bare letters alone have at most
  6 members forever (kernel); the bare part of L is at most PSORD(q' mod 210) <= 5, with 28 of the
  48 classes at 2 (kernel).
- **Mirror.** The symmetry k -> -k of the opening set; column 0 and the antipode are always open;
  records come in mirror pairs; the symmetry group is exactly Z/2 (kernel).
- **d_0.** The first opening after column 0, which on the real machine is the column of the first
  twin prime pair above p. F_2 >= 2 d_0 always, so the budget inequality at column 0 asks the next
  twin pair to sit within half the budget (a twin-Bertrand statement, open).
- **Counterfactual family.** All machines with the same gears and the real gears' symmetric
  two-tooth shape but the teeth moved. What holds on every member is structural; what fails on
  some member needs the real teeth. The budget inequality fails on up to 0.6% of members, the
  chain statement on up to 0.1%, and L reaches 5 where the real machine has 2.
- **Pair statement, chain statement.** The two halves of the budget inequality: F_2(M) <= F(M) + q'
  (one hole) and Q*_J(M) <= F(M) + q' for J >= 3 (chains). Both open uniformly; both hold at every
  computed step.
- **Walk.** From any column, step forward through blocked columns to the next opening; the layered
  walk builds the bigger machine's walk from the smaller one's plus the new gear's hits (kernel:
  the nested next-opening formula). From column 0 its length is d_0; from q^2 it lands on a twin
  within 2 to 79 columns at every prime to 100,003 (exact).
- **Repulsion.** Big blocked stretches have short neighbours; the one-hole record sits below what
  independent gaps would give (measured; the suppression law and the renewal ladder are the
  measured and the rigorous sides of it).
- **Lane, round, kernel audit.** The proof search runs in rounds of parallel AI lanes (Constructor,
  Mechanic, Lateral, Formalist, Harvester, LP-duality); every claim carries its gate command and
  output line; `cd proofs; lake env lean AxiomCheck.lean` must report no sorryAx.

---

# Origin: Prime Number Experiments

The project began as a manual workbook: spreadsheets and hand-written code, learning basic number
theory from no mathematical background while going. AI assistants entered over the past year, and
the proof-search programme above grew out of that. The write-up below is the original, with its
proofs corrected on 5 September 2026 (the trial-divisor bound and the gap-search range were wrong
as first written; the corrected algorithm was checked against every prime below 200,000).

## Current Algorithm

The main work is in **[test11.ipynb](notebooks/test11.ipynb)** - an algorithm that finds the next
prime after any given number using modular arithmetic patterns.

### How It Works

The algorithm uses modular arithmetic to calculate the next prime after any given number by finding
"blocked slots" where primes cannot exist.

**Core Algorithm (for finding the next prime after a known prime):**
1. For a given prime p, take as trial divisors all primes q with q^2 <= p + g for the candidate gap
   g under test (in practice: all primes up to sqrt(p + G) for the largest gap G you are prepared
   to search; for p below 200,000 no next-prime gap exceeds 86)
2. Calculate `-p % q` for each trial divisor - this gives the distance to the next multiple of q
3. These distances represent "blocked slots" where the next prime cannot be located
4. Cycle each trial divisor through multiple iterations to find all blocked slots up to G
5. Search for the first even number (starting from 2) that is not in the blocked set
6. That unblocked position is the gap to the next prime: `next_prime = p + gap`

**Example with p = 97:**
- Trial divisors: [2, 3, 5, 7] (the primes with q^2 <= 97 + g for every gap g up to 24)
- `-97 % 2 = 1` -> blocks gaps {1, 3, 5, 7, 9, ...}
- `-97 % 3 = 2` -> blocks gaps {2, 5, 8, ...}
- `-97 % 5 = 3` -> blocks gaps {3, 8, ...}
- `-97 % 7 = 1` -> blocks gaps {1, 8, ...}
- Combined blocked gaps: {1, 2, 3, 5, 7, 8, 9, ...}
- First unblocked even gap: 4 -> `97 + 4 = 101`

**Why the divisor bound must grow with the candidate.** With divisors below sqrt(p) only, p + g can
be the square of, or a product of, two primes just above sqrt(p) and escape every block: that rule
reports 9 after 7, 25 after 23, 49 after 47, 121 after 113, 169 after 167, 289 after 283 (85 wrong
answers below 200,000). And the search cannot be capped at sqrt(p): the next prime after 13 is 17
(gap 4 > 3.6) and after 31 is 37 (gap 6 > 5.6). "The next gap is at most sqrt(p)" is an unproved
conjecture of Andrica type, so the search continues until an unblocked gap appears.

**Generalised Version (for any number n):**
The same process from any starting number n: know all primes q with q^2 <= n + g, then search from
the appropriate starting position (1 if n is even, 2 if n is odd), stepping by 2.

**Example with n = 100:**
- Trial divisors: [2, 3, 5, 7] (the primes with q^2 <= 100 + g for the gaps tested)
- `-100 % 2 = 0` -> blocks gaps {0, 2, 4, 6, 8, ...}
- `-100 % 3 = 2` -> blocks gaps {2, 5, 8, ...}
- `-100 % 5 = 0` -> blocks gaps {0, 5, ...}
- `-100 % 7 = 5` -> blocks gaps {5, ...}
- Combined blocked gaps: {0, 2, 4, 5, 6, 8, ...}
- First unblocked gap starting from 1: 1 -> `100 + 1 = 101`

**Mathematical Foundation:**
Any composite number m has a prime factor at most sqrt(m). Using all primes q with q^2 <= m as
trial divisors and cycling their modular patterns identifies every position where a composite must
occur. The first candidate that no pattern blocks is prime.

## Proof: Deriving next_prime(p) from p

**Given:** A prime p >= 5
**Goal:** Prove the algorithm correctly finds next_prime(p)

**Theorem:** Let p >= 5 be prime. For a gap g >= 1 let G(g) be the set of gaps h <= g with
h = -p (mod q) for some prime q with q^2 <= p + g. Then the smallest even g with g not in G(g)
satisfies p + g = next_prime(p).

**Proof:**

**Step 1 - Construction of G(g):** For each prime q with q^2 <= p + g, compute r = -p mod q and add
the values {r, r+q, r+2q, ...} up to g. So G(g) is exactly the set of gaps h <= g with h = -p (mod q)
for some such q.

**Step 2 - Gaps in G(g) give composites:** If h is in G(g) then h = -p (mod q) for some prime q with
q^2 <= p + g, so q divides p + h. Since q <= sqrt(p + g) < p + h (as p >= 5 and h >= 1 give
(p + h)^2 > p + g whenever g < p^2 + 2p, which covers every gap the search meets), q is a proper
divisor, and p + h is composite.

**Step 3 - Completeness of G(g):** Suppose p + h is composite for some h <= g. Then p + h has a
prime factor q with q^2 <= p + h <= p + g, so q is a trial divisor, and q dividing p + h means
h = -p (mod q). Hence h is in G(g). (This is the step that fails if the divisors stop below
sqrt(p): a composite p + h can be a product of two primes between sqrt(p) and sqrt(p + h).)

**Step 4 - A gap outside G(g) yields a prime:** By Step 3's contrapositive, if g is not in G(g) then
p + g is not composite, so p + g is prime.

**Step 5 - It is next_prime(p):** Every even h < g lies in G(h), hence in G(g) because the divisor
set for g contains the divisor set for h; so by Step 2 every p + h with even h < g is composite,
and odd h give even numbers, composite for p >= 3. Therefore no prime lies strictly between p and
p + g, and p + g = next_prime(p).

**Conclusion:** The algorithm identifies next_prime(p) by pre-computing the gaps that must yield
composites and taking the first gap that avoids them, provided the trial divisors grow with the
candidate. It is the sieve of Eratosthenes run forward from p.

## Proof: Generalised Version for Any Number n

**Given:** Any integer n >= 2
**Goal:** Prove the generalised algorithm correctly finds next_prime(n)

**Theorem:** Let n >= 2 and, for a gap g >= 1, let G(g) be the set of gaps h <= g with
h = -n (mod q) for some prime q with q^2 <= n + g. Starting from g = 1 if n is even and g = 2 if n
is odd, and stepping by 2, the first g not in G(g) satisfies n + g = next_prime(n).

**Proof:**

**Step 1 - Construction of G(g):** As above, with n in place of p.

**Step 2 - Gaps in G(g) give composites:** If h is in G(g), some prime q with q^2 <= n + g divides
n + h. If q were n + h itself, then (n + h)^2 <= n + g <= n + h + (g - h), forcing n + h <= 1 for
the gaps the search meets (g < n^2 + 2n); impossible for n >= 2. So q is a proper divisor and n + h
is composite.

**Step 3 - Completeness:** A composite n + h with h <= g has a prime factor q with
q^2 <= n + h <= n + g, so q is a trial divisor and h = -n (mod q), hence h is in G(g).

**Step 4 - Parity:** All primes above 2 are odd, so candidates n + h of the wrong parity are
skipped by the starting position and the step of 2; for n >= 2 this loses nothing.

**Step 5 - The first unblocked gap is the next prime:** By Step 3 the first g outside G(g) gives a
prime n + g, and by Step 2 every earlier candidate of the right parity is composite, so n + g is
next_prime(n).

**Conclusion:** The generalised algorithm works by the same argument, with the blocked set G(g)
capturing every composite-yielding gap as long as the divisor bound grows with the candidate.

## Observations: how the blocked stretches form

Everything below is on record with its status (kernel-checked, exact on full periods, or measured);
the pointers are docs/proof-search/alignment-rules.md (section numbers in brackets) and
research/proof/theory_tree.md.

**Formation.** Adding a gear q' to a machine is laying q' copies of the old pattern end to end, one
per phase, and every old opening dies in exactly two of the copies (kernel, [2.1]). So every new
blocked stretch is a merge of consecutive old stretches, and the record grows only by merging
(kernel, [2.3]). Which old stretches can merge is decided by residues alone: the openings between
them must sit on the new gear's two teeth, which means their gaps are 0 or +-a mod q' with the
nonzero classes alternating, the "legal word" grammar (kernel, [2.2]-[2.3]). The record of the
bigger machine is therefore computable from the smaller one, three ways that agree: the widest run
of old openings on two residues plus its flanks, maximised over the q' phases on one old period
(the record law, [3.1]); the attainment identity max(F_2, max_J Q*_J) = F(M+q') ([3.2]); and the
exact record algorithm over legal words ([3.3]). Exact at every computed step to 59.

**Made at the top.** A record stretch is a near-perfect tiling: the small gears behave in it as in a
random stretch of the same length, the top three or four gears each remove the two or three
openings a random stretch would keep, and the top gear alone accounts for one to three columns,
every one an old opening (exact, full periods to 23; [3.11], research/proof/manager_notes.md).

**Self-similarity.** The record of one machine is built from a near-record piece of the machine
below, which was built the same way one level further down: the ancestor is a runner-up of the
lower machine (not its record) in 7 of 8 steps, and the chain goes one to five generations deep
([3.11], "records recruit runner-ups"). The nested next-opening formula computes the enlarged
machine's walk from the lower machine's walk plus the new gear's hits, layer by layer (kernel,
[3.1]); the walk from q^2 lands on a twin pair within 2 to 79 columns at every prime to 100,003
([4.7]). Inside the section the machine below is exact and the new gear is silent: its whole new
workload is its square and its products with the primes up to the horizon, one to three columns
(kernel, [2.9]).

**Mirroring.** The opening set is symmetric under k -> -k; column 0 and the antipode (P +- 1)/2 are
always open (kernel, [1.4], [3.12]); the symmetry group is exactly Z/2, so the mirror is worth one
factor of two and never more ([3.12]). Record stretches come in mirror pairs at k and P - k - span,
and no self-mirror stretch is ever word-legal at depth three or more, so every span count is even
([3.12]). The mirror is also an exact symmetry of the LP certificates.

**Repulsion.** Big blocked stretches repel: the gap after a long gap is shorter than an ordinary gap
(after the record: 2.0, 3.0, 3.7, 2.6, 3.0 columns at machines 11 to 23 against ordinary gaps 2.9
to 4.7), the one-hole record sits far below what independent gaps would give, and every three-gap
run with a big middle stays within the budget while the three-gap record always has a tiny middle
between two big flanks (measured, both the two-class machine and the ordinary Jacobsthal sieve;
theory_tree.md branch 5). No mechanism is proved.

**What needs the real teeth.** On the family of machines with the same gears and moved teeth, the
formation rules above all hold (the identity, the record law, the mirror), while the sizes do not:
the budget inequality fails at up to 0.6% of members, the increment law at 13-22%, and the alignment
depth reaches 5 where the real machine has 2 ([5]). The real machine is a low-record outlier among
its own counterfactuals (11th to 26th percentile), and its teeth sit exactly on the residue class
where the relaxed machine's survivors are densest ([5.4]-[5.5]). Coherence of the tooth spacings
(the real spacing is one third of every gear) does not explain the outlier (theory_tree.md branch 6).

## Areas for further investigation

- **The one size statement.** The record stays below the window's growth, F(y) < y^2/6, for every
  y. Every formation rule is proved; this is not. The per-step form contains a twin-Bertrand
  postulate at column 0 (the window's first opening within half the budget), the direct window form
  sits below the dimension-2 sieve limit, and every class-count-only method is provably
  insufficient; the real teeth must enter and no instrument on record turns them into a size bound
  (theory_tree.md, branches 1-3).
- **The padded half of the alignment depth.** The bare half is capped at 5 forever (kernel); the
  padded half grows 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3 across the corpus and nothing bounds it. The
  smallest unproved statements are flank statements: the two gaps around an occurrence of the
  letter a sum to at most F + b, around the padded letter q' to at most F, around the pair (a, b)
  to at most F (research/proof/chain_from_teeth.md).
- **The mechanism of repulsion.** Why the neighbours of a big stretch are small, as a residue
  statement about the gears' phases at an opening (the left tiling is the negated right tiling,
  gear by gear); the negative adjacency correlation is structural on 95% of the family and unproved.
- **Record genealogy as a theory.** If the assembly of records from runner-ups has bounded
  branching, the growth per step is bounded by the pieces' growth. Exact at 8 steps, untested as a
  theory.
- **The renewal factor.** A closed-form lower bound on the chance of no opening strictly between
  two exposed endpoints; named as the entire remaining gap between the rigorous exposure bound and
  sufficiency, never built ([6.5]).
- **The walk as the object.** Any proof must locate the next opening after a point at every scale;
  the layered walk from column 0 and from q^2 is that object, measured short and never bounded.
- **The signal-processing and non-iterative ideas of the origin** (stacking sin(remainder/n), CRT
  without iteration) were pursued: the walk's Fourier transform is the pole-phase law and carries no
  information of its own beyond the opening set's, and the L1 character mass is identical across all
  counterfactual teeth (docs/novel/walk-transform-pole-identity.md); the CRT form of the record is the
  realisability CSP with no period ([2.8]).
