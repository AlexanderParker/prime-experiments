# The nth prime as a closed form, from what the machine has established (2026-09-11)

Companion to `next_gap_closed_form.md`. Assembled from results on record; each piece carries its
status. "Closed form" here has the same meaning as there: an explicit finite expression in the
machine's objects that evaluates to p_n with a certificate. No formula in n alone is on the
record, and section 5 says exactly what one would require. Script:
`research/stack/r7/nth_prime.py` (both checks below reproduce from it).

## 1. The objects, by construction

- **The stack by squares from base 2.** Cut c_1 = 2. Given c_k, let p_k be the first prime at
  or above c_k and c_{k+1} = p_k^2. Section k is the integers in [c_k, c_{k+1}); machine k is
  the set of primes at most p_k (its gears). The first five cuts are 2, 4, 25, 841, 727609,
  529420677769.
- **The square-root rule** (proof skeleton step 3; kernel `OneStepE.blocked_iff_sqrt`,
  `CoreLeftover.twin_of_rough`): a member of section k greater than 1 with no prime factor
  among the gears of machine k is prime, because its least prime factor would otherwise exceed
  p_k and its square would exceed c_{k+1}. Conversely every prime in section k either is a gear
  of machine k (only possible in the two lowest sections, where p_k lies inside its own section)
  or has no gear as a factor. So on section k the primes are exactly the open members of
  machine k, together with the gears that sit in the section.
- **The count of a section**, n_k = the number of open members of machine k in section k (plus
  its resident gears). N_k = n_1 + ... + n_{k-1} is the number of primes below c_k.
- **The walk of machine k** is the sequence of its open members in increasing order. The next
  open member after y is y + mex of the union over gears g of the progression
  { ((-y) mod g) + j g : j >= 0 } truncated at a bound B, certified exact when the mex is below
  B (`next_gap_closed_form.md` section 2, single-tooth version; the free-regime half is the
  kernel's `mex_form`).

## 2. The assembled function

    p_n  =  c_k  +  W_k( n - N_k ),      k the unique index with N_k < n <= N_{k+1},

where W_k(j) is the offset of the j-th open member of machine k in section k, produced by the
walk started at c_k (j steps of the certified mex). Every quantity is finite and explicit: the
cuts by squaring, the machines as the primes below a cut (themselves p_1 .. p_{pi(p_k)}, all
already produced), the counts n_k by the count of section 3, and the position by the walk.

**Checked** against sympy's `prime(n)` at 15 of 15 values of n across the first four sections
(n = 1 .. 58175):

| section k | [c_k, c_{k+1}) | top gear p_k | N_k | n_k (primes in the section) |
|---|---|---|---|---|
| 1 | [2, 4) | 2 | 0 | 2 |
| 2 | [4, 25) | 5 | 2 | 7 |
| 3 | [25, 841) | 29 | 9 | 137 |
| 4 | [841, 727609) | 853 | 146 | 58,462 |
| 5 | [727609, 5.29 x 10^11) | 727,613 | 58,608 | (not walked; 5.3 x 10^11 members) |

with p_147 = 853 (the first member of section 4 is its own top gear), p_1000 = 7919,
p_10000 = 104729, p_58175 = 721859 all exact.

## 3. The count of a stretch, in closed form (the core / tail split)

For the count n_k, or for the number of primes on any stretch, the record gives one expression
that is not a walk. Take a stretch (x, x + 6L] and the core t = 6L + 1 (the gears at most t).
Whenever x + 6L < t^3:

    pi(x + 6L) - pi(x)  =  Omega_core(x, L)  -  S_tail(x, L),

    Omega_core(x, L)  =  #{ m in (x, x + 6L] : no prime <= t divides m }
                      =  sum over subsets D of the primes <= t of (-1)^|D| ( floor((x + 6L)/prod D) - floor(x/prod D) ),
    S_tail(x, L)      =  #{ m in (x, x + 6L] : m = P1 P2, P1 <= P2 primes > t }.

Why: a member of the stretch with no prime factor at most t and below t^3 is a prime or a product
of exactly two primes above t (kernel `CoreLeftover.primeOrSemiprime_of_rough_lt_cube`); the core's
open members are therefore the primes plus those products, and the products are the tail's
strikes on core-open members (each tail gear strikes at most one position in the stretch: that is
the core / tail split of the loaded record rule, kernel `TopMachineRecord.loaded_record_rule`,
read for a single tooth). Both terms are finite sums: Omega_core over the core's subsets (or by a
sieve of the stretch by the core), S_tail over the primes P1 in (t, sqrt(x + 6L)] and their
multiples in the stretch.

**Checked** at 12 of 12 random stretches, x from 9.6 x 10^6 to 8.5 x 10^8, L in {50, 100, 200,
400} (t = 301 .. 2401):

| x | L | t | pi(x+6L) - pi(x) | Omega_core | S_tail | Omega_core - S_tail |
|---|---|---|---|---|---|---|
| 153,794,500 | 100 | 601 | 29 | 54 | 25 | 29 |
| 508,069,464 | 400 | 2401 | 116 | 171 | 55 | 116 |
| 847,885,253 | 400 | 2401 | 122 | 182 | 60 | 122 |
| 31,437,866 | 400 | 2401 | 143 | 165 | 22 | 143 |
| 11,651,171 | 50 | 301 | 23 | 31 | 8 | 23 |

(the other seven rows are in the script's output; all agree).

**Prior art, stated plainly.** With a = pi(t) this is Meissel's formula (1870), pi(x) = phi(x, a)
+ a - 1 - P_2(x, a) for a >= pi(x^(1/3)), restricted to an interval: KNOWN VARIANT. What the
machine adds is only the reading: the core decides the open pattern, the tail contributes single
strikes, and every tail strike on a core-open member is a charge of two tail primes (the same
object as the twin leftover's P1 P2 members in `core_leftover.md`). It is recorded because it
is the one count on the record that is not a walk, and because its twin version
(`CoreLeftover.leftover_eq_card_twins`) is the step's object.

## 4. The folded form (the anchor as a coordinate)

Gears 2, 3, 5 leave eight residues in every cycle of 30 (1, 7, 11, 13, 17, 19, 23, 29). So for
n >= 4,

    p_n  =  30 m_n + r_n,   r_n in {1, 7, 11, 13, 17, 19, 23, 29},

and the walk of section 2 runs over the eight residues of one cycle, the walk of every later
machine over the eight residues of each cycle with the remaining gears 7 .. p_k as the mex's
progressions. Nothing changes in the certificate; the walk is shorter by the factor 30/8.

## 5. What it is, and what it is not

- It is exact and self-certifying: p_n is the position at which the machine's own count reaches
  n, and that position is produced by the machine's certified walk on the section that holds it.
  The information consumed is the residues of the current position modulo the gears below the
  cut, nothing else.
- It is not a formula in n alone. A formula in n would be a closed form for W_k(j), the j-th
  open position of a machine on a stretch, i.e. for the iterated mex over the gears' residues;
  no expression on the record compresses the iterate, and the same wall stands here as for the
  next gap (`next_gap_closed_form.md` section 4): a bound on any single step is the record of
  the machine {primes <= sqrt(p)} on a stretch, the object of the step.
- The known formulas for p_n in n alone (sums of floors over the counting function, of the
  Willans kind) encode the sieve count in a formula and are not on the record; they are not
  sought here (the project's rule: find new maths, do not translate known results).
- The count's mean is on the record as a statement about machines, not about n: each machine
  removes half the numbers it sees and a quarter of the slots that reach it (proof skeleton
  step 11, proved as a count via Mertens), which is why the count of section k grows like the
  section's length over the logarithm of its cut. That is the one line the record spends on the
  asymptotic; it is not re-derived here.

## 6. Verdict

FACT, not a route: the nth prime is the machine's walk indexed by the machine's count, with a
certificate at every step; the stretch count has a closed form with two finite sums under the
hypothesis x + 6L < (6L + 1)^3 (Meissel's formula read on the machine, KNOWN VARIANT); no
formula in n alone exists on the record, and the obstruction is the same object as the step.
