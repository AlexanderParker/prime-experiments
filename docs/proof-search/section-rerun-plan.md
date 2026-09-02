# Section-view rerun plan for the historical checks

Manager, round 29 (2026-09-02). Human's direction: every pattern check is to be run on the NEW
section each machine adds (numbers in (p^2, q^2), slots p^2 < 6k+1 < q^2 for the rung p -> q),
never on the whole window - the old part was already checked by the smaller machine and is
noise - and the finding is to be the MECHANISM that surfaces the twins (which words, which gear
interactions), not counts or averages, which are logarithmic and always return the same verdict.
This document inventories the historical checks, says what each becomes in section view, and
fixes the order of work. Standing rule recorded in memory (section-view-mechanism).

## 0. The exact mechanism of a section (forced, no computation)

In the section (p^2, q^2) every blocked number is s * m with s its smallest prime factor, a gear
<= p, and m in (p^2/s, q^2/s) with no factor below s. So gear s's kill list in the section is

    K_s(p -> q) = s x { m in (p^2/s, q^2/s) : spf(m) >= s },

and for s > q^(2/3) every such m is prime. The new twins are the slots k with neither 6k-1 nor
6k+1 in any K_s. Read across sections, gear s walks through the s-rough numbers in order, each
section consuming the next band (p^2/s, q^2/s); the top gears consume primes just above q
(gear p's band is (p, q^2/p): the primes q, q_2, ...), the bottom gears consume numbers near
p^2/5. A new twin is a slot that no gear's current band reaches. This is the sieve of
Eratosthenes in section coordinates; it is stated here because it is the exact object every
section-view rerun is about, and because it says where the kills come from: from the primes
found at smaller scales, multiplied by the gears. research/section_mechanism_r29.py prints it
slot by slot and gear by gear (log research/data/r29/section_mechanism.log, sections 29 -> 31,
47 -> 53, 197 -> 199).

## 1. Inventory

  Check (doc)                          Whole-window / period object     Section-view form
  ------------------------------------ -------------------------------- ----------------------------------------
  word tree (word-tree.md 1-5)         window fusion tree               done: per-section trees (7.4) - runs, not
                                                                        the human's object
  twin paths (6)                       window twins                     done: provenance of new twins (8)
  sections (7.1-7.3)                   -                                counts only; superseded by 8
  provenance (8)                       -                                THE TEMPLATE for the rest
  2n-gap reordering / sort step        period odometer theorem          the digits of the new twins (their phase
  (two-n-gap-reordering.md)                                             vectors) - measured uniform in 8; the 2n
                                                                        law is a full-odometer statement and does
                                                                        not restrict to an interval; rerun = list
                                                                        the new twins in odometer order for small
                                                                        sections and look at the carries (item A)
  matrix formulation, merge =          period operator                  the delete step of gear s in every later
  lift-tensor-delete                                                    section is K_s above: an explicit list of
  (matrix-formulation.md)                                               primes in an explicit band (item B)
  tooth counterfactual                 period F percentile              the same section under moved teeth: which
  (tooth-counterfactual-percentile.md)                                  new twins die, which slots open, listed;
                                                                        the real teeth are the only choice for
                                                                        which every kill is a factorisation, so
                                                                        the counterfactual breaks the shared-m
                                                                        structure of K_s (item C)
  mirror parity laws, gear cells,      period involution / histogram    NO section form: the mirror maps the
  DFT of the gap histogram                                              section elsewhere in the period, and the
                                                                        histogram is a count. Not rerun.
  chains, dictionary, record law,      F, period-scale gaps             NO section form: F lives outside the
  increment law, LP certificates                                        window (word-tree.md 3). Not rerun.

## 2. Order of work

  A. Odometer order of the new twins (small sections, exact listing): the human's sort step
     applied to the new part only. Expected: the carries are those of the full odometer
     restricted to the vectors that land in the interval; look at the listing before deciding
     whether anything beyond CRT is there.
  B. The delete lists K_s per section as words: which primes m each gear consumes in each
     section, and the cross-section provenance of every kill (the section that produced the
     prime m). Exact; no counts.
  C. Section counterfactual: move one gear's teeth and list what changes in the new section.
  D. Anything from round 29's openers (docs/proof-search/agents-shared.md) that touches the
     window is to be rerun on sections before its result is read.

Rules for all items: pre-register, print the objects, counts only as sanity checks, refutations
labelled.
