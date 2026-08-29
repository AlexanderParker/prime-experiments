# Unit 1 — submission memo for the human

**From:** harvester lane, round 27, 2026-08-29.
**Purpose:** to let you decide whether, where and how to submit. **The decision
is yours; this memo is not a recommendation to submit.** Sources: Unit 1 as
assembled (`docs/novel/j2-upper-bound.md` §11), `j2-lower-ladder.md`,
`layered-erdos-rankin.md`, `twin-percentile.md`, `paired-jacobsthal-values.md`,
and the new `jk-family.md`. Standing gates re-run clean today:
`j2_referee.py`, `j2_citesweep.py`, `j2_odcpages.py`, `jk_family.py` — all
GREEN.

---

## 1. What the paper claims

**Title.** *The paired Jacobsthal function: first upper bounds, a first lower
bound from the paired structure, and the structure of its maximisers.*

`j_2 = h_2` is Ziller–Morack's paired Jacobsthal function (arXiv:1706.00317,
2017): the largest gap between consecutive `n` with `n` and `n+E` both coprime
to the modulus, maximised over even `E`. They conjecture
`h_2(p_n#) < p_n^2 - p_n` and prove **no upper bound of any strength**; no
follow-up supplies one.

The paper supplies the ladder:

| rung | statement | constants |
|---|---|---|
| 1 | `j_2 <= 2·3^{n-1}/V_n + 1 < 3^{n+1}(log p_n)^2` | all explicit |
| 3E | quasi-polynomial `j_2 < p_n^{9.30 loglog p_n}`, asymptotic constant exactly `2λ_* = 7.182242` | all explicit |
| 2E″ | `j_2 <= 7.2671e11 · p_n^15 (log p_n)^10 + 1` | all explicit |
| **2G** | `j_2 <= C p_n^{8.04162}(8.04162 log p_n + 1)(log p_n)^2 + 1`, `log₁₀C = 57.5` | all explicit |
| 2G-∞ | `j_2 <<_ε p_n^{s+ε}` for every `s > 7.93727` | computable |
| 2 | `j_2 <<_ε p_n^{4.266+ε}` | **not explicit, and not makeable so** |

plus two lower bounds — `h_2 >= (1.349+o(1)) z log z` at every finite scale
(P1), and `h_2 >= (0.01275+o(1)) z (log z)^3 (lll z)^2/(ll z)^4`
asymptotically (P2′) — an exact-value / maximiser-structure section (the
per-difference family `F_d`, the twin percentile, the shallow-extension cap
law), and a printed falsification target: **one exact `h_2` beyond
`p_n = 73`** discriminates the two live growth readings, which are now a full
`log z` apart.

## 2. What the paper does not claim

Eight numbered items, written for the referee, at §11c. The load-bearing ones:
**no progress on Conjecture 6** (proved 4.266 against its 2, and the gap is
parity, not arithmetic); **no new sieve theory** — the rungs are Legendre,
Brun and Friedlander–Iwaniec applied to a dimension-2 density, and *the
contribution is that the ladder was empty, not that the rungs are hard*; **the
computational half is replication plus structure**, given Ziller–Morack's
ancillary files; **the order `z log z` of (P1) is not new** — FKMPT
arXiv:1802.07604 Remark 7 names this exact sieving system and records that
order as "the 'trivial' bound", without proof or constant; and **no
twin-prime-gap corollary**, because a pigeonhole argument those authors call
trivial already beats it.

## 3. The three strongest points a referee will see

1. **The ladder was genuinely empty and is now four explicit rungs deep.**
   Nine years, a stated conjecture, an OEIS sequence, and no bound of any
   strength. The paper also *proves where explicitness stops* and extends that
   boundary first-hand (from the DHR differential-delay system to the
   `Λ²Λ⁻` family, via Blight's thesis), so a referee cannot ask for the obvious
   improvement — the answer is already in the paper.
2. **(P2′) is a real theorem and a new construction.** A layered Erdős–Rankin
   covering with *two* classes per prime, where the second layer is a shifted
   Eratosthenes and the joint survivors are twin primes. It is parity-free for
   a structural reason (it stops at Rankin level; the FGKT/Maynard upgrade
   would need a *lower* bound for twins). It gives two full powers of `loglog`
   where the five authors who named the system hoped only for "a small power".
3. **The paper audits itself, visibly.** Two standing gates recompute every
   recomputable number and every citation number by independent code; five
   self-found retractions are printed rather than buried; and the whole thing
   is reproducible from a repository. Referees rarely see a submission that
   names its own weakest constant.

## 4. The three weakest points a referee will see

1. **"This is an exercise."** Every upper rung is a textbook sieve applied to
   `ω(p) = 2`. The honest defence is in the paper (the ladder was empty), and
   round 27 strengthens it: the ladder is **uniform in the sifting dimension**,
   so `j_1` (ordinary Jacobsthal) and `j_2` are the first two members of a
   family `j_k` that nobody has written down (`docs/novel/jk-family.md`).
   Including that section converts "one function, standard tools" into "a
   family, and the family is the contribution." **I recommend including it.**
2. **Audience.** arXiv:1706.00317 has **exactly one citation in nine years**
   — its own companion note. zbMATH returns *no records* for "paired
   Jacobsthal". The referee's real question is not "is this correct?" but "who
   is this for?", and the honest answer is: Ziller, Morack, the ordinary-
   Jacobsthal computation community, and the long-gaps community via §(P2′).
   That is a small readership and the paper should say why it should be
   larger (the `j_k` family, and the parity ceiling being uniform in `k`).
3. **The lower half is asymptotic and the computational half is replication.**
   (P2′) has no finite-`z` content, no writable `x_0` (the `o(1)` decays like
   `1/logloglog log x`), and no kernel check; (P1) is the bound to quote at any
   `z` anyone will evaluate, and its *order* is in print as trivial. The
   exact-value table replicates Ziller–Morack's own computation; what is ours
   there is `F_d`, the percentile, and the cap law.

*(A fourth, smaller: every Opera de Cribro citation was verified from
publisher-preview page images, not a copy held in hand. Round 27 closed the
last three unread equations — (5.38), (6.69), p. 74 — and the pages
cross-check each other, but a submission should cite theorem numbers and say
nothing about typography.)*

## 5. Venue class — assessment, not a recommendation

* **arXiv math.NT first, whatever else you decide.** Ziller–Morack's work
  lives on arXiv; Holt's related programme lives on arXiv and primegaps.info
  rather than in journals. A preprint reaches the entire actual readership of
  this problem in a day, and it timestamps the priority claim on `j_k`.
* **If (P2′) travels with it:** a specialist analytic-number-theory journal is
  in range — *Journal of Number Theory*, *Ramanujan Journal*, *Acta
  Arithmetica* (a stretch), *Mathematika*. (P2′) is the piece that makes it a
  research paper rather than a note.
* **If the paper stays elementary/computational:** *INTEGERS (EJCNT)* or
  *Journal of Integer Sequences* — both a natural fit, both fast, and JIS suits
  the A288815 / exact-value half.
* **Not** a general-audience or top-5 venue. Nothing here would survive that
  filter, and submitting there costs months.
* **Worth considering:** writing to Ziller and Morack before or alongside
  submission. They are simultaneously the entire prior readership, the natural
  referees, and the people who can compute `h_2` at `p_n = 151` — the single
  number that would most improve the paper.

## 6. The question that is yours alone: AI-assistance disclosure

This work was produced by an agent team under your direction, with
machine-verified gates. **I have not decided anything about this and should
not.** The facts you need:

* No journal permits an AI system as an author. That is universal and not in
  question; authorship is yours.
* Most major publishers (Elsevier, Springer, AMS, and the arXiv moderation
  policy) now require **disclosure of generative-AI use**, typically in an
  acknowledgement or a short methods statement. Policies differ on whether
  assistance in *deriving* results, as opposed to *writing prose*, must be
  disclosed — and this paper is the former far more than the latter.
* A minority of venues and individual editors treat AI-assisted submissions
  more sceptically, and a few desk-reject.
* **The asymmetry worth weighing:** this paper's strongest defence is that
  every recomputable claim is reproduced by independent code and every citation
  number is gate-checked. Disclosure makes that apparatus *the point* rather
  than a curiosity. Non-disclosure that later surfaces would damage exactly the
  credibility the gates were built to earn.
* Three shapes to choose between: (a) a one-line acknowledgement naming the
  assistance and pointing at the reproduction repository; (b) a short methods
  paragraph describing the gate discipline and the self-found retractions;
  (c) no statement. My only input is that (a) and (b) are cheap and (c) is a
  bet.

## 7. Still to do before anything is sent

LaTeX, and the scope decision: does `F_d` travel with Unit 1, and does the
`j_k` section travel inside it or as a separate note? Nothing else is
outstanding — the round-23 blockers, the round-24 openings and the ODC
page-image caveat are all closed, both standing gates are green, and the
prior-art checks (including OEIS at `k = 3`) are dated today.
