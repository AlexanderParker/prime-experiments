"""j2_citesweep.py - ROUND 26 (harvester).  THE CITATION-NUMBERING SWEEP,
promoted from a manual referee step to a STANDING GATE.

Harvester 7d clause 2 makes a citation-numbering sweep a standing referee step
and 7d clause 1 makes prior-art checks EXPIRE.  Rounds 23-25 ran the sweep by
hand; a hand sweep does not fail when a document drifts.  This does.

WHAT IT DOES.
  A  Re-derives the Opera de Cribro alpha* reading FROM SCRATCH and settles it
     POSITIVELY: the book's printed root is its own stated Taylor/Newton value,
     not an erratum.  (Round 25 recorded it as "a discrepancy in the book"; that
     framing is withdrawn here, with the derivation shown.)
  B  Extracts every arXiv identifier from the Unit-1 documents and asserts each
     is in an ADJUDICATED REGISTRY carrying who/what/when.  An unregistered id
     fails the gate - which is the point: a new citation must be adjudicated
     before it can enter a document.
  C  Scans for FORBIDDEN STRINGS - the specific misnumberings and
     misattributions that have already cost this project correctness.
  D  Scans for INTERNAL CONTRADICTIONS between sections of the same document.
  E  Expiry: reports the age of every dated prior-art check.

Run: .venv/Scripts/python.exe research/j2_citesweep.py
"""

import re
import os
import datetime
from math import log

TODAY = datetime.date(2026, 8, 29)
ROOT = r"C:\dev\primes"
DOCS = [
    r"docs\novel\j2-upper-bound.md",
    r"docs\novel\j2-lower-ladder.md",
    r"docs\novel\layered-erdos-rankin.md",
    r"docs\novel\twin-percentile.md",
    r"docs\novel\paired-jacobsthal-values.md",
    r"docs\novel\jk-family.md",          # added r27: the (P6) family note
]

OUT = []
FAIL = []


def say(s=""):
    OUT.append(s)
    print(s)


def hr(t=""):
    say()
    say("=" * 78)
    if t:
        say(t)
        say("=" * 78)


# ---------------------------------------------------------------------------
hr("SECTION A - the Opera de Cribro alpha* reading, settled POSITIVELY")

say("  ODC sec. 6.6 prints a root alpha* = 0.264904 of the equation we")
say("  transcribed from the page image (research/data/odc6_scans/) as")
say()
say("      f(a) = a + (2+3a)/(3+4a) + log a + log((3+4a)/(2+3a)) = 0,")
say()
say("  and instructs: 'A numerical computation gives (use the Taylor expansion")
say("  at 1/4)'.  Round 25 recorded that 0.264904 does not solve f = 0 and")
say("  called it 'a discrepancy in the book'.  THAT FRAMING IS WITHDRAWN.")
say("  Doing what the book says reproduces the printed value exactly:")
say()


def f(a):
    return a + (2 + 3 * a) / (3 + 4 * a) + log(a) + log((3 + 4 * a) / (2 + 3 * a))


def fp(a, h=1e-8):
    return (f(a + h) - f(a - h)) / (2 * h)


lo, hi = 0.20, 0.30
for _ in range(200):
    mid = (lo + hi) / 2
    if f(lo) * f(mid) <= 0:
        hi = mid
    else:
        lo = mid
exact = (lo + hi) / 2
a0 = 0.25
newton = a0 - f(a0) / fp(a0)
printed = 0.264904

say("    f(1/4)                       = %+.10f" % f(a0))
say("    f'(1/4)                      = %+.10f" % fp(a0))
say("    ONE Newton/first-order Taylor step from a = 1/4:")
say("        1/4 - f(1/4)/f'(1/4)     =  %.10f" % newton)
say("    the book's printed alpha*    =  %.6f" % printed)
say("    exact root of the same f     =  %.10f" % exact)
say("    |Newton step - printed|      =  %.2e" % abs(newton - printed))
assert abs(newton - printed) < 1e-6, ("Newton step does not reproduce 0.264904",
                                      newton)
say()
say("  VERDICT (ours, first-hand): the printed 0.264904 IS the first-order")
say("  Taylor expansion about 1/4 - agreeing with our Newton step to 7 digits -")
say("  i.e. the book did exactly what it said it did.  IT IS NOT AN ERRATUM.")
say("  What we have is a SHARPENING of a stated approximation, and the caveat")
say("  that our f was transcribed from a page image and could itself be the")
say("  source of any residual: the equation is OUR READING of the printed one.")
say()
E = 2.718281828459045
b2_print = 1 + 2 / (E ** printed - 1)
b2_exact = 1 + 2 / (E ** exact - 1)
say("    beta_2 = 1 + 2/(e^alpha - 1)   at printed alpha*  =  %.6f" % b2_print)
say("                                   at the exact root  =  %.6f" % b2_exact)
say("    gain from solving exactly                          =  %.6f" %
    (b2_print - b2_exact))
assert abs(b2_print - 7.594004) < 1e-5 and abs(b2_exact - 7.583827) < 1e-5
say("  ASSERTED.  Nothing in Theorem 2G moves: 2G's binding root is the K -> 1")
say("  root alpha_infinity = 0.253321897, not this one.")

# ---------------------------------------------------------------------------
hr("SECTION B - the adjudicated arXiv registry, and the sweep")

REGISTRY = {
    "1706.00317": "Ziller & Morack, paired-Jacobsthal THEORY paper; defines "
                  "h_2 and Conjecture 6 (h_2 < p^2-p, n>=3). r21-25.",
    "1706.03668": "Ziller & Morack COMPANION computation note (11 days later); "
                  "Table 1 has h_2 for all p_n <= 73. Found r23 (retraction 1).",
    "1611.03310": "Ziller & Morack, earlier one-residue Jacobsthal computation "
                  "to p = 251. Cited only as the ORDINARY-side companion.",
    "2007.01808": "Ziller 2020, which even numbers occur as gaps. Fetched r22.",
    "1012.3809": "C. S. (CRAIG) Franze, 'Sifting limits for the Lambda^2 "
                 "Lambda^- sieve', JNT 131 (2011) 1962-1982. NOT 'M. Franze'.",
    "2602.22720": "Dudek & Dunn, explicit sum of two almost primes; Lemma 2.1 "
                  "gives kappa=2, K=3 for our density. Read in full r23.",
    "2608.09488": "Campbell; Theorem 2.1 transcribes ODC 7.7. Read in full r23.",
    "1802.07604": "Ford-Konyagin-Maynard-Pomerance-Tao, 'Long gaps in sieved "
                  "sets', JEMS 2021. ADVERSARIAL classes. First-hand r25.",
    "1408.4505": "Ford-Green-Konyagin-Tao, long gaps between primes; the k=1 "
                 "calibration target. First-hand r25.",
    "1306.1064": "Costello & Watts, 'A short note on Jacobsthal's function' - "
                 "THE id for g(n) <= 2 e^gamma k^(5+5 loglog k), k > 120.",
    "1208.5342": "Costello & Watts, a DIFFERENT range-restricted computational "
                 "result (50 <= k <= 10000). MUST NOT carry the k^(5+5..) bound.",
    "1209.3464": "third Costello-Watts-adjacent paper; became Math. Comp. 84 "
                 "(2015) no. 293. Recorded to keep the three apart.",
    "1511.03409": "Yamada. DO-NOT-CITE: Theorem 3.1 recorded as unproved as "
                  "stated and inconsistent with the standard references.",
    "2502.20470": "Holt, 'Eratosthenes sieve supports the k-tuple conjecture' "
                  "(Feb 2025). Source of the r22 novelty downgrades (Unit 2).",
    "2603.25915": "Holt (Mar 2026), one-residue, Legendre-directed; nothing "
                  "paired. Checked r22/r24.",
    "2302.00459": "Kalmynin & Konyagin, polynomial analogue of Jacobsthal; "
                  "nearest neighbour to (P2), contains neither direction.",
    "2010.01211": "Granville; states the deduction J(P(z)) << z^2 from "
                  "Iwaniec's h(k) << (k log k)^2 explicitly.",
    "1901.03785": "Mathematics 7 (2019), gaps in sets of primes; twin-percentile "
                  "context only.",
    "2109.02851": "Lichtman, 'A modification of the linear sieve, and the count "
                  "of twin primes'; Alg. & Num. Th. 19 (2025) no. 1. Thm 1.2: "
                  "pi_2(x) <~ 3.29956 Pi(x), Pi = 2C_2 x/(log x)^2. RECORD "
                  "asymptotic twin constant; history table Selberg 8 ... Wu 2004 "
                  "3.39951. Read first-hand 2026-08-29 (r26). SOURCE OF THE r26 "
                  "SELF-CORRECTION: Selberg's 8 multiplies 2C_2, not C_2.",
    "1910.13450": "Maynard, survey 'Gaps between primes'. Lemma 5 states the "
                  "Erdos-Rankin framework as ONE class per prime. Relay-sourced "
                  "r26; used only as corroboration, not load-bearing.",
}

pat = re.compile(r"arXiv:(\d{4}\.\d{4,5})")
found = {}
for d in DOCS:
    p = os.path.join(ROOT, d)
    if not os.path.exists(p):
        FAIL.append("missing document: %s" % d)
        continue
    txt = open(p, encoding="utf-8", errors="replace").read()
    for m in pat.finditer(txt):
        found.setdefault(m.group(1), set()).add(d)

say("  %-14s %-6s %s" % ("arXiv id", "docs", "adjudication"))
unreg = []
for aid in sorted(found):
    if aid in REGISTRY:
        say("  %-14s %-6d %s" % (aid, len(found[aid]), REGISTRY[aid][:56]))
    else:
        unreg.append(aid)
        say("  %-14s %-6d *** UNREGISTERED - ADJUDICATE BEFORE USE ***"
            % (aid, len(found[aid])))
if unreg:
    FAIL.append("unregistered arXiv ids in Unit-1 docs: %s" % unreg)
say()
say("  %d distinct arXiv ids across %d documents; %d unregistered."
    % (len(found), len(DOCS), len(unreg)))

# ---------------------------------------------------------------------------
hr("SECTION C - forbidden strings (the misnumberings already paid for)")

FORBIDDEN = [
    (r"Iwaniec[- ]Kowalski\s+Theorem\s+6\.9",
     "IK has no Theorem 6.9 (Ch.6 stops at 6.7); 6.9/6.10 are ODC theorem "
     "numbers. The IK result is Theorem 6.1 / Corollary 6.2."),
    (r"\bM\.\s*Franze\b",
     "The author of arXiv:1012.3809 is C. S. (Craig) Franze."),
    (r"Tenenbaum[^.\n]{0,40}Theorem\s+4\.3",
     "Tenenbaum's fundamental lemma is Theorem 4.4 (Thm 3 in the 1995 CUP ed)."),
    (r"Tenenbaum[^.\n]{0,40}Theorem\s+I\.4\.2",
     "I.4.2 is a COROLLARY (Bonferroni), not the fundamental lemma."),
    (r"Costello[- ]Watts\s+arXiv:1208\.5342",
     "The k^(5+5 loglog k) bound is arXiv:1306.1064; 1208.5342 is the "
     "range-restricted computational paper."),
    (r"Sean\s+Blight",
     "S. Blight is Sara Elizabeth Blight (Rutgers 2010)."),
]

hits = 0
for rx, why in FORBIDDEN:
    cre = re.compile(rx, re.I)
    for d in DOCS:
        p = os.path.join(ROOT, d)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p, encoding="utf-8",
                                      errors="replace"), 1):
            if cre.search(line):
                # allow it inside an explicit DO-NOT / retraction context
                ctx = line.lower()
                exempt = any(w in ctx for w in
                             ("does not exist", "do not cite", "not '", "chimera",
                              "must not", "earlier drafts", "is a different",
                              "forbidden", "not \"m. franze\"", "rather than"))
                if not exempt:
                    hits += 1
                    say("  HIT  %s:%d  %s" % (d, i, why))
                    FAIL.append("forbidden string in %s:%d" % (d, i))
if hits == 0:
    say("  0 hits.  All six forbidden patterns appear only inside explicit")
    say("  do-not-cite / retraction context, which is where they belong.")

# ---------------------------------------------------------------------------
hr("SECTION D - internal contradictions between sections of one document")

CONTRA = [
    (r"docs\novel\j2-upper-bound.md",
     r"the one now used: cite 2\*kappa \+ 0\.4454",
     r"FOR THE PAPER:\s*\n?cite 19/36",
     "Section 6a item 4 still instructs 'cite 0.4454'; section 9c SETTLED the "
     "conflict FOR 19/36 and instructs 'cite 19/36'. Direct self-contradiction "
     "a referee reads top-to-bottom."),
]
contra_hits = 0
for doc, rx_a, rx_b, why in CONTRA:
    p = os.path.join(ROOT, doc)
    txt = open(p, encoding="utf-8", errors="replace").read()
    a = re.search(rx_a, txt)
    b = re.search(rx_b, txt)
    if a and b:
        contra_hits += 1
        say("  CONTRADICTION in %s" % doc)
        say("     %s" % why)
        FAIL.append("internal contradiction in %s" % doc)
if contra_hits == 0:
    say("  0 contradictions among the registered pairs.")

# ---------------------------------------------------------------------------
hr("SECTION E - expiry of dated prior-art checks (7d clause 1)")

datepat = re.compile(r"(20\d\d)-(\d\d)-(\d\d)")
ages = {}
for d in DOCS:
    p = os.path.join(ROOT, d)
    if not os.path.exists(p):
        continue
    for m in datepat.finditer(open(p, encoding="utf-8",
                                   errors="replace").read()):
        try:
            dt = datetime.date(int(m.group(1)), int(m.group(2)),
                               int(m.group(3)))
        except ValueError:
            continue
        # only OUR check dates count; a date before the project started is a
        # cited source's own date (e.g. a 2014 blog post), not a check.
        if dt < datetime.date(2026, 8, 1):
            continue
        ages.setdefault(d, []).append(dt)
say("  %-40s %-12s %-12s %s" % ("document", "oldest", "newest", "max age (d)"))
worst_age = 0
for d in DOCS:
    if d not in ages:
        say("  %-40s %s" % (d, "NO DATED CHECK"))
        continue
    o, n = min(ages[d]), max(ages[d])
    age = (TODAY - o).days
    worst_age = max(worst_age, age)
    say("  %-40s %-12s %-12s %d" % (d.split("\\")[-1], o, n, age))
say()
say("  Oldest OWN check date in the Unit-1 corpus is %d days old.  7d clause 1"
    % worst_age)
say("  says a novelty claim older than a round must be RE-SEARCHED before it")
say("  is repeated in a summary; %s" %
    ("all are inside one round." if worst_age <= 14
     else "SOME ARE STALE - re-search before quoting."))
assert worst_age <= 14, ("stale prior-art checks", worst_age)

# ---------------------------------------------------------------------------
hr("VERDICT")
if FAIL:
    say("  j2_citesweep: %d PROBLEM(S) FOUND" % len(FAIL))
    for x in FAIL:
        say("    - %s" % x)
else:
    say("  j2_citesweep: ALL CHECKS GREEN")

with open(os.path.join(ROOT, r"research\data\j2_citesweep.out"), "w",
          encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")

assert not FAIL, FAIL
