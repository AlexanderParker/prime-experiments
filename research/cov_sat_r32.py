"""cov_sat_r32.py -- the COVERABILITY SPECTRUM COV(M) by SAT, second build.

READ THIS FIRST.  alignment-rules.md section 6.5 calls COV(M) a
"[named construct; NOT BUILT]".  THAT IS STALE.  COV(M) WAS BUILT IN ROUND 20
by the mechanic lane and lives in **research/cov_sat.py**; commit fe4c390 says
"COV-SAT reaches machine 41 complete", and mechanic.md K1 records exact gap
spectra and complete hole lists at m11..m37 plus m41, and F_j at m23/m29/m31.
This file was written on the mistaken premise that the construct did not
exist, and it briefly OVERWROTE research/cov_sat.py before the collision was
caught; that file has been restored byte-for-byte to its committed state and
nothing of round 20 is lost.

What this second build adds, and what it does not, is set out in
research/proof/cov_spectrum.md section 0a.  Briefly: it is an independent
re-implementation that agrees with round 20 at every machine; it adds the
LEFT-FLANK MONOTONE form (which makes bisection valid, though it is not
cheaper than round 20's spectrum scan at m37); it adds Q*_J, the WORD-LEGAL
spectrum, which round 20 did not build (round 20's Q_j uses the size shadow
"middles >= a", not the sharp predicate); and it carries the ladder past the
wall on the lower-bound side only, to m61/m67/m71.

The construct:

    COV(M) = { L : an interval of L consecutive columns is coverable by the
               gears 5..M, with both flanking columns spared }

and the spectrum it decides,

    F(M)   = max { L + 1 : L in COV(M) }                   (max-gap units)
    F_J(M) = max { L + 1 : L coverable with <= J-1 of the L interior
                           columns ALSO spared, both flanks spared }

with F_1 = F.  No period is ever built.  This is section 2.8's realisability
CSP (alignment as feasibility, no period) posed as a pure covering problem and
handed to CaDiCaL.

---------------------------------------------------------------------------
THE ENCODING
---------------------------------------------------------------------------
Vocabulary is alignment-rules.md section 0.  A column k is the pair 6k-1,6k+1.
Gear q >= 5 STRIKES column k iff k = +-u_q (mod q), u_q = round(q/6); the other
q-2 residues are its OPENINGS.  Machine M = {5..p} has one gear per prime in
[5, p].

Anchor the candidate stretch at an unknown column c and write t = 0..L+1 for
the offsets, so t = 0 and t = L+1 are the flanks and t = 1..L the interior.
Put s_q = -c mod q.  Then

    gear q strikes offset t   <=>   t = s_q +- u_q  (mod q)
                              <=>   s_q = t -+ u_q  (mod q),

i.e. EXACTLY TWO phase values of gear q make it strike a given offset.  The
moduli are distinct primes, so by CRT the phase vector (s_q) ranges over all of
prod Z_q as c ranges over the period: a phase vector IS a column, and no period
has to be enumerated.  The whole problem is therefore

    variables   y_{q,s}, s in Z_q, "gear q has phase s"          exactly-one
    flanks      ~y_{q,t -+ u_q}  for t in {0, L+1}, every gear    (unit clauses)
    interior    OR_q ( y_{q,t-u_q} v y_{q,t+u_q} )  for t = 1..L  (one clause)

with NO auxiliary variables at all: the flank condition just deletes up to four
phase values per gear, and each interior column is one clause of width
2*pi(M).  For J > 1 add one selector o_t per interior column ("t is spared"),
an AT-MOST-(J-1) cardinality constraint over them, and replace the interior
clause by

    o_t v OR_q (...)                       t is covered unless it is spared
    ~o_t v ~y_{q,t -+ u_q}                 a spared column is struck by nobody

AT-MOST J-1 rather than EXACTLY J-1 is deliberate and loses nothing: F_j is
non-decreasing in j, so max over stretches with <= J-1 interior openings is
F_J itself.  A witness with fewer interior openings is still a witness for a
stretch of that length, hence still a valid lower bound on F_J.

SOUNDNESS OF THE CLIMB.  "Both flanks spared" is NOT downward closed in L, so
"L SAT, L+1 UNSAT" does not by itself bound F_J.  The monotone predicate is
the LEFT-FLANK-ONLY one, C_J(L) = "some stretch of L columns, at most J-1 of
them open, has an open column immediately to its left"; a sub-run has no more
openings than the run, so C_J(L) => C_J(L-1), and max{L : C_J(L)} = F_J - 1
(slide the window right of the nearest open column at or below its left end).
Every UPPER bound below is therefore taken in the left form; every LOWER bound
is a both-flanks witness, which exhibits an actual run of gaps.

---------------------------------------------------------------------------
WHAT COUNTS AS AN ANSWER
---------------------------------------------------------------------------
* Every SAT answer comes with the witness PHASE VECTOR (s_q)_q and that vector
  is re-verified by `verify_witness` -- plain integer residue arithmetic, no
  solver, no CNF -- before the value is reported.
* Every UNSAT answer is SOLVER-CERTIFIED, never "proved": it is recorded as
  "CaDiCaL 1.9.5 via python-sat, encoding cov_sat.build, no covering found".
  The phrase used throughout is "none has been found", never "none exists".
* The GATE is the machines with a full period on record (m11..m23, and the
  m29..m59 corpus rows): SAT must return the scanned value, both directions.

Runs in .venv-sat (python-sat / CaDiCaL):
    .venv-sat/Scripts/python.exe research/cov_sat.py <cmd> ...

Commands:
  gate                      -- reproduce (a): F and F_2 at m11..m23 vs corpus
  one <p> <L> [J]           -- decide one (M={5..p}, L, J); prints the witness
  check <p> <J> <v>         -- two-sided decision of F_J(M) = v
  ladder <p> <J> <lo> <hi>  -- climb L in the monotone "left" form until the
                               first UNSAT; writes research/data/proof/*.json
  legal <p> <J> <lo> <hi>   -- Q*_J(M; q'): as F_J but the EXACTLY J-1 spared
                               interiors must be word-legal (all struck by one
                               phase of the incoming gear q'); the whole range
                               is scanned because Q*_J is not monotone in L
"""
from __future__ import annotations

import json
import os
import sys
import time

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

SOLVER = "cadical195"
SOLVER_NAME = "CaDiCaL 1.9.5 (python-sat 'cadical195')"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "proof")


# --------------------------------------------------------------------------
# the machine
# --------------------------------------------------------------------------
def primes_upto(n):
    s = [True] * (n + 1)
    s[0] = s[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, n + 1, i):
                s[j] = False
    return [i for i in range(2, n + 1) if s[i]]


def gears(p):
    """The gears of machine M = {5..p}."""
    return [q for q in primes_upto(p) if q >= 5]


def tooth(q):
    """u_q = 6^{-1} mod q = round(q/6).  Teeth are +-u_q."""
    u = (q + 3) // 6            # round(q/6) for q = 1,5 mod 6
    assert (6 * u - 1) % q == 0 or (6 * u + 1) % q == 0, (q, u)
    return u


def next_prime(p):
    n = p + 1
    while True:
        if all(n % d for d in range(2, int(n ** 0.5) + 1)):
            return n
        n += 1


def letters(qp):
    """a = 2 round(q'/6), b = q' - a for the incoming gear q'."""
    a = 2 * tooth(qp)
    return a, qp - a


# --------------------------------------------------------------------------
# the encoding
# --------------------------------------------------------------------------
def build(p, L, J=1, legal_qp=None, flanks="both"):
    """CNF for 'some stretch of L consecutive columns is covered by the gears
    5..p with the flank(s) spared and at most J-1 interior columns also spared'.

    flanks = "both": columns 0 and L+1 both spared.  This is the EXTREMAL form;
    a SAT answer here exhibits an actual run of <= J gaps of total L+1, so it
    certifies F_J(M) >= L+1.  It is NOT downward closed in L, so it must not be
    used to climb.
    flanks = "left": only column 0 is spared.  Write C_J(L) for this predicate.
    C_J IS downward closed (a sub-run has no more openings than the run) and
        max { L : C_J(L) } = F_J(M) - 1
    exactly: one direction is the maximal gap-run itself, the other slides the
    window right of the nearest open column at or below its left end, which
    can only remove interior openings.  So the UPPER bound direction -- the
    solver-certified UNSAT that says "no stretch this long has been found" --
    must be taken in the "left" form, and a climb in the "left" form is sound.

    legal_qp: if given (= the incoming gear q'), the J-1 spared interior
    columns are required to be word-legal, and the count is EXACTLY J-1, which
    makes the answer Q*_J(M; q') rather than F_J(M).

    Word-legality is encoded at its source rather than through T2/T3.  The
    record's predicate is "every middle in V = {0, +a, -a} mod q' and the
    letter word T3-alternating"; that predicate holds of a set of points iff
    the points all lie on ONE phase of gear q' -- i.e. iff q' can strike every
    one of them at once (differences of two elements of {s+u', s-u'} are
    0 and +-2u' = +-a, and the alternation is exactly the two-tooth
    bookkeeping).  So q' is added as one more phase variable, constrained to
    strike every spared interior column.  The FLANKS are deliberately left
    unconstrained with respect to q', because the record's Q*_J puts no
    condition on the two outer gaps.

    Returns (cnf, y, gs) or (None, None, gs) if the flank constraints alone
    empty some gear's domain.
    """
    gs = gears(p)
    pool = IDPool()
    y = {}
    cnf = CNF()

    flank_offsets = (0, L + 1) if flanks == "both" else (0,)
    for q in gs:
        u = tooth(q)
        dead = set()
        for t in flank_offsets:
            dead.add((t - u) % q)
            dead.add((t + u) % q)
        live = [s for s in range(q) if s not in dead]
        if not live:
            return None, None, gs           # flanks alone kill this gear
        for s in live:
            y[(q, s)] = pool.id(("y", q, s))
        lits = [y[(q, s)] for s in live]
        cnf.append(lits)                                    # at least one
        for i in range(len(lits)):                          # at most one
            for j in range(i + 1, len(lits)):
                cnf.append([-lits[i], -lits[j]])

    def strike_lits(t):
        """The phase literals that make some gear strike offset t."""
        out = []
        for q in gs:
            u = tooth(q)
            for s in ((t - u) % q, (t + u) % q):
                if (q, s) in y:
                    out.append(y[(q, s)])
        return out

    if J <= 1:
        for t in range(1, L + 1):
            cl = strike_lits(t)
            if not cl:
                return None, None, gs
            cnf.append(cl)
        return cnf, y, gs

    o = {t: pool.id(("o", t)) for t in range(1, L + 1)}
    for t in range(1, L + 1):
        cl = strike_lits(t)
        cnf.append([o[t]] + cl)                 # covered unless spared
        for lit in cl:
            cnf.append([-o[t], -lit])           # spared => struck by nobody
    olits = [o[t] for t in range(1, L + 1)]
    if legal_qp is None:
        if J - 1 < L:                       # otherwise the cap is vacuous
            cnf.extend(CardEnc.atmost(lits=olits, bound=J - 1, vpool=pool,
                                      encoding=EncType.seqcounter).clauses)
        return cnf, y, gs

    # Q*_J: EXACTLY J-1 spared interiors, all struck by one phase of q'.
    if J - 1 > L:
        return None, None, gs               # not enough interior columns
    if J - 1 == L:
        for t in range(1, L + 1):
            cnf.append([o[t]])
    else:
        cnf.extend(CardEnc.equals(lits=olits, bound=J - 1, vpool=pool,
                                  encoding=EncType.seqcounter).clauses)
    qp, up = legal_qp, tooth(legal_qp)
    z = {s: pool.id(("z", s)) for s in range(qp)}
    cnf.append([z[s] for s in range(qp)])
    for i in range(qp):
        for j in range(i + 1, qp):
            cnf.append([-z[i], -z[j]])
    for t in range(1, L + 1):
        # o_t  =>  q' strikes t, i.e. its phase is one of the two values
        cnf.append([-o[t], z[(t - up) % qp], z[(t + up) % qp]])
    return cnf, y, gs


# --------------------------------------------------------------------------
# the solver call, and the INDEPENDENT check of every witness
# --------------------------------------------------------------------------
def decide(p, L, J=1, legal_qp=None, verbose=True, timeout=None,
           flanks="both"):
    cnf, y, gs = build(p, L, J, legal_qp, flanks)
    if cnf is None:
        if verbose:
            print(f"    m{p} L={L} J={J}  UNSAT (by construction: a gear's "
                  f"domain is empty)", flush=True)
        return False, dict(vars=0, clauses=0, conflicts=0, propagations=0,
                           secs=0.0, trivial=True), None
    t0 = time.time()
    with Solver(name=SOLVER, bootstrap_with=cnf) as s:
        if timeout:
            import threading
            timer = threading.Timer(timeout, s.interrupt)
            timer.start()
            sat = s.solve_limited(expect_interrupt=True)
            timer.cancel()
        else:
            sat = s.solve()
        st = s.accum_stats()
        stats = dict(vars=cnf.nv, clauses=len(cnf.clauses),
                     conflicts=st.get("conflicts"),
                     decisions=st.get("decisions"),
                     propagations=st.get("propagations"),
                     secs=round(time.time() - t0, 3))
        wit = None
        if sat:
            model = set(l for l in s.get_model() if l > 0)
            wit = {}
            for q in gs:
                hit = [s_ for (qq, s_) in y if qq == q and y[(qq, s_)] in model]
                assert len(hit) == 1, f"gear {q}: {len(hit)} phases in model"
                wit[q] = hit[0]
    if verbose:
        tag = "SAT  " if sat else ("UNSAT" if sat is False else "TIMEOUT")
        print(f"    m{p:<3d} L={L:<4d} J={J} {flanks[0]}  {tag} "
              f"vars={stats['vars']:6d} "
              f"clauses={stats['clauses']:8d} conf={stats['conflicts']} "
              f"{stats['secs']:.2f}s", flush=True)
    return sat, stats, wit


def verify_witness(p, L, J, wit, legal_qp=None, flanks="both"):
    """INDEPENDENT re-verification: plain residue arithmetic on the phase
    vector, no CNF, no solver.  Returns the list of spared interior offsets.

    A phase vector (s_q) means the stretch starts at the column c with
    c = -s_q (mod q); gear q strikes offset t iff t = s_q +- u_q (mod q).
    """
    gs = gears(p)
    assert set(wit) == set(gs), "witness does not name exactly the gears"

    def struck(t):
        for q in gs:
            u = tooth(q)
            if (t - wit[q]) % q in (u % q, (-u) % q):
                return q
        return None

    assert struck(0) is None, f"flank 0 is struck by gear {struck(0)}"
    if flanks == "both":
            assert struck(L + 1) is None, \
            f"flank {L+1} is struck by gear {struck(L + 1)}"
    spared = [t for t in range(1, L + 1) if struck(t) is None]
    if legal_qp is None:
        assert len(spared) <= J - 1, \
            f"{len(spared)} interior columns spared, budget is {J-1}"
        return spared
    assert len(spared) == J - 1, \
        f"{len(spared)} interior columns spared, Q*_{J} needs exactly {J-1}"
    # word-legality, checked BOTH ways and required to agree:
    # (i) some phase of q' strikes every spared interior column;
    qp, up = legal_qp, tooth(legal_qp)
    phases = [s for s in range(qp)
              if all((t - s) % qp in (up % qp, (-up) % qp) for t in spared)]
    # (ii) every middle lies in V = {0, +a, -a} mod q' and the nonzero-class
    #      middles strictly alternate (T2 + T3, the record's own predicate).
    a, _ = letters(qp)
    ok = {v % qp for v in (0, a, -a)}
    seq, t2 = [], True
    for i in range(len(spared) - 1):
        v = (spared[i + 1] - spared[i]) % qp
        t2 = t2 and v in ok
        if v != 0:
            seq.append(v)
    t3 = all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))
    assert bool(phases) == (t2 and t3), \
        f"the two legality tests disagree at {spared}: phases={phases}, " \
        f"T2={t2}, T3={t3}"
    assert phases, f"spared set {spared} is not word-legal mod {qp}"
    return spared


def column_of(wit):
    """CRT: the actual column c of the witness stretch's left flank, mod P."""
    c, mod = 0, 1
    for q, s in sorted(wit.items()):
        want = (-s) % q
        while c % q != want:
            c += mod
        mod *= q
    return c, mod


# --------------------------------------------------------------------------
# per-run result files, one per (M, L, J)
# --------------------------------------------------------------------------
def record(p, L, J, sat, stats, wit, legal_qp=None, spared=None,
           flanks="both"):
    os.makedirs(OUT, exist_ok=True)
    tag = "Q" if legal_qp else "F"
    fn = os.path.join(OUT, f"cov_{tag}_m{p}_L{L}_J{J}_{flanks}.json")
    doc = dict(machine=p, L=L, J=J, value_if_sat=L + 1, legal_qp=legal_qp,
               flanks=flanks,
               result=("SAT" if sat else "UNSAT" if sat is False else "TIMEOUT"),
               solver=SOLVER_NAME, encoding="research/cov_sat.py:build",
               stats=stats, witness=(None if wit is None
                                     else {str(k): v for k, v in wit.items()}),
               spared=spared, when=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(fn, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return fn


# --------------------------------------------------------------------------
# deciding F_J exactly
# --------------------------------------------------------------------------
def ladder(p, J=1, lo=1, hi=None, legal_qp=None, timeout=None, verbose=True,
           flanks="left"):
    """Climb L from lo in the (monotone) "left" form until the first UNSAT.
    F_J(M) = L_max + 1 with L_max the largest SAT L.  Q*_J (legal_qp set) is
    NOT monotone, so there the whole range [lo, hi] is scanned and the maximum
    SAT L is returned -- never a first-UNSAT stop.
    """
    best, best_wit, ev = None, None, []
    stop_on_unsat = legal_qp is None
    L = lo
    while hi is None or L <= hi:
        sat, stats, wit = decide(p, L, J, legal_qp, verbose=verbose,
                                 timeout=timeout, flanks=flanks)
        spared = None
        if sat:
            spared = verify_witness(p, L, J, wit, legal_qp, flanks)
            best, best_wit = L, wit
        ev.append(dict(L=L, result=("SAT" if sat else
                                    "UNSAT" if sat is False else "TIMEOUT"),
                       stats=stats, spared=spared))
        record(p, L, J, sat, stats, wit, legal_qp, spared, flanks)
        if sat is None:
            break
        if not sat and stop_on_unsat:
            break
        L += 1
    return (None if best is None else best + 1), best_wit, ev


# --------------------------------------------------------------------------
# the corpus, and the GATE
# --------------------------------------------------------------------------
CORPUS_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
            41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
CORPUS_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
             41: 103, 43: 116, 47: 134, 53: 159, 59: 173}


def bisect(p, J=1, lo=1, hi=None, timeout=None):
    """F_J(M) by BINARY SEARCH on the monotone left-flank predicate C_J.

    Invariant: C_J(lo) is SAT (or lo is a value known SAT), C_J(hi+1) UNSAT.
    F is non-decreasing in the machine -- adding a gear only strikes more
    columns, so every struck run survives -- which is what makes F_J(previous
    machine) - 1 a legitimate SAT starting point.  Each probe is recorded.
    Returns (F_J or None, best_wit, evidence, bracket).
    """
    ev, best, best_wit = [], None, None
    lo_ok, hi_bad = None, None

    def probe(L):
        nonlocal best, best_wit
        sat, stats, wit = decide(p, L, J, None, timeout=timeout, flanks="left")
        spared = None
        if sat:
            spared = verify_witness(p, L, J, wit, None, "left")
            if best is None or L > best:
                best, best_wit = L, wit
        ev.append(dict(L=L, result=("SAT" if sat else "UNSAT" if sat is False
                                    else "TIMEOUT"), stats=stats,
                       spared=spared))
        record(p, L, J, sat, stats, wit, None, spared, "left")
        return sat

    s = probe(lo)
    if s is not True:
        return None, None, ev, (None, lo)
    lo_ok = lo
    if hi is None:
        hi = lo * 2
    while probe(hi) is True:
        lo_ok, hi = hi, hi * 2
    hi_bad = hi
    while hi_bad - lo_ok > 1:
        mid = (lo_ok + hi_bad) // 2
        r = probe(mid)
        if r is None:
            return None, best_wit, ev, (lo_ok, hi_bad)
        if r:
            lo_ok = mid
        else:
            hi_bad = mid
    return lo_ok + 1, best_wit, ev, (lo_ok, hi_bad)


def two_sided(p, J, value, legal_qp=None, timeout=None):
    """Decide F_J(M) = value exactly, in the two forms the soundness argument
    needs:
      LOWER  L = value-1 in the "both" form -- a real run of <= J gaps summing
             to `value`, witness re-verified => F_J >= value.
      UPPER  L = value   in the "left" form -- the monotone predicate C_J;
             UNSAT there is the solver-certified claim that no stretch of that
             length has been found, and by downward closure it rules out every
             LONGER stretch too => F_J <= value.
    Returns (ok, sat_stats, unsat_stats, wit, spared)."""
    lo, slo, wl = decide(p, value - 1, J, legal_qp, timeout=timeout,
                         flanks="both")
    spared = (verify_witness(p, value - 1, J, wl, legal_qp, "both")
              if lo else None)
    # record the lower bound BEFORE starting the (much longer) upper-bound
    # decision, so that killing the run does not throw the witness away
    record(p, value - 1, J, lo, slo, wl, legal_qp, spared, "both")
    hi, shi, _ = decide(p, value, J, legal_qp, timeout=timeout, flanks="left")
    record(p, value, J, hi, shi, None, legal_qp, None, "left")
    return (lo and hi is False), slo, shi, wl, spared


def gate():
    """(a): F and F_2 by SAT at m11..m23, against the scanned corpus."""
    print(f"[COV GATE] solver = {SOLVER_NAME}")
    print("           encoding = research/cov_sat.py:build")
    print("           every SAT witness re-verified by verify_witness "
          "(residue arithmetic, no solver)\n")
    bad = []
    for J, table in ((1, CORPUS_F), (2, CORPUS_F2)):
        name = "F  " if J == 1 else "F_2"
        for p in (11, 13, 17, 19, 23):
            v = table[p]
            ok, slo, shi, wit, spared = two_sided(p, J, v)
            c, mod = column_of(wit) if wit else (None, None)
            print(f"    {name}(m{p:<2d}) = {v:<3d}  {'OK ' if ok else 'BAD'}  "
                  f"SAT conf={slo['conflicts']} ({slo['secs']:.2f}s) / "
                  f"UNSAT conf={shi['conflicts']} ({shi['secs']:.2f}s)"
                  f"  spared={spared}  column c={c}", flush=True)
            if not ok:
                bad.append((name, p, v))
    assert not bad, f"SAT disagrees with the scanned corpus at {bad}"
    print("\nALL ASSERTIONS GREEN")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "gate":
        gate()
        return
    cmd = sys.argv[1]
    if cmd == "one":
        p, L = int(sys.argv[2]), int(sys.argv[3])
        J = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        sat, stats, wit = decide(p, L, J)
        if sat:
            sp = verify_witness(p, L, J, wit, None)
            print(f"    witness verified; spared interiors {sp}")
            print(f"    phases {wit}")
        record(p, L, J, sat, stats, wit, None, sp if sat else None)
    elif cmd == "check":
        # two-sided check of a claimed value: L = v-1 SAT, L = v UNSAT
        p, J, v = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        to = float(sys.argv[5]) if len(sys.argv) > 5 else None
        ok, slo, shi, wit, spared = two_sided(p, J, v, timeout=to)
        c, mod = column_of(wit) if wit else (None, None)
        print(f"CHECK F_{J}(m{p}) = {v}  {'OK' if ok else 'MISMATCH'}  "
              f"SATconf={slo['conflicts']} UNSATconf={shi['conflicts']} "
              f"({slo['secs']:.2f}s / {shi['secs']:.2f}s) spared={spared}")
        print(f"  phases {wit}")
        print(f"  column c={c}")
    elif cmd == "table":
        # collect every per-run result file into the ladder it certifies
        import glob
        lb, ub, tmo = {}, {}, {}
        for fn in glob.glob(os.path.join(OUT, "cov_F_m*.json")):
            d = json.load(open(fn))
            k = (d["machine"], d["J"])
            if d["result"] == "SAT" and d.get("flanks") != "left":
                lb[k] = max(lb.get(k, 0), d["L"] + 1)
            elif d["result"] == "SAT":
                lb[k] = max(lb.get(k, 0), d["L"] + 1)
            elif d["result"] == "UNSAT" and d.get("flanks") == "left":
                ub[k] = min(ub.get(k, 10 ** 9), d["L"])
            elif d["result"] == "TIMEOUT":
                tmo.setdefault(k, []).append(d["L"])
        print(f"{'M':>4} {'J':>2} {'lower':>7} {'upper':>7} {'exact':>7} "
              f"{'corpus':>7}")
        for k in sorted(set(lb) | set(ub)):
            p, J = k
            lo, hi = lb.get(k), ub.get(k)
            ex = lo if (lo and hi and lo == hi) else None
            corp = (CORPUS_F if J == 1 else CORPUS_F2 if J == 2 else {}).get(p)
            flag = "" if (corp is None or ex is None) else \
                ("  OK" if ex == corp else "  MISMATCH")
            print(f"{p:>4} {J:>2} {str(lo):>7} {str(hi):>7} {str(ex):>7} "
                  f"{str(corp):>7}{flag}")
        for k, ls in sorted(tmo.items()):
            print(f"  TIMEOUT m{k[0]} J={k[1]} at L in {sorted(set(ls))}")
    elif cmd == "ub":
        # one-sided UPPER bound: left-flank UNSAT at L says F_J(M) <= L.
        # This is the cheap end beyond the scan wall: an UNSAT far ABOVE the
        # true F_J is much easier than the tight one at F_J itself, and
        # L = U(previous machine) + q' is exactly the budget inequality.
        p, J, L = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        to = float(sys.argv[5]) if len(sys.argv) > 5 else None
        sat, st, w = decide(p, L, J, None, timeout=to, flanks="left")
        record(p, L, J, sat, st, w, None, None, "left")
        if sat is False:
            print(f"UB   F_{J}(m{p}) <= {L}   solver-certified UNSAT "
                  f"({SOLVER_NAME}), conf={st['conflicts']} {st['secs']:.1f}s")
        elif sat:
            print(f"UB   F_{J}(m{p}) >= {L + 1}  -- SAT, so {L} is NOT an "
                  f"upper bound; conf={st['conflicts']} {st['secs']:.1f}s")
        else:
            print(f"UB   F_{J}(m{p}) vs {L}: UNDECIDED (timeout), "
                  f"conf={st['conflicts']} {st['secs']:.1f}s")
    elif cmd == "lb":
        # one-sided LOWER bound: both-flanks SAT at L, witness re-verified,
        # says F_J(M) >= L+1.  Climbs while witnesses keep being found.
        p, J = int(sys.argv[2]), int(sys.argv[3])
        L = int(sys.argv[4])
        step = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        to = float(sys.argv[6]) if len(sys.argv) > 6 else None
        best = None
        while True:
            sat, st, w = decide(p, L, J, None, timeout=to, flanks="both")
            if not sat:
                break
            sp = verify_witness(p, L, J, w, None, "both")
            record(p, L, J, sat, st, w, None, sp, "both")
            best = (L, w, sp, st)
            print(f"LB   F_{J}(m{p}) >= {L + 1}  witness verified, "
                  f"spared={sp}, conf={st['conflicts']} {st['secs']:.1f}s",
                  flush=True)
            L += step
        if best:
            c, _ = column_of(best[1])
            print(f"BEST F_{J}(m{p}) >= {best[0] + 1}   column c={c}")
            print(f"  phases {best[1]}")
    elif cmd == "bisect":
        p, J = int(sys.argv[2]), int(sys.argv[3])
        lo = int(sys.argv[4])
        hi = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" \
            else None
        to = float(sys.argv[6]) if len(sys.argv) > 6 else None
        F, wit, ev, br = bisect(p, J, lo, hi, timeout=to)
        if F is None:
            print(f"RESULT F_{J}(m{p}) UNDECIDED; bracket {br} "
                  f"(>= {(br[0] or 0) + 1} from a verified witness)")
        else:
            print(f"RESULT F_{J}(m{p}) = {F}   ({len(ev)} probes)")
        # exhibit the extremal both-flanks witness at L = F-1
        if F is not None:
            sat, st, w = decide(p, F - 1, J, None, timeout=to, flanks="both")
            if sat:
                sp = verify_witness(p, F - 1, J, w, None, "both")
                record(p, F - 1, J, sat, st, w, None, sp, "both")
                c, _ = column_of(w)
                print(f"  extremal witness spared={sp} column c={c}")
                print(f"  phases {w}")
    elif cmd in ("ladder", "legal"):
        p, J = int(sys.argv[2]), int(sys.argv[3])
        lo = int(sys.argv[4]); hi = int(sys.argv[5])
        to = float(sys.argv[6]) if len(sys.argv) > 6 else None
        qp = next_prime(p) if cmd == "legal" else None
        fl = "both" if qp else "left"
        F, wit, ev = ladder(p, J, lo, hi, qp, timeout=to, flanks=fl)
        n_unsat = sum(1 for e in ev if e["result"] == "UNSAT")
        n_to = sum(1 for e in ev if e["result"] == "TIMEOUT")
        print(f"RESULT {'Q*' if qp else 'F'}_{J}(m{p}) "
              f"{'=' if not n_to else '>='} {F}   "
              f"({len(ev)} decisions, {n_unsat} UNSAT, {n_to} TIMEOUT)")
        print(f"  phases {wit}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
