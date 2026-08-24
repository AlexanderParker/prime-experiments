"""Round 21 (mechanic): COV-COUNT - exact pattern COUNTS (not just decisions)
by projected model enumeration over the COV-SAT phase encoding.

The bijection that makes counting exact: a phase vector (a_q)_q corresponds
to exactly ONE slot k mod P by CRT, and every k gives one phase vector.  So

    #{ k in [0,P) : X exposed, Y blocked at k }  =  # models of the phase CNF
                                                    projected to phase vars.

Counting = enumerate models, adding a blocking clause over the PHASE
variables only after each (projection makes auxiliary-variable multiplicity
irrelevant).  Every model is CRT'd to its k and machine-verified by assert.
This is exact and cheap precisely when the count is small - which is the
regime Constructor's renewal-ladder zero certificates live in (count = 0 is
a single UNSAT, no counting at all; the extreme patterns near the (D)
boundary have counts 0..a few hundred).  For abundant patterns the tool
reports ">cap" and the ladder's IE bounds remain the right instrument.

Pattern language (Constructor's #(X exposed, Y blocked), renewal ladder R38):
  window span S anchored at 0; X = exposed positions (always contains 0, S
  unless --loose); Y = blocked positions.  Two modes:
    strict (default): Y = all of (0,S) minus X  (word occurrences, gaps)
    --blocked i,j,..: Y = exactly these; all other interior positions FREE
                      (the ladder's own object)
Usage:
  python cov_count.py count y S --open 8            # word (8,15) at m19: X={0,8,23}
  python cov_count.py count y S                     # gap of exactly S
  python cov_count.py count y S --blocked 1,2,5     # ladder pattern, rest free
  python cov_count.py validate                      # anchors vs full-period census
  [--cap N] enumeration cap (default 100000)
Run with the .venv-sat python (or uv run --with python-sat).
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
from cov_sat import gears_of, crt, verify_window, blocked_by  # noqa: E402
from pysat.solvers import Cadical153 as SatSolver  # noqa: E402
from pysat.formula import IDPool  # noqa: E402


def build_pattern(S, qs, opens, blocked=None):
    """CNF: positions in X = {0, S} u opens exposed; positions in Y blocked
    (Y = all interior minus X if blocked is None); other positions free.
    Returns (clauses, phase) or None (some gear cannot avoid X)."""
    X = {0, S} | set(opens)
    Y = set(blocked) if blocked is not None else set(range(1, S)) - X
    assert not (X & Y)
    pool = IDPool()
    clauses = []
    phase = {}
    cover = {i: [] for i in Y}
    for q in qs:
        u = pow(6, -1, q)
        s = (-2 * u) % q
        forb = set()
        for x in X:
            forb.add(x % q)
            forb.add((x - s) % q)
        vs = []
        for a in range(q):
            if a in forb:
                continue
            var = pool.id(("x", q, a))
            phase[(q, a)] = var
            vs.append(var)
            b = (a + s) % q
            for i in Y:
                r = i % q
                if r == a or r == b:
                    cover[i].append(var)
        if not vs:
            return None
        clauses.append(vs)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                clauses.append([-vs[i], -vs[j]])
    for i in Y:
        if not cover[i]:
            return None
        clauses.append(cover[i])
    return clauses, phase


def verify_pattern(k, S, opens, blocked, qs):
    """Direct machine check of the loose pattern at slot k."""
    X = {0, S} | set(opens)
    Y = set(blocked) if blocked is not None else set(range(1, S)) - X
    for i in X:
        if any(blocked_by(k + i, q) for q in qs):
            return False
    for i in Y:
        if not any(blocked_by(k + i, q) for q in qs):
            return False
    return True


def count_pattern(y, S, opens=(), blocked=None, cap=100_000, quiet=False):
    """Exact count per period of machine y, by projected enumeration.
    Returns (count, witnesses[:10], solver_calls) - count == cap+ means
    aborted at cap (reported as lower bound)."""
    qs = gears_of(y)
    inst = build_pattern(S, qs, opens, blocked)
    if inst is None:
        return 0, [], 0            # structurally impossible (endpoint parity)
    clauses, phase = inst
    n = 0
    wits = []
    calls = 0
    with SatSolver(bootstrap_with=clauses) as m:
        while n < cap:
            calls += 1
            if not m.solve():
                return n, wits, calls
            model = set(l for l in m.get_model() if l > 0)
            pos = [var for var in phase.values() if var in model]
            res, mod = [], []
            for (q, a), var in phase.items():
                if var in model:
                    res.append((pow(6, -1, q) - a) % q)
                    mod.append(q)
            k = crt(res, mod)
            assert verify_pattern(k, S, opens, blocked, qs), (y, S, k)
            if len(wits) < 10:
                wits.append(k)
            n += 1
            m.add_clause([-v for v in pos])     # block this phase vector
    return n, wits, calls


# Anchors: (y, S, opens, expected full-period count, source)
ANCHORS = [
    (19, 23, (8,), 31, "r17 flank census: word (8,15) at 19->23, occ 31"),
    (23, 31, (10,), 138, "r17: word (10,21) at 23->29/31, occ 138"),
    (29, 41, (10, 31), 4, "r11+r17: word (10,21,10) at 29->31, exactly 4"),
    (29, 52, (21, 31), 0, "r17: word (21,10,21) at 29->31, ZERO occurrences"),
    (23, 29, (), 6, "C12 gap-tail: hist_23[29] = 6"),
    (19, 24, (), 0, "C14 hole list: 24 is a hole of machine 19"),
    (23, 24, (), 0, "C14 hole list: 24 is a hole of machine 23"),
]


def validate():
    print("COV-COUNT validation: exact enumeration vs full-period censuses")
    ok = True
    for y, S, opens, want, src in ANCHORS:
        t0 = time.time()
        n, wits, calls = count_pattern(y, S, opens)
        good = n == want
        ok &= good
        print(f"  m{y} S={S} opens={list(opens)}: count={n} want={want} "
              f"({calls} solver calls, {time.time()-t0:.1f}s) "
              f"{'AGREES' if good else 'MISMATCH'}  [{src}]"
              + (f"  wits {wits[:4]}" if wits else ""), flush=True)
    # known addresses for (10,21,10): r17 lists 220171102, 406081827,
    # 672200337, 858111062
    n, wits, _ = count_pattern(29, 41, (10, 31))
    known = {220171102, 406081827, 672200337, 858111062}
    got = set(w % 1_078_282_205 if False else w for w in wits)
    addr_ok = got == known
    ok &= addr_ok
    print(f"  (10,21,10) witness addresses {sorted(got)} == r17 census: "
          f"{'AGREES' if addr_ok else 'MISMATCH ' + str(sorted(known))}")
    print(f"=> {'ALL ANCHORS AGREE' if ok else 'MISMATCH - do not use'}")
    return ok


def main():
    args = sys.argv[1:]
    def popopt(name, default=None, cast=str):
        if name in args:
            i = args.index(name)
            v = cast(args[i + 1])
            del args[i:i + 2]
            return v
        return default
    cap = popopt("--cap", 100_000, int)
    opens = popopt("--open", "", str)
    blocked = popopt("--blocked", None, str)
    opens = tuple(int(x) for x in opens.split(",") if x)
    blocked = [int(x) for x in blocked.split(",")] if blocked else None
    if args[0] == "validate":
        validate()
        return
    y, S = int(args[1]), int(args[2])
    t0 = time.time()
    n, wits, calls = count_pattern(y, S, opens, blocked, cap)
    tag = f"m{y} S={S} opens={list(opens)}" + (
        f" blocked={blocked}" if blocked else " (strict)")
    if n >= cap:
        print(f"{tag}: count >= {cap} (cap hit; use IE/ladder bounds) "
              f"({calls} calls, {time.time()-t0:.0f}s)")
    else:
        print(f"{tag}: count = {n} EXACT per period "
              f"({calls} solver calls, {time.time()-t0:.0f}s)"
              + (f"  first witnesses {wits}" if wits else "  (ZERO CERT)"))


if __name__ == "__main__":
    main()
