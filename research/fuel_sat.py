"""Round 20 (mechanic): FUEL EXISTENCE BY CRT+SAT - N_k > 0 decided at
FULL period, no scan.

A valid k-tuple at step M -> q' is k CONSECUTIVE openings co-deletable by
one phase of gear q': their slot positions mod q' all lie in a tooth pair
{c, c + s}, s = 2u' mod q'.  Anchoring the first opening at 0, the k-1
gaps (g_1..g_{k-1}) must have cumulative sums whose residues mod q' stay
inside {0, s} or {0, -s}, each gap <= F(M), and the tuple must be k
consecutive openings - which is exactly a solve_window instance with the
spared positions FORCED at the cumulative sums.  Enumerate the finitely
many legal gap words, SAT each: any SAT witness (machine-verified) proves
N_k > 0; all-UNSAT proves N_k = 0 over the whole period.

Validation: 31->37 k=4 must be SAT exactly on the r11 words
(12,25,12)/(25,12,25) (N4 = 216, full-period), and 31->37 k=5 must be
all-UNSAT (kwin31 full period found zero k=5 tuples).

Usage: uv run --with python-sat python research/fuel_sat.py y qprime kmax
"""
import sys
import time
from itertools import product

sys.path.insert(0, "research")
sys.path.insert(0, ".")
from cov_sat import (gears_of, build_gap_instance, SatSolver, crt,
                     verify_window, MEASURED_F)

HOLES = {31: {54, 56, 57},
         37: {73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87},
         41: {84, 87, 89},   # from COV predict (partial; used only to skip)
         }


def allowed_values(res, q1, F, holes):
    return [v for v in range(1, F + 1)
            if v % q1 == res % q1 and v not in holes]


def legal_words(k, q1, s, F, holes):
    """All gap words (g_1..g_{k-1}) whose cumulative residues stay in
    {0, s} (both tooth-pair orientations covered by s and q1 - s)."""
    words = set()
    for step_set in ({0, s % q1}, {0, (-s) % q1}):
        # cumulative residues r_0 = 0, r_i in step_set
        def rec(prefix, cur):
            if len(prefix) == k - 1:
                words.add(tuple(prefix))
                return
            for r in step_set:
                d = (r - cur) % q1
                for v in allowed_values(d, q1, F, holes):
                    rec(prefix + [v], r)
        rec([], 0)
    return sorted(words)


def decide(y, q1, k):
    qs = gears_of(y)
    F = MEASURED_F[y]
    holes = HOLES.get(y, set())
    u = pow(6, -1, q1)
    s = (2 * u) % q1
    words = legal_words(k, q1, s, F, holes)
    print(f"step {y}->{q1}, k={k}: s = {s} mod {q1}, F({y}) = {F}, "
          f"{len(words)} legal gap words", flush=True)
    found = []
    for w in words:
        S = sum(w)
        sp = []
        acc = 0
        for g in w[:-1]:
            acc += g
            sp.append(acc)
        t0 = time.time()
        inst = build_gap_instance(S, qs, spared_budget=k - 2)
        if inst is None:
            print(f"  word {w}: UNSAT (endpoint) ({time.time()-t0:.0f}s)",
                  flush=True)
            continue
        clauses, phase, spare, pool = inst
        for p in sp:
            clauses.append([spare[p]])
        with SatSolver(bootstrap_with=clauses) as m:
            if m.solve():
                model = set(l for l in m.get_model() if l > 0)
                res, mod = [], []
                for (q, a), var in phase.items():
                    if var in model:
                        res.append((pow(6, -1, q) - a) % q)
                        mod.append(q)
                kk = crt(res, mod)
                assert verify_window(kk, S, sp, qs), (w, kk)
                # co-deletability check mod q1
                pos = [0] + sp + [S]
                rr = {p % q1 for p in pos}
                assert len(rr) <= 2, (w, rr)
                if len(rr) == 2:
                    d = (max(rr) - min(rr)) % q1
                    assert d in (s, (q1 - s) % q1), (w, rr, s)
                print(f"  word {w}: SAT  witness k = {kk}  (openings at "
                      f"k+{pos})  ({time.time()-t0:.0f}s)", flush=True)
                found.append((w, kk))
            else:
                print(f"  word {w}: UNSAT ({time.time()-t0:.0f}s)",
                      flush=True)
    if found:
        print(f"=> N_{k}({y}->{q1}) > 0 over the FULL period: "
              f"{len(found)} realizable words", flush=True)
    else:
        print(f"=> N_{k}({y}->{q1}) = 0 over the FULL period "
              f"(every legal word refuted)", flush=True)
    return found


if __name__ == "__main__":
    y, q1, kmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for k in range(kmax, kmax + 1) if len(sys.argv) < 5 else []:
        decide(y, q1, k)
