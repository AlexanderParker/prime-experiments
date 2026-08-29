"""Round 27 (constructor): PHASE SATURATION AS A FREE ORACLE TIER, and the
ninth rung retried with it.

R72 left 41 -> 43 uncertified and diagnosed it precisely: the rung is
ORACLE-bound.  The superset-only oracle stalls at bound 222 against the budget
134 (too inflated at arity 4 to finish); the exact-CRT hybrid is correct but
each machine-41 refutation costs seconds, and the round-26 run was cancelled on
cost.

This round adds a THIRD tier that costs nothing at all.  Mechanic's round-26
phase-saturation theorem refutes a gap tuple outright when its prefix-sum set
X has no translate inside some gear's exposed set:

    gear g blocks slots k = +-c_g (mod g), c_g = 6^{-1} mod g, so an
    occurrence needs some k0 with k0 + X inside E_g = Z_g \\ {+-c_g}.

That is a few dozen integer operations per tuple, it is EXACT (a refutation is
a proof), it is machine-free apart from the gear list, and - unlike the
superset - it does not come from any scan.  So the oracle becomes

    (1) superset says ABSENT  -> NO, free          (sound: superset)
    (2) phase saturation fires -> NO, free          (sound: theorem)
    (3) otherwise              -> CRT, exact but expensive   [--crt]

and every deletion any tier licenses is a genuine refutation.

Two things are measured here, both of which the round needed:
  * how many of the loop's ACTUAL queries tier (2) answers for free - not how
    many of the 4.2M dictionary entries it screens, which is a different and
    much less useful number;
  * whether tiers (1)+(2) alone, with NO solver call whatever, bring the
    41 -> 43 closure under the budget.

Usage:
  .venv/Scripts/python.exe research/chain_ps.py --step 41 --topk 256 [--crt]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_cegar                                       # noqa: E402
import chain_dict_oracle as CDO                          # noqa: E402
import crt_dict                                          # noqa: E402

DDIR = os.path.join(HERE, "data")


def _exposed(g):
    c = pow(6, -1, g)
    return frozenset(r for r in range(g) if r != c % g and r != (-c) % g)


class PSScreen:
    """Mechanic's phase-saturation refuter, gears of M."""

    def __init__(self, y):
        self.gears = [p for p in range(5, y + 1)
                      if all(p % d for d in range(2, int(p ** .5) + 1))]
        self.E = {g: _exposed(g) for g in self.gears}
        self.hits = 0
        self.calls = 0

    def refutes(self, tup):
        self.calls += 1
        X, s = [0], 0
        for v in tup:
            s += v
            X.append(s)
        for g in self.gears:
            Eg = self.E[g]
            xs = {x % g for x in X}
            if len(xs) > g - 2:
                self.hits += 1
                return True
            ok = False
            for t in range(g):
                if all((t + x) % g in Eg for x in xs):
                    ok = True
                    break
            if not ok:
                self.hits += 1
                return True
        return False


class ScreenedSupersetOracle(CDO.SupersetDictOracle):
    """superset NO -> NO; else phase saturation; else (optionally) CRT."""

    def __init__(self, y, use_crt=False, node_budget=4_000_000):
        super().__init__(y)
        self.ps = PSScreen(y)
        self.use_crt = use_crt
        self.nb = node_budget
        self.ps_kills = 0
        self.sup_kills = 0
        self.crt_calls = 0
        self.crt_kills = 0
        self.yes = 0

    def __call__(self, tup):
        if not super().__call__(tup):
            self.sup_kills += 1
            return False
        if self.ps.refutes(tup):
            self.ps_kills += 1
            return False
        if self.use_crt:
            self.crt_calls += 1
            t0 = time.time()
            try:
                ok = crt_dict.realised(self.y, tup, node_budget=self.nb)
            except Exception:
                self.undecided += 1
                self.yes += 1
                return True
            self.secs += time.time() - t0
            if not ok:
                self.crt_kills += 1
                return False
        self.yes += 1
        return True


def main():
    args = sys.argv[1:]

    def opt(n, d):
        return type(d)(args[args.index(n) + 1]) if n in args else d

    y = opt("--step", 41)
    topk = opt("--topk", 256)
    use_crt = "--crt" in args
    F, Q1, EXACT = chain_cegar.STEPS[y]
    print("=== step %d -> %d   F = %d   budget %d   (tiers: superset, phase "
          "saturation%s)" % (y, Q1, F, F + Q1, ", CRT" if use_crt else ""),
          flush=True)
    t0 = time.time()
    orc = ScreenedSupersetOracle(y, use_crt=use_crt)
    print("  superset levels: |D_1| = %d, |D_2| = %d, |D_3| = %d, |D_4| = %d"
          "   F = %d, F_2 <= %d   (%.0f s to load)"
          % (len(orc.D[1]), len(orc.D[2]), len(orc.D[3]), len(orc.D[4]),
             orc.F, orc.F2, time.time() - t0), flush=True)
    # soundness gate: the screen must not refute anything the exact m37
    # dictionary says is realised (a false kill would invalidate every rung).
    import csv
    ps37 = PSScreen(37)
    bad, n = [], 0
    with open(os.path.join(DDIR, "gap_tuples_37_4.csv")) as f:
        for r in csv.reader(f):
            if r[0] == "g1":
                continue
            t = tuple(int(x) for x in r)
            n += 1
            if ps37.refutes(t):
                bad.append(t)
            if len(bad) > 3:
                break
    assert not bad, ("PHASE SATURATION FALSE KILL", bad[:3])
    print("  soundness gate: 0 false kills on all %d realised m37 4-tuples"
          % n, flush=True)

    f2 = opt("--f2", 0)
    if f2:
        print("  seeded with the two-gap number F_2(%d) = %d (R72, exact,"
              " scan-free and non-circular)" % (y, f2), flush=True)
    r = chain_cegar.run_step(y, orc, topk=topk, f2=f2)
    chain_cegar.report(r, orc, F, Q1, EXACT)
    print("\n  ORACLE TIER BREAKDOWN over the loop's own query stream:")
    tot = orc.sup_kills + orc.ps_kills + orc.crt_kills + orc.yes
    print("    superset ABSENT      : %7d  (free)" % orc.sup_kills)
    print("    phase saturation     : %7d  (free)  -- %.1f%% of all NOs "
          "that the superset could not answer"
          % (orc.ps_kills,
             100.0 * orc.ps_kills / max(1, orc.ps_kills + orc.crt_kills)))
    print("    CRT refutations      : %7d  (%.0f s)" % (orc.crt_kills,
                                                        orc.secs))
    print("    YES (no deletion)    : %7d" % orc.yes)
    print("    total queries        : %7d" % tot)
    out = os.path.join(DDIR, "chain_ps_%d%s%s.json"
                       % (y, "_crt" if use_crt else "",
                          "_f2%d" % f2 if f2 else ""))
    json.dump(dict(y=y, status=r["status"], bound=r.get("bound"),
                   budget=F + Q1, it=r["it"], topk=topk,
                   sup_kills=orc.sup_kills, ps_kills=orc.ps_kills,
                   crt_kills=orc.crt_kills, yes=orc.yes,
                   killed4=[list(t) for t in r.get("killed4", [])],
                   killed2=[list(t) for t in r.get("killed2", [])],
                   secs=time.time() - t0), open(out, "w"))
    print("\n  written to %s   (%.0f s total)" % (out, time.time() - t0))


if __name__ == "__main__":
    main()
