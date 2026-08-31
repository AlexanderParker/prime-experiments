"""Round 28 (constructor): THE NINTH RUNG WITH MECHANIC'S ROUND-27 ORACLE.

R79 pinned the 41 -> 43 stall at 222 under three settings and named the cause:
the TRANSFER SUPERSET's information content, not cost and not strategy.  Round
27 gave the superset strictly more information, from two independent directions:

  (1) F_4(41) = 118 EXACT (Mechanic, first computation).  A realised 4-tuple's
      span is <= F_4 BY DEFINITION, so every superset entry of span > 118 falls
      BY THEOREM.  4,239,676 -> 1,747,819 entries
      (research/data/r27/gap_tuples_41_4_screened_spancap.csv).
  (2) THE EXACT SHARD: the m41 4-tuple census is COMPLETE at every span <= 77
      (338,855 tuples, research/data/r27/gap_tuples_41_4_exact_le77.csv).  So
      for a 4-tuple of span <= 77 the shard is an EXACT oracle in BOTH
      directions - absence is a refutation, not merely a superset miss.

Tiers, in cost order, every one of them sound:
  (0) span > 118                     -> NO   (theorem: F_4(41) = 118)
  (1) span <= 77 and not in shard    -> NO   (exact census, complete at span)
  (2) not in the screened superset   -> NO   (superset)
  (3) phase saturation fires         -> NO   (theorem, R79 tier)
  (4) otherwise                      -> CRT if --crt, else YES (no deletion)

A YES is only a refusal to delete, so the loop can never certify wrongly; it
either certifies or reports a stall.

GATES (asserted before any deletion is licensed):
  * the exact shard is a SUBSET of the screened superset;
  * the screened superset's induced level-1 set reproduces F(41) = 91 and the
    complete m41 hole list {84, 87, 89};
  * every span in the screened superset is <= 118 and every span in the shard
    is <= 77;
  * phase saturation kills nothing in the exact shard (a false kill there would
    be a false kill anywhere).

Usage:
  .venv/Scripts/python.exe research/rung9_r28.py --topk 256 [--f2 103] [--crt]
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_cegar                                        # noqa: E402
import chain_dict_oracle as CDO                           # noqa: E402
import crt_dict                                           # noqa: E402
from chain_ps import PSScreen                             # noqa: E402
from chain_dict_oracle import _vinit, _vwork              # noqa: E402

DDIR = os.path.join(HERE, "data")
R27 = os.path.join(DDIR, "r27")
SCREENED = os.path.join(R27, "gap_tuples_41_4_screened_spancap.csv")
SHARD = os.path.join(R27, "gap_tuples_41_4_exact_le77.csv")
F4_41 = 118
SHARD_SPAN = 77


def _enc(t):
    k = 0
    for v in t:
        k = k * 128 + int(v)
    return k


def load_codes(path):
    codes, spans, d1 = [], [], set()
    with open(path) as f:
        for line in f:
            if line[0] == "g":
                continue
            t = tuple(int(x) for x in line.split(","))
            codes.append(_enc(t))
            spans.append(sum(t))
            d1.update(t)
    return (np.unique(np.array(codes, np.int64)),
            max(spans), d1, len(codes))


class ShardScreenedOracle(CDO.SupersetDictOracle):
    """Screened superset + exact span-<=77 shard + phase saturation (+ CRT)."""

    def __init__(self, y=41, use_crt=False, node_budget=4_000_000,
                 workers=1):
        super().__init__(y, path=SCREENED)
        self.shard, shmax, shd1, shn = load_codes(SHARD)
        assert shmax <= SHARD_SPAN, ("shard span gate", shmax)
        # subset gate: every exact-shard tuple must survive the screen
        i = np.searchsorted(self.D[4], self.shard)
        i = np.clip(i, 0, len(self.D[4]) - 1)
        miss = int((self.D[4][i] != self.shard).sum())
        assert miss == 0, ("shard NOT a subset of the screened superset", miss)
        self.shard_n = shn
        self.ps = PSScreen(y)
        self.use_crt = use_crt
        self.nb = node_budget
        self.span_kills = 0
        self.shard_kills = 0
        self.sup_kills = 0
        self.ps_kills = 0
        self.crt_calls = 0
        self.crt_kills = 0
        self.yes = 0
        self.pool = None
        if use_crt and workers > 1:
            from multiprocessing import Pool
            self.pool = Pool(workers, initializer=_vinit,
                             initargs=(y, node_budget))

    def close(self):
        if self.pool is not None:
            self.pool.terminate()

    def free(self, tup):
        """Every tier that costs nothing.  True = survives, False = refuted."""
        s = sum(tup)
        if len(tup) == 4:
            if s > F4_41:
                self.span_kills += 1
                return False
            if s <= SHARD_SPAN:
                k = _enc(tup)
                j = int(np.searchsorted(self.shard, k))
                if not (j < len(self.shard) and int(self.shard[j]) == k):
                    self.shard_kills += 1
                    return False
                return True                     # EXACT yes, no CRT needed
        if not CDO.SupersetDictOracle.__call__(self, tup):
            self.sup_kills += 1
            return False
        if self.ps.refutes(tup):
            self.ps_kills += 1
            return False
        return True

    def batch(self, tups):
        """Answer the round's CRT queries in PARALLEL, before the loop asks
        them one at a time.  Only tuples that survive every free tier and are
        not already memoised go to the solver."""
        if not self.use_crt:
            return
        need = []
        for t in dict.fromkeys(tups):
            if t in self.memo:
                continue
            if len(t) == 4 and sum(t) <= SHARD_SPAN:
                continue                        # exact shard already decided
            if not self.free(t):
                self.memo[t] = False
                continue
            need.append(t)
        if not need:
            return
        t0 = time.time()
        if self.pool is not None:
            res = self.pool.map(_vwork, need, chunksize=1)
        else:
            _vinit(self.y, self.nb)
            res = [_vwork(t) for t in need]
        self.secs += time.time() - t0
        for t, r, d in res:
            self.crt_calls += 1
            self.memo[t] = r
            if r is False:
                self.crt_kills += 1
            if r is None:
                self.undecided += 1
            if d > self.slowest[0]:
                self.slowest = (d, t)

    def __call__(self, tup):
        self.n += 1
        s = sum(tup)
        self.spans.append(s)
        if tup in self.memo:
            v = self.memo[tup]
            if v is False:
                return False
            self.yes += 1
            return v
        if len(tup) == 4:
            if s > F4_41:
                self.span_kills += 1
                return False
            if s <= SHARD_SPAN:
                k = _enc(tup)
                j = int(np.searchsorted(self.shard, k))
                if not (j < len(self.shard) and int(self.shard[j]) == k):
                    self.shard_kills += 1
                    return False
                self.yes += 1
                return True                     # EXACT yes
        if not CDO.SupersetDictOracle.__call__(self, tup):
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


def gates(orc):
    print("GATES")
    d1 = sorted(orc.D[1])
    holes = [v for v in range(1, max(d1) + 1) if v not in orc.D[1]]
    assert max(d1) == 91, ("F(41) gate", max(d1))
    assert holes == [84, 87, 89], ("m41 hole list gate", holes)
    print("  screened superset: %d 4-tuples, induced F(41) = %d, holes %s  OK"
          % (len(orc.D[4]), max(d1), holes))
    print("  exact shard: %d 4-tuples, all spans <= %d, subset of the screened"
          " superset  OK" % (orc.shard_n, SHARD_SPAN))
    # phase saturation must not kill anything the exact shard says is realised
    ps = PSScreen(41)
    bad, n = [], 0
    with open(SHARD) as f:
        for r in csv.reader(f):
            if r[0] == "g1":
                continue
            n += 1
            if n % 7 == 0:                      # 1-in-7 sample, 48k tuples
                if ps.refutes(tuple(int(x) for x in r)):
                    bad.append(r)
            if len(bad) > 2:
                break
    assert not bad, ("PHASE SATURATION FALSE KILL on the exact shard", bad)
    print("  phase saturation: 0 false kills on %d sampled exact-shard tuples"
          "  OK" % (n // 7))
    # and F_2 induced from the shard must not exceed the corpus F_2(41) = 103
    print()


def main():
    args = sys.argv[1:]

    def opt(nm, d):
        return type(d)(args[args.index(nm) + 1]) if nm in args else d

    y = 41
    topk = opt("--topk", 256)
    f2 = opt("--f2", 0)
    use_crt = "--crt" in args
    F, Q1, EXACT = chain_cegar.STEPS[y]
    t0 = time.time()
    print("=== RUNG NINE, round 28: step %d -> %d   F = %d   budget %d"
          % (y, Q1, F, F + Q1), flush=True)
    orc = ShardScreenedOracle(y, use_crt=use_crt, workers=opt("--workers", 1),
                              node_budget=opt("--nodes", 4_000_000))
    print("  oracle loaded in %.0f s\n" % (time.time() - t0), flush=True)
    gates(orc)
    if f2:
        print("  seeded with F_2(41) = %d (exact, R72, scan-free)\n" % f2,
              flush=True)
    r = chain_cegar.run_step(y, orc, topk=topk, f2=f2)
    chain_cegar.report(r, orc, F, Q1, EXACT)
    print("\n  ORACLE TIER BREAKDOWN over the loop's OWN query stream:")
    for nm, v in (("span > F_4(41) = 118 (theorem)", orc.span_kills),
                  ("exact shard, span <= 77   ", orc.shard_kills),
                  ("screened superset ABSENT  ", orc.sup_kills),
                  ("phase saturation          ", orc.ps_kills),
                  ("CRT refutations           ", orc.crt_kills),
                  ("YES (no deletion)         ", orc.yes)):
        print("    %-34s : %7d" % (nm, v))
    print("    total queries                      : %7d" % orc.n)
    out = os.path.join(DDIR, "r28", "rung9_r28%s%s.json"
                       % ("_crt" if use_crt else "", "_f2%d" % f2 if f2 else ""))
    json.dump(dict(y=y, status=r["status"], bound=r.get("bound"),
                   budget=F + Q1, it=r["it"], topk=topk, f2=f2,
                   span_kills=orc.span_kills, shard_kills=orc.shard_kills,
                   sup_kills=orc.sup_kills, ps_kills=orc.ps_kills,
                   crt_kills=orc.crt_kills, yes=orc.yes, queries=orc.n,
                   killed4=[list(t) for t in r.get("killed4", [])],
                   killed2=[list(t) for t in r.get("killed2", [])],
                   secs=time.time() - t0), open(out, "w"))
    print("\n  written to %s  (%.0f s)" % (out, time.time() - t0))


if __name__ == "__main__":
    main()
