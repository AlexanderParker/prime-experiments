"""Round 26 (constructor): THE MINIMAL CERTIFICATE OF A RUNG.

R67 item (ii) asked for a bound on the CEGAR QUERY COUNT.  Round 26 finds the
question is asked about the wrong object: the query count is a property of the
SEARCH STRATEGY, not of the step.  Measured at 37 -> 41 the same rung costs
5,771 queries at topk = 256 with the two-gap number given and 12,695 at
topk = 256 without it; round 25's topk = 1 gives different numbers again.

The strategy-free object is the CERTIFICATE: the SET OF REALISABILITY
REFUTATIONS that, applied to the machine-free system MF_4(F, q'), brings its
max-plus closure to or below the budget F + q'.  That set is what a kernel
proof would have to carry, and its minimum size is a property of the step
alone.  This script computes a MINIMAL certificate (minimal in the sense of
irredundant: no member can be removed) by reverse deletion:

    start from the greedy loop's deletion set K (every member an exhaustive
    CRT/census refutation);
    for each t in K in turn, restore t and re-close; if the bound is still
    <= budget, t was redundant - drop it permanently; else put it back.

Restoring an edge or state can only RAISE the closure, so the test is exact
and the result is an irredundant certificate.  (Irredundant, not necessarily
minimum: finding the minimum is a hitting-set problem.  The number reported is
therefore an UPPER bound on the true minimum and a huge improvement on the
greedy count.)

Usage:  python research/cert_minimise.py --steps 13,17,19,23,29
        python research/cert_minimise.py --steps 31 --order span
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_cegar                                       # noqa: E402
import chain_dict_oracle                                 # noqa: E402
from chain_cegar import close_sorted, NEG                # noqa: E402
from machinefree_cert import build_mf_edges              # noqa: E402

DDIR = os.path.join(HERE, "data")


class System:
    """MF_4 with a deletable set of value 4-tuples and value pairs."""

    def __init__(self, F, q1, f2=0):
        S, esrc, edst, ew, Rs, Ls, tup = build_mf_edges(F, q1, 35, 4)
        self.S = S
        self.Rs0, self.Ls0 = Rs.copy(), Ls.copy()
        if f2:
            okst = (Ls + Rs) <= f2
            keep = okst[esrc] & okst[edst]
            esrc, edst, ew, tup = (esrc[keep], edst[keep], ew[keep],
                                   tup[keep])
            self.Ls0 = np.where(okst, self.Ls0, NEG)
            self.Rs0 = np.where(okst, self.Rs0, NEG)
        order = np.argsort(esrc, kind="stable")
        self.esrc, self.edst = esrc[order], edst[order]
        self.ew, self.tup = ew[order], tup[order]
        self.usrc, self.starts = np.unique(self.esrc, return_index=True)
        t64 = self.tup.astype(np.int64)
        k4 = ((t64[:, 0] * 128 + t64[:, 1]) * 128 + t64[:, 2]) * 128 + \
            t64[:, 3]
        self.by4 = {}
        for i, k in enumerate(k4.tolist()):
            self.by4.setdefault(k, []).append(i)
        self.by4 = {k: np.array(v, np.int64) for k, v in self.by4.items()}
        pk = (self.Ls0 * 128 + self.Rs0).astype(np.int64)
        self.bypair = {}
        for i, k in enumerate(pk.tolist()):
            if self.Ls0[i] > NEG // 2:
                self.bypair.setdefault(k, []).append(i)
        self.bypair = {k: np.array(v, np.int64) for k, v in self.bypair.items()}

    def bound(self, kill4, kill2):
        live = np.ones(len(self.esrc), bool)
        Ls, Rs = self.Ls0.copy(), self.Rs0.copy()
        dead = np.zeros(self.S, bool)
        for p in kill2:
            js = self.bypair.get(p[0] * 128 + p[1])
            if js is not None and len(js):
                dead[js] = True
                Ls[js] = NEG
                Rs[js] = NEG
        if dead.any():
            live &= ~(dead[self.esrc] | dead[self.edst])
        for t in kill4:
            js = self.by4.get(((t[0] * 128 + t[1]) * 128 + t[2]) * 128 + t[3])
            if js is not None:
                live[js] = False
        hh, b = close_sorted(self.S, self.edst, self.ew, live, Rs, Ls,
                             self.usrc, self.starts)
        return None if hh is None else b


def minimise(y, path=None, f2=0, order="greedy", verbose=True):
    F, q1, EXACT = chain_cegar.STEPS[y]
    budget = F + q1
    rec = json.load(open(path)) if path else None
    if rec is None:
        # run the greedy loop first (census oracle when one exists)
        if y in chain_dict_oracle.DICT_CSV:
            orc = chain_dict_oracle.ExactDictOracle(y)
        else:
            orc = chain_cegar.CRTOracle(y, 2_000_000)
        r = chain_cegar.run_step(y, orc, topk=1, f2=f2, verbose=False)
        assert r["status"] == "CERTIFIED", (y, r["status"])
        rec = dict(killed4=[list(t) for t in r["killed4"]],
                   killed2=[list(t) for t in r["killed2"]],
                   q4=r["q4"], q2=r["q2"])
    k4 = [tuple(t) for t in rec["killed4"]]
    k2 = [tuple(t) for t in rec["killed2"]]
    sysm = System(F, q1, f2=f2)
    b0 = sysm.bound(k4, k2)
    assert b0 is not None and b0 <= budget, (y, b0, budget)
    items = [("4", t) for t in k4] + [("2", t) for t in k2]
    if order == "span":                     # try the biggest spans first
        items.sort(key=lambda it: -sum(it[1]))
    t0 = time.time()
    cur4, cur2 = set(k4), set(k2)
    dropped = 0
    for i, (kind, t) in enumerate(items):
        if kind == "4":
            cur4.discard(t)
        else:
            cur2.discard(t)
        b = sysm.bound(cur4, cur2)
        if b is None or b > budget:
            (cur4 if kind == "4" else cur2).add(t)       # needed - restore
        else:
            dropped += 1
        if verbose and (i + 1) % 250 == 0:
            print("      %d/%d tested, %d dropped, %.0fs"
                  % (i + 1, len(items), dropped, time.time() - t0), flush=True)
    b1 = sysm.bound(cur4, cur2)
    assert b1 is not None and b1 <= budget, (y, b1, budget)
    return dict(y=y, q1=q1, F=F, budget=budget, bound=b1,
                greedy=len(items), greedy4=len(k4), greedy2=len(k2),
                minimal=len(cur4) + len(cur2), min4=len(cur4),
                min2=len(cur2), queries=rec.get("q4", 0) + rec.get("q2", 0),
                cert4=sorted(cur4), cert2=sorted(cur2),
                secs=time.time() - t0)


def main():
    args = sys.argv[1:]
    ys = ([int(x) for x in args[args.index("--steps") + 1].split(",")]
          if "--steps" in args else [13, 17, 19, 23, 29])
    order = args[args.index("--order") + 1] if "--order" in args else "greedy"
    f2 = int(args[args.index("--f2") + 1]) if "--f2" in args else 0
    path = args[args.index("--path") + 1] if "--path" in args else None
    rows = []
    for y in ys:
        print("\n=== minimising the certificate at %d -> %d" % (y,
              chain_cegar.STEPS[y][1]), flush=True)
        r = minimise(y, path=path, f2=f2, order=order)
        print("   greedy deletions %d (%d + %d)  ->  IRREDUNDANT %d "
              "(%d + %d)   bound %d <= %d   %.0fs"
              % (r["greedy"], r["greedy4"], r["greedy2"], r["minimal"],
                 r["min4"], r["min2"], r["bound"], r["budget"], r["secs"]),
              flush=True)
        rows.append(r)
    print("\n\nTHE MINIMAL CERTIFICATE OF EACH RUNG")
    print("  M    q'   budget  queries  greedy dels  irredundant  ratio")
    for r in rows:
        print("  %-4d %-4d %6d  %7d  %11d  %11d  %5.1fx"
              % (r["y"], r["q1"], r["budget"], r["queries"], r["greedy"],
                 r["minimal"], r["greedy"] / max(1, r["minimal"])))
    json.dump([{k: v for k, v in r.items() if k not in ("cert4", "cert2")}
               for r in rows],
              open(os.path.join(DDIR, "r26_cert_min.json"), "w"))
    for r in rows:
        json.dump(dict(y=r["y"], cert4=r["cert4"], cert2=r["cert2"]),
                  open(os.path.join(DDIR, "r26_cert_%d.json" % r["y"]), "w"))
    print("\ncertificates written to research/data/r26_cert_*.json")


if __name__ == "__main__":
    main()
