"""sr_identities.py -- the exact identities the gap-multiplicity function satisfies.

Machines m5..m23 sieved whole over a full period. For each machine:
  (a) sum_v m(v)     = prod (q - 2)
  (b) sum_v v m(v)   = prod q
  (c) m(v) even for v >= 2, m(1) odd            (parity theorem, docs/proofs/03 (e))
  (d) m(1)           = prod (q - 4)
  (e) m(2)           = 2 . 4 . prod_{q>=11} (q - 4)
  (f) A(v) := #{k : k, k+v both open} = prod_q c_q(v)   (the autocorrelation sum rule;
      recorded as depth-sum-identity = Holt arXiv:2502.20470 Cor. 1 -- instrument check)
  and the budget shares of the uncoupled sizes.

Writes results/sr_identities.txt
"""
import os, sys, json
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
PR = [5, 7, 11, 13, 17, 19, 23]


def u_of(g):
    return pow(6, -1, g)


def teeth(g):
    u = u_of(g)
    return (u % g, (-u) % g)


def c_local(q, v):
    """#{r mod q : r and r+v both open for gear q}."""
    u = u_of(q)
    s = {u % q, (-u) % q, (u - v) % q, ((-u) - v) % q}
    return q - len(s)


def coupling(q, v):
    """'pad' if q|v, 'letter' if v = +-d mod q, else None."""
    if v % q == 0:
        return "pad"
    d = (2 * u_of(q)) % q
    if v % q == d % q or v % q == (-d) % q:
        return "letter"
    return None


def sieve(gears):
    P = 1
    for g in gears:
        P *= g
    blocked = np.zeros(P, dtype=bool)
    for g in gears:
        for t in teeth(g):
            blocked[t::g] = True
    return P, ~blocked


def main():
    lines = []
    W = lines.append
    for i in range(len(PR)):
        gears = PR[: i + 1]
        P, op = sieve(gears)
        idx = np.flatnonzero(op)
        N = idx.size
        gaps = np.diff(np.concatenate([idx, [idx[0] + P]]))
        F = int(gaps.max())
        bc = np.bincount(gaps)
        m = {int(v): int(bc[v]) for v in np.flatnonzero(bc)}
        prod_q = 1
        prod_q2 = 1
        prod_q4 = 1
        for g in gears:
            prod_q *= g
            prod_q2 *= g - 2
            prod_q4 *= g - 4
        tot = sum(m.values())
        totlen = sum(v * c for v, c in m.items())
        par_bad = [v for v in m if v >= 2 and m[v] % 2 == 1]
        m2pred = None
        if 5 in gears and 7 in gears:
            m2pred = 2 * 4
            for g in gears:
                if g >= 11:
                    m2pred *= g - 4
        W(f"=== machine {{5..{gears[-1]}}}  P={P}  N={N}  F={F}  |Spec|={len(m)}")
        W(f"  (a) sum m(v)   = {tot}  vs prod(q-2) = {prod_q2}   {'OK' if tot == prod_q2 else 'FAIL'}")
        W(f"  (b) sum v m(v) = {totlen}  vs P = {P}   {'OK' if totlen == P else 'FAIL'}")
        W(f"  (c) parity: odd-count sizes >=2: {par_bad if par_bad else 'none'}; m(1)={m.get(1,0)} "
          f"({'odd' if m.get(1,0)%2 else 'EVEN'})   {'OK' if not par_bad and m.get(1,0)%2==1 else 'FAIL'}")
        W(f"  (d) m(1) = {m.get(1,0)} vs prod(q-4) = {prod_q4}   {'OK' if m.get(1,0)==prod_q4 else 'FAIL'}")
        if m2pred is not None:
            W(f"  (e) m(2) = {m.get(2,0)} vs 8.prod_{{q>=11}}(q-4) = {m2pred}   "
              f"{'OK' if m.get(2,0)==m2pred else 'FAIL'}  ratio m(2)/m(1) = {m.get(2,0)/max(m.get(1,0),1):.6f}")
        # (f) autocorrelation
        bad = []
        for v in range(1, F + 1):
            A = int(np.count_nonzero(op & np.roll(op, -v)))
            pr = 1
            for g in gears:
                pr *= c_local(g, v)
            if A != pr:
                bad.append((v, A, pr))
        W(f"  (f) A(v) = prod c_q(v) for v=1..{F}: {'0 exceptions' if not bad else bad}")
        # uncoupled sizes and their budget share
        unc = []
        for v in range(2, F + 1):
            if all(coupling(g, v) is None for g in gears):
                unc.append(v)
        share_n = sum(m.get(v, 0) for v in unc) / tot
        share_p = sum(v * m.get(v, 0) for v in unc) / P
        W(f"  uncoupled sizes 2..F: {unc}")
        W(f"    counts: {[m.get(v,0) for v in unc]}")
        W(f"    share of sum m(v): {share_n:.3e}    share of sum v m(v): {share_p:.3e}")
        # full spectrum for later use
        W("  spectrum: " + " ".join(f"{v}:{m[v]}" for v in sorted(m)))
        sys.stdout.flush()
        json.dump({str(k): v for k, v in m.items()},
                  open(os.path.join(OUT, f"spec_m{gears[-1]}.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "sr_identities.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
