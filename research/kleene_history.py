"""Round 23 (constructor): THE HISTORY LADDER - how much bounded state does the
Kleene certificate need, and what exactly is the J=5 object at 29 -> 31?

Background.  R46 wrote F(M+q') as a max-plus Kleene star on states
(opening i, tooth s) with K[(i,s),(i+1,s')] = d_i when d_i qualifies and the
T3 transition is legal; (D) at alpha = 3 is L^T (x) K* (x) R <= F(M) + q'.
R47 replaced the opening by a BOUNDED class (last gap value; corridor phase
mod 35/385/5005 with the value) and found the class-level closure CERTIFIES
through 23 -> 29 but FAILS at 29 -> 31: bounds 99 / 99 / 91 against a budget
of 74 (exact 58).  Mechanic's marked qualifying spectrum fails at the same
step, localised at J = 5 (85 against 74, true 71).

This script builds the ladder R47 named as its own next construct:

    ABSTRACTION A_m(mod):  state = the last (m-1) gap VALUES ending at the
    current opening (packed base 64), optionally with the corridor phase of
    the opening mod `mod`, and the tooth.
      * an EDGE state -> state' exists iff the m-tuple of consecutive gaps it
        encodes is REALISED somewhere in the period, and the letter class of
        the middle gap makes the T3 transition legal;
      * edge weight = d_i, which is a component of the state, so the weight
        is EXACT (R47's crudest choice - max over sources - disappears);
      * base R(state) = d_i (last digit), EXACT;
      * flank L(state) = d_{i-1} (second digit), EXACT for m >= 3.
    m = 2 reproduces R47's "value only" state (there L must be maxed over the
    class, because d_{i-1} is not in the state).

    Every real chain maps to an abstract walk of the same weight, so the
    class-level closure is a SOUND upper bound on F(M+q') at every m, and it
    is non-increasing in m.  A_m is the m-POINT RELAXATION of the joint
    realizability that R41 named the counting boundary, so A_m is nilpotent
    (has a finite closure) exactly when the m-point relaxation refutes the
    infinite alternating word - i.e. when m > A_relax(M) of R45.

Also computed in the SAME streaming pass, all exact and full period:
  * F(M), the exact Kleene value L^T (x) K* (x) R, and its layer maxima
    (layer k = window of k+2 gaps with k qualifying alternating interiors,
    i.e. qualmax_{k+2});
  * the complete inventory of the deepest realised chains (word, flanks,
    window sum, address) - the exact "J = 5" object at 29 -> 31;
  * the SIZE-FLOOR spectrum Q_J(M; 2u') for J = 2..8 (Mechanic's object:
    max sum of J consecutive gaps whose J-1 interiors are all >= 2u'), with
    a witness for each J.
So one pass produces the whole relaxation ladder at a step:

    qualmax_J  (residue + T3)  <=  Q_J  (size floor)  <=  A_m bounds.

Usage: uv run python research/kleene_history.py y [y ...]
         [--seg N] [--specs m:mod,m:mod,...]  (mod 0 = value only)
"""
import os
import sys
import time
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91}
OVL = 24            # openings carried across a segment boundary
MAXK = 8            # max chain links tracked (assert if ever saturated)
BASE = 64           # gap values are < 64 at every machine used here
JMAX = 8            # depth of the size-floor spectrum


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def next_prime(y):
    p = y + 1
    while not all(p % d for d in range(2, int(p ** 0.5) + 1)):
        p += 1
    return p


class HistSystem:
    """Sound class-level max-plus system for the m-point history abstraction.

    Collected by streaming: the set of realised state keys and the set of
    realised edges.  Both weight and flank are functions of the key, so
    nothing has to be maximised over a class when m >= 3.
    """

    def __init__(self, m, mod, q1):
        self.m = m
        self.mod = mod                       # 0 / None = value only
        self.q1 = q1
        self.hspan = BASE ** (m - 1)
        self.keyspan = self.hspan * (mod if mod else 1)
        assert self.keyspan < 2 ** 31, (m, mod, self.keyspan)
        self.name = ("A_%d value only" % m if not mod
                     else "A_%d + phase mod %d" % (m, mod))
        self.seen = np.zeros(self.keyspan, bool)
        self.edges = set()
        self.Lmax = np.zeros(self.keyspan, np.int32) if m == 2 else None

    def keys(self, d, ph, i0, i1):
        """key for every index in [i0-1, i1+1] (needs one on each side)."""
        m, mod = self.m, self.mod
        k = np.zeros(i1 - i0 + 3, np.int64)
        sl = slice(i0 - 1, i1 + 2)
        for j in range(m - 1):
            k += d[i0 - 1 - j:i1 + 2 - j].astype(np.int64) * (BASE ** j)
        if mod:
            k += ph[sl].astype(np.int64) * self.hspan
        return k

    def absorb(self, d, cls, ph, i0, i1):
        k = self.keys(d, ph, i0, i1)          # index t <-> gap index i0-1+t
        own = k[1:i1 - i0 + 1]                # indices i0 .. i1-1
        self.seen[own] = True
        if self.Lmax is not None:
            np.maximum.at(self.Lmax, own,
                          d[i0 - 1:i1 - 1].astype(np.int32))
        c = cls[i0:i1]
        for s, leg, land in (
                (0, (c == 0) | (c == 1), np.where(c == 0, 0, 1)),
                (1, (c == 0) | (c == -1), np.where(c == 0, 1, 0))):
            sel = np.flatnonzero(leg)
            if not len(sel):
                continue
            src = own[sel] * 2 + s
            dst = k[2:i1 - i0 + 2][sel] * 2 + land[sel]
            eid = src.astype(np.int64) * (1 << 32) + dst
            self.edges.update(np.unique(eid).tolist())

    def close(self, budget, exact, layers=True):
        ks = sorted({e >> 32 for e in self.edges} |
                    {e & 0xFFFFFFFF for e in self.edges} |
                    {int(k) * 2 + s for k in np.flatnonzero(self.seen)
                     for s in (0, 1)})
        idx = {k: i for i, k in enumerate(ks)}
        S = len(ks)
        keyof = np.array([k >> 1 for k in ks], np.int64)
        d0 = (keyof % BASE)                                  # d_i
        d1 = (keyof // BASE) % BASE                           # d_{i-1}
        Rs = d0.astype(np.int64)
        if self.m == 2:
            Ls = self.Lmax[keyof].astype(np.int64)
        else:
            Ls = d1.astype(np.int64)
        esrc = np.array([idx[e >> 32] for e in self.edges], np.int64)
        edst = np.array([idx[e & 0xFFFFFFFF] for e in self.edges], np.int64)
        ewv = d0[esrc].astype(np.int64)      # weight = d_i, exact
        # layer decomposition of the abstract bound
        lay = []
        cur = Rs.copy()
        NEG = -(1 << 40)
        for _ in range(MAXK + 2):
            lay.append(int((Ls + cur).max()))
            nxt = np.full(S, NEG, np.int64)
            np.maximum.at(nxt, esrc, ewv + cur[edst])
            if nxt.max() <= NEG // 2:
                break
            cur = nxt
        # closure
        hh = Rs.copy()
        cyclic = False
        for _ in range(S + 2):
            new = hh.copy()
            np.maximum.at(new, esrc, ewv + hh[edst])
            if np.array_equal(new, hh):
                break
            hh = new
        else:
            cyclic = True
        bound = None if cyclic else int((Ls + hh).max())
        print("     %-24s states %8d edges %8d  %s"
              % (self.name, S, len(self.edges),
                 "CYCLIC -> closure = +inf (vacuous)" if cyclic else
                 "bound %3d  vs F+q' = %d  %s (exact %d)"
                 % (bound, budget,
                    "CERTIFIES (D)" if bound <= budget else "FAILS by %+d"
                    % (bound - budget), exact)))
        if not cyclic and layers:
            print("        abstract layer maxima k = 0.. : %s" % lay)
            # reconstruct one maximising abstract walk
            st = int(np.argmax(Ls + hh))
            walk = [int(Ls[st])]
            seen_st = set()
            while st not in seen_st:
                seen_st.add(st)
                walk.append(int(d0[st]))
                cand = np.flatnonzero(esrc == st)
                if not len(cand):
                    break
                gains = ewv[cand] + hh[edst[cand]]
                if not len(gains) or gains.max() <= hh[st] - 1e-9 and \
                        hh[st] == d0[st]:
                    break
                j = int(cand[int(np.argmax(gains))])
                if ewv[j] + hh[edst[j]] < hh[st]:
                    break
                st = int(edst[j])
            print("        maximising abstract window (L, then gaps): %s "
                  "sum %d" % (walk, sum(walk)))
        return bound, cyclic, lay


def run(y, seg=64_000_000, specs=((2, 0), (2, 35), (3, 0), (3, 35), (4, 0))):
    gears = primes(5, y)
    P = prod(gears)
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    floor_a = a                      # Mechanic's size floor 2u' for this step
    uvals = [pow(6, -1, g) for g in gears]
    sysx = [HistSystem(m, mod, q1) for m, mod in specs]
    maxmod = max([mod for _, mod in specs] + [1])
    F_old = 0
    F_new = 0
    layers = np.zeros(MAXK + 1, np.int64)
    QJ = np.zeros(JMAX + 1, np.int64)
    QJw = {}
    deep = {}                    # k -> list of (word, gL, gR, sum, address)
    ngaps = 0
    tail = None
    head = None
    t0 = time.time()

    def eat(ops, first_new):
        nonlocal F_old, F_new, ngaps
        d = np.diff(ops)
        n = len(d)
        if n <= 2 * (MAXK + OVL + 4):
            return
        assert int(d.max()) < BASE, "gap value exceeds the packing base"
        d = d.astype(np.int16)          # memory-lean: gaps are < 64
        cls = np.full(n, 9, np.int8)
        r = d % q1
        cls[r == 0] = 0
        cls[r == a] = 1
        cls[r == b] = -1
        nxt = np.arange(1, n)
        cm = np.where(cls[:-1] == 0, 0, 1)
        cp = np.where(cls[:-1] == 0, 1, 0)
        lm = (cls[:-1] == 0) | (cls[:-1] == 1)
        lp = (cls[:-1] == 0) | (cls[:-1] == -1)
        h = np.stack([d.astype(np.int32), d.astype(np.int32)])
        used = 0
        for _ in range(MAXK):
            hn = h.copy()
            hn[0][:-1] = np.where(lm, np.maximum(h[0][:-1],
                                                 d[:-1] + h[cm, nxt]),
                                  h[0][:-1])
            hn[1][:-1] = np.where(lp, np.maximum(h[1][:-1],
                                                 d[:-1] + h[cp, nxt]),
                                  h[1][:-1])
            if np.array_equal(hn, h):
                break
            h = hn
            used += 1
        assert used < MAXK, "chain longer than MAXK"
        i0 = max(OVL, first_new)
        i1 = n - MAXK - OVL
        if i1 <= i0:
            return
        sl = slice(i0, i1)
        ngaps += i1 - i0
        F_old = max(F_old, int(d[sl].max()))
        L = d[i0 - 1:i1 - 1]
        F_new = max(F_new, int(max((L + h[0][sl]).max(),
                                   (L + h[1][sl]).max())))
        # exact layer maxima (qualmax_{k+2}) and deep-chain inventory
        NEGB = -(1 << 28)
        cur = np.stack([d.astype(np.int32), d.astype(np.int32)])
        for lay in range(MAXK + 1):
            if cur.max() < NEGB / 2:
                break
            v = int(max((L + cur[0][sl]).max(), (L + cur[1][sl]).max()))
            layers[lay] = max(layers[lay], v)
            if lay >= 2:            # record every realised chain of >= 2 links
                hit = np.flatnonzero((cur[0][sl] > NEGB // 2) |
                                     (cur[1][sl] > NEGB // 2))
                for t in hit.tolist():
                    i = i0 + t
                    w = [int(x) for x in d[i:i + lay]]
                    rec = (tuple(w), int(d[i - 1]), int(d[i + lay]),
                           int(d[i - 1]) + sum(w) + int(d[i + lay]),
                           int(ops[i]))
                    deep.setdefault(lay, set()).add(rec)
            nx = np.full_like(cur, NEGB)
            nx[0][:-1] = np.where(lm, d[:-1] + cur[cm, nxt], NEGB)
            nx[1][:-1] = np.where(lp, d[:-1] + cur[cp, nxt], NEGB)
            cur = nx
        # size-floor spectrum Q_J(M; floor_a): J consecutive gaps, the J-1
        # interiors all >= floor_a
        big = d >= floor_a
        for J in range(2, JMAX + 1):
            if i1 - i0 < J + 2:
                continue
            # window = J consecutive gaps d[i..i+J-1]; the J-2 INTERIOR gaps
            # d[i+1..i+J-2] are the mutual distances of the J-1 interior
            # openings and carry the size floor (J = 2 has none, so Q_2 = F_2)
            ok = np.ones(i1 - i0, bool)
            tot = d[sl].astype(np.int32)
            for t in range(1, J):
                if t <= J - 2:
                    ok &= big[i0 + t:i1 + t]
                tot = tot + d[i0 + t:i1 + t]
            tot = np.where(ok, tot, -1)
            j = int(np.argmax(tot))
            if int(tot[j]) > QJ[J]:
                QJ[J] = int(tot[j])
                i = i0 + j
                QJw[J] = ([int(x) for x in d[i:i + J]], int(ops[i]))
        # abstraction systems
        ph = (ops[:-1] % maxmod).astype(np.int32) if maxmod > 1 else None
        for S in sysx:
            S.absorb(d, cls, None if not S.mod else (ops[:-1] % S.mod), i0,
                     i1)

    for lo in range(0, P, seg):
        hi = min(P, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = (np.flatnonzero(~ex) + lo).astype(np.int64)
        del ex
        if head is None:
            head = op[:4 * OVL].copy()
        ops = op if tail is None else np.concatenate([tail, op])
        eat(ops, OVL if tail is None else OVL)
        tail = ops[-(2 * (MAXK + OVL + 4) + 2):].copy()
        del op, ops
        if (lo // seg) % 4 == 0 and P > 5e8:
            print("  seg to %.4g (%.1f%%) %.0fs" % (hi, 100 * hi / P,
                                                    time.time() - t0),
                  flush=True)
    # cyclic seam + the head/tail openings skipped by the OVL margins
    eat(np.concatenate([tail, head + P]), 1)

    print("\n=== machine %d -> %d  (period %d, %d openings scanned)"
          % (y, q1, P, ngaps))
    print("  letters a=%d b=%d ; F(M) = %d ; size floor 2u' = %d"
          % (a, b, F_old, floor_a))
    print("  L (x) K* (x) R = %d   vs   F(M+q') known = %s"
          % (F_new, KNOWN_F.get(q1)))
    if q1 in KNOWN_F:
        assert F_new == KNOWN_F[q1], (y, F_new, KNOWN_F[q1])
        print("     IDENTITY VERIFIED (exact, full period)")
    budget = F_old + q1
    nz = [int(v) for v in layers if v > 0]
    print("  EXACT layer maxima (qualmax_{k+2}), k = 0.. : %s" % nz)
    print("  SIZE-FLOOR spectrum Q_J(%d; %d), J = 2..%d : %s"
          % (y, floor_a, JMAX, [int(QJ[J]) for J in range(2, JMAX + 1)]))
    for J in range(2, JMAX + 1):
        if J in QJw:
            print("       Q_%d = %3d  witness gaps %s at opening %d"
                  % (J, QJ[J], QJw[J][0], QJw[J][1]))
    for k in sorted(deep):
        recs = sorted(deep[k], key=lambda z: -z[3])
        wc = {}
        for w, gl, gr, tot, addr in recs:
            wc[w] = wc.get(w, 0) + 1
        print("  chains of exactly %d links (%d realised, window = %d gaps, "
              "%d distinct words):" % (k, len(recs), k + 2, len(wc)))
        print("       word multiset: %s"
              % sorted(wc.items(), key=lambda z: -z[1])[:10])
        for w, gl, gr, tot, addr in recs[:12]:
            print("       word %-16s flanks (%2d,%2d) window sum %3d  at %d"
                  % (str(w), gl, gr, tot, addr))
        if len(recs) > 12:
            print("       ... %d more" % (len(recs) - 12))
    print("  CERTIFICATE (C3): %d <= F + q' = %d   margin %+d (%.3f q')"
          % (F_new, budget, budget - F_new, (budget - F_new) / q1))
    assert F_new <= budget
    assert max(nz) == F_new
    assert QJ[2] == layers[0], (QJ[2], layers[0])    # both are F_2(M)
    for k, v in enumerate(nz):
        assert v <= QJ[k + 2] or k + 2 > JMAX, (k, v, QJ[k + 2])
    dump = os.environ.get("KH_DUMP4")
    if dump:
        for S in sysx:
            if S.m == 4 and not S.mod:
                with open(dump, "w") as fh:
                    for e in sorted(S.edges):
                        src, dst = e >> 32, e & 0xFFFFFFFF
                        ks, kd = src >> 1, dst >> 1
                        fh.write("%d %d %d %d\n"
                                 % ((ks // BASE // BASE) % BASE,
                                    (ks // BASE) % BASE, ks % BASE,
                                    kd % BASE))
                print("  dumped %d realised A_4 edges to %s"
                      % (len(S.edges), dump))
    print("  HISTORY LADDER (sound class-level closures):")
    out = []
    for S in sysx:
        bnd, cyc, lay = S.close(budget, F_new)
        assert cyc or bnd >= F_new, (S.name, bnd, F_new)
        out.append((S.name, bnd, cyc))
    print("  (%.0f s)" % (time.time() - t0))
    return dict(y=y, q1=q1, F_old=F_old, F_new=F_new, budget=budget,
                layers=nz, QJ=[int(QJ[J]) for J in range(2, JMAX + 1)],
                ladder=out)


def main():
    args = sys.argv[1:]
    seg = 64_000_000
    specs = [(2, 0), (2, 35), (3, 0), (3, 35), (4, 0)]
    if "--seg" in args:
        i = args.index("--seg")
        seg = int(float(args[i + 1]))
        del args[i:i + 2]
    if "--specs" in args:
        i = args.index("--specs")
        specs = [tuple(int(v) for v in s.split(":"))
                 for s in args[i + 1].split(",")]
        del args[i:i + 2]
    res = []
    for y in args:
        res.append(run(int(y), seg=seg, specs=specs))
    print("\n=== SUMMARY: the history ladder")
    names = [n for n, _, _ in res[0]["ladder"]]
    print("   step        exact  budget  " +
          "  ".join("%-18s" % n for n in names))
    for r in res:
        cells = []
        for _, b, c in r["ladder"]:
            cells.append("%-18s" % ("CYCLIC" if c else
                                    ("%d %s" % (b, "OK" if b <= r["budget"]
                                                else "FAIL"))))
        print("   %2d -> %-3d %6d %7d  %s"
              % (r["y"], r["q1"], r["F_new"], r["budget"], "  ".join(cells)))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
