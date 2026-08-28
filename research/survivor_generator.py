"""Round 24 (constructor): THE SURVIVOR-EXTENDED KLEENE GENERATOR.

WHY.  R53 measured that (D) at 29 -> 31 is certified by counterexample-guided
refinement from the machine-free system GIVEN ONE INTEGER, F_2(29) = 55 -
lemma 1's left-hand side at the old machine.  The brief asks what
machine-independent fact substitutes for that integer.  research/twogap_table.py
(round 24, item 1) shows the two machine-free suppliers both saturate at 2F:
the gap HISTOGRAM (tight bound F + G_2, and G_2 = F because maximal gaps are
mirror-paired) and the CORRIDOR (R52's layer-0 column, 2F or 2F-1 at all seven
steps).  So no unitary invariant and no bounded-modulus corridor can supply it.

THIS SCRIPT SUPPLIES IT FROM ONE GEAR DOWN INSTEAD.

R46 wrote F(M+q') = L (x) K* (x) R on states (killed opening i, tooth s):
K carries the T2/T3-legal spacings between consecutive KILLED openings, L is
the left flank d_{i-1} and R the right flank d_j.  The observation of this
round is that F_2(M+q') - the two-gap statement AT THE NEW MACHINE - is the
SAME generator with ONE extra transition:

    a two-gap SKIP  x_k --(d_i + d_{i+1})--> y_1  which passes through the
    single SURVIVING opening i+1 that separates the two new gaps.

    THEOREM (survivor identity).  Let p_0 < ... < p_n be consecutive openings
    of M.  A window of TWO consecutive gaps of M + q' is exactly a window in
    which every interior opening is killed by q' except ONE.  The killed
    openings all sit at the same q'-phase, so their consecutive spacings obey
    T2/T3 (R40); the spacing straddling the survivor is d_i + d_{i+1}, and the
    survivor lives iff d_i alone is NOT a legal T3 transition from the current
    tooth.  Hence

        F_2(M+q') = L (x) K* (x) SIGMA (x) K* (x) R ,

    SIGMA[(i,s),(i+2,s')] = d_i + d_{i+1} when cls(d_i) is ILLEGAL from s and
    cls(d_i + d_{i+1}) is legal from s with landing tooth s'.  (Endpoints may
    be relaxed to arbitrary openings: enlarging a window that has one interior
    survivor only enlarges the enclosing two-gap window, so the maximum is
    unchanged.)  Boundary cases k = 0 (the left chain empty) are the branch B
    terms below; l = 0 (right chain empty) is the SIGMA-then-stop term.

CONSEQUENCE.  The two-gap statement at M + q' is layer 0 of the SAME max-plus
system over M.  So the m-point history abstraction A_m of R49 - whose whole
machine input is the dictionary of REALISED gap m-tuples of M - bounds it too,
and the "one extra integer" of R53 is not an extra obligation at all: it is a
consequence of the dictionary the certificate is already querying.  In
particular the realised-PAIR sub-dictionary of A_m is exactly F_2(M).

WHAT IS COMPUTED, in one streaming full-period pass per machine:
  * EXACT F(M+q')     (cross-checked against the known ladder)
  * EXACT F_2(M+q')   (cross-checked against the independent pair census)
  * the A_m closure of the survivor-extended system, built ONLY from realised
    m-tuples (skip edges are COMPOSED from two realised m-tuples, so no
    (m+1)-tuple fact is ever used), reported against the next step's budget.

Usage: uv run python research/survivor_generator.py y [y ...] [--seg N] [--m 4]
"""
import os
import sys
import time
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
# independently censused (research/twogap_table.py, full-period lag-1 pairs)
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68}
OVL = 32
MAXK = 8
BASE = 64


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def next_prime(y):
    p = y + 1
    while not all(p % d for d in range(2, int(p ** 0.5) + 1)):
        p += 1
    return p


def classify(vals, q1, a, b):
    """T2 class of a spacing: 0 (padded), 1 (letter a), -1 (letter b), 9."""
    r = vals % q1
    c = np.full(r.shape, 9, np.int8)
    c[r == 0] = 0
    c[r == a] = 1
    c[r == b] = -1
    return c


def legal(s, c):
    """Is class c a legal T3 transition out of tooth s?  (0 -> 0/1, a: 0->1,
    b: 1->0.)"""
    return (c == 0) | (c == (1 if s == 0 else -1))


def land(s, c):
    return np.where(c == 0, s, 1 - s)


def run(y, seg=48_000_000, m=4):
    gears = primes(5, y)
    P = prod(gears)
    q1 = next_prime(y)
    q2 = next_prime(q1)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    uvals = [pow(6, -1, g) for g in gears]
    F_old = 0
    F_new = 0
    F2_new = 0
    F2_old = 0
    wit = {}
    # m = 4 uses a packed bool table (BASE^4 = 16.7M); m = 5 would need 1 GB
    # packed, so it falls back to a Python set of packed keys.
    assert m in (4, 5), "m in {4, 5}"
    tuples = (np.zeros(BASE ** m, bool) if m == 4 else set())
    ngaps = 0
    tail = None
    head = None
    t0 = time.time()

    def eat(ops, first_new):
        nonlocal F_old, F_new, F2_new, F2_old, ngaps
        d = np.diff(ops)
        n = len(d)
        if n <= 2 * (MAXK + OVL + 8):
            return
        assert int(d.max()) < BASE, "gap value exceeds the packing base"
        d = d.astype(np.int32)
        cls = classify(d, q1, a, b)
        # class of the two-gap spacing d_i + d_{i+1}
        d2 = np.zeros(n, np.int32)
        d2[:-1] = d[:-1] + d[1:]
        cls2 = classify(d2, q1, a, b)
        NEG = -(1 << 28)

        # ---- plain chain value h[s][i] (R46's K* (x) R), exact ----
        h = np.stack([d.copy(), d.copy()])
        nxt = np.arange(1, n)
        used = 0
        for _ in range(MAXK):
            hn = h.copy()
            for s in (0, 1):
                lg = legal(s, cls[:-1])
                ld = land(s, cls[:-1])
                hn[s][:-1] = np.where(lg, np.maximum(h[s][:-1],
                                                     d[:-1] + h[ld, nxt]),
                                      h[s][:-1])
            if np.array_equal(hn, h):
                break
            h = hn
            used += 1
        assert used < MAXK, "chain longer than MAXK"

        # ---- survivor value G[s][i]: exactly one skip, from killed i ----
        # if cls(d_i) is LEGAL from s the opening i+1 is killed -> recurse;
        # otherwise i+1 SURVIVES and we either stop (l = 0) or skip into a
        # fresh chain at i+2 (l >= 1).
        G = np.full((2, n), NEG, np.int32)
        for s in (0, 1):
            lg = legal(s, cls[:-2])
            stop = d[:-2] + d[1:-1]                      # d_i + d_{i+1}
            lg2 = legal(s, cls2[:-2])
            ld2 = land(s, cls2[:-2])
            cont = np.where(lg2, stop + h[ld2, np.arange(2, n)], NEG).astype(np.int32)
            G[s][:-2] = np.where(lg, NEG, np.maximum(stop.astype(np.int32), cont))
        for _ in range(MAXK):
            Gn = G.copy()
            for s in (0, 1):
                lg = legal(s, cls[:-1])
                ld = land(s, cls[:-1])
                Gn[s][:-1] = np.where(lg, np.maximum(G[s][:-1],
                                                     d[:-1] + G[ld, nxt]),
                                      G[s][:-1])
            if np.array_equal(Gn, G):
                break
            G = Gn

        i0 = max(OVL, first_new)
        i1 = n - MAXK - OVL
        if i1 <= i0:
            return
        sl = slice(i0, i1)
        ngaps += i1 - i0
        F_old = max(F_old, int(d[sl].max()))
        L = d[i0 - 1:i1 - 1]
        F_new = max(F_new, int(max((L + h[0][sl]).max(), (L + h[1][sl]).max())))
        F2_old = max(F2_old, int((d[i0 - 1:i1 - 1] + d[sl]).max()))

        # branch A: at least one killed opening before the survivor
        for s in (0, 1):
            v = L + G[s][sl]
            j = int(np.argmax(v))
            if int(v[j]) > F2_new:
                F2_new = int(v[j])
                i = i0 + j
                wit["A"] = ("branchA", s, [int(x) for x in d[i - 1:i + 6]],
                            int(ops[i]))
        # branch B: the left chain is EMPTY - p_0 = i, survivor i+1
        #   B0 (l = 0): d_i + d_{i+1}  == F_2(M) itself
        vb0 = d[i0 - 1:i1 - 1] + d[sl]
        if int(vb0.max()) > F2_new:
            F2_new = int(vb0.max())
            wit["B0"] = ("branchB0", int(vb0.max()))
        #   B1 (l >= 1): chain starts at i+2 with tooth t and the survivor
        #   at i+1 must live: the BACKWARD transition into tooth t must be
        #   illegal, which is legal_{1-t}(cls(d_{i+1})).
        for t in (0, 1):
            alive = ~legal(1 - t, cls[i0:i1])          # cls(d_{i+1}) with
            #   i+1 running over [i0, i1) means p_0 = i0-1 ... shift below
            v = (d[i0 - 2:i1 - 2] + d[i0 - 1:i1 - 1]
                 + h[t][i0:i1])
            v = np.where(~legal(1 - t, cls[i0 - 1:i1 - 1]), v, NEG)
            j = int(np.argmax(v))
            if int(v[j]) > F2_new:
                F2_new = int(v[j])
                i = i0 + j
                wit["B1"] = ("branchB1", t, [int(x) for x in d[i - 2:i + 4]],
                             int(ops[i]))
            del alive

        # ---- realised gap m-tuples of M (the whole machine input of A_m) ----
        lo, hi = i0 - 1, min(i1 + m, n)
        if hi - lo > m:
            k = np.zeros(hi - lo - m + 1, np.int64)
            for j in range(m):
                k = k * BASE + d[lo + j:hi - m + 1 + j].astype(np.int64)
            if m == 4:
                tuples[k] = True
            else:
                tuples.update(np.unique(k).tolist())

    for lo in range(0, P, seg):
        hi = min(P, lo + seg)
        # The box is shared with other lanes and round-22/23 both lost a
        # long pass to a transient memory squeeze (R22 iv, R23 iv).  Retry
        # rather than die: a refused 8 MiB allocation is not a finding.
        for attempt in range(60):
            try:
                ex = np.zeros(hi - lo, bool)
                for g, u in zip(gears, uvals):
                    ex[(u - lo) % g::g] = True
                    ex[(-u - lo) % g::g] = True
                op = (np.flatnonzero(~ex) + lo).astype(np.int64)
                del ex
                break
            except MemoryError:
                ex = None
                print("  MemoryError at %d, retry %d in 20 s"
                      % (lo, attempt), flush=True)
                time.sleep(20)
        else:
            raise MemoryError("segment %d unallocatable after 60 retries" % lo)
        if head is None:
            head = op[:4 * OVL].copy()
        ops = op if tail is None else np.concatenate([tail, op])
        for attempt in range(60):
            try:
                eat(ops, OVL if tail is None else OVL)
                break
            except MemoryError:
                print("  MemoryError in eat at %d, retry %d in 20 s"
                      % (lo, attempt), flush=True)
                time.sleep(20)
        else:
            raise MemoryError("eat at %d unallocatable" % lo)
        tail = ops[-(2 * (MAXK + OVL + 8) + 2):].copy()
        del op, ops
        if (lo // seg) % 8 == 0 and P > 5e8:
            print("  seg to %.4g (%.1f%%) %.0fs"
                  % (hi, 100 * hi / P, time.time() - t0), flush=True)
    eat(np.concatenate([tail, head + P]), 1)

    print("\n=== machine %d  ->  %d   (period %d, %d gaps)"
          % (y, q1, P, ngaps))
    print("  F(M) = %d   F_2(M) = %d   letters a=%d b=%d" % (F_old, F2_old, a, b))
    print("  EXACT  F(M+q')   = %d   (known %s)" % (F_new, KNOWN_F.get(q1)))
    print("  EXACT  F_2(M+q') = %d   (known %s)" % (F2_new, KNOWN_F2.get(q1)))
    if q1 in KNOWN_F:
        assert F_new == KNOWN_F[q1], (y, F_new, KNOWN_F[q1])
    if y in KNOWN_F2:
        assert F2_old == KNOWN_F2[y], (y, F2_old, KNOWN_F2[y])
    if q1 in KNOWN_F2:
        assert F2_new == KNOWN_F2[q1], (y, F2_new, KNOWN_F2[q1])
        print("     SURVIVOR IDENTITY VERIFIED (exact, full period)")
    for kk in sorted(wit):
        print("     witness %s: %s" % (kk, wit[kk]))
    assert F2_new >= F_new, (F2_new, F_new)
    print("  the NEXT step's two-gap budget: F(M+q') + q'' = %d + %d = %d"
          "   -> margin %+d"
          % (F_new, q2, F_new + q2, F_new + q2 - F2_new))
    ntup = int(tuples.sum()) if m == 4 else len(tuples)
    print("  realised gap %d-tuples of M: %d" % (m, ntup))
    print("  (%.0f s)" % (time.time() - t0))
    return dict(y=y, q1=q1, q2=q2, F_old=F_old, F2_old=F2_old, F_new=F_new,
                F2_new=F2_new, tuples=tuples, P=P, ngaps=ngaps, a=a, b=b, m=m)


# ---------------------------------------------------------------------------
# The A_m abstraction of the survivor-extended system, built from the realised
# m-tuple dictionary ONLY (skip edges are composed from two realised m-tuples).
# ---------------------------------------------------------------------------

def abstract(res, m=None):
    m = m or res["m"]
    q1, a, b = res["q1"], res["a"], res["b"]
    t_ = res["tuples"]
    tups = (np.flatnonzero(t_).astype(np.int64) if m == 4
            else np.array(sorted(t_), np.int64))
    dig = np.zeros((len(tups), m), np.int64)
    t = tups.copy()
    for j in range(m - 1, -1, -1):
        dig[:, j] = t % BASE
        t //= BASE
    src = tups // BASE                      # first m-1 digits
    dst = tups % (BASE ** (m - 1))          # last m-1 digits
    keys = np.unique(np.concatenate([src, dst]))
    kidx = {int(k): i for i, k in enumerate(keys.tolist())}
    K = len(keys)
    kd = np.zeros((K, m - 1), np.int64)
    t = keys.copy()
    for j in range(m - 2, -1, -1):
        kd[:, j] = t % BASE
        t //= BASE
    di_of_key = kd[:, -1]                   # d_i
    NEG = -(1 << 40)

    esrc = np.array([kidx[int(x)] for x in src], np.int64)
    edst = np.array([kidx[int(x)] for x in dst], np.int64)
    ew = dig[:, m - 2]                      # weight = d_i = last digit of src
    ecls = classify(ew, q1, a, b)

    # states are (key, tooth)
    def sid(ki, s):
        return ki * 2 + s

    S = K * 2
    Rs = np.repeat(di_of_key, 2)
    Ls = np.repeat(kd[:, -2], 2) if m >= 3 else None
    # ---- ordinary edges ----
    osrc, odst, oew = [], [], []
    for s in (0, 1):
        sel = np.flatnonzero(legal(s, ecls))
        if not len(sel):
            continue
        ld = land(s, ecls[sel])
        osrc.append(esrc[sel] * 2 + s)
        odst.append(edst[sel] * 2 + ld)
        oew.append(ew[sel])
    osrc = np.concatenate(osrc)
    odst = np.concatenate(odst)
    oew = np.concatenate(oew).astype(np.int64)

    # ---- skip transitions, as a TWO-HOP composition of realised m-tuples ----
    # A skip leaves the killed opening i, passes the SURVIVING opening i+1 and
    # lands on the killed opening i+2, so its target state is key_{i+2}: two
    # structural hops from key_i.  Composing pairwise would materialise
    # sum_k indeg(k)*outdeg(k) edges; instead note that for a fixed source
    # tuple and tooth the landing tooth is determined, so the second hop can be
    # maximised independently:
    #     Q_t[k'] = max over realised tuples j with src_j = k' of hplain[dst_j, t]
    # and the skip value out of (key_i, s) is
    #     max over tuples i with src_i = key_i, guards satisfied,
    #         of (d_i + d_{i+1}) + Q_{land(s, cls2)}[dst_i].
    # This is EXACTLY the pairwise composition, in O(#tuples).
    di_t = dig[:, m - 2]                   # d_i   for tuple index
    dn_t = dig[:, m - 1]                   # d_{i+1}
    w2 = di_t + dn_t
    cls2t = classify(w2, q1, a, b)

    def close(src_, dst_, w_, base):
        hh = base.copy()
        for _ in range(S + 2):
            new = hh.copy()
            if len(src_):
                np.maximum.at(new, src_, w_ + hh[dst_])
            if np.array_equal(new, hh):
                return hh, False
            hh = new
        return None, True

    hplain, cyc = close(osrc, odst, oew, Rs.astype(np.int64))
    if cyc:
        return dict(cyclic=True)
    plain_bound = int((Ls + hplain).max())
    # Q_t[k'] = best plain continuation starting one structural hop past k'
    Q = np.full((2, K), NEG, np.int64)
    for t_ in (0, 1):
        np.maximum.at(Q[t_], esrc, hplain[edst * 2 + t_])
    # G: exactly one skip.  base = skip transitions (with continuation), plus
    # the "stop right after the survivor" term d_i + d_{i+1} (right chain
    # empty).  Guard: cls(d_i) ILLEGAL from s (so opening i+1 survives).
    Gbase = np.full(S, NEG, np.int64)
    nskip = 0
    for s in (0, 1):
        guard = ~legal(s, ecls)                      # i+1 survives
        # stop term
        sel = np.flatnonzero(guard)
        if len(sel):
            np.maximum.at(Gbase, esrc[sel] * 2 + s, w2[sel])
        # continue term
        sel = np.flatnonzero(guard & legal(s, cls2t))
        nskip += len(sel)
        if len(sel):
            ld = land(s, cls2t[sel])
            val = w2[sel] + Q[ld, edst[sel]]
            val = np.where(Q[ld, edst[sel]] <= NEG // 2, NEG, val)
            np.maximum.at(Gbase, esrc[sel] * 2 + s, val)
    ssrc = np.zeros(nskip, np.int64)                 # count only
    G, cyc2 = close(osrc, odst, oew, Gbase)
    if cyc2:
        return dict(cyclic=True)
    bA = int((Ls + G).max())
    # branch B0 = max realised adjacent pair = F_2(M), read off the dictionary
    b0 = int(max((dig[:, j] + dig[:, j + 1]).max() for j in range(m - 1)))
    # branch B1 needs d_i, d_{i+1} at state key_{i+2}: digits -3 and -2
    bB1 = NEG
    if m >= 4:
        cl2 = classify(kd[:, -2], q1, a, b)
        for t_ in (0, 1):
            alive = ~legal(1 - t_, cl2)
            v = np.where(alive, kd[:, -3] + kd[:, -2] + hplain[np.arange(K) * 2 + t_],
                         NEG)
            bB1 = max(bB1, int(v.max()))
    bound = max(bA, b0, bB1)
    return dict(cyclic=False, plain=plain_bound, F2bound=bound, bA=bA, b0=b0,
                bB1=bB1, states=S, oedges=len(osrc), sedges=len(ssrc))


def main():
    args = sys.argv[1:]
    seg = 48_000_000
    m = 4
    if "--seg" in args:
        i = args.index("--seg")
        seg = int(float(args[i + 1]))
        del args[i:i + 2]
    if "--m" in args:
        i = args.index("--m")
        m = int(args[i + 1])
        del args[i:i + 2]
    out = []
    for y in args:
        res = run(int(y), seg=seg, m=m)
        t0 = time.time()
        ab = abstract(res)
        if ab.get("cyclic"):
            print("  A_%d survivor closure: CYCLIC (vacuous)" % m)
        else:
            print("  A_%d over M: %d states, %d ordinary + %d skip edges"
                  % (m, ab["states"], ab["oedges"], ab["sedges"]))
            print("     plain closure  -> F(M+q')   <= %d   (exact %d) %s"
                  % (ab["plain"], res["F_new"],
                     "EXACT" if ab["plain"] == res["F_new"] else ""))
            print("     survivor closure -> F_2(M+q') <= %d   (exact %d) %s"
                  % (ab["F2bound"], res["F2_new"],
                     "EXACT" if ab["F2bound"] == res["F2_new"] else ""))
            print("       branch A %d, branch B0 (= F_2(M)) %d, branch B1 %d"
                  % (ab["bA"], ab["b0"], ab["bB1"]))
            nb = res["F_new"] + res["q2"]
            print("     two-gap statement at M+q':  %d <= F(M+q') + q'' = %d"
                  "  %s (margin %+d)"
                  % (ab["F2bound"], nb,
                     "CERTIFIES" if ab["F2bound"] <= nb else "FAILS",
                     nb - ab["F2bound"]))
            assert ab["plain"] >= res["F_new"]
            assert ab["F2bound"] >= res["F2_new"]
            assert ab["b0"] == res["F2_old"], (ab["b0"], res["F2_old"])
        print("  (abstraction %.0f s)" % (time.time() - t0))
        out.append((res, ab))
    print("\n=== SUMMARY: the survivor generator")
    print("  step        F(M)  F_2(M) | F(M+q')  F_2(M+q')  budget'  |"
          "  A_%d bound on F_2(M+q')" % m)
    for res, ab in out:
        nb = res["F_new"] + res["q2"]
        cell = "CYCLIC" if ab.get("cyclic") else "%d %s" % (
            ab["F2bound"], "OK" if ab["F2bound"] <= nb else "FAIL")
        print("  %2d -> %-3d  %5d %6d  | %6d %9d %8d  |  %s"
              % (res["y"], res["q1"], res["F_old"], res["F2_old"],
                 res["F_new"], res["F2_new"], nb, cell))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
