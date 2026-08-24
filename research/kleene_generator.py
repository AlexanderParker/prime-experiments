"""Round 22 (constructor): THE ARITY-FREE GENERATOR - (D) as ONE max-plus
Kleene-star identity, and its dual certificate.

Round 21 (R41) proved that adding a gear is an exact Kronecker recursion

    B_new S_new = (B_M S_M) (x) S'  +  (E_M S_M) (x) (B' S')      (*)

and that no function of the marginal data bounds the nilpotency index of the
sum: delta <= q' is a >= 3-point joint-realizability statement, and the
truncation arity moves with the machine.  This script answers the question
that raises - IS NILPOTENCY ADDITIVITY ARITY-FREE? - by writing the index of
the sum as a KLEENE STAR, which is one equation about all orders at once.

THE ALGEBRA.  Note that in (*) the second summand is nilpotent of index 2
((Y (x) Z)^2 = Y^2 (x) Z^2 = 0 because a single gear's two teeth are never
adjacent), the first is nilpotent of index F(M) (S' is invertible), and a
nonzero word in the expansion is a blocked/exposed pattern of M whose
"exposed" positions are all q'-killed - the merge law.  On the state space

    states = (opening i of M, current tooth s in {+,-})

define the max-plus (tropical) matrix K and the two flank vectors

    K[(i,s), (i+1,s')] = d_i   if d_i qualifies and s -> s' is the T3
                               transition of its letter class; else -inf
    L(i) = d_{i-1}   (left flank)        R(i,s) = d_i   (right flank)

Then, with (x) = max-plus product and K* = (+)_{m>=0} K^m the Kleene star,

    THEOREM (verified here at every scannable step):
        F(M + q')  =  L^T (x) K* (x) R      exactly,
    and (D) at alpha = 3 is the SINGLE inequality  L^T (x) K* (x) R <= F+q'.

K is nilpotent, so K* is a finite sum - but the identity never names the
truncation depth: its m-th layer is exactly qualmax_{m+2}, so one algebra
generates every layer.  That is the arity-free form.

    COROLLARY (tropical dual certificate).  (D) holds at a step iff there is
    a potential h on states with
        (C1) h(i,s) >= d_i
        (C2) h(i,s) >= d_i + h(i+1,s')  for every legal qualifying transition
        (C3) d_{i-1} + h(i,s) <= F(M) + q'
    Necessity: h = K* (x) R.  Sufficiency: any super-solution dominates the
    star.  No depth index appears anywhere in (C1)-(C3).

THE DECISIVE TEST run here: can h be replaced by a FINITE table - a function
of a bounded local state (last gap value, corridor phase mod 35 / 385, tooth)?
That is the only way the certificate becomes machine-free.  The class-level
max-plus closure is computed and reported; a positive cycle in the class
graph means the abstraction is NOT nilpotent and the certificate is vacuous
at that modulus.

Usage: uv run python research/kleene_generator.py [y ...]   (default 11..29)
"""
import os
import sys
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")

NEG = -(1 << 60)
# F(M+q') along the consecutive chain, for the identity test
KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91}


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def next_prime(y):
    p = y + 1
    while not all(p % d for d in range(2, int(p ** 0.5) + 1)):
        p += 1
    return p


def build(y, mods=(35, 385, 5005)):
    """Full-period openings, cyclic gaps, T3 letter classes, corridor phases.

    Memory-lean: gaps and classes are int8, phases int16, and the opening
    positions are dropped as soon as the phases are taken (machine 29 has
    2.15e8 openings)."""
    gears = primes(5, y)
    P = prod(gears)
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    np.logical_not(ex, out=ex)
    assert P < 2 ** 31
    op = np.flatnonzero(ex).astype(np.int32)
    del ex
    d = np.diff(np.concatenate([op, np.array([op[0] + P], np.int32)]))
    assert d.max() < 127
    ph = {M: (op % M).astype(np.int16) for M in mods if M <= P}
    del op
    d = d.astype(np.int8)
    r = d.astype(np.int16) % q1
    cls = np.full(len(d), 9, np.int8)       # 9 = not qualifying
    cls[r == 0] = 0
    cls[r == a] = 1
    cls[r == b] = -1
    return dict(y=y, q1=q1, a=a, b=b, P=P, ph=ph, d=d, cls=cls,
                gears=gears)


def value_function(d, cls):
    """h(i,s) = K* (x) R : the largest chain-sum startable at opening i with
    the kill sitting on tooth s.  Iterated to the max-plus fixed point; the
    number of iterations needed IS the nilpotency index of K."""
    n = len(d)
    nxt = np.roll(np.arange(n, dtype=np.int32), -1)
    # legality masks: from tooth s, letter class c
    #   c = 0 (padded): legal from both, tooth unchanged
    #   c = +1        : legal only from s = -1, lands on +1
    #   c = -1        : legal only from s = +1, lands on -1
    leg_m = (cls == 0) | (cls == 1)      # legal from s = -1
    to_m = np.where(cls == 0, 0, 1).astype(np.int8)   # landing tooth
    leg_p = (cls == 0) | (cls == -1)     # legal from s = +1
    to_p = np.where(cls == 0, 1, 0).astype(np.int8)
    h = np.stack([d.astype(np.int16), d.astype(np.int16)])   # [tooth-, +]
    it = 0
    while True:
        it += 1
        hn = h.copy()
        cand_m = d + h[to_m, nxt]
        hn[0] = np.where(leg_m, np.maximum(h[0], cand_m), h[0])
        cand_p = d + h[to_p, nxt]
        hn[1] = np.where(leg_p, np.maximum(h[1], cand_p), h[1])
        if np.array_equal(hn, h):
            break
        h = hn
        assert it < 200, "K is not nilpotent - positive cycle in the chain"
    return h, it


def report(y, verbose=True):
    m = build(y)
    d, cls, q1 = m["d"], m["cls"], m["q1"]
    n = len(d)
    h, iters = value_function(d, cls)
    L = np.roll(d, 1).astype(np.int16)      # left flank d_{i-1}
    F_new = int(max((L + h[0]).max(), (L + h[1]).max()))
    F_old = int(d.max())
    F2_old = int((d.astype(np.int16) + np.roll(d, -1)).max())
    print("\n=== machine %d -> %d  (period %d, %d openings)"
          % (y, q1, m["P"], n))
    print("  letters a=%d b=%d ; F(M)=%d  F2(M)=%d" % (m["a"], m["b"],
                                                       F_old, F2_old))
    print("  KLEENE STAR: index of K (iterations to the fixed point) = %d"
          % iters)
    print("  L (x) K* (x) R = %d   vs   F(M+q') known = %s"
          % (F_new, KNOWN_F.get(q1)))
    if q1 in KNOWN_F:
        assert F_new == KNOWN_F[q1], (y, F_new, KNOWN_F[q1])
        print("     IDENTITY VERIFIED (exact)")
    # layer decomposition: qualmax_{m+2} = max over chains of exactly m links
    NEG16 = np.int16(-30000)
    hm = [np.stack([d.astype(np.int16), d.astype(np.int16)])]
    hh = hm[0]
    nxt = np.roll(np.arange(n, dtype=np.int32), -1)
    leg_m = (cls == 0) | (cls == 1)
    to_m = np.where(cls == 0, 0, 1).astype(np.int8)
    leg_p = (cls == 0) | (cls == -1)
    to_p = np.where(cls == 0, 1, 0).astype(np.int8)
    layers = []
    cur = hm[0]
    for lay in range(0, iters + 1):
        v = int((L + cur).max()) if cur.max() > -15000 else None
        layers.append(v)
        nxtv = np.full_like(cur, NEG16)
        c_m = np.where(leg_m, d + cur[to_m, nxt], NEG16).astype(np.int16)
        c_p = np.where(leg_p, d + cur[to_p, nxt], NEG16).astype(np.int16)
        nxtv[0] = c_m
        nxtv[1] = c_p
        if nxtv.max() <= -15000:
            break
        cur = nxtv
    print("  layer maxima  (chain of exactly k links => window of k+2 gaps):")
    print("     k = " + "  ".join("%6d" % k for k in range(len(layers))))
    print("     max " + "  ".join("%6s" % v for v in layers))
    # the certificate
    budget = F_old + q1
    print("  CERTIFICATE (C3): max(L + h) = %d  <=  F + q' = %d   margin %+d "
          "(%.3f q')" % (F_new, budget, budget - F_new,
                         (budget - F_new) / q1))
    assert F_new <= budget, "(D) FAILS at this step"
    # verify (C1),(C2) hold for h and that h is the LEAST super-solution
    assert (h >= d).all()
    hplus = h[1]
    hminus = h[0]
    ok = True
    ok &= bool((~leg_m | (hminus >= d + h[to_m, nxt])).all())
    ok &= bool((~leg_p | (hplus >= d + h[to_p, nxt])).all())
    assert ok, "h is not a super-solution"
    # leastness: some state is tight at every constraint it activates
    tight = ((hminus == np.where(leg_m, d + h[to_m, nxt], d)).all() and
             (hplus == np.where(leg_p, d + h[to_p, nxt], d)).all())
    print("  h is a super-solution: YES ; h is the LEAST one (every state "
          "tight): %s" % tight)
    m["h"] = h
    m["F_new"] = F_new
    m["iters"] = iters
    m["margin"] = budget - F_new
    m["layers"] = layers
    return m


def _group_max(keys, vals, S):
    out = np.full(S, NEG, np.int64)
    np.maximum.at(out, keys, vals)
    return out


def _group_min(keys, vals, S):
    out = np.full(S, 1 << 60, np.int64)
    np.minimum.at(out, keys, vals)
    return out


def abstraction_chunked(m, chunk=8_000_000):
    """Same class-level certificate as abstraction_test, but every per-opening
    array is built one chunk at a time, so machine 29 (2.15e8 openings) fits
    in ~1 GB.  Only states that CONTAIN the gap value are available here (the
    destination class must be a function of (source state, next letter))."""
    d, cls, q1 = m["d"], m["cls"], m["q1"]
    n = len(d)
    h = m["h"]
    ccode = np.where(cls == 9, 3, cls.astype(np.int32) + 1)
    budget = int(d.max()) + q1
    print("  abstraction test (chunked): (budget F+q' = %d, exact %d)"
          % (budget, m["F_new"]))
    cands = [("value only", None)]
    for M in sorted(m["ph"]):
        cands.append(("(phase mod %d, value)" % M, M))
    for name, M in cands:
        st = (d.astype(np.int32) if M is None
              else m["ph"][M].astype(np.int32) * 128 + d)
        keys = st * 4 + ccode
        del st
        kmax = int(keys.max()) + 1
        pres = np.zeros(kmax, np.int8)
        pres[keys] = 1
        lut = np.cumsum(pres, dtype=np.int32) - 1
        C = int(lut[-1]) + 1
        S = 2 * C
        if S * 1024 > 3e9:
            print("     %-26s SKIPPED (edge id space %d)" % (name, S * 1024))
            del pres, lut, keys
            continue
        inv = lut[keys].astype(np.int32)
        del pres, lut, keys
        # per-class base/flank and h-spread, chunked
        Rc = np.zeros(C, np.int32)
        Lc = np.zeros(C, np.int32)
        hi = np.full((2, C), -1, np.int32)
        lo = np.full((2, C), 1 << 30, np.int32)
        ew = np.zeros(S * 1024, np.int16)
        ds = np.zeros(S * 1024, np.int32)
        for lo_i in range(0, n, chunk):
            hi_i = min(n, lo_i + chunk)
            sl = slice(lo_i, hi_i)
            iv = inv[sl]
            dv = d[sl].astype(np.int32)
            np.maximum.at(Rc, iv, dv)
            np.maximum.at(Lc, iv, d[(np.arange(lo_i, hi_i) - 1) % n]
                          .astype(np.int32))
            for t in (0, 1):
                np.maximum.at(hi[t], iv, h[t][sl].astype(np.int32))
                np.minimum.at(lo[t], iv, h[t][sl].astype(np.int32))
            nx = (np.arange(lo_i, hi_i, dtype=np.int64) + 1) % n
            cv = cls[sl]
            invn = inv[nx]
            dkn = (d[nx].astype(np.int32) * 4 + ccode[nx])
            for s, legmask, tolanding in (
                    (0, (cv == 0) | (cv == 1),
                     np.where(cv == 0, 0, 1).astype(np.int32)),
                    (1, (cv == 0) | (cv == -1),
                     np.where(cv == 0, 1, 0).astype(np.int32))):
                sel = np.flatnonzero(legmask)
                if not len(sel):
                    continue
                srcs = 2 * iv[sel] + s
                dsts = 2 * invn[sel] + tolanding[sel]
                eid = srcs.astype(np.int64) * 1024 + dkn[sel] * 2 + \
                    tolanding[sel]
                np.maximum.at(ew, eid, dv[sel].astype(np.int16))
                ds[eid] = dsts
        spread = int(max((hi[t] - lo[t]).max() for t in (0, 1)))
        keep = np.flatnonzero(ew)
        esrc = (keep // 1024).astype(np.int64)
        edst = ds[keep].astype(np.int64)
        ewv = ew[keep].astype(np.int64)
        del ew, ds, inv
        Rs = np.stack([Rc, Rc], 1).reshape(-1).astype(np.int64)
        Ls = np.stack([Lc, Lc], 1).reshape(-1).astype(np.int64)
        hh = Rs.copy()
        cyclic = False
        for _ in range(S + 2):
            cand = ewv + hh[edst]
            new = hh.copy()
            np.maximum.at(new, esrc, cand)
            if np.array_equal(new, hh):
                break
            hh = new
        else:
            cyclic = True
        bound = None if cyclic else int((Ls + hh).max())
        print("     %-26s states %7d  h-spread %4d  %s"
              % (name, S, spread,
                 "CYCLIC -> class closure = +inf (vacuous)" if cyclic else
                 "bound %d  vs  F+q' = %d   %s (exact %d)"
                 % (bound, budget,
                    "CERTIFIES (D)" if bound <= budget else "FAILS by %+d"
                    % (bound - budget), m["F_new"])))


def abstraction_test(m, mods=(35, 385, 5005)):
    """Is (D) certifiable from a BOUNDED LOCAL STATE - a finite table rather
    than a per-opening potential?

    For an abstraction alpha : opening -> finite class, build the SOUND
    class-level max-plus system: an edge (alpha(i), s) -> (alpha(i+1), s')
    with weight max{ d_i : the real transition realises this class edge },
    base R_c = max{ d_i : alpha(i) = c }, flank L_c = max{ d_{i-1} }.  Its
    Kleene star hhat dominates h class-wise, so max(L_c + hhat_c) is a valid
    upper bound on F(M+q').  Reported:
      * h-spread   : is the EXACT h already a function of the class?
      * class graph: acyclic (finite closure) or cyclic (bound = +inf,
                     the certificate is vacuous at that state space)
      * bound      : the class-level certificate value vs F + q'
    """
    d, cls, q1 = m["d"], m["cls"], m["q1"]
    n = len(d)
    h = m["h"]
    d64 = d.astype(np.int32)
    L = np.roll(d64, 1)
    nxt = np.roll(np.arange(n, dtype=np.int32), -1)
    leg_m = (cls == 0) | (cls == 1)
    to_m = np.where(cls == 0, 0, 1).astype(np.int8)
    leg_p = (cls == 0) | (cls == -1)
    to_p = np.where(cls == 0, 1, 0).astype(np.int8)
    ccode = np.where(cls == 9, 3, cls.astype(np.int64) + 1)  # 0..3, no
    # collisions when packed as key*4 + ccode
    budget = int(d.max()) + q1
    print("  abstraction test: can (D) be certified from a bounded local "
          "state?   (budget F+q' = %d, exact %d)" % (budget, m["F_new"]))
    big = n > 30_000_000        # too big for a general edge sort
    cands = [("value only", lambda: d64.astype(np.int64), True)]
    for M in sorted(m["ph"]):
        cands.append(("phase mod %d" % M,
                      lambda M=M: m["ph"][M].astype(np.int64) * 128, False))
        cands.append(("(phase mod %d, value)" % M,
                      lambda M=M: m["ph"][M].astype(np.int64) * 128 + d64,
                      True))
    for name, f, hasval in cands:
        if big and not hasval:
            print("     %-26s SKIPPED (value-free state, needs the general "
                  "edge sort - too big at %d openings)" % (name, n))
            continue
        # raw key: (state) * 4 + letter class.  Bounded, so compact it with
        # bincount instead of a 2e8-element sort.
        keys = f() * 4 + ccode
        kmax = int(keys.max()) + 1
        if kmax > 3e8:
            print("     %-26s SKIPPED (key space %d too large)" % (name, kmax))
            continue
        pres = np.zeros(kmax, np.int8)
        pres[keys] = 1
        lut = (np.cumsum(pres, dtype=np.int64) - 1)
        C = int(lut[-1]) + 1
        inv = lut[keys].astype(np.int32)
        del pres, lut
        S = 2 * C                                    # states = class x tooth
        # h-spread inside a class
        spread = 0
        for t in (0, 1):
            hi = _group_max(inv, h[t].astype(np.int64), C)
            lo = _group_min(inv, h[t].astype(np.int64), C)
            spread = max(spread, int((hi - lo).max()))
        # sound class-level edges, weight = max realised gap
        src = np.r_[2 * inv[leg_m], 2 * inv[leg_p] + 1].astype(np.int64)
        dst = np.r_[2 * inv[nxt[leg_m]] + to_m[leg_m],
                    2 * inv[nxt[leg_p]] + to_p[leg_p]].astype(np.int64)
        w = np.r_[d64[leg_m], d64[leg_p]].astype(np.int64)
        if big:
            # bounded edge id: with the value in the state, the destination
            # class is a function of (src state, destination letter, tooth)
            dk = np.r_[(d64[nxt[leg_m]] * 4 + ccode[nxt[leg_m]]),
                       (d64[nxt[leg_p]] * 4 + ccode[nxt[leg_p]])]
            eid = src * 1024 + dk.astype(np.int64) * 2 + (dst & 1)
            if eid.max() > 6e8:
                print("     %-26s SKIPPED (edge id space too large)" % name)
                continue
            ew = np.zeros(int(eid.max()) + 1, np.int64)
            np.maximum.at(ew, eid, w)
            ds = np.zeros_like(ew)
            ds[eid] = dst
            keep = np.flatnonzero(ew)
            esrc, edst, ew = keep // 1024, ds[keep], ew[keep]
            del dk, eid, ds, keep
        else:
            ek = src * S + dst
            uek, einv = np.unique(ek, return_inverse=True)
            ew = _group_max(einv, w, len(uek))
            esrc, edst = uek // S, uek % S
        # base and flank per state (tooth-independent here)
        Rc = _group_max(inv, d64, C)
        Lc = _group_max(inv, L, C)
        Rs = np.repeat(Rc, 2).reshape(C, 2).T.reshape(-1) if False \
            else np.stack([Rc, Rc], 1).reshape(-1)
        Ls = np.stack([Lc, Lc], 1).reshape(-1)
        # longest path by Bellman-Ford-style iteration; a value that keeps
        # growing past S rounds certifies a cycle
        hh = Rs.copy()
        cyclic = False
        for _ in range(S + 2):
            cand = ew + hh[edst]
            new = hh.copy()
            np.maximum.at(new, esrc, cand)
            if np.array_equal(new, hh):
                break
            hh = new
        else:
            cyclic = True
        bound = None if cyclic else int((Ls + hh).max())
        print("     %-26s states %7d  h-spread %4d  %s"
              % (name, S, spread,
                 "CYCLIC -> class closure = +inf (vacuous)" if cyclic else
                 "bound %d  vs  F+q' = %d   %s (exact %d)"
                 % (bound, budget,
                    "CERTIFIES (D)" if bound <= budget else "FAILS by %+d"
                    % (bound - budget), m["F_new"])))


def main():
    ys = [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19, 23]
    ms = []
    for y in ys:
        m = report(y)
        if len(m["d"]) > 30_000_000:
            abstraction_chunked(m)
        else:
            abstraction_test(m)
        ms.append(m)
    print("\n=== SUMMARY: the Kleene identity across steps")
    print("   step        index(K)   L(x)K*(x)R   F(M)+q'   margin   layers")
    for m in ms:
        print("   %2d -> %-3d   %6d   %10d %9d  %+7d   %s"
              % (m["y"], m["q1"], m["iters"], m["F_new"],
                 m["F_new"] + m["margin"], m["margin"], m["layers"]))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
