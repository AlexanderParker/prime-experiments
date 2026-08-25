"""Round 23 (constructor): IS THE HISTORY CERTIFICATE MACHINE-FREE?

research/kleene_history.py shows that the m-point history abstraction A_m of
the Kleene generator (R46) certifies (D) at every scannable step, and is EXACT
at m = 4.  But A_m's edge set is the set of m-tuples of gap values REALISED by
the machine, so it still needs the machine's period.  This script replaces
"realised" by a MACHINE-FREE over-approximation and measures what survives:

    MF_m(Mod):  state = (corridor phase r mod Mod, the last m-1 gap values,
                tooth), where Mod is 35 (gears 5,7) or 385 (gears 5,7,11) and
                every opening phase implied by the state must lie in the
                EXPOSED set E mod Mod.  An edge exists whenever the corridor
                walk stays in E, the middle gap qualifies mod q', and the T3
                alternation permits it.  Gap values run over 1..F only.

Every realised m-tuple is corridor-admissible, so MF_m's edge set CONTAINS
A_m's and the closure is a SOUND upper bound on F(M + q') - and it depends
only on the two numbers (F, q'), never on the machine's period.  If it
certifies, (D) is machine-free at that step.

Reported per layer, because the layers are not equal in difficulty:
  layer 0 is F_2(M) <= F + q' - LEMMA 1, a two-gap statement with no chain in
  it at all; layers >= 1 are the actual merge chains.  X11 already records
  that bounded-modulus corridors "constrain where, never how big", so layer 0
  is expected to be the machine-free wall; the question this script answers is
  whether the DEEP layers - the ones (D) was always thought to be about - are
  machine-free once lemma 1 is granted.

Usage: uv run python research/machinefree_cert.py [--mods 35,385] [--ms 2,3,4]
"""
import sys
import time
from math import prod

import numpy as np

# (F(M), q') along the consecutive chain, and the known exact F(M+q')
STEPS = [(11, 7, 13, 11), (13, 11, 17, 18), (17, 18, 19, 25),
         (19, 25, 23, 34), (23, 34, 29, 43), (29, 43, 31, 58),
         (31, 58, 37, 88)]
NEG = -(1 << 40)


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def exposed(Mod):
    ex = np.zeros(Mod, bool)
    for g in primes(5, Mod):
        if Mod % g:
            continue
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    return ~ex


def build_mf(F, q1, Mod, m):
    """States and edges of the machine-free system.  Returns arrays."""
    inE = exposed(Mod)
    E = np.flatnonzero(inE)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    ds = np.arange(1, F + 1)
    cls = np.full(F + 1, 9, np.int8)
    for d in ds:
        r = int(d) % q1
        if r == 0:
            cls[d] = 0
        elif r == a:
            cls[d] = 1
        elif r == b:
            cls[d] = -1
    states = {}                 # (r, hist tuple, tooth) -> index
    slist = []
    back = {}                   # r -> list of valid backward hist prefixes
    fwd = {}                    # r -> list of valid d_i
    for r in E.tolist():
        fwd[r] = [int(d) for d in ds if inE[(r + int(d)) % Mod]]
        pref = [()]
        for _ in range(m - 2):
            nxt = []
            for p in pref:
                base = (r - sum(p)) % Mod
                for d in ds.tolist():
                    if inE[(base - d) % Mod]:
                        nxt.append((d,) + p)
            pref = nxt
        back[r] = pref
    for r in E.tolist():
        for p in back[r]:
            for d in fwd[r]:
                for s in (0, 1):
                    states[(r, p + (d,), s)] = len(slist)
                    slist.append((r, p + (d,), s))
    S = len(slist)
    esrc, edst, ew, tups = [], [], [], []
    for (r, hist, s), si in states.items():
        di = hist[-1]
        c = cls[di]
        if c == 9:
            continue
        if c == 0:
            land = s
        elif c == 1:
            if s != 0:
                continue
            land = 1
        else:
            if s != 1:
                continue
            land = 0
        r2 = (r + di) % Mod
        h2 = hist[1:]
        for dn in fwd[r2]:
            j = states.get((r2, h2 + (dn,), land))
            if j is not None:
                esrc.append(si)
                edst.append(j)
                ew.append(di)
                tups.append(hist + (dn,))
    tup = np.array(tups, np.int16) if tups else np.zeros((0, m), np.int16)
    Rs = np.array([h[-1] for _, h, _ in slist], np.int64)
    Ls = (np.array([h[-2] for _, h, _ in slist], np.int64) if m >= 3
          else np.full(S, F, np.int64))       # m = 2: flank unknown, use F
    return (S, np.array(esrc, np.int64), np.array(edst, np.int64),
            np.array(ew, np.int64), Rs, Ls, tup)


def build_mf_edges(F, q1, Mod, m):
    """Same system, with the value m-tuple of every edge returned (round 23,
    for research/cegar_cert.py)."""
    return build_mf(F, q1, Mod, m)


def closure(S, esrc, edst, ew, Rs, Ls, maxlay=12):
    lay = []
    cur = Rs.copy()
    for _ in range(maxlay):
        lay.append(int((Ls + cur).max()))
        nxt = np.full(S, NEG, np.int64)
        if len(esrc):
            np.maximum.at(nxt, esrc, ew + cur[edst])
        if nxt.max() <= NEG // 2:
            break
        cur = nxt
    else:
        return None, lay, True
    hh = Rs.copy()
    for _ in range(S + 2):
        new = hh.copy()
        if len(esrc):
            np.maximum.at(new, esrc, ew + hh[edst])
        if np.array_equal(new, hh):
            break
        hh = new
    else:
        return None, lay, True
    return int((Ls + hh).max()), lay, False


def main():
    args = sys.argv[1:]
    mods = [35, 385]
    ms = [3, 4]
    if "--mods" in args:
        mods = [int(x) for x in args[args.index("--mods") + 1].split(",")]
    if "--ms" in args:
        ms = [int(x) for x in args[args.index("--ms") + 1].split(",")]
    print("MACHINE-FREE CERTIFICATE MF_m(Mod): edges = corridor-admissible "
          "m-tuples, values 1..F.\nSound upper bound on F(M+q') depending "
          "only on (F, q').\n")
    for y, F, q1, exact in STEPS:
        budget = F + q1
        print("=== step %d -> %d :  F = %d, q' = %d, budget %d, exact %d"
              % (y, q1, F, q1, budget, exact))
        for Mod in mods:
            for m in ms:
                t0 = time.time()
                S, esrc, edst, ew, Rs, Ls, _tp = build_mf(F, q1, Mod, m)
                bnd, lay, cyc = closure(S, esrc, edst, ew, Rs, Ls)
                deep = max(lay[1:]) if len(lay) > 1 else None
                tag = ("CYCLIC" if cyc else
                       ("%d %s" % (bnd, "CERTIFIES" if bnd <= budget
                                   else "FAILS by %+d" % (bnd - budget))))
                print("    MF_%d mod %-3d  states %8d edges %8d  %-22s "
                      "layers %s   deep(>=1) %s  %.0fs"
                      % (m, Mod, S, len(esrc), tag, lay, deep,
                         time.time() - t0))
                if deep is not None and not cyc:
                    assert bnd >= exact, (y, Mod, m, bnd, exact)
        print()
    print("all assertions passed")


if __name__ == "__main__":
    main()
