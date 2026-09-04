"""Round 30 (mechanic), probe (a): L AS A RESIDUE-RUN STATISTIC, AND WHAT
SUPPRESSES IT.

By the chain law (anchor-235 9d) and D_g = A_kill(M -> g) = L + 1 (C50 / R89),
the longest realised legal word L_g(M) for a prime g > M is the longest run of
CONSECUTIVE gaps of machine M whose residues mod g lie in the three classes
{0, +d, -d}, d = 2 * 6^{-1} mod g, with the nonzero classes strictly
alternating (T3; padded letters transparent).  Naive counting - N = prod(q-2)
gaps, each in class with chance 3/g, independent - predicts longest runs of
order ln(N)/ln(g/3): about 13-18 at g = 53.  Measured L(47) = 4.

This file measures, on the REAL gap sequence of M (full period for M <= 29;
machine 31 and 37 streamed from the memory-mapped machine-29 list of
chain_depth_r29.py, no full-period array beyond machine 29), for every prime
g in (M, 200] with a non-empty legal alphabet:

  V1 RAW    longest run of consecutive gaps all in class (no alternation);
  V2 T3     longest run with alternation = L_g(M) (= D_g - 1);
  occ_L     the number of length-L windows of consecutive gaps that are legal
            alternating words (occurrence counts, L = 1..V2);
  the gap-value histogram of M (exact, cyclically closed), from which the
  class densities p0, p+, p- and two MODELS are computed at report time:
  MODEL-U   ln(N)/ln(g/3)                      (uniform residues, the brief's)
  MODEL-D   independent letters with the REAL class densities: raw
            ln(N)/ln(1/(p0+p+ + p-)); with T3 the transfer-matrix growth rate
            lam = p0 + sqrt(p+ p-), ln(N)/ln(1/lam); and E[occ_L] = N * (total
            probability of a legal alternating class word of length L), by a
            3-state DP.

The attaining runs are recorded in SECTION VIEW (slot of the opening before
the run, its gaps, residues, tooth word) so a run is an exhibited object, not
a statistic.  Every slice asserts max gap <= F(M) (a free gate on the source).

Exactness across slices: each slice is prepended with the last K = 4096 gaps
of the previous one and the code asserts that no run reaches K, so every run
is seen whole in some slice.  The cyclic closure appends the first opening +
the period.

usage:
  uv run python research/resrun_r30.py scan M [chunks]     M in 11..37
        (for M = 37, 'chunks' = how many of the 1147 lower-period chunks to
         stream: a DELIBERATE PARTIAL SWEEP with its support recorded)
  uv run python research/resrun_r30.py report [M ...]
  uv run python research/resrun_r30.py gate                (V2 == D_g - 1
        against anchor235/chain_depth.py's row at g = 7..29)
"""
import json
import os
import sys
import time
from math import log, prod, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "data", "r30")
KNOWN_F = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58,
           37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
K = 4096
SLICE = 1 << 23


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


PRIMES = [p for p in range(5, 201) if is_prime(p)]


def gears(y):
    return [p for p in PRIMES if p <= y]


def tooth(q):
    u = pow(6, -1, q)
    return u, (-u) % q


def dg(g):
    return (2 * pow(6, -1, g)) % g


def alphabet(M, g):
    """legal VALUES <= F(M) for gear g, with their T3 class."""
    d, F = dg(g), KNOWN_F[M]
    out = []
    for v in range(1, F + 1):
        r = v % g
        if r == 0:
            out.append((v, 0))
        elif r == d:
            out.append((v, 1))
        elif r == (g - d) % g:
            out.append((v, -1))
    return out


def lut_for(M, g):
    lut = np.full(256, 9, np.int8)
    for v, c in alphabet(M, g):
        lut[v] = c
    return lut


# ---------------------------------------------------------------- sources
def full_period_openings(M):
    G = gears(M)
    P = prod(G)
    w = np.ones(P, bool)
    for q in G:
        for u in tooth(q):
            w[u % q::q] = False
    return np.flatnonzero(w).astype(np.int64), P


def slices(M, nchunks=None):
    """yield (pos int64 slice, tag).  Consecutive slices are contiguous in
    slot order; the caller carries the seam.  The last slice ends with the
    cyclic closure (first opening + period)."""
    if M <= 23:
        X, P = full_period_openings(M)
        yield np.concatenate([X, [X[0] + P]]), "full"
        return
    import chain_depth_r29 as CD
    X, R = CD.load()
    if M == 29:
        P = CD.P29
        for a in range(0, CD.N29, SLICE):
            b = min(a + SLICE, CD.N29)
            pos = X[a:b].astype(np.int64)
            if b == CD.N29:
                pos = np.concatenate([pos, [int(X[0]) + P]])
            yield pos, f"m29[{a}:{b}]"
        return
    if M == 31:
        chunks = [(0, j) for j in range(31)]
        g, P = 37, CD.P31
    elif M == 37:
        chunks = CD.chunk_list(41)
        if nchunks is not None:
            chunks = chunks[:nchunks]
        g, P = 41, CD.P37
    else:
        raise ValueError(M)
    first = None
    for ci, (a, b) in enumerate(chunks):
        for pos, _ in CD.stream_chunk(X, R, g, a, b):
            if first is None:
                first = int(pos[0])
            last = ci == len(chunks) - 1
            yield pos, f"chunk({a},{b})"
    if nchunks is None or nchunks >= len(CD.chunk_list(g)):
        yield np.array([first + P], np.int64), "closure"


# ---------------------------------------------------------------- the runs
def run_stats(cls, alt):
    """cls: int8 classes (0, 1, -1, or 9 = out of class).  Returns
    (lengths array, start array): for each end index j the longest legal run
    ending at j and its start."""
    n = len(cls)
    idx = np.arange(n, dtype=np.int32)
    q = np.where(cls == 9, idx + 1, 0).astype(np.int32)
    if alt:
        nz = np.flatnonzero((cls == 1) | (cls == -1)).astype(np.int32)
        if len(nz) > 1:
            same = cls[nz[1:]] == cls[nz[:-1]]
            j = nz[1:][same]                     # each j once: plain assign
            q[j] = np.maximum(q[j], nz[:-1][same] + 1)
    s = np.maximum.accumulate(q)
    return idx - s + 1, s


class Best:
    def __init__(self):
        self.L = 0
        self.wit = None

    def offer(self, L, wit):
        if L > self.L:
            self.L, self.wit = int(L), wit


def scan(M, nchunks=None, gmax=200):
    F = KNOWN_F[M]
    G = [g for g in PRIMES if M < g <= gmax and alphabet(M, g)]
    luts = {g: lut_for(M, g) for g in G}
    raw = {g: Best() for g in G}
    t3 = {g: Best() for g in G}
    occ = {g: np.zeros(64, np.int64) for g in G}
    hist = np.zeros(256, np.int64)
    ngaps = 0
    carry = None
    t0 = time.time()
    nsl = 0
    for pos, tag in slices(M, nchunks):
        if carry is not None:
            pos = np.concatenate([carry, pos])
            base = len(carry) - 1          # gaps with index >= base are new
        else:
            base = 0
        if len(pos) < 2:
            carry = pos[-K - 1:]
            continue
        gaps = np.diff(pos)
        assert gaps.max() <= F, (M, tag, int(gaps.max()), F)
        g8 = gaps.astype(np.uint8)
        newg = g8[base:]
        hist += np.bincount(newg, minlength=256)[:256]
        ngaps += len(newg)
        for g in G:
            cls = luts[g][g8]
            for alt, best in ((False, raw[g]), (True, t3[g])):
                ln, st = run_stats(cls, alt)
                if alt:
                    # occurrence counts: windows ending at NEW indices only
                    lnew = ln[base:]
                    lnew = lnew[lnew > 0]
                    if len(lnew):
                        h = np.bincount(np.minimum(lnew, 63))
                        occ[g][:len(h)] += h
                j = int(np.argmax(ln))
                L = int(ln[j])
                assert L < K, ("run reaches the carry", M, g, L)
                if L > best.L and j >= base:
                    s = int(st[j])
                    gl = [int(v) for v in gaps[s:j + 1]]
                    best.offer(L, dict(
                        slot=int(pos[s]), gaps=gl,
                        residues=[v % g for v in gl],
                        cls=[int(luts[g][v]) for v in gl]))
        carry = pos[-K - 1:]
        nsl += 1
        if nsl % 8 == 0 or M <= 23:
            print(f"  M={M} slice {nsl} ({tag}) t={time.time()-t0:.0f}s "
                  f"gaps={ngaps:,}  L_{G[0]}: raw {raw[G[0]].L} t3 {t3[G[0]].L}",
                  flush=True)
    # occ_L is the count of windows of length EXACTLY... no: ln[j] is the
    # longest legal run ending at j, so #{j: ln[j] >= L} = #legal L-windows.
    out = dict(M=M, F=F, ngaps=ngaps, nchunks=nchunks,
               partial=(M == 37 and nchunks is not None
                        and nchunks < 1147),
               hist={str(v): int(c) for v, c in enumerate(hist) if c},
               seconds=round(time.time() - t0, 1), g={})
    for g in G:
        cum = np.cumsum(occ[g][::-1])[::-1]        # cum[L] = #{ln >= L}
        out["g"][str(g)] = dict(
            d=dg(g), alphabet=alphabet(M, g),
            raw=raw[g].L, raw_wit=raw[g].wit,
            t3=t3[g].L, t3_wit=t3[g].wit,
            occ=[int(cum[L]) for L in range(1, t3[g].L + 1)])
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"resrun_m{M}{'_c%d' % nchunks if nchunks else ''}.json")
    with open(fn, "w") as f:
        json.dump(out, f)
    print(f"  written {fn}  ({out['seconds']}s, {ngaps:,} gaps)")
    return out


# ---------------------------------------------------------------- models
def densities(hist, g, N):
    d = dg(g)
    p = {0: 0, 1: 0, -1: 0}
    for v, c in hist.items():
        v = int(v)
        r = v % g
        if r == 0:
            p[0] += c
        elif r == d:
            p[1] += c
        elif r == (g - d) % g:
            p[-1] += c
    return {k: v / N for k, v in p.items()}


def model_occ(p, N, Lmax):
    """E[#legal alternating L-windows] under independent letters: 3-state DP
    (last nonzero class none / + / -)."""
    out = []
    a, b, c = 1.0, 0.0, 0.0          # weights: none, last +, last -
    for L in range(1, Lmax + 1):
        a, b, c = (a * p[0],
                   a * p[1] + c * p[1] + b * p[0],
                   a * p[-1] + b * p[-1] + c * p[0])
        out.append(N * (a + b + c))
    return out


def model_len(p, N, alt):
    if alt:
        lam = p[0] + sqrt(p[1] * p[-1])
    else:
        lam = p[0] + p[1] + p[-1]
    if lam <= 0:
        return 0.0
    if lam >= 1:
        return float("inf")
    return log(N) / log(1 / lam)


def report(Ms):
    rows = []
    for M in Ms:
        cand = sorted(f for f in os.listdir(OUT)
                      if f.startswith(f"resrun_m{M}") and f.endswith(".json"))
        if not cand:
            print(f"  M={M}: no scan on disk")
            continue
        J = json.load(open(os.path.join(OUT, cand[-1])))
        N = J["ngaps"]
        Nfull = prod(q - 2 for q in gears(M))
        qn = next(g for g in PRIMES if g > M)
        print(f"\n=== MACHINE {M}  F = {J['F']}  gaps scanned {N:,} of "
              f"{Nfull:,} ({100.0*N/Nfull:.2f}%){'  PARTIAL' if J['partial'] else ''}"
              f"  next prime q' = {qn}")
        print("     g   d  |Lam|  V1raw  V2t3 | modelU  modelD-raw  modelD-t3 |"
              "  occ_1..occ_V2  (measured / model)")
        for gs, e in sorted(J["g"].items(), key=lambda kv: int(kv[0])):
            g = int(gs)
            p = densities(J["hist"], g, N)
            mu = log(Nfull) / log(g / 3)
            md_raw = model_len(p, Nfull, False)
            md_t3 = model_len(p, Nfull, True)
            eo = model_occ(p, N, max(1, e["t3"]) + 1)   # occ on the SCANNED N
            occs = "  ".join(f"{o}/{m:.3g}" for o, m in zip(e["occ"] + [0], eo))
            mark = " <- q'" if g == qn else ""
            print(f"  {g:4d} {e['d']:3d} {len(e['alphabet']):4d}   {e['raw']:3d}"
                  f"    {e['t3']:3d}  | {mu:5.1f}    {md_raw:6.1f}     "
                  f"{md_t3:6.1f}   | {occs}{mark}")
            rows.append(dict(M=M, g=g, V1=e["raw"], V2=e["t3"], modelU=mu,
                             modelD_raw=md_raw, modelD_t3=md_t3,
                             occ=e["occ"], model_occ=eo, alphabet=e["alphabet"],
                             p=p, next=(g == qn)))
        e = J["g"][str(qn)]
        for nm in ("raw", "t3"):
            w = e[nm + "_wit"]
            if w:
                print(f"  attaining {nm.upper()} run for g = q' = {qn}: slot "
                      f"{w['slot']}, gaps {w['gaps']}, residues mod {qn} "
                      f"{w['residues']}, classes {w['cls']}")
    return rows


def gate():
    """V2 must equal D_g - 1 of anchor235/chain_depth.py at g = 7..29, and
    the full-period gap count must be prod(q-2)."""
    want_D = {7: 2, 11: 1, 13: 2, 17: 2, 19: 2, 23: 3, 29: 2}
    for M, g in ((5, 7), (7, 11), (11, 13), (13, 17), (17, 19), (19, 23),
                 (23, 29)):
        if M == 5:
            G = [5]
        elif M == 7:
            G = [5, 7]
        else:
            G = gears(M)
        P = prod(G)
        w = np.ones(P, bool)
        for q in G:
            for u in tooth(q):
                w[u % q::q] = False
        X = np.flatnonzero(w).astype(np.int64)
        pos = np.concatenate([X, X + P])            # doubled: seam runs seen
        gaps = np.diff(pos)
        assert len(X) == prod(q - 2 for q in G)
        F = int(gaps.max())
        d = dg(g)
        lut = np.full(F + 1, 9, np.int8)
        for v in range(1, F + 1):
            r = v % g
            lut[v] = 0 if r == 0 else 1 if r == d else -1 if r == (g - d) % g else 9
        cls = lut[gaps]
        ln, _ = run_stats(cls, True)
        L = int(ln.max())
        assert L + 1 == want_D[g], (M, g, L + 1, want_D[g])
        print(f"  M={{5..{M}}} g={g:2d}: F = {F:2d}, longest T3 run L = {L}, "
              f"D_g = L+1 = {L+1} = chain_depth.py's {want_D[g]}  OK")
    print("\nALL ASSERTIONS PASSED (V2 = D_g - 1 at 7 rungs)")


def models_from_ghist(Ms=(31, 37), gmax=130):
    """MODEL-U / MODEL-D rows for machines whose EXACT cyclic gap histogram is
    on disk (research/data/r26/ghist_<M>.csv, round-26 lap-phase transfer),
    so the density side of the ladder exists even where the run scan is
    partial (m37) or slow (m31)."""
    import csv
    for M in Ms:
        fn = os.path.join(HERE, "data", "r26", f"ghist_{M}.csv")
        hist = {}
        for row in csv.DictReader(open(fn)):
            if int(row["y"]) == M:
                hist[row["gap"]] = int(row["count"])
        N = sum(hist.values())
        Nfull = prod(q - 2 for q in gears(M))
        assert N == Nfull, (M, N, Nfull)
        qn = next(g for g in PRIMES if g > M)
        print(f"\n=== MACHINE {M} MODELS from the exact cyclic histogram "
              f"(N = {N:,} = prod(q-2))")
        print("     g   d |Lam| modelU modelD-raw modelD-t3  E[occ_1..6]")
        for g in [p for p in PRIMES if M < p <= gmax and alphabet(M, p)]:
            p = densities(hist, g, N)
            mu = log(N) / log(g / 3)
            eo = model_occ(p, N, 6)
            print(f"  {g:4d} {dg(g):3d} {len(alphabet(M, g)):4d}  {mu:5.1f}   "
                  f"{model_len(p, N, False):6.1f}    {model_len(p, N, True):6.1f}   "
                  + "  ".join(f"{v:.3g}" for v in eo)
                  + ("  <- q'" if g == qn else ""))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "models":
        models_from_ghist()
    elif cmd == "scan":
        M = int(sys.argv[2])
        nch = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
        gmax = int(sys.argv[4]) if len(sys.argv) > 4 else 200
        scan(M, nch, gmax)
    elif cmd == "report":
        Ms = [int(x) for x in sys.argv[2:]] or [11, 13, 17, 19, 23, 29, 31, 37]
        report(Ms)
    elif cmd == "gate":
        gate()
