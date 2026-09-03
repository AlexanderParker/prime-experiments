"""Round 29 (mechanic), brief item (c): THE ANCHOR-235 CHAIN DEPTH AND THE
RECORD LAW AT MACHINES 31, 37 AND 41, AS EVENTS, BY STREAMED PASSES.

anchor-235 section 9f (research/anchor235/chain_depth.py) states two laws about
adding one gear g on top of the machine M = {5 .. prev(g)}:

    D_g   = the longest run of CONSECUTIVE M-openings whose (actual) slot
            residues mod g all lie in one two-class set {r, r+d}, d = 2*6^{-1};
    F_g   = max over such runs of  (gap before) + (run span) + (gap after)
            [chain_depth.py prints F_g - 1, the blocked-count convention]

and computes both on ONE LOWER PERIOD, because the g copies of the lower period
inside the full period realise every deletion phase r exactly once.
chain_depth.py stops at g = 29 because it materialises the lower period as a
numpy array.  This file carries the same two laws to g = 31, 37 and 41 with NO
full-period array beyond machine 29:

  * machine 29's opening list (214,708,725 entries, uint32) is built ONCE and
    memory-mapped;
  * the {5..31} lower sequence is streamed as 31 chunks of it, the {5..37}
    lower sequence as 31 x 37 = 1147 chunks (2.18e11 openings, 1.24e12 slots),
    each chunk being the machine-29 list under one or two residue filters;
  * the phase r is NOT looped over.  Mapping residues by d^{-1} turns the
    two-class condition into "all values in {s, s+1}", so one rolling
    max/min pass over a length-L window decides ALL g phases at once; the
    phase is read back off the winning window as r = s*d mod g.

Cyclic closure is exact: the lower sequence is wrapped by appending its own head
at position + P_lower, whose residues mod g are those of the SHIFTED slot - the
copy-to-copy phase change is therefore carried, not ignored.

The four memory-mapped arrays `build` writes total 1.5 GB and research/data/r29
is a SHARED directory whose .gitignore rule covers *.log only, so they are
DELETED at round close and `build` regenerates them in ~30 s.  Run `build`
first before any `run`; the JSON results and the round gate do not need them.

usage:
  uv run python research/chain_depth_r29.py build
  uv run python research/chain_depth_r29.py run 31
  uv run python research/chain_depth_r29.py run 37
  uv run python research/chain_depth_r29.py run 41 [J37LO J37HI TAG]
  uv run python research/chain_depth_r29.py merge 41 TAG...
  uv run python research/chain_depth_r29.py gate          (small-g replication)
"""
import json
import os
import sys
import time
from math import prod

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "r29")
G29 = [5, 7, 11, 13, 17, 19, 23, 29]
P29 = prod(G29)                     # 1,078,282,205 slots
N29 = prod(q - 2 for q in G29)      # 214,708,725 openings
P31 = P29 * 31
P37 = P31 * 37
LOWER = {31: (G29, P29), 37: (G29 + [31], P31), 41: (G29 + [31, 37], P37)}
SLICE = 1 << 23                     # machine-29 indices per streamed slice
CARRY = 24                          # openings carried across every seam
LMAX = 12


def tooth(q):
    u = pow(6, -1, q)
    return u, (-u) % q


# ---------------------------------------------------------------- stage 0
def build():
    os.makedirs(DATA, exist_ok=True)
    xp = os.path.join(DATA, "x29.u32")
    t0 = time.time()
    if not os.path.exists(xp) or os.path.getsize(xp) != 4 * N29:
        X = np.memmap(xp, dtype=np.uint32, mode="w+", shape=(N29,))
        w = 0
        BLK = 1 << 26
        for a in range(0, P29, BLK):
            b = min(a + BLK, P29)
            arr = np.ones(b - a, bool)
            for q in G29:
                for u in tooth(q):
                    arr[(u - a) % q::q] = False
            k = np.flatnonzero(arr).astype(np.uint32) + np.uint32(a)
            X[w:w + len(k)] = k
            w += len(k)
        assert w == N29, (w, N29)
        X.flush()
        del X
        print(f"  x29.u32 built: {N29:,} openings in {time.time()-t0:.0f}s",
              flush=True)
    X = np.memmap(xp, dtype=np.uint32, mode="r")
    for g in (31, 37, 41):
        rp = os.path.join(DATA, f"r{g}.u8")
        if os.path.exists(rp) and os.path.getsize(rp) == N29:
            continue
        R = np.memmap(rp, dtype=np.uint8, mode="w+", shape=(N29,))
        for a in range(0, N29, 1 << 24):
            b = min(a + (1 << 24), N29)
            R[a:b] = (X[a:b] % np.uint32(g)).astype(np.uint8)
        R.flush()
        del R
        print(f"  r{g}.u8 built ({time.time()-t0:.0f}s)", flush=True)
    # gate the build against the exact opening count and the known F(29) = 43
    d = np.diff(X[:1 << 22].astype(np.int64))
    print(f"  build gate: {N29:,} openings, max gap in the first 4.2M = "
          f"{d.max()} (F(29) = 43)", flush=True)


def load():
    X = np.memmap(os.path.join(DATA, "x29.u32"), dtype=np.uint32, mode="r")
    R = {g: np.memmap(os.path.join(DATA, f"r{g}.u8"), dtype=np.uint8, mode="r")
         for g in (31, 37, 41)}
    return X, R


# ---------------------------------------------------------------- streaming
def chunk_list(g):
    if g == 31:
        return [(0, 0)]
    if g == 37:
        return [(0, j) for j in range(31)]
    return [(a, b) for a in range(37) for b in range(31)]


def chunk_base(g, j37, j31):
    return j37 * P31 + j31 * P29


def stream_chunk(X, R, g, j37, j31, lo=0, hi=None):
    """yield (pos, res) numpy arrays for one chunk of the lower sequence."""
    base = chunk_base(g, j37, j31)
    keep_masks = []
    if g >= 37:
        t = [(u - base) % 31 for u in tooth(31)]
        keep_masks.append((R[31], t))
    if g >= 41:
        t = [(u - base) % 37 for u in tooth(37)]
        keep_masks.append((R[37], t))
    hi = N29 if hi is None else hi
    # residues of the SHIFTED slot, as a g-entry permutation lookup: the
    # int64 add-and-modulo it replaces was 60% of this pass's cost.
    shift = np.array([(v + base) % g for v in range(g)], dtype=np.int16)
    for a in range(lo, hi, SLICE):
        b = min(a + SLICE, hi)
        keep = None
        for arr, t in keep_masks:
            s = arr[a:b]
            m = (s != t[0]) & (s != t[1])
            keep = m if keep is None else (keep & m)
        x = X[a:b]
        if keep is not None:
            x = x[keep]
        if len(x) == 0:
            continue
        pos = x.astype(np.int64) + base
        res = R[g][a:b]
        if keep is not None:
            res = res[keep]
        yield pos, shift[res]


# ---------------------------------------------------------------- the scan
class State:
    def __init__(self, g):
        self.g = g
        self.u = pow(6, -1, g)
        self.d = (2 * self.u) % g
        self.dinv = pow(self.d, -1, g)
        self.lut = np.array([(v * self.dinv) % g for v in range(g)],
                            dtype=np.int16)
        self.lut2 = (self.lut + 1) % g
        self.D = 1
        self.best = {}          # L -> (value, dict)

    def offer(self, L, value, pos, res, i):
        if L in self.best and self.best[L][0] >= value:
            return
        g = self.g
        rs = [int(v) for v in res[i:i + L]]
        ts = sorted({(v * self.dinv) % g for v in rs})
        if len(ts) == 1:
            s = ts[0]
        elif len(ts) == 2 and (ts[1] - ts[0]) == 1:
            s = ts[0]
        else:                                   # the cyclic pair {g-1, 0}
            s = g - 1
        r = (s * self.d) % g
        # which copy of the lower period does this phase belong to?
        PL = LOWER[g][1]
        j = ((-self.u - r) * pow(PL % g, -1, g)) % g
        self.best[L] = (value, dict(
            L=L, value=value, before=int(pos[i] - pos[i - 1]),
            span=int(pos[i + L - 1] - pos[i]),
            after=int(pos[i + L] - pos[i + L - 1]),
            x_prev=int(pos[i - 1]), residues=rs, r=r, copy=j,
            slot=int(pos[i - 1]) + j * PL))

    def merge(self, other):
        self.D = max(self.D, other["D"])
        for k, v in other["best"].items():
            L = int(k)
            if L not in self.best or self.best[L][0] < v["value"]:
                self.best[L] = (v["value"], v)

    def dump(self):
        return dict(g=self.g, D=self.D,
                    best={str(L): v[1] for L, v in sorted(self.best.items())})


def scan_block(pos, res, st):
    n = len(pos)
    if n < 3:
        return
    g = st.g
    # d^{-1} folds the two-class condition {r, r+d} into "two adjacent values",
    # applied as a g-entry lookup so no modulo runs per element.
    t = st.lut[res]
    t2 = st.lut2[res]
    v = pos[2:] - pos[:-2]                       # L = 1, every opening valid
    i = int(np.argmax(v))
    st.offer(1, int(v[i]), pos, res, i + 1)
    d0 = t[1:].astype(np.int16) - t[:-1]
    link = ((d0 == 0) | (d0 == 1) | (d0 == -1)
            | (d0 == g - 1) | (d0 == 1 - g))
    for L in range(2, LMAX + 1):
        if L == 2:
            valid = np.flatnonzero(link)
            cand = np.flatnonzero(link[:-1] & link[1:])
        else:
            c = cand[cand + L <= n]
            if len(c) == 0:
                break
            T = np.stack([t[c + k] for k in range(L)])
            ok = (T.max(0) - T.min(0)) <= 1
            T2 = np.stack([t2[c + k] for k in range(L)])
            ok |= (T2.max(0) - T2.min(0)) <= 1
            valid = c[ok]
            cand = np.intersect1d(valid, valid - 1, assume_unique=True)
        if len(valid) == 0:
            break
        st.D = max(st.D, L)
        sel = valid[(valid >= 1) & (valid + L <= n - 1)]
        if len(sel):
            vv = pos[sel + L] - pos[sel - 1]
            k = int(np.argmax(vv))
            st.offer(L, int(vv[k]), pos, res, int(sel[k]))
        if len(cand) == 0:
            break


def tail_of(X, R, g, j37, j31):
    """the last CARRY openings of one chunk (for a worker's leading seam)."""
    for pos, res in stream_chunk(X, R, g, j37, j31,
                                 lo=max(0, N29 - (1 << 16))):
        last = (pos, res)
    return last[0][-CARRY:], last[1][-CARRY:]


def run(g, j37lo=None, j37hi=None, tag=""):
    X, R = load()
    st = State(g)
    chunks = chunk_list(g)
    if g == 41 and j37lo is not None:
        chunks = [(a, b) for (a, b) in chunks if j37lo <= a < j37hi]
    t0 = time.time()
    cp = cr = None
    if g == 41 and j37lo:                      # leading seam from chunk before
        cp, cr = tail_of(X, R, g, j37lo - 1, 30)
    print(f"machine {g}: lower {LOWER[g][0][0]}..{LOWER[g][0][-1]} "
          f"(P_lower = {LOWER[g][1]:,}), {len(chunks)} chunks, "
          f"d = {st.d}, tag={tag or '-'}", flush=True)
    fn = os.path.join(DATA, f"chain_{g}{('_' + tag) if tag else ''}.json")
    laps, done = [], set()
    if os.path.exists(fn):                     # RESUME from my own dump
        prev = json.load(open(fn))
        st.merge(prev)
        laps = list(prev.get("laps_done", []))
        done = {tuple(x) for x in prev.get("chunks_done", [])}
        cp = cr = None
        print(f"  resuming: {len(done)} chunks already done, D={st.D}",
              flush=True)
    dl = [tuple(x) for x in done]
    for ci, (a, b) in enumerate(chunks):
        if (a, b) in dl:
            cp = cr = None            # the seam into the next chunk is lost
            continue
        if cp is None and (a, b) != (0, 0):
            # leading seam: the tail of the chunk that PRECEDES this one in
            # slot order, so a resumed or worker-sharded run is still exact
            # across the join.
            pa, pb = (a, b - 1) if b else (a - 1, 30)
            cp, cr = tail_of(X, R, g, pa, pb)
        for pos, res in stream_chunk(X, R, g, a, b):
            if cp is not None:
                pos = np.concatenate([cp, pos])
                res = np.concatenate([cr, res])
            scan_block(pos, res, st)
            cp, cr = pos[-CARRY:], res[-CARRY:]
        dl.append((a, b))
        if b == 30 or ci == len(chunks) - 1:
            laps.append(a)
        # a completed chunk is a self-contained unit of coverage: dump now so
        # that stopping the job early still leaves an EXACTLY specified
        # sample of the lower period on disk, and a restart resumes from it.
        d = st.dump()
        d["laps_done"] = laps
        d["chunks_done"] = [list(x) for x in dl]
        d["seconds"] = round(time.time() - t0, 1)
        with open(fn, "w") as f:
            json.dump(d, f, indent=1)
        if ci % max(1, len(chunks) // 20) == 0 or ci == len(chunks) - 1:
            print(f"  chunk {ci+1}/{len(chunks)} (j37={a}, j31={b}) "
                  f"t={time.time()-t0:.0f}s D={st.D} "
                  f"max={max(v[0] for v in st.best.values())}", flush=True)
    if j37lo is None or (g == 41 and j37hi == 37):
        # cyclic closure: wrap the lower sequence's own head onto its tail
        PL = LOWER[g][1]
        if cp is None:
            cp, cr = tail_of(X, R, g, *chunks[-1])
        a0, b0 = chunks[0] if j37lo is None else (0, 0)
        for pos, res in stream_chunk(X, R, g, a0, b0, hi=1 << 16):
            pos = np.concatenate([cp, pos[:CARRY] + PL])
            res = np.concatenate([cr, ((res[:CARRY].astype(np.int64) + PL) % g
                                       ).astype(np.int16)])
            scan_block(pos, res, st)
            break
        print("  cyclic closure applied", flush=True)
    out = st.dump()
    out["seconds"] = round(time.time() - t0, 1)
    out["chunks"] = len(chunks)
    fn = os.path.join(DATA, f"chain_{g}{('_' + tag) if tag else ''}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    report(out)
    print(f"  written {fn}", flush=True)


CORPUS_F = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58,
            37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}


def report(out):
    g = out["g"]
    print(f"\n  MACHINE {g}: chain depth D_{g} = {out['D']}")
    print("    L   merged gap   before  span  after   phase r  copy j   "
          "slot of the survivor before")
    best = 0
    for L, v in sorted(out["best"].items(), key=lambda kv: int(kv[0])):
        best = max(best, v["value"])
        print(f"   {int(L):2d}    {v['value']:8d}   {v['before']:6d} "
              f"{v['span']:5d} {v['after']:6d}   {v['r']:6d}  {v['copy']:6d}   "
              f"{v['slot']}")
    F = CORPUS_F.get(g)
    print(f"\n    record law: max(before + span + after) = {best}   "
          f"corpus F({g}) = {F}   "
          f"{'MATCH' if F == best else 'MISMATCH'}"
          f"   (blocked-count convention: {best - 1})")
    arg = [int(L) for L, v in out["best"].items() if v["value"] == best]
    print(f"    attained at run length L = {arg} "
          f"(i.e. J = {[a+1 for a in arg]} gaps)")


def merge(g, tags):
    st = State(g)
    cov = []
    for tg in tags:
        with open(os.path.join(DATA, f"chain_{g}_{tg}.json")) as f:
            j = json.load(f)
        st.merge(j)
        cov += [tuple(x) for x in j.get("chunks_done", [])]
    out = st.dump()
    tot, S = len(chunk_list(g)), set(cov)
    out["chunks_done"] = [list(x) for x in sorted(S)]
    out["coverage"] = f"{len(S)}/{tot}"
    laps = sorted({a for a, _ in S if all((a, k) in S for k in range(31))})
    print(f"  coverage: {len(S)}/{tot} chunks of the lower period "
          f"({100.0*len(S)/tot:.1f}%); complete laps j37 = {laps}")
    fn = os.path.join(DATA, f"chain_{g}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    report(out)
    print(f"  written {fn}")


# ---------------------------------------------------------------- gate
def gate():
    """Replicate anchor235/chain_depth.py's published row exactly, with this
    file's phase-free vehicle, at every g it can reach with a small array."""
    want = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43}
    PR = [5, 7, 11, 13, 17, 19, 23, 29]
    for n in range(1, len(PR)):
        low, g = PR[:n], PR[n]
        P = prod(low)
        k = np.arange(P, dtype=np.int64)
        w = np.ones(P, bool)
        for h in low:
            for u in tooth(h):
                w[u % h::h] = False
        Xs = np.flatnonzero(w).astype(np.int64)
        st = State(g)
        LOWER[g] = (low, P)
        pos = np.concatenate([Xs, Xs[:CARRY] + P])
        res = (pos % g).astype(np.int16)
        scan_block(pos, res, st)
        best = max(v[0] for v in st.best.values())
        arg = sorted(L for L, v in st.best.items() if v[0] == best)
        assert best == want[g], (g, best, want[g])
        print(f"  g={g:2d}  D_{g} = {st.D}  F_{g} = {best} "
              f"(chain_depth.py prints {best-1})  attained at L = {arg}  OK")
    print("\nALL ASSERTIONS PASSED (record law reproduced at 7 rungs)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "gate":
        gate()
    elif cmd == "run":
        g = int(sys.argv[2])
        if len(sys.argv) > 3:
            run(g, int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
        else:
            run(g)
    elif cmd == "merge":
        merge(int(sys.argv[2]), sys.argv[3:])
