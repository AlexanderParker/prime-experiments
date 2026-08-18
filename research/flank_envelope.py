"""Round 17 (mechanic): THE FLANK-ENVELOPE CENSUS.

For a step M -> q' the tolerance route's open part (D) is

    FS_max(w) <= F(M) + q' - span(w)   for every compatible qualifying word w

with FS = (gap before the word's first opening) + (gap after its last), and
span(w) = sum of the word's letters.  This tool measures, exactly and at full
period where reachable:

 (1) per compatible word: occurrence count, the JOINT (gL, gR) distribution,
     FS_max with the ADDRESS of the attaining occurrence, the largest single
     flank with its address, and both (D) margins (alpha = 3 and 2.5);
 (2) the UNCONDITIONAL envelope: for every word length ell and every span s
     (any letters, compatible or not), the max single flank and max flank sum
     over all occurrences - the configuration-free version of the Constructor's
     monotone envelope;
 (3) the SPECTRUM CEILING that bounds both: an occurrence of a length-ell word
     is ell+2 consecutive gaps, so span + FS <= F_{ell+2}(M) IDENTICALLY.
     Hence (D) at a step is IMPLIED by F_{ell+2}(M) <= F(M) + q' for every
     compatible word length ell.  F_1..F_8 are computed in the same pass.

Streaming, chunk-flushed, resumable-free (single pass), full period unless
--limit.  Wrap-around handled for full-period runs.

Usage: uv run python research/flank_envelope.py y q' [--limit N] [--seg N]
"""
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")

MAXJ = 8          # spectrum depth F_1..F_MAXJ
MAXELL = 6        # unconditional envelope word lengths 1..MAXELL
SMAX = 640        # span axis for the unconditional envelope
FMAX = 128        # flank axis for the joint histograms (gaps < 128 here)
FS2 = 256         # flank-sum axis


_GLOB = {}


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s).tolist()


def literal_cap(q1):
    """Exact max member count of a literal alternating chain mod 35."""
    E35 = np.ones(35, bool)
    for q in (5, 7):
        c = pow(6, -1, q)
        E35[c::q] = False
        E35[(q - c) % q::q] = False
    s1 = (2 * round(q1 / 6)) % 35
    best = 0
    for r in range(35):
        for phase in (0, 1):
            run = mx = 0
            for i in range(140):
                j, par = divmod(i + phase, 2)
                pos = (r + j * q1 + (s1 if par else 0)) % 35
                if E35[pos]:
                    run += 1
                    mx = max(mx, run)
                else:
                    run = 0
            best = max(best, mx)
    return best


def words(q1):
    a = 2 * round(q1 / 6)
    b = q1 - a
    L = literal_cap(q1)
    out = []
    for ell in range(1, L):
        for start in (a, b):
            w = tuple(start if j % 2 == 0 else (a + b - start)
                      for j in range(ell))
            if w not in out:
                out.append(w)
    return out


def valid_starts(w, q1):
    c = pow(6, -1, q1)
    teeth = {c % q1, (q1 - c) % q1}
    out = []
    for r in sorted(teeth):
        p, ok = r, True
        for x in w:
            p = (p + x) % q1
            if p not in teeth:
                ok = False
                break
        if ok:
            out.append(r)
    return out


class WordState:
    __slots__ = ("w", "ell", "span", "starts", "occ", "joint",
                 "fs_max", "fs_pair", "fs_addr", "mf_max", "mf_addr",
                 "mf_side", "top")

    def __init__(self, w, starts):
        self.w = w
        self.ell = len(w)
        self.span = sum(w)
        self.starts = starts
        self.occ = 0
        self.joint = np.zeros((FMAX, FMAX), np.int64)
        self.fs_max = -1
        self.fs_pair = (0, 0)
        self.fs_addr = -1
        self.mf_max = -1
        self.mf_addr = -1
        self.mf_side = ""
        self.top = []      # (fs, addr, gL, gR) for the top few occurrences


def run(y, q1, limit=None, seg=64_000_000, verbose=True):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, limit)
    full = K >= P
    uvals = [pow(6, -1, g) for g in gears]

    W = [WordState(w, valid_starts(w, q1)) for w in words(q1)]
    maxell = max([ws.ell for ws in W] + [MAXELL])
    ctx = maxell + 4                       # openings of overlap to carry

    Fj = np.zeros(MAXJ + 1, np.int64)
    ghist = np.zeros(FMAX, np.int64)
    # unconditional envelope: per ell, per span -> max single flank / max FS
    Emax = np.full((MAXELL + 1, SMAX), -1, np.int32)
    Hmax = np.full((MAXELL + 1, SMAX), -1, np.int32)
    Ecnt = np.zeros((MAXELL + 1, SMAX), np.int64)
    Haddr = np.full((MAXELL + 1, SMAX), -1, np.int64)
    _GLOB.clear()
    for e in range(1, MAXELL + 1):
        _GLOB[e] = (-1, -1, -1, -1, -1)

    tail = np.array([], dtype=np.int64)
    first_ops = None
    total_open = 0
    t0 = time.time()
    a = 0
    while a < K:
        b = min(K, a + seg)
        ex = np.zeros(b - a, bool)
        for g, u in zip(gears, uvals):
            ex[(u - a) % g::g] = True
            ex[(-u - a) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + a
        total_open += len(op)
        if first_ops is None and len(op) >= ctx:
            first_ops = op[:ctx].copy()
        ops = np.concatenate([tail, op])
        _process(ops, a, Fj, Emax, Hmax, Ecnt, Haddr, W, ghist)
        tail = ops[-ctx:].copy() if len(ops) >= ctx else ops.copy()
        a = b
        if verbose:
            print(f"  seg to {a:.4g} ({100*a/K:.1f}%) "
                  f"{time.time()-t0:.0f}s", flush=True)
    if full and first_ops is not None:
        # close the cycle: the period wraps, so the first openings recur at +P
        ops = np.concatenate([tail, first_ops + P])
        _process(ops, tail[-1] + 1, Fj, Emax, Hmax, Ecnt, Haddr, W, ghist)

    return dict(y=y, q1=q1, P=P, K=K, full=full, openings=total_open,
                gears=gears, Fj=Fj, W=W, Emax=Emax, Hmax=Hmax, Ecnt=Ecnt,
                Haddr=Haddr, glob=dict(_GLOB), ghist=ghist,
                secs=time.time() - t0)


def _process(ops, anew, Fj, Emax, Hmax, Ecnt, Haddr, W, ghist):
    """Fold one overlapping opening block.  An occurrence is counted iff its
    LAST needed opening is new (>= anew) - each is seen exactly once."""
    if len(ops) < 3:
        return
    d = np.diff(ops)
    c = np.concatenate([[0], np.cumsum(d)])
    dn = d[ops[1:] >= anew]
    if len(dn):
        ghist += np.bincount(np.minimum(dn, FMAX - 1), minlength=FMAX)
    for j in range(1, MAXJ + 1):
        if len(d) >= j:
            Fj[j] = max(Fj[j], int((c[j:] - c[:-j]).max()))

    # unconditional envelope: word = d[i..i+ell-1], flanks d[i-1], d[i+ell]
    # aggregated by bincount over (span, value) keys - no sorts, O(n) per ell
    n = len(d)
    for ell in range(1, MAXELL + 1):
        if n < ell + 2:
            break
        i = np.arange(1, n - ell)                    # word start index
        span = (c[i + ell] - c[i]).astype(np.int64)
        lf = d[i - 1].astype(np.int64)
        rf = d[i + ell].astype(np.int64)
        sel = (ops[i + ell + 1] >= anew) & (span < SMAX)
        if not sel.any():
            continue
        s = span[sel]
        lfs, rfs = lf[sel], rf[sel]
        mf = np.maximum(lfs, rfs)
        fs = lfs + rfs
        cnt = np.bincount(s * FS2 + fs,
                          minlength=SMAX * FS2).reshape(SMAX, FS2)
        has = cnt > 0
        Ecnt[ell] += cnt.sum(1)
        rowmax = np.where(has.any(1),
                          FS2 - 1 - np.argmax(has[:, ::-1], axis=1), -1)
        np.maximum(Hmax[ell], rowmax.astype(np.int32), out=Hmax[ell])
        cnt2 = np.bincount(s * FMAX + np.minimum(mf, FMAX - 1),
                           minlength=SMAX * FMAX).reshape(SMAX, FMAX)
        has2 = cnt2 > 0
        rowmax2 = np.where(has2.any(1),
                           FMAX - 1 - np.argmax(has2[:, ::-1], axis=1), -1)
        np.maximum(Emax[ell], rowmax2.astype(np.int32), out=Emax[ell])
        # address of the per-ell global max flank sum
        j = int(np.argmax(fs))
        if int(fs[j]) > _GLOB[ell][0]:
            _GLOB[ell] = (int(fs[j]), int(ops[i[sel][j]]), int(s[j]),
                          int(lfs[j]), int(rfs[j]))

    # per-word census
    for ws in W:
        ell = ws.ell
        if n < ell + 2:
            continue
        m = d[1:n - ell] == ws.w[0]
        for j in range(1, ell):
            m &= d[1 + j:n - ell + j] == ws.w[j]
        idx = np.flatnonzero(m) + 1
        if len(idx) == 0:
            continue
        idx = idx[ops[idx + ell + 1] >= anew]
        if len(idx) == 0:
            continue
        lf = d[idx - 1].astype(np.int64)
        rf = d[idx + ell].astype(np.int64)
        ws.occ += len(idx)
        ok = (lf < FMAX) & (rf < FMAX)
        ws.joint += np.bincount(lf[ok] * FMAX + rf[ok],
                                minlength=FMAX * FMAX).reshape(FMAX, FMAX)
        fs = lf + rf
        j = int(np.argmax(fs))
        if int(fs[j]) > ws.fs_max:
            ws.fs_max = int(fs[j])
            ws.fs_pair = (int(lf[j]), int(rf[j]))
            ws.fs_addr = int(ops[idx[j]])
        jl, jr = int(np.argmax(lf)), int(np.argmax(rf))
        if int(lf[jl]) >= int(rf[jr]):
            cand, addr, side = int(lf[jl]), int(ops[idx[jl]]), "L"
        else:
            cand, addr, side = int(rf[jr]), int(ops[idx[jr]]), "R"
        if cand > ws.mf_max:
            ws.mf_max, ws.mf_addr, ws.mf_side = cand, addr, side
        # keep the top-5 flank sums overall
        k = min(5, len(fs))
        sel = np.argpartition(fs, -k)[-k:]
        for t in sel:
            ws.top.append((int(fs[t]), int(ops[idx[t]]),
                           int(lf[t]), int(rf[t])))
        ws.top = sorted(ws.top, reverse=True)[:5]


def report(r):
    y, q1 = r["y"], r["q1"]
    Fj = r["Fj"]
    F = int(Fj[1])
    cov = r["K"] / r["P"]
    tag = "FULL PERIOD" if r["full"] else f"PREFIX {100*cov:.3f}%"
    print(f"\n=== step {y} -> {q1}   {tag}   openings {r['openings']:,} "
          f"  {r['secs']:.0f}s")
    print(f"  F_j (j=1..{MAXJ}): " + " ".join(str(int(x)) for x in Fj[1:]))
    print(f"  increments      : " +
          " ".join(str(int(Fj[j + 1] - Fj[j])) for j in range(1, MAXJ)))
    print(f"  F = {F}   F + q' = {F + q1}   "
          f"(D at alpha=3: span + FS <= {F + q1})")
    print("  SPECTRUM SUFFICIENCY (span+FS <= F_{ell+2} identically):")
    for ell in range(1, MAXJ - 1):
        v = int(Fj[ell + 2])
        print(f"    ell={ell}: F_{ell+2} = {v:4d}  vs F+q' = {F+q1:4d}  "
              f"{'IMPLIES (D)' if v <= F + q1 else f'short by {v-F-q1}'}")
    print("  WORDS (compatible = the letter walk can start on a tooth):")
    rows = []
    for ws in sorted(r["W"], key=lambda z: (z.span, z.ell)):
        comp = bool(ws.starts)
        if ws.occ == 0:
            print(f"    w={str(ws.w):18s} span={ws.span:3d} "
                  f"{'COMPAT' if comp else 'incomp'}  occ=0")
            continue
        need3 = F + q1 - ws.span
        ceil_sp = int(Fj[ws.ell + 2]) - ws.span if ws.ell + 2 <= MAXJ else -1
        print(f"    w={str(ws.w):18s} span={ws.span:3d} ell={ws.ell} "
              f"{'COMPAT' if comp else 'incomp'} occ={ws.occ:>12,} "
              f"FS_max={ws.fs_max:3d} at ({ws.fs_pair[0]},{ws.fs_pair[1]}) "
              f"k={ws.fs_addr:,}  maxflank={ws.mf_max:3d} "
              f"({ws.mf_max/F:.2f}F, {ws.mf_side} @ k={ws.mf_addr:,})  "
              f"need3={need3:4d} margin3={need3-ws.fs_max:+5d}  "
              f"specceil={ceil_sp:4d} slack={ceil_sp-ws.fs_max:+5d}")
        rows.append(ws)
    comp_all = [ws.ell for ws in r["W"] if ws.starts]
    if comp_all:
        La = max(comp_all)
        needa = int(Fj[La + 2]) if La + 2 <= MAXJ else -1
        print(f"  A PRIORI max compatible word length (all, occ or not) "
              f"L_cap = {La}; F_{La+2} = {needa} vs F+q' = {F+q1}: "
              f"{'(D) PROVED A PRIORI BY SPECTRUM' if 0 <= needa <= F+q1 else 'NOT implied'}")
    comp_ells = [ws.ell for ws in r["W"] if ws.starts and ws.occ > 0]
    if comp_ells:
        L = max(comp_ells)
        need = int(Fj[L + 2]) if L + 2 <= MAXJ else -1
        print(f"  MAX COMPATIBLE WORD LENGTH L = {L}; spectrum test "
              f"F_{L+2} = {need} vs F+q' = {F+q1}: "
              f"{'(D) PROVED BY SPECTRUM' if 0 <= need <= F+q1 else 'not implied'}")
        cm = min((F + q1 - ws.span - ws.fs_max)
                 for ws in r["W"] if ws.starts and ws.occ > 0)
        print(f"  MIN (D) MARGIN over compatible words: {cm:+d} "
              f"({cm/q1:.3f} q')")
    gh = r["ghist"]
    tot = int(gh.sum())
    tail = np.cumsum(gh[::-1])[::-1]        # tail[g] = #gaps >= g
    print("  RARITY NULL (flanks drawn independently from the machine's own "
          "gap distribution):")
    for ws in sorted(r["W"], key=lambda z: (z.span, z.ell)):
        if ws.occ == 0:
            continue
        # predicted max flank: largest g with 2*occ*P(gap>=g) >= 1
        pred = 0
        for g in range(FMAX - 1, 0, -1):
            if 2 * ws.occ * tail[g] >= tot:
                pred = g
                break
        # predicted max flank SUM: largest v with occ*P(gL+gR>=v) >= 1
        pm = gh / tot
        conv = np.convolve(pm, pm)
        ctail = np.cumsum(conv[::-1])[::-1]
        predfs = 0
        for v in range(len(ctail) - 1, 0, -1):
            if ws.occ * ctail[v] >= 1:
                predfs = v
                break
        print(f"    w={str(ws.w):18s} occ={ws.occ:>12,}  maxflank obs="
              f"{ws.mf_max:3d} null={pred:3d}   FS_max obs={ws.fs_max:3d} "
              f"null={predfs:3d}   (obs-null: {ws.mf_max-pred:+d}, "
              f"{ws.fs_max-predfs:+d})")
    print("  UNCONDITIONAL ENVELOPE  E(ell,s) = max single flank, "
          "H = max flank sum (any letters):")
    for ell in range(1, MAXELL + 1):
        ss = np.flatnonzero(r["Ecnt"][ell] > 0)
        if len(ss) == 0:
            continue
        top = [(int(s), int(r["Emax"][ell][s]), int(r["Hmax"][ell][s]))
               for s in ss]
        viol = [(s1, s2) for i, (s1, e1, _) in enumerate(top)
                for (s2, e2, _) in top[i + 1:] if e2 > e1]
        print(f"    ell={ell}: spans {ss.min()}..{ss.max()} "
              f"({len(ss)} values), maxE={max(t[1] for t in top)}, "
              f"maxH={max(t[2] for t in top)}, "
              f"monotone violations (s'>s with E(s')>E(s)): {len(viol)}")
        g = r["glob"][ell]
        print(f"           global max FS = {g[0]} at k={g[1]:,} "
              f"(span {g[2]}, flanks {g[3]},{g[4]})")


def write_csv(r):
    y, q1 = r["y"], r["q1"]
    os.makedirs(DDIR, exist_ok=True)
    F = int(r["Fj"][1])
    p1 = os.path.join(DDIR, "flank_envelope_words.csv")
    new = not os.path.exists(p1) or os.path.getsize(p1) == 0
    with open(p1, "a") as f:
        if new:
            f.write("y,qp,coverage,word,ell,span,compat,starts,occ,F,"
                    "Fell2,FSmax,gL,gR,addr_fs,maxflank,addr_mf,side,"
                    "need3,margin3,specceil,specslack\n")
        for ws in r["W"]:
            if ws.occ == 0:
                continue
            fe = int(r["Fj"][ws.ell + 2]) if ws.ell + 2 <= MAXJ else -1
            need3 = F + q1 - ws.span
            f.write(f"{y},{q1},{r['K']/r['P']:.6f},"
                    f"\"{'-'.join(map(str, ws.w))}\",{ws.ell},{ws.span},"
                    f"{int(bool(ws.starts))},\"{ws.starts}\",{ws.occ},{F},"
                    f"{fe},{ws.fs_max},{ws.fs_pair[0]},{ws.fs_pair[1]},"
                    f"{ws.fs_addr},{ws.mf_max},{ws.mf_addr},{ws.mf_side},"
                    f"{need3},{need3-ws.fs_max},{fe-ws.span},"
                    f"{fe-ws.span-ws.fs_max}\n")
    p2 = os.path.join(DDIR, "flank_envelope_joint.csv")
    new = not os.path.exists(p2) or os.path.getsize(p2) == 0
    with open(p2, "a") as f:
        if new:
            f.write("y,qp,word,span,gL,gR,count\n")
        for ws in r["W"]:
            if ws.occ == 0:
                continue
            nz = np.flatnonzero(ws.joint.ravel())
            for z in nz:
                gl, gr = divmod(int(z), FMAX)
                f.write(f"{y},{q1},\"{'-'.join(map(str, ws.w))}\","
                        f"{ws.span},{gl},{gr},{int(ws.joint[gl, gr])}\n")
    p3 = os.path.join(DDIR, "flank_envelope_uncond.csv")
    new = not os.path.exists(p3) or os.path.getsize(p3) == 0
    with open(p3, "a") as f:
        if new:
            f.write("y,qp,ell,span,count,maxflank,maxFS,addr_maxFS\n")
        for ell in range(1, MAXELL + 1):
            for s in np.flatnonzero(r["Ecnt"][ell] > 0):
                f.write(f"{y},{q1},{ell},{int(s)},{int(r['Ecnt'][ell][s])},"
                        f"{int(r['Emax'][ell][s])},{int(r['Hmax'][ell][s])},"
                        f"{int(r['Haddr'][ell][s])}\n")
    p4 = os.path.join(DDIR, "flank_envelope_spectra.csv")
    new = not os.path.exists(p4) or os.path.getsize(p4) == 0
    with open(p4, "a") as f:
        if new:
            f.write("y,period,scanned,coverage," +
                    ",".join(f"F{j}" for j in range(1, MAXJ + 1)) + "\n")
        f.write(f"{y},{r['P']},{r['K']},{r['K']/r['P']:.6f}," +
                ",".join(str(int(r["Fj"][j])) for j in range(1, MAXJ + 1))
                + "\n")
    p5 = os.path.join(DDIR, "flank_envelope_gaphist.csv")
    new = not os.path.exists(p5) or os.path.getsize(p5) == 0
    with open(p5, "a") as f:
        if new:
            f.write("y,coverage,gap,count\n")
        for g in np.flatnonzero(r["ghist"]):
            f.write(f"{y},{r['K']/r['P']:.6f},{int(g)},"
                    f"{int(r['ghist'][g])}\n")
    print(f"  wrote {p1}, {p2}, {p3}, {p4}, {p5}")


def main():
    args = sys.argv[1:]
    limit = None
    seg = 64_000_000
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(float(args[i + 1]))
        del args[i:i + 2]
    if "--seg" in args:
        i = args.index("--seg")
        seg = int(float(args[i + 1]))
        del args[i:i + 2]
    global MAXELL
    if "--maxell" in args:
        i = args.index("--maxell")
        MAXELL = int(args[i + 1])
        del args[i:i + 2]
    quiet = "--quiet" in args
    if quiet:
        args.remove("--quiet")
    y, q1 = int(args[0]), int(args[1])
    r = run(y, q1, limit=limit, seg=seg, verbose=not quiet)
    report(r)
    write_csv(r)
    for ws in r["W"]:
        if ws.occ and ws.top:
            print(f"  TOP-5 flank sums, w={ws.w}: " +
                  ", ".join(f"FS={a} at k={b:,} ({c},{d})"
                            for a, b, c, d in ws.top))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
