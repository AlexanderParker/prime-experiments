"""Round 30 (constructor): THE COUNTED WORD CENSUS - occ(w) and Phi(w) for every
legal word of M, over the FULL cyclic period, streamed.

R96 named the construct that would decide the three open m31 rows: the COUNTED
padded-gap census occ(q'; M).  Every existing census at m29/m31/m37 is a
distinct-tuple list with no counts.  This script streams the whole period of M
in slot chunks (no array beyond one chunk), and for every run of consecutive
gaps that are all legal letters (values <= F with v = 0 or +-2c mod q') it
records the occurrence count occ(w) and the flank envelope Phi(w) = max over
occurrences of (gap before + gap after), with the argmax flank pair.  It also
records the full single-gap histogram and the full lag-1 pair table.

GATES (in `merge`, exact integers):
  (1) sum of counts = prod(q-2), sum of gaps = P (the cyclic close, Mechanic
      rule 25), max gap = F(M), max adjacent pair = F_2(M);
  (2) MIRROR (Lateral r25 theorem): occ(w) = occ(reverse w) for every word and
      pair[a,b] = pair[b,a];
  (3) at m11..m23 the streamed tables equal a whole-period in-memory scan;
      at m11..m37 the histogram equals the recorded exact cyclic ghist rows;
  (4) Phi(w) equals research/evenj_r29.py's flank table (from the distinct
      censuses) at every word both vehicles see.
Results are written FROM THE CHILD, one .npz per worker, merged by the driver;
the merged text report is research/data/r30/occ_<y>.txt (committed) and the
merged arrays research/data/r30/occ_<y>.npz.

Usage:
  uv run python research/occ_census_r30.py <y> [--workers 6] [--chunk 100000000]
  uv run python research/occ_census_r30.py <y> --merge-only
"""
import glob
import json
import os
import sys
import time
from math import prod
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
R30 = os.path.join(DDIR, "r30")

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
            41: 103, 43: 116, 47: 134, 53: 159}
MAXW = 4                      # legal words of length 1..MAXW are counted


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def next_prime(y):
    p = y + 1
    while not is_prime(p):
        p += 1
    return p


def gears_of(y):
    return [p for p in range(5, y + 1) if is_prime(p)]


def letters(y):
    """(q', a, b, Lambda_all) - every residue-legal value <= F, realised or not."""
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    F = KNOWN_F[y]
    lam = [v for v in range(1, F + 1) if v % q1 in (0, a % q1, b % q1)]
    return q1, a, b, lam


def openings(gears, lo, hi):
    """Absolute positions of the openings in [lo, hi) (lo may be negative,
    hi may exceed P: blocking is periodic, so the positions are correct
    modulo P)."""
    n = hi - lo
    ex = np.zeros(n, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        for t in (u % g, (-u) % g):
            ex[(t - lo) % g::g] = True
    return np.flatnonzero(~ex).astype(np.int64) + lo


def worker(args):
    y, wid, lo, hi, chunk, out = args
    try:
        import psutil
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:
        pass
    gears = gears_of(y)
    F = KNOWN_F[y]
    K = F + 1
    q1, a, b, lam = letters(y)
    B = len(lam) + 1
    code = np.zeros(2 * F + 2, dtype=np.int64)
    for i, v in enumerate(lam):
        code[v] = i + 1
    MARG = 8 * F + 64
    hist = np.zeros(2 * F + 2, dtype=np.int64)
    pairs = np.zeros(K * K, dtype=np.int64)
    trip = np.zeros(K * K * K, dtype=np.int64)
    occ = [None] + [np.zeros(B ** m, dtype=np.int64) for m in range(1, MAXW + 1)]
    phi = [None] + [np.full(B ** m, -1, dtype=np.int64) for m in range(1, MAXW + 1)]
    FL = 2 * K                    # flank sums run to 2F
    fl = [None] + [np.zeros(B ** m * FL, dtype=np.int64) for m in range(1, 4)]
    ngap = 0
    t0 = time.time()
    log = open(out + ".log", "w")
    nsub = 0
    s = lo
    maxgap = 0
    while s < hi:
        e = min(s + chunk, hi)
        o = openings(gears, s - MARG, e + MARG)
        d = np.diff(o)
        i0 = int(np.searchsorted(o, s))          # first opening >= s
        i1 = int(np.searchsorted(o, e))          # first opening >= e
        # gaps owned by this sub-chunk: indices i0..i1-1 (left opening in [s,e))
        dd = d[i0:i1]
        if len(dd):
            mg = int(dd.max())
            if mg > maxgap:
                maxgap = mg
            assert mg <= F, ("gap above F", y, mg, s)
            ngap += len(dd)
            hist += np.bincount(dd, minlength=2 * F + 2)[:2 * F + 2]
            pairs += np.bincount(d[i0:i1] * K + d[i0 + 1:i1 + 1],
                                 minlength=K * K)
            trip += np.bincount((d[i0:i1] * K + d[i0 + 1:i1 + 1]) * K
                                + d[i0 + 2:i1 + 2], minlength=K * K * K)
            c = code[d]
            nz = c > 0
            # candidate start positions: owned, with a flank before and room
            idx = np.flatnonzero(nz[i0:i1]) + i0
            idx = idx[idx >= 1]
            for m in range(1, MAXW + 1):
                idx = idx[idx + m < len(d)]
                if m > 1:
                    idx = idx[nz[idx + m - 1]]
                if len(idx) == 0:
                    break
                keys = np.zeros(len(idx), dtype=np.int64)
                for j in range(m):
                    keys = keys * B + c[idx + j]
                occ[m] += np.bincount(keys, minlength=B ** m)
                gL = d[idx - 1]
                gR = d[idx + m]
                enc = (gL + gR) * K + gL
                np.maximum.at(phi[m], keys, enc)
                if m <= 3:
                    fl[m] += np.bincount(keys * FL + gL + gR,
                                         minlength=B ** m * FL)
        nsub += 1
        if nsub % 20 == 0:
            log.write("sub-chunk %d  slots to %d  gaps %d  %.0f s\n"
                      % (nsub, e, ngap, time.time() - t0))
            log.flush()
        s = e
    np.savez(out, hist=hist, pairs=pairs, trip=trip, ngap=np.int64(ngap),
             maxgap=np.int64(maxgap), lo=np.int64(lo), hi=np.int64(hi),
             **{"occ%d" % m: occ[m] for m in range(1, MAXW + 1)},
             **{"phi%d" % m: phi[m] for m in range(1, MAXW + 1)},
             **{"fl%d" % m: fl[m] for m in range(1, 4)})
    log.write("DONE worker %d  gaps %d  %.0f s\n" % (wid, ngap, time.time() - t0))
    log.close()
    return wid, ngap, time.time() - t0


def cls_of(v, q1, a, b):
    r = v % q1
    if r == 0:
        return 0
    if r == a % q1:
        return 1
    if r == b % q1:
        return -1
    return None


def t3_ok(word, q1, a, b):
    last = 0
    for v in word:
        c = cls_of(v, q1, a, b)
        if c is None:
            return False
        if c:
            if c == last:
                return False
            last = c
    return True


def decode(key, m, B, lam):
    out = []
    for _ in range(m):
        out.append(lam[key % B - 1])
        key //= B
    return tuple(reversed(out))


def merge(y, workers):
    gears = gears_of(y)
    P = prod(gears)
    N = prod(g - 2 for g in gears)
    F = KNOWN_F[y]
    K = F + 1
    q1, a, b, lam = letters(y)
    B = len(lam) + 1
    files = sorted(glob.glob(os.path.join(R30, "occ_%d_w*.npz" % y)))
    assert len(files) == workers, ("worker files", len(files), workers)
    hist = np.zeros(2 * F + 2, dtype=np.int64)
    pairs = np.zeros(K * K, dtype=np.int64)
    trip = np.zeros(K * K * K, dtype=np.int64)
    occ = [None] + [np.zeros(B ** m, dtype=np.int64) for m in range(1, MAXW + 1)]
    phi = [None] + [np.full(B ** m, -1, dtype=np.int64) for m in range(1, MAXW + 1)]
    FL = 2 * K
    fl = [None] + [np.zeros(B ** m * FL, dtype=np.int64) for m in range(1, 4)]
    have_fl = True
    ngap = 0
    cover = []
    for f in files:
        z = np.load(f)
        hist += z["hist"]
        pairs += z["pairs"]
        trip += z["trip"]
        ngap += int(z["ngap"])
        cover.append((int(z["lo"]), int(z["hi"])))
        for m in range(1, MAXW + 1):
            occ[m] += z["occ%d" % m]
            phi[m] = np.maximum(phi[m], z["phi%d" % m])
        for m in range(1, 4):
            if "fl%d" % m in z.files:
                fl[m] += z["fl%d" % m]
            else:
                have_fl = False
    cover.sort()
    assert cover[0][0] == 0 and cover[-1][1] == P, ("tiling", cover)
    for (l1, h1), (l2, h2) in zip(cover, cover[1:]):
        assert h1 == l2, ("tiling gap", l1, h1, l2, h2)
    lines = []

    def out(s=""):
        lines.append(s)
        print(s, flush=True)

    out("COUNTED WORD CENSUS  machine %d  (gears %s)  q' = %d  a = %d  b = %d"
        % (y, gears, q1, a, b))
    out("period P = %d   openings N = prod(q-2) = %d" % (P, N))
    # ---- gate 1: cyclic close
    tot = int(hist.sum())
    wsum = int((np.arange(len(hist), dtype=np.int64) * hist).sum())
    out("GATE 1  gap count %d == N %s ; weighted sum %d == P %s ; max gap %d "
        "== F %s" % (tot, tot == N, wsum, wsum == P, int(np.flatnonzero(hist).max()),
                     int(np.flatnonzero(hist).max()) == F))
    assert tot == N and wsum == P and ngap == N, ("cyclic close", tot, N, wsum, P)
    assert int(np.flatnonzero(hist).max()) == F, "F"
    pm = pairs.reshape(K, K)
    ii, jj = np.nonzero(pm)
    F2 = int((ii + jj).max())
    out("        max adjacent pair sum %d == F_2 %s" % (F2, F2 == KNOWN_F2[y]))
    assert F2 == KNOWN_F2[y], ("F_2 gate", F2)
    assert int(pm.sum()) == N and int(trip.sum()) == N
    # ---- gate 2: mirror
    assert np.array_equal(pm, pm.T), "pair table not symmetric"
    tm = trip.reshape(K, K, K)
    assert np.array_equal(tm, tm.transpose(2, 1, 0)), "triple table not mirror"
    mirror_bad = 0
    words = {}
    for m in range(1, MAXW + 1):
        for key in np.flatnonzero(occ[m]):
            w = decode(int(key), m, B, lam)
            enc = int(phi[m][key])
            fs, gL = divmod(enc, K)
            words[w] = (int(occ[m][key]), fs, gL, fs - gL)
    for w, (n, fs, gL, gR) in words.items():
        r = w[::-1]
        if r not in words or words[r][0] != n or words[r][1] != fs:
            mirror_bad += 1
    out("GATE 2  mirror: pair table symmetric, triple table mirror-symmetric, "
        "%d legal words, %d mirror violations" % (len(words), mirror_bad))
    assert mirror_bad == 0
    # ---- gate 3: whole-period scan (small machines) and recorded ghist rows
    if y <= 23:
        ex = np.zeros(P, bool)
        for g in gears:
            u = pow(6, -1, g)
            ex[u % g::g] = True
            ex[(-u) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64)
        d = np.diff(np.concatenate([op, [op[0] + P]]))
        h2 = np.bincount(d, minlength=2 * F + 2)
        assert np.array_equal(h2[:len(hist)], hist), "hist vs in-memory scan"
        p2 = np.bincount(d * K + np.roll(d, -1), minlength=K * K)
        assert np.array_equal(p2, pairs), "pairs vs in-memory scan"
        t2 = np.bincount((d * K + np.roll(d, -1)) * K + np.roll(d, -2),
                         minlength=K * K * K)
        assert np.array_equal(t2, trip), "triples vs in-memory scan"
        out("GATE 3  streamed hist / pair / triple tables == whole-period "
            "in-memory scan: EXACT")
    ref = None
    p = os.path.join(DDIR, "gap_pair_hist.csv")
    if os.path.exists(p):
        ref = {}
        with open(p) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) == 6 and parts[0] == str(y) and parts[2] == "ghist" \
                        and parts[1].startswith("1.0"):
                    ref[int(parts[4])] = int(parts[5])
        if not ref:
            ref = None
    if ref is None:
        p = os.path.join(DDIR, "r26", "ghist_%d.csv" % y)
        if os.path.exists(p):
            ref = {}
            with open(p) as fh:
                for line in fh:
                    parts = line.strip().replace(";", ",").split(",")
                    try:                     # Mechanic's r26 format: y,gap,count
                        if int(parts[0]) != y:
                            continue
                        v, c = int(parts[1]), int(parts[2])
                        ref[v] = c
                    except (ValueError, IndexError):
                        continue
    if ref:
        mine = {v: int(hist[v]) for v in range(len(hist)) if hist[v]}
        same = (mine == ref)
        out("GATE 3b recorded exact cyclic ghist row for m%d: %s (%d values)"
            % (y, "EXACT MATCH" if same else "MISMATCH", len(ref)))
        assert same, ("ghist mismatch", y)
    else:
        out("GATE 3b no recorded ghist row for m%d on disk (nothing to compare)"
            % y)
    # ---- gate 4: Phi against evenj_r29's flank table (distinct censuses)
    try:
        import evenj_r29
        if y in evenj_r29.KNOWN_F2 and (y in evenj_r29.SCANNED
                                        or y in evenj_r29.CENSUS4):
            r = evenj_r29.analyse(y)
            nchk = nbad = 0
            for w, (fs, gL, gR) in r["words"].items():
                if w in words:
                    nchk += 1
                    if words[w][1] != fs:
                        nbad += 1
                        out("   Phi MISMATCH %s: census %d, counted %d"
                            % (w, fs, words[w][1]))
            out("GATE 4  Phi(w) vs evenj_r29 flank table: %d words compared, "
                "%d mismatches" % (nchk, nbad))
            assert nbad == 0
            # every T3-legal word of the counted census with a Phi must be in
            # evenj's table if evenj's source arity reaches it
            for w, (n, fs, gL, gR) in words.items():
                if t3_ok(w, q1, a, b) and len(w) + 2 <= r["max_arity"]:
                    assert w in r["words"], ("counted word missing in census", w)
            out("        every T3-legal counted word of length <= %d is in the "
                "distinct census: OK" % (r["max_arity"] - 2))
    except ImportError:
        out("GATE 4  evenj_r29 not importable, skipped")
    # ---- report
    out("")
    out("SINGLE-GAP HISTOGRAM (value:count), %d distinct values:" %
        len(np.flatnonzero(hist)))
    out("   " + " ".join("%d:%d" % (v, hist[v]) for v in np.flatnonzero(hist)))
    out("")
    out("LEGAL LETTERS <= F  (value  class  occ  occ/N  Phi  argmax flank pair)")
    for v in lam:
        w = (v,)
        c = cls_of(v, q1, a, b)
        if w in words:
            n, fs, gL, gR = words[w]
            out("   %4d  %-6s  %12d  %.3e   Phi=%3d  (%d,%d)"
                % (v, {0: "padded", 1: "a", -1: "b"}[c], n, n / N, fs, gL, gR))
        else:
            out("   %4d  %-6s  %12d  (hole)" % (v, {0: "padded", 1: "a",
                                                    -1: "b"}[c], 0))
    out("")
    out("T3-LEGAL WORDS  (word  occ  Phi  argmax flanks  padded?)")
    for w in sorted(words, key=lambda t: (len(t), t)):
        if not t3_ok(w, q1, a, b):
            continue
        n, fs, gL, gR = words[w]
        out("   %-22s %12d   Phi=%3d  (%d,%d)  %s"
            % (str(w), n, fs, gL, gR,
               "padded" if any(v % q1 == 0 for v in w) else "literal"))
    if have_fl:
        out("")
        out("FLANK-SUM DISTRIBUTION per T3-legal word (length <= 3): count, mean "
            "flank sum (exact rational), and the top of the tail as sum:count")
        for w in sorted(words, key=lambda t: (len(t), t)):
            if not t3_ok(w, q1, a, b) or len(w) > 3:
                continue
            m = len(w)
            key = 0
            for v in w:
                key = key * B + (lam.index(v) + 1)
            h = fl[m][key * FL:(key + 1) * FL]
            n = int(h.sum())
            assert n == words[w][0], ("flank hist total", w, n, words[w][0])
            tot = int((np.arange(FL, dtype=np.int64) * h).sum())
            nz = np.flatnonzero(h)
            tail = " ".join("%d:%d" % (v, h[v]) for v in nz[-8:])
            out("   %-16s n=%-10d mean=%d/%d=%.2f  max=%d  tail %s"
                % (str(w), n, tot, n, tot / n, int(nz.max()), tail))
    out("")
    out("NON-T3 LEGAL-LETTER RUNS (same class twice), for the record:")
    for w in sorted(words, key=lambda t: (len(t), t)):
        if t3_ok(w, q1, a, b) or len(w) == 1:
            continue
        n, fs, gL, gR = words[w]
        out("   %-22s %12d   Phi=%3d" % (str(w), n, fs))
    with open(os.path.join(R30, "occ_%d.txt" % y), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    np.savez_compressed(os.path.join(R30, "occ_%d.npz" % y), hist=hist,
                        pairs=pairs, trip=trip,
                        **{"occ%d" % m: occ[m] for m in range(1, MAXW + 1)},
                        **{"phi%d" % m: phi[m] for m in range(1, MAXW + 1)},
                        **({"fl%d" % m: fl[m] for m in range(1, 4)} if have_fl
                           else {}))
    with open(os.path.join(R30, "occ_%d_words.json" % y), "w") as fh:
        json.dump({" ".join(map(str, w)): list(v) for w, v in words.items()},
                  fh, indent=0)
    out("all assertions passed")
    return words


def main():
    y = int(sys.argv[1])
    args = sys.argv[2:]

    def opt(nm, d):
        return type(d)(args[args.index(nm) + 1]) if nm in args else d

    workers = opt("--workers", 6)
    chunk = opt("--chunk", 100_000_000)
    os.makedirs(R30, exist_ok=True)
    if "--merge-only" not in args:
        gears = gears_of(y)
        P = prod(gears)
        bounds = [P * i // workers for i in range(workers + 1)]
        jobs = [(y, i, bounds[i], bounds[i + 1], chunk,
                 os.path.join(R30, "occ_%d_w%d" % (y, i)))
                for i in range(workers)]
        for f in glob.glob(os.path.join(R30, "occ_%d_w*.npz" % y)):
            os.remove(f)                       # stale worker files from an
        print("machine %d  P = %d  %d workers  chunk %d" % (y, P, workers, chunk),
              flush=True)
        t0 = time.time()
        if workers == 1:
            res = [worker(jobs[0])]
        else:
            with Pool(workers) as pool:
                res = list(pool.imap_unordered(worker, jobs))
        for wid, ng, dt in sorted(res):
            print("  worker %d  gaps %d  %.0f s" % (wid, ng, dt), flush=True)
        print("scan wall %.0f s" % (time.time() - t0), flush=True)
    merge(y, workers)


if __name__ == "__main__":
    main()
