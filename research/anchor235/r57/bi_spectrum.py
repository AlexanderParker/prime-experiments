"""bi_spectrum.py -- the size side: the next machine's whole gap spectrum from the old
machine's J-run size-and-residue dictionary.

    m_{M+q'}(v)  =  sum_{J=1..J_max}  sum_{n : g_n + ... + g_{n+J-1} = v}  eps_J(n)

with eps_J(n) the number of copies in which that J-run fuses into exactly one new gap (bi_core).
Verified against the directly built spectrum at every rung up to 23 -> 29, and against the
corpus gates at 29 -> 31.

Also: the truncation table (what the depth-K dictionary gets wrong for K < J_max) and the size
of the realised J-tuple dictionary D_J(M).

Usage:  uv run python research/anchor235/r57/bi_spectrum.py [maxtop]
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bi_core import (OUT, PRIMES, machine_gaps, letters, word_counts, eps_of_window, wrap_pad)

CHUNK = 20_000_000


def spectrum_by_J(gaps, lt, q, jmax, vmax):
    """hist[J][v] = the multiplicity contributed by J-runs of span v."""
    N = gaps.size
    gp = wrap_pad(gaps, jmax)
    lp = wrap_pad(lt, jmax)
    out = {}
    for J in range(1, jmax + 1):
        h = np.zeros(vmax + 1, dtype=np.int64)
        pos = 0
        while pos < N:
            end = min(pos + CHUNK, N)
            span = np.zeros(end - pos, dtype=np.int32)
            for k in range(J):
                span += gp[pos + k:end + k]
            e = eps_of_window([lp[pos + k:end + k] for k in range(J)], q)
            for w in np.unique(e):
                if w <= 0:
                    continue
                sel = span[e == w]
                if sel.size:
                    h[:] += np.bincount(sel, minlength=vmax + 1)[:vmax + 1] * int(w)
            pos = end
        out[J] = h
    return out


def dict_sizes(gaps, kmax):
    """|D_J(M)| = the number of DISTINCT J-tuples of consecutive gap sizes, J = 1..kmax.
    Streamed: level J's identifiers are rebuilt from level 1 each pass, so nothing of the size
    of the machine is ever held as int64."""
    N = gaps.size
    vals = np.flatnonzero(np.bincount(gaps))
    V = vals.size
    vidx = np.full(int(gaps.max()) + 1, -1, dtype=np.int32)
    vidx[vals] = np.arange(V, dtype=np.int32)
    gp = wrap_pad(gaps, kmax)
    maps = []          # maps[k] : (id at level k+1) from id_k * V + vidx
    sizes = [V]
    nid = V
    for k in range(1, kmax):
        seen = np.zeros(nid * V, dtype=bool)
        pos = 0
        while pos < N:
            end = min(pos + CHUNK, N)
            ids = vidx[gp[pos:end]].astype(np.int64)
            for j in range(1, k):
                ids = maps[j - 1][ids * V + vidx[gp[pos + j:end + j]]]
            code = ids * V + vidx[gp[pos + k:end + k]]
            seen[np.unique(code)] = True
            pos = end
        hit = np.flatnonzero(seen)
        m = np.full(nid * V, -1, dtype=np.int64)
        m[hit] = np.arange(hit.size, dtype=np.int64)
        maps.append(m)
        nid = hit.size
        sizes.append(nid)
    return sizes


def main():
    maxtop = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    lines = []
    W = lines.append
    W("bi_spectrum -- the next machine's spectrum from the old machine's J-run dictionary")
    W("")
    W("rung | J_max | F(M+q') | |Spec| | sum m | sum v m | P(M+q') | direct match | "
      "truncation errors by K")
    tail = []
    for k in range(len(PRIMES) - 1):
        top, q = PRIMES[k], PRIMES[k + 1]
        if top > maxtop:
            break
        t0 = time.time()
        P, f0, gaps = machine_gaps(top)
        N = gaps.size
        lt = letters(gaps, q)
        L = 0
        while word_counts(lt, L + 1)[0] > 0:
            L += 1
        jmax = L + 2
        vmax = int(gaps.max()) * (jmax + 1) + 4
        hists = spectrum_by_J(gaps, lt, q, jmax + 1, vmax)
        total = np.zeros(vmax + 1, dtype=np.int64)
        for J in range(1, jmax + 2):
            total += hists[J]
        Fnew = int(np.flatnonzero(total).max())
        spec = int((total > 0).sum())
        summ = int(total.sum())
        sumvm = int((total * np.arange(vmax + 1)).sum())
        # the K-truncation table
        trunc = []
        run = np.zeros(vmax + 1, dtype=np.int64)
        for K in range(1, jmax + 2):
            run = run + hists[K]
            err = int(np.abs(run - total).sum())
            trunc.append((K, err))
        # direct comparison where the machine can be built
        match = "-"
        if q <= 29:
            _, _, gnew = machine_gaps(q)
            dspec = np.bincount(gnew, minlength=vmax + 1)[:vmax + 1]
            match = "EXACT" if np.array_equal(dspec, total) else "MISMATCH"
            del gnew
        W(f"{top}->{q} | {jmax} | {Fnew} | {spec} | {summ} | {sumvm} | {P*q} | {match} | " +
          " ".join(f"K={K}:{e}" for K, e in trunc))
        print(lines[-1], flush=True)
        # per-J mass
        mx = [int(np.flatnonzero(hists[J]).max()) if hists[J].any() else 0
              for J in range(1, jmax + 2)]
        tail.append((top, q, jmax, [int(hists[J].sum()) for J in range(1, jmax + 2)],
                     total, dict_sizes(gaps, min(jmax + 1, 6)), time.time() - t0, mx))
    W("")
    W("== mass by J (number of new gaps of order J, from the size formula) ==")
    W("rung | n_1 .. n_{J_max+1}")
    for top, q, jmax, mass, total, ds, secs, mx in tail:
        W(f"{top}->{q} | " + " ".join(str(x) for x in mass) + f"   [{secs:.1f}s]")
    W("")
    W("== the realised J-tuple dictionary of M: |D_J| ==")
    W("machine | |D_1| |D_2| ... ")
    for top, q, jmax, mass, total, ds, secs, mx in tail:
        W(f"m{top} | " + " ".join(str(x) for x in ds))
    W("")
    W("== the extremes: the largest span a J-run of positive weight attains (= Q*_J) ==")
    W("rung | J = 1 .. J_max+1 | F(M+q') = max | F(M) | q' | F(M) + q'")
    for top, q, jmax, mass, total, ds, secs, mx in tail:
        _, _, gg = machine_gaps(top)
        W(f"{top}->{q} | " + " ".join(str(x) for x in mx) + f" | {max(mx)} | "
          f"{int(gg.max())} | {q} | {int(gg.max()) + q}")
    W("")
    W("== spectra (v : m(v)) ==")
    for top, q, jmax, mass, total, ds, secs, mx in tail:
        nz = np.flatnonzero(total)
        W(f"m{q}: F = {int(nz.max())}, |Spec| = {nz.size}, absent in [1, F]: " +
          str(sorted(set(range(1, int(nz.max()) + 1)) - set(int(x) for x in nz))))
        W("   " + " ".join(f"{int(v)}:{int(total[v])}" for v in nz))
    open(os.path.join(OUT, "bi_spectrum.txt"), "w").write("\n".join(lines))
    print("\nwrote results/bi_spectrum.txt")


if __name__ == "__main__":
    main()
