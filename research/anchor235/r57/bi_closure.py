"""bi_closure.py -- does the hierarchy close?  How deep a dictionary of M is needed to write
down the depth-m dictionary of M + q' exactly.

Engine: the z-walk, an independent vehicle that uses no word theory at all.  For a window of
consecutive gaps starting at the opening x_0, put z = x_0 - r (mod q') for the deletion phase r;
then the opening at offset o is struck iff (z + o) mod q' is in {0, d}.  Letting z run over all
q' residues runs over all q' copies exactly once (file 05 (A)).  So for each position n and each
z we can read off directly which of the following openings survive, hence the first m new gaps
of M + q' that start at n, and how many OLD gaps they span.

    K_m(M -> q')  =  the largest number of consecutive old gaps spanned by m consecutive new gaps
                  =  the least depth of the old dictionary that determines the depth-m
                     dictionary of the new machine.

The depth-m dictionary of M + q' (with multiplicity) is then computed from M's depth-K_m
size dictionary alone, and gated against the directly built machine.

Usage:  uv run python research/anchor235/r57/bi_closure.py [maxtop] [mmax]
"""
import os
import sys
import time
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bi_core import OUT, PRIMES, u_of, machine_gaps, letters, word_counts, wrap_pad

CHUNK = 4_000_000


def zwalk_dictionary(gaps, q, mmax, kcap):
    """Returns (dicts, khist) where dicts[m] is a Counter over m-tuples of consecutive NEW gap
    sizes with multiplicity over the whole period of M + q', and khist[m] is a histogram of the
    number of old gaps those m new gaps span."""
    N = gaps.size
    d = (2 * u_of(q)) % q
    gp = wrap_pad(gaps, kcap)
    dicts = {m: Counter() for m in range(1, mmax + 1)}
    khist = {m: Counter() for m in range(1, mmax + 1)}
    pos = 0
    while pos < N:
        end = min(pos + CHUNK, N)
        n = end - pos
        ar = np.arange(n)
        off = np.zeros((kcap + 1, n), dtype=np.int32)
        for i in range(1, kcap + 1):
            off[i] = off[i - 1] + gp[pos + i - 1:end + i - 1]
        offq = (off % q).astype(np.uint8)
        for z in range(q):
            # opening at offset i is struck iff (off_i + z) mod q in {0, d}
            a, b = (-z) % q, (d - z) % q
            strk = (offq == a) | (offq == b)
            if strk[0].all():
                continue
            # nextfree[i] = least j >= i with the opening at offset j unstruck (127 = none)
            nextfree = np.full((kcap + 2, n), 127, dtype=np.int8)
            for i in range(kcap, -1, -1):
                nextfree[i] = np.where(strk[i], nextfree[i + 1], np.int8(i))
            ok = ~strk[0]
            idx = np.zeros(n, dtype=np.int64)
            vs = []
            for m in range(1, mmax + 1):
                nxt = nextfree[np.minimum(idx + 1, kcap + 1), ar].astype(np.int64)
                ok = ok & (nxt <= kcap)
                v = np.where(ok, off[np.minimum(nxt, kcap), ar] - off[idx, ar], 0)
                vs.append(v)
                idx = np.where(ok, nxt, idx)
                sel = np.flatnonzero(ok)
                if sel.size == 0:
                    break
                key = np.zeros(sel.size, dtype=np.int64)
                for vv in vs:
                    key = key * 256 + vv[sel]
                for kv, c in zip(*np.unique(key, return_counts=True)):
                    dicts[m][int(kv)] += int(c)
                for kv, c in zip(*np.unique(idx[sel], return_counts=True)):
                    khist[m][int(kv)] += int(c)
        pos = end
    return dicts, khist


def direct_dictionary(gnew, mmax):
    """The depth-m dictionaries of a directly built machine, with multiplicity."""
    N = gnew.size
    gp = wrap_pad(gnew, mmax)
    out = {}
    for m in range(1, mmax + 1):
        key = np.zeros(N, dtype=np.int64)
        for i in range(m):
            key = key * 256 + gp[i:i + N]
        c = Counter()
        for kv, cnt in zip(*np.unique(key, return_counts=True)):
            c[int(kv)] += int(cnt)
        out[m] = c
    return out


def main():
    maxtop = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    mmax = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    lines = []
    W = lines.append
    W("bi_closure -- the depth needed in M to write the depth-m dictionary of M + q'")
    W("")
    W("rung | J_max | m | K_m (least depth of M that suffices) | m * J_max | "
      "|dictionary| | mass | gate vs direct build | error at depth K_m - 1")
    for k in range(len(PRIMES) - 1):
        top, q = PRIMES[k], PRIMES[k + 1]
        if top > maxtop:
            break
        t0 = time.time()
        P, f0, gaps = machine_gaps(top)
        lt = letters(gaps, q)
        L = 0
        while word_counts(lt, L + 1)[0] > 0:
            L += 1
        jmax = L + 2
        kcap = mmax * jmax + 2
        dicts, khist = zwalk_dictionary(gaps, q, mmax, kcap)
        direct = None
        if q <= 29:
            _, _, gnew = machine_gaps(q)
            direct = direct_dictionary(gnew, mmax)
            del gnew
        for m in range(1, mmax + 1):
            Km = max(khist[m])
            mass = sum(dicts[m].values())
            gate = "-"
            if direct is not None:
                gate = "EXACT" if dicts[m] == direct[m] else "MISMATCH"
            lost = sum(c for kk, c in khist[m].items() if kk > Km - 1)
            W(f"{top}->{q} | {jmax} | {m} | {Km} | {m*jmax} | {len(dicts[m])} | {mass} | "
              f"{gate} | {lost}")
            print(lines[-1], flush=True)
        W(f"   [{time.time()-t0:.1f}s]  span histogram (old gaps per m new gaps): " +
          "; ".join(f"m={m}: " + ",".join(f"{kk}:{c}" for kk, c in sorted(khist[m].items()))
                    for m in range(1, mmax + 1)))
    open(os.path.join(OUT, "bi_closure.txt"), "w").write("\n".join(lines))
    print("\nwrote results/bi_closure.txt")


if __name__ == "__main__":
    main()
