"""Saturated-run census (mechanic round 7).

A SATURATED (load-1) run = maximal run of consecutive slots k where exactly
one of (6k-1, 6k+1) is prime (pure n1: one prime + one composite per slot;
no twins, no doubles inside). These are ABSOLUTE integer objects - no gear
set, no window: primality only. So this scans k = 1..K once and derives
every window census by truncation at [k_lo(y), k_hi(y)].

Scan: segmented primality sieve (classes of primes q in [5, sqrt(6K+1)],
own-value slots unmarked). Runs of (pL XOR pR) extracted with cross-segment
carry. Records kept: every maximal run with L >= 8 (k_start, L) in memory;
CSVs: individual runs L >= 10; per-decade renewal counts for L >= 8; record
progression; side words + alternation for L >= 12 (recomputed by
Miller-Rabin, cheap).

Window census (y-ladder): runs intersected with [k_lo, k_hi]; a run
straddling k_lo is truncated (flagged); depth decile = start position in
the window. Boundary member = y convention does not arise (absolute
primality used throughout; the window's slot-1 member y is prime here,
matching rounds 1-6's P convention).

Outputs (append): research/data/satruns_ge10.csv, satruns_renewal.csv,
satruns_records.csv, satruns_windows.csv.
Usage: uv run python research/saturated_runs.py [K_slots] [y ladder...]
Defaults: K = 12e9, ladder 2003 10007 50021 200003.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime


def scan(K, seg=16_000_000, k_start=1, progress_every=0):
    """maximal load-1 runs with L >= 8 in k in [k_start, K].
    NOTE with k_start > 1: runs straddling k_start are truncated at
    k_start - start a little early and dedupe if that matters."""
    import math
    Q = math.isqrt(6 * K + 1)
    sieve_ps = [q for q in primes_upto(Q) if q >= 5]
    uvals = [pow(6, -1, q) for q in sieve_ps]
    starts_all = []
    lens_all = []
    carry_len = 0
    carry_start = 0
    t0 = time.time()
    nseg = 0
    for a in range(k_start, K + 1, seg):
        b = min(K + 1, a + seg)
        nseg += 1
        if progress_every and nseg % progress_every == 0:
            done = (b - k_start) / (K + 1 - k_start)
            print(f"  ... {100*done:.1f}% (k={b}, {time.time()-t0:.0f}s)",
                  flush=True)
        n = b - a
        exL = np.zeros(n, bool)
        exR = np.zeros(n, bool)
        for q, u in zip(sieve_ps, uvals):
            exL[(u - a) % q::q] = True
            exR[(-u - a) % q::q] = True
        if a == 1:  # own-value slots: member == q is prime, unmark
            for q in sieve_ps:
                if (q + 1) % 6 == 0:
                    exL[(q + 1) // 6 - 1] = False
                if (q - 1) % 6 == 0:
                    exR[(q - 1) // 6 - 1] = False
        sat = exL ^ exR  # exactly one member prime
        pad = np.empty(n + 2, np.int8)
        pad[0] = pad[-1] = 0
        pad[1:-1] = sat
        d = np.diff(pad)
        rs = np.flatnonzero(d == 1)
        re = np.flatnonzero(d == -1)
        if len(rs):
            starts = rs + a  # global k of run start
            lens = (re - rs).astype(np.int64)
            if sat[0] and carry_len:
                starts[0] = carry_start
                lens[0] += carry_len
            elif carry_len:
                starts_all.append(np.array([carry_start]))
                lens_all.append(np.array([carry_len]))
            if sat[-1]:
                carry_start, carry_len = int(starts[-1]), int(lens[-1])
                starts, lens = starts[:-1], lens[:-1]
            else:
                carry_len = 0
            keep = lens >= 8
            starts_all.append(starts[keep])
            lens_all.append(lens[keep])
        else:
            if carry_len:
                starts_all.append(np.array([carry_start]))
                lens_all.append(np.array([carry_len]))
            carry_len = 0
    if carry_len:
        starts_all.append(np.array([carry_start]))
        lens_all.append(np.array([carry_len]))
    starts = np.concatenate(starts_all) if starts_all else np.array([], int)
    lens = np.concatenate(lens_all) if lens_all else np.array([], int)
    print(f"scan K={K}: {len(starts)} runs with L>=8 "
          f"({time.time()-t0:.0f}s)")
    return starts, lens


def side_word(k0, L):
    w = []
    for k in range(k0, k0 + L):
        pl = is_prime(6 * k - 1)
        pr = is_prime(6 * k + 1)
        assert pl != pr, (k0, L, k)
        w.append("L" if pl else "R")
    return "".join(w)


def main():
    args = [int(a) for a in sys.argv[1:]]
    K = args[0] if args else 12_000_000_000
    ladder = args[1:] or [2003, 10007, 50021, 200003]
    starts, lens = scan(K)
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)

    def opencsv(name, header):
        path = os.path.join(ddir, name)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a")
        if new:
            f.write(header + "\n")
        return f
    # individual runs L >= 10
    f = opencsv("satruns_ge10.csv", "k_start,member_start,L")
    for k0, L in zip(starts[lens >= 10].tolist(), lens[lens >= 10].tolist()):
        f.write(f"{k0},{6*k0-1},{L}\n")
    f.close()
    # record progression
    f = opencsv("satruns_records.csv",
                "k_start,member_start,L,side_word,alternating")
    rec = 0
    print("record progression (new max L as k increases):")
    for k0, L in zip(starts.tolist(), lens.tolist()):
        if L > rec:
            rec = L
            w = side_word(k0, L)
            alt = all(w[i] != w[i + 1] for i in range(len(w) - 1))
            f.write(f"{k0},{6*k0-1},{L},{w},{alt}\n")
            print(f"  L={L:>3} at k={k0} (member {6*k0-1}) word={w} "
                  f"alternating={alt}")
    f.close()
    # renewal per decade of member
    f = opencsv("satruns_renewal.csv",
                "decade,slots_in_decade,L8,L9,L10,L11,L12,L13plus")
    mem = 6 * starts - 1
    dec = np.floor(np.log10(mem)).astype(int)
    print(f"{'decade':>7} {'slots':>12} {'L8':>7} {'L9':>6} {'L10':>6} "
          f"{'L11':>5} {'L12':>5} {'L13+':>5}")
    for d in range(int(dec.min()), int(dec.max()) + 1):
        m = dec == d
        lo_k = max(1, (10 ** d + 1) // 6)
        hi_k = min(K, (10 ** (d + 1) + 1) // 6)
        nsl = hi_k - lo_k + 1
        row = [int((m & (lens == L)).sum()) for L in (8, 9, 10, 11, 12)]
        r13 = int((m & (lens >= 13)).sum())
        f.write(f"{d},{nsl}," + ",".join(map(str, row)) + f",{r13}\n")
        print(f"{d:>7} {nsl:>12} {row[0]:>7} {row[1]:>6} {row[2]:>6} "
              f"{row[3]:>5} {row[4]:>5} {r13:>5}")
    f.close()
    # window censuses: truncation at k_lo, deciles
    f = opencsv("satruns_windows.csv",
                "y,decile,L8,L9,L10plus,maxL,maxL_k_start,truncated_at_klo")
    print("\nwindow censuses (runs L>=8 by depth decile):")
    for y in ladder:
        k_lo = -((-(y - 1)) // 6)
        k_hi = (y * y + 1) // 6
        if k_hi > K:
            print(f"  y={y}: window exceeds scan range, skipped")
            continue
        W = k_hi - k_lo + 1
        s, L = starts.copy(), lens.copy()
        endk = s + L - 1
        inw = (endk >= k_lo) & (s <= k_hi)
        s, L, endk = s[inw], L[inw], endk[inw]
        trunc = s < k_lo
        L = np.where(trunc, endk - k_lo + 1, L)
        L = np.minimum(L, k_hi - np.maximum(s, k_lo) + 1)
        s = np.maximum(s, k_lo)
        keep = L >= 8
        s, L, trunc = s[keep], L[keep], trunc[keep]
        dec10 = np.minimum((10 * (s - k_lo)) // W, 9)
        mx = int(L.max()) if len(L) else 0
        mxk = int(s[np.argmax(L)]) if len(L) else 0
        print(f"  y={y}: {len(s)} runs, maxL={mx} at k={mxk} "
              f"(trunc events {int(trunc.sum())})")
        for d in range(10):
            m = dec10 == d
            f.write(f"{y},{d},{int((m & (L == 8)).sum())},"
                    f"{int((m & (L == 9)).sum())},"
                    f"{int((m & (L >= 10)).sum())},"
                    f"{int(L[m].max()) if m.any() else 0},"
                    f"{int(s[m][np.argmax(L[m])]) if m.any() else 0},"
                    f"{int((m & trunc).sum())}\n")
    f.close()
    print("wrote satruns_ge10.csv, satruns_records.csv, satruns_renewal.csv, "
          "satruns_windows.csv")


if __name__ == "__main__":
    main()
