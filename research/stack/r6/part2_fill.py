"""Fill base_and_step.md Part II from the r2 result files on disk: the chain sections' ends and
records (Q3, Q4) from results/twins_1e9.npy, and the scan's shape (Q5) from
results/record_scan_rows.npz.  Nothing is re-sieved: both files are read as they are.

Usage: uv run python research/stack/r6/part2_fill.py
"""
import json
import math
import os

import gmpy2
import numpy as np

R2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stack", "r2", "results")
R2 = os.path.join("research", "stack", "r2", "results")
SOFF = {11: 0, 17: 1, 29: 2}


def slot_index(n):
    r = n % 30
    off = np.where(r == 11, 0, np.where(r == 17, 1, 2))
    return 3 * (n // 30) + off


def first_slot_at_or_above(c):
    j = c // 30
    for jj in (j, j + 1):
        for e, o in ((11, 0), (17, 1), (29, 2)):
            if 30 * jj + e >= c:
                return 3 * jj + o
    raise RuntimeError


def record_on(tw, lo, hi):
    i0 = int(np.searchsorted(tw, lo)); i1 = int(np.searchsorted(tw, hi))
    s_first = first_slot_at_or_above(lo); s_last = first_slot_at_or_above(hi) - 1
    nslots = s_last - s_first + 1
    t = tw[i0:i1]
    if len(t) == 0:
        return dict(slots=nslots, record=nslots, at=lo, twins=0, first=None, last=None, head=nslots, tail=nslots)
    si = slot_index(t)
    runs = np.diff(si) - 1
    head = int(si[0] - s_first); tail = int(s_last - si[-1])
    best, where = head, lo
    if len(runs):
        k = int(runs.argmax())
        if runs[k] > best:
            best, where = int(runs[k]), int(t[k])
    if tail > best:
        best, where = tail, int(t[-1])
    return dict(slots=int(nslots), record=int(best), at=int(where), twins=int(len(t)),
                first=int(t[0]), last=int(t[-1]), head=head, tail=tail)


def chains(N):
    out = {}
    for b in (3, 5, 7, 11, 13):
        p = int(gmpy2.next_prime(b - 1)) if not gmpy2.is_prime(b) else b
        cuts = [b]; ps = [p]
        while True:
            c = ps[-1] ** 2
            cuts.append(c)
            if c > N:
                break
            ps.append(int(gmpy2.next_prime(c)))
        out[b] = (cuts, ps)
    return out


def main():
    tw = np.load(os.path.join(R2, "twins_1e9.npy"))
    N = 10 ** 9
    print(f"twins to 1e9 on file: {len(tw):,}  (pi_2(1e9) = 3,424,506)")
    print()
    print("== Q3/Q4: the chain sections ==")
    print("base  section                          full  slots        twins      first twin  +offset  last twin  end-gap  record  rec/slots")
    for b, (cuts, ps) in chains(N).items():
        for k in range(1, len(cuts) - 1):
            lo, hi = cuts[k], cuts[k + 1]
            if lo >= N:
                break
            full = hi <= N
            hi_eff = min(hi, N)
            r = record_on(tw, lo, hi_eff)
            off = (r["first"] - lo) if r["first"] else None
            endgap = (hi_eff - r["last"]) if r["last"] else None
            print(f"{b:4d}  [{lo:,}, {hi:,})".ljust(40)
                  + f"{'full' if full else 'pref'}  {r['slots']:11,}  {r['twins']:9,}  {str(r['first']):>11}  {str(off):>6}  "
                    f"{str(r['last']):>11}  {str(endgap):>7}  {r['record']:6d}  {r['record']/r['slots']:.3e}")
    print()
    print("== Q5: the scan (record_scan_rows.npz), recomputed summaries ==")
    A = np.load(os.path.join(R2, "record_scan_rows.npz"))["rows"]
    q, qp, lo, hi, nslots, best, where, ntw, tfirst, tlast = (A[:, i] for i in range(10))
    ratio = best / nslots
    print(f"sections: {len(A)}; q from {int(q.min())} to {int(q.max())}; empty sections (0 twins): {int((ntw == 0).sum())}")
    print(f"max record/section ratio {ratio.max():.4f} at q={int(q[int(ratio.argmax())])} "
          f"({int(best[int(ratio.argmax())])} of {int(nslots[int(ratio.argmax())])} slots)")
    for a in (100, 1000, 10000, 20000):
        m = q >= a
        print(f"  max ratio over q >= {a:6d}: {ratio[m].max():.5f} at q={int(q[m][int(ratio[m].argmax())])}"
              f"   (cuts above 0.02: {int((ratio[m] > 0.02).sum())})")
    rec_numbers = best * 10.0
    ln2 = np.log(lo.astype(float)) ** 2
    z = rec_numbers / ln2
    band = (z >= 0.5) & (z <= 4)
    print(f"record (numbers) / (ln q^2)^2: min {z.min():.3f} (q={int(q[int(z.argmin())])}), median {np.median(z):.3f}, "
          f"max {z.max():.3f} (q={int(q[int(z.argmax())])})")
    print(f"  in the pre-registered band [0.5, 4]: {band.mean() * 100:.2f}%  ({int((z < 0.5).sum())} below, {int((z > 4).sum())} above)")
    for lo_, hi_ in ((0.5, 8), (2, 9), (2.5, 9)):
        print(f"  in [{lo_}, {hi_}]: {float(((z >= lo_) & (z <= hi_)).mean()) * 100:.2f}%")
    print(f"largest record: {int(best.max())} slots at q={int(q[int(best.argmax())])} (position {int(where[int(best.argmax())])})")
    print(f"section length: min {int(nslots.min())} slots (q={int(q[int(nslots.argmin())])}), max {int(nslots.max())} (q={int(q[int(nslots.argmax())])})")
    print(f"twins per section: min {int(ntw.min())} at q={int(q[int(ntw.argmin())])}; sections with <= 2 twins: {int((ntw <= 2).sum())}")
    ii = np.argsort(ntw)[:8]
    print("  sparsest sections (q, section, slots, twins, record): "
          + "; ".join(f"({int(q[i])}, [{int(lo[i])},{int(hi[i])}), {int(nslots[i])}, {int(ntw[i])}, {int(best[i])})" for i in ii))
    print(f"first-twin offset above the cut: max {int((tfirst - lo).max())} numbers at q={int(q[int((tfirst - lo).argmax())])}; "
          f"median {int(np.median(tfirst - lo))}")
    print(f"last-twin gap to the section end: max {int((hi - tlast).max())} numbers at q={int(q[int((hi - tlast).argmax())])}")
    # head / tail runs against the interior record
    print()
    print("-- the record's position: head (cut to first twin), tail (last twin to end), or interior --")
    head = slot_index(tfirst) - np.array([first_slot_at_or_above(int(c)) for c in lo])
    tail = np.array([first_slot_at_or_above(int(c)) - 1 for c in hi]) - slot_index(tlast)
    is_head = best == head
    is_tail = best == tail
    print(f"record = head run: {int(is_head.sum())}; = tail run: {int(is_tail.sum())}; interior only: "
          f"{int((~is_head & ~is_tail).sum())}")
    print()
    print("-- ratio record/length by decade of q --")
    print("q range            n     median ratio   max ratio (q)      median record  max record  median slots")
    for a, b in ((7, 100), (100, 1000), (1000, 10000), (10000, 31700)):
        m = (q >= a) & (q < b)
        if m.any():
            print(f"[{a}, {b})".ljust(18) + f"{int(m.sum()):5d}   {np.median(ratio[m]):.3e}   "
                  f"{ratio[m].max():.4f} ({int(q[m][int(ratio[m].argmax())])})".ljust(18)
                  + f"   {np.median(best[m]):8.1f}   {int(best[m].max()):8d}   {np.median(nslots[m]):12.1f}")


if __name__ == "__main__":
    main()
