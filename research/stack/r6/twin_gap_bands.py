"""Twin-gap histogram of a section, model-free, by a segmented sieve.

A section is [lo, hi). Twins are pairs (n, n+2) with n = 5 mod 6 and both prime. The gap
between consecutive twins n1 < n2 is (n2 - n1) / 6 slots. This script writes the histogram of
gap lengths (all gaps, in slots) for the twins with n in [lo, hi), and prints the 50-slot band
counts from a chosen floor together with a geometric-tail prediction of each band from the
four bands below it (the manager's fourth measurement of leftover_depth.md section 5).

Usage: uv run python twin_gap_bands.py lo hi tag [band_floor] [seg]
  e.g. uv run python twin_gap_bands.py 16129 260467321 base3s4 200
Results: research/stack/r6/results/twin_gap_bands_<tag>.npz and .txt (untracked).
"""
import sys, time, math
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)


def small_primes(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def twin_starts_in(lo, hi, primes):
    """All n in [lo, hi) with n = 5 mod 6 and n, n+2 prime (n+2 may equal hi or exceed it)."""
    # sieve the odd numbers in [lo, hi + 2]
    a = lo | 1  # first odd >= lo
    b = hi + 2
    m = (b - a) // 2 + 1  # odd numbers a, a+2, ..., <= b
    s = np.ones(m, dtype=bool)
    for p in primes[1:]:  # skip 2
        if p * p > b:
            break
        start = max(p * p, ((a + p - 1) // p) * p)
        if start % 2 == 0:
            start += p
        if start > b:
            continue
        s[(start - a) // 2::p] = False
    if a <= 1:
        s[(1 - a) // 2] = False if a <= 1 else s[0]
    # n prime and n+2 prime: index i and i+1
    tw = s[:-1] & s[1:]
    idx = np.nonzero(tw)[0]
    n = a + 2 * idx
    n = n[(n % 6 == 5) & (n >= lo) & (n < hi)]
    return n


def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); tag = sys.argv[3]
    floor = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    seg = int(sys.argv[5]) if len(sys.argv) > 5 else 400_000_000
    primes = small_primes(int(math.isqrt(hi + 2)) + 1)
    hist = {}
    last = None
    t0 = time.time()
    nseg = (hi - lo + seg - 1) // seg
    for k in range(nseg):
        a = lo + k * seg; b = min(hi, a + seg)
        n = twin_starts_in(a, b, primes)
        if last is not None and len(n):
            n = np.concatenate(([last], n))
        if len(n) >= 2:
            g = np.diff(n) // 6
            u, c = np.unique(g, return_counts=True)
            for gg, cc in zip(u.tolist(), c.tolist()):
                hist[gg] = hist.get(gg, 0) + cc
        if len(n):
            last = int(n[-1])
        if k % 20 == 0 or k == nseg - 1:
            print(f"seg {k+1}/{nseg} up to {b}  twins so far {sum(hist.values())+1}  {time.time()-t0:.0f}s", flush=True)
            np.savez(RES / f"twin_gap_bands_{tag}.npz", g=np.array(sorted(hist)), c=np.array([hist[x] for x in sorted(hist)]), done=b)
    np.savez(RES / f"twin_gap_bands_{tag}.npz", g=np.array(sorted(hist)), c=np.array([hist[x] for x in sorted(hist)]), done=hi)
    G = np.array(sorted(hist)); C = np.array([hist[x] for x in G])
    lines = [f"section [{lo}, {hi}), twins {C.sum()+1}, max gap {G.max()} slots, record twin-free run = max gap - 1"]
    # 50-slot bands from floor
    def band(x, y):
        return int(C[(G >= x) & (G < y)].sum())
    bands = []
    x = floor
    while x < G.max() + 50:
        bands.append((x, x + 50, band(x, x + 50))); x += 50
    lines.append("band [x, x+50) | observed | geometric tail from the four bands below | ratio")
    for i, (x, y, o) in enumerate(bands):
        if i >= 4 and all(bands[j][2] > 0 for j in range(i - 4, i)):
            # fit log-linear decay through the four bands below (least squares on log counts)
            xs = np.array([bands[j][0] for j in range(i - 4, i)], dtype=float)
            ys = np.log(np.array([bands[j][2] for j in range(i - 4, i)], dtype=float))
            A = np.vstack([xs, np.ones(4)]).T
            sl, ic = np.linalg.lstsq(A, ys, rcond=None)[0]
            pred = math.exp(sl * x + ic)
            lines.append(f"[{x}, {y}) | {o} | {pred:.1f} | {o/pred:.2f}")
        else:
            lines.append(f"[{x}, {y}) | {o} | - | -")
    # cumulative N(>= x) for the record neighbourhood
    lines.append("N(>= x): " + ", ".join(f"{x}:{int(C[G>=x].sum())}" for x in range(floor, int(G.max()) + 1, 25)))
    txt = "\n".join(lines)
    (RES / f"twin_gap_bands_{tag}.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
