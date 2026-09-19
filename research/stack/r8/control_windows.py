"""Control for claim 1's spread: same-length windows at the same heights as each rung's stretch,
centred at s'^2 + 6u with u uniform in [W, 10W]. Writes control_0_3000.csv."""
import sys, os, time, csv
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], "3000"]
import sieve_rungs as sr


def sieve_window(centre, W):
    """columns j in [jlo, jlo+W) of members centre + 6j -+ 1; centre = 0 mod 6."""
    jlo = -(W // 2)
    top = centre + 6 * (jlo + W) + 1
    pmax = int(np.sqrt(top)) + 2
    P = sr._P[sr._P <= pmax]
    inv6 = sr._INV6[: len(P)]
    r_minus = ((-(centre - 1)) % P) * inv6 % P
    r_plus = ((-(centre + 1)) % P) * inv6 % P
    f_minus = (r_minus - jlo) % P
    f_plus = (r_plus - jlo) % P
    arrs = []
    for first in (f_minus, f_plus):
        a = np.ones(W, dtype=bool)
        nsmall = int(np.searchsorted(P, sr.SMALL))
        for i in range(nsmall):
            a[int(first[i])::int(P[i])] = False
        lo = nsmall
        n = len(P)
        while lo < n:
            pmin = int(P[lo])
            hi = min(int(np.searchsorted(P, 2 * pmin)), n)
            h = W // pmin + 1
            off = first[lo:hi, None] + P[lo:hi, None] * np.arange(h, dtype=np.int64)[None, :]
            a[off[off < W]] = False
            lo = hi
        arrs.append(a)
    return arrs[0] & arrs[1]


def work(args):
    sp, seed = args
    c = sp // 6
    W = 4 * c - 1
    rng = np.random.default_rng(seed + 7)
    u = int(rng.integers(W, 10 * W + 1))
    centre = sp * sp + 6 * u
    tw = sieve_window(centre, W)
    return (sp, u, int(tw.sum()))


if __name__ == "__main__":
    t0 = time.time()
    sr._init()
    # selfcheck against gmpy2 on a small window
    import gmpy2
    tw = sieve_window(30 * 30 + 6 * 50, 4 * 5 - 1)
    for i in range(len(tw)):
        jj = -9 + i
        m = 900 + 300 + 6 * jj
        assert bool(tw[i]) == bool(gmpy2.is_prime(m - 1) and gmpy2.is_prime(m + 1))
    print("selfcheck ok")
    D = np.genfromtxt(os.path.join(HERE, "rungs_0_3000.csv"), delimiter=",", names=True)
    tasks = [(int(sp), int(sp)) for sp in D["sp"]]
    out = os.path.join(HERE, "control_0_3000.csv")
    with Pool(2, initializer=sr._init) as pool, open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sp", "u", "T_ctrl"])
        done = 0
        for r in pool.imap_unordered(work, tasks, chunksize=4):
            w.writerow(r)
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {done} in {time.time()-t0:.0f}s -> {out}")
