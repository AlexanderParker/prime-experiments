"""Prefix census at the window bottom (mechanic round 2, for the Constructor).

Constructor's C2 condition lives in window prefixes (y, t]. For the first T
slots of the window of y (slots k from k_lo = ceil((y-1)/6)), this computes
exactly, per prefix length t:

    P(t)   prime members among the first t slots (actual primality; the
           member equal to y itself counts as prime)
    n0(t)  slots with both members prime (twins; n0>y variant requires
           both members > y)
    n1(t)  slots with exactly one composite member
    n2(t)  slots with both members composite (doubles)
    margin(t) = N(t) - P(t) = t - P(t) = n2(t) - n0(t)   (C2: X needs >= 0)

plus the onset lags: first slot (1-based from the window bottom) that is a
twin above y, and first slot that is a double. Primality by deterministic
Miller-Rabin (valid to 3.3e24), so any y in 64-bit range is affordable -
only the first T slots are ever touched.

Output: summary table to stdout; full per-t curves for all y to
research/data/prefix_census.csv (columns y,t,k,mlo,P,n0,n1,n2,margin).

Usage: uv run python research/prefix_census.py [T] [y1 y2 ...]
Defaults: T=200, ladder 101 503 1009 2003 5003 10007 20011 50021 100003
          1000003 10000019 100000007.
"""
import os
import sys

_SPRP = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n):
    if n < 2:
        return False
    for p in _SPRP:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SPRP:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prefix(y, T=200):
    k_lo = -((-(y - 1)) // 6)
    rows = []
    P = n0 = n1 = n2 = 0
    first_twin = first_twin_gt = first_double = None
    for t in range(1, T + 1):
        k = k_lo + t - 1
        mlo, mhi = 6 * k - 1, 6 * k + 1
        pl, pr = is_prime(mlo), is_prime(mhi)
        P += pl + pr
        if pl and pr:
            n0 += 1
            if first_twin is None:
                first_twin = t
            if first_twin_gt is None and mlo > y:
                first_twin_gt = t
        elif pl or pr:
            n1 += 1
        else:
            n2 += 1
            if first_double is None:
                first_double = t
        rows.append((y, t, k, mlo, P, n0, n1, n2, t - P))
    return rows, dict(y=y, first_twin=first_twin, first_twin_gt=first_twin_gt,
                      first_double=first_double)


def main():
    args = [int(a) for a in sys.argv[1:]]
    T = args[0] if args else 200
    ys = args[1:] or [101, 503, 1009, 2003, 5003, 10007, 20011, 50021,
                      100003, 1000003, 10000019, 100000007]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "prefix_census.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    all_rows = []
    print(f"{'y':>10} {'1st_twin':>8} {'1st_tw>y':>8} {'1st_dbl':>8} "
          f"{'minMarg':>8} {'argmin':>7} {'lastNeg':>8} {'marg(T)':>8} "
          f"{'P(T)':>6} {'n0':>5} {'n1':>5} {'n2':>5}")
    for y in ys:
        rows, s = prefix(y, T)
        all_rows += rows
        margins = [r[8] for r in rows]
        mn = min(margins)
        argmin = margins.index(mn) + 1
        neg = [t + 1 for t, m in enumerate(margins) if m < 0]
        last_neg = neg[-1] if neg else 0
        end = rows[-1]
        print(f"{y:>10} {s['first_twin'] or '-':>8} "
              f"{s['first_twin_gt'] or '-':>8} {s['first_double'] or '-':>8} "
              f"{mn:>8} {argmin:>7} {last_neg:>8} {end[8]:>8} "
              f"{end[4]:>6} {end[5]:>5} {end[6]:>5} {end[7]:>5}")
    with open(out, "w") as f:
        f.write("y,t,k,member_lo,P,n0,n1,n2,margin\n")
        for r in all_rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"\nwrote {len(all_rows)} rows to {out}")


if __name__ == "__main__":
    main()
