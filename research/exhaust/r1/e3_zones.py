"""Item 3: THE ZONES AND THE REDUNDANCY LEMMA IN THE STACK'S COORDINATE (O-X2).

On a range of N raw integers with the machine {all primes <= Q} and the exhaust above Q, a gear
sits in one of three zones with respect to the RANGE (not with respect to the tier):

  REPEATING      g^2 <= N            it turns past its own square; it has strikes g m with m >= g
  NON-REPEATING  sqrt(N) < g <= N/2  every strike is g m with m < g
  SILENT         g > N/2             the only multiple in [1, N] is g itself

Counts are exact and closed: gear g strikes floor(N/g) numbers of [1, N], and in pair coordinates
the positions n with g | n or g | n + 2, i.e. at most 2 floor(N/g) + 2 positions.

The redundancy lemma (range form) is checked exhaustively with a least-prime-factor sieve.
"""

import json
import os
import sys
from math import isqrt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import primes_upto  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def spf_sieve(N):
    """spf[n] = least prime factor of n, for n <= N (spf[0] = spf[1] = 0)."""
    spf = np.zeros(N + 1, dtype=np.int64)
    for p in range(2, isqrt(N) + 1):
        if spf[p] == 0:
            blk = spf[p * p:: p]
            blk[blk == 0] = p
            if spf[p] == 0:
                spf[p] = p
    rest = np.flatnonzero(spf[2:] == 0) + 2
    spf[rest] = rest
    return spf


def census(N):
    pr = primes_upto(N)
    s = isqrt(N)
    strikes = N // pr                       # floor(N/g), exact per gear
    rep = pr[pr <= s]
    non = pr[(pr > s) & (pr <= N // 2)]
    sil = pr[pr > N // 2]
    c = {
        "N": N, "sqrtN": s, "pi_N": int(pr.size),
        "n_repeating": int(rep.size), "n_nonrepeating": int(non.size), "n_silent": int(sil.size),
        "strikes_repeating": int((N // rep).sum()),
        "strikes_nonrepeating": int((N // non).sum()),
        "strikes_silent": int(sil.size),
        "strikes_total": int(strikes.sum()),
    }
    c["home_strikes"] = c["pi_N"]
    c["echoes_nonrepeating"] = c["strikes_nonrepeating"] - c["n_nonrepeating"]
    c["echoes_silent"] = 0
    c["strikes_above_sqrtN"] = c["strikes_nonrepeating"] + c["strikes_silent"]
    c["home_above_sqrtN"] = c["n_nonrepeating"] + c["n_silent"]
    c["echo_above_sqrtN"] = c["strikes_above_sqrtN"] - c["home_above_sqrtN"]
    c["identity_ok"] = (c["strikes_repeating"] + c["strikes_nonrepeating"]
                        + c["strikes_silent"] == c["strikes_total"])
    return c


def verify_redundancy(N):
    """Exhaustive: every strike by a gear g > sqrt(N) on [1, N] is a home strike or an echo."""
    s = isqrt(N)
    spf = spf_sieve(N)
    n = np.arange(N + 1, dtype=np.int64)
    # the strikes of gears above sqrt(N) are exactly the numbers n whose largest prime factor
    # exceeds sqrt(N); such an n has at most one such factor.  Home iff n is prime.
    # A number n is struck by a gear > sqrt(N) iff n has a prime factor > sqrt(N).
    # cofactor m = n / p; check spf(m) < p for m > 1.
    big = np.zeros(N + 1, dtype=np.int64)   # the (unique) prime factor > sqrt(N), or 0
    pr = primes_upto(N)
    for p in pr[pr > s]:
        p = int(p)
        big[p::p] = p
    idx = np.flatnonzero(big > 0)
    idx = idx[idx > 0]
    p = big[idx]
    m = idx // p
    home = int((m == 1).sum())
    comp = idx[m > 1]
    pc, mc = p[m > 1], m[m > 1]
    ok = spf[mc] < pc
    bad = int((~ok).sum())
    # a second, independent check: two factors above sqrt(N) would need n > N
    two = 0
    for pp in pr[pr > s]:
        pp = int(pp)
        if pp * pp > N:
            break
        two += 1
    return {"N": N, "sqrtN": s, "struck_by_big_gear": int(idx.size),
            "home_strikes": home, "echo_strikes": int(comp.size),
            "echo_exceptions": bad,
            "gears_above_sqrtN_with_square_in_range": two}


def verify_silent_pairs(N):
    """A silent gear (g > N/2) touches exactly the two pair positions g and g - 2."""
    pr = primes_upto(N)
    sil = pr[pr > N // 2]
    bad = 0
    for g in sil:
        g = int(g)
        mult = np.arange(0, N + 1, g)                 # the multiples of g in [0, N]
        pos = set(int(x) for x in mult if 1 <= x <= N - 2)
        pos |= set(int(x - 2) for x in mult if 1 <= x - 2 <= N - 2)
        want = {x for x in (g, g - 2) if 1 <= x <= N - 2}
        if pos != want:
            bad += 1
    return {"N": N, "n_silent_checked": int(sil.size), "exceptions": bad}


def main():
    out = {}
    print("=== the three gear zones of a range, exact counts ===")
    print(f"{'N':>10} {'pi(N)':>9} {'#rep':>7} {'#non':>9} {'#sil':>9} "
          f"{'strikes rep':>13} {'strikes non':>12} {'strikes sil':>11} {'total':>12}")
    rows = []
    for N in [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]:
        c = census(N)
        rows.append(c)
        print(f"{N:>10} {c['pi_N']:>9} {c['n_repeating']:>7} {c['n_nonrepeating']:>9} "
              f"{c['n_silent']:>9} {c['strikes_repeating']:>13} {c['strikes_nonrepeating']:>12} "
              f"{c['strikes_silent']:>11} {c['strikes_total']:>12}")
    out["census"] = rows
    print("\nabove sqrt(N): strikes / home / echo, and the echo share")
    for c in rows:
        print(f"  N={c['N']:>9}: {c['strikes_above_sqrtN']:>9} strikes = "
              f"{c['home_above_sqrtN']:>8} home + {c['echo_above_sqrtN']:>8} echo "
              f"({100 * c['echo_above_sqrtN'] / c['strikes_above_sqrtN']:.2f}% echo); "
              f"identity ok: {c['identity_ok']}")

    print("\n=== the redundancy lemma, exhaustive ===")
    red = []
    for N in [10 ** 5, 10 ** 6, 10 ** 7]:
        r = verify_redundancy(N)
        red.append(r)
        print(f"  N={N:>9}: {r['struck_by_big_gear']:>8} numbers struck by a gear > sqrt(N) = "
              f"{r['home_strikes']:>8} home + {r['echo_strikes']:>8} echo; "
              f"exceptions {r['echo_exceptions']}; gears > sqrt(N) whose square is <= N: "
              f"{r['gears_above_sqrtN_with_square_in_range']}")
    out["redundancy"] = red

    print("\n=== silent gears in pair coordinates ===")
    sp = []
    for N in [10 ** 4, 10 ** 5]:
        r = verify_silent_pairs(N)
        sp.append(r)
        print(f"  N={N}: {r['n_silent_checked']} silent gears checked, "
              f"exceptions {r['exceptions']}")
    out["silent_pairs"] = sp

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e3_zones.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
