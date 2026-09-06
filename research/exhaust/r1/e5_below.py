"""Item 5: WHAT THE EXHAUST ADDS BELOW, INSIDE AND ABOVE THE WINDOW.

Q = q#, the motor-and-wheels cut; the machine at or below the cut is {all primes <= Q}; the
exhaust is every prime above Q.

  (a) [1, Q]        the exhaust's smallest gear exceeds the range: it acts not at all.
  (b) (Q, Q^2]      the window.  Every exhaust incidence is classified home / echo / neither, and
                    the action on the motor-and-wheels OPEN pairs is counted separately.
  (c) (Q^2, Q^3]    where two exhaust gears can meet on one number.  The first strike that is
                    neither home nor echo, its height, and the density of such strikes with
                    height; and the depth floor(log_Q x) - 1, the number of exhaust gears that can
                    sit on one number at height x.
"""

import json
import os
import sys
from math import isqrt, log, prod

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import primes_in, primes_upto  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def rough_arrays(Q, X):
    """rough[n] iff n has no prime factor <= Q (n >= 1); prime[n]; omega_big[n] = number of prime
    factors above Q counted with multiplicity."""
    rough = np.ones(X + 1, dtype=bool)
    rough[0] = False
    for p in primes_in(1, Q):
        rough[p::p] = False
    rough[1] = True
    pm = np.zeros(X + 1, dtype=bool)
    pm[primes_upto(X)] = True
    return rough, pm


def classify_window(q, X=None):
    """Every exhaust incidence on (Q, Q^2], classified."""
    Q = prod(primes_in(1, q))
    top = Q * Q if X is None else X
    smallp = primes_in(1, Q)
    rough, pm = rough_arrays(Q, top + 2)
    # open pairs of {primes <= Q} on (Q, Q^2 - 2]
    openpair = [n for n in range(Q + 1, top - 1) if rough[n] and rough[n + 2]]
    twin = [n for n in openpair if pm[n] and pm[n + 2]]
    # exhaust incidences: gear p > Q, position n in (Q, top-2], p | n or p | n+2
    home = echo = neither = 0
    neither_examples = []
    strikes_on_open = 0
    for n in range(Q + 1, top - 1):
        for member in (n, n + 2):
            if member <= Q:
                continue
            # the prime factors of member above Q
            m = member
            for p in primes_in(1, Q):
                while m % p == 0:
                    m //= p
            # m is the Q-rough part; its prime factors are the exhaust gears striking here
            if m == 1:
                continue
            # factor m (it is <= top and Q-rough)
            fac = []
            mm = m
            d = 2
            while d * d <= mm:
                while mm % d == 0:
                    fac.append(d)
                    mm //= d
                d += 1
            if mm > 1:
                fac.append(mm)
            for p in fac:
                if member == p:
                    home += 1
                elif member // p > 1 and any(member % r == 0 for r in primes_in(1, Q)):
                    echo += 1
                else:
                    neither += 1
                    if len(neither_examples) < 5:
                        neither_examples.append((int(p), int(n), int(member)))
            if rough[n] and rough[n + 2]:
                strikes_on_open += len(fac)
    return {"q": q, "Q": Q, "window": [Q + 1, top], "open_pairs": len(openpair),
            "open_pairs_that_are_twin": len(twin),
            "open_pair_list": openpair[:40],
            "exhaust_home": home, "exhaust_echo": echo, "exhaust_neither": neither,
            "neither_examples": neither_examples,
            "exhaust_strikes_on_open_pairs": strikes_on_open,
            "strikes_per_open_pair": (strikes_on_open / len(openpair)) if openpair else None}


def above_window(q, X):
    """(Q^2, X]: the Q-rough composites - the numbers on which the exhaust does new work."""
    Q = prod(primes_in(1, q))
    rough, pm = rough_arrays(Q, X)
    n = np.arange(X + 1)
    eff = rough & ~pm & (n > 1)          # Q-rough and not prime => at least two exhaust gears
    idx = np.flatnonzero(eff)
    p1 = int(primes_upto(2 * Q)[np.searchsorted(primes_upto(2 * Q), Q, side="right")])
    heights = []
    for e in range(2, 9):
        x = min(X, Q ** e)
        if x < Q * Q:
            continue
        c_eff = int(eff[:x + 1].sum())
        c_rough = int((rough[:x + 1] & (n[:x + 1] > 1)).sum())
        heights.append({"x": int(x), "label": f"Q^{e}", "rough": c_rough, "effective": c_eff,
                        "share": round(c_eff / c_rough, 5) if c_rough else None,
                        "depth_floor_logQ_x_minus_1": int(log(x) / log(Q)) - 1})
    # pair view: the first pair open under {primes <= Q} that is NOT a twin prime pair
    first_bad = None
    for m in range(Q + 1, X - 1):
        if rough[m] and rough[m + 2] and not (pm[m] and pm[m + 2]):
            first_bad = m
            break
    return {"q": q, "Q": Q, "X": X, "p1": p1, "p1_sq": p1 * p1,
            "first_effective": int(idx[0]) if idx.size else None,
            "first_effective_is_p1sq": (int(idx[0]) == p1 * p1) if idx.size else None,
            "height_above_Q2": int(idx[0]) - Q * Q if idx.size else None,
            "first_non_twin_open_pair": first_bad,
            "first_non_twin_above_Q2": (first_bad - Q * Q) if first_bad else None,
            "heights": heights}


def density_curve(q, X):
    """Above the window: how much of the exhaust's work is NEW, by height.

    An open pair of {primes <= Q} at height x is a twin prime pair iff neither member is a
    Q-rough composite.  The share that is not is exactly what the exhaust has left to do.
    """
    Q = prod(primes_in(1, q))
    rough, pm = rough_arrays(Q, X + 2)
    op = rough[:X + 1] & rough[2:X + 3]
    tw = pm[:X + 1] & pm[2:X + 3]
    idx = np.flatnonzero(op)
    idx = idx[idx > Q]
    rows = []
    x = Q * Q
    while x <= X:
        sel = idx[idx <= x]
        n_op = int(sel.size)
        n_tw = int(tw[sel].sum())
        rows.append({"x": int(x), "open_pairs_above_Q": n_op, "twin": n_tw,
                     "exhaust_effective": n_op - n_tw,
                     "share": round((n_op - n_tw) / n_op, 5) if n_op else None})
        x *= 10
    return {"q": q, "Q": Q, "X": X, "rows": rows}


def main():
    out = {}
    print("=== (a) below the cut: [1, Q] ===")
    for q in [5, 7]:
        Q = prod(primes_in(1, q))
        rough, pm = rough_arrays(Q, Q + 2)
        op = [n for n in range(1, Q - 1) if rough[n] and rough[n + 2]]
        print(f"q={q}: Q = q# = {Q}; motor+wheels-open pairs in [1, {Q}]: {len(op)} {op}; "
              f"exhaust gears with a multiple in [1, {Q}]: 0 (every gear exceeds the range)")
        out.setdefault("below", []).append({"q": q, "Q": Q, "open_pairs": op,
                                            "exhaust_strikes": 0})

    print("\n=== (b) the window (Q, Q^2] ===")
    for q in [5, 7]:
        r = classify_window(q)
        out.setdefault("window", []).append(r)
        print(f"q={r['q']}: window ({r['Q']}, {r['Q'] ** 2}]; "
              f"motor+wheels-open pairs {r['open_pairs']}, of which twin primes "
              f"{r['open_pairs_that_are_twin']}")
        print(f"   exhaust incidences: home {r['exhaust_home']}, echo {r['exhaust_echo']}, "
              f"NEITHER {r['exhaust_neither']}")
        print(f"   exhaust strikes landing on an open pair: "
              f"{r['exhaust_strikes_on_open_pairs']} "
              f"= {r['strikes_per_open_pair']} per open pair")
        print(f"   first open pairs: {r['open_pair_list'][:12]}")

    print("\n=== (c) above the window, (Q^2, Q^3] ===")
    for q, X in [(5, 27000), (7, 9261000)]:
        r = above_window(q, X)
        out.setdefault("above", []).append(r)
        print(f"q={r['q']}: Q={r['Q']}, p1={r['p1']}, p1^2={r['p1_sq']}")
        print(f"   first number on which the exhaust does new work (neither home nor echo): "
              f"{r['first_effective']} (= p1^2: {r['first_effective_is_p1sq']}), "
              f"height Q^2 + {r['height_above_Q2']}")
        print(f"   first pair open under the primes <= Q that is NOT a twin prime: "
              f"{r['first_non_twin_open_pair']} (Q^2 + {r['first_non_twin_above_Q2']})")
        for h in r["heights"]:
            print(f"   up to {h['label']} = {h['x']}: Q-rough {h['rough']}, of which "
                  f"composite (exhaust-effective) {h['effective']} "
                  f"({100 * h['share']:.2f}%), depth {h['depth_floor_logQ_x_minus_1']}")

    print("\n=== (c2) the share of open pairs above Q^2 that the exhaust must still kill ===")
    for q, X in [(5, 10 ** 8), (7, 10 ** 8)]:
        r = density_curve(q, X)
        out.setdefault("density", []).append(r)
        print(f"q={q}, Q={r['Q']}:")
        for row in r["rows"]:
            print(f"   up to {row['x']:>10}: open pairs above Q {row['open_pairs_above_Q']:>9}, "
                  f"twin {row['twin']:>8}, exhaust-effective {row['exhaust_effective']:>9} "
                  f"({100 * row['share']:.3f}%)")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e5_below.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
