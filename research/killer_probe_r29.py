"""Manager killer-word probe - round 29.

Pre-registration: research/data/r29/manager_killer_prereg.md (P-K1 .. P-K6).

Part A  - the real twin machine: origin gap g_0(M) against the record F(M) and the
          reduction window W(M) = (q_next^2 - 1)/6, plus the origin-front fusion depth at
          each rung (how many leading old openings the incoming gear kills).
Part B  - the counterfactual family V(y) (teeth at +-v_q, v_q free), exhaustive at
          y = 7, 11, 13, 17, 19: family max F against W; violators of (D); the CHAINING test
          (does any violator have a violating child?); two-rung slack; origin placement.
Part C  - m23 children of every m19 violator (11 sieves each); --full23 runs all 142,560.

Exact integers, asserts as gates, no fits.
Usage: python killer_probe_r29.py [--upto 19] [--workers 8] [--full23]
"""
import argparse
import itertools
import multiprocessing as mp
import sys
import time

import numpy as np

GEARS = [5, 7, 11, 13, 17, 19, 23]
NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def tooth(q):
    return pow(6, -1, q)


# ----------------------------------------------------------------- Part A: real machine
def first_survivors(gears, count):
    """The first `count` openings k >= 1 of the twin machine with these gears."""
    out, k = [], 1
    teeth = [(q, tooth(q) % q, (-tooth(q)) % q) for q in gears]
    while len(out) < count:
        if all(k % q != a and k % q != b for q, a, b in teeth):
            out.append(k)
        k += 1
    return out


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def part_a(qmax, F_known):
    print("\n=== PART A: real twin machine - origin gap vs record vs window ===")
    ps = [p for p in primes_upto(qmax * 2) if p >= 5]
    print(f"{'q':>4} {'q_next':>6} {'g_0':>5} {'F(M)':>6} {'W':>7} {'g0/W':>7} {'F/W':>6} "
          f"{'moved':>5} {'front_kills':>11}")
    moved = total = 0
    max_front = 0
    prev_open = None
    prev_g0 = None
    for i, q in enumerate(ps):
        if q > qmax:
            break
        gears = [p for p in ps if p <= q]
        qn = ps[i + 1]
        W = (qn * qn - 1) // 6
        opens = first_survivors(gears, 12)
        g0 = opens[0]
        gate(is_prime(6 * g0 - 1) and is_prime(6 * g0 + 1),
             f"m{q}: first survivor k_0={g0} is a twin prime pair ({6*g0-1},{6*g0+1})")
        gate(g0 <= W, f"m{q}: g_0 = {g0} inside the window W = {W}")
        front = 0
        if prev_open is not None:
            # how many leading openings of the previous machine did gear q kill?
            u = tooth(q)
            for k in prev_open:
                if k % q == u % q or k % q == (q - u) % q:
                    front += 1
                else:
                    break
            gate((front > 0) == (g0 != prev_g0), f"m{q}: origin moved iff a front kill")
        max_front = max(max_front, front)
        total += 1
        if front > 0:
            moved += 1
        F = F_known.get(q)
        Fs = str(F) if F is not None else "-"
        FW = f"{F / W:.3f}" if F is not None else "-"
        if q <= 60 or front > 0:
            print(f"{q:>4} {qn:>6} {g0:>5} {Fs:>6} {W:>7} {g0 / W:>7.4f} {FW:>6} "
                  f"{'yes' if front else '':>5} {front:>11}")
        prev_open, prev_g0 = opens, g0
    print(f"origin gap moved at {moved}/{total} rungs (q <= {qmax}); "
          f"max leading openings killed at the origin front in one rung = {max_front}")
    gate(moved * 2 < total, f"P-K1: origin moved at fewer than half the rungs ({moved}/{total})")
    gate(max_front <= 2, f"P-K1: origin-front fusion depth <= 2 at every rung (max {max_front})")


# ------------------------------------------------------------- Part B: counterfactuals
def member_stats(args):
    """(F, origin_gap, record_at_origin) for one symmetric-teeth sieve."""
    gears, vs, P = args
    blocked = np.zeros(P, dtype=bool)
    for q, v in zip(gears, vs):
        blocked[v % q::q] = True
        blocked[(-v) % q::q] = True
    op = np.flatnonzero(~blocked)
    n = op.size
    g = np.empty(n, dtype=np.int64)
    g[:-1] = np.diff(op)
    g[-1] = op[0] + P - op[-1]
    F = int(g.max())
    assert op[0] == 0, "slot 0 must be open in every symmetric member"
    g0 = int(op[1]) if n > 1 else P
    return F, g0, bool(g0 == F), n


def part_b(upto, workers):
    print("\n=== PART B: counterfactual family - K1 size, violators, chaining ===")
    levels = [n for n in range(2, len(GEARS) + 1) if GEARS[n - 1] <= upto]
    ps = [p for p in primes_upto(200) if p >= 5]
    Fd, G0d, ROd = {}, {}, {}
    with mp.Pool(workers) as pool:
        for n in levels:
            gears = GEARS[:n]
            y = gears[-1]
            P = int(np.prod(gears))
            space = [list(range(1, (q - 1) // 2 + 1)) for q in gears]
            vecs = list(itertools.product(*space))
            t0 = time.time()
            res = pool.map(member_stats, [(gears, list(v), P) for v in vecs], chunksize=16)
            nref = res[0][3]
            gate(all(r[3] == nref for r in res) and nref == int(np.prod([q - 2 for q in gears])),
                 f"m{y}: all {len(vecs)} members have prod(q-2) = {nref} openings "
                 f"[{time.time() - t0:.0f}s]")
            for v, r in zip(vecs, res):
                Fd[v] = r[0]; G0d[v] = r[1]; ROd[v] = r[2]
            twin = tuple(min(tooth(q) % q, (-tooth(q)) % q) for q in gears)
            qn = ps[ps.index(y) + 1]
            W = (qn * qn - 1) // 6
            Fmax = max(Fd[v] for v in vecs)
            g0max = max(G0d[v] for v in vecs)
            n_rec_origin = sum(ROd[v] for v in vecs)
            print(f"  m{y:<3} |V|={len(vecs):>6}  F(twin)={Fd[twin]:>3}  max_V F={Fmax:>3}  "
                  f"W={W:>4}  W/maxF={W / Fmax:.2f}  max_V g_0={g0max:>3}  "
                  f"members whose record IS the origin gap: {n_rec_origin}")
            gate(Fmax < W, f"P-K2: m{y} family max F = {Fmax} < W = {W}")
            if y >= 11:  # pre-registered levels are 11..19; m7 is the degenerate 2-gear machine
                gate(1.5 <= W / Fmax <= 3.0, f"P-K2: m{y} W/maxF = {W / Fmax:.2f} in [1.5, 3.0]")

    # violators and chaining
    print("\n  --- (D) violators per step and the chaining test ---")
    viol = {}
    for n in levels[1:]:
        gears = GEARS[:n]; q = gears[-1]; y0 = gears[-2]
        vs_here = [v for v in Fd if len(v) == n]
        sl = {v: Fd[v] - Fd[v[:-1]] - q for v in vs_here}
        viol[n] = [v for v in vs_here if sl[v] > 0]
        print(f"  step m{y0}->m{q}: family {len(vs_here)}, violators {len(viol[n])}, "
              f"max slack {max(sl.values()):+d}, "
              f"twin slack {sl[tuple(min(tooth(g) % g, (-tooth(g)) % g) for g in gears)]:+d}")
        for v in viol[n]:
            print(f"      violator v={v}: F_old={Fd[v[:-1]]}, F_new={Fd[v]}, slack {sl[v]:+d}, "
                  f"g_0={G0d[v]}, record at origin: {ROd[v]}")
            gate(not ROd[v], f"P-K5: violator {v} record is not the origin gap")
    for n in levels[1:-1]:
        q2 = GEARS[n]
        chained = 0
        for v in viol[n]:
            kids = [v + (w,) for w in range(1, (q2 - 1) // 2 + 1)]
            ks = [Fd[k] - Fd[v] - q2 for k in kids]
            print(f"      children of violator {v} at ->m{q2}: slacks {ks}")
            chained += sum(1 for s in ks if s > 0)
        gate(chained == 0, f"P-K3: no violator at ->m{GEARS[n-1]} has a violating child at ->m{q2} "
                           f"({len(viol[n])} violators checked)")
        # conditional violation rate: children of violators vs children of everyone
        vs_next = [v for v in Fd if len(v) == n + 1]
        base = sum(1 for v in vs_next if Fd[v] - Fd[v[:-1]] - q2 > 0) / len(vs_next)
        print(f"      base violation rate at ->m{q2}: {100 * base:.2f}%")
    # two-rung slack
    for n in levels[2:]:
        q1, q2 = GEARS[n - 2], GEARS[n - 1]
        vs_here = [v for v in Fd if len(v) == n]
        two = [Fd[v] - Fd[v[:-2]] - q1 - q2 for v in vs_here]
        print(f"  two-rung slack m{GEARS[n-3]}->m{q2}: max {max(two):+d}, "
              f"count > 0: {sum(1 for s in two if s > 0)}/{len(two)}, "
              f"count >= +2: {sum(1 for s in two if s >= 2)}")
        gate(max(two) <= 1, f"P-K4: two-rung slack max {max(two):+d} <= +1 at m{GEARS[n-3]}->m{q2}")
    return Fd, viol


def part_c(Fd, viol, upto, workers, full23):
    if upto < 19 or 6 not in viol:
        return
    print("\n=== PART C: m23 children of every m19 violator ===")
    gears = GEARS[:7]; P = int(np.prod(gears)); q2 = 23
    if full23:
        space = [list(range(1, (q - 1) // 2 + 1)) for q in gears]
        parents = [v for v in Fd if len(v) == 6]
        kids = [v + (w,) for v in parents for w in range(1, 12)]
        print(f"  --full23: {len(kids)} sieves of P = {P}")
    else:
        kids = [v + (w,) for v in viol[6] for w in range(1, 12)]
        print(f"  {len(kids)} sieves of P = {P} ({len(viol[6])} violators x 11 teeth)")
    t0 = time.time()
    with mp.Pool(min(workers, 6)) as pool:
        res = pool.map(member_stats, [(gears, list(v), P) for v in kids], chunksize=4)
    print(f"  done in {time.time() - t0:.0f}s")
    chained = 0
    ps = [p for p in primes_upto(200) if p >= 5]
    W = (29 * 29 - 1) // 6
    Fmax = 0
    for k, r in zip(kids, res):
        Fd[k] = r[0]
        Fmax = max(Fmax, r[0])
        s = r[0] - Fd[k[:-1]] - q2
        if s > 0:
            chained += 1
            par_slack = Fd[k[:-1]] - Fd[k[:-2]] - 19
            print(f"      violator at ->m23: {k}, slack {s:+d}, parent slack {par_slack:+d}, "
                  f"record at origin: {r[2]}")
    if full23:
        print(f"  m23 full family: max F = {Fmax}, W = {W}, W/maxF = {W / Fmax:.2f}, "
              f"violators at 19->23: {chained}")
        chain2 = [k for k in kids if Fd[k] - Fd[k[:-1]] - 23 > 0 and Fd[k[:-1]] - Fd[k[:-2]] - 19 > 0]
        gate(len(chain2) == 0, f"P-K6 (full): no two-rung chain 17->19->23 ({len(chain2)} found)")
        two = [Fd[k] - Fd[k[:-2]] - 19 - 23 for k in kids]
        # P-K4 was pre-registered as "max <= +1" and REFUTED here on 2026-09-02: the full
        # family has one member, (1,3,1,2,5,8,5), at two-rung +4 (parent -7, child +11).
        # Recorded, not gated - see research/data/r29/manager_killer_prereg.md scorecard.
        print(f"  P-K4 (m23): two-rung slack max {max(two):+d} "
              f"(pre-registered <= +1: {'HELD' if max(two) <= 1 else 'REFUTED'}), "
              f"members > +1: {[k for k in kids if Fd[k] - Fd[k[:-2]] - 42 > 1]}")
    else:
        gate(chained == 0, f"P-K6: no m19 violator has a violating m23 child ({len(kids)} checked)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=19)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--full23", action="store_true")
    ap.add_argument("--qmax", type=int, default=200)
    a = ap.parse_args()
    # corpus record values (kernel-checked / census, agents-shared.md) for the real machine
    F_known = {29: 43, 31: 58, 37: 88, 41: 91, 43: 103, 47: 118, 53: 145}
    # small machines computed here exactly and cross-checked against the corpus
    for n in range(1, 7):
        gears = GEARS[:n]; P = int(np.prod(gears))
        vs = [min(tooth(q) % q, (-tooth(q)) % q) for q in gears]
        F_known[gears[-1]] = member_stats((gears, vs, P))[0]
    gate(F_known[11] == 7 and F_known[13] == 11 and F_known[17] == 18 and F_known[19] == 25,
         "small-machine records match the corpus (7, 11, 18, 25)")
    part_a(a.qmax, F_known)
    Fd, viol = part_b(a.upto, a.workers)
    part_c(Fd, viol, a.upto, a.workers, a.full23)
    print(f"\nALL {NGATE} ASSERTION GATES PASSED")


if __name__ == "__main__":
    main()
