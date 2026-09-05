r"""bm_exact.py - the block second moment, computed EXACTLY on the real machine.

The distortion engine (BBMST Invent. math. 228 (2022) Thm 3.1) with delta_i = 1/2, run on an
interval of columns with the parts taken to be ARITHMETIC BLOCKS of length beta instead of the
congruence fibres (classes mod Q_{i-1}).  The per-part algebra is partition-agnostic:

    on a part B with a = alpha_i(B) = P_{i-1}(B_i ∩ B) / P_{i-1}(B), delta = 1/2,
        P_i = max{0, 2 - 1/a} * P_{i-1}   on B_i ∩ B      (0 whenever a <= 1/2)
        P_i = min{1/(1-a), 2} * P_{i-1}   on B \ B_i
    mass on B is preserved exactly, and the loss is P_i(B_i ∩ B) = P_{i-1}(B) * max{0, 2a - 1}.

B_i = struck by gear i AND not struck by any earlier gear (BBMST's "newly covered at stage i").

Budget:   eta_B = sum_i E_{i-1}[alpha_i^2]     (the theorem's hypothesis, needs < 1)
True loss: sum_i P_i(B_i) = sum_i E_{i-1}[max{0, 2 alpha_i - 1}]   (what eta_B bounds)

Outputs (results/, untracked): bm_exact.txt
"""

import math
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def primes_upto(n):
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i:: i] = bytearray(len(s[i * i:: i]))
    return [i for i in range(n + 1) if s[i]]


ALLP = primes_upto(4000)


def gears(q):
    return [p for p in ALLP if 5 <= p <= q]


def next_prime(q):
    for p in ALLP:
        if p > q:
            return p
    raise ValueError


def window(q):
    qp = next_prime(q)
    k0 = (q + 1) // 6 + 1
    k1 = (qp * qp - 1) // 6
    return k0, k1, k1 - k0 + 1


def strike_mask(k0, L, g):
    """boolean array of length L: column k0+j struck by gear g"""
    u = pow(6, -1, g)
    m = np.zeros(L, dtype=bool)
    for r in {u % g, (-u) % g}:
        j0 = (r - k0) % g
        if j0 < L:
            m[j0::g] = True
    return m


def run_blocks(k0, L, gs, beta_rule, track_dead=False):
    """Run the block distortion recursion.  beta_rule(i, g, Qprev) -> block length at stage i.
    Returns (eta, loss, dead_mass, per_gear) with per_gear a list of (g, beta, M1, M2, nblocks)."""
    w = np.full(L, 1.0 / L)
    covered = np.zeros(L, dtype=bool)
    eta = 0.0
    loss = 0.0
    per = []
    Q = 1
    logQ = 0.0
    for i, g in enumerate(gs):
        beta = max(1, min(L, int(beta_rule(i, g, Q, logQ))))
        idx = np.arange(0, L, beta)
        newly = strike_mask(k0, L, g) & ~covered
        T = np.add.reduceat(w, idx)
        A = np.add.reduceat(np.where(newly, w, 0.0), idx)
        good = T > 0
        a = np.zeros_like(T)
        a[good] = A[good] / T[good]
        m1 = float(np.sum(T * a))
        m2 = float(np.sum(T * a * a))
        eta += m2
        loss += float(np.sum(T * np.maximum(0.0, 2.0 * a - 1.0)))
        # expand alpha to columns
        ac = np.repeat(a, np.diff(np.append(idx, L)))
        f_in = np.maximum(0.0, 2.0 - np.divide(1.0, ac, out=np.zeros_like(ac), where=ac > 0))
        f_out = np.minimum(np.divide(1.0, 1.0 - ac, out=np.full_like(ac, 2.0), where=ac < 1), 2.0)
        w = np.where(newly, w * f_in, w * f_out)
        covered |= newly
        per.append((g, beta, m1, m2, len(idx)))
        Q *= g
        logQ += math.log(g)
    dead = 0.0
    if track_dead:
        dead = float(np.mean(covered))
    return eta, loss, dead, per


def beta_grid(L):
    """a log-spaced grid of fixed block lengths, always including 1, 2, 3 and L"""
    xs = {1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200,
          256, 320, 400, 512, 640, 800, 1024, 1600, 2048, 4096, 8192, 16384, 32768, 65536}
    xs = sorted(x for x in xs if x < L)
    return xs + [L]


def main():
    say("=" * 100)
    say("A.  THE EXACT BLOCK BUDGET ON THE REAL WINDOW (real teeth, exact survivor counts)")
    say("    eta_B = sum_i E[alpha_i^2] over blocks of length beta; loss = sum_i E[max(0,2a-1)]")
    say("    'covered' = fraction of the window's columns struck by some gear (= eta_B at beta=1)")
    say("=" * 100)
    QS = [59, 97, 199, 499, 997]
    summary = []
    for q in QS:
        gs = gears(q)
        k0, k1, L = window(q)
        say(f"\n  q = {q}: window columns {k0}..{k1}, L = {L}, {len(gs)} gears")
        say(f"  {'beta':>8} {'n blocks':>9} {'eta_B':>9} {'true loss':>10} {'eta_B<1':>8}")
        best = (1e9, None)
        cross = None
        prev = None
        for beta in beta_grid(L):
            eta, loss, dead, per = run_blocks(k0, L, gs, lambda i, g, Q, lQ, b=beta: b,
                                              track_dead=(beta == 1))
            if beta == 1:
                covfrac = dead
            say(f"  {beta:>8} {len(range(0, L, beta)):>9} {eta:>9.5f} {loss:>10.5f} "
                f"{'yes' if eta < 1 else 'NO':>8}")
            if eta < best[0]:
                best = (eta, beta)
            if prev is not None and prev[1] >= 1.0 > eta:
                cross = beta
            prev = (beta, eta)
        say(f"  covered fraction of the window = {covfrac:.5f}   (openings "
            f"{round((1-covfrac)*L)} of {L})")
        say(f"  best fixed beta = {best[1]} with eta_B = {best[0]:.5f};  "
            f"sum 4/g^2 = {sum(4.0/(g*g) for g in gs):.5f}")
        # variable-beta rules
        rules = [
            ("beta_i = g_i", lambda i, g, Q, lQ: g),
            ("beta_i = 4 g_i", lambda i, g, Q, lQ: 4 * g),
            ("beta_i = 16 g_i", lambda i, g, Q, lQ: 16 * g),
            ("beta_i = geomean(gears used)",
             lambda i, g, Q, lQ: math.exp(lQ / i) if i >= 1 else g),
            ("beta_i = Q_{<i} (fibre spacing)",
             lambda i, g, Q, lQ: math.exp(min(lQ, 40.0))),
        ]
        say(f"  {'variable rule':>32} {'eta_B':>9} {'true loss':>10}")
        for name, r in rules:
            eta, loss, _, _ = run_blocks(k0, L, gs, r)
            say(f"  {name:>32} {eta:>9.5f} {loss:>10.5f}")
        summary.append((q, L, best[1], best[0], covfrac))
    say()
    say("  SUMMARY")
    say(f"  {'q':>5} {'L=W(q)':>8} {'best beta':>10} {'min eta_B':>10} {'eta_B(1)=covered':>17}")
    for q, L, b, e, c in summary:
        say(f"  {q:>5} {L:>8} {b:>10} {e:>10.5f} {c:>17.5f}")
    say()

    say("=" * 100)
    say("B.  THE SAME BUDGET ON A FULLY COVERED STRETCH (the validity gate)")
    say("    A covered stretch has no opening, so the theorem forbids eta_B < 1 there.")
    say("    Stretches: the longest opening-free run of each machine found by scanning columns")
    say("    1..SCAN.  F(M) from the certified ladder.")
    say("=" * 100)
    SCAN = 20_000_000
    FLAD = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
    for q in [11, 13, 17, 19, 23, 29, 31]:
        gs = gears(q)
        # find the longest covered run in columns 1..SCAN
        blocked = np.zeros(SCAN + 1, dtype=bool)
        for g in gs:
            u = pow(6, -1, g)
            for r in {u % g, (-u) % g}:
                blocked[r % g:: g] = True
        blocked[0] = False
        # longest run of True
        d = np.diff(np.concatenate(([0], blocked.view(np.int8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        lens = ends - starts
        b = int(np.argmax(lens))
        run_len = int(lens[b])
        run_start = int(starts[b])
        del blocked, d, starts, ends, lens
        say(f"\n  m{q}: longest covered run in 1..{SCAN} is {run_len} columns at {run_start}"
            f"   (certified F = {FLAD[q]})")
        say(f"  {'beta':>8} {'eta_B':>9} {'true loss':>10} {'>=1?':>6}")
        ok = True
        for beta in sorted(set([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, run_len])):
            if beta > run_len:
                continue
            eta, loss, _, _ = run_blocks(run_start, run_len, gs,
                                         lambda i, g, Q, lQ, bb=beta: bb)
            if eta < 1.0:
                ok = False
            say(f"  {beta:>8} {eta:>9.5f} {loss:>10.5f} {'ok' if eta >= 1 else 'FAIL':>6}")
        say(f"  gate: {'PASSED' if ok else 'FAILED'} (eta_B >= 1 at every beta on a covered run)")
    say()


if __name__ == "__main__":
    main()
    with open(os.path.join(OUT, "bm_exact.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nwritten:", os.path.join(OUT, "bm_exact.txt"))
