"""Paths through the word tree to each twin - manager, round 29.

Pre-registration: research/data/r29/twin_path_prereg.md (U1..U5).

A twin at slot k is an opening at every level p (sub-machine m_p = gears 5..p); its path is
the pair of flank chains L_p(k), R_p(k) = distance to the nearest opening of m_p on each
side. Walking left from k, the death rungs r(k-1), r(k-2), ... of the flank slots determine
the chain: L_p = position of the first slot with death rung > p, so the gears that touch the
flank are the PREFIX MAXIMA of that sequence and the number of old openings fused by such a
gear is the count of slots equal to it before the next prefix maximum.

Usage: python twin_path_r29.py [--machines 53,199,997,4999] [--examples 53,997]
"""
import argparse
import math
from collections import Counter

import numpy as np

from word_tree_r29 import spf_sieve, death_rungs, runs_of

NGATE, NFAIL = 0, 0


def gate(cond, msg):
    # non-fatal here so every pre-registered statement is scored in one run
    global NGATE, NFAIL
    NGATE += 1
    if not cond:
        NFAIL += 1
        print("  GATE FAIL: " + msg)
    else:
        print("  ASSERT ok: " + msg)

GEARS_SMALL = [5, 7, 11, 13, 17, 19]


def flank_events(seq):
    """seq = death rungs walking away from the twin (stops before the next opening).
    Returns list of (gear, fused_count, position) for the prefix maxima, in order."""
    events = []
    cur = 0
    pos = []
    for d, v in enumerate(seq):
        if v > cur:
            cur = v
            pos.append((d, v))
    for j, (d, g) in enumerate(pos):
        d2 = pos[j + 1][0] if j + 1 < len(pos) else len(seq)
        fused = sum(1 for x in seq[d:d2] if x == g)
        events.append((g, fused, d))
    return events


def L_at_level(seq, p):
    for d, v in enumerate(seq):
        if v > p:
            return d + 1
    return len(seq) + 1


def exact_gap_dist(p_max):
    """cyclic gap distribution of the sub-machine with gears 5..p_max (period scan)."""
    gears = [g for g in [5, 7, 11, 13, 17, 19, 23] if g <= p_max]
    P = 1
    for g in gears:
        P *= g
    blocked = np.zeros(P, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        blocked[u % g::g] = True
        blocked[(-u) % g::g] = True
    op = np.flatnonzero(~blocked)
    gaps = np.diff(np.concatenate((op, [op[0] + P])))
    c = Counter(gaps.tolist())
    n = len(gaps)
    return {k: v / n for k, v in c.items()}


def tv(d1, d2):
    keys = set(d1) | set(d2)
    return 0.5 * sum(abs(d1.get(k, 0) - d2.get(k, 0)) for k in keys)


def iid_records_mean(rung_probs, lengths):
    """expected number of strict prefix maxima in n iid draws from rung_probs, averaged over
    the observed flank lengths (n = flank length - 1 = number of blocked slots)."""
    vals = sorted(rung_probs)
    pr = np.array([rung_probs[v] for v in vals])
    cdf_below = np.concatenate(([0.0], np.cumsum(pr)[:-1]))  # P(x < v)
    nmax = max(lengths)
    # E[records in n draws] = sum_{j=1..n} sum_v pr_v * cdf_below_v^(j-1)
    j = np.arange(nmax)
    per_j = (pr[None, :] * cdf_below[None, :] ** j[:, None]).sum(axis=1)  # term for draw j+1
    cum = np.cumsum(per_j)  # cum[n-1] = expected records in n draws
    lens = np.array(lengths)
    return float(np.where(lens > 0, cum[np.maximum(lens - 1, 0)], 0.0).mean())


def print_path(k, seq_l, seq_r, tag):
    ev_l = flank_events(seq_l)
    ev_r = flank_events(seq_r)
    print(f"    {tag}: twin at slot {k} ({6 * k - 1}, {6 * k + 1}); "
          f"left gap {len(seq_l) + 1}, right gap {len(seq_r) + 1}")
    print(f"      left  death rungs outward: {seq_l}")
    print(f"      right death rungs outward: {seq_r}")
    for side, ev, seq in (("left ", ev_l, seq_l), ("right", ev_r, seq_r)):
        L_prev = 1
        parts = []
        for g, fused, d in ev:
            # flank at level g: first slot with rung > g
            L_new = L_at_level(seq, g)
            prev_gear_slots = seq[:L_new - 1]
            lower = [x for x in prev_gear_slots if x < g]
            pg = max(lower) if lower else None
            blk = np.array([x <= pg for x in prev_gear_slots]) if pg else np.zeros(len(prev_gear_slots), dtype=bool)
            tup = [e - s + 1 for s, e in runs_of(blk)] if len(blk) else []
            parts.append(f"gear {g} fuses {fused} old opening(s): flank {L_prev} -> {L_new}, "
                         f"tuple below {tup}")
            L_prev = L_new
        print(f"      {side} path ({len(ev)} events): " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", type=str, default="53,199,997,4999")
    ap.add_argument("--examples", type=str, default="53,997")
    a = ap.parse_args()
    machines = [int(x) for x in a.machines.split(",")]
    examples = {int(x) for x in a.examples.split(",") if x}

    qmax = max(machines)
    primes = [int(p) for p in np.flatnonzero(spf_sieve(qmax * 2) == np.arange(qmax * 2 + 1)) if p >= 5]
    q_next = {p: primes[i + 1] for i, p in enumerate(primes[:-1])}
    Wmax = (q_next[qmax] ** 2 - 2) // 6
    spf = spf_sieve(6 * Wmax + 1)
    r_all = death_rungs(Wmax, spf)

    exact = {p: exact_gap_dist(p) for p in GEARS_SMALL}
    summary = {}
    for q in machines:
        W = (q_next[q] ** 2 - 2) // 6
        lo = q // 6 + 1
        r = r_all[:W]
        twins = [int(i) for i in np.flatnonzero(r == 0) if i >= lo - 1]
        seqs = []
        skipped = 0
        for i in twins:
            # left
            j = i - 1
            sl = []
            while j >= 0 and r[j] != 0:
                sl.append(int(r[j])); j -= 1
            j2 = i + 1
            sr = []
            while j2 < W and r[j2] != 0:
                sr.append(int(r[j2])); j2 += 1
            if j < lo - 1 or j2 >= W:
                skipped += 1
                continue
            seqs.append((i + 1, sl, sr))
        n = len(seqs)
        print(f"\n=== m{q}: W={W}, twins in window {len(twins)}, with both flanks inside {n} "
              f"(skipped {skipped}) ===")
        ev_l = [flank_events(sl) for _, sl, _ in seqs]
        ev_r = [flank_events(sr) for _, _, sr in seqs]
        nl = [len(e) for e in ev_l]; nr = [len(e) for e in ev_r]
        Lq = [len(sl) + 1 for _, sl, _ in seqs]; Rq = [len(sr) + 1 for _, _, sr in seqs]
        maxg_l = [max(sl) if sl else 0 for _, sl, _ in seqs]
        maxg_r = [max(sr) if sr else 0 for _, _, sr in seqs]
        # fusion arity by gear class
        ar7 = max([f for e in ev_l + ev_r for g, f, _ in e if g == 7] + [0])
        ar5 = max([f for e in ev_l + ev_r for g, f, _ in e if g == 5] + [0])
        ar_big = max([f for e in ev_l + ev_r for g, f, _ in e if g >= 11] + [0])
        ar_big_who = Counter(f for e in ev_l + ev_r for g, f, _ in e if g >= 11)
        # growth per event: at each single-kill event with gear >= 11, the flank absorbs the
        # neighbouring run beyond one old opening; record new/old flank ratio and whether
        # the absorbed run is at least as long as the flank it joins
        ratios, absorbed_ge = [], 0
        n_ev = 0
        for (_, sl, _), ev in zip(seqs, ev_l):
            L_prev = 1
            for g, fused, d in ev:
                L_new = L_at_level(sl, g)
                if g >= 11 and fused == 1:
                    n_ev += 1
                    ratios.append(L_new / L_prev)
                    if L_new - L_prev - 1 >= L_prev - 1:
                        absorbed_ge += 1
                L_prev = L_new
        if ratios:
            rs = np.array(ratios)
            print(f"  growth per single-kill event (gears >= 11, left flank, {n_ev} events): "
                  f"median new/old flank {np.median(rs):.2f}, mean {rs.mean():.2f}, "
                  f"absorbed run >= current flank in {absorbed_ge / n_ev:.3f}")
        # rung distribution of blocked window slots (for the iid model)
        rw = r[lo - 1:]
        bw = rw[rw > 0]
        cnt = Counter(bw.tolist())
        rung_probs = {k: v / bw.size for k, v in cnt.items()}
        iid_l = iid_records_mean(rung_probs, Lq)
        frac_big_l = sum(1 for g in maxg_l if g > q / 2) / n
        frac_big_r = sum(1 for g in maxg_r if g > q / 2) / n
        # event-count and gap distributions
        dist = lambda xs: {k: v / len(xs) for k, v in Counter(xs).items()}
        tv_LR = tv(dist(Lq), dist(Rq))
        print(f"  mean flank gap L_q {np.mean(Lq):.2f} R_q {np.mean(Rq):.2f} max {max(Lq)} / {max(Rq)}; "
              f"TV(L_q, R_q) = {tv_LR:.4f}")
        print(f"  fusion events per twin: left mean {np.mean(nl):.3f} right mean {np.mean(nr):.3f} "
              f"max {max(nl)} / {max(nr)}; distribution left {sorted(Counter(nl).items())}")
        print(f"  iid-records model (Mertens rungs, observed flank lengths): {iid_l:.3f} "
              f"(observed/model {np.mean(nl) / iid_l:.3f})")
        print(f"  fusion arity: gear 5 max {ar5}, gear 7 max {ar7}, gears >= 11 max {ar_big} "
              f"(arity counts {sorted(ar_big_who.items())})")
        print(f"  largest gear on flank > q/2: left {frac_big_l:.3f} right {frac_big_r:.3f}")
        # the gear that touches a flank: how often each gear class is the LAST (largest)
        lastc = Counter()
        for g in maxg_l:
            lastc["<=7" if g <= 7 else "<=q/10" if g <= q / 10 else "<=q/2" if g <= q / 2 else ">q/2"] += 1
        print(f"  largest flank gear class: {dict((k, round(v / n, 3)) for k, v in lastc.items())}")
        # first gear on the flank (r(k-1)) distribution vs unconditional death-rung distribution
        first = Counter(sl[0] if sl else 0 for _, sl, _ in seqs)
        print(f"  r(k-1) for twins: adjacent twin {first[0] / n:.3f} 5 {first[5] / n:.3f} 7 {first[7] / n:.3f} 11 {first[11] / n:.3f} "
              f"| unconditional blocked-slot rung: 5 {rung_probs.get(5, 0):.3f} 7 {rung_probs.get(7, 0):.3f} "
              f"11 {rung_probs.get(11, 0):.3f} | residue prediction P(r(k-1)=5 | twin) = 2/3")
        if q == 997:
            print("  U1 low-level flank distributions vs exact m_p gap distribution:")
            for p in GEARS_SMALL:
                Lp = [L_at_level(sl, p) for _, sl, _ in seqs]
                d = tv(dist(Lp), exact[p])
                print(f"    p={p:>2}: mean L_p {np.mean(Lp):.3f} exact mean gap "
                      f"{sum(k * v for k, v in exact[p].items()):.3f} TV {d:.4f}")
                gate(d < 0.05, f"U1 p={p}: TV {d:.4f} < 0.05")
            gate(tv_LR < 0.03, f"U5 TV(L_q,R_q) {tv_LR:.4f} < 0.03 at m997")
            gate(abs(np.mean(nl) - np.mean(nr)) <= 0.05 * np.mean(nl), "U5 mean event counts agree to 5% at m997")
        if q <= 997:
            gate(ar7 <= 2, f"U2 gear-7 arity <= 2 at m{q} (max {ar7})")
            gate(ar_big <= 3, f"U2 gears >= 11 arity <= 3 at m{q} (max {ar_big})")
        summary[q] = (float(np.mean(nl)), iid_l, frac_big_l)
        if q in examples:
            print(f"  --- example paths at m{q} ---")
            by_L = sorted(range(n), key=lambda t: Lq[t])
            picks = [(0, "first twin"), (1, "second twin"), (2, "third twin"),
                     (by_L[n // 2], "median-flank twin"), (by_L[-1], "largest-left-flank twin")]
            for idx, tag in picks:
                k, sl, sr = seqs[idx]
                print_path(k, sl, sr, tag)
    print("\n=== U3 path length ===")
    for q, (m, iid, fb) in summary.items():
        print(f"  m{q}: mean left events {m:.3f}, iid model {iid:.3f}, ratio {m / iid:.3f}; "
              f"flank max gear > q/2: {fb:.3f}")
    ms = [summary[q][0] for q in machines]
    gate(all(2 <= m <= 6 for m in ms), "U3 mean left events in [2, 6] at every machine")
    gate(all(ms[i] < ms[i + 1] for i in range(len(ms) - 1)), "U3 mean left events increase with q")
    gate(ms[-1] - ms[0] < 3, f"U3 increase m{machines[0]} -> m{machines[-1]} < 3 ({ms[-1] - ms[0]:.3f})")
    gate(all(abs(summary[q][0] / summary[q][1] - 1) <= 0.25 for q in machines),
         "U3 iid-records model within 25% at every machine")
    if 53 in summary:
        gate(0.25 <= summary[53][2] <= 0.55, f"U4 m53 big-gear flank fraction {summary[53][2]:.3f} in [0.25, 0.55]")
    if 4999 in summary:
        gate(0.05 <= summary[4999][2] <= 0.20, f"U4 m4999 big-gear flank fraction {summary[4999][2]:.3f} in [0.05, 0.20]")
    print(f"\nGATES: {NGATE - NFAIL} passed, {NFAIL} failed of {NGATE}")


if __name__ == "__main__":
    main()
