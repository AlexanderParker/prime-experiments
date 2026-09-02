"""Word tree inside the reduction window - manager, round 29.

Pre-registration: research/data/r29/word_tree_prereg.md (T1..T5).

For machine M (gears 5..q) the reduction window is (q/6, W], W = (q_next^2 - 2)/6 (so 6W+1 < q_next^2). Every
slot k in the window is either a twin prime pair (6k-1, 6k+1) or blocked; a blocked slot's
DEATH RUNG r(k) is the smallest prime p >= 5 dividing 6k-1 or 6k+1 (inside the window every
composite has such a factor <= q). The word tree of a blocked run [a,b] at level p is: the
runs of machine m_p inside [a,b] (its n-tuple), recursively down the gears; leaves are the
individual kills labelled by their gear.

Usage: python word_tree_r29.py [--qmax 2000] [--trees 53,199,997]
"""
import argparse
import sys

import numpy as np

NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def spf_sieve(n):
    """smallest prime factor for 0..n (0/1 -> 0)."""
    spf = np.zeros(n + 1, dtype=np.int64)
    spf[2::2] = 2
    for i in range(3, int(n ** 0.5) + 1, 2):
        if spf[i] == 0:
            sl = spf[i * i::2 * i]
            sl[sl == 0] = i
            spf[i * i::2 * i] = sl
    odd = np.arange(3, n + 1, 2)
    z = odd[spf[odd] == 0]
    spf[z] = z
    spf[1] = 0
    return spf


def death_rungs(W, spf):
    """r(k) for k = 1..W: smallest prime >= 5 dividing 6k-1 or 6k+1; 0 if both prime."""
    k = np.arange(1, W + 1, dtype=np.int64)
    lo, hi = 6 * k - 1, 6 * k + 1
    a = spf[lo]; b = spf[hi]
    # a number that is prime has spf == itself; it contributes no gear (gear = proper factor)
    a = np.where(a == lo, np.iinfo(np.int64).max, a)
    b = np.where(b == hi, np.iinfo(np.int64).max, b)
    r = np.minimum(a, b)
    r[r == np.iinfo(np.int64).max] = 0
    return r  # index i <-> slot i+1


def runs_of(blocked):
    """maximal runs of True in a boolean array -> list of (start, end) inclusive indices."""
    if not blocked.any():
        return []
    d = np.diff(np.concatenate(([0], blocked.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def tree_of_run(r, a, b, gears):
    """n-tuple decomposition of the run [a,b] (slot indices into r) level by level.

    Returns list of (gear p, kills at this level, number of children runs) for the gears
    that killed at least one slot in the run, ascending, plus the sealing-gear set."""
    sub = r[a:b + 1]
    present = sorted(set(int(x) for x in np.unique(sub)))
    levels = []
    for p in present:
        # runs of machine m_p inside [a,b]: slots with death rung <= p are blocked at level p
        blk = (sub <= p)
        nruns = len(runs_of(blk))
        kills = int((sub == p).sum())
        levels.append((p, kills, nruns))
    return levels, present


def print_tree(r, a, b, gears, qmax_level, indent="    "):
    """Draw the fusion tree of run [a,b] top-down: at each level show the tuple of
    sub-runs of the previous level and the kills that joined them."""
    sub = r[a:b + 1]
    present = sorted(set(int(x) for x in np.unique(sub)), reverse=True)
    print(f"{indent}run slots [{a + 1}, {b + 1}] length {b - a + 1}, sealing gears {sorted(present)}")
    for p in present:
        lower = [g for g in present if g < p]
        prev_level = max(lower) if lower else None
        blk_prev = (sub <= prev_level) if prev_level else np.zeros_like(sub, dtype=bool)
        runs_prev = runs_of(blk_prev)
        kills = np.flatnonzero(sub == p)
        tup = [e - s + 1 for s, e in runs_prev]
        print(f"{indent}  gear {p:>5}: kills {len(kills):>3} at offsets {kills.tolist()[:12]}"
              f"{' ...' if len(kills) > 12 else ''} -> fuses a {len(tup)}-tuple {tup[:16]}"
              f"{' ...' if len(tup) > 16 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmax", type=int, default=2000)
    ap.add_argument("--trees", type=str, default="53,199,997")
    a = ap.parse_args()
    trees = {int(x) for x in a.trees.split(",") if x}

    primes = [int(p) for p in np.flatnonzero(spf_sieve(a.qmax * 2) == np.arange(a.qmax * 2 + 1)) if p >= 2]
    ps = [p for p in primes if p >= 5]
    q_last = max(p for p in ps if p <= a.qmax)
    q_next_last = ps[ps.index(q_last) + 1]
    Wmax = (q_next_last ** 2 - 2) // 6
    spf = spf_sieve(6 * Wmax + 1)
    r_all = death_rungs(Wmax, spf)

    F_known = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 29: 43, 31: 58, 37: 88, 41: 91,
               43: 103, 47: 118, 53: 145}
    print("=== word tree inside the window: per-machine summary ===")
    print(f"{'q':>5} {'W':>8} {'#open':>6} {'G_W':>5} {'G_W/W':>7} {'F(M)':>5} {'#seal':>5} "
          f"{'last':>5} {'last/q':>6} {'depth':>5} {'<=7':>5} {'<=13':>5} {'<=19':>5}")
    t1_bad, t2_bad, t3_bad, t4_big, t4_n, t5_max = 0, 0, 0, 0, 0, 0
    t2_list, t5_list = [], []
    t3c_bad = 0
    show = {5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 97, 199, 499, 997, 1999} | trees
    for i, q in enumerate(ps):
        if q > a.qmax:
            break
        qn = ps[i + 1]
        W = (qn * qn - 2) // 6  # 6W+1 < q_next^2 strictly
        lo = q // 6 + 1  # first slot with 6k-1 > q
        r = r_all[:W].copy()
        # death rung relative to THIS machine: gears > q do not exist yet, but inside the
        # window no slot needs them (gate below)
        gate_needed = int((r[lo - 1:] > q).sum())
        if gate_needed:
            gate(False, f"m{q}: a window slot needs a gear > q ({gate_needed} slots)")
        blocked = (r > 0)
        win = blocked[lo - 1:]
        n_open = int((~win).sum())
        rs = runs_of(win)
        if rs:
            s, e = max(rs, key=lambda t: t[1] - t[0])
            s += lo - 1; e += lo - 1
            G = e - s + 1  # blocked slots; gap between openings = G + 1
        else:
            s = e = None; G = 0
        gap = G + 1 if G else 0
        levels, present = tree_of_run(r, s, e, ps) if G else ([], [])
        nseal = len(present)
        last = max(present) if present else 0
        depth = len(levels)
        rw = r[lo - 1:]
        bw = rw[rw > 0]
        f7 = float((bw <= 7).mean()) if bw.size else 0.0
        f13 = float((bw <= 13).mean()) if bw.size else 0.0
        f19 = float((bw <= 19).mean()) if bw.size else 0.0
        F = F_known.get(q)
        if q >= 50 and gap / W >= 0.10:
            t1_bad += 1
        if F is not None and gap >= F:
            t2_bad += 1
            t2_list.append((q, gap, F))
        if q >= 100 and not (abs(f7 - 4 / 7) <= 0.03 and abs(f13 - 0.7033) <= 0.03):
            t3_bad += 1
        # corrected normalisation: Mertens fractions are of ALL window slots, the table's
        # fractions are of BLOCKED slots; multiply by the blocked density
        dens = 1 - n_open / (W - lo + 1)
        if q >= 100 and not (abs(f7 * dens - 4 / 7) <= 0.01 and abs(f13 * dens - 0.7033) <= 0.01):
            t3c_bad += 1
        if G:
            t4_n += 1
            if last > q / 2:
                t4_big += 1
        t5_max = max(t5_max, depth)
        if q <= 2000 and depth > 12:
            t5_list.append((q, depth))
        if q in show:
            print(f"{q:>5} {W:>8} {n_open:>6} {gap:>5} {gap / W:>7.4f} {str(F) if F else '-':>5} "
                  f"{nseal:>5} {last:>5} {last / q:>6.3f} {depth:>5} {f7:>5.3f} {f13:>5.3f} {f19:>5.3f}")
        if q in trees and G:
            print(f"  --- fusion tree of the maximal in-window run at m{q} (gap {gap} of W={W}) ---")
            print_tree(r, s, e, ps, q)
    nq = sum(1 for p in ps if p <= a.qmax)
    print(f"\nmachines: {nq} (q <= {a.qmax}); window max {Wmax}")
    gate(t1_bad == 0, f"T1: G_W/W < 0.10 at every q >= 50 ({t1_bad} exceptions)")
    # T2 as pre-registered (strict at every corpus rung) is REFUTED at m5 and m7, where the
    # in-window gap EQUALS the record (2 = 2, 5 = 5: the window is the whole period there).
    # Recorded honestly; the gate below is the surviving statement (q >= 11).
    print(f"T2: gap >= F(M) at {t2_list} -> pre-registered strict form "
          f"{'HELD' if not t2_list else 'REFUTED (equality at the two smallest machines)'}")
    gate(all(q < 11 for q, _, _ in t2_list), "T2': in-window max gap < F(M) at every corpus rung q >= 11")
    # T3 as pre-registered is REFUTED: the prediction normalised by blocked slots, Mertens
    # normalises by all slots. The corrected form is the gate.
    print(f"T3: as pre-registered (fractions of BLOCKED slots within 0.03 of 4/7 and 0.7033): "
          f"{'HELD' if t3_bad == 0 else f'REFUTED ({t3_bad} exceptions; prediction mis-normalised)'}")
    gate(t3c_bad == 0, f"T3': death-rung fractions x blocked density within 0.01 of Mertens 4/7 and "
                       f"0.7033 at every q >= 100 ({t3c_bad} exceptions)")
    print(f"T4: last sealer > q/2 at {t4_big}/{t4_n} machines "
          f"({'HELD' if t4_big * 2 < t4_n else 'REFUTED'} as pre-registered 'fewer than half')")
    print(f"T5: fusion depth <= 12 at q <= 2000: {'HELD' if not t5_list else 'REFUTED'} "
          f"(first exceptions {t5_list[:5]}, max depth {t5_max})")
    print(f"\nALL {NGATE} ASSERTION GATES PASSED")


if __name__ == "__main__":
    main()
