# Branch 5d.i, follow-ups: (a) which gears are PINNED across the record set and which are free -
# the exact refinement of the parent's "anchor + 7 + top fixed, middle free"; (b) the
# COVERAGE-MAXIMALITY ledger - for every gear, the number of offsets of the record length it
# strikes at the record's phase against the maximum over its phases, for the record, for the
# window's best run, and for a random stretch; (c) the independence baseline for the completion
# counts of f1 (how far below an independent filling the true number sits).
# Record starts for m29 and m31 come from f1_record_frames.py's full-period scan and are
# re-verified here (blocked run of exactly F-1 columns, flanked by openings).
# Self-contained; numpy only.  Run: uv run python research/anchor235/r35/f3_ledger.py
import os
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "f3_ledger.txt")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


LAD = [5, 7, 11, 13, 17, 19, 23, 29, 31]
FK = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
# from f1 (full period, chunked scan)
BIG = {29: [200906186, 877375978],
       31: [1468940243, 11582483683, 21844264616, 31957808056]}


def gears_of(q):
    return [g for g in LAD if g <= q]


def blocked(k, gears):
    return any(k % g in teeth(g) for g in gears)


def starts_of(q):
    gears = gears_of(q)
    F = FK[q]
    if q in BIG:
        ss = BIG[q]
        for s in ss:                       # re-verify: F-1 blocked columns, both flanks open
            assert not blocked(s - 1, gears) and not blocked(s + F - 1, gears)
            assert all(blocked(s + j, gears) for j in range(F - 1))
        return ss
    P = prod(gears)
    w = np.ones(P, bool)
    for g in gears:
        for t in teeth(g):
            w[t::g] = False
    op = np.flatnonzero(w)
    gaps = np.diff(np.concatenate([op, [op[0] + P]]))
    assert int(gaps.max()) == F
    return [int(op[j]) + 1 for j in np.flatnonzero(gaps == F)]


say("Branch 5d.i / f3.  Which gears are pinned; the coverage ledger; the independence baseline.")
say("")
say("=== (a) The degeneracy profile: distinct start residues per gear over the record set.")
say("  A gear with 1 distinct residue is PINNED by the record; 2 means a mirror pair only.")
say("  machine  F   #records   " + "  ".join(f"{g:>4}" for g in LAD))
prof = {}
for q in [13, 17, 19, 23, 29, 31]:
    gears = gears_of(q)
    ss = starts_of(q)
    row = []
    for g in LAD:
        row.append(len({s % g for s in ss}) if g in gears else 0)
    prof[q] = (ss, row)
    say(f"  m{q:<7} {FK[q]:<3} {len(ss):<10} "
        + "  ".join((f"{v:>4}" if v else "   .") for v in row))
say("  (0/'.' = the gear is not in the machine.)")
say("")
say("  The same after quotienting by the mirror (one start kept from each mirror pair):")
say("  machine  F   #classes   " + "  ".join(f"{g:>4}" for g in LAD))
for q in [13, 17, 19, 23, 29, 31]:
    gears = gears_of(q)
    ss, _ = prof[q]
    P = prod(gears)
    F = FK[q]
    keep = []
    seen = set()
    for s in ss:
        m = (P - s - F + 2) % P
        if m in seen:
            continue
        seen.add(s)
        keep.append(s)
    row = [len({s % g for s in keep}) if g in gears else 0 for g in LAD]
    say(f"  m{q:<7} {F:<3} {len(keep):<10} "
        + "  ".join((f"{v:>4}" if v else "   .") for v in row))

say("")
say("=== (b) The coverage ledger.  For a stretch of L columns starting at s, gear g strikes")
say("  c_g(s) = #{j < L : s+j = +-u_g mod g} of them; m_g(L) = max over the phases of s.")
say("  A record is a stretch of L = F-1 whose gears' strikes cover every offset.")
say("")
say("  machine  L=F-1  sum c_g  sum m_g  gears at max / total   overlap = sum c_g - L")
for q in [13, 17, 19, 23, 29, 31]:
    gears = gears_of(q)
    L = FK[q] - 1
    ss, _ = prof[q]
    mg = {}
    for g in gears:
        a, b = teeth(g)
        mg[g] = max(sum(1 for j in range(L) if (r + j) % g in (a, b)) for r in range(g))
    for s in ss[:2]:
        cg = {}
        for g in gears:
            a, b = teeth(g)
            cg[g] = sum(1 for j in range(L) if (s + j) % g in (a, b))
        atmax = sum(1 for g in gears if cg[g] == mg[g])
        say(f"  m{q:<7} {L:<6} {sum(cg.values()):<8} {sum(mg.values()):<8} "
            f"{atmax}/{len(gears)}                 {sum(cg.values()) - L}")
        say(f"      per gear c_g / m_g: "
            + "  ".join(f"{g}:{cg[g]}/{mg[g]}" for g in gears))
    # control: the distribution of "gears at max" over uniform random phases
    rng = np.random.default_rng(12345)
    tally = []
    for _ in range(20000):
        n = 0
        for g in gears:
            a, b = teeth(g)
            r = int(rng.integers(g))
            if sum(1 for j in range(L) if (r + j) % g in (a, b)) == mg[g]:
                n += 1
        tally.append(n)
    tally = np.array(tally)
    say(f"      control (20,000 uniform phase vectors): gears at max mean {tally.mean():.2f}, "
        f"max {int(tally.max())}, P(>= record's) = {(tally >= atmax).mean():.4f}")

say("")
say("=== (b2) Which gears sit at their coverage maximum in a record, and how likely that is")
say("  for a uniform phase.  share_max(g, L) = (# phases attaining m_g) / g.")
say("  machine  L      " + "  ".join(f"{g:>10}" for g in LAD))
for q in [13, 17, 19, 23, 29, 31]:
    gears = gears_of(q)
    L = FK[q] - 1
    ss, _ = prof[q]
    cells = []
    for g in LAD:
        if g not in gears:
            cells.append("         .")
            continue
        a, b = teeth(g)
        cnt = [sum(1 for j in range(L) if (r + j) % g in (a, b)) for r in range(g)]
        m = max(cnt)
        share = cnt.count(m) / g
        atmax = sum(1 for s in ss if cnt[s % g] == m) / len(ss)
        cells.append(f"{atmax:4.2f}/{share:4.2f}")
    say(f"  m{q:<7} {L:<6} " + "  ".join(f"{c:>10}" for c in cells))
say("  (cell = share of the machine's records with that gear at its coverage maximum / share of")
say("   phases that attain the maximum, i.e. the chance for a column picked at random.)")

say("")
say("=== (b3) How rare a record start is, against the size of the window.")
say("  machine  P            #records   density      window cols   expected records in a window")
for q in [13, 17, 19, 23, 29, 31]:
    gears = gears_of(q)
    P = prod(gears)
    ss, _ = prof[q]
    nxt = {13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37}[q]
    lo = q // 6 + 1
    hi = (nxt * nxt - 1) // 6
    wcols = hi - lo + 1
    dens = len(ss) / P
    say(f"  m{q:<7} {P:<12} {len(ss):<10} {dens:<12.3e} {wcols:<13} {dens * wcols:.3e}")

say("")
say("=== (d) P1 and P2 scored exactly: the top gear's corridor (its strike columns mod 35) and")
say("  its word (the offsets between consecutive strikes), both reduced by the mirror.")
say("  machine  #records  distinct corridors (mod 35, up to c -> -c)   distinct words (up to reversal)")
for q in [13, 17, 19, 23, 29, 31]:
    gears = gears_of(q)
    L = FK[q] - 1
    ss, _ = prof[q]
    a, b = teeth(q)
    cors, words = set(), set()
    for s in ss:
        offs = [j for j in range(L) if (s + j) % q in (a, b)]
        c = frozenset((s + j) % 35 for j in offs)
        cm = frozenset((-x) % 35 for x in c)
        cors.add(min(sorted(c), sorted(cm)) and tuple(sorted(min(sorted(c), sorted(cm)))))
        w = tuple(offs[i + 1] - offs[i] for i in range(len(offs) - 1))
        words.add(min(w, w[::-1]))
    say(f"  m{q:<7} {len(ss):<9} {len(cors):<44} {len(words)}")
    say(f"      corridors {sorted(cors)}")
    say(f"      words {sorted(words)}   (letters a = {2 * min(a, b)}, b = {q - 2 * min(a, b)})")

say("")
say("=== (c) The independence baseline for the completion counts of f1.")
say("  If the middle gears filled the frame's open offsets independently, the chance that a")
say("  given offset is left open by all of them is pi = prod (1 - 2/g), so the expected number")
say("  of completions among the N = P/(35q) frame columns is N (1 - pi)^|R|.")
say("  machine  N          |R|   pi       expected   actual   actual/expected")
ACT = {13: (11, 2, 1), 17: (143, 3, 2), 19: (2431, 6, 2), 23: (46189, 10, 2),
       29: (1062347, 16, 1), 31: (30808063, 20, 1)}
for q in [13, 17, 19, 23, 29, 31]:
    mid = [g for g in gears_of(q) if g not in (5, 7, q)]
    pi = prod((g - 2) / g for g in mid) if mid else 1.0
    N, R, C = ACT[q]
    exp = N * (1 - pi) ** R
    say(f"  m{q:<7} {N:<10} {R:<5} {pi:<8.4f} {exp:<10.3f} {C:<8} {C / exp:.3f}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written", OUT)
