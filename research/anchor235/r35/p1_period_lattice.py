# Branch 5d.ii, question 1: the deletion profile of the PERIOD record.
# For M = {5..q}, q = 7..23: F(M), every record stretch, the per-gear drop F(M) - F(M minus g)
# (gear 5 included, which the r34 table omits), the sole-strike map of each record stretch, and
# the full deletion LATTICE F(M minus S) over every subset S of gears (subadditivity test).
# Max-gap convention: F = max distance between consecutive openings, wrap included.
# Self-contained; numpy only.  Run: uv run python research/anchor235/r35/p1_period_lattice.py
import numpy as np, os, time, itertools
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "p1_period_lattice.txt")
lines = []
def say(s=""):
    print(s); lines.append(s)

def openings(gears, P):
    w = np.ones(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        w[u::g] = False
        w[g - u::g] = False
    return np.flatnonzero(w)

def gaps_of(op, P):
    return np.diff(np.concatenate([op, [op[0] + P]]))

def record(gears):
    """F and the list of record stretch starts (opening before the stretch)."""
    if not gears:
        return 1, []
    P = prod(gears)
    op = openings(gears, P)
    gp = gaps_of(op, P)
    F = int(gp.max())
    starts = [int(op[j]) for j in np.flatnonzero(gp == F)]
    return F, starts

def strikers(k, gears):
    out = []
    for g in gears:
        u = pow(6, -1, g)
        if k % g in (u, g - u):
            out.append(g)
    return out

ladder = [7, 11, 13, 17, 19, 23]
say("Branch 5d.ii Q1: deletion profile of the period record.")
say("drop(g) = F(M) - F(M minus g).  L4 (pair_statement.md) forces every drop > 0 whenever the")
say("record's endpoints have flanks summing above F; measured here directly for every gear.")
t0 = time.time()
prof = {}
for i, q in enumerate(ladder):
    gears = [5] + ladder[:i + 1]
    P = prod(gears)
    F, starts = record(gears)
    say(f"\n=== M = {{5..{q}}}  P = {P}  F = {F}  record stretches: {len(starts)} (starts {starts[:8]})")
    # per-gear drop
    row = []
    for g in gears:
        Fg, _ = record([h for h in gears if h != g])
        row.append((g, Fg, F - Fg))
    prof[q] = row
    say("  gear      F(M-g)   drop   drop/F")
    for g, Fg, d in row:
        say(f"  {g:>4}   {Fg:>7}   {d:>4}   {d / F:6.3f}")
    say(f"  profile falls with g? {[d for _, _, d in row]}  argmax gear = {max(row, key=lambda r: r[2])[0]}")
    # sole-strike map of every record stretch (first 4 shown in full)
    say("  sole-strike map of the record stretches (columns are relative to the stretch start x+1;")
    say("  'k*' = gear g is the SOLE striker of that column):")
    for x in starts[:4]:
        cols = list(range(x + 1, x + F))
        per = {g: [] for g in gears}
        sole = {g: [] for g in gears}
        for j, k in enumerate(cols):
            st = strikers(k, gears)
            for g in st:
                per[g].append(j)
            if len(st) == 1:
                sole[st[0]].append(j)
        say(f"    x = {x} (frac {(x + 1) / P:.4f}):  " +
            "  ".join(f"{g}:{len(per[g])}k/{len(sole[g])}s" for g in gears))
        say("      sole columns: " + "  ".join(f"{g}:{sole[g]}" for g in gears))
        chk = [g for g in gears if not sole[g]]
        say(f"      gears with NO sole column in this stretch: {chk if chk else 'none'}")
    # deletion lattice over subsets
    say("  deletion lattice F(M minus S) (subadditivity: F - F(M-S) <= sum of drops?)")
    single = {g: d for g, _, d in row}
    viol = 0
    tot = 0
    worst = None
    lat = {}
    for r in range(1, len(gears) + 1):
        for S in itertools.combinations(gears, r):
            FS, _ = record([h for h in gears if h not in S])
            lat[S] = FS
            joint = F - FS
            bound = sum(single[g] for g in S)
            tot += 1
            if joint > bound:
                viol += 1
            if r == 2:
                pass
            if worst is None or (bound - joint) < worst[0]:
                worst = (bound - joint, S, joint, bound)
    say(f"    subsets tested {tot}; subadditivity violations {viol}; tightest slack {worst}")
    say("    pairs (S, F(M-S), joint drop, sum of single drops):")
    for S in itertools.combinations(gears, 2):
        say(f"      {S}  F={lat[S]:>4}  joint={F - lat[S]:>3}  singles={single[S[0]] + single[S[1]]:>3}")
    say(f"    all gears removed but 5: F = {lat[tuple(g for g in gears if g != 5)]}; "
        f"only 5 removed: F = {lat[(5,)]}")
    say(f"  [{time.time() - t0:.1f}s]")

say("\nSummary of the period profile (drop by gear, machines down the rows):")
say("  M      " + "  ".join(f"{g:>5}" for g in [5] + ladder))
for q in ladder:
    d = {g: v for g, _, v in prof[q]}
    say(f"  {{5..{q:>2}}}  " + "  ".join(f"{d.get(g, ''):>5}" for g in [5] + ladder))

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", OUT)
