"""hc_letters.py -- non-circular version of the letter test.

For every layer of every record stretch (m11..m31) and of every window longest stretch
(q = 23..997), take every pair of CONSECUTIVE openings struck by the next gear g and
classify the distance dl:
    pad          dl = 0 (mod g)                       -> half-column is not u_g
    letter k=0   dl in {a_g, b_g}                     -> half/quarter-column IS u_g  (P1)
    letter k>=1  dl = a_g or b_g plus a multiple of g -> half-column is not u_g
    illegal      anything else                        -> would refute the chain law
and report how the record's and the window's distances split.  Also reports how far each
piece of the top-layer word half-columns: into the frame (c <= q/6) or into the window.

Writes results/hc_letters.txt.
"""
import os
import numpy as np
from sympy import primerange

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

GEARS9 = [5, 7, 11, 13, 17, 19, 23, 29, 31]
DEEP = {29: (43, [200906185, 877375977]), 31: (58, [1468940242, 21844264615])}


def u_of(g):
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def teeth(g):
    u = u_of(g)
    return (u % g, (-u) % g)


def col_index(v):
    if v % 2 == 0:
        return v // 2
    return (v - 1) // 4 if v % 4 == 1 else (v + 1) // 4


def classify(dl, g):
    a, b = 2 * u_of(g), g - 2 * u_of(g)
    if dl % g == 0:
        return "pad"
    if dl % g == a % g:
        return "letter_a" if dl == a else "letter_a+kg"
    if dl % g == b % g:
        return "letter_b" if dl == b else "letter_b+kg"
    return "ILLEGAL"


def records(top):
    gears = [g for g in GEARS9 if g <= top]
    P = 1
    for g in gears:
        P *= g
    blocked = np.zeros(P, dtype=bool)
    for g in gears:
        for t in teeth(g):
            blocked[t % g::g] = True
    idx = np.flatnonzero(~blocked)
    d = np.diff(idx)
    F = int(d.max())
    return F, [int(idx[i]) for i in np.flatnonzero(d == F)][:2]


def analyse(gears, x, V, tag, lines, tally):
    for i, g in enumerate(gears):
        low = gears[:i + 1]
        nxt = gears[i + 1] if i + 1 < len(gears) else None
        if nxt is None:
            continue
        ops = [k for k in range(x, x + V + 1)
               if not any(k % gg in teeth(gg) for gg in low)]
        struck = [k for k in ops[1:-1] if k % nxt in teeth(nxt)]
        for j in range(len(struck) - 1):
            dl = struck[j + 1] - struck[j]
            cl = classify(dl, nxt)
            tally[cl] = tally.get(cl, 0) + 1
            onhome = col_index(dl) == u_of(nxt)
            key = ("home" if onhome else "away")
            tally[key] = tally.get(key, 0) + 1
            if cl.startswith("letter") and cl.endswith("kg"):
                tally["letter_kg"] = tally.get("letter_kg", 0) + 1
            if cl == "ILLEGAL":
                lines.append("   ILLEGAL %s gear %d distance %d" % (tag, nxt, dl))
            if cl in ("letter_a", "letter_b") and not onhome:
                lines.append("   LETTER MISS %s gear %d distance %d" % (tag, nxt, dl))


def main():
    lines = []
    # ---- the records ----
    tally = {}
    for top in [11, 13, 17, 19, 23, 29, 31]:
        gears = [g for g in GEARS9 if g <= top]
        if top in DEEP:
            F, pos = DEEP[top]
        else:
            F, pos = records(top)
        for x in pos:
            analyse(gears, x, F, "m%d@%d" % (top, x), lines, tally)
    lines.append("RECORD stretches m11..m31 (2 per machine), every consecutive struck pair:")
    for k in sorted(tally):
        lines.append("   %-14s %d" % (k, tally[k]))
    nonpad = sum(v for k, v in tally.items() if k.startswith("letter"))
    lines.append("   distances landing on the new gear's HOME column: %d of %d (all), %d of %d (non-pad)"
                 % (tally.get("home", 0), tally.get("home", 0) + tally.get("away", 0),
                    tally.get("letter_a", 0) + tally.get("letter_b", 0),
                    tally.get("letter_a", 0) + tally.get("letter_b", 0) + tally.get("letter_kg", 0)))
    lines.append("")

    # ---- the window's longest stretches ----
    tally2 = {}
    halfloc = {"frame": 0, "window": 0}
    rows = []
    for q in primerange(23, 1000):
        gears = list(primerange(5, q + 1))
        lo, hi = q // 6 + 1, (q * q - 1) // 6 + 1
        n = hi - lo
        b = np.zeros(n, dtype=bool)
        for g in gears:
            for t in teeth(g):
                b[(t - lo) % g::g] = True
        idx = np.flatnonzero(~b) + lo
        d = np.diff(idx)
        V = int(d.max())
        x = int(idx[int(np.argmax(d))])
        analyse(gears, x, V, "win%d" % q, lines, tally2)
        c = col_index(V)
        halfloc["frame" if c <= q // 6 else "window"] += 1
        rows.append((q, V, c, q // 6))
    lines.append("WINDOW longest stretch, rungs q = 23..997 (160 rungs):")
    for k in sorted(tally2):
        lines.append("   %-14s %d" % (k, tally2[k]))
    lines.append("   distances landing on the closing gear's HOME column: %d of %d (all), %d of %d (non-pad)"
                 % (tally2.get("home", 0), tally2.get("home", 0) + tally2.get("away", 0),
                    tally2.get("letter_a", 0) + tally2.get("letter_b", 0),
                    tally2.get("letter_a", 0) + tally2.get("letter_b", 0) + tally2.get("letter_kg", 0)))
    lines.append("   half-column of the stretch length: in the FRAME (c <= q/6) at %d rungs, "
                 "in the WINDOW at %d rungs" % (halfloc["frame"], halfloc["window"]))
    lines.append("   the rungs whose half-column is still in the window: %s"
                 % [(q, V, c, fr) for (q, V, c, fr) in rows if c > fr])
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "hc_letters.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
