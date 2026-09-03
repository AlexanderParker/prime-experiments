"""Round 29 (mechanic), brief item (a): THE CRT SLOTS FOR FORMALIST.

Formalist's verdict 36 says realisability is cheap in the kernel once a witness
lives as a SLOT rather than as a phase vector.  This script turns each recorded
F_2 record of the project into exactly that, from the definition and importing
nothing from the tools that found it:

  * the explicit slot y on the FULL machine (0 <= y < P(machine)),
  * the span and the interior opening offset,
  * a BLOCKER CERTIFICATE - for every other slot of the span, the smallest gear
    that blocks it, so "every interior slot is blocked" becomes span-many single
    modular equalities instead of one existential,
  * the residue vector of y modulo every gear (the CRT coordinates), so the
    whole check is finite arithmetic on numerals,
  * the two neighbours OUTSIDE the window (the previous opening below y and the
    next opening above y + span), which pin the window as a maximal adjacent
    triple rather than merely as three openings.

Slot convention: slot k is the pair (6k-1, 6k+1); gear q blocks slot k iff
k = +-6^{-1} (mod q).  Openings of machine y = slots blocked by no gear
5 <= q <= y.

usage: uv run python research/crt_slots_r29.py
"""
from math import prod


def gears(y):
    return [p for p in range(5, y + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def teeth(y):
    return {q: (pow(6, -1, q), (-pow(6, -1, q)) % q) for q in gears(y)}


def is_open(k, T):
    return all(k % q not in t for q, t in T.items())


def blocker(k, T):
    for q, t in T.items():
        if k % q in t:
            return q, k % q
    return None


def report(y, k, offs, label):
    T = teeth(y)
    G = list(T)
    P = prod(G)
    assert 0 <= k < P, (label, "slot outside the period", k, P)
    span = offs[-1]
    oset = set(offs)
    # (i) every offset in offs is an opening; every other slot of the span is
    #     blocked, and we name the blocking gear.
    cert = []
    for t in range(span + 1):
        op = is_open(k + t, T)
        assert op == (t in oset), (label, "membership mismatch at offset", t)
        if not op:
            cert.append((t,) + blocker(k + t, T))
    # (ii) the two neighbours outside the window
    lo = k - 1
    while not is_open(lo, T):
        lo -= 1
    hi = k + span + 1
    while not is_open(hi, T):
        hi += 1
    gaps = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
    assert sum(gaps) == span
    print(f"\n=== {label} : machine {y}, gears {G[0]}..{G[-1]} "
          f"({len(G)} gears), P = {P:,}")
    print(f"  slot y            = {k}")
    print(f"  opening offsets   = {offs}   gap word {gaps}   span {span}")
    print(f"  interior openings = {offs[1:-1]}")
    print(f"  blocked interior  = {len(cert)} slots, all certified by a gear")
    print(f"  neighbour BELOW   = {lo}  (gap {k - lo} down to it)")
    print(f"  neighbour ABOVE   = {hi}  (gap {hi - (k + span)} up to it)")
    print(f"  five consecutive openings: {[lo] + [k + o for o in offs] + [hi]}")
    print(f"  full gap word incl. flanks: "
          f"{[k - lo] + gaps + [hi - (k + span)]}")
    print(f"  residue vector y mod q: "
          f"{ {q: k % q for q in G} }")
    print(f"  teeth  q: (+u, -u) : { {q: T[q] for q in G} }")
    print("  BLOCKER CERTIFICATE (offset -> gear, residue):")
    line = "   "
    for t, q, r in cert:
        s = f" {t}:{q}"
        if len(line) + len(s) > 76:
            print(line)
            line = "   "
        line += s
    if line.strip():
        print(line)
    return dict(y=y, k=k, offs=offs, span=span, lo=lo, hi=hi, P=P,
                below=k - lo, above=hi - (k + span))


W = [
    (41, 21157523372970, [0, 28, 103], "F_2(41) = 103"),
    (53, 327666424664536738, [0, 77, 159], "F_2(53) = 159"),
    (59, 307199471342884027665, [0, 100, 173], "F_2(59) = 173  (witness A)"),
    (59, 13260587016151412007, [0, 73, 173], "F_2(59) = 173  (witness B)"),
]

if __name__ == "__main__":
    print("ROUND-29 CRT SLOTS - every F_2 record as an explicit slot of its "
          "own machine")
    R = [report(*w) for w in W]
    assert [r["span"] for r in R] == [103, 159, 173, 173]
    # the two machine-59 witnesses: mirror pair of the machine-59 period?
    P59 = R[2]["P"]
    a, b = R[2]["k"], R[3]["k"]
    print(f"\n=== MIRROR TEST at machine 59 (P = {P59:,})")
    print(f"  y_A + y_B                 = {a + b}")
    print(f"  P(59) - 173               = {P59 - 173}")
    print(f"  mirror pair (y_A + y_B + 173 = P)?  "
          f"{'YES' if a + b + 173 == P59 else 'NO'}")
    print(f"  gap words {[100, 73]} and {[73, 100]} are exact reverses: YES")
    print(f"  flank gaps A (below, above) = "
          f"({R[2]['below']}, {R[2]['above']});  "
          f"B = ({R[3]['below']}, {R[3]['above']})")
    print(f"  flanks are a mirror pair?  "
          f"{'YES' if (R[2]['below'], R[2]['above']) == (R[3]['above'], R[3]['below']) else 'NO'}")
    print("\nALL ASSERTIONS PASSED")
