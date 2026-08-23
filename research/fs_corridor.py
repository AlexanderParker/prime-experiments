"""Round 20 lateral: THE FLANK-SUM CORRIDOR LAW - (D) at the 4-point level.

A word occurrence with flanks is 4 anchor openings at 0, gL, gL+s, s+T
(s = span, T = gL+gR = the flank sum FS; shifting the frame, the points are
{r-gL, r, r+s, r+s+gR} for machine phase r). By the completeness lemma
(round 17) a 4-point shape can be blocked outright only by gears q <= 8,
i.e. 5 and 7: feasibility is decided ENTIRELY mod 35.

    feas(s, T)  <=>  exists r, gL mod 35 : all four points in E mod 35.

This script (1) builds the full 35 x 35 (s mod 35, T mod 35) feasibility
table - the general corridor law for flank sums; (2) tests, for every
word-step in the Mechanic's census, whether any part of the (D)-critical
interval T in (FSmax, need3] is corridor-forbidden - i.e. whether (D) is
ever corridor-FORCED rather than merely renewal-suppressed.
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

def exposed(m):
    out = set()
    for q in (5, 7):
        u = pow(6, -1, q)
        pass
    E = set()
    for r in range(35):
        ok = True
        for q in (5, 7):
            u = pow(6, -1, q)
            if r % q in (u % q, (-u) % q):
                ok = False
        if ok:
            E.add(r)
    return E

E = exposed(35)
assert len(E) == 15

def feas(s, T):
    for r in range(35):
        for gL in range(35):
            pts = (r - gL, r, r + s, r + s + T - gL)
            if all(p % 35 in E for p in pts):
                return True
    return False

print("=" * 76)
print("PART 1: the (s, T) mod-35 feasibility table for flank sums")
table = {}
for s in range(35):
    for T in range(35):
        table[(s, T)] = feas(s, T)
blocked = [(s, T) for (s, T), v in table.items() if not v]
print(f"  blocked (s mod 35, T mod 35) classes: {len(blocked)} of 1225")
if blocked:
    print("  " + str(blocked[:40]))
else:
    print("  NO class is blocked: with two free phases (machine phase r and")
    print("  flank split gL) the corridor NEVER forbids a flank-sum value.")
    print("  Corollary: (D) is NEVER corridor-forced at the 4-point level -")
    print("  every FS value above the requirement is arithmetically feasible")
    print("  and is excluded (when it is) only by counting/renewal. The same")
    print("  lesson as gap 24 (r18): selection plus rarity, not obstruction.")

print("=" * 76)
print("PART 2: refinement - fix the flank split (gL mod 35 given), only the")
print("machine phase free. Blocked (gL, s, gR) triples do exist:")
cnt = 0
tot = 0
for s in range(35):
    for gL in range(35):
        for gR in range(35):
            tot += 1
            ok = False
            for r in range(35):
                pts = (r - gL, r, r + s, r + s + gR)
                if all(p % 35 in E for p in pts):
                    ok = True
                    break
            if not ok:
                cnt += 1
print(f"  blocked (gL, s, gR) mod-35 triples: {cnt} of {tot} "
      f"({100*cnt/tot:.1f}%)")
print("  (so individual flank SPLITS are corridor-forbidden, but the flank")
print("   SUM always has an allowed split - the disjunction saves it.)")

print("=" * 76)
print("PART 3: word-steps - is any (D)-critical interval corridor-blocked?")
rows = list(csv.DictReader(open(os.path.join(DATA,
                                             "flank_envelope_words.csv"))))
anyblocked = False
for r in rows:
    s = int(r["span"]); fsmax = int(r["FSmax"]); need3 = int(r["need3"])
    bad = [T for T in range(fsmax + 1, need3 + 1) if not table[(s % 35, T % 35)]]
    if bad:
        anyblocked = True
        print(f"  y={r['y']} q'={r['qp']} word {r['word']}: corridor blocks "
              f"T in {bad}")
if not anyblocked:
    print(f"  none of the {len(rows)} word-steps has ANY corridor-blocked T")
    print("  in (FSmax, need3] - consistent with Part 1 (no class blocked).")
