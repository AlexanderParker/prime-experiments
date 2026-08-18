"""Round 15 lateral: (1) frame check; (2) the 37->41 double-padding question,
theory ahead of the census - both branches quantified.

PART 1 - FRAME. All lateral gap numbers are in SLOT units: slot k is the pair
(6k-1, 6k+1), openings are surviving slots, a gap is a difference of slot
indices. Conversions: member-space = 6 x slot, corpus halved coordinates
= 3 x slot. So lateral's "padded link of exactly q'" IS harvester's "3q'"
once the frame factor is applied.

PART 2 - two padded links at 37->41. Two mechanisms can forbid them:
  SPECTRUM (round 14): p padded links with j literal links between occupy
    j+2 consecutive gaps summing >= 2q' + j*L, so need F_{j+2}(M) >= 2q' + jL.
  CORRIDOR (rounds 8-9): every opening lies in the 15-residue exposed set E
    mod 35, so consecutive openings o, o+a*q', o+(a+b)*q' need all three
    residues in E - a pure mod-35 feasibility question, INDEPENDENT of the
    spectrum and unaffected by the fact that machine-37 F_j values are only
    prefix lower bounds.
"""
from math import prod
from split_gap_law import primes

E = sorted(k for k in range(35) if k % 5 not in (1, 4) and k % 7 not in (1, 6))
Eset = set(E)

def frame_check():
    print("=" * 74)
    print("PART 1: FRAME - lateral works in SLOT units")
    print(f"  slot k = pair (6k-1, 6k+1); gap = difference of slot indices.")
    print(f"  member-space gap = 6 x slot gap;  corpus halved gap = 3 x slot gap.")
    print(f"  => lateral 'padded link = exactly q'' == harvester '3q'' (halved).")
    print("  independent verification of the 3x factor against the corpus:")
    for y, slot, halved in [(43, 103, 309)]:
        print(f"    corpus F(2,{y}) = {halved}; 3 x {slot} = {3*slot} "
              f"{'OK' if 3*slot == halved else 'MISMATCH'}   "
              f"({slot} appears in mechanic's machine-37 F_j spectrum)")
    print("  measured padded-link values (slot units), all steps: exactly q' -")
    print("    23, 29, 31, 37 at steps 19->23, 23->29, 29->31, 31->37")
    print("    i.e. 69, 87, 93, 111 in halved units = 3q' exactly.")

def corridor_table():
    print("=" * 74)
    print("PART 2a: CORRIDOR feasibility of two ADJACENT padded links")
    print(f"  exposed set E mod 35 ({len(E)}): {E}")
    print("  adjacent padded links of sizes a*q', b*q' need residues")
    print("  r, r+a*q', r+(a+b)*q' all in E (mod 35).")
    print(f"  {'step':>9} {'q':>3} {'q mod 35':>9} {'(a,b)':>7} {'feasible r':>28}")
    for y, qp in [(19, 23), (23, 29), (29, 31), (31, 37), (37, 41), (41, 43)]:
        g = qp % 35
        for (a, b) in [(1, 1), (1, 2), (2, 1)]:
            good = [r for r in E if (r + a * g) % 35 in Eset
                    and (r + (a + b) * g) % 35 in Eset]
            tag = (", ".join(map(str, good[:6])) + ("..." if len(good) > 6 else "")
                   ) if good else "NONE - corridor-IMPOSSIBLE"
            print(f"  {y:>4}->{qp:<3} {qp:>3} {g:>9} {str((a,b)):>7} {tag:>28}")

def spectrum_table():
    print("=" * 74)
    print("PART 2b: SPECTRUM feasibility at 37->41 (F_j are PREFIX LOWER BOUNDS)")
    qp = 41
    u = pow(6, -1, qp); s = (2 * u) % qp; L = min(s, qp - s)
    F = [88, 90, 95, 103, 112, 115]         # machine 37, 16.2% prefix
    print(f"  q'={qp}, L={L}; machine-37 F_j >= {F}")
    for j in range(0, 5):
        need = 2 * qp + j * L
        have = F[j + 1] if j + 1 < len(F) else None
        if have is None:
            continue
        verdict = ("allowed" if have >= need else
                   f"excluded by {need-have} - BUT F_j is a lower bound, so "
                   f"NOT airtight")
        print(f"    j={j}: need F_{j+2} >= {need:>3}, measured F_{j+2} >= {have:>3}"
              f"   {verdict}")
    print(f"  => the spectrum alone cannot settle 37->41: only j=0 is cleanly")
    print(f"     allowed, and j>=1 exclusions rest on lower bounds.")

def branches():
    print("=" * 74)
    print("PART 2c: BOTH BRANCHES, quantified")
    qp, s = 41, (2 * pow(6, -1, 41)) % 41
    F37 = 88
    pmax_count = (F37 + 5 * qp / 6) / qp
    print(f"  q'={qp}, s={s}, F(37)>={F37}")
    print(f"  constructor's count cap: p <= (F + 5q'/6)/q' = {pmax_count:.2f}"
          f"  -> p <= 2 (p=3 arithmetically impossible)")
    print()
    print("  BRANCH A - a double-padded run IS found:")
    for p in (1, 2):
        ceil = (4 + p) * qp + 2 * s
        print(f"    p={p}: span <= (4+p)q' + 2s = {ceil} = {ceil/qp:.2f} q'")
    print(f"    => the ceiling does NOT collapse; it degrades by exactly one q'")
    print(f"       (5.68 q' -> 6.68 q'). Structure is forced: with F_2(37) < 123")
    print(f"       both padded links must be exactly q', and the run is")
    print(f"       [literal chain] --q'-- [kill] --q'-- [literal chain].")
    print()
    print("  BRANCH B - none found (padded links repel):")
    g = qp % 35
    good = [r for r in E if (r + g) % 35 in Eset and (r + 2 * g) % 35 in Eset]
    print(f"    MECHANISM IDENTIFIED: q' mod 35 = {g}; three openings at")
    print(f"    r, r+{g}, r+{2*g} (mod 35) all in E has {len(good)} solutions"
          f" -> {'possible' if good else 'IMPOSSIBLE'}.")
    print(f"    So adjacent padded links are forbidden by the (5,7) corridor")
    print(f"    ALONE - no spectrum input, unaffected by prefix lower bounds.")

if __name__ == "__main__":
    frame_check()
    corridor_table()
    spectrum_table()
    branches()
