"""Harvester round 20: WHY IS 13 EXTREMAL - the mechanism, against Ziller-Morack's
FULL published table (arXiv:1706.03668 Table 1, h_2 exact for all p_n <= 73).

Round 17 explained the dip locally (twin step x1.42 bound growth + clean profile
extension x2.27). This script interrogates the mechanism with the 12 extra ZM data
points and the merge-law round's exact argmax trajectories at 19 and 23, as EXACT
EVENTS (no fits):

  A. Slack quantisation: h_2 = 0 mod 6 and the bound p^2-p has fixed residue mod 6
     (0 if p = 1 mod 6, 2 if p = 5 mod 6), so the conjecture's slack is quantised.
     EVENT: the minimal admissible slack is attained at p = 5 (slack 2) and p = 13
     (slack 6) and NOWHERE else through 73. "3.8% margin" is really "one quantum".
  B. The step law: relative margin falls at ALL six twin steps (landing prime >= 13),
     rises at ALL five gap-6 steps, and is genuinely mixed at gap-4 steps.
     Absolute slack falls ONLY at twin steps (13, 31, 61).
  C. The jump ratio r = Delta(maxF)/q' (halved units): 11->13 is the unique step
     with r > 2.6 in all 18 steps of the ZM table (3.231 vs runner-up 2.553).
  D. Persistence events: winners at 13 are clean extensions of winners at 11
     (the LAST persistence step); winners at 17 restrict to non-winners at 13.
     The 19-argmax (merge-law round) ranks ~35,849th at 17 - verified here from a
     full exhaustive 17-scan.
  E. Per-difference single-step increments of 3.23-4.43 q' EXIST in the family
     (d = 688 at 11->13; the 19- and 23-argmax trajectories), vs the round-14
     budget audit's structured-d worst 1.846 and twins' own worst 2.432: any
     uniform increment budget alpha <= 3 is FALSE over the full family.
  F. Twin/family-max ratio ladder through y = 43 using ZM's h_2 as the external
     denominator (validation for the twin-percentile statement).

Machine note: single-threaded numpy; heaviest piece is the exhaustive 17-family
scan (127,627 differences x P = 255,255) plus one P = 111.5M single-class sieve.
"""
import numpy as np
from math import prod, gcd

# ---------------------------------------------------------------- shared sieve
def F_of(gears, e, P, buf=None):
    """max cyclic gap of {n mod P : n != 0 and n != -e mod q for all gears q}"""
    a = np.ones(P, bool) if buf is None else buf
    if buf is not None:
        a[:] = True
    for q in gears:
        a[0::q] = False
        a[(-e) % q::q] = False
    idx = np.flatnonzero(a)
    if idx.size < 2:
        return 0
    g = np.diff(np.append(idx, idx[0] + P))
    return int(g.max())

def delta_profile(e, gears):
    return tuple(min(e % q, q - e % q) for q in gears)

# ------------------------------------------------- A. ZM table + quantisation
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
          67, 71, 73]
H2 =     [2, 6, 18, 30, 66, 150, 192, 258, 366, 450, 570, 708, 894, 1044, 1284,
          1422, 1656, 1902, 2190, 2460, 2622]   # ZM arXiv:1706.03668 Table 1

print("=" * 78)
print("A. SLACK QUANTISATION (bound B = p^2 - p; slack = B - h_2)")
print("=" * 78)
print(f"{'p':>4} {'h_2':>6} {'B':>6} {'slack':>6} {'min adm.':>8} {'margin %':>9}"
      f"  event")
min_slack_events = []
for p, h in zip(PRIMES, H2):
    if p < 5:
        continue
    B = p * p - p
    slack = B - h
    assert h % 6 == 0, (p, h)
    min_adm = 6 if p % 6 == 1 else 2          # B mod 6 = 0 or 2 resp.
    assert B % 6 == (0 if p % 6 == 1 else 2), p
    assert slack % 6 == min_adm % 6 or (min_adm == 6 and slack % 6 == 0), p
    ev = "<== MINIMAL ADMISSIBLE SLACK" if slack == min_adm else ""
    if slack == min_adm:
        min_slack_events.append(p)
    print(f"{p:>4} {h:>6} {B:>6} {slack:>6} {min_adm:>8} {100*(1-h/B):>8.1f}  {ev}")
assert min_slack_events == [5, 13], min_slack_events
print(f"\nEVENT: minimal admissible slack attained exactly at p = 5 and p = 13 "
      f"(both twin-step landings); h_2(13) = 156 - 6, i.e. in ZM condensed units "
      f"omega_2 = 24 = cap 25 minus ONE. Never again through 73.")

# ------------------------------------------------- B. step law (exact events)
print()
print("=" * 78)
print("B. THE STEP LAW (18 steps of the ZM table)")
print("=" * 78)
print(f"{'step':>9} {'gap':>4} {'dh/q_':>6} {'margin %':>17} {'d margin':>9} "
      f"{'d slack':>8}")
sign_by_gap = {2: [], 4: [], 6: []}
slack_down_steps = []
r_list = []
for i in range(3, len(PRIMES) - 1):          # steps landing at p >= 11's successor
    p0, p1 = PRIMES[i], PRIMES[i + 1]
    h0, h1 = H2[i], H2[i + 1]
    B0, B1 = p0 * p0 - p0, p1 * p1 - p1
    g = p1 - p0
    m0, m1 = 1 - h0 / B0, 1 - h1 / B1
    r = (h1 - h0) / 2 / p1                    # halved-frame jump per q'
    r_list.append((p1, g, r))
    dm = m1 - m0
    ds = (B1 - h1) - (B0 - h0)
    if ds < 0:
        slack_down_steps.append(p1)
    if p1 >= 13:
        sign_by_gap[g].append(1 if dm > 0 else (-1 if dm < 0 else 0))
    print(f"{p0:>4}->{p1:<4} {g:>4} {r:>6.3f} {100*m0:>7.2f} -> {100*m1:<7.2f}"
          f" {100*dm:>+8.2f} {ds:>+8}")
assert all(s == -1 for s in sign_by_gap[2]) and len(sign_by_gap[2]) == 6
assert all(s == +1 for s in sign_by_gap[6]) and len(sign_by_gap[6]) == 5
assert 1 in sign_by_gap[4] and -1 in sign_by_gap[4]
assert slack_down_steps == [13, 31, 61], slack_down_steps
print(f"\nEVENTS: margin falls at ALL 6 twin steps (>=13), rises at ALL 5 gap-6 "
      f"steps, gap-4 mixed ({sign_by_gap[4]}).")
print(f"Absolute slack falls ONLY at twin steps: {slack_down_steps} (3 of 6).")

# jump-ratio uniqueness
rs = sorted(r_list, key=lambda t: -t[2])
print(f"\nC. JUMP RATIOS r = Delta(maxF)/q' (halved): top three "
      f"{[(f'->{p}', round(r,3)) for p, _, r in rs[:3]]}")
assert rs[0][0] == 13 and abs(rs[0][2] - 42 / 13) < 1e-12
assert rs[1][2] < 2.6, rs[1]
print(f"EVENT: 11->13 is the UNIQUE step with r > 2.6 (3.231; runner-up "
      f"{rs[1][2]:.3f} at ->{rs[1][0]}).")
print("Mechanism of the sign law: margin is stable when h grows like the bound, "
      "d(B)/B ~ 2g/p vs d(h)/h ~ 2r/p (h ~ p^2/2), so the crossover gap is g ~ r "
      f"~ {np.mean([r for _,_,r in r_list]):.2f} - hence gap 2 always down, gap 6 "
      "always up, gap 4 on the knife edge. The dip needs r >> g: only 11->13.")

# ------------------------------------------- D. persistence + family anatomy
print()
print("=" * 78)
print("D. PERSISTENCE EVENTS (exhaustive family scans, y = 5..17)")
print("=" * 78)
SETS = {5: [3, 5], 7: [3, 5, 7], 11: [3, 5, 7, 11], 13: [3, 5, 7, 11, 13],
        17: [3, 5, 7, 11, 13, 17]}
fam = {}
for y, gears in SETS.items():
    P = prod(gears)
    buf = np.empty(P, bool)
    F = np.zeros(P // 2 + 1, np.int32)
    for e in range(1, P // 2 + 1):
        F[e] = F_of(gears, e, P, buf)
    fam[y] = (gears, P, F)
    print(f"  y={y:>2}: maxF={F[1:].max():>3} (h_2={2*int(F[1:].max())}), "
          f"twin F_1={F[1]:>3}, winners={int((F[1:]==F[1:].max()).sum())}")
assert 2 * fam[13][2][1:].max() == 150 and 2 * fam[17][2][1:].max() == 192
assert fam[13][2][1] == 33 and fam[17][2][1] == 54   # twin ladder values

def winners(y):
    gears, P, F = fam[y]
    mx = F[1:].max()
    return [int(e) for e in np.flatnonzero(F == mx) if e >= 1], int(mx)

def restrict(e, Psmall):
    r = e % Psmall
    return min(r, Psmall - r)

for ys, yb in ((5, 7), (7, 11), (11, 13), (13, 17)):
    wb, mb = winners(yb)
    ws, ms = winners(ys)
    Ps = fam[ys][1]
    restr = [restrict(e, Ps) for e in wb]
    vals = [int(fam[ys][2][r]) if r >= 1 else 0 for r in restr]
    clean = all(v == ms for v in vals)
    print(f"  {ys:>2}->{yb:<2}: winners at {yb} restrict to F_{ys} values "
          f"min={min(vals)} max={max(vals)} (family max {ms}) -> "
          f"{'CLEAN EXTENSION (all winners extend winners)' if clean else 'NOT winners below'}")
    if (ys, yb) == (11, 13):
        assert clean, vals
    if (ys, yb) == (13, 17):
        assert not clean and max(vals) < ms, vals

# winners at 13 anatomy: same e gains 3.231 q' in one step
w13, _ = winners(13)
assert len(w13) == 16
for e in w13:
    r = restrict(e, fam[11][1])
    assert fam[11][2][r] == 33 and fam[13][2][e] == 75
print(f"  All 16 winners at 13 (incl. e=344) have F_11 = 33, F_13 = 75: the SAME "
      f"fixed difference gains 42 = 3.231 q' in the single step 11->13.")

# clean extensions of the 13-winner profile at 17: how good are they?
gears17, P17, F17 = fam[17]
best_ext, best_ext_e = 0, None
for e in w13:
    # lift e to all 17-classes with the same residues mod 15015: e + k*15015
    for k in range(17):
        cand = restrict(e + k * 15015, P17)
        if cand < 1:
            continue
        f = int(F17[cand])
        if f > best_ext:
            best_ext, best_ext_e = f, cand
mx17 = int(F17[1:].max())
print(f"  Best 17-extension of any 13-winner: F = {best_ext} (e = {best_ext_e}) "
      f"vs true max {mx17} -> clean extension DIES at 13->17 "
      f"(deficit {mx17 - best_ext}).")
assert best_ext < mx17

# rank of the 19-argmax at 17 (merge-law round claim: 35,848 strictly above)
e19arg_restr = restrict(1_532_627, P17)
v = int(F17[e19arg_restr])
rank_above = int((F17[1:] > v).sum())
twin_rank_above = int((F17[1:] > F17[1]).sum())
n17 = P17 // 2
print(f"  19-argmax e=1,532,627 restricts to e={e19arg_restr} at 17: F = {v} "
      f"(the twin's own value), {rank_above} classes strictly above "
      f"(of {n17}).")
print(f"  Twin at 17: F = {int(F17[1])}, {twin_rank_above} classes strictly above "
      f"({100*twin_rank_above/n17:.1f}% of family, incl. non-coprime).")
assert v == 54, v

# --------------------------------- E. per-difference single-step increments
print()
print("=" * 78)
print("E. PER-DIFFERENCE SINGLE-STEP INCREMENTS (the budget events)")
print("=" * 78)
# e = 344 verified above: 33 -> 75, 3.231 q'.
# 19-argmax trajectory: F_17 = 54 (verified above) -> F_19 = 129.
G19 = [3, 5, 7, 11, 13, 17, 19]
P19 = prod(G19)
f19 = F_of(G19, 1_532_627, P19)
print(f"  e=1,532,627: F_17 = 54 -> F_19 = {f19}  "
      f"(increment {f19 - 54} = {(f19-54)/19:.3f} q')")
assert f19 == 129
# 23-argmax trajectory: F_19 = 81 -> F_23 = 183 (merge-law round values).
e23 = 107_207_699
f23_at19 = F_of(G19, e23 % P19, P19)
G23 = G19 + [23]
P23 = prod(G23)
f23 = F_of(G23, e23, P23)
print(f"  e=107,207,699: F_19 = {f23_at19} -> F_23 = {f23}  "
      f"(increment {f23 - f23_at19} = {(f23-f23_at19)/23:.3f} q')")
assert f23_at19 == 81 and f23 == 183
print("""
EVENT: fixed differences with single-step increments 3.231 q' (11->13),
3.947 q' (17->19), 4.435 q' (19->23) EXIST - and the two large ones are the
family argmaxes found by the merge-law round. The round-14 budget audit's
structured-d worst was 1.846 q', twins' own worst 2.432 q' (31->37).
CONSEQUENCE: no uniform increment budget alpha <= 3 (in q' units) holds over
the full even-difference family; the tolerance-route constant is twin-lane
(more generally, structured-d) specific, and the sequence 3.23, 3.95, 4.43 is
non-decreasing in the three known family-argmax steps.""")

# --------------------------------- F. twin / family-max ratio ladder (ZM ext.)
print("=" * 78)
print("F. TWIN / FAMILY-MAX RATIO LADDER (external denominator: ZM h_2)")
print("=" * 78)
TWIN_F = {5: int(fam[5][2][1]), 7: int(fam[7][2][1]), 11: 21, 13: 33, 17: 54,
          19: 75, 23: 102, 29: 129, 31: 174, 37: 264, 41: 273, 43: 309}
assert fam[11][2][1] == 21
zm = dict(zip(PRIMES, H2))
print(f"{'y':>4} {'twin F':>7} {'maxF=h_2/2':>11} {'ratio':>7} {'extreme/twin':>13}")
ratios = {}
for y, tf in TWIN_F.items():
    mx = zm[y] // 2
    ratios[y] = tf / mx
    print(f"{y:>4} {tf:>7} {mx:>11} {tf/mx:>7.3f} {mx/tf:>13.2f}")
lo, hi = min(ratios, key=ratios.get), max(ratios, key=ratios.get)
print(f"\nTwin share of the family max over 12 machines: "
      f"{ratios[lo]:.3f} (y={lo}) .. {ratios[hi]:.3f} (y={hi}); "
      f"twin attains the max only at y = 5, 7; from y = 11 on the extreme is "
      f"1.34x - 2.27x the twin value, median "
      f"{np.median([1/r for y, r in ratios.items() if y >= 11]):.2f}x.")
assert ratios[13] == 33 / 75 and abs(ratios[37] - 264/354) < 1e-12
assert all(r < 1 for y, r in ratios.items() if y >= 11)
assert ratios[5] == 1.0 or ratios[5] < 1  # printed; y=5 checked below
print(f"(y=5 twin F = {TWIN_F[5]} vs max {zm[5]//2}; y=7 twin F = {TWIN_F[7]} "
      f"vs max {zm[7]//2})")

print("\nALL ASSERTIONS PASSED.")
