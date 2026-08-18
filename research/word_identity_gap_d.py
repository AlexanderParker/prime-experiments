"""Harvester round 13, chunk 2: does Constructor's WORD IDENTITY transfer to gap d?

  F(M+q') = max( F2(M), max over compatible qualifying words w of
                 [span(w) + FS_max(w; M)] )

Structural inputs per Constructor: (a) gcd(P_M, q') = 1 - the q' CRT copies
realize every residue shift, so every compatible word fires; (b) tooth
ALTERNATION inside a chain. Both look d-agnostic. Tested in halved coordinates
(gear q blocks n = 0, -e mod q; e = d/2) against EXACT F values.

Two frame subtleties, handled explicitly (they bit the first pass):
 * chains are cyclic - enumeration must start at an UNKILLED survivor, else
   wrap-around chains get split and tiers come out short;
 * a chain's letters are the distances between consecutive FRAME-ADMISSIBLE
   kills (admissible = surviving gear 3), not the raw residue distances:
   for twins at q'=31 the letters are 30, 63 in halved units = the corpus's
   10, 21 in slot units (x3), never {1, 30}.

Checks per (d, M, q'):
  W1  tier decomposition reproduces F(M+q') exactly (code sanity)
  W2  identity shape: F(M+q') = max(F2(M), max_{k>=2} tier_k)
  W3  tooth alternation inside every chain: zero violations
  W4  every realized letter is a sum of consecutive frame letters, and the
      frame letter pair is predicted from q', e and gear 3 alone
  W5  degenerate q' | e: gear q' has ONE tooth - what changes
"""
import numpy as np

def survivors(gears, e, P):
    n = np.arange(P)
    alive = np.ones(P, bool)
    for q in gears:
        alive &= (n % q != 0) & (n % q != (-e) % q)
    return np.flatnonzero(alive)

def frame_letters(e, q1):
    """Distances between consecutive frame-admissible (gear-3-surviving) kills."""
    n = np.arange(3 * q1 * 2)
    kill = (n % q1 == 0) | (n % q1 == (-e) % q1)
    adm = (n % 3 != 0) & (n % 3 != (-e) % 3)
    idx = np.flatnonzero(kill & adm)
    return sorted(set(np.diff(idx).tolist())), idx.size

def analyse(e, gears, q1):
    P = 1
    for q in gears:
        P *= q
    S = survivors(gears, e, P)
    m = len(S)
    gaps = np.diff(np.append(S, S[0] + P))
    F, F2 = int(gaps.max()), int((gaps + np.roll(gaps, -1)).max())
    # ALL q' CRT phases: gcd(P, q') = 1, so old survivor s is killed in copy c
    # iff (s + c) = 0 or -e mod q'. The max over the new period needs every phase.
    phases = range(q1)
    Pn = P * q1
    Sn = survivors(list(gears) + [q1], e, Pn)
    Fnew = int(np.diff(np.append(Sn, Sn[0] + Pn)).max())
    single = ((-e) % q1 == 0)
    # cyclic chain enumeration starting from an unkilled survivor
    tiers, letters, alt_viol, lits = {}, set(), 0, 0
    for c in phases:
      killed = (((S + c) % q1) == 0) | (((S + c) % q1) == (-e) % q1)
      if killed.all():
          continue
      start0 = int(np.flatnonzero(~killed)[0])
      i = 1
      while i <= m:
        j = (start0 + i) % m
        if killed[j]:
            run = []
            while killed[(start0 + i) % m] and len(run) <= m:
                run.append((start0 + i) % m)
                i += 1
            k = len(run)
            st = (run[0] - 1) % m
            span = int(sum(gaps[(st + z) % m] for z in range(k + 1)))
            tiers[k] = max(tiers.get(k, 0), span)
            teeth = [0 if (S[r] + c) % q1 == 0 else 1 for r in run]
            if any(teeth[z] == teeth[z + 1] for z in range(k - 1)):
                alt_viol += 1
            if k >= 2:
                w = [int((S[run[z + 1]] - S[run[z]]) % P) for z in range(k - 1)]
                letters |= set(w)
                if set(w) <= set(frame_letters(e, q1)[0]):
                    lits += 1
        i += 1
    hi = max([v for k, v in tiers.items() if k >= 2], default=0)
    fl = frame_letters(e, q1)[0]
    ok4 = all(any(L == s for s in _sums(fl, 40)) for L in letters) if letters else True
    return dict(F=F, F2=F2, Fnew=Fnew, tiers=dict(sorted(tiers.items())),
                W1=(max(F, max(tiers.values()) if tiers else F) == Fnew),
                W2=(max(F2, hi) == Fnew), W3=(alt_viol == 0), W4=ok4,
                letters=sorted(letters)[:4], frame=fl, single=single, lits=lits)

def _sums(fl, kmax):
    """all sums of up to kmax consecutive alternating frame letters"""
    if len(fl) == 1:
        return {fl[0] * j for j in range(1, kmax + 1)}
    a, b = fl[0], fl[1]
    out = set()
    for st in (a, b):
        tot, cur = 0, st
        for _ in range(kmax):
            tot += cur
            out.add(tot)
            cur = b if cur == a else a
    return out

print("d  gears                 q'   F   F2 Fnew  W1 W2 W3 W4  frame letters  tiers")
for d, gears, q1 in [(2, [3,5,7,11], 13), (2, [3,5,7,11,13], 17),
                     (2, [3,5,7,11,13,17], 19),
                     (4, [3,5,7,11], 13), (4, [3,5,7,11,13], 17),
                     (6, [3,5,7,11], 13), (6, [3,5,7,11,13], 17),
                     (6, [3,5,7,11,13,17], 19),
                     (10, [3,5,7,11], 13), (30, [3,5,7,11,13], 17),
                     (12, [3,5,7,11,13], 17)]:
    r = analyse(d // 2, gears, q1)
    y = lambda b: 'Y' if b else 'N'
    print(f"{d:>2}  {str(gears):<21}{q1:>3} {r['F']:>4} {r['F2']:>4} {r['Fnew']:>4}  "
          f"{y(r['W1'])}  {y(r['W2'])}  {y(r['W3'])}  {y(r['W4'])}  {str(r['frame']):<14} {r['tiers']}")

print("\nW5 degenerate q' | e (ONE tooth):")
for d, gears, q1 in [(26, [3,5,7,11], 13), (34, [3,5,7,11,13], 17)]:
    r = analyse(d // 2, gears, q1)
    print(f"  d={d} q'={q1}: single_tooth={r['single']} F={r['F']} F2={r['F2']} "
          f"Fnew={r['Fnew']} W1={r['W1']} W2={r['W2']} W3={r['W3']} "
          f"frame={r['frame']} tiers={r['tiers']}")
