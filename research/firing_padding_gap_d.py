"""Harvester round 14: the firing law and PADDING for general Polignac gap d.

Halved coordinates: position n, pair (2n+1, 2n+1+2e), gear q blocks n = 0, -e mod q.
Gear q' has TWO teeth, A: n = 0 and B: n = -e (mod q'), separated by e.

CORRECTED FIRING LAW (general d) - the d-restatement of lateral's merge law:
  a killed run is a run of consecutive M-survivors all killed by q'; the LETTER
  between adjacent members is the single M-gap g between them, and
      g = 0 (mod q')  -> PADDED link  (same tooth)
      g = +-e (mod q') -> LITERAL link (opposite teeth)
      anything else    -> ILLEGAL (cannot be two kills)
  non-zero letters ALTERNATE in sign (forced: +e is B->A, -e is A->B);
  zero letters are insertable freely. Then
      F(M+q') = max over LEGAL runs of span = o[i+k] - o[i-1]   (k >= 0)
  computed from the OLD machine alone; k=0 gives F(M), k=1 gives F2(M).

SELF-CORRECTION carried out here: my round-13 "tooth alternation fails for
3 | e" counted SAME-TOOTH adjacency as a violation. Under the corrected law a
same-tooth adjacency is a legal PADDED link. The observation was real, the
label was wrong - and the real content is that padding is CHEAPER when 3 | e.

CHECKS
  A  soundness  : every realized run is legal            (violations counted)
  B  firing     : every legal run is realized in some phase, and conversely
                  (both directions, over all q' CRT phases)
  C  identity   : old-machine prediction = exact F(M+q')
  D  padding    : cost of a padded link per d class, and the winner's type
"""
import numpy as np

def survivors(gears, e, P):
    n = np.arange(P)
    a = np.ones(P, bool)
    for q in gears:
        a &= (n % q != 0) & (n % q != (-e) % q)
    return np.flatnonzero(a)

def letter_type(g, e, q1):
    r = g % q1
    if r == 0:
        return 'Z'
    if r == e % q1:
        return 'P'
    if r == (-e) % q1:
        return 'N'
    return None

def analyse(e, gears, q1, KMAX=7):
    P = 1
    for q in gears:
        P *= q
    o = survivors(gears, e, P)
    m = len(o)
    gaps = np.diff(np.append(o, o[0] + P))
    F, F2 = int(gaps.max()), int((gaps + np.roll(gaps, -1)).max())
    types = [letter_type(int(g), e, q1) for g in gaps]
    # legal runs from the OLD machine alone
    legal = {0: np.ones(m, bool)}
    for k in range(1, KMAX + 1):
        ok = np.zeros(m, bool)
        for i in range(m):
            good, last = True, None
            for z in range(k - 1):
                t = types[(i + z) % m]
                if t is None:
                    good = False
                    break
                if t != 'Z':
                    if last is not None and t == last:
                        good = False
                        break
                    last = t
            ok[i] = good
        legal[k] = ok
    # exact new F
    Pn = P * q1
    on = survivors(list(gears) + [q1], e, Pn)
    Fnew = int(np.diff(np.append(on, on[0] + Pn)).max())
    # spans and prediction
    cum = np.concatenate([[0], np.cumsum(gaps)])
    def span(i, k):
        s = 0
        for z in range(k + 1):
            s += int(gaps[(i - 1 + z) % m])
        return s
    pred, pred_kind, best = 0, None, None
    for k in range(0, KMAX + 1):
        idx = np.flatnonzero(legal[k]) if k > 0 else np.arange(m)
        for i in idx:
            sp = span(int(i), k)
            if sp > pred:
                pred = sp
                has_z = any(types[(int(i) + z) % m] == 'Z' for z in range(max(0, k - 1)))
                pred_kind = ('padded' if has_z else 'literal') if k >= 2 else f'k={k}'
                best = (int(i), k)
    # realized runs over all phases, both directions vs legal
    realized = {k: np.zeros(m, bool) for k in range(0, KMAX + 1)}
    sound_viol = 0
    # absolute positions over two periods: rolling would corrupt the wrap element
    # (gcd(P, q') = 1, so o[0] and o[0] + P have DIFFERENT kill status)
    o2 = np.concatenate([o, o + P])
    for c in range(q1):
        kill2 = (((o2 + c) % q1) == 0) | (((o2 + c) % q1) == (-e) % q1)
        for k in range(1, KMAX + 1):
            w = np.ones(m, bool)
            for z in range(k):
                w &= kill2[z:z + m]
            realized[k] |= w
            if k == 2:
                for i in np.flatnonzero(w):
                    if types[int(i)] is None:
                        sound_viol += 1
    fire_miss = sum(int((legal[k] & ~realized[k]).sum()) for k in range(1, KMAX + 1))
    over = sum(int((realized[k] & ~legal[k]).sum()) for k in range(1, KMAX + 1))
    # padding cost: which gaps can serve as a padded link
    padgaps = sorted({int(g) for g in gaps if int(g) % q1 == 0})
    return dict(F=F, F2=F2, Fnew=Fnew, pred=pred, kind=pred_kind, k=best[1],
                sound=sound_viol, fire_miss=fire_miss, over=over,
                padgaps=padgaps[:3], has_pad=len(padgaps) > 0,
                minpad=(padgaps[0] if padgaps else None), m=m)

print("d   gears                q'   F   F2  Fnew pred  ok  winner        "
      "sound fire over  min padded gap")
rows = [(2,[3,5,7,11],13),(2,[3,5,7,11,13],17),(2,[3,5,7,11,13,17],19),
        (4,[3,5,7,11],13),(4,[3,5,7,11,13],17),(4,[3,5,7,11,13,17],19),
        (6,[3,5,7,11],13),(6,[3,5,7,11,13],17),(6,[3,5,7,11,13,17],19),
        (12,[3,5,7,11],13),(12,[3,5,7,11,13],17),(12,[3,5,7,11,13,17],19),
        (10,[3,5,7,11,13],17),(30,[3,5,7,11,13],17)]
for d, gears, q1 in rows:
    r = analyse(d // 2, gears, q1)
    ok = 'Y' if r['pred'] == r['Fnew'] else 'N'
    mp = r['minpad'] if r['minpad'] else '-'
    print(f"{d:>2}  {str(gears):<20}{q1:>3} {r['F']:>4} {r['F2']:>4} {r['Fnew']:>5} "
          f"{r['pred']:>4}  {ok}   {r['kind']:<12} {r['sound']:>4} {r['fire_miss']:>4} "
          f"{r['over']:>4}   {mp}  (3q'={3*q1}, q'={q1})")
