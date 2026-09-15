"""Single flips on whole field sets (owner, 2026-09-15): for each set of gears defined by a
field property, one flip whose mirror is the product of ALL the gears in the set (the set as
one mirror, not its gears one at a time), and the same for combinations (unions and
intersections of two sets, and each set joined with the base).  Methodically: origin, set,
mirror, direction, period; the landing; in the window or not; a twin, or the distance in
slots to the nearest twin (capped).

Field sets, per machine q (gears from 5 unless stated):
  base        the spiral base (lowest gears with product at most q/2), 2 and 3 included
  sqout       gears above the base at most sqrt q          sqin      gears above sqrt q
  coltwin     gears whose own column (6g-1, 6g+1) is a twin pair
  colopen     gears whose column is open to every base gear (from 5)
  col5, col7  gears whose column is painted by 5 (by 7)
  twinL/R     left / right members of twin gear pairs      solo      gears in no twin pair
  m1, m5      gears = 1 mod 6 / = 5 mod 6
  divq-1, divq+1   gears dividing q - 1 / q + 1            top3      the three top gears
Origins: home (-1, 1) and the top twin gear pair.  Directions up and down.  Periods 1 and
'enter' (the first period putting the landing above q).  A mirror above q^2 overshoots the
window at one period; recorded as such.
usage: uv run python research/stack/r8/field_set_flips.py 2000
"""
import sys, itertools, math
import numpy as np
from sympy import primerange, isprime
from pathlib import Path

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1)); qs = qs[:40] + qs[40::4]
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    def twin_dist(L, q):
        # distance in slots to the nearest twin inside the window, capped at 60; None if outside
        if not (q < L <= q * q - 2): return None
        for d in range(0, 61):
            for c in (L - 6 * d, L + 6 * d):
                if q < c <= q * q - 2 and sv[c] and sv[c + 2]: return d
        return 61
    names = ['base', 'sqout', 'sqin', 'coltwin', 'colopen', 'col5', 'col7', 'twinL', 'twinR', 'solo', 'm1', 'm5', 'divq-1', 'divq+1', 'top3']
    combos = [(n,) for n in names] + [(a, b, 'U') for a, b in itertools.combinations(names[1:], 2)] + [(a, b, 'I') for a, b in itertools.combinations(names[1:], 2)] + [('base', b, 'U') for b in names[1:]]
    stats = {}
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; B = 1
        for p in ps:
            if B * p <= q // 2: B *= p; base.append(p)
            else: break
        bg = [x for x in base if x >= 5]; g5 = [x for x in ps if x >= 5]; r = int(q ** 0.5)
        sets = {'base': set(base), 'sqout': {x for x in g5 if x not in base and x <= r}, 'sqin': {x for x in g5 if x > r},
                'coltwin': {x for x in g5 if sv[6 * x - 1] and sv[6 * x + 1]},
                'colopen': {x for x in g5 if all((6 * x - 1) % b and (6 * x + 1) % b for b in bg)},
                'col5': {x for x in g5 if 5 in bg and ((6 * x - 1) % 5 == 0 or (6 * x + 1) % 5 == 0)},
                'col7': {x for x in g5 if 7 in bg and ((6 * x - 1) % 7 == 0 or (6 * x + 1) % 7 == 0)},
                'twinL': {x for x in g5 if sv[x + 2]}, 'twinR': {x for x in g5 if sv[x - 2]}, 'solo': {x for x in g5 if not (sv[x + 2] or sv[x - 2])},
                'm1': {x for x in g5 if x % 6 == 1}, 'm5': {x for x in g5 if x % 6 == 5},
                'divq-1': {x for x in g5 if (q - 1) % x == 0}, 'divq+1': {x for x in g5 if (q + 1) % x == 0}, 'top3': set(ps[-3:])}
        tg = [x for x in ps if x + 2 in ps]; origins = {'home': -1, 'toptwin': tg[-1] if tg else None}
        for combo in combos:
            if len(combo) == 1: S = sets[combo[0]]
            elif combo[2] == 'U': S = sets[combo[0]] | sets[combo[1]]
            else: S = sets[combo[0]] & sets[combo[1]]
            if not S: continue
            M = 1
            for g in S:
                M *= g
                if M > 4 * q * q: break
            for oname, O in origins.items():
                if O is None: continue
                for d in (1, -1):
                    for per in ('1', 'enter'):
                        if M > 4 * q * q: L = None
                        else:
                            if per == '1': k = 1
                            else:
                                k = 1
                                while O + 2 * k * M * d <= q and k < 10 ** 6: k += 1
                                if d < 0: k = 1
                            L = O + 2 * k * M * d
                        key = (combo, oname, d, per); st = stats.setdefault(key, dict(n=0, over=0, inwin=0, twin=0, dsum=0, dn=0, dist=[]))
                        st['n'] += 1
                        if L is None or L > q * q - 2 or L < 0: st['over'] += 1; continue
                        td = twin_dist(L, q)
                        if td is None: st['over'] += 1; continue
                        st['inwin'] += 1
                        if td == 0: st['twin'] += 1
                        st['dsum'] += math.log2(1 + td); st['dn'] += 1
    rows = []
    for key, st in stats.items():
        combo, o, d, per = key
        name = combo[0] if len(combo) == 1 else f"{combo[0]} {'U' if combo[2] == 'U' else 'I'} {combo[1]}"
        rows.append((st['twin'], st['inwin'], st['n'], round(st['dsum'] / st['dn'], 3) if st['dn'] else None, name, o, 'up' if d > 0 else 'down', per))
    rows.sort(key=lambda r: (-r[0], -(r[1])))
    out = [__doc__.strip(), "", f"machines {len(qs)} (11 to {Q}); rules {len(rows)}", "",
           f"{'twin':>5} {'inwin':>5} {'n':>4} {'mean log2(1+d)':>15}  set / combination, origin, direction, period   (top 40 by twins)"]
    for t, w, n, md, name, o, d, per in rows[:40]:
        out.append(f"{t:>5} {w:>5} {n:>4} {str(md):>15}  {name:<24} {o:<8} {d:<5} {per}")
    out.append(""); out.append("single sets, all variants:")
    for t, w, n, md, name, o, d, per in sorted([r for r in rows if ' ' not in r[4]], key=lambda r: (r[4], r[5], r[6], r[7])):
        out.append(f"{t:>5} {w:>5} {n:>4} {str(md):>15}  {name:<24} {o:<8} {d:<5} {per}")
    Path("research/stack/r8/results_field_set_flips.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:48]))

if __name__ == "__main__":
    main()
