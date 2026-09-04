"""Prover C r33 -- score the pre-registration on the family rows (chain_teeth_r33_fam_m<y>.json)."""
import json
import sys
from collections import Counter

sys.path.insert(0, 'research/proof')
from chain_teeth_r33 import gears_of, next_prime, real_tooth, letter_a, is_T, sep_of, margins  # noqa: E402


def slice_report(name, sel, q1):
    v = [r for r in sel if r['viol']]
    marg = sorted(r['F'] + q1 - r['chain'] for r in sel if r['chain'] is not None)
    s = f"  {name}: rows {len(sel)}, chain violators {len(v)}"
    if marg:
        s += f", chain margin min {marg[0]} / 1st pct {marg[len(marg) // 100]} / median {marg[len(marg) // 2]}"
    if v:
        s += (f"; max excess {max(max(r['viol'].values()) for r in v)};"
              f" a-distribution {dict(sorted(Counter(r['a'] for r in v).items()))}")
    print(s)
    return v


def main(y):
    rows = json.load(open(f'research/proof/chain_teeth_r33_fam_m{y}.json'))
    gears = gears_of(y)
    q1 = next_prime(y)
    vp = real_tooth(q1)
    ap = letter_a(q1, vp)
    print(f"\n=== m{y}: gears {gears}, q'={q1}, pinned v'={vp} a={ap}, {len(rows)} rows ===")
    T = [r for r in rows if is_T(gears, r['teeth'])]
    Lr = [r for r in rows if r['v1'] == vp]
    TL = [r for r in T if r['v1'] == vp]
    slice_report("ALL (free tooth, all teeth)", rows, q1)
    slice_report("(L) pinned only", Lr, q1)
    vT = slice_report("(T) only, tooth free", T, q1)
    slice_report("(T) only, tooth free, a >= 2", [r for r in T if r['a'] >= 2], q1)
    slice_report("(T)+(L) sub-family", TL, q1)
    T57 = [r for r in Lr if sep_of(5, r['teeth'][0]) >= 2 and sep_of(7, r['teeth'][1]) >= 2]
    slice_report("(T) at gears 5,7 only + (L)", T57, q1)
    Thi = [r for r in Lr if all(sep_of(q, v) >= 2 for q, v in zip(gears, r['teeth']) if q >= 11)]
    slice_report("(T) at gears >= 11 only + (L)", Thi, q1)
    vP = [r for r in Lr if r['viol']]
    print(f"  pinned violators: degenerate gear sets {dict(sorted(Counter(tuple(q for q, v in zip(gears, r['teeth']) if sep_of(q, v) == 1) for r in vP).items()))}")
    if vT:
        cnt = Counter(r['a'] for r in vT)
        print(f"  PC2: (T)-only violators at a = {ap - 1}: {cnt.get(ap - 1, 0)}, a = {ap + 1}: {cnt.get(ap + 1, 0)};"
              f" largest share of one letter {max(cnt.values()) / len(vT):.2f}")
    else:
        print("  PC2: no (T)-only violators")
    by = {}
    for r in T:
        if r['chain'] is not None:
            by.setdefault(r['a'], []).append(r['F'] + q1 - r['chain'])
    print("  (T)-only min chain margin by letter a: " + "; ".join(f"a={a}: {min(v)}" for a, v in sorted(by.items())))
    keys = (("min sep, all gears", lambda r: min(sep_of(q, v) for q, v in zip(gears, r['teeth']))),
            ("min sep, gears >= 11", lambda r: min(sep_of(q, v) for q, v in zip(gears, r['teeth']) if q >= 11)),
            ("min sep/q x100 (5-bins)", lambda r: 5 * int(100 * min(sep_of(q, v) / q for q, v in zip(gears, r['teeth'])) // 5)))
    for lab, key in keys:
        by = {}
        for r in Lr:
            if r['chain'] is not None:
                by.setdefault(key(r), []).append((r['F'] + q1 - r['chain'], bool(r['viol'])))
        print(f"  PINNED rows by {lab}: " + "; ".join(
            f"{s}: n={len(v)} viol {sum(b for _, b in v)} min {min(m for m, _ in v)} mean {sum(m for m, _ in v) / len(v):.1f}"
            for s, v in sorted(by.items())))
    for k, name in ((0, "S1 Phi(a) <= F+b"), (1, "S2 Phi(q') <= F"), (2, "S3 Phi(a,b) <= F"), (3, "S1b Phi(b) <= F+a")):
        ev = [((margins(r, q1)[k] if k < 3 else (r['F'] + r['a'] - r['PhiB'] if r['PhiB'] is not None else None)), r) for r in rows]
        ev = [(m, r) for m, r in ev if m is not None]
        fails = [r for m, r in ev if m < 0]
        fT = [r for r in fails if is_T(gears, r['teeth'])]
        fL = [r for r in fails if r['v1'] == vp]
        fTL = [r for r in fT if r['v1'] == vp]
        subev = [m for m, r in ev if is_T(gears, r['teeth']) and r['v1'] == vp]
        Tev = [m for m, r in ev if is_T(gears, r['teeth'])]
        Lev = [m for m, r in ev if r['v1'] == vp]
        print(f"  {name}: evaluated {len(ev)}/{len(rows)}, fails {len(fails)} [with (T) {len(fT)}, with (L) {len(fL)}, both {len(fTL)}];"
              f" min margin: all {min(m for m, _ in ev) if ev else None}, (T) {min(Tev) if Tev else None},"
              f" (L) {min(Lev) if Lev else None}, (T)+(L) {min(subev) if subev else None} (evaluated {len(subev)}/{len(TL)})")
        if fails:
            print(f"      failing a-distribution {dict(sorted(Counter(r['a'] for r in fails).items()))};"
                  f" degenerate gears among failures {dict(sorted(Counter(q for r in fails for q, v in zip(gears, r['teeth']) if sep_of(q, v) == 1).items()))}")
    tight = Counter()
    for r in TL:
        ms = list(margins(r, q1)) + [r['F'] + r['a'] - r['PhiB'] if r['PhiB'] is not None else None]
        cand = [(m, kk) for m, kk in zip(ms, ("S1", "S2", "S3", "S1b")) if m is not None]
        if cand:
            tight[min(cand)[1]] += 1
    print(f"  PC4 tight statement per sub-family row: {dict(tight)}; S3 vacuous at {sum(1 for r in TL if r['PhiAB'] is None)}/{len(TL)};"
          f" S2 vacuous at {sum(1 for r in TL if r['PhiQ'] is None)}/{len(TL)}")
    Jc = Counter()
    for r in TL:
        if r['chain'] is None:
            continue
        J = min(J for J in r['argmax'] if r['argmax'][J][1] + sum(r['argmax'][J][0]) + r['argmax'][J][2] == r['chain'])
        w, gL, gR, lit, i = r['argmax'][J]
        cls = ''.join('0' if v % q1 == 0 else ('a' if v % q1 == r['a'] else 'b') for v in w)
        Jc[(int(J), cls, 'lit' if lit else 'pad')] += 1
    print(f"  sub-family: shape of the chain-maximising cell: {dict(sorted(Jc.items()))}")
    worst = sorted((r['F'] + q1 - r['chain'], tuple(r['teeth']), r['F'], r['F2'], r['chain']) for r in TL if r['chain'] is not None)[:5]
    print(f"  sub-family five smallest margins (margin, teeth, F, F_2, chain): {worst}")
    real = tuple(real_tooth(q) for q in gears)
    rr = [r for r in TL if tuple(r['teeth']) == real]
    if rr:
        print(f"  real machine row: F={rr[0]['F']} chain {rr[0]['chain']} margin {rr[0]['F'] + q1 - rr[0]['chain']}")


if __name__ == '__main__':
    for y in [int(t) for t in sys.argv[1:]]:
        main(y)
