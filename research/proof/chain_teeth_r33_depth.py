"""Prover C r33 -- the depth mechanism: admissible word depth at gears 5 and 7 (bare-word lemma)
versus the chain violators, per incoming letter.  Uses the family rows (chain_teeth_r33_fam_m<y>.json)
or, if absent, prover B's violator file (chain_family_r32_viol_m<y>.json)."""
import json, os, sys, itertools
from collections import Counter
sys.path.insert(0, 'research/proof')
from chain_teeth_r33 import gears_of, next_prime, real_tooth, letter_a, is_T, sep_of
from chain_teeth_r33_bare import legal_words, admissible

def depth57(a, q1, v5, v7, maxlen=6, literal_only=False):
    words = legal_words(a, q1 - a, q1, maxlen)
    if literal_only:
        words = [w for w in words if q1 not in w]
    return max((len(w) for w in words if admissible(w, 5, v5) and admissible(w, 7, v7)), default=0)

for y in [int(t) for t in sys.argv[1:]]:
    gears = gears_of(y); q1 = next_prime(y); vp = real_tooth(q1); ap = letter_a(q1, vp)
    fam = f'research/proof/chain_teeth_r33_fam_m{y}.json'
    if os.path.exists(fam):
        rows = json.load(open(fam)); V = [r for r in rows if r['viol']]; src = 'fam rows'
    else:
        rows = None; V = json.load(open(f'research/proof/chain_family_r32_viol_m{y}.json')); src = 'prover B violators'
    print(f"\n=== m{y} q'={q1} pinned a={ap} ({src}) ===")
    print("  admissible depth at (5,7) with v5=1, v7=1 | v7=2  [legal / literal-only], per incoming letter a:")
    for a in range(1, (q1 - 1) // 2 + 1):
        d11 = depth57(a, q1, 1, 1); d12 = depth57(a, q1, 1, 2)
        l11 = depth57(a, q1, 1, 1, literal_only=True); l12 = depth57(a, q1, 1, 2, literal_only=True)
        vT = [r for r in V if is_T(gears, r['teeth']) and r['a'] == a]
        Jc = dict(sorted(Counter(int(J) for r in vT for J in r['viol']).items()))
        nrows = (sum(1 for r in rows if is_T(gears, r['teeth']) and r['a'] == a) if rows else None)
        Lmax = max((r['L'] for r in vT), default=None)
        print(f"    a={a:<2}{' (pinned)' if a == ap else '         '} depth {d11}/{d12} [lit {l11}/{l12}] | (T)-only violators {len(vT)}"
              f"{f' of {nrows} rows' if nrows else ''}; violating J cells {Jc}; max L among them {Lmax}")
    # pinned violators: their (v5, v7) and admissible depth with those teeth
    pin = [r for r in V if r['v1'] == vp]
    print(f"  pinned violators {len(pin)}: (v5, v7) -> count, admissible depth with those teeth, J cells:")
    for (v5, v7), n in sorted(Counter((r['teeth'][0], r['teeth'][1]) for r in pin).items()):
        sel = [r for r in pin if r['teeth'][0] == v5 and r['teeth'][1] == v7]
        Jc = dict(sorted(Counter(int(J) for r in sel for J in r['viol']).items()))
        print(f"    (v5,v7)=({v5},{v7}) x{n}: depth {depth57(ap, q1, v5, v7)} [lit {depth57(ap, q1, v5, v7, literal_only=True)}] J cells {Jc}")
    # J=3 violators anywhere: letters and teeth
    j3 = [r for r in V if '3' in r['viol']]
    print(f"  J=3 violators {len(j3)}: (a, v5, v7, (T)?) -> count: {dict(sorted(Counter((r['a'], r['teeth'][0], r['teeth'][1], is_T(gears, r['teeth'])) for r in j3).items()))}")
