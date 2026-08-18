"""Constructor round 12: THE WORD-INDEXED TOLERANCE THEOREM (corrected).

KEY STRUCTURAL FACT (consecutive steps): gcd(P_M, q') = 1, so the q' CRT
copies of M's period inside the joint period realize EVERY residue shift.
Hence (i) the F2 pair's middle opening is deleted in some copy:
F(M+q') >= F2(M) always; (ii) every occurrence of a COMPATIBLE literal word
(one whose letter walk can start on a tooth: 1-2 valid start residues mod q')
fires in exactly that many copies: its merge IS realized. The word-indexed
formula is therefore an IDENTITY, not just a ceiling:

  F(M+q') = max( F2(M),
                 max over compatible qualifying words w of span(w)+FS_max(w) )

where qualifying words = literal alternating words (<= litcap members,
round 11) plus padded words (some letter >= q', value = 0 or +-2c mod q').
Tolerance needs: every tier <= F(M) + 2.5q'. The missing bound = FS_max
(flank sums at word occurrences) at the binding word - computed here per step.
"""
import numpy as np
from math import prod
import sys
sys.path.insert(0, "research")
from fuel_bound import literal_cap, gapword, GEARS

STEPS = [(11, 13), (13, 17), (17, 19), (19, 23), (23, 29), (29, 31)]
FK = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43}
F2K = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55}
FNEW = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}


def words(q1):
    a = 2 * round(q1 / 6)
    b = q1 - a
    L = literal_cap(q1)
    out = []
    for ell in range(1, L):
        for start in (a, b):
            w = tuple(start if j % 2 == 0 else (a + b - start)
                      for j in range(ell))
            if w not in out:
                out.append(w)
    return out


def valid_starts(w, q1):
    """start residues r with r and every r+prefix-sum in teeth {c, q'-c}."""
    c = pow(6, -1, q1)
    teeth = {c % q1, (q1 - c) % q1}
    out = []
    for r in teeth:
        p = r
        ok = True
        for x in w:
            p = (p + x) % q1
            if p not in teeth:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def analyse(y, q1):
    gaps = gapword(y).astype(np.int32)
    n = len(gaps)
    F, F2 = FK[y], F2K[y]
    W = words(q1)
    print(f"\n=== step {y}->{q1}: litcap={literal_cap(q1)}")
    Clit, binding = 0, None
    for w in W:
        ell = len(w)
        vs = valid_starts(w, q1)
        # word at gaps[i..i+ell-1], i in [1, n-ell-1); flanks i-1, i+ell
        m = gaps[1:n - ell] == w[0]
        for j in range(1, ell):
            m &= gaps[1 + j:n - ell + j] == w[j]
        idx = np.flatnonzero(m) + 1
        if len(idx) == 0:
            print(f"    w={w}: 0 occurrences (starts {vs})")
            continue
        fs = gaps[idx - 1] + gaps[idx + ell]
        top = sum(w) + int(fs.max())
        tag = "COMPATIBLE" if vs else "incompatible (never fires)"
        print(f"    w={w}: occ={len(idx):,} FS_max={int(fs.max())} "
              f"tier={top} ({tag}, starts={vs})")
        if vs and top > Clit:
            Clit, binding = top, w
    # padded tier: windows whose middles are all qualifying VALUES
    # (= 0 or +-2c mod q1) with at least one middle >= q1
    a = 2 * round(q1 / 6)
    qual = np.zeros(n, bool)
    for v in range(a, int(gaps.max()) + 1):
        if v % q1 in (0, (2 * pow(6, -1, q1)) % q1,
                      (-2 * pow(6, -1, q1)) % q1):
            qual |= gaps == v
    huge = gaps >= q1
    Cpad = 0
    for ell in range(1, 7):
        ok = np.ones(n - ell - 1, bool)
        anyh = np.zeros(n - ell - 1, bool)
        for j in range(1, ell + 1):
            ok &= qual[j:n - ell - 1 + j]
            anyh |= huge[j:n - ell - 1 + j]
        sel = np.flatnonzero(ok & anyh)
        if len(sel) == 0:
            continue
        s = np.zeros(len(sel), np.int64)
        for j in range(ell + 2):
            s += gaps[sel + j]
        Cpad = max(Cpad, int(s.max()))
    ident = max(F, F2, Clit, Cpad)
    budget = 2.5 * q1 / 3
    ok = "=" if ident == FNEW[q1] else f"!= {FNEW[q1]} MISMATCH"
    print(f"  tiers: F={F}  F2={F2}(+{F2-F})  C_lit={Clit}(+{Clit-F}) "
          f"binding {binding}  C_pad={Cpad}"
          f"({'+%d' % (Cpad-F) if Cpad else 'absent'})")
    print(f"  IDENTITY: max tiers = {ident} {ok} (known F(M+q'))  "
          f"incr = {ident - F} vs budget {budget:.1f} "
          f"[{'WITHIN' if ident - F <= budget else 'EXCEEDS'}]")


if __name__ == "__main__":
    for y, q1 in STEPS:
        analyse(y, q1)
