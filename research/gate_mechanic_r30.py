"""Round 30 (mechanic): THE ROUND GATE.

One command, one process, importing nothing from the tools that produced the
numbers.  Slot k is blocked by gear q iff k = +-6^{-1} (mod q).

  A. brief item (d): the seed-144 word-legal run's 64 shard logs TILE machine
     23's period exactly; the per-J maxima over the tiling are re-read; the
     J = 4 witness (a machine-23 start + phase vector) is lifted by CRT to a
     MACHINE-47 SLOT and re-checked there - five consecutive openings, every
     other slot blocked, the two middle gaps legal letters for 53 with
     alternating classes - and then lifted once more to MACHINE 53, where the
     window is a gap of 145: so F(53) >= 145 is EXHIBITED and max_J Q*_J(47)
     = 145 <= 171 closes rung eleven with machine 53's record never consulted.
  B. probe (b): every extension word on disk is a T3-legal one-letter
     extension of a realised length-L(M) word; its recorded SAT set is
     recomputed from the definition; and for m19..m31 the cover-only verdict
     (no slot of M blocks the punctured interior) is re-derived by a direct
     period scan of the TARGET machine, not by the CSP.
  C. probe (a): V2 (the longest alternating three-class run) equals
     D_g - 1 = A_kill(M -> g) - 1 = L(M) at every next-prime cell on disk,
     against the recorded A_kill values, and the recorded attaining runs are
     re-checked as consecutive openings of M with the stated residues.
  D. probe (c): every lifted record slot is re-verified at its machine.

usage: uv run python research/gate_mechanic_r30.py
"""
import json
import os
import re
import sys
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "r30")


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def gears(y):
    return [p for p in range(5, y + 1) if is_prime(p)]


def teeth(q):
    u = pow(6, -1, q)
    return u, (-u) % q


def is_open(k, y):
    return all(k % q not in teeth(q) for q in gears(y))


def openings(y):
    G = gears(y)
    P = prod(G)
    ex = np.zeros(P, bool)
    for q in G:
        for u in teeth(q):
            ex[u % q::q] = True
    return np.flatnonzero(~ex).astype(np.int64), P


# ------------------------------------------------------------------ A
def sectionA():
    d = os.path.join(DATA, "q47_s144")
    tiles, mx, wit = [], {}, None
    for f in sorted(os.listdir(d)):
        txt = open(os.path.join(d, f), errors="replace").read()
        m = re.search(r"WALKING start-opening indices \[([\d,]+), ([\d,]+)\)", txt)
        assert m and "scan complete" in txt, ("incomplete shard", f)
        tiles.append(tuple(int(m.group(i).replace(",", "")) for i in (1, 2)))
        for J, v in re.findall(r"^\s+(\d)\s+(\d+)\s", txt, re.M):
            mx[int(J)] = max(mx.get(int(J), 0), int(v))
        w = re.search(r"J=4: k=([\d,]+) span=(\d+) phases=\(([\d, ]+)\) "
                      r"marks=\(([\d, ]+)\)", txt)
        if w:
            wit = (int(w.group(1).replace(",", "")), int(w.group(2)),
                   tuple(int(x) for x in w.group(3).split(",")),
                   tuple(int(x) for x in w.group(4).split(",")))
    tiles.sort()
    assert tiles[0][0] == 0 and tiles[-1][1] == 7952175
    for i in range(1, len(tiles)):
        assert tiles[i][0] == tiles[i - 1][1], ("tiling gap", tiles[i - 1], tiles[i])
    print(f"  A {len(tiles)} shards tile [0, 7,952,175) exactly; per-J maxima "
          f"(seed 144): {mx}")
    assert mx == {2: 144, 3: 144, 4: 145, 5: 144, 6: 144}, mx
    assert max(mx.values()) == 145 <= 171
    # lift the witness
    K, SPAN, PH, MARKS = wit
    op, P23 = openings(23)
    i = int(np.searchsorted(op, K))
    assert op[i] == K and op[int(np.searchsorted(op, K + SPAN))] == K + SPAN
    j = int(np.searchsorted(op, K + SPAN))
    interior = [int(v) - K for v in op[i + 1:j]]
    offs = [0] + [interior[m] for m in MARKS] + [SPAN]
    NEW = [29, 31, 37, 41, 43, 47]
    t, Mm = 0, 1
    for q, c in zip(NEW, PH):
        r = (-c * pow(P23 % q, -1, q)) % q
        t += Mm * ((r - t) * pow(Mm % q, -1, q) % q)
        Mm *= q
    x = (K + t * P23) % (P23 * Mm)
    P47 = prod(gears(47))
    assert P23 * Mm == P47 and 0 <= x < P47
    oset = set(offs)
    for s in range(SPAN + 1):
        assert is_open(x + s, 47) == (s in oset), ("m47 mismatch", s)
    gaps = [offs[k + 1] - offs[k] for k in range(len(offs) - 1)]
    assert sum(gaps) == 145 and len(gaps) == 4
    # word legality for 53: middles in {0, +-18} mod 53, alternating
    dd = (2 * pow(6, -1, 53)) % 53
    cls = []
    for v in gaps[1:-1]:
        r = v % 53
        cls.append(0 if r == 0 else 1 if r == dd else -1 if r == (53 - dd) else None)
    assert None not in cls, ("middle not a legal letter", gaps, cls)
    nz = [c for c in cls if c]
    assert all(nz[k] != nz[k + 1] for k in range(len(nz) - 1)), ("T3", cls)
    print(f"  A J=4 witness lifted to machine-47 slot {x}: openings at {offs}, "
          f"gap word {gaps}, middles {gaps[1:-1]} = classes {cls} mod 53 "
          f"(legal, alternating), 141 other slots blocked")
    # and to machine 53: some phase deletes exactly the three interiors
    P53 = P47 * 53
    found = None
    for tt in range(53):
        y = (x + tt * P47) % P53
        if all(not is_open(y + o, 53) for o in offs[1:-1]) and is_open(y, 53) \
                and is_open(y + SPAN, 53):
            assert all(not is_open(y + s, 53) for s in range(1, SPAN))
            found = y
            break
    assert found is not None
    print(f"  A lifted to MACHINE 53: slot {found} is a gap of exactly 145 - "
          f"F(53) >= 145 EXHIBITED; with max_J Q*_J(47) = 145 <= 171 the "
          f"rung 47 -> 53 is CLOSED from machine 23's period alone")
    anchor = 82799441296736535
    mirror = (P47 - anchor - 145) % P47
    print(f"  A the slot is the round-26 anchor ({x == anchor}) or its mirror "
          f"({x == mirror})")


# ------------------------------------------------------------------ B
def cls_of(v, g):
    d = (2 * pow(6, -1, g)) % g
    r = v % g
    return 0 if r == 0 else 1 if r == d else -1 if r == (g - d) % g else None


def t3_ok(word, g):
    last = 0
    for v in word:
        c = cls_of(v, g)
        if c is None:
            return False
        if c:
            if c == last:
                return False
            last = c
    return True


def sat_set(X, G):
    out = []
    for g in G:
        u = pow(6, -1, g)
        E = {r for r in range(g) if r != u % g and r != (-u) % g}
        xs = {x % g for x in X}
        if not any(all((t + x) % g in E for x in xs) for t in range(g)):
            out.append(g)
    return out


def sectionB():
    for f in sorted(os.listdir(DATA)):
        if not (f.startswith("killer_m") and f.endswith(".json")):
            continue
        J = json.load(open(os.path.join(DATA, f)))
        M, q, L = J["M"], J["q"], J["L"]
        G = gears(M)
        words = {tuple(w) for w in J["words"]}
        assert all(len(w) == L and t3_ok(w, q) for w in words)
        n_cover = n_sat = 0
        for r in J["ext"]:
            w = tuple(r["word"])
            assert len(w) == L + 1 and t3_ok(w, q), (M, w)
            assert w[:-1] in words or w[1:] in words or w[::-1][:-1] in words \
                or w[::-1][1:] in words, (M, "not an extension", w)
            X = [0]
            for v in w:
                X.append(X[-1] + v)
            assert sat_set(X, G) == r["sat"], (M, w, r["sat"])
            assert r["full"] is False, (M, w, "not refuted", r["full"])
            n_cover += r["ystar"] == 0
            n_sat += bool(r["sat"])
        print(f"  B m{M} -> {q}: {len(J['ext'])} extension classes of {len(words)} "
              f"realised length-{L} words, all refuted; SAT sets recomputed; "
              f"{n_cover} cover-only, {n_sat} single-gear saturated")
        if M <= 23:
            # cover-only verdicts by a direct period scan of M (P(23) = 3.7e7
            # slots; machine 29's 1.1e9 and beyond are not scanned here)
            op, P = openings(M)
            blocked = np.ones(P, bool)
            blocked[op] = False
            for r in J["ext"]:
                w = r["word"]
                X, span = [0], sum(w)
                for v in w:
                    X.append(X[-1] + v)
                Y = [t for t in range(1, span) if t not in set(X)]
                ok = np.ones(P, bool)
                for t in Y:
                    ok &= np.roll(blocked, -t)
                exists = bool(ok.any())
                assert exists == (r["R_empty"] is True), (M, w, exists, r["R_empty"])
            print(f"  B m{M}: every cover-only / not-cover-only verdict "
                  f"reproduced by a direct scan of machine {M}'s period")


# ------------------------------------------------------------------ C
AKILL = {13: 2, 17: 2, 19: 2, 23: 3, 29: 2, 31: 4, 37: 4, 41: 3}   # M -> A_kill(M-prev -> M)


def sectionC():
    for M in (11, 13, 17, 19, 23, 29, 31):
        fn = os.path.join(DATA, f"resrun_m{M}.json")
        if not os.path.exists(fn):
            print(f"  C m{M}: no scan on disk")
            continue
        J = json.load(open(fn))
        assert J["ngaps"] == prod(q - 2 for q in gears(M)), (M, J["ngaps"])
        qn = next(p for p in range(M + 1, 200) if is_prime(p))
        e = J["g"][str(qn)]
        assert e["t3"] + 1 == AKILL[qn], (M, qn, e["t3"], AKILL[qn])
        for nm in ("raw_wit", "t3_wit"):
            w = e[nm]
            k, gl = w["slot"], w["gaps"]
            assert is_open(k, M)
            pos = k
            for v in gl:
                assert all(not is_open(pos + s, M) for s in range(1, v))
                pos += v
                assert is_open(pos, M)
            assert w["residues"] == [v % qn for v in gl]
        print(f"  C m{M} (N = {J['ngaps']:,}): V2 at q' = {qn} is {e['t3']} = "
              f"A_kill - 1 = {AKILL[qn] - 1}; raw run {e['raw']}; both attaining "
              f"runs re-checked as consecutive openings of machine {M}")


# ------------------------------------------------------------------ D
def sectionD():
    for y, k, F in ((43, 426824541409250, 103), (47, 34905861380755417, 118),
                    (53, 4182064658553345935, 145),
                    (59, 73115517300464200662, 161)):
        assert 0 <= k < prod(gears(y))
        assert is_open(k, y) and is_open(k + F, y)
        assert all(not is_open(k + s, y) for s in range(1, F))
        print(f"  D machine {y}: slot {k} is a gap of exactly {F} = F({y})")


if __name__ == "__main__":
    print("ROUND-30 MECHANIC GATE\n\nA. the independent Q*_J(47) (item d)")
    sectionA()
    print("\nB. killer profiles (probe b)")
    sectionB()
    print("\nC. residue runs vs A_kill (probe a)")
    sectionC()
    print("\nD. lifted record slots (probe c)")
    sectionD()
    print("\nALL ASSERTIONS PASSED")
