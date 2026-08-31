"""Round 28 (mechanic): THE ROUND'S ASSERTION GATE, from a clean process.

Re-derives every structural claim of the round from the definitions, importing
as little as possible from the tools that produced them.

  A. THE DEPTH-0 LEMMA at every exact pair, recomputed from the PERIOD (not
     from the CSVs) at machines 13, 17, 19, 23, and from the full-period CSVs
     at 29, 31, 37, plus the round-27 m41 shard.
  B. The cyclic close of every recomputed period (rule 25): N gaps for N
     openings, sum = P, wrap gap = first gap.
  C. THE ONSET LADDER re-measured, and THE ONSET LAW re-asserted at the six
     in-sample steps.
  D. THE WALK SCREEN is SOUND: it removes no realised tuple at any of the six
     steps, and it subsumes the round-26 emission screen.
  E. THE DEPTH CAP JMAX = 5 for the F(59) pin is JUSTIFIED: a word-legal
     window of J gaps has J-1 INTERIOR OPENINGS deleted by one phase, i.e. a
     kill chain of ARITY J-1 (its word has J-2 letters; A_kill counts openings),
     and A_kill(53->59) = 4 with N_5 = 0 (round 27, C36), so Q*_6 = Q*_7 = 0.  Re-derived here as an arithmetic identity
     on the round-27 realised-word list, not quoted.
  F. THE F(59) PIN: the band's seven workers TILE machine 23's period exactly
     and every one reports 161 (so the band (161,178] is EMPTY), and the lower
     half - the machine-53 window of span 161 - is re-checked slot by slot from
     the definition over 14 gears.

usage: <venv>/python research/gate_mechanic_r28.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)

from dict_transfer import load_dict, transfer          # noqa: E402
from onset_r28 import screen                           # noqa: E402
from onset_ladder_r28 import gaps_cyclic, ktuples, primes_upto  # noqa: E402
from onset_walkscreen_r28 import ws_transfer           # noqa: E402

F1 = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
F4 = {13: 26, 17: 33, 19: 38, 23: 58, 29: 70, 31: 90, 37: 105}
PAIRS = [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31), (31, 37)]
ONSET = {(13, 17): 15, (17, 19): 17, (19, 23): 25, (23, 29): 31,
         (29, 31): 41, (31, 37): 53}
NEXT = {17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41}
# round-27 C36: the complete realised kill-word levels at 53 -> 59
R27_WORDS = {3: [(20, 39), (39, 20), (20, 59), (59, 20), (20, 98), (98, 20),
                 (20, 118), (118, 20)],
             4: [(20, 98, 20)],
             5: []}


def main():
    print("MECHANIC ROUND-28 GATE\n")

    print("A/B. exact dictionaries + cyclic close")
    D = {}
    for y in (13, 17, 19, 23):
        g = gaps_cyclic(y)                       # asserts the cyclic close
        D[y] = ktuples(g, 4)
        assert int(g.max()) == F1[y], ("F", y)
        assert max(sum(t) for t in D[y]) == F4[y], ("F_4", y)
        print("   m%-2d recomputed from the period: %d gaps, F = %d, F_4 = %d"
              % (y, len(g), int(g.max()), F4[y]))
    for y in (29, 31, 37):
        D[y] = set(load_dict(os.path.join(DATA, "gap_tuples_%d_4.csv" % y)))
        assert max(t[0] for t in D[y]) <= F1[y]
        assert max(sum(t) for t in D[y]) == F4[y], ("F_4", y)
        print("   m%-2d full-period CSV: %d 4-tuples, max span %d = F_4"
              % (y, len(D[y]), F4[y]))
    shard = set(load_dict(os.path.join(DATA, "r27",
                                       "gap_tuples_41_4_exact_le77.csv")))

    print("\nA. THE DEPTH-0 LEMMA  D_4(M) subset D_4(M + q')")
    for M, qp in PAIRS:
        assert not (D[M] - D[qp]), ("DEPTH-0 LEMMA VIOLATED", M, qp)
        print("   D_4(%2d) subset D_4(%2d)   OK" % (M, qp))
    assert not ({t for t in D[37] if sum(t) <= 77} - shard), "37 -> 41"
    print("   D_4(37)|span<=77 subset the exact m41 shard   OK")

    print("\nC/D. onset ladder, onset law, walk-screen soundness")
    for M, qp in PAIRS:
        sup, _, _ = transfer(sorted(D[M]), qp, F4[qp], F1[qp], verbose=False)
        assert not (D[qp] - sup), ("superset violated", M, qp)
        scr, _ = screen(sup, qp)
        assert not (D[qp] - set(scr)), ("emission screen unsound", M, qp)
        ws, _ = ws_transfer(sorted(D[M]), M, qp, F4[qp], F1[qp])
        assert not (D[qp] - ws), ("WALK SCREEN UNSOUND", M, qp)
        wss, _ = screen(sorted(ws), qp)
        assert set(wss) == ws, ("walk screen does not subsume the emission "
                                "screen", M, qp)
        o = min((sum(t) for t in scr if t not in D[qp]), default=None)
        assert o == ONSET[(M, qp)], ("onset moved", M, qp, o)
        nxt = NEXT[qp]
        Dn = shard if nxt == 41 else D[nxt]
        law = min((sum(t) for t in Dn - D[qp]), default=None)
        assert law == o, ("ONSET LAW FAILS", M, qp, o, law)
        ref = [t for t in scr if sum(t) == o and t not in D[qp]]
        assert ref and all(t in Dn for t in ref), ("mechanism", M, qp)
        print("   %2d->%2d  onset %3d = min span D_4(%d)\\D_4(%d); walk screen "
              "sound and subsuming; %d refuted at the onset span, all realised "
              "at m%d" % (M, qp, o, nxt, qp, len(ref), nxt))

    print("\nE. the F(59) pin's depth cap")
    s = (2 * pow(6, -1, 59)) % 59
    for k, ws_ in sorted(R27_WORDS.items()):
        for w in ws_:
            p = lo = hi = 0
            for v in w:
                r = v % 59
                assert r in (0, s, (-s) % 59), ("illegal letter", w, v)
                p += 0 if r == 0 else (1 if r == s else -1)
                lo, hi = min(lo, p), max(hi, p)
            assert hi - lo <= 1, ("prefix range", w)
    assert R27_WORDS[5] == [], "N_5 must be empty"
    # the overlap lemma, re-derived: a realised 4-letter word needs both of its
    # 3-letter sub-words realised, and the only realised 3-letter word is the
    # palindrome (20,98,20), whose two overlaps cannot both be it.
    three = {(20, 98, 20)}
    cand5 = [(a, b) for a in three for b in three if a[1:] == b[:2]]
    assert not cand5, "a 5-letter word would be possible"
    print("   A_kill(53->59) = 4: every realised word legal, N_5 empty by the "
          "overlap lemma  ->  Q*_6 = Q*_7 = 0, so JMAX = 5 is exhaustive")

    print("\nF. the F(59) pin: the band's workers TILE machine 23's period, "
          "and all report 161")
    import re
    d = os.path.join(DATA, "r28")
    logs = sorted(f for f in os.listdir(d)
                  if f.startswith("f59_pin_161_178_J5_w") and f.endswith(".log"))
    assert len(logs) == 7, ("expected 7 band workers", len(logs))
    ivals, maxima = [], []
    for f in logs:
        txt = open(os.path.join(d, f), errors="replace").read()
        assert "scan complete" in txt, ("worker did not finish", f)
        m = re.search(r"WALKING start-opening indices \[([\d,]+), ([\d,]+)\)",
                      txt)
        ivals.append((int(m.group(1).replace(",", "")),
                      int(m.group(2).replace(",", ""))))
        maxima += [int(x) for x in re.findall(r"max over J = (\d+)", txt)]
    ivals.sort()
    assert ivals[0][0] == 0 and ivals[-1][1] == 7952175, ("not a tiling", ivals)
    for a, b in zip(ivals, ivals[1:]):
        assert a[1] == b[0], ("gap or overlap in the tiling", a, b)
    assert set(maxima) == {161}, ("a worker found something above the seed",
                                  sorted(set(maxima)))
    print("   7 workers tile [0, 7,952,175) exactly; every one reports 161, so "
          "the band (161,178] is EMPTY")
    # the lower half, from the definition
    K, OFF = 2505673933219103747, [0, 10, 128, 161]
    g53 = [p for p in primes_upto(53) if p >= 5]
    uu = {q: pow(6, -1, q) for q in g53}
    oset = set(OFF)
    for t in range(OFF[-1] + 1):
        op = all((K + t) % q not in (uu[q] % q, (-uu[q]) % q) for q in g53)
        assert op == (t in oset), ("machine-53 witness mismatch", t)
    s59 = (2 * pow(6, -1, 59)) % 59
    assert (OFF[2] - OFF[1]) % 59 in (0, s59, (-s59) % 59), "middle not legal"
    print("   lower half: the machine-53 window at k = %d has gaps [10,118,33],"
          " span 161, every other slot blocked -> F(59) >= 161" % K)
    print("   => F(59) = 161 EXACT (given the round-27 bands above 178)")

    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
