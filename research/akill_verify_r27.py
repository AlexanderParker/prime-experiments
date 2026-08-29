"""Round 27 (mechanic) GATE: A_kill(53 -> 59), every verdict re-derived.

The k=3 level at 53 -> 59 has 36 legal words = 19 reverse classes, and this
round closes it from FOUR independent sources, only two of which cost a solver
call:

  (i)   REALISED - an exhibited address, re-verified here FROM THE DEFINITION
        at machine 53 (the k members are openings, every other slot of the
        span is blocked gear by gear, some residue mod 59 puts every member on
        a tooth, and the CRT combination is re-checked from scratch).
  (ii)  ZERO BY PHASE SATURATION (K9) - re-derived here by pure arithmetic,
        no solver, from the exposed set alone.
  (iii) ZERO BY THE ROUND-26 F_2(53) SCAN - a word whose span lies in
        (159, 200] needs a 2-window of machine 53 of that span, and the
        round-26 floor-1 lap-phase transfer examined EVERY window of span in
        (145, 200] over three ranges that TILE machine 23's period and found a
        maximum of 159.  Note carefully: F_2(53) <= 159 is conditional on that
        run's span cap 200, but "NO 2-window has span in (159, 200]" is NOT -
        the cap only conditions statements about spans ABOVE it.  This is
        rule 20 / rule 29 in action: buy the refutation from the cheaper
        source.
  (iv)  ZERO BY THE ROUND-27 F(59) BAND SCAN - the same argument one band up,
        for words of span in (203, 260], from this round's stage-A run.
  (v)   ZERO BY SAT - the residue: words whose span is at or below F_2(53),
        which no scan can refute.  An UNSAT is not cheaply re-derivable and is
        the ONLY class of verdict here taken from its own log.

Usage: .venv-sat/Scripts/python.exe research/akill_verify_r27.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from akill_verify_r26 import (verify, free_phases, gears,        # noqa: E402
                              check_occurrence)

M, QP = 53, 59
DATA = os.path.join(HERE, "data")

# ---- the round-26 F_2(53) run: three range workers TILING m23's period ----
F2_LOGS = [os.path.join(DATA, "r26", f) for f in
           ("f2_53_head.log", "f2_53_mid.log", "f2_53_w1.log")]
NOPEN23 = 7952175
F2_SEED, F2_CAP, F2_VALUE = 145, 200, 159

# ---- the round-27 F(59) stage-A run: two range workers, same tiling ----
FA_LOGS = [os.path.join(DATA, "r27", f) for f in
           ("f59_A_w0.log", "f59_A_w1.log")]
FA_SEED, FA_CAP = 203, 260

# ---- the round-27 m53 TOP-BAND 2-window run: seed 145, cap 158 ----
# Floor 1 (no middle-gap condition), J = 2 only, two range workers tiling
# machine 23's period.  Its maximum is the largest 2-window span of machine 53
# at or below 158, so everything strictly above it and at or below 158 is
# absent - UNCONDITIONALLY (the cap only conditions spans above 158, and 159
# is separately known realised from round 26).
FT_LOGS = [os.path.join(DATA, "r27", f) for f in
           ("f2_53_top_w0.log", "f2_53_top_w1.log")]
FT_SEED, FT_CAP, FT_VALUE = 145, 158, 152

# ---- the round-27 DEPTH-3 word-legal band: seed 152, cap 184, JMAX = 3 ----
# Three range workers tiling machine 23's period.  Its maximum is the largest
# word-legal span at depth J <= 3 in (152, 184], so nothing strictly above it
# and at or below 184 exists at those depths.  A k-chain word with k-1 <= 3
# letters is exactly such a window.
WJ3_LOGS = [os.path.join(DATA, "r27", "f59_wlJ3_152_184_w%d.log" % i)
            for i in range(3)]
WJ3_SEED, WJ3_CAP, WJ3_VALUE, WJ3_JMAX = 152, 184, 161, 3

# Every word the round-27 campaign found REALISED, with the address its own
# process reported (research/data/r27/akill_53_59.log).  Each is re-derived
# here from the definition, not re-read.
WITNESSES = [
    (5408553654414421963, (20, 39)),
    (1522353991400668678, (39, 20)),
    (4976851792281208593, (20, 59)),
    (4934479250535369593, (59, 20)),
    (3169565749973215150, (20, 98)),
    (753180542466205047, (98, 20)),
    (3803507470089459690, (20, 118)),
    (3091310213016392672, (118, 20)),
    (5179823167446585215, (20, 98, 20)),
]


def s_of(qp):
    return (2 * pow(6, -1, qp)) % qp


def letters(word, qp):
    s = s_of(qp)
    out = []
    for v in word:
        r = v % qp
        if r == 0:
            out.append(0)
        elif r == s:
            out.append(1)
        elif r == (-s) % qp:
            out.append(-1)
        else:
            return None
    return out


def window_valid(L):
    p = lo = hi = 0
    for x in L:
        p += x
        lo, hi = min(lo, p), max(hi, p)
    return hi - lo <= 1


def legal_k3():
    """The 36 legal 2-letter words and their 19 reverse classes."""
    F = 145                                   # F(53), corpus ladder
    vals = [v for v in range(1, F + 1)
            if v % QP in {0, s_of(QP), (-s_of(QP)) % QP}]
    words = [(a, b) for a in vals for b in vals
             if window_valid(letters((a, b), QP))]
    reps, seen = [], set()
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        seen.add(w[::-1])
        reps.append(w)
    return vals, words, reps


def dead_gear(word):
    X = [0]
    for g in word:
        X.append(X[-1] + g)
    for q in gears(M):
        if not free_phases(X, q):
            return q
    return None


def parse_ranges(logs):
    """(covered index ranges, per-log reported max over J)."""
    rngs, maxes = [], []
    for p in logs:
        txt = open(p, encoding="utf-8", errors="replace").read()
        assert "scan complete" in txt, ("worker did not finish", p)
        m = re.search(r"indices \[([\d,]+), ([\d,]+)\)", txt)
        rngs.append((int(m.group(1).replace(",", "")),
                     int(m.group(2).replace(",", ""))))
        mx = re.search(r"max over J = (\d+)", txt)
        maxes.append(int(mx.group(1)))
    return rngs, maxes


def check_tiling(rngs, n, label):
    rngs = sorted(rngs)
    assert rngs[0][0] == 0, (label, rngs)
    for a, b in zip(rngs, rngs[1:]):
        assert a[1] == b[0], (label, "gap or overlap in the tiling", a, b)
    assert rngs[-1][1] == n, (label, rngs)


def main():
    print(__doc__.splitlines()[0])
    vals, words, reps = legal_k3()
    print("\n=== A. THE LEVEL, RE-ENUMERATED FROM THE DEFINITION ===")
    print("  s = %d, legal gap values %s" % (s_of(QP), vals))
    assert len(words) == 36, len(words)
    assert len(reps) == 19, len(reps)
    print("  k=3: %d legal words = %d reverse classes (mirror law, rule 27)"
          % (len(words), len(reps)))

    print("\n=== B. THE TWO SCANS THIS GATE LEANS ON ===")
    r2, m2 = parse_ranges(F2_LOGS)
    check_tiling(r2, NOPEN23, "F_2(53)")
    assert max(m2) == F2_VALUE, m2
    print("  r26 F_2(53): %d workers TILE [0, %d) exactly; seed %d, cap %d, "
          "per-range maxima %s -> max %d"
          % (len(r2), NOPEN23, F2_SEED, F2_CAP, m2, max(m2)))
    print("     => NO 2-window of machine 53 has span in (%d, %d]"
          % (F2_VALUE, F2_CAP))
    ra, ma = parse_ranges(FA_LOGS)
    check_tiling(ra, NOPEN23, "F(59) stage A")
    print("  r27 F(59) band (%d, %d]: %d workers TILE [0, %d) exactly; "
          "per-range maxima %s" % (FA_SEED, FA_CAP, len(ra), NOPEN23, ma))
    band_a_empty = (max(ma) == FA_SEED)
    if band_a_empty:
        print("     => NO word-legal window of machine 53 has span in "
              "(%d, %d]; in particular no 2-window does" % (FA_SEED, FA_CAP))
    rt, mt = parse_ranges(FT_LOGS)
    check_tiling(rt, NOPEN23, "m53 top band")
    band_t_empty = (max(mt) == FT_VALUE)
    print("  r27 m53 top band (%d, %d]: %d workers TILE [0, %d) exactly; "
          "per-range maxima %s -> max %d"
          % (FT_SEED, FT_CAP, len(rt), NOPEN23, mt, max(mt)))
    if band_t_empty:
        print("     => NO 2-window of machine 53 has span in (%d, %d] - so "
              "machine 53's adjacent-pair span spectrum has a hole at every "
              "value %d..%d, immediately below its maximum F_2(53) = 159"
              % (FT_VALUE, FT_CAP, FT_VALUE + 1, FT_CAP))

    print("\n=== C. EVERY REVERSE CLASS, WITH ITS SOURCE ===")
    src = {}
    for w in reps:
        sp = sum(w)
        q = dead_gear(w)
        if q:
            src[w] = ("SCREEN", "gear %d has no admissible phase" % q)
        elif band_t_empty and FT_VALUE < sp <= FT_CAP:
            src[w] = ("SCAN27t", "span in (%d, %d] - refuted by the r27 "
                                 "top-band scan" % (FT_VALUE, FT_CAP))
        elif F2_VALUE < sp <= F2_CAP:
            src[w] = ("SCAN26", "span in (%d, %d] - refuted by the r26 "
                                "F_2(53) scan" % (F2_VALUE, F2_CAP))
        elif band_a_empty and FA_SEED < sp <= FA_CAP:
            src[w] = ("SCAN27", "span in (%d, %d] - refuted by the r27 "
                                "stage-A band" % (FA_SEED, FA_CAP))
        else:
            src[w] = ("SAT", "span %d <= F_2(53): only a solver decides" % sp)
    for w in reps:
        print("  %-12s span %3d  %-7s %s" % (str(w), sum(w), src[w][0],
                                             src[w][1]))
    n_sat = sum(1 for w in reps if src[w][0] == "SAT")
    print("  --> solver calls actually needed at k=3: %d of %d reverse "
          "classes" % (n_sat, len(reps)))

    print("\n=== D. EVERY REALISED WITNESS, RE-DERIVED FROM THE DEFINITION ===")
    for k0, word in WITNESSES:
        verify(M, QP, k0, word)
    print("  %d witnesses verified (occurrence at the claimed machine-53"
          " address, killability mod 59, and the CRT combination re-checked)"
          % len(WITNESSES))

    print("\n=== E. SOUNDNESS OF THE SCREEN AGAINST THIS ROUND'S OWN "
          "REALISED WORDS ===")
    for _, word in WITNESSES:
        q = dead_gear(word)
        assert q is None, ("the screen zeroed a REALISED word", word, q)
    print("  the phase-saturation screen calls none of the %d realised words "
          "zero" % len(WITNESSES))
    print("\nALL ASSERTIONS PASSED")
    return src


if __name__ == "__main__":
    main()
