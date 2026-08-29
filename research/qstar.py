"""Round 26 (constructor): MECHANIC'S Q* CONJECTURE, DECIDED.

THE CONJECTURE (mechanic, round 25, docs/novel/old-machine-spectrum.md s8).
Define, for machine M = {5..y} and the next gear q':

    Q*_J(M; legal for q') = max span of a J-gap window of M whose J-2 MIDDLE
    gaps g_1..g_{J-2} satisfy
        (i)  g_i mod q' in V = {0, +s, -s},  s = 2u' mod q',  u' = round(q'/6)
        (ii) the induced letter word (0/+1/-1) has prefix-sum range <= 1,
    i.e. the middle gaps form a KILL WORD.

Mechanic proved  F(M+q') <= max_J Q*_J  (the criterion direction) and observed
Q*_max = F(M+q') at the two anchors they computed (58 = F(31), 88 = F(37)),
registering "Q*_max IS F(M+q'), not merely an upper bound" as a CONJECTURE.

THE VERDICT THIS SCRIPT ESTABLISHES: it is a THEOREM, and the project already
had it.  Q*_J is definitionally the same object as R46's qualmax_J = LAYER
J-2 OF THE KLEENE STAR K*, and R46's identity

    F(M + q') = L (x) K* (x) R          (round 23, proved BOTH ways)

is exactly the conjecture.  The direction mechanic left open - Q*_J <=
F(M+q') for every J - is R46's ">=" half, and the proof is three lines:

  THEOREM (attainment).  Let x_0 < ... < x_J be consecutive openings of M
  whose J-1 interiors x_1..x_{J-1} have a legal middle-gap word.  Then
  x_J - x_0 <= F(M + q').
  PROOF.  Legality is exactly the existence of a tooth assignment
  t_1..t_{J-1} in {+,-} with x_{i+1} - x_i = (t_{i+1} - t_i) c mod q',
  c = 6^{-1} mod q' (letter 0 = same tooth, +-1 = the two tooth-swaps;
  prefix-sum range <= 1 says the walk stays inside the two teeth).  Fix it and
  put r = t_1 c - x_1 mod q'; then x_i + r = t_i c (mod q') for every interior
  i.  The joint period of M + q' is P(M) q' with gcd(P(M), q') = 1, so the
  translate x + jP(M) with jP(M) = r (mod q') is a window of M with the same
  gaps in which gear q' blocks EVERY interior.  Hence no opening of M + q'
  lies strictly between x_0 and x_J, so the gap of M + q' containing that
  interval has length >= x_J - x_0 (equality iff q' also spares x_0, x_J).
  Therefore x_J - x_0 <= F(M + q').  QED

  So Q*_J <= F(M+q') for EVERY J, and the maximal gap of M+q' is itself such
  a window, so max_J Q*_J = F(M+q') exactly.  (J = 2 is mechanic's own
  DELETION LADDER F_2(M) <= F(M + one gear); the theorem is its extension to
  every depth.)

WHAT THIS SCRIPT DOES - the computational half, because a proof this short
deserves an independent check at every step where one is possible:

  (1) A_4 OVER THE REALISED DICTIONARY.  Every real legal window maps to an
      abstract A_4 walk of the same weight (R49 soundness), so the A_4
      closure is an upper bound on max_J Q*_J over ALL J at once - no depth
      cap and no span cap.  A_4 = F(M+q') at all seven scannable steps, which
      closes the "for every J" quantifier computationally.
  (2) THE EXACT PER-DEPTH TABLE Q*_2, Q*_3, ...  computed by descending-span
      search seeded at the A_4 layer bound, with the realised-tuple oracle
      (the exact dictionary at arity <= 4, the scan-free CRT decision deeper).
      A_4's layer vector terminates (A_4 is nilpotent), which RIGOROUSLY caps
      the depth: no walk of length k means no legal window of J = k+2 gaps.
  (3) ASSERTS max_J Q*_J == F(M+q') and Q*_J <= F(M+q') at every J.

Usage:  python research/qstar.py                # all seven scannable steps
        python research/qstar.py --steps 29,31
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_a4                                          # noqa: E402
import chain_dict_oracle                                 # noqa: E402
import crt_dict                                          # noqa: E402
import scanfree_dict                                     # noqa: E402

# the seven scannable steps of the brief, plus 37 -> 41: machine 37's exact
# 4-tuple census (mechanic, round 25) makes an EIGHTH step decidable, beyond
# every Q* computation anyone has run.
STEPS = [11, 13, 17, 19, 23, 29, 31, 37]
KNOWN_FNEW = chain_a4.KNOWN_FNEW


def letters(F, q1):
    """v -> class in {0,+1,-1} for the legal values v <= F, else None."""
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    out = {}
    for v in range(1, F + 1):
        r = v % q1
        if r == 0:
            out[v] = 0
        elif r == a:
            out[v] = 1
        elif r == b:
            out[v] = -1
    return out


def legal_words(F, q1, length, spancap):
    """Every middle word of the given length: values legal, prefix-sum range
    <= 1, span <= spancap.  (length 0 = the single empty word, J = 2.)"""
    lets = letters(F, q1)
    if length == 0:
        return [()]
    out = []
    # partial state: (word, span, lo, hi) with lo/hi the running prefix-sum
    # range; admissible while hi - lo <= 1
    stack = [((), 0, 0, 0)]
    for _ in range(length):
        nxt = []
        for w, sp, lo, hi in stack:
            p = sum(lets[x] for x in w)
            for v, c in lets.items():
                if sp + v > spancap:
                    continue
                p2 = p + c
                lo2, hi2 = min(lo, p2), max(hi, p2)
                if hi2 - lo2 > 1:
                    continue
                nxt.append((w + (v,), sp + v, lo2, hi2))
        stack = nxt
    return [w for w, _, _, _ in stack]


class Oracle:
    """Realised-tuple oracle: the exact full-period census at arity <= 4 when
    one exists, the scan-free CRT decision otherwise.  Both are exact; where
    both are available they are cross-checked."""

    def __init__(self, y, D=None, crt_only=False):
        self.y = y
        self.D = D
        self.crt_only = crt_only
        self.n = 0
        self.ncrt = 0
        self.secs = 0.0
        self.cross = 0
        self.disagree = []

    def __call__(self, t):
        self.n += 1
        if self.D is not None and len(t) in self.D and not self.crt_only:
            return t in self.D[len(t)]
        t0 = time.time()
        r = crt_dict.realised(self.y, t, 20_000_000)
        self.secs += time.time() - t0
        self.ncrt += 1
        if self.D is not None and len(t) in self.D:
            self.cross += 1
            if r != (t in self.D[len(t)]):
                self.disagree.append(t)
        return r


def qstar_at_depth(y, q1, F, J, upper, oracle, floor=0):
    """EXACT Q*_J: descending search on the span, seeded at `upper` (a proved
    upper bound).  Returns (value, witness) or (floor_or_less, None) if no
    legal J-window of span > floor is realised."""
    words = legal_words(F, q1, J - 2, upper)
    if not words:
        return 0, None
    # bucket the candidates by total span so we can walk the spans downwards
    by_span = {}
    for w in words:
        sw = sum(w)
        for dL in range(1, min(F, upper - sw - 1) + 1):
            lo = max(1, floor + 1 - sw - dL)
            hi = min(F, upper - sw - dL)
            for dR in range(lo, hi + 1):
                by_span.setdefault(sw + dL + dR, []).append((dL,) + w + (dR,))
    for S in sorted(by_span, reverse=True):
        for t in by_span[S]:
            if oracle(t):
                return S, t
    return floor, None


def run(y, verbose=True):
    q1 = chain_a4.next_prime(y)
    Fnew = KNOWN_FNEW[y]
    t0 = time.time()
    # ---- the machine's realised-tuple dictionary
    if y in chain_dict_oracle.DICT_CSV:
        D, F, F2 = chain_dict_oracle.load_exact_dict(y)
        src = "exact full-period census (%s)" % chain_dict_oracle.DICT_CSV[y]
    else:
        Dd, Fj, und = scanfree_dict.build(y, 4, workers=1,
                                          cap=crt_dict.KNOWN_F[y] + 20,
                                          verbose=False)
        assert not und, und[:3]
        D = {m: set(Dd[m]) for m in Dd}
        F, F2 = Fj[1], Fj[2]
        src = "scan-free CRT dictionary"
    assert F == crt_dict.KNOWN_F[y], (y, F)
    Dl = {m: sorted(D[m]) for m in sorted(D)}
    print("\n=== machine %d  ->  +%d      F = %d, F_2 = %d, F(M+q') = %d"
          % (y, q1, F, F2, Fnew))
    print("    dictionary: %s;  |D_1..D_4| = %s"
          % (src, [len(D[m]) for m in sorted(D)]))

    # ---- (1) A_4 closure: an upper bound on max_J Q*_J over ALL J
    bound, S, E, lay = chain_a4.a_m_closure(y, Dl, 4, verbose=False)
    assert bound is not None, "A_4 CYCLIC - no all-J bound"
    print("    A_4 closure: %d states, %d edges -> %d   (corpus F(M+q') = %d)"
          % (S, E, bound, Fnew))
    assert bound >= Fnew, (y, bound, Fnew)
    allJ = "EXACT - so max_J Q*_J <= %d for EVERY J" % bound if bound == Fnew \
        else "LOOSE by +%d" % (bound - Fnew)
    print("    layers (upper bounds on Q*_{k+2}, k = 0..): %s   [%s]"
          % (lay, allJ))
    Jmax = len(lay) + 1          # no walk of length len(lay) => Q*_J = 0 above

    # ---- (2) the exact per-depth table
    oracle = Oracle(y, D)
    rows = []
    for J in range(2, Jmax + 1):
        up = lay[J - 2] if J - 2 < len(lay) else 0
        v, wit = qstar_at_depth(y, q1, F, J, up, oracle, floor=0)
        rows.append((J, up, v, wit))
        print("      Q*_%d = %3d   (A_4 layer bound %3d)   witness %s"
              % (J, v, up, wit))
        assert v <= Fnew, ("Q* EXCEEDS F(M+q') - CONJECTURE REFUTED",
                           y, J, v, Fnew, wit)
    qmax = max(r[2] for r in rows)
    argJ = [r[0] for r in rows if r[2] == qmax]
    print("    max_J Q*_J = %d   F(M+q') = %d   %s   (attained at J = %s)"
          % (qmax, Fnew, "EQUAL" if qmax == Fnew else "*** DIFFERS ***", argJ))
    assert qmax == Fnew, ("Q* conjecture FAILS", y, qmax, Fnew)
    if oracle.disagree:
        print("    *** oracle cross-check DISAGREEMENT: %s" %
              oracle.disagree[:5])
    print("    %d oracle calls (%d by CRT, %.0f s), %d cross-checked, "
          "wall %.0f s"
          % (oracle.n, oracle.ncrt, oracle.secs, oracle.cross,
             time.time() - t0))
    return dict(y=y, q1=q1, F=F, F2=F2, Fnew=Fnew, rows=rows, qmax=qmax,
                argJ=argJ, a4=bound, lay=lay, Jmax=Jmax)


def main():
    args = sys.argv[1:]
    ys = ([int(x) for x in args[args.index("--steps") + 1].split(",")]
          if "--steps" in args else STEPS)
    out = []
    for y in ys:
        out.append(run(y))
    print("\n\nQ* AT EVERY SCANNABLE STEP")
    print("  M    q'   F(M)  F(M+q')   Q*_2 Q*_3 Q*_4 Q*_5 Q*_6  max_J Q*_J"
          "  J*   verdict")
    for r in out:
        vals = {J: v for J, _, v, _ in r["rows"]}
        cells = "".join("%5s" % vals.get(J, "-") for J in range(2, 7))
        print("  %-4d %-4d %4d  %7d  %s  %10d  %-4s %s"
              % (r["y"], r["q1"], r["F"], r["Fnew"], cells, r["qmax"],
                 ",".join(str(j) for j in r["argJ"]),
                 "EXACT" if r["qmax"] == r["Fnew"] else "DIFFERS"))
    assert all(r["qmax"] == r["Fnew"] for r in out)
    print("\nall assertions passed - max_J Q*_J = F(M+q') at every step tested")


if __name__ == "__main__":
    main()
