"""hr_twoclass_r30.py -- Holt-Rudd's copy/closure counting in the two-class gear machine.

HARVESTER lane, round 30 follow-on.  Small exact check, single process, seconds.

Holt & Rudd (arXiv:1408.6002) build G(p_{k+1}#) from G(p_k#) by concatenating
p_{k+1} copies and closing gaps at the multiples of p_{k+1}; Theorem 2.3 says
each closure occurs in EXACTLY ONE copy (CRT), and Lemma 3.1 says that for a
constellation of span g < 2 p_{k+1} the j+1 closures land in DISTINCT copies.

Two-class translation (anchor-235.md 9d/9f: the g copies of the lower period
realise every deletion phase once - phase_bijective).  Machine M = gears 5..y,
period P, openings O (N of them, cyclic).  New gear q' with teeth
T = {u', -u'}, u' = 6^{-1} mod q'.  Copy i (i = 0..q'-1) holds the openings
o + iP, and in copy i the opening o is killed iff o + iP in T, i.e. copy i IS
the phase r_i = iP mod q' (a bijection, P being a unit mod q').  Claims:

  A  each opening is killed in EXACTLY TWO copies (two-class Theorem 2.3);
  B  for a window of j+1 consecutive openings with offsets X, the number of
     copies sparing the whole window is EXACTLY q' - |X u (X+s)| (mod q'),
     s = 2u' mod q' -- Holt-Rudd's count with the coincidences made explicit;
  C  if span(window) < s_min(q') = min(s, q'-s) then |X u (X+s)| = 2(j+1):
     all 2(j+1) killing copies are DISTINCT (two-class Lemma 3.1, threshold
     s_min instead of 2p_{k+1}); and the threshold is SHARP -- the smallest
     span at which two points of one window die in one copy equals the
     smallest realised legal letter;
  D  the number of copies in which a run of k >= 2 consecutive openings dies
     ENTIRELY is 1 (legal word with a literal letter) or 2 (legal, all letters
     padded) -- so the counting bounds the MULTIPLICITY of a chain per
     instance and takes realisation as INPUT; it never bounds the chain length;
  E  the longest run of consecutive kills in the concatenation of the q'
     copies equals the recorded A_kill (= L + 1 = D_q'), read off the same data.
"""
from __future__ import annotations

import sys
import numpy as np

REC_AKILL = {11: 2, 13: 2, 17: 2, 19: 3, 23: 2}   # even-j-mechanism.md 1.4(a): A_kill = L+1


def primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(range(i * i, n + 1, i)))
    return [i for i in range(2, n + 1) if s[i]]


def machine(y):
    gs = [q for q in primes_upto(y) if q >= 5]
    P = 1
    for q in gs:
        P *= q
    k = np.arange(P, dtype=np.int64)
    open_ = np.ones(P, dtype=bool)
    for q in gs:
        u = pow(6, -1, q)
        open_ &= ((k % q) != u) & ((k % q) != (q - u) % q)
    O = np.nonzero(open_)[0]
    return gs, P, O


def next_prime(y):
    p = y + 1
    while True:
        if all(p % d for d in range(2, int(p ** 0.5) + 1)):
            return p
        p += 1


def legal_word_class(gaps, q, s):
    """0 illegal; 1 legal with a literal letter; 2 legal all padded."""
    cls = []
    for g in gaps:
        m = g % q
        if m == 0:
            cls.append(0)
        elif m == s:
            cls.append(1)
        elif m == (q - s) % q:
            cls.append(-1)
        else:
            return 0
    nz = [c for c in cls if c]
    for a, b in zip(nz, nz[1:]):
        if a == b:
            return 0            # T3: nonzero classes must alternate
    return 2 if not nz else 1


def check(y, full_windows=True):
    gs, P, O = machine(y)
    N = len(O)
    q = next_prime(y)
    u = pow(6, -1, q)
    s = (2 * u) % q
    smin = min(s, q - s)
    KMAX = 8
    Oc = np.concatenate([O, O[:KMAX + 2] + P])          # cyclic extension (next copy)
    res_c = (Oc % q).astype(np.int64)
    gaps = np.diff(np.concatenate([O, [O[0] + P]]))       # cyclic gaps, N of them
    print(f"\nmachine {y}: gears {gs}, P={P}, N={N}, q'={q}, u'={u}, s=2u'={s}, s_min={smin}")

    def kill_phase(r):
        """killed flags over the extended opening list Oc, in phase r"""
        return (res_c == (u - r) % q) | (res_c == (q - u - r) % q)

    # ---- A: each opening dies in exactly two phases (= two copies) ---------
    kills = np.zeros(N, dtype=np.int64)
    KR = [kill_phase(r) for r in range(q)]
    for r in range(q):
        kills += KR[r][:N]
    assert np.all(kills == 2), "A FAILED"
    print(f"  A  every opening killed in exactly 2 of the {q} copies   OK   (total kills {2*N} = 2N)")

    # ---- B, C: per-window sparing counts vs the forbidden-set formula -------
    nB = 0; nC = 0; first_coinc = None
    rng = np.random.default_rng(y)
    for j in range(1, KMAX):                       # window = j gaps, j+1 points
        if full_windows and N * q <= 3_000_000:
            idx = np.arange(N)
        else:
            idx = rng.choice(N, size=min(N, 20000), replace=False)
        spared = np.zeros(len(idx), dtype=np.int64)
        for r in range(q):
            hit = np.zeros(len(idx), dtype=bool)
            for t in range(j + 1):
                hit |= KR[r][idx + t]
            spared += ~hit
        for a, t0 in enumerate(idx):
            X = [int(Oc[t0 + t] - Oc[t0]) % q for t in range(j + 1)]
            forb = set(X) | set((x + s) % q for x in X)
            assert spared[a] == q - len(forb), ("B FAILED", y, j, int(t0), int(spared[a]), len(forb))
            nB += 1
            span = int(Oc[t0 + j] - Oc[t0])
            if span < smin:
                assert len(forb) == 2 * (j + 1), ("C FAILED", y, j, int(t0), span)
                nC += 1
            elif len(forb) < 2 * (j + 1):
                if first_coinc is None or span < first_coinc:
                    first_coinc = span
    realised = set(int(g) for g in gaps)
    legal_letters = sorted(g for g in realised if g % q in (0, s, (q - s) % q))
    print(f"  B  #sparing copies = q' - |X u (X+s)|  on {nB} windows, j <= {KMAX-1}   OK")
    print(f"  C  span < s_min = {smin}  =>  all 2(j+1) killing copies distinct, {nC} windows   OK")
    print(f"     first coincidence (two points of one window dying in one copy) at span {first_coinc};"
          f" smallest realised legal letter = {legal_letters[0]}   "
          f"{'OK' if first_coinc == legal_letters[0] else 'MISMATCH'}")
    assert first_coinc == legal_letters[0]

    # ---- E: runs of consecutive kills in the CONCATENATION of the q' copies -
    # copy i is phase r_i = iP mod q'; runs may cross copy seams, so carry them.
    runs = []                      # (global start, length) in the q'N-long cycle
    carry_start = None; carry_len = 0
    for i in range(q):
        r = (i * P) % q
        kr = KR[r][:N]
        d = np.diff(np.concatenate([[0], kr.astype(np.int8), [0]]))
        starts = np.nonzero(d == 1)[0]; ends = np.nonzero(d == -1)[0]
        for a, b in zip(starts, ends):
            a = int(a); b = int(b)
            if a == 0 and carry_len:
                gstart, length = carry_start, carry_len + (b - a)
            else:
                gstart, length = i * N + a, b - a
            if b == N:
                carry_start, carry_len = gstart, length
            else:
                runs.append((gstart, length))
                carry_start, carry_len = None, 0
    if carry_len:
        if runs and runs[0][0] == 0:
            g0, l0 = runs[0]
            runs[0] = (carry_start, carry_len + l0)
        else:
            runs.append((carry_start, carry_len))
    longest = max(l for _, l in runs)
    print(f"  E  longest run of consecutive kills in the {q}-copy concatenation = {longest} = A_kill = D_q';"
          f" recorded {REC_AKILL[y]}   {'OK' if longest == REC_AKILL[y] else 'MISMATCH'}")
    assert longest == REC_AKILL[y]

    # ---- D: multiplicity of a full k-run, over copies, is 1 or 2 -------------
    byk = {}
    for gstart, k in runs:
        if k >= 2:
            byk.setdefault(k, set()).add(gstart % N)
    nD = 0; hist = {}
    for k, t0s in sorted(byk.items()):
        t0 = np.array(sorted(t0s), dtype=np.int64)
        cnt = np.zeros(len(t0), dtype=np.int64)
        for r in range(q):
            allk = np.ones(len(t0), dtype=bool)
            for t in range(k):
                allk &= KR[r][t0 + t]
            cnt += allk
        for a, tt in enumerate(t0):
            wg = [int(Oc[tt + t + 1] - Oc[tt + t]) for t in range(k - 1)]
            cls = legal_word_class(wg, q, s)
            assert cls in (1, 2) and int(cnt[a]) == cls, ("D FAILED", y, int(tt), k, wg, cls, int(cnt[a]))
            nD += 1
            hist[(k, cls)] = hist.get((k, cls), 0) + 1
    print(f"  D  {nD} distinct maximal runs of >= 2 kills: #copies carrying the run = 1 (literal word) / 2 (all padded)"
          f"   OK   histogram (k, mult): {dict(sorted(hist.items()))}")
    F = int(gaps.max())
    print(f"     F(M) = {F} >= s_min = {smin}: the Holt-Rudd threshold excludes every window carrying F   "
          f"{'OK' if F >= smin else 'note: F < s_min'}")


if __name__ == "__main__":
    ys = [int(a) for a in sys.argv[1:]] or [11, 13, 17, 19, 23]
    for y in ys:
        check(y)
    print("\nhr_twoclass_r30: ALL ASSERTIONS GREEN")
