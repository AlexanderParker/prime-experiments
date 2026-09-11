"""ol2_core.py -- shared machinery for the order law beyond 41 (round 71).

The covering-problem instrument itself is r70/ol_pattern.py, imported UNCHANGED, so every
membership verdict here is produced by the operator that was gated at m23, m29 and m37 in
research/proof/order_law_37_41.md.  This module adds only:

  * gear lists and letter alphabets for the engines 41, 43, 47, 53;
  * the certified record table F(q) (never guessed -- every value is quoted from the corpus);
  * a memoised, process-parallel wrapper round the membership decision;
  * the word enumerators (level-2 admissible fusing J-words) used by every step.

Usage: imported by ol2_dict.py, ol2_step.py, ol2_gate.py.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R70 = os.path.abspath(os.path.join(HERE, "..", "r70"))
sys.path.insert(0, R70)

from ol_pattern import realised_word, realised_word_search  # noqa: E402

OUT = os.path.join(HERE, "results")

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]

# The certified record table.  F(y) = longest opening-free stretch of the machine {5..y}.
# Corpus: docs/proof-search/agents-shared.md line 40 and mechanic.md 3060 --
#   F = 2, 5, 7, 11, 18, 25, 34, 43, 58, 88, 91, 103, 118, 145, 161 at y = 5..59.
# Every one of these is EXACT (the ladder is complete to 53; F(59) = 161 was computed on
# machine 23's period).  Nothing here is a bound.
F_RECORD = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58,
            37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}

# F_2(M), the widest stretch of M carrying one interior opening, where the corpus has it EXACTLY:
#   F_2(41) = 103 (alignment-rules.md 3.7 m41 row), F_2(47) = 134 (agents-shared.md, "F_2(47) =
#   134 EXACT - a first computation, superseding the standing range [119,141]"),
#   F_2(53) = 159 (alignment-rules.md, the lap-phase transfer vehicle).
# Where only the deletion-ladder cap is on record (m43), the cap F_2(M) <= F(M + q') is used and
# the fact that it is a BOUND is carried through every statement that depends on it.
F2_RECORD = {37: 90, 41: 103, 47: 134, 53: 159}
F2_CAP_ONLY = {43: 118}     # F_2(43) <= F(47) = 118, deletion ladder; not known exactly

# L(M) with respect to the NEXT gear, from monotone_functional.md 3.5 / M5:
#   L = 1, 1, 1, 2, 1, 3, 3, 2, 2 at m11 .. m41.
L_RECORD = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2}


def _ladder_cap(y, j):
    """The certified deletion-ladder cap on F_j(M): F_j(M) <= F(M + the next j-1 primes)
    (docs/proofs/07-deletion-ladder.md, E17, proved)."""
    i = PRIMES.index(y)
    k = i + (j - 1)
    if j == 2 and y in F2_RECORD:
        return F2_RECORD[y]                       # the exact value where the corpus has it
    if k < len(PRIMES) and PRIMES[k] in F_RECORD:
        return F_RECORD[PRIMES[k]]
    return None


def fj_cap(y, j):
    """A certified cap on F_j(M), the widest realised j-window.  Two sources, both exact:

      (i)  the deletion ladder, F_j(M) <= F(M + the next j-1 primes);
      (ii) SUBADDITIVITY: if (g_1 .. g_j) is realised then so are (g_1 .. g_a) and
           (g_{a+1} .. g_j), so F_j(M) <= F_a(M) + F_b(M) for every a + b = j.

    (ii) matters where the record table runs out: F(61) is not on record, so the ladder gives m47
    nothing beyond j = 3 and m53 nothing beyond j = 2, while subadditivity still caps every j.
    A j-window wider than the cap is NOT realised and needs no covering problem at all.
    Returns None only if no cap is available."""
    best = {}
    for m in range(1, j + 1):
        c = _ladder_cap(y, m)
        for a in range(1, m // 2 + 1):
            ca, cb = best.get(a), best.get(m - a)
            if ca is not None and cb is not None:
                c = ca + cb if c is None else min(c, ca + cb)
        best[m] = c
    return best[j]


def gears_upto(y):
    return [p for p in PRIMES if p <= y]


def next_gear(y):
    i = PRIMES.index(y)
    return PRIMES[i + 1]


def u_of(g):
    return pow(6, -1, g)


def d_of(q):
    return (2 * u_of(q)) % q


def letter_class(v, q):
    """0 = PAD, 1 = UP (+d), 2 = DOWN (-d), 3 = BAD."""
    d = d_of(q)
    r = v % q
    if r == 0:
        return 0
    if r == d:
        return 1
    if r == (-d) % q:
        return 2
    return 3


# ------------------------------------------------------------------ membership, memoised

_MEMO = {}
_POOL = None
_SOFT_FAIL = set()      # words whose verdict costs more than the soft budget: never re-attempted
                        # softly (the same hard window recurs in hundreds of candidate words),
                        # always still available to the full solver
_STATS = {"calls": 0, "nodes": 0, "fallback": 0, "secs": 0.0}


def decide(word, gears, budget=40_000_000):
    """Exact: is the tuple `word` of consecutive gap sizes a realised window of the machine?"""
    key = tuple(int(x) for x in word)
    hit = _MEMO.get(key)
    if hit is not None:
        return hit
    t0 = time.time()
    c = {}
    try:
        v = bool(realised_word_search(key, gears, budget=budget, counter=c))
    except RuntimeError:
        _STATS["fallback"] += 1
        v = bool(realised_word(key, gears))
    _MEMO[key] = v
    _STATS["calls"] += 1
    _STATS["nodes"] += c.get("nodes", 0)
    _STATS["secs"] += time.time() - t0
    return v


def load_memo(path):
    """Verdicts are expensive and permanent facts about the gears -- keep them across runs."""
    import json
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        for k, v in json.load(f).items():
            _MEMO[tuple(int(x) for x in k.split(","))] = bool(v)
    return len(_MEMO)


def save_memo(path):
    import json
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({",".join(map(str, k)): int(v) for k, v in _MEMO.items() if len(k) >= 2}, f)
    os.replace(tmp, path)
    return len(_MEMO)


def stats():
    return dict(_STATS)


def memo():
    return _MEMO


# ------------------------------------------------------------------ parallel batch decision

def _work(job):
    """Decide one word.  The search solver first; if it exhausts its node budget the budget is
    raised twentyfold, and only then is the meet-in-the-middle reference solver called (which can
    itself run out of room at 13 or 14 gears).  A word that survives all three is reported
    UNDECIDED rather than guessed - no verdict here is ever a default."""
    word, gears, budget, soft = job
    t0 = time.time()
    c = {}
    fb = 0
    try:
        v = bool(realised_word_search(tuple(word), gears, budget=budget, counter=c))
    except RuntimeError:
        if soft:
            # SOFT pass: a verdict that costs more than `budget` search nodes is deferred, not
            # guessed.  The caller resolves the deferrals with the full solver, and only for the
            # words that are still candidates -- a word with one cheap NO among its subwindows is
            # already rejected and its expensive subwindows are never proved.
            return tuple(word), None, c.get("nodes", 0), 0, time.time() - t0
        fb = 1
        try:
            v = bool(realised_word_search(tuple(word), gears, budget=20 * budget, counter=c))
        except RuntimeError:
            fb = 2
            try:
                # small bcap: the meet-in-the-middle reference is the memory hog and several
                # of these run at once
                v = bool(realised_word(tuple(word), gears, bcap=1_500_000))
            except (MemoryError, ValueError):
                fb = 3
                v = None
    return tuple(word), v, c.get("nodes", 0), fb, time.time() - t0


def decide_many(words, gears, procs=8, budget=40_000_000, chunk=1, log=None, soft=False):
    """Decide a list of words in parallel.  Returns {word: True/False/None} plus a stats dict.
    With `soft`, a word whose verdict costs more than `budget` search nodes comes back None."""
    import multiprocessing as mp
    words = [tuple(int(x) for x in w) for w in words]
    # longest spans first: with chunk = 1 and dynamic scheduling that keeps the tail short
    todo = sorted((w for w in dict.fromkeys(words)
                   if _MEMO.get(w, "?") == "?" and not (soft and w in _SOFT_FAIL)),
                  key=sum, reverse=True)
    st = {"calls": 0, "nodes": 0, "fallback": 0, "undecided": 0, "secs": 0.0, "wall": 0.0}
    if todo:
        t0 = time.time()
        jobs = [(w, gears, budget, soft) for w in todo]
        if len(todo) <= 2:
            pool = None
        else:
            global _POOL
            if _POOL is None or _POOL[0] != procs:
                if _POOL is not None:
                    _POOL[1].terminate()
                # ONE pool for the whole run: on Windows every mp.Pool() respawns interpreters
                # (a few seconds each), and the scans call this function twice per span level, so
                # a per-call pool leaves the workers idle most of the wall clock.
                _POOL = (procs, mp.Pool(procs))
            pool = _POOL[1]
        if pool is None:
            it = (_work(j) for j in jobs)
        else:
            it = pool.imap_unordered(_work, jobs, chunksize=chunk)
        if True:
            done = 0
            for w, v, nodes, fb, secs in it:
                if v is not None:
                    _MEMO[w] = v            # a deferred verdict is never cached as a verdict
                    _SOFT_FAIL.discard(w)
                else:
                    _SOFT_FAIL.add(w)
                st["calls"] += 1
                st["nodes"] += nodes
                st["fallback"] += 1 if fb else 0
                st["undecided"] += 1 if v is None else 0
                st["secs"] += secs
                done += 1
                if log and done % max(1, len(jobs) // 20) == 0:
                    print(f"    {done:,}/{len(jobs):,} decided, {time.time()-t0:.0f}s",
                          flush=True)
        st["wall"] = time.time() - t0
    return {w: _MEMO.get(w) for w in words}, st


# ------------------------------------------------------------------ word enumeration

def enumerate_fusing(J, vals, pairs, q):
    """Every J-word over the realised gap values `vals` whose adjacent pairs all lie in `pairs`
    (level-2 admissible) and which FUSES at some phase z of q': offsets o_1..o_{J-1} struck,
    o_0 and o_J unstruck.  Returns {word: one phase that works}."""
    d = d_of(q)
    struck = {0, d}
    out = {}
    for z in range(q):
        if z in struck:
            continue
        stack = [((), 0)]
        for i in range(J):
            nxt = []
            for w, off in stack:
                last = w[-1] if w else None
                for g in vals:
                    if last is not None and (last, g) not in pairs:
                        continue
                    o = off + g
                    hit = (o + z) % q in struck
                    if i < J - 1:
                        if hit:
                            nxt.append((w + (g,), o))
                    else:
                        if not hit:
                            nxt.append((w + (g,), o))
            stack = nxt
            if not stack:
                break
        for w, off in stack:
            out.setdefault(w, z)
    return out


def dp_relaxed(vals, pairs, q, J, k, Dk_member):
    """Kept deliberately simple: for k = 1 and k = 2 the level-k relaxation is a DP; higher k is
    done by explicit enumeration in ol2_step.py.  `Dk_member` unused for k <= 2."""
    raise NotImplementedError
