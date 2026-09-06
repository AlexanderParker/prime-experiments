"""pc_core.py -- shared helpers for node 4.i.b.ii.a (CAP THE PADDED WORD).

Everything is exact integer arithmetic.  The closure step (for the ladder m29, m31, m37) is
imported unchanged from research/anchor235/r61/lc_core.py, the instrument of ladder_closure.md and
monotone_functional.md.

Vocabulary (research/proof/pad_cap.md 0.1): for the incoming gear q', u = 6^{-1} mod q', d = 2u,
a = min(d, q'-d), b = q' - a.  A gap value v is a legal letter iff v mod q' in {0, d, -d}; classes
PAD (0), UP (+d), DOWN (-d); bare letters a, b (< q'); padded letters = legal values >= q'; the
small alphabet {a, b, q'}; skip letters = padded letters other than q'.
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R61 = os.path.abspath(os.path.join(HERE, "..", "r61"))
if R61 not in sys.path:
    sys.path.insert(0, R61)
from lc_core import PRIMES, base_gaps, closure_step, dict_from_gaps, sieve_machine, u_of  # noqa

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

E35 = [0, 2, 3, 5, 7, 10, 12, 17, 18, 23, 25, 28, 30, 32, 33]
CORPUS_L = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2, 43: 2, 47: 4, 53: 3}
CORPUS_LPAD = {11: 0, 13: 0, 17: 0, 19: 1, 23: 1, 29: 1, 31: 2, 37: 2, 41: 2, 43: 2, 47: 3, 53: 3}
CORPUS_F = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
            43: 103, 47: 118, 53: 145, 59: 161}


def next_gear(y):
    return PRIMES[PRIMES.index(y) + 1]


def gears_of(y):
    return [p for p in PRIMES if p <= y]


def letter_data(q):
    u = u_of(q)
    d = (2 * u) % q
    a = min(d, q - d)
    return {"q": q, "u": u, "d": d, "a": a, "b": q - a}


def teeth(g):
    u = u_of(g)
    return (u % g, (-u) % g)


def strikes(g, X):
    """Boolean: gear g strikes column X (scalar or array)."""
    t1, t2 = teeth(g)
    r = np.asarray(X) % g
    return (r == t1) | (r == t2)


def letters_of(vals, q):
    """0 PAD, 1 UP (+d), 2 DOWN (-d), 3 BAD."""
    d = (2 * u_of(q)) % q
    vals = np.asarray(vals, dtype=np.int64)
    r = vals % q
    lt = np.full(vals.shape, 3, dtype=np.uint8)
    lt[r == 0] = 0
    lt[r == d] = 1
    lt[r == (-d) % q] = 2
    return lt


def is_legal_word(vals, q):
    lt = letters_of(vals, q)
    if (lt == 3).any():
        return False
    last = 0
    for c in lt:
        if c != 0:
            if c == last:
                return False
            last = c
    return True


def word_kind(vals, q):
    """'bare' (only a, b), 'small' (a, b, q' with at least one q'), 'skip' (some other letter)."""
    L = letter_data(q)
    s = set(int(v) for v in vals)
    if s <= {L["a"], L["b"]}:
        return "bare"
    if s <= {L["a"], L["b"], q}:
        return "small"
    return "skip"


# ------------------------------------------------------------------ maximal legal runs

def maximal_legal_words(gaps_ext, q, n_valid, min_len=1):
    """All maximal legal words of the (extended) gap array whose START index is < n_valid.
    gaps_ext = gaps followed by enough of a lookahead so no word starting below n_valid is cut.
    Returns a list of (start, tuple(word)).  A maximal legal word: a run of consecutive legal
    letters with no two consecutive nonzero letters equal, that cannot be extended either way."""
    lt = letters_of(gaps_ext, q)
    legal = lt != 3
    n = gaps_ext.size
    out = []
    # runs of consecutive legal letters
    idx = np.flatnonzero(legal)
    if idx.size == 0:
        return out
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])  # inclusive
    for s0, e0 in zip(starts, ends):
        if s0 >= n_valid:
            break
        # within a run of legal letters, split at consecutive equal nonzero classes.
        # maximal legal words: from each candidate start (run start or the position after an
        # equality break) to the next equality break.
        cls = lt[s0:e0 + 1]
        # positions of equality breaks: i such that cls[i] != 0 and the previous nonzero == cls[i]
        last = 0
        last_pos = -1
        brk = []  # (last_pos, i): word cannot contain both last_pos and i
        for i, c in enumerate(cls):
            if c != 0:
                if c == last:
                    brk.append((last_pos, i))
                last = c
                last_pos = i
        # maximal words: segments [w_start, w_end] with w_start = 0 or (prev break's first
        # index + 1), w_end = (next break's second index - 1) or len - 1
        seg_starts = [0] + [p + 1 for p, _ in brk]
        seg_ends = [i - 1 for _, i in brk] + [len(cls) - 1]
        for ws, we in zip(seg_starts, seg_ends):
            if we - ws + 1 >= min_len and s0 + ws < n_valid:
                out.append((int(s0 + ws), tuple(int(v) for v in gaps_ext[s0 + ws:s0 + we + 1])))
    return out


def subwords(words_counter, k):
    """Set of realised legal k-words from a Counter of maximal legal words."""
    s = set()
    for w in words_counter:
        for i in range(len(w) - k + 1):
            s.add(w[i:i + k])
    return s


def longest_admissible(realised_k, q, k):
    """L^{(k)}: the longest legal word all of whose k-subwords are in realised_k (a set of
    k-tuples).  Returns (length, cyclic) with length = None when cyclic (infinite)."""
    nodes = list(realised_k)
    if not nodes:
        return 0, False
    index = {w: i for i, w in enumerate(nodes)}
    # edges w -> w' iff w[1:] == w'[:-1] and w + (w'[-1],) legal
    by_prefix = defaultdict(list)
    for w in nodes:
        by_prefix[w[:-1]].append(w)
    adj = [[] for _ in nodes]
    for w in nodes:
        for w2 in by_prefix.get(w[1:], []):
            if is_legal_word(w + (w2[-1],), q):
                adj[index[w]].append(index[w2])
    # longest path in edges, with cycle detection (iterative DFS)
    n = len(nodes)
    state = [0] * n  # 0 unvisited, 1 on stack, 2 done
    best = [0] * n
    sys.setrecursionlimit(max(10000, 4 * n + 100))

    def dfs(i):
        state[i] = 1
        b = 0
        for j in adj[i]:
            if state[j] == 1:
                raise ValueError("cycle")
            if state[j] == 0:
                dfs(j)
            b = max(b, best[j] + 1)
        best[i] = b
        state[i] = 2

    try:
        for i in range(n):
            if state[i] == 0:
                dfs(i)
    except ValueError:
        return None, True
    return k + max(best), False


def k_L_table(words_counter, q, L):
    """L^{(k)} for k = 1..L+1 and k_L."""
    rows = {}
    kL = None
    for k in range(1, L + 2):
        rk = subwords(words_counter, k)
        val, cyc = longest_admissible(rk, q, k)
        if val is not None and val < L:
            val = L  # words shorter than k are realised anyway
        rows[k] = "inf" if cyc else val
        if kL is None and not cyc and val == L:
            kL = k
    return rows, kL


# ------------------------------------------------------------------ the skeleton (pad_cap.md 0.1)

def skeleton(word, q):
    """Multiplier sets (S_0, S_1), the class-1 residue s, the multiplier sequence, runs and
    skips.  word = tuple of gap values (a realised legal word)."""
    d = (2 * u_of(q)) % q
    offs = [0]
    for v in word:
        offs.append(offs[-1] + v)
    s = None
    for o in offs:
        if o % q != 0:
            s = o % q
            break
    S0, S1, seq = [], [], []
    for o in offs:
        r = o % q
        if r == 0:
            S0.append(o // q)
            seq.append((0, o // q))
        else:
            assert r == s, (word, q, o, s)
            S1.append((o - s) // q)
            seq.append((1, (o - s) // q))
    def runs(S):
        if not S:
            return [], 0
        rr, cur = [], [S[0]]
        for m in S[1:]:
            if m == cur[-1] + 1:
                cur.append(m)
            else:
                rr.append(cur)
                cur = [m]
        rr.append(cur)
        skips = (S[-1] - S[0] + 1) - len(S)
        return rr, skips
    r0, sk0 = runs(S0)
    r1, sk1 = runs(S1)
    jumps = [seq[i + 1][1] - seq[i][1] for i in range(len(seq) - 1)]
    return {"s": s, "S0": S0, "S1": S1, "seq": seq, "runs0": [len(r) for r in r0],
            "runs1": [len(r) for r in r1], "skips0": sk0, "skips1": sk1, "jumps": jumps,
            "max_run": max([len(r) for r in r0 + r1] or [0]), "m_max": max(S0 + S1)}


# ------------------------------------------------------------------ junctions and the counting cap

def capacity(g, Lrun):
    """max_g(L): the most columns of a run of L that gear g can strike (docs/proofs/20 Lemma 2)."""
    d = (2 * u_of(g)) % g
    ag = min(d, g - d)
    r = Lrun % g
    e = 2 if r > ag else (1 if r >= 1 else 0)
    return 2 * (Lrun // g) + e


def junction_analysis(X0, word, gears):
    """For a realised legal word starting at global column X0 (all arithmetic on X, no period):
    per opening: gears striking X-1 and X+1; per gap: the gears striking each interior column,
    the sole strikers, the needed set; shared needed gears between consecutive gaps; residues."""
    offs = [0]
    for v in word:
        offs.append(offs[-1] + v)
    X = [X0 + o for o in offs]
    per_open = []
    for x in X:
        assert not any(strikes(g, x) for g in gears), ("junction not open", x)
        per_open.append({"X": int(x), "left": [g for g in gears if strikes(g, x - 1)],
                         "right": [g for g in gears if strikes(g, x + 1)],
                         "mod5": int(x % 5), "mod7": int(x % 7), "mod35": int(x % 35)})
    per_gap = []
    for i, v in enumerate(word):
        inter = np.arange(X[i] + 1, X[i + 1])
        hits = np.zeros((len(gears), inter.size), dtype=bool)
        for gi, g in enumerate(gears):
            hits[gi] = strikes(g, inter)
        cnt = hits.sum(axis=0)
        assert (cnt >= 1).all(), ("interior column open", i)
        sole = [gears[gi] for gi in range(len(gears)) if (hits[gi] & (cnt == 1)).any()]
        used = [gears[gi] for gi in range(len(gears)) if hits[gi].any()]
        per_gap.append({"v": int(v), "needed": sole, "striking": used,
                        "strikes_by_gear": {int(gears[gi]): int(hits[gi].sum())
                                            for gi in range(len(gears)) if hits[gi].any()},
                        "gear5_strikes": int(hits[gears.index(5)].sum()) if 5 in gears else None,
                        "gear5_cap": capacity(5, int(v) - 1) if 5 in gears else None})
    shared = []
    for i in range(len(word) - 1):
        shared.append(sorted(set(per_gap[i]["needed"]) & set(per_gap[i + 1]["needed"])))
    S = offs[-1]
    Lw = len(word)
    cap_sum = sum(capacity(g, S + 1) for g in gears)
    neigh = sorted(set([x - 1 for x in X] + [x + 1 for x in X]))
    return {"X0": int(X0), "word": [int(v) for v in word], "span": int(S),
            "openings": per_open, "gaps": per_gap, "shared_needed": shared,
            "counting": {"need_interior": int(S - Lw), "capacity": int(cap_sum),
                         "ratio": round(cap_sum / max(1, S - Lw), 3),
                         "junction_need": len(neigh),
                         "junction_ratio": round(cap_sum / len(neigh), 3),
                         "gears_above_span": [g for g in gears if g > S + 1]}}


# ------------------------------------------------------------------ CORRCAP_3 per class

def a_of_class(c):
    return (c - 1) // 3 if c % 6 == 1 else (c + 1) // 3


def exposed_set(gears):
    """Residues mod prod(gears) open under all the gears (E_35 for {5,7})."""
    P = 1
    for g in gears:
        P *= g
    r = np.arange(P)
    ok = np.ones(P, dtype=bool)
    for g in gears:
        ok &= ~strikes(g, r)
    return P, np.flatnonzero(ok)


def corrcap(letters, classes, gears):
    """Longest legal word over the given letters (list of (value, class) with class 0 pad,
    1, 2 nonzero) whose prefix-sum walk stays in the exposed set of `gears` from some start
    residue.  Returns (length or None if infinite, witness word or None).
    State: (residue, last nonzero class).  Longest path with cycle detection."""
    P, E = exposed_set(gears)
    Eset = set(int(e) for e in E)
    nodes = [(r, last) for r in Eset for last in (0, 1, 2)]
    index = {nd: i for i, nd in enumerate(nodes)}
    adj = [[] for _ in nodes]
    for (r, last), i in index.items():
        for v, c in zip(letters, classes):
            if c != 0 and c == last:
                continue
            r2 = (r + v) % P
            if r2 not in Eset:
                continue
            j = index[(r2, last if c == 0 else c)]
            adj[i].append((j, v))
    n = len(nodes)
    state = [0] * n
    best = [0] * n
    nxt = [None] * n
    sys.setrecursionlimit(max(10000, 4 * n + 100))

    def dfs(i):
        state[i] = 1
        b, bn = 0, None
        for j, v in adj[i]:
            if state[j] == 1:
                raise ValueError("cycle")
            if state[j] == 0:
                dfs(j)
            if best[j] + 1 > b:
                b, bn = best[j] + 1, (j, v)
        best[i], nxt[i] = b, bn
        state[i] = 2

    try:
        for i in range(n):
            if state[i] == 0:
                dfs(i)
    except ValueError:
        return None, None
    starts = [index[(r, 0)] for r in Eset]
    i0 = max(starts, key=lambda i: best[i])
    w, i = [], i0
    while nxt[i] is not None:
        j, v = nxt[i]
        w.append(int(v))
        i = j
    return best[i0], w


def corrcap3(c, gears=(5, 7)):
    """CORRCAP_3(c): the small alphabet {a_c, b_c, c} at class c (c coprime to prod gears*6)."""
    a = a_of_class(c)
    b = c - a
    return corrcap([a, b, c], [1, 2, 0], list(gears))


def psord(c, gears=(5, 7)):
    a = a_of_class(c)
    b = c - a
    return corrcap([a, b], [1, 2], list(gears))
