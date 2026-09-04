"""Round 30 (mechanic), probes (a)-word-vehicle and (b): THE KILLER PROFILE OF
WORD EXTENSIONS, and the word-level ladder V3 / V4.

L(M) is the longest REALISED legal word (letters = gap values <= F(M) whose
residue mod q' is 0 or +-d, d = 2*6^{-1} mod q'; nonzero classes strictly
alternate).  Every one-letter extension of a length-L(M) word is therefore
REFUTED.  This file asks WHICH GEAR refutes it.

Two attributions, both exact:

  SAT(w)  = the set of gears g of M whose single-gear FREE set is empty -
            FREE_g(X) = Z_g minus (X mod g) minus (X - s_g mod g) - the
            phase-saturation screen (Mechanic r26).  |FREE_g| >= g - 2|X|, so
            a single gear can only saturate when g < 2|X|: the corridor gears.
  y*(w)   = the OPEN-CONSTRAINT KILLER PREFIX.  In the realisability CSP every
            gear has two roles: its phase must keep the L+2 points of X open
            (the OPEN half) and the phases must jointly block every interior
            point (the COVER half).  R(S) = the CSP in which only the gears in
            S carry the open constraint and every other gear's phase is FREE
            (it may sit anywhere, and still helps cover).  R(S) only gets
            harder as S grows, so y* = min{ y' : R({g <= y'}) infeasible } is
            well defined and monotone-searchable; y* <= min SAT(w) when SAT is
            non-empty, and y* = M means the word dies only when the TOP
            gear's open constraint is imposed - the cover half needs every
            gear.  "Killers among 5, 7, 11" <=> y* <= 11.

Realised length-L words are grown level by level from the legal alphabet by
the overlap lemma (both (n-1)-subwords of a realised n-word are realised),
each candidate decided by crt_dict.decide_cover; at machine 47 the realised
length-4 words are taken from Constructor's round-29 exhaustive decision
((18,35,18,35) and its reverse - the only survivors of 40 T3-legal candidates
under the exact caps F_1..F_3 and phase saturation) because arity-2 decisions
at machine 47 cost minutes each and are not the object here.

Mode 'words' (probe (a), V3/V4): for a list of primes g > M, the longest
T3-legal word over the legal alphabet of g that survives the spectrum caps
and phase saturation with NO cover decision (V3, capped at length 12), and
with realisability (V4 = L_g(M)) where the CRT budget allows.

usage:
  uv run python research/wordkill_r30.py kill M [--workers 4] [--nodes N]
  uv run python research/wordkill_r30.py words M [--gmax 200] [--crt]
  uv run python research/wordkill_r30.py report M ...
"""
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                          # noqa: E402
from perj_scanfree import spectrum, exposed, next_prime  # noqa: E402

OUT = os.path.join(HERE, "data", "r30")
KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
# exact spectra on record (C11-UPDATE r28/r29) where perj_scanfree has none;
# every entry is an exact value or a 100%-coverage upper bound, so it is a
# SOUND filter (it only removes words that cannot be realised).
SPEC_EXTRA = {43: {1: 103, 2: 116, 3: 125, 4: 132},
              47: {1: 118, 2: 134, 3: 145, 4: 174, 5: 174, 6: 177}}
KNOWN_L = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2,
           43: 2, 47: 4}
GIVEN_WORDS = {47: [(18, 35, 18, 35), (35, 18, 35, 18)]}


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def gears(y):
    return [p for p in range(5, y + 1) if is_prime(p)]


def dg(g):
    return (2 * pow(6, -1, g)) % g


def cls_of(v, g):
    r, d = v % g, dg(g)
    return 0 if r == 0 else 1 if r == d else -1 if r == (g - d) % g else None


def alphabet(M, g, vals=None):
    out = []
    for v in range(1, KNOWN_F[M] + 1):
        if vals is not None and v not in vals:
            continue
        c = cls_of(v, g)
        if c is not None:
            out.append((v, c))
    return out


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


def spec_ok(vs, Fspec):
    for j in range(1, min(len(vs), max(Fspec)) + 1):
        lim = Fspec.get(j)
        if lim is None:
            continue
        for i in range(0, len(vs) - j + 1):
            if sum(vs[i:i + j]) > lim:
                return False
    return True


def points(word):
    X = [0]
    for v in word:
        X.append(X[-1] + v)
    Y = [t for t in range(1, X[-1]) if t not in set(X)]
    return X, Y


def sat_set(X, G):
    """gears whose single-gear free set is empty."""
    out = []
    for g in G:
        Eg = exposed(g)
        xs = {x % g for x in X}
        if not any(all((t + x) % g in Eg for x in xs) for t in range(g)):
            out.append(g)
    return out


def decide_R(qs, X, Y, constrained, node_budget):
    """crt_dict.decide_cover with the OPEN constraint imposed only on the
    gears in `constrained`; every other gear's phase is free.  Same search."""
    n = len(qs)
    Yl = list(Y)
    ny = len(Yl)
    pos = {t: j for j, t in enumerate(Yl)}
    FULL = (1 << ny) - 1
    masks = []
    for q in qs:
        u = pow(6, -1, q)
        forb = set()
        if q in constrained:
            for x in X:
                forb.add((u - x) % q)
                forb.add((-u - x) % q)
        d = {}
        for a in range(q):
            if a in forb:
                continue
            m = 0
            for t in Yl:
                if (a + t - u) % q == 0 or (a + t + u) % q == 0:
                    m |= 1 << pos[t]
            d[a] = m
        if not d:
            return False, 0
        masks.append(d)
    options = [[] for _ in range(ny)]
    for gi in range(n):
        for a, m in masks[gi].items():
            mm = m
            while mm:
                b = mm & -mm
                options[b.bit_length() - 1].append((gi, a))
                mm ^= b
    for j in range(ny):
        if not options[j]:
            return False, 0
    asg = [None] * n
    nodes = [0]

    def rec(covered):
        nodes[0] += 1
        if nodes[0] > node_budget:
            raise crt_dict.Budget()
        unc = FULL & ~covered
        if unc == 0:
            return True
        need = bin(unc).count("1")
        cap = 0
        for gi in range(n):
            if asg[gi] is not None:
                continue
            best = 0
            for m in masks[gi].values():
                c = bin(m & unc).count("1")
                if c > best:
                    best = c
            cap += best
            if cap >= need:
                break
        if cap < need:
            return False
        bestj, bestlo = -1, None
        mm = unc
        while mm:
            b = mm & -mm
            j = b.bit_length() - 1
            mm ^= b
            lo = [(gi, a) for (gi, a) in options[j] if asg[gi] is None]
            if not lo:
                return False
            if bestlo is None or len(lo) < len(bestlo):
                bestj, bestlo = j, lo
                if len(lo) == 1:
                    break
        bestlo.sort(key=lambda ga: -bin(masks[ga[0]][ga[1]] & unc).count("1"))
        for (gi, a) in bestlo:
            asg[gi] = a
            if rec(covered | masks[gi][a]):
                return True
            asg[gi] = None
        return False

    try:
        ok = rec(0)
    except crt_dict.Budget:
        return None, nodes[0]
    return ok, nodes[0]


def killer_job(args):
    """one extension word: SAT set, then y* by bisection on the prefix."""
    M, word, nodes = args
    t0 = time.time()
    G = gears(M)
    X, Y = points(word)
    S = sat_set(X, G)
    # full decision first (the record needs 'refuted' to be exact)
    full, nf = decide_R(G, X, Y, set(G), nodes)
    res = dict(word=list(word), span=X[-1], sat=S, full=full, nodes_full=nf,
               ystar=None, ystar_lo=None, ystar_hi=None, undecided=[])
    if full is not False:
        res["seconds"] = round(time.time() - t0, 1)
        return res                        # realised or undecided: no y*
    # R(empty): NO open constraint at all - only "every interior point
    # blocked".  Infeasible means the blocked pattern itself does not occur
    # in M (a COVER-ONLY kill, y* = 0): no tooth position of any open point
    # is needed to refute the word.
    r0, _ = decide_R(G, X, Y, set(), nodes)
    res["R_empty"] = r0
    if r0 is False:
        res["ystar"] = 0
        res["ystar_lo"] = res["ystar_hi"] = 0
        res["seconds"] = round(time.time() - t0, 1)
        return res
    if r0 is None:
        res["undecided"].append(0)
    # THE CHEAP-ENDS LADDER (replaces a bisection that stalled at m37: R(S)
    # with a small S is a hard relaxed instance, and the question the brief
    # asks is only "corridor or beyond").  R is monotone in S, so the first
    # infeasible prefix in 5 < 7 < 11 < 13 is y*; if R({5,7,11,13}) is still
    # feasible the killer is BEYOND the corridor and one more call decides
    # whether the top gear itself is needed (R(all but the top) feasible <=>
    # y* = M).  A Budget answer stops the ladder and is recorded as a bracket.
    relax = max(2_000_000, nodes // 4)
    prev = None                          # last prefix known feasible
    for pre in ([5], [5, 7], [5, 7, 11], [5, 7, 11, 13]):
        if pre[-1] > G[-1]:
            break
        ok, _ = decide_R(G, X, Y, set(pre), relax)
        res.setdefault("R_prefix", {})[pre[-1]] = ok
        if ok is False:
            res["ystar"] = res["ystar_lo"] = res["ystar_hi"] = pre[-1]
            res["seconds"] = round(time.time() - t0, 1)
            return res
        if ok is None:
            res["undecided"].append(pre[-1])
            break
        prev = pre[-1]
    nxt = G[G.index(prev) + 1] if prev is not None and prev in G and G.index(prev) + 1 < len(G) else None
    res["ystar_lo"], res["ystar_hi"] = nxt, G[-1]
    if prev == 13 and len(G) > 5:
        okm, _ = decide_R(G, X, Y, set(G[:-1]), relax)
        res["R_prefix"][G[-2]] = okm
        if okm is True:
            res["ystar"] = res["ystar_lo"] = res["ystar_hi"] = G[-1]
        elif okm is False:
            res["ystar_hi"] = G[-2]
        else:
            res["undecided"].append(G[-2])
    res["seconds"] = round(time.time() - t0, 1)
    return res


def realised_job(args):
    M, word, nodes = args
    t0 = time.time()
    try:
        ok = crt_dict.realised(M, word, node_budget=nodes)
    except crt_dict.Budget:
        ok = None
    return word, ok, round(time.time() - t0, 1)


def canonical(w):
    return min(tuple(w), tuple(w)[::-1])


def grow_words(M, q1, Fspec, vals, workers, nodes, upto):
    """realised T3-legal words of length 2..upto by overlap growth."""
    A = alphabet(M, q1, vals)
    letters = [v for v, _ in A]
    level = {}
    # level 2: every T3-legal pair under the spectrum, decided by CRT (level
    # 1 - the realised letters - is decided only when L(M) = 1, since arity-1
    # decisions are the EXPENSIVE end of the CRT cost curve at large M)
    start = 1 if upto <= 2 else 2
    if start == 1:
        cands = [(a,) for a in letters]
    else:
        cands = [(a, b) for a in letters for b in letters
                 if t3_ok((a, b), q1) and spec_ok([a, b], Fspec)]
    for n in range(start, upto + 1):
        if n == 2 and start == 1:
            prev = level[1]
            cands = [(a, b) for (a,) in prev for (b,) in prev
                     if t3_ok((a, b), q1) and spec_ok([a, b], Fspec)]
        elif n > 2:
            prev = level[n - 1]
            cands = sorted({w + (x,) for w in prev for x in letters
                            if w[1:] + (x,) in prev and t3_ok(w + (x,), q1)
                            and spec_ok(list(w) + [x], Fspec)})
        classes = sorted({canonical(w) for w in cands})
        t0 = time.time()
        with Pool(min(workers, max(1, len(classes)))) as p:
            res = p.map(realised_job, [(M, w, nodes) for w in classes],
                        chunksize=1)
        yes = set()
        und = []
        for w, ok, dt in res:
            if ok:
                yes.add(w)
                yes.add(w[::-1])
            elif ok is None:
                und.append(w)
        level[n] = yes
        print(f"  M={M} level {n}: {len(cands)} T3+spectrum candidates, "
              f"{len(classes)} reverse classes, {len(yes)} realised, "
              f"{len(und)} undecided  [{time.time()-t0:.0f}s]", flush=True)
        if und:
            print(f"    UNDECIDED: {und}", flush=True)
        if not yes:
            break
    return level


def kill(M, workers, nodes):
    q1 = next_prime(M)
    L = KNOWN_L[M]
    Fspec, vals, exact = spectrum(M)
    if Fspec is None:
        Fspec, vals = SPEC_EXTRA[M], None
    if M == 41:
        vals = None                    # the superset's value set is not exact
    print(f"=== machine {M}, q' = {q1}, d = {dg(q1)}, F = {KNOWN_F[M]}, "
          f"L(M) = {L}, alphabet {[v for v, _ in alphabet(M, q1)]}, "
          f"spectrum {Fspec}")
    if M in GIVEN_WORDS:
        words = set(GIVEN_WORDS[M])
        print(f"  realised length-{L} words TAKEN AS GIVEN (Constructor r29): "
              f"{sorted(words)}")
    else:
        level = grow_words(M, q1, Fspec, vals, workers, nodes, L + 1)
        assert level.get(L), ("no realised word at length L", M, L)
        assert not level.get(L + 1), ("a realised word at length L+1 - "
                                      "L(M) is wrong", M, level.get(L + 1))
        words = level[L]
        print(f"  realised length-{L} words ({len(words)}): {sorted(words)}")
    # extensions (both ends, every class value <= F including holes)
    A = [v for v, _ in alphabet(M, q1)]
    ext = set()
    for w in words:
        for x in A:
            for e in (w + (x,), (x,) + w):
                if t3_ok(e, q1):
                    ext.add(e)
    classes = sorted({canonical(e) for e in ext})
    print(f"  one-letter extensions: {len(ext)} words, {len(classes)} reverse "
          f"classes; deciding at {nodes:,} nodes on {workers} workers",
          flush=True)
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"killer_m{M}.json")
    res = []
    out = dict(M=M, q=q1, L=L, words=sorted(words), gears=gears(M), ext=res,
               classes=len(classes), seconds=0)
    with Pool(min(workers, len(classes))) as p:
        for r in p.imap_unordered(killer_job, [(M, e, nodes) for e in classes]):
            res.append(r)
            ys = (str(r["ystar"]) if r["ystar"] is not None else
                  f"[{r['ystar_lo']},{r['ystar_hi']}]")
            print(f"    done {len(res)}/{len(classes)}: {r['word']} sat={r['sat']} "
                  f"full={r['full']} R_empty={r.get('R_empty')} y*={ys} "
                  f"[{r.get('seconds', 0):.0f}s]", flush=True)
            out["seconds"] = round(time.time() - t0, 1)
            with open(fn, "w") as f:          # dump after EVERY class (rule 38)
                json.dump(out, f, indent=1)
    show(out)
    print(f"  written {fn}")


def show(out):
    M = out["M"]
    print(f"\n  KILLER PROFILE machine {M} -> {out['q']} (L = {out['L']}):")
    print("    extension word                span  SAT gears     y*     full  s"
          "      (y* = 0: cover-only kill, no open constraint needed)")
    hist = {}
    for r in sorted(out["ext"], key=lambda r: (r["ystar_hi"] if r["ystar_hi"] is not None else 999, r["word"])):
        ys = (str(r["ystar"]) if r["ystar"] is not None else
              f"[{r['ystar_lo']},{r['ystar_hi']}]" if r["ystar_hi"] else "-")
        fl = {True: "REALISED", False: "refuted", None: "UNDEC"}[r["full"]]
        print(f"    {str(r['word']):28s} {r['span']:4d}  {str(r['sat']):12s} "
              f"{ys:>8s}  {fl:8s} {r.get('seconds', 0):.0f}")
        k = r["ystar"] if r["ystar"] is not None else ("bracket" if r["full"] is False else fl)
        hist[k] = hist.get(k, 0) + 1
    print(f"    y* histogram: "
          f"{dict(sorted(hist.items(), key=lambda kv: str(kv[0])))}")


def words_mode(M, gmax, crt, workers, nodes):
    """probe (a) word vehicle: V3 (alphabet + spectrum + saturation, no cover)
    and V4 (+ realisability) for every prime g in (M, gmax]."""
    Fspec, vals, exact = spectrum(M)
    if Fspec is None:
        Fspec, vals = SPEC_EXTRA[M], None
    if M == 41:
        vals = None
    G = gears(M)
    E = {g: exposed(g) for g in G}
    rows = []
    for g in [p for p in range(M + 1, gmax + 1) if is_prime(p)]:
        A = alphabet(M, g, vals)
        letters = [v for v, _ in A]
        if not letters:
            rows.append(dict(g=g, alphabet=[], V3=0, V4=0, note="no letter"))
            continue
        # V3: grow words with T3 + spectrum + saturation only
        words = [(v,) for v in letters]
        words = [w for w in words if not sat_set(points(w)[0], G)]
        V3, cyc = (1 if words else 0), False
        lvl = {1: set(words)}
        n = 1
        while words and n < 12:
            n += 1
            nxt = set()
            for w in lvl[n - 1]:
                for x in letters:
                    e = w + (x,)
                    if e[1:] in lvl[n - 1] and t3_ok(e, g) and spec_ok(list(e), Fspec) \
                            and not sat_set(points(e)[0], G):
                        nxt.add(e)
            lvl[n] = nxt
            words = list(nxt)
            if words:
                V3 = n
        if n == 12 and words:
            cyc = True
        row = dict(g=g, d=dg(g), alphabet=letters, V3=V3, V3_cycle=cyc,
                   V3_counts={k: len(v) for k, v in lvl.items() if v})
        if crt:
            # V4: realisability, level by level from the V3 survivors
            real = None
            for k in range(1, V3 + 1):
                cl = sorted({canonical(w) for w in lvl[k]
                             if k == 1 or (real is None or
                                           (w[:-1] in real and w[1:] in real))})
                if not cl:
                    break
                with Pool(min(workers, len(cl))) as p:
                    res = p.map(realised_job, [(M, w, nodes) for w in cl],
                                chunksize=1)
                yes = set()
                und = [w for w, ok, _ in res if ok is None]
                for w, ok, _ in res:
                    if ok:
                        yes.add(w)
                        yes.add(w[::-1])
                row.setdefault("V4_levels", {})[k] = dict(
                    cand=len(cl), realised=len(yes), undecided=len(und))
                if not yes:
                    break
                real = yes
                row["V4"] = k
                row["V4_words"] = sorted(yes)
            row.setdefault("V4", 0)
        rows.append(row)
        print(f"  M={M} g={g:3d} d={dg(g):3d} letters {letters}  V3 = {V3}"
              f"{' (cycle, capped 12)' if cyc else ''}  counts "
              f"{row['V3_counts']}"
              + (f"  V4 = {row['V4']} {row.get('V4_words', '')}" if crt else ""),
              flush=True)
    fn = os.path.join(OUT, f"words_m{M}{'_crt' if crt else ''}.json")
    with open(fn, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"  written {fn}")


if __name__ == "__main__":
    args = sys.argv[1:]

    def opt(nm, d):
        return type(d)(args[args.index(nm) + 1]) if nm in args else d

    cmd = args[0]
    if cmd == "kill":
        kill(int(args[1]), opt("--workers", 4), opt("--nodes", 20_000_000))
    elif cmd == "words":
        words_mode(int(args[1]), opt("--gmax", 200), "--crt" in args,
                   opt("--workers", 4), opt("--nodes", 20_000_000))
    elif cmd == "report":
        for M in args[1:]:
            show(json.load(open(os.path.join(OUT, f"killer_m{M}.json"))))
