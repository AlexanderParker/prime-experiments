"""The smooth vectors of the long openings stored by opening_spectrum.py (real manifold).

For each stored opening (start n, length L, slot count): the block n..n+L+1 (L >= 2) or the pair
{n, n+2} (L = 1); its smooth vector s_i = q-smooth part of each member; its turn t = (n-1)//Q;
ember flag (a member equal to its smooth part, i.e. fuel 1); the onset law t >= max s_i for
ember-free openings (exception count); the first opening of each length and its vector; the
aligned openings' vectors (slot position inside the block, the smooth part of the middle member);
the residue histogram of starts mod q# against uniform, by turn band.

usage: uv run python research/valves/r4/opening_vectors.py q Q
"""
import sys, os, json, math
import numpy as np

q = int(sys.argv[1]); Q = int(sys.argv[2])
here = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(here, "results", f"spectrum_q{q}_Q{Q}_real_G{Q}.json")))
engine = [p for p in [2, 3, 5, 7, 11, 13] if p <= q]; qsharp = math.prod(engine)
slot_set = set(d["slots_mod_qsharp"])


def smooth_part(m):
    s = 1
    for p in engine:
        while m % p == 0:
            m //= p; s *= p
    return s


lines = [f"q = {q}, Q = {Q}: stored openings of length >= {min(int(k) for k in d['stored'])}"]
exc_onset = 0; n_checked = 0
summary = {}
for Lk in sorted(d["stored"], key=int):
    L = int(Lk); items = d["stored"][Lk]
    members = (lambda n: [n, n + 2]) if L == 1 else (lambda n: list(range(n, n + L + 2)))
    rows = []
    for n, sl in items:
        ms = members(n)
        sv = [smooth_part(m) for m in ms]
        ember = any(m == s and m > Q for m, s in zip(ms, sv))
        t = (n - 1) // Q
        rows.append((n, sl, sv, ember, t))
        if not ember:
            n_checked += 1
            if t < max(sv):
                exc_onset += 1
    rows.sort()
    first = rows[0]
    first_al = next((r for r in rows if r[1] > 0), None)
    al = [r for r in rows if r[1] > 0]; nal = [r for r in rows if r[1] == 0]
    emb = sum(1 for r in rows if r[3])
    mx_al = np.mean([max(r[2]) for r in al]) if al else float('nan')
    mx_nal = np.mean([max(r[2]) for r in nal]) if nal else float('nan')
    lines.append(f"L = {L}: {len(rows)} openings (stored cap {len(items)} of {d['count_L'][Lk]}), {len(al)} aligned, {emb} with an ember; "
                 f"mean max smooth part: aligned {mx_al:.1f}, not aligned {mx_nal:.1f}; "
                 f"first: n = {first[0]} (turn {first[4]}, vector {first[2]}, slots {first[1]}{', ember' if first[3] else ''}); "
                 f"first aligned: " + (f"n = {first_al[0]} (turn {first_al[4]}, vector {first_al[2]}{', ember' if first_al[3] else ''})" if first_al else "none"))
    # the slot position inside aligned blocks and the middle member's smooth part
    if al and L >= 2:
        pos = {}
        mids = {}
        for n, sl, sv, ember, t in al:
            for i in range(L):
                if (n + i) % qsharp in slot_set:
                    pos[i] = pos.get(i, 0) + 1
                    mids[sv[i + 1]] = mids.get(sv[i + 1], 0) + 1
        lines.append(f"   slot position i inside the block (0..{L-1}): {dict(sorted(pos.items()))}; smooth part of the member between the twin: "
                     f"{dict(sorted(mids.items())[:12])}")
    # earliest five openings and their vectors
    lines.append("   earliest: " + "; ".join(f"{r[0]} t{r[4]} {r[2]}{' E' if r[3] else ''}{' *' if r[1] else ''}" for r in rows[:6]))
    # residues of starts mod q#: the top residues
    rh = np.array(d["res_hist"][Lk]); tot = rh.sum()
    top = np.argsort(-rh)[:6]
    lines.append(f"   start residues mod {qsharp}: used {int((rh > 0).sum())} of {qsharp}; max {int(rh.max())} at r = {int(rh.argmax())}, min {int(rh.min())}; "
                 f"uniform would be {tot / qsharp:.1f}; top: " + ", ".join(f"r={int(r)}:{int(rh[r])}" for r in top))
    summary[L] = {"n": len(rows), "aligned": len(al), "ember": emb, "first": first[:2] + (first[2], first[3], first[4]),
                  "first_aligned": (first_al[:2] + (first_al[2], first_al[3], first_al[4])) if first_al else None}
lines.append(f"onset law (turn >= max smooth part, ember-free openings): {exc_onset} exceptions in {n_checked}")
print("\n".join(lines))
json.dump(summary, open(os.path.join(here, "results", f"vectors_q{q}_Q{Q}.json"), "w"), default=str)
