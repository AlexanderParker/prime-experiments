"""Opening spectrum of the manifold on the quiet zone (Q, Q^2], and the engine's twin slots inside
each opening.  Raw line.  An OPENING is a maximal run of L consecutive manifold-open pairs
(n, n+2), (n+1, n+3), ..., (n+L-1, n+L+1).  A twin SLOT is a pair with both members coprime to q#.

usage: uv run python research/valves/r4/opening_spectrum.py q Q [model] [Gmax]
  model: real            the manifold: gear g strikes the integers = 0 (mod g)   (phase zero)
         rand:<seed>     free-phase copy: gear g strikes the integers = a_g (mod g), a_g random
         adv             the V12 one-tooth adversary on the fuel (phases from research/valves/r2/results)
  Gmax: only the gears <= Gmax are used (truncated manifold); default Q.
Writes results/spectrum_q{q}_Q{Q}_{model}_G{Gmax}.json and prints a bounded summary.
"""
import sys, os, json, math, time
import numpy as np

q = int(sys.argv[1]); Q = int(sys.argv[2])
model = sys.argv[3] if len(sys.argv) > 3 else "real"
Gmax = int(sys.argv[4]) if len(sys.argv) > 4 else Q
N = Q * Q                      # pairs with lower member n in (Q, N - 2]
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)
t0 = time.time()


def sieve(n):
    is_p = np.ones(n + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = False
    return is_p


isp = sieve(max(Q, Gmax) + 2)
engine = [int(p) for p in np.flatnonzero(isp) if p <= q]
gears = [int(p) for p in np.flatnonzero(isp) if q < p <= Gmax]
qsharp = math.prod(engine)
qprime = int(gears[0]) if gears else None
ceiling = qprime - 3 if qprime else None
LMAX = max(ceiling or 1, 1) + 4          # room to detect a violation of the ceiling

# ---- phases
rng = None
if model == "real":
    phase = {g: 0 for g in gears}
elif model.startswith("rand:"):
    rng = np.random.default_rng(int(model.split(":")[1]))
    phase = {g: int(rng.integers(g)) for g in gears}
elif model == "adv":
    src = os.path.join(here, "..", "r2", "results", f"inv_adv_q{q}_Q{Q}_M60.json")
    used = json.load(open(src))["used"]
    phase = {g: int(used.get(str(g), 0)) for g in gears}
else:
    raise SystemExit("model must be real, rand:<seed> or adv")

# ---- the engine's slot pattern mod q# and its longest blocked run of pairs
res = np.arange(qsharp)
slot_pat = np.ones(qsharp, dtype=bool)
for p in engine:
    slot_pat &= (res % p != 0) & ((res + 2) % p != 0)
slots = np.flatnonzero(slot_pat)
gaps = np.diff(np.concatenate([slots, [slots[0] + qsharp]]))
engine_blocked_run = int(gaps.max() - 1)
# S_L: residues r such that some r + i (0 <= i < L) is a slot
S_L = {}
for L in range(1, LMAX + 1):
    S_L[L] = int(np.sum([any(slot_pat[(r + i) % qsharp] for i in range(L)) for r in range(qsharp)]))

# ---- CRT predictions: f(k) = prod_g (g - k)/g
def f(k):
    s = 0.0
    for g in gears:
        if g <= k:
            return 0.0
        s += math.log1p(-k / g)
    return math.exp(s)

crt_exact = {}
for L in range(1, LMAX + 1):
    if L == 1:
        d = f(2) - 2 * f(4) + f(5)
    else:
        d = f(L + 2) - 2 * f(L + 3) + f(L + 4)
    crt_exact[L] = d * (N - Q)
crt_atleast = {L: f(L + 2) * (N - Q) if L >= 2 else f(2) * (N - Q) for L in range(1, LMAX + 1)}

# ---- adversary fuel (whole-range, only for Q <= 10^4)
adv_fuel = None
smooth_list = None
if model == "adv":
    ok = np.ones(N + 3, dtype=bool)
    for g in gears:
        ok[phase[g]::g] = False
    for p in engine:
        ok[::p] = False
    ok[:Q + 1] = False
    adv_fuel = np.flatnonzero(ok).astype(np.int64)
    # q-smooth numbers up to N + 2
    smooth_list = []
    def gen(idx, cur):
        if idx == len(engine):
            smooth_list.append(cur); return
        p = engine[idx]; v = cur
        while v <= N + 2:
            gen(idx + 1, v); v *= p
    gen(0, 1)
    smooth_list = np.array(sorted(smooth_list), dtype=np.int64)

# ---- segmented scan
SEG = 20_000_000 if N > 20_000_000 else N + 8
turns = Q + 2
cnt_L_turn = np.zeros((LMAX + 2, turns), dtype=np.int64)        # openings of length L in turn t
aligned_L_turn = np.zeros((LMAX + 2, turns), dtype=np.int64)    # ... holding >= 1 slot
slots_hist = np.zeros((LMAX + 2, 8), dtype=np.int64)            # number of slots per opening
res_hist = np.zeros((LMAX + 2, qsharp), dtype=np.int64)         # start residue mod q#
first_pos = {}; first_aligned = {}
LSTORE = max(1, (ceiling or 1) - 1)
stored = {}                                                     # L -> list of (n, slots)
STORE_CAP = 20000
ember_count = 0
open_pairs_total = 0
n_over = 0

a = Q + 1                        # first pair position
last_reported = 0
while a <= N - 2:
    b0 = min(a + SEG, N - 1)     # provisional end (exclusive) of pair positions
    hi = min(b0 + 64, N + 1)     # integers needed: up to hi + 2
    lo_int = a
    length = hi + 3 - lo_int
    opn = np.ones(length, dtype=bool)
    if model == "adv":
        opn[:] = False
        for s in smooth_list:
            if s > hi + 2: break
            fmax = (hi + 2) // s; fmin = max(Q + 1, (lo_int + s - 1) // s)
            if fmax < fmin:
                continue
            lo_i = np.searchsorted(adv_fuel, fmin); hi_i = np.searchsorted(adv_fuel, fmax, side="right")
            if hi_i > lo_i:
                opn[adv_fuel[lo_i:hi_i] * s - lo_int] = True
            if s > Q and lo_int <= s <= hi + 2:
                opn[s - lo_int] = True            # ember (fuel 1)
    else:
        for g in gears:
            start = (phase[g] - lo_int) % g
            opn[start::g] = False
    # pair-open over positions a .. hi
    po = opn[:hi + 1 - lo_int] & opn[2:hi + 3 - lo_int]
    # choose the segment end b (exclusive) as the first position >= b0 with po False
    if b0 >= N - 1:
        b = N - 1
        po = po[:b - a]
    else:
        j = b0 - a
        while po[j]:
            j += 1
        b = a + j
        po = po[:b - a]
    # runs of True in po
    d = np.diff(np.concatenate([[0], po.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1)
    Lr = ends - starts
    S = starts + a
    if len(Lr):
        n_over += int((Lr > (ceiling or 10**9)).sum())
        Lc = np.minimum(Lr, LMAX + 1)
        # slots inside each run
        eo_full = np.tile(slot_pat, (b - a) // qsharp + 2)
        off = a % qsharp
        eo = eo_full[off:off + (b - a)]
        C = np.concatenate([[0], np.cumsum(eo, dtype=np.int64)])
        sl = C[ends] - C[starts]
        aligned = sl > 0
        tr = (S - 1) // Q
        np.add.at(cnt_L_turn, (Lc, tr), 1)
        np.add.at(aligned_L_turn, (Lc[aligned], tr[aligned]), 1)
        np.add.at(slots_hist, (Lc, np.minimum(sl, 7)), 1)
        np.add.at(res_hist, (Lc, S % qsharp), 1)
        for L in np.unique(Lc):
            L = int(L)
            m = Lc == L
            if L not in first_pos:
                first_pos[L] = int(S[m].min())
            ma = m & aligned
            if ma.any() and L not in first_aligned:
                first_aligned[L] = int(S[ma].min())
            if L >= LSTORE:
                lst = stored.setdefault(L, [])
                if len(lst) < STORE_CAP:
                    idx = np.flatnonzero(m)[:STORE_CAP - len(lst)]
                    lst.extend([(int(S[i]), int(sl[i])) for i in idx])
        open_pairs_total += int(po.sum())
    a = b
    if (a - Q) // SEG > last_reported:
        last_reported = (a - Q) // SEG
        print(f"  ... at {a} ({100 * a / N:.0f}%), {time.time() - t0:.0f}s", flush=True)

cnt_L = cnt_L_turn.sum(axis=1); aligned_L = aligned_L_turn.sum(axis=1)
Lpresent = [L for L in range(1, LMAX + 2) if cnt_L[L] > 0]
longest = max(Lpresent) if Lpresent else 0

out = {"q": q, "Q": Q, "model": model, "Gmax": Gmax, "gears": len(gears), "qprime": qprime, "ceiling": ceiling,
       "qsharp": qsharp, "slots_mod_qsharp": slots.tolist(), "engine_blocked_run": engine_blocked_run,
       "S_L": S_L, "phase_nonzero": sum(1 for g in gears if phase[g]),
       "open_pairs": open_pairs_total, "over_ceiling": n_over, "longest_present": longest,
       "count_L": {L: int(cnt_L[L]) for L in Lpresent},
       "aligned_L": {L: int(aligned_L[L]) for L in Lpresent},
       "crt_exact_L": {L: crt_exact.get(L, 0.0) for L in range(1, LMAX + 1)},
       "crt_density_longest": f(ceiling + 2) if ceiling else None,
       "slots_hist": {L: slots_hist[L].tolist() for L in Lpresent},
       "res_hist": {L: res_hist[L].tolist() for L in Lpresent},
       "first_pos": first_pos, "first_aligned": first_aligned,
       "cnt_L_turn": {L: cnt_L_turn[L].tolist() for L in Lpresent},
       "aligned_L_turn": {L: aligned_L_turn[L].tolist() for L in Lpresent},
       "stored": {L: v for L, v in stored.items()},
       "seconds": time.time() - t0}
tag = model.replace(":", "")
path = os.path.join(outdir, f"spectrum_q{q}_Q{Q}_{tag}_G{Gmax}.json")
json.dump(out, open(path, "w"))

print(f"q={q} Q={Q} model={model} Gmax={Gmax}: {len(gears)} gears, q'={qprime}, ceiling q'-3={ceiling}; "
      f"engine slots mod {qsharp}: {len(slots)}, longest blocked pair run {engine_blocked_run}; |S_L|/q#: "
      + ", ".join(f"L={L}:{S_L[L]}/{qsharp}" for L in range(1, (ceiling or 1) + 1)))
print(f"open pairs {open_pairs_total}; openings over the ceiling: {n_over}; longest present {longest}; {time.time() - t0:.0f}s")
print("L | count | CRT exact | ratio | aligned | frac | |S_L|/q# | first | first aligned | slots hist")
for L in Lpresent:
    c = int(cnt_L[L]); al = int(aligned_L[L]); ce = crt_exact.get(L, 0.0)
    print(f"{L} | {c} | {ce:.1f} | {c / ce if ce else float('nan'):.3f} | {al} | {al / c:.3f} | {S_L.get(L, 0) / qsharp:.3f} | "
          f"{first_pos.get(L)} | {first_aligned.get(L)} | {slots_hist[L][:4].tolist()}")
