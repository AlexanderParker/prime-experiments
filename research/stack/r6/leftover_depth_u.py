"""The leftover at depth u (branch R4.d.i.c, round 6).

On a section [c, c') of a chain, for each stretch length L' the core is the primes <= t' = 6L' + 1,
a member is core-free iff it is t'-rough, a slot is core-open iff both members are core-free, and
the depth of a slot is u = ln n / ln t'.  Per (L', u-bin of width 0.1):

  * the Omega-census of core-free members (how many have 1, 2, 3, 4, >=5 prime factors);
  * the PP share among core-open slots (a core-open slot is PP iff both members are prime);
  * the mean core leftover K over the stretches starting in the bin;
  * twin-free stretches of length L' -- STARTS and RUNS -- against the independent-slot
    prediction with the bin's own PP share, analytically and by simulation of that model;
  * min K over the twin-free stretches.

Plus: the exact Omega thresholds (first core-free member with Omega = 2, 3, 4 against
nextprime(t')^2, ^3, ^4), and the fuels (largest prime factor) of the composite core-free members
sitting in core-open slots inside twin gaps of at least L' slots.

usage: uv run python research/stack/r6/leftover_depth_u.py BASE SECTION L1,L2,... [NSIM] [TAG]
"""
import sys, os, math, json
import numpy as np
from sympy import nextprime, isprime

BW = 0.1                      # depth bin width
CHUNK = 4_000_000

base = int(sys.argv[1])
ksec = int(sys.argv[2])
Ls = [int(x) for x in sys.argv[3].split(',')]
NSIM = int(sys.argv[4]) if len(sys.argv) > 4 else 20
TAG = sys.argv[5] if len(sys.argv) > 5 else f"b{base}s{ksec}"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
log_path = os.path.join(OUT, f"{TAG}.txt")
LOG = open(log_path, "w", encoding="utf-8")
def say(s=""):
    LOG.write(str(s) + "\n"); LOG.flush()


def sieve(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return s


def spf_sieve(n):
    """smallest prime factor of each composite in [0, n]; 0 at 0, 1 and at the primes."""
    sp = np.zeros(n + 2, dtype=np.int32)
    for i in range(2, int(n ** 0.5) + 1):
        if sp[i] == 0:
            blk = sp[i * i::i]
            blk[blk == 0] = i
    return sp


# ---------------------------------------------------------------- the section
cuts = [base]; firsts = [base]
for _ in range(ksec):
    f = firsts[-1]; c = f * f; cuts.append(c); firsts.append(c if isprime(c) else nextprime(c))
lo, hi = cuts[ksec - 1], cuts[ksec]
pk = firsts[ksec - 1]                      # the section's first gear: hi = pk^2
isp = sieve(hi + 40)
n0 = lo + ((5 - lo) % 6)
slots = np.arange(n0, hi, 6, dtype=np.int64)
if slots[-1] + 2 >= hi: slots = slots[:-1]
S = slots.size
lowp = isp[slots]; upp = isp[slots + 2]
twin = lowp & upp
ntw = int(twin.sum())

# twin gaps in slots; blocklen[j] = length of the twin-free block holding slot j (0 at twins,
# and the value at a twin slot is never read: every use masks on a non-prime member).
tidx = np.nonzero(twin)[0]
gaps = (np.diff(np.concatenate([[-1], tidx, [S]])) - 1).astype(np.int32)
Lrec = int(gaps.max())
blocklen = np.repeat(gaps, gaps + 1)[:S]
del tidx

# smallest prime factor (>= 5) of each member via the gear phases; 0 when the member has no prime
# factor <= sqrt(hi), i.e. when the member is prime.
sq = int(math.isqrt(hi)) + 1
smallp = np.nonzero(sieve(sq))[0]
smallp = smallp[smallp >= 5]
DT = np.int16 if sq < 32000 else np.int32
spf_low = np.zeros(S, dtype=DT); spf_up = np.zeros(S, dtype=DT)
for p in smallp[::-1]:
    p = int(p); inv = pow(6, -1, p)
    spf_low[((0 - n0) * inv) % p::p] = p
    spf_up[(((-2) % p - n0) * inv) % p::p] = p

lnn = np.log(slots.astype(np.float64)).astype(np.float32)
del slots

nprimes_below_cut = int(sieve(lo)[5:].sum())
say(f"# section: base {base}, section {ksec} = [{lo}, {hi}); slots {S}; twins {ntw}; "
    f"section record {Lrec} slots; first gear p_k = {pk}; sqrt(hi) = {sq}; "
    f"primes in [5, c) = {nprimes_below_cut}; NSIM = {NSIM}")

cst = np.concatenate([[0], np.cumsum(twin, dtype=np.int32)])     # cumulative twins


def members(idx, which):
    return (n0 + 6 * idx.astype(np.int64)) + which


def factor_stats(vals, sp0, spf_small):
    """Omega (int8) and the largest prime factor (int64) of each value."""
    mm = vals.copy()
    om = np.zeros(mm.size, dtype=np.int8)
    mx = np.zeros(mm.size, dtype=np.int64)
    first = True
    while True:
        alive = mm > 1
        if not alive.any(): break
        pm = alive & isp[mm]
        if pm.any():
            om[pm] += 1
            t = mx[pm]; np.maximum(t, mm[pm], out=t); mx[pm] = t
            mm[pm] = 1
        rest = mm > 1
        if not rest.any(): break
        d = sp0[rest] if first else spf_small[mm[rest]].astype(np.int64)
        mm[rest] //= d
        om[rest] += 1
        t = mx[rest]; np.maximum(t, d, out=t); mx[rest] = t
        first = False
    return om, mx


results = {"base": base, "section": [int(lo), int(hi)], "slots": int(S), "twins": ntw,
           "record": Lrec, "pk": int(pk), "nsim": NSIM, "L": {}}

for Lp in Ls:
    tp = 6 * Lp + 1
    q = int(nextprime(tp))
    ncore = int(sieve(tp)[5:].sum())
    ntail = max(0, nprimes_below_cut - ncore)
    lt = math.log(tp)
    u = lnn / np.float32(lt)
    b0 = int(math.floor(float(u.min()) / BW)); b1 = int(math.floor(float(u.max()) / BW))
    NB = b1 - b0 + 1
    binidx = np.clip((u / np.float32(BW)).astype(np.int32) - b0, 0, NB - 1)
    del u
    rl = (spf_low == 0) | (spf_low > tp)
    ru = (spf_up == 0) | (spf_up > tp)
    # a member at most t' is divisible by its own prime factors, all at most t': never core-free.
    # (spf is 0 for a prime above sqrt(hi), so this has to be excluded by hand when t' > sqrt(hi).)
    if tp >= n0:
        cutj = (tp - n0) // 6 + 1
        rl[:cutj] = False
        ru[:max(0, (tp - n0 - 2) // 6 + 1)] = False
    op = rl & ru
    nstart = S - Lp + 1
    spf_small = spf_sieve(hi // tp + 2)

    # ---- Omega census of core-free members, per bin; and the first member at each Omega
    om_hist = np.zeros((NB, 6), dtype=np.int64)          # column j = Omega j (5 = ">= 5")
    firsts_om = {}
    for which, rough, spfa in ((0, rl, spf_low), (2, ru, spf_up)):
        idx_all = np.nonzero(rough)[0]
        for s0 in range(0, idx_all.size, CHUNK):
            idx = idx_all[s0:s0 + CHUNK]
            vals = members(idx, which)
            om, _mx = factor_stats(vals, spfa[idx].astype(np.int64), spf_small)
            omc = np.minimum(om, 5).astype(np.int64)
            flat = binidx[idx].astype(np.int64) * 6 + omc
            om_hist += np.bincount(flat, minlength=NB * 6).reshape(NB, 6)
            for j in (2, 3, 4, 5):
                sel = (om == j) if j < 5 else (om >= 5)
                if sel.any():
                    v = int(vals[sel].min())
                    if j not in firsts_om or v < firsts_om[j]: firsts_om[j] = v
            del vals, om, _mx, idx, omc, flat
        del idx_all
    thresholds = {j: {"q_pow": q ** j, "first_seen": firsts_om.get(j),
                      "u_threshold": j * math.log(q) / lt,
                      "u_first": (math.log(firsts_om[j]) / lt) if j in firsts_om else None}
                  for j in (2, 3, 4, 5)}

    # ---- per-bin slot counts and the PP share among core-open slots
    cnt_slots = np.bincount(binidx, minlength=NB)
    cnt_open = np.bincount(binidx[op], minlength=NB)
    ppmask = twin & op
    cnt_pp = np.bincount(binidx[ppmask], minlength=NB)
    pbin = np.where(cnt_open > 0, cnt_pp / np.maximum(cnt_open, 1), 0.0)
    pslot = np.where(op, pbin[binidx], 0.0).astype(np.float32)
    del ppmask

    # ---- K, twin-free starts and runs, analytic prediction under independent slots
    cso = np.concatenate([[0], np.cumsum(op, dtype=np.int32)])
    # log(1 - p) per slot; a bin with p = 1 (every core-open slot is a twin, below depth 2) gets a
    # finite floor of -50 so that a stretch holding such a slot has predicted probability 0.
    _pc = pslot.astype(np.float64)
    _lp = np.where(_pc >= 1.0, -50.0, np.log1p(-np.minimum(_pc, 0.999999999)))
    logq = np.concatenate([[0.0], np.cumsum(_lp, dtype=np.float64)])
    del _pc, _lp
    sumK = np.zeros(NB); cntst = np.zeros(NB, dtype=np.int64)
    cnt_kpos = np.zeros(NB, dtype=np.int64)
    tf_starts = np.zeros(NB, dtype=np.int64); tf_runs = np.zeros(NB, dtype=np.int64)
    pr_starts = np.zeros(NB); pr_runs = np.zeros(NB)
    tf_starts_k = np.zeros(NB, dtype=np.int64); tf_runs_k = np.zeros(NB, dtype=np.int64)
    pr_starts_k = np.zeros(NB); pr_runs_k = np.zeros(NB)
    minK_tf = np.full(NB, 1 << 30, dtype=np.int64)
    kposmask = np.zeros(nstart, dtype=bool)
    for s0 in range(0, nstart, CHUNK):
        s1 = min(s0 + CHUNK, nstart)
        b = binidx[s0:s1].astype(np.int64)
        K = (cso[s0 + Lp:s1 + Lp] - cso[s0:s1]).astype(np.int64)
        kp = K > 0
        kposmask[s0:s1] = kp
        sumK += np.bincount(b, weights=K, minlength=NB)
        cntst += np.bincount(b, minlength=NB)
        cnt_kpos += np.bincount(b[kp], minlength=NB)
        w = np.exp(logq[s0 + Lp:s1 + Lp] - logq[s0:s1])
        pr_starts += np.bincount(b, weights=w, minlength=NB)
        pr_starts_k += np.bincount(b[kp], weights=w[kp], minlength=NB)
        prev = pslot[s0 - 1:s1 - 1].astype(np.float64) if s0 > 0 else \
            np.concatenate([[0.0], pslot[0:s1 - 1].astype(np.float64)])
        pr_runs += np.bincount(b, weights=w * prev, minlength=NB)
        pr_runs_k += np.bincount(b[kp], weights=(w * prev)[kp], minlength=NB)
        tf = (cst[s0 + Lp:s1 + Lp] - cst[s0:s1]) == 0
        if tf.any():
            tf_starts += np.bincount(b[tf], minlength=NB)
            tf_starts_k += np.bincount(b[tf & kp], minlength=NB)
            prevtw = twin[s0 - 1:s1 - 1] if s0 > 0 else np.concatenate([[True], twin[0:s1 - 1]])
            rs = tf & prevtw
            if rs.any():
                tf_runs += np.bincount(b[rs], minlength=NB)
                tf_runs_k += np.bincount(b[rs & kp], minlength=NB)
            kk = K[tf]; bb = b[tf]
            order = np.argsort(bb, kind='stable'); bb = bb[order]; kk = kk[order]
            edges = np.searchsorted(bb, np.arange(NB + 1))
            for bi in range(NB):
                if edges[bi + 1] > edges[bi]:
                    m = int(kk[edges[bi]:edges[bi + 1]].min())
                    if m < minK_tf[bi]: minK_tf[bi] = m
            del kk, bb, order, edges, rs, prevtw
        del b, K, w, tf, prev, kp
    del logq, cso

    # ---- the model's own null distribution of the run count (second, independent measurement)
    sim_runs = np.zeros((NSIM, NB), dtype=np.int64)
    sim_starts = np.zeros((NSIM, NB), dtype=np.int64)
    sim_runs_k = np.zeros((NSIM, NB), dtype=np.int64)
    rng = np.random.default_rng(20260910 + Lp)
    for s in range(NSIM):
        st = rng.random(S, dtype=np.float32) < pslot
        cs = np.concatenate([[0], np.cumsum(st, dtype=np.int32)])
        for s0 in range(0, nstart, CHUNK):
            s1 = min(s0 + CHUNK, nstart)
            b = binidx[s0:s1].astype(np.int64)
            tf = (cs[s0 + Lp:s1 + Lp] - cs[s0:s1]) == 0
            if tf.any():
                sim_starts[s] += np.bincount(b[tf], minlength=NB)
                prevtw = st[s0 - 1:s1 - 1] if s0 > 0 else np.concatenate([[True], st[0:s1 - 1]])
                rs = tf & prevtw
                if rs.any():
                    sim_runs[s] += np.bincount(b[rs], minlength=NB)
                    sim_runs_k[s] += np.bincount(b[rs & kposmask[s0:s1]], minlength=NB)
                del prevtw, rs
            del b, tf
        del st, cs

    # ---- the fuels: composite core-free members in core-open slots inside twin gaps >= L'
    inrun = blocklen >= Lp
    fuel = {"members": 0, "omega2": 0, "omega3": 0, "omega4plus": 0, "p2_gt_n_over_pk": 0,
            "exceptions_omega2": 0, "satisfy_omega3plus": 0, "fuel_above_cut": 0}
    for which, rough, pr, spfa in ((0, rl, lowp, spf_low), (2, ru, upp, spf_up)):
        sel = np.nonzero(op & inrun & rough & ~pr)[0]
        for s0 in range(0, sel.size, CHUNK):
            idx = sel[s0:s0 + CHUNK]
            vals = members(idx, which)
            om, mx = factor_stats(vals, spfa[idx].astype(np.int64), spf_small)
            cof = vals // mx                     # n / (largest prime factor)
            good = cof < pk                      # equivalent to P2 > n / p_k
            fuel["members"] += int(idx.size)
            fuel["omega2"] += int((om == 2).sum())
            fuel["omega3"] += int((om == 3).sum())
            fuel["omega4plus"] += int((om >= 4).sum())
            fuel["p2_gt_n_over_pk"] += int(good.sum())
            fuel["exceptions_omega2"] += int(((om == 2) & ~good).sum())
            fuel["satisfy_omega3plus"] += int(((om >= 3) & good).sum())
            fuel["fuel_above_cut"] += int((mx >= lo).sum())
            del idx, vals, om, mx, cof, good
        del sel
    del inrun, spf_small

    # ---- report
    say()
    say(f"## L' = {Lp}: t' = {tp}, nextprime(t') = {q}, core gears {ncore}, tail gears {ntail}, "
        f"starts {nstart}, depth {b0*BW:.1f}-{(b1+1)*BW:.1f}")
    say(f"   core-free members {int(om_hist.sum())} "
        f"(Omega 1..5+: {[int(x) for x in om_hist.sum(axis=0)[1:]]}); "
        f"core-open slots {int(cnt_open.sum())}; PP {int(cnt_pp.sum())}; "
        f"mean K {sumK.sum()/max(1,cntst.sum()):.4f}; min K over twin-free "
        f"{int(minK_tf.min()) if minK_tf.min() < (1<<30) else None}")
    say(f"   twin-free starts {int(tf_starts.sum())} (predicted {pr_starts.sum():.2f}); "
        f"runs {int(tf_runs.sum())} (predicted {pr_runs.sum():.2f}; "
        f"sim {sim_runs.sum(axis=1).mean():.2f} +- {sim_runs.sum(axis=1).std(ddof=1):.2f})")
    say(f"   restricted to K > 0: starts with K > 0 {int(cnt_kpos.sum())}; twin-free "
        f"{int(tf_starts_k.sum())} (predicted {pr_starts_k.sum():.2f}); runs "
        f"{int(tf_runs_k.sum())} (predicted {pr_runs_k.sum():.2f}; sim "
        f"{sim_runs_k.sum(axis=1).mean():.2f} +- {sim_runs_k.sum(axis=1).std(ddof=1):.2f})")
    say("   Omega thresholds: " + "; ".join(
        f"Om={j}: q^{j}={thresholds[j]['q_pow']}, first={thresholds[j]['first_seen']}, "
        f"u_thr={thresholds[j]['u_threshold']:.4f}, u_first="
        + (f"{thresholds[j]['u_first']:.4f}" if thresholds[j]['u_first'] else "-")
        for j in (2, 3, 4)))
    say(f"   fuels (composite core-free members in core-open slots of twin gaps >= L'): "
        f"{fuel['members']} members, Omega2 {fuel['omega2']}, Omega3 {fuel['omega3']}, "
        f"Omega>=4 {fuel['omega4plus']}; P2 > n/p_k for {fuel['p2_gt_n_over_pk']} "
        f"(Omega2 exceptions {fuel['exceptions_omega2']}, Omega>=3 satisfying "
        f"{fuel['satisfy_omega3plus']}); largest factor >= c for {fuel['fuel_above_cut']}")
    say("u | slots | core-free | Om1 | Om2 | Om3 | Om4 | Om5+ | prime share | core-open | "
        "PP share | starts | starts K>0 | mean K | tf starts | pred starts | tf runs | pred runs |"
        " sim mean | sim sd | z | tf runs K>0 | pred runs K>0 | sim K>0 mean | sim K>0 sd | z K>0 "
        "| min K on tf")
    rows = []
    for bi in range(NB):
        cf = int(om_hist[bi].sum())
        if cf == 0 and int(cnt_slots[bi]) == 0: continue
        sm = float(sim_runs[:, bi].mean()); ssd = float(sim_runs[:, bi].std(ddof=1))
        smk = float(sim_runs_k[:, bi].mean()); ssdk = float(sim_runs_k[:, bi].std(ddof=1))
        z = (tf_runs[bi] - pr_runs[bi]) / ssd if ssd > 0 else None
        zk = (tf_runs_k[bi] - pr_runs_k[bi]) / ssdk if ssdk > 0 else None
        mk = int(minK_tf[bi]) if minK_tf[bi] < (1 << 30) else None
        row = dict(u=round((b0 + bi) * BW, 2), slots=int(cnt_slots[bi]), corefree=cf,
                   om1=int(om_hist[bi][1]), om2=int(om_hist[bi][2]), om3=int(om_hist[bi][3]),
                   om4=int(om_hist[bi][4]), om5=int(om_hist[bi][5]),
                   prime_share=float(om_hist[bi][1] / cf) if cf else None,
                   open=int(cnt_open[bi]), pp_share=float(pbin[bi]), starts=int(cntst[bi]),
                   starts_kpos=int(cnt_kpos[bi]),
                   meanK=float(sumK[bi] / cntst[bi]) if cntst[bi] else None,
                   tf_starts=int(tf_starts[bi]), pred_starts=float(pr_starts[bi]),
                   tf_runs=int(tf_runs[bi]), pred_runs=float(pr_runs[bi]),
                   sim_mean=sm, sim_sd=ssd, z=z,
                   tf_runs_k=int(tf_runs_k[bi]), pred_runs_k=float(pr_runs_k[bi]),
                   sim_mean_k=smk, sim_sd_k=ssdk, z_k=zk,
                   tf_starts_k=int(tf_starts_k[bi]), pred_starts_k=float(pr_starts_k[bi]),
                   minK_tf=mk)
        rows.append(row)
        say(f"{row['u']:.1f} | {row['slots']} | {cf} | {row['om1']} | {row['om2']} | {row['om3']}"
            f" | {row['om4']} | {row['om5']} | "
            + (f"{row['prime_share']:.6f}" if row['prime_share'] is not None else "-")
            + f" | {row['open']} | {row['pp_share']:.6f} | {row['starts']} | "
              f"{row['starts_kpos']} | "
            + (f"{row['meanK']:.4f}" if row['meanK'] is not None else "-")
            + f" | {row['tf_starts']} | {row['pred_starts']:.2f} | {row['tf_runs']} | "
              f"{row['pred_runs']:.2f} | {sm:.2f} | {ssd:.2f} | "
            + (f"{z:+.2f}" if z is not None else "-")
            + f" | {row['tf_runs_k']} | {row['pred_runs_k']:.2f} | {smk:.2f} | {ssdk:.2f} | "
            + (f"{zk:+.2f}" if zk is not None else "-")
            + " | " + (str(mk) if mk is not None else "-"))
    results["L"][str(Lp)] = {
        "t": tp, "q": q, "core": ncore, "tail": ntail, "starts": int(nstart),
        "thresholds": {str(j): thresholds[j] for j in (2, 3, 4, 5)}, "fuel": fuel, "rows": rows,
        "tot": {"corefree": int(om_hist.sum()), "open": int(cnt_open.sum()),
                "pp": int(cnt_pp.sum()), "om": [int(x) for x in om_hist.sum(axis=0)],
                "meanK": float(sumK.sum() / max(1, cntst.sum())),
                "tf_starts": int(tf_starts.sum()), "tf_runs": int(tf_runs.sum()),
                "pred_starts": float(pr_starts.sum()), "pred_runs": float(pr_runs.sum()),
                "sim_runs_mean": float(sim_runs.sum(axis=1).mean()),
                "sim_runs_sd": float(sim_runs.sum(axis=1).std(ddof=1)),
                "starts_kpos": int(cnt_kpos.sum()),
                "tf_starts_k": int(tf_starts_k.sum()), "tf_runs_k": int(tf_runs_k.sum()),
                "pred_starts_k": float(pr_starts_k.sum()), "pred_runs_k": float(pr_runs_k.sum()),
                "sim_runs_k_mean": float(sim_runs_k.sum(axis=1).mean()),
                "sim_runs_k_sd": float(sim_runs_k.sum(axis=1).std(ddof=1)),
                "minK_tf": int(minK_tf.min()) if minK_tf.min() < (1 << 30) else None}}
    del binidx, rl, ru, op, cnt_slots, cnt_open, cnt_pp, pbin, pslot
    del sim_runs, sim_starts, sim_runs_k, kposmask

with open(os.path.join(OUT, f"{TAG}.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=str)
say()
say("done")
LOG.close()
print(f"wrote {log_path}")
