"""Dead-cycle record F_c of the anchor-30 machine {5..q}, exact over the full period in cycles.

Cycle j = numbers 30j+11, 13, 17, 19, 29, 31 = columns 5j+2, 5j+3, 5j+5 (numbers 6k -+ 1).
Gear g >= 7 kills the number 30j+e iff j = -e * 30^{-1} (mod g).  Sieve over j in blocks with
period-g tiles; per cycle a 3-bit state (bit t set iff slot t is blocked).  Also computed from
the same pass: the column openings (5j+o for every open slot), hence the column record F,
the gap histogram and the record gap phases mod 5 - all exact over the whole period.

Usage: uv run python research/anchor235/r34/cycle_record.py [qmax ...]   (default 7 11 13 17 19 23 29)
Writes research/anchor235/r34/results/cycle_record_<q>.json and a text summary.
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

E = (11, 13, 17, 19, 29, 31)          # in-cycle offsets, slot t = e index // 2
OFFS = (2, 3, 5)                       # column offsets of the three slots


def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def kill_pattern(g):
    """uint8 array of length g: bit t set iff gear g blocks slot t of cycles j = index (mod g)."""
    inv30 = pow(30, -1, g)
    pat = np.zeros(g, np.uint8)
    for i, e in enumerate(E):
        r = (-e * inv30) % g
        pat[r] |= 1 << (i // 2)
    # cross-check against the column rule k = +-u_g (mod g), 6u_g = g -+ 1, k = 5j + o
    u = pow(6, -1, g)
    inv5 = pow(5, -1, g)
    chk = np.zeros(g, np.uint8)
    for t, o in enumerate(OFFS):
        for tooth in (u, -u):
            chk[((tooth - o) * inv5) % g] |= 1 << t
    assert np.array_equal(pat, chk), g
    return pat


def runs_true(b, gstart, carry):
    """Runs of True in boolean b (global start gstart).  carry = (start, length) of an open run
    ending exactly at gstart, or None.  Returns (list of closed runs, new carry)."""
    d = np.diff(np.concatenate((np.zeros(1, np.int8), b.view(np.int8), np.zeros(1, np.int8))))
    starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1)
    out = []
    if len(starts) == 0:
        return ([carry] if carry else []), None
    starts = starts.astype(np.int64) + gstart; ends = ends.astype(np.int64) + gstart
    if carry is not None:
        if starts[0] == gstart:
            starts[0] = carry[0]
        else:
            out.append(carry)
    newcarry = None
    if ends[-1] == gstart + len(b):
        newcarry = (int(starts[-1]), int(ends[-1] - starts[-1]))
        starts, ends = starts[:-1], ends[:-1]
    out.extend(zip(starts.tolist(), (ends - starts).tolist()))
    return out, newcarry


class TopRuns:
    """Keeps every run of length >= (max length - keep_below)."""
    def __init__(self, keep_below=1):
        self.best = -1; self.runs = []; self.kb = keep_below
    def add(self, runs):
        for s, L in runs:
            if L > self.best:
                self.best = L
                self.runs = [r for r in self.runs if r[1] >= L - self.kb]
            if L >= self.best - self.kb:
                self.runs.append((s, L))
    def report(self):
        top = sorted(r for r in self.runs if r[1] == self.best)
        below = sorted(r for r in self.runs if self.best - self.kb <= r[1] < self.best)
        return {"max": self.best, "runs": top, "n_max": len(top), "n_below": len(below),
                "below": below[:40]}


def machine(qmax, block=1 << 22, columns=True, hist_max=512):
    gears = [p for p in primes_upto(qmax) if p >= 7]
    Pc = 1
    for g in gears: Pc *= g
    pats = {g: kill_pattern(g) for g in gears}
    t0 = time.time()
    dead = TopRuns(1); wall = TopRuns(0); colrec = TopRuns(2)
    carry_dead = carry_wall = None
    last_open_col = 0                       # column 0 is open at every machine (the anchor of the mirror)
    ghist = np.zeros(hist_max, np.int64)
    state_hist = np.zeros(8, np.int64)
    nblocks = (Pc + block - 1) // block
    for bi in range(nblocks):
        s = bi * block; B = min(block, Pc - s)
        st = np.zeros(B, np.uint8)
        for g in gears:
            off = s % g
            reps = (B + off) // g + 1
            st |= np.tile(pats[g], reps)[off:off + B]
        state_hist += np.bincount(st, minlength=8)
        isdead = st == 7
        r, carry_dead = runs_true(isdead, s, carry_dead); dead.add(r)
        nopen = 3 - (((st & 1) + ((st >> 1) & 1) + ((st >> 2) & 1)).astype(np.int8))
        r, carry_wall = runs_true(nopen <= 1, s, carry_wall); wall.add(r)
        if columns:
            j = np.arange(s, s + B, dtype=np.int64)
            openmask = np.stack([(st >> t) & 1 == 0 for t in range(3)], axis=1)      # (B,3)
            cols = (5 * j)[:, None] + np.array(OFFS, np.int64)[None, :]
            oc = cols[openmask]                                                        # sorted
            if len(oc):
                gaps = np.diff(np.concatenate(([last_open_col], oc)))
                ghist += np.bincount(np.minimum(gaps, hist_max - 1), minlength=hist_max)
                # record gaps: keep the top ones with their start column
                gmax = int(gaps.max())
                idx = np.flatnonzero(gaps >= gmax - 2)
                starts = np.concatenate(([last_open_col], oc))[idx]
                colrec.add(list(zip(starts.tolist(), gaps[idx].tolist())))
                last_open_col = int(oc[-1])
        if bi % 32 == 0 or bi == nblocks - 1:
            print(f"  {qmax}: block {bi+1}/{nblocks}  dead max {dead.best}  wall max {wall.best}  F {colrec.best}  {time.time()-t0:.0f}s", flush=True)
    if carry_dead: dead.add([carry_dead])
    if carry_wall: wall.add([carry_wall])
    assert last_open_col == 5 * Pc or not columns, (last_open_col, 5 * Pc)   # column P = 0 is open
    out = {"qmax": qmax, "gears": gears, "period_cycles": Pc, "period_columns": 5 * Pc,
           "state_hist": state_hist.tolist(), "dead_cycles": int(state_hist[7]),
           "open_slot_hist": [int(state_hist[[k for k in range(8) if 3 - bin(k).count('1') == c]].sum()) for c in range(4)],
           "F_c": dead.report(), "H_1": wall.report(), "seconds": round(time.time() - t0, 1)}
    if columns:
        nz = np.flatnonzero(ghist)
        out["F"] = colrec.best
        out["gap_hist_top"] = {int(g): int(ghist[g]) for g in nz[-12:]}
        rg = colrec.report()
        out["record_gaps"] = [(s, L, s % 5) for s, L in rg["runs"]]
        out["record_gap_phases"] = {ph: sum(1 for s, L in rg["runs"] if s % 5 == ph) for ph in (0, 2, 3)}
        out["near_record_gaps"] = [(s, L, s % 5) for s, L in rg["below"]]
        out["openings_per_period"] = int(ghist.sum())
    return out


def crt_expect(gears):
    """Exact CRT counts per period of cycles with 3,2,1,0 open slots (independent residues per gear)."""
    from itertools import combinations
    Pc = 1
    for g in gears: Pc *= g
    def prob_all_open(slots):
        p = 1.0
        for g in gears:
            pat = kill_pattern(g)
            bad = sum(1 for r in range(g) if any((pat[r] >> t) & 1 for t in slots))
            p *= (g - bad) / g
        return p
    p1 = [prob_all_open([t]) for t in range(3)]
    p2 = {c: prob_all_open(list(c)) for c in combinations(range(3), 2)}
    p3 = prob_all_open([0, 1, 2])
    # inclusion-exclusion for "all three dead"
    dead = 1 - sum(p1) + sum(p2.values()) - p3
    return {"P_slot_open": p1, "P_pair_open": {str(k): v for k, v in p2.items()}, "P_all_open": p3,
            "P_dead": dead, "expected_dead_per_period": dead * Pc}


if __name__ == "__main__":
    qs = [int(a) for a in sys.argv[1:]] or [7, 11, 13, 17, 19, 23, 29]
    cols = os.environ.get("COLUMNS_OFF") is None
    summary = []
    for q in qs:
        res = machine(q, columns=cols)
        exp = crt_expect(res["gears"])
        res["crt"] = exp
        assert abs(exp["expected_dead_per_period"] - res["dead_cycles"]) < 1e-6 * max(1, res["dead_cycles"]) + 0.5, (exp, res["dead_cycles"])
        with open(os.path.join(RES, f"cycle_record_{q}.json"), "w") as f:
            json.dump(res, f, indent=1)
        Fc = res["F_c"]["max"]; F = res.get("F")
        line = (f"{{5..{q}}}: period {res['period_cycles']} cycles, dead {res['dead_cycles']} "
                f"(CRT {exp['expected_dead_per_period']:.1f}), open-slot hist {res['open_slot_hist']}, "
                f"F_c = {Fc} at {res['F_c']['runs'][:12]} ({res['F_c']['n_max']} runs of {Fc}, {res['F_c']['n_below']} of {Fc-1}), "
                f"H_1 = {res['H_1']['max']} at {res['H_1']['runs'][:4]}")
        if cols:
            line += (f", F = {F} (floor((F-2)/5) = {(F-2)//5}), record gaps by start mod 5: {res['record_gap_phases']}, "
                     f"first record gaps (start, gap, start mod 5) = {res['record_gaps'][:6]}, "
                     f"near-record gaps = {res['near_record_gaps'][:8]}, top of spectrum {res['gap_hist_top']}")
        print(line, flush=True)
        summary.append(line)
    with open(os.path.join(RES, "cycle_record_summary.txt"), "a") as f:
        f.write("\n".join(summary) + "\n")
