"""The generated survivor set, exactly: the chains from the bases 3, 5, 7, 11, 13.

Anchor 2, 3, 5 as the clock (cycle j = 30j .. 30j + 29, slots (30j + 11, 13), (17, 19), (29, 31)).
Cuts c_1 = base, p_k = first prime >= c_k, c_{k+1} = p_k^2; section k + 1 = [c_{k+1}, c_{k+2}).
Machine 1 = the numbers in [7, c_2) coprime to 30 that survive striking by the smaller such numbers
(the base construction); machine k + 1 = the numbers in section k + 1 coprime to 30 that survive
striking by the multiples of every member of machines 1..k.  No primality test enters the
construction.  The check is an independent Eratosthenes sieve over the section (0 mismatches
expected) and gmpy2.is_prime on samples.

Per full section the four sets on the section are compared: (a) G = the generated set; (b) B = G
minus the twin members (the survivors of the saturated counter-machine V17, whose gears are the
lower primes and the section's twin members); (c) C = G minus a random subset of members of the
same size as the twin members; (d) D = the Liouville-negative charges: n = s P with P a prime >=
p_{k+1}, s coprime to 30 with an even number of prime factors (so D contains G).  Measured per
set: size; pairs at distance 2; the gap spectrum of consecutive members; forbidden patterns (a
member in class 0 of a lower gear; members at n, n + 2, n + 4; consecutive gaps (2, 4) or (4, 2));
maximality (numbers of the section coprime to the lower product that are not members); the
class census mod 7, 11, 13 and 30 with its largest deviation in standard deviations and exact
ties; the Legendre character sums mod 7, 11, 13; the mirror n -> M - 2 - n (M the multiple of 30
nearest the section's centre) hit count against chance; and the record: the longest run of
consecutive slots without a pair at distance 2, from the section's first slot to its last.  For
G the record is also computed directly as the longest run of slots struck by machines 1..k.

Sections whose end exceeds N (default 10^9) are run as prefixes [c_{k+1}, N): construction check
(gears above sqrt N skipped: by the echo theorem they add nothing below N), ends, first twin,
record; the counter-machines and D are not built on prefixes.

Usage: uv run python research/stack/r2/generated.py [N] [bases...]     (about 400 MB, one core)
"""
import sys, os, json, time, math, random
import numpy as np
import gmpy2

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
SEG = 1 << 24
GAPMAX = 60


def small_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s).astype(np.int64)


def coprime30_mask(lo, L):
    r = (np.arange(lo, lo + L, dtype=np.int64)) % 30
    return (r % 2 == 1) & (r % 3 != 0) & (r % 5 != 0)


def strike(lo, L, gears, mask):
    """mask &= (no gear divides n), striking multiples m*g with m >= 2 only (a gear never strikes itself)."""
    for g in gears:
        g = int(g)
        start = max(2 * g, -(-lo // g) * g)
        if start < lo + L:
            mask[start - lo::g] = False
    return mask


def slot_index(n):
    r = n % 30
    off = np.where(r == 11, 0, np.where(r == 17, 1, 2))
    return 3 * (n // 30) + off


def first_slot_at_or_above(c):
    j = c // 30
    for jj in (j, j + 1):
        for e, o in ((11, 0), (17, 1), (29, 2)):
            if 30 * jj + e >= c:
                return 3 * jj + o
    raise RuntimeError


def base_machine(c2):
    """the numbers in [7, c2) coprime to 30 surviving the strikes of the smaller ones: machine 1."""
    L = c2 - 7
    mask = coprime30_mask(7, L)
    for n in range(7, c2):
        if mask[n - 7]:
            mask[2 * n - 7::n] = False
    return np.flatnonzero(mask).astype(np.int64) + 7


class SetStats:
    """streaming statistics of one set of odd numbers on a section (members given per segment, sorted)."""

    def __init__(self, name, lo, hi):
        self.name, self.lo, self.hi = name, lo, hi
        self.n = 0
        self.pairs2 = 0
        self.gaps = np.zeros(GAPMAX + 2, dtype=np.int64)       # gaps[g] for even g <= GAPMAX; gaps[GAPMAX+1] = larger
        self.maxgap = 0
        self.maxgap_at = None
        self.prev = None                                          # last member of the previous segment
        self.prevgap = None
        self.pat_2_4 = 0                                          # consecutive gaps (2, 4) or (4, 2)
        self.triple = 0                                           # n, n+2, n+4 members
        self.cls = {g: np.zeros(g, dtype=np.int64) for g in (7, 11, 13, 30)}
        self.chi = {g: 0 for g in (7, 11, 13)}
        self.first = None
        self.last = None
        self.first_pair = None
        self.last_pair = None
        self.pair_slots = []                                      # slot indices of the pairs (lower member)
        self.mirror_hits = 0
        self.M = 30 * round((lo + hi) / 30)
        self._mirror_pending = {}                                 # members whose image is in a later segment
        self.legendre = {g: np.array([0] + [1 if pow(a, (g - 1) // 2, g) == 1 else -1 for a in range(1, g)]) for g in (7, 11, 13)}

    def feed(self, members, seg_lo, seg_hi):
        m = members
        if len(m) == 0:
            return
        self.n += len(m)
        if self.first is None:
            self.first = int(m[0])
        self.last = int(m[-1])
        # consecutive gaps, with carry
        if self.prev is not None:
            allm = np.concatenate(([self.prev], m))
        else:
            allm = m
        d = np.diff(allm)
        if len(d):
            small = d[d <= GAPMAX]
            self.gaps[:GAPMAX + 1] += np.bincount(small, minlength=GAPMAX + 1)
            self.gaps[GAPMAX + 1] += int((d > GAPMAX).sum())
            k = int(d.argmax())
            if d[k] > self.maxgap:
                self.maxgap, self.maxgap_at = int(d[k]), int(allm[k])
            # pattern (2,4)/(4,2) in consecutive gaps, with the carried gap
            dd = d
            if self.prevgap is not None:
                dd = np.concatenate(([self.prevgap], d))
            self.pat_2_4 += int(((dd[:-1] == 2) & (dd[1:] == 4)).sum() + ((dd[:-1] == 4) & (dd[1:] == 2)).sum())
            self.prevgap = int(d[-1])
        self.prev = int(m[-1])
        # pairs at distance 2 (both members inside allm)
        s = set()
        ip = np.flatnonzero(d == 2) if len(d) else np.array([], dtype=np.int64)
        pairs = allm[ip]
        # a pair whose lower member is the carried element was not counted in the previous segment: fine, allm includes it once
        self.pairs2 += len(pairs)
        if len(pairs):
            if self.first_pair is None:
                self.first_pair = int(pairs[0])
            self.last_pair = int(pairs[-1])
            self.pair_slots.append(slot_index(pairs))
        # triples n, n+2, n+4: consecutive gaps (2, 2)
        if len(d) > 1:
            self.triple += int(((d[:-1] == 2) & (d[1:] == 2)).sum())
        # class census and characters
        for g in (7, 11, 13, 30):
            self.cls[g] += np.bincount(m % g, minlength=g)
        for g in (7, 11, 13):
            self.chi[g] += int(self.legendre[g][m % g].sum())
        # mirror: image M - 2 - n; count members whose image is a member (images inside the section only)
        img = self.M - 2 - m
        inside = (img >= self.lo) & (img < self.hi)
        img_in_seg = inside & (img >= seg_lo) & (img < seg_hi)
        ms = set(m.tolist())
        self.mirror_hits += sum(1 for x in img[img_in_seg].tolist() if x in ms)
        # images in earlier segments: check against pending; images in later segments: store
        earlier = inside & (img < seg_lo)
        for x in img[earlier].tolist():
            if x in self._mirror_pending:
                self.mirror_hits += 1
        later = inside & (img >= seg_hi)
        for x in m[later].tolist():
            self._mirror_pending[x] = True
        # note: hits are counted once per ordered member whose image is a member (symmetric pairs count twice)

    def record(self):
        s_first = first_slot_at_or_above(self.lo)
        s_last = first_slot_at_or_above(self.hi) - 1
        nslots = s_last - s_first + 1
        if not self.pair_slots:
            return nslots, nslots, self.lo, 0
        si = np.concatenate(self.pair_slots)
        runs = np.diff(si) - 1
        head = int(si[0] - s_first)
        tail = int(s_last - si[-1])
        best, where = head, self.lo
        if len(runs):
            k = int(runs.argmax())
            if runs[k] > best:
                best, where = int(runs[k]), int(30 * (si[k] // 3) + (11, 17, 29)[int(si[k] % 3)])
        if tail > best:
            best, where = tail, int(30 * (si[-1] // 3) + (11, 17, 29)[int(si[-1] % 3)])
        return nslots, best, where, int(len(si))

    def summary(self, coprime30_count):
        nslots, best, where, npairs = self.record()
        dens = self.n / coprime30_count if coprime30_count else 0
        out = {"name": self.name, "size": self.n, "pairs_at_2": self.pairs2, "first": self.first, "last": self.last,
               "first_pair": self.first_pair, "last_pair": self.last_pair,
               "gaps": {str(g): int(self.gaps[g]) for g in range(2, GAPMAX + 1, 2) if self.gaps[g]},
               "gaps_above_max": int(self.gaps[GAPMAX + 1]), "max_gap": self.maxgap, "max_gap_at": self.maxgap_at,
               "pattern_2_4": self.pat_2_4, "triples": self.triple,
               "class0": {str(g): int(self.cls[g][0]) for g in (7, 11, 13)},
               "census_max_dev_sd": {}, "census_ties": {}, "chi": self.chi,
               "chi_over_sqrt": {str(g): round(self.chi[g] / math.sqrt(max(self.n, 1)), 3) for g in (7, 11, 13)},
               "mirror_hits": self.mirror_hits,
               "mirror_chance": round(dens * self.n, 1) if coprime30_count else None,
               "mirror_ratio": round(self.mirror_hits / (dens * self.n), 3) if self.n and dens else None,
               "slots": nslots, "record_slots": best, "record_at": where, "record_ratio": round(best / nslots, 6)}
        for g in (7, 11, 13):
            c = self.cls[g][1:]
            mean = c.mean()
            sd = math.sqrt(mean) if mean > 0 else 1
            out["census_max_dev_sd"][str(g)] = round(float(np.abs(c - mean).max() / sd), 2)
            vals = c.tolist()
            out["census_ties"][str(g)] = len(vals) - len(set(vals))
        c30 = self.cls[30]
        out["census_mod30"] = {str(r): int(c30[r]) for r in (1, 7, 11, 13, 17, 19, 23, 29)}
        return out


def run_section(lo, hi, gears, name, full, N, rng, pnext):
    """gears = every member of machines 1..k (sorted).  Returns the section's report."""
    t0 = time.time()
    hi_eff = min(hi, N)
    sq = math.isqrt(hi_eff) + 1
    gear_list = gears[gears <= sq] if not full or len(gears) > 100000 else gears
    skipped_gears = int(len(gears) - len(gear_list))
    small = small_primes(sq + 1)                       # the independent primality sieve's primes
    small = small[small >= 7]
    stats = {"G": SetStats("G", lo, hi_eff)}
    if full:
        stats["B"] = SetStats("B", lo, hi_eff)
        stats["C"] = SetStats("C", lo, hi_eff)
        stats["D"] = SetStats("D", lo, hi_eff)
    mismatches = 0
    coprime_count = 0
    missing_B = missing_C = 0
    div_by_lower_D = 0
    D_size_extra = 0
    struck_run = 0
    struck_best = 0
    struck_best_at = None
    struck_carry = 0
    struck_carry_at = None
    prev_last_member = None
    sample_members = []
    sample_non = []
    seg_lo = lo
    nseg = 0
    while seg_lo < hi_eff:
        seg_hi = min(seg_lo + SEG, hi_eff)
        L = seg_hi - seg_lo + 4                          # +4 so that n + 2 and n + 4 of the last members are visible
        cop = coprime30_mask(seg_lo, L)
        surv = strike(seg_lo, L, gear_list, cop.copy())  # the construction: coprime to 30 and to every lower gear
        # the independent check: Eratosthenes by the small primes (not the machine lists)
        isp = cop.copy()
        for p in small.tolist():
            if p * p > seg_hi + 4:
                break
            start = max(p * p, -(-seg_lo // p) * p)
            if start < seg_lo + L:
                isp[start - seg_lo::p] = False
        core = slice(0, seg_hi - seg_lo)
        mismatches += int((surv[core] != isp[core]).sum())
        coprime_count += int(cop[core].sum())
        members = np.flatnonzero(surv[core]).astype(np.int64) + seg_lo
        # struck-slot run computed from the survivor mask directly
        idx = np.arange(seg_lo, seg_hi, dtype=np.int64)
        r30 = idx % 30
        slot_lower = (r30 == 11) | (r30 == 17) | (r30 == 29)
        sl = np.flatnonzero(slot_lower)
        open_slot = surv[sl] & surv[sl + 2]
        struck = ~open_slot
        # longest run of True with carry (struck_carry = length of the run open at the previous segment's end,
        # struck_carry_at = the lower member of the slot where that run began)
        if len(struck):
            padded = np.concatenate(([False], struck, [False])).astype(np.int8)
            dpad = np.diff(padded)
            starts = np.flatnonzero(dpad == 1)
            ends = np.flatnonzero(dpad == -1)
            lens = (ends - starts).astype(np.int64)
            pos = [int(idx[sl[s]]) for s in starts.tolist()]
            if len(lens):
                if struck[0] and struck_carry:
                    lens[0] += struck_carry
                    pos[0] = struck_carry_at
                elif struck_carry > struck_best:
                    struck_best, struck_best_at = struck_carry, struck_carry_at
                k = int(lens.argmax())
                if lens[k] > struck_best:
                    struck_best, struck_best_at = int(lens[k]), pos[k]
                if struck[-1]:
                    struck_carry, struck_carry_at = int(lens[-1]), pos[-1]
                else:
                    struck_carry = 0
            else:
                if struck_carry > struck_best:
                    struck_best, struck_best_at = struck_carry, struck_carry_at
                struck_carry = 0
        if nseg % 7 == 0 and len(members):
            take = rng.choice(members, size=min(2000, len(members)), replace=False)
            sample_members.extend(take.tolist())
            non = np.flatnonzero(cop[core] & ~surv[core]).astype(np.int64) + seg_lo
            if len(non):
                sample_non.extend(rng.choice(non, size=min(2000, len(non)), replace=False).tolist())
        stats["G"].feed(members, seg_lo, seg_hi)
        if full:
            # twin members: n with n + 2 a member, and n with n - 2 a member (both inside surv with the +4 margin)
            mem_idx = members - seg_lo
            up = surv[mem_idx + 2]
            down = np.zeros(len(members), dtype=bool)
            ge2 = mem_idx >= 2
            down[ge2] = surv[mem_idx[ge2] - 2]
            if seg_lo > lo:
                pass                                          # a member at seg_lo whose partner is seg_lo - 2: handled by carry below
            twin_member = up | down
            # partner across the segment boundary (n = seg_lo or seg_lo + ... with n - 2 in the previous segment)
            if len(members) and members[0] - seg_lo < 2 and seg_lo > lo:
                if prev_last_member is not None and members[0] - prev_last_member == 2:
                    twin_member[0] = True
            B = members[~twin_member]
            ntw = int(twin_member.sum())
            drop = rng.choice(len(members), size=min(ntw, len(members)), replace=False)
            keep = np.ones(len(members), dtype=bool)
            keep[drop] = False
            C = members[keep]
            missing_B += ntw
            missing_C += int(len(members) - len(C))
            stats["B"].feed(B, seg_lo, seg_hi)
            stats["C"].feed(C, seg_lo, seg_hi)
            # D: Omega parity of the smooth part; residual after dividing out every lower gear
            resid = np.arange(seg_lo, seg_lo + L, dtype=np.int64)
            cnt = np.zeros(L, dtype=np.int16)
            for g in gears.tolist():
                pe = g
                while pe < seg_lo + L:
                    start = -(-seg_lo // pe) * pe
                    if start < seg_lo + L:
                        resid[start - seg_lo::pe] //= g
                        cnt[start - seg_lo::pe] += 1
                    pe *= g
            charge = cop & (resid > 1)                         # residual is 1 or a prime >= p_{k+1}
            Dmask = charge & (cnt % 2 == 0)
            Dm = np.flatnonzero(Dmask[core]).astype(np.int64) + seg_lo
            div_by_lower_D += int((Dmask[core] & ~surv[core]).sum())
            stats["D"].feed(Dm, seg_lo, seg_hi)
        prev_last_member = int(members[-1]) if len(members) else None
        seg_lo = seg_hi
        nseg += 1
        if nseg % 8 == 0:
            print(f"    {name}: segment {nseg} at {seg_lo:,} ({time.time() - t0:.0f}s)", flush=True)
    if struck_carry > struck_best:
        struck_best = struck_carry
    # samples through gmpy2
    bad_members = sum(1 for x in sample_members if not gmpy2.is_prime(x))
    bad_non = sum(1 for x in sample_non if gmpy2.is_prime(x))
    rep = {"section": [lo, hi], "full": full, "prefix_to": hi_eff if not full else None,
           "lower_gears": int(len(gears)), "gears_used": int(len(gear_list)), "gears_skipped_above_sqrt": skipped_gears,
           "coprime30_numbers": coprime_count, "mismatches": mismatches,
           "gmpy2_samples": {"members": len(sample_members), "members_not_prime": bad_members,
                             "non_members": len(sample_non), "non_members_prime": bad_non},
           "struck_slot_record": struck_best, "struck_slot_record_at": struck_best_at,
           "seconds": round(time.time() - t0, 1)}
    rep["sets"] = {k: v.summary(coprime_count) for k, v in stats.items()}
    rep["sets"]["G"]["smallest_is_first_prime_at_or_above_cut"] = rep["sets"]["G"]["first"] == pnext
    rep["sets"]["G"]["first_twin_offset"] = (rep["sets"]["G"]["first_pair"] - lo) if rep["sets"]["G"]["first_pair"] else None
    rep["sets"]["G"]["last_twin_gap_to_end"] = (hi_eff - rep["sets"]["G"]["last_pair"]) if (full and rep["sets"]["G"]["last_pair"]) else None
    if full:
        rep["sets"]["B"]["missing_coprime"] = missing_B
        rep["sets"]["C"]["missing_coprime"] = missing_C
        rep["sets"]["G"]["missing_coprime"] = 0 if mismatches == 0 else None
        rep["sets"]["D"]["missing_coprime"] = 0
        rep["sets"]["D"]["members_divisible_by_lower_gear"] = div_by_lower_D
        for k in ("G", "B", "C"):
            rep["sets"][k]["members_divisible_by_lower_gear"] = 0     # by construction (subsets of the survivor set); class0 counts confirm at 7, 11, 13
    return rep


def main(N, bases):
    t0 = time.time()
    rng = np.random.default_rng(20260909)
    report = {"N": N, "chains": {}}
    for b in bases:
        p = int(b) if gmpy2.is_prime(b) else int(gmpy2.next_prime(b))
        cuts = [b]
        ps = [p]
        while True:
            c = ps[-1] ** 2
            cuts.append(c)
            if c > N:
                break
            ps.append(int(gmpy2.next_prime(c)))
        print(f"base {b}: cuts {cuts[:6]} first gears {ps[:6]}", flush=True)
        machines = [base_machine(cuts[1])]
        gears = machines[0].copy()
        chain = {"cuts": cuts, "first_gears": ps, "machine1": [int(machines[0][0]), int(machines[0][-1]), int(len(machines[0]))], "sections": []}
        for k in range(1, len(cuts) - 1):
            lo, hi = cuts[k], cuts[k + 1]
            full = hi <= N
            if lo >= N:
                break
            name = f"base{b}_s{k + 1}"
            print(f"  section {k + 1} = [{lo:,}, {hi:,}) {'full' if full else 'prefix'} with {len(gears):,} lower gears", flush=True)
            rep = run_section(lo, hi, gears, name, full, N, rng, ps[k])
            rep["k"] = k + 1
            chain["sections"].append(rep)
            print(f"    mismatches {rep['mismatches']}, members {rep['sets']['G']['size']:,}, twins {rep['sets']['G']['pairs_at_2']:,}, "
                  f"first twin at +{rep['sets']['G']['first_twin_offset']}, record {rep['sets']['G']['record_slots']} of {rep['sets']['G']['slots']} slots "
                  f"(struck-run {rep['struck_slot_record']}), {rep['seconds']}s", flush=True)
            if full:
                # the next machine's gears: the survivors themselves, rebuilt from the construction (the sieve of this section)
                # (kept as the primes of the section: identical by the 0-mismatch check; regenerated cheaply from the small sieve)
                nxt = small_primes(hi)
                nxt = nxt[(nxt >= lo) & (nxt < hi)]
                assert rep["mismatches"] == 0
                gears = np.concatenate([gears, nxt])
        report["chains"][str(b)] = chain
        with open(os.path.join(RES, "generated.json"), "w") as f:
            json.dump(report, f, indent=1)
    report["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "generated.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("done", report["seconds"], "s")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    bases = [int(x) for x in sys.argv[2:]] or [3, 5, 7, 11, 13]
    main(N, bases)
