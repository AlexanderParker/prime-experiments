//! h2ladder — standalone computational test: can the merge law compute the
//! paired Jacobsthal function h_2 incrementally, matching Ziller-Morack
//! (arXiv:1706.03668, Table 1), and faster than from-scratch construction?
//!
//! Two frames, reconciled explicitly (see docs/novel/merge-law-h2-test.md):
//!
//! * **h_2 frame** (Ziller-Morack, halved coordinates as in
//!   research/jacobsthal_family.py): positions n, gears q = 3, 5, ..., y,
//!   gear q blocks n = 0 and n = -e (mod q) for the reduced difference e
//!   (even difference 2e; gear 2 is absorbed by the halving, factor 2 on all
//!   lengths). F_e(y) = max cyclic gap of the survivors;
//!   **h_2 = 2 * max over e of F_e** — the max over ALL differences, not the
//!   twin difference.
//! * **twin slot frame** (this crate): slot k = (6k-1, 6k+1), gears 5..y,
//!   gear q blocks k = +-u (mod q), 6u = 1 (mod q). This is the e = 1 class
//!   of the h_2 frame with gear 3's single free class mod 3 divided out:
//!   F_1 (halved) = F_adjacent = 3 * F_slot.
//!
//! The merge law (docs/novel/merge-law.md) applies to BOTH frames: adding a
//! coprime gear q' deletes two residue classes per lap, the deleted pair
//! shifting by -P per lap, so a run of consecutive openings is deleted
//! together in some lap iff its members' positions all lie in a two-element
//! residue set {c, c+s} mod q' (s = tooth separation: e mod q' in the h_2
//! frame, 2u mod q' in the slot frame). F(M+q') is then read off the OLD
//! word alone. The ChainScan below is that criterion as an O(1)/element
//! automaton; it is validated against direct construction in this file's
//! `sample19` mode and in research/merge_h2_ladder.py.
//!
//! Subcommands:
//!   twin           RAM twin ladder 17 -> 19 -> 23 -> 29 -> 31 (merge + construction cross-checks)
//!   construct <y>  segmented-sieve construction of F_slot(y) (feasible to y = 37)
//!   twin37 | twin41 | twin43   streamed deep rungs (period never materialised)
//!   h2_17          exhaustive h_2(17) over all difference classes (base of the family ladder)
//!   h2_19          EXACT h_2(19) by merging every level-17 class word with gear 19
//!   sample19 <K>   validate merge against from-scratch construction at 19 on K random classes + timing
//!   h2_23          EXACT h_2(23) by merging every level-19 class word with gear 23

use std::env;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use gearsuite::slot::teeth;

// ---------------------------------------------------------------------------
// The merge-law scanner: exact chain condition as a streaming automaton.
// ---------------------------------------------------------------------------

/// Streaming evaluation of F(M + q') from the openings of M, fed in order.
///
/// A maximal run of consecutive openings whose residues mod q' all lie in a
/// two-element set {c, c+s} is deleted together in some lap (every c occurs,
/// because gcd(P, q') = 1); the resulting new gap spans from the opening
/// before the run to the opening after it. Plain (undeleted) old gaps also
/// survive. `best` is the running maximum over both.
///
/// The very first fed opening has no left flank, so a run containing it is
/// under-counted; callers re-feed a wrap/overlap margin so every run is seen
/// once with full flanks.
struct ChainScan {
    q: u32,
    s: u32,
    best: u64,
    started: bool,
    last_x: u64,
    // candidate run (maximal, ending at the previous opening)
    val_a: u32,
    val_b: u32,
    has_b: bool,
    run_prev_x: u64,   // opening just before the run start (left flank)
    last_val: u32,
    block_prev_x: u64, // opening just before the trailing equal-residue block
}

impl ChainScan {
    fn new(q: u32, s: u32) -> Self {
        ChainScan {
            q,
            s: s % q,
            best: 0,
            started: false,
            last_x: 0,
            val_a: 0,
            val_b: 0,
            has_b: false,
            run_prev_x: 0,
            last_val: 0,
            block_prev_x: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, x: u64, r: u32) {
        if !self.started {
            self.started = true;
            self.last_x = x;
            self.val_a = r;
            self.has_b = false;
            self.run_prev_x = x;
            self.last_val = r;
            self.block_prev_x = x;
            return;
        }
        let g = x - self.last_x;
        if g > self.best {
            self.best = g;
        }
        let q = self.q;
        // all residues are < q, so a + s < 2q: conditional subtraction replaces `%`
        #[inline(always)]
        fn addm(a: u32, b: u32, q: u32) -> u32 {
            let t = a + b;
            if t >= q {
                t - q
            } else {
                t
            }
        }
        let r_s = addm(r, self.s, q);
        let fits = r == self.val_a
            || (self.has_b && r == self.val_b)
            || (!self.has_b && (r == addm(self.val_a, self.s, q) || self.val_a == r_s));
        if fits {
            if !self.has_b && r != self.val_a {
                self.val_b = r;
                self.has_b = true;
            }
            if r != self.last_val {
                self.last_val = r;
                self.block_prev_x = self.last_x;
            }
        } else {
            // the maximal run ended at the previous opening
            let span = x - self.run_prev_x;
            if span > self.best {
                self.best = span;
            }
            // restart: longest compatible suffix is the trailing equal-residue
            // block (plus the new opening), or the new opening alone
            if r == addm(self.last_val, self.s, q) || self.last_val == r_s {
                self.val_a = self.last_val;
                self.val_b = r;
                self.has_b = true;
                self.run_prev_x = self.block_prev_x;
            } else {
                self.val_a = r;
                self.has_b = false;
                self.run_prev_x = self.last_x;
            }
            self.last_val = r;
            self.block_prev_x = self.last_x;
        }
        self.last_x = x;
    }

    #[inline(always)]
    fn push_x(&mut self, x: u64) {
        let r = (x % self.q as u64) as u32;
        self.push(x, r);
    }
}

/// F(M + q') for a cyclic word given as sorted positions in [0, P).
/// `margin` openings are re-fed shifted by P so wrap runs are seen whole.
fn scan_cyclic_u32(positions: &[u32], p: u64, q: u32, s: u32, margin: usize) -> u64 {
    let mut sc = ChainScan::new(q, s);
    for &x in positions {
        sc.push_x(x as u64);
    }
    for &x in positions.iter().take(margin.min(positions.len())) {
        sc.push_x(x as u64 + p);
    }
    sc.best
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

fn primes_between(lo: u64, hi: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut n = lo.max(2);
    while n <= hi {
        let mut prime = n >= 2;
        let mut d = 2;
        while d * d <= n {
            if n % d == 0 {
                prime = false;
                break;
            }
            d += 1;
        }
        if prime {
            out.push(n);
        }
        n += 1;
    }
    out
}

fn threads() -> usize {
    env::var("THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18)
}

/// Max cyclic gap of sorted positions over [0, P).
fn max_cyclic_gap_u32(pos: &[u32], p: u64) -> u64 {
    let mut best = 0u64;
    for w in pos.windows(2) {
        let g = (w[1] - w[0]) as u64;
        if g > best {
            best = g;
        }
    }
    best.max(p - pos[pos.len() - 1] as u64 + pos[0] as u64)
}

// ---------------------------------------------------------------------------
// Twin slot frame: RAM ladder and streamed deep rungs.
// ---------------------------------------------------------------------------

/// Openings of the slot machine {5..y} over one period, by plain sieve.
fn slot_openings_u32(y: u64) -> (Vec<u32>, u64) {
    let gears = primes_between(5, y);
    let p: u64 = gears.iter().product();
    let mut alive = vec![true; p as usize];
    for &q in &gears {
        let (a, b) = teeth(q);
        let mut k = a;
        while k < p {
            alive[k as usize] = false;
            k += q;
        }
        let mut k = b;
        while k < p {
            alive[k as usize] = false;
            k += q;
        }
    }
    let mut pos = Vec::new();
    for (k, &al) in alive.iter().enumerate() {
        if al {
            pos.push(k as u32);
        }
    }
    (pos, p)
}

/// Thin a positions word by gear q (teeth in absolute position space),
/// producing the full new word: q laps, each a copy of the old word with the
/// two newly blocked residue classes deleted.
fn thin_positions(base: &[u32], p_base: u64, q: u64, t1: u64, t2: u64) -> (Vec<u32>, u64) {
    let p_new = p_base * q;
    assert!(p_new <= u32::MAX as u64, "thinned word exceeds u32 positions");
    let mut out = Vec::with_capacity(base.len() * (q as usize - 2));
    // residues of base positions mod q, computed once
    let rbase: Vec<u32> = base.iter().map(|&x| (x as u64 % q) as u32) .collect();
    let (t1, t2) = (t1 as u32, t2 as u32);
    for l in 0..q {
        let off = l * p_base;
        let roff = (off % q) as u32;
        for (i, &x) in base.iter().enumerate() {
            let mut r = rbase[i] + roff;
            if r >= q as u32 {
                r -= q as u32;
            }
            if r != t1 && r != t2 {
                out.push((x as u64 + off) as u32);
            }
        }
    }
    (out, p_new)
}

/// A big word stored as u8 gaps (cyclic): openings first, first+g0, ...
struct Word8 {
    gaps: Vec<u8>,
    first: u64,
    period: u64,
}

/// Build the slot-frame twin word at machine 29 (214,708,725 openings) by
/// sieving machine 23 and thinning with gear 29 — never materialising
/// positions of the 29-word (gaps fit in u8: F_slot(29) = 43).
fn build_word29() -> Word8 {
    let t0 = Instant::now();
    let (w23, p23) = slot_openings_u32(23);
    println!(
        "  machine 23 sieved: {} openings, period {}, F_slot = {} ({:.1}s)",
        w23.len(),
        p23,
        max_cyclic_gap_u32(&w23, p23),
        t0.elapsed().as_secs_f64()
    );
    let q = 29u64;
    let (t1, t2) = teeth(q);
    let p29 = p23 * q;
    let mut gaps = Vec::with_capacity(w23.len() * 27);
    let rbase: Vec<u32> = w23.iter().map(|&x| (x as u64 % q) as u32).collect();
    let mut first = u64::MAX;
    let mut prev = 0u64;
    let mut maxg = 0u64;
    for l in 0..q {
        let off = l * p23;
        let roff = (off % q) as u32;
        for (i, &x) in w23.iter().enumerate() {
            let mut r = rbase[i] + roff;
            if r >= q as u32 {
                r -= q as u32;
            }
            if r != t1 as u32 && r != t2 as u32 {
                let ax = x as u64 + off;
                if first == u64::MAX {
                    first = ax;
                } else {
                    let g = ax - prev;
                    if g > maxg {
                        maxg = g;
                    }
                    gaps.push(g as u8);
                }
                prev = ax;
            }
        }
    }
    // closing wrap gap
    let g = first + p29 - prev;
    if g > maxg {
        maxg = g;
    }
    gaps.push(g as u8);
    println!(
        "  machine 29 word built by merge laps: {} openings, period {}, F_slot = {} ({:.1}s total)",
        gaps.len(),
        p29,
        maxg,
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(gaps.len(), 214_708_725, "A(29) mismatch");
    assert_eq!(maxg, 43, "F_slot(29) must be 43 (adjacent 129)");
    Word8 {
        gaps,
        first,
        period: p29,
    }
}

/// The RAM twin ladder 17 -> 31 with merge values and in-RAM construction
/// cross-checks at every rung.
fn twin_ladder() {
    println!("TWIN SLOT-FRAME LADDER (merge law vs construction), lengths in slots; adjacent = 3x");
    let (w17, p17) = slot_openings_u32(17);
    println!(
        "  machine 17: {} openings, period {}, F_slot = {}",
        w17.len(),
        p17,
        max_cyclic_gap_u32(&w17, p17)
    );

    // 17 -> 19
    let q = 19u64;
    let s = (2 * gearsuite::slot::tooth_offset(q)) % q; // separation 2u mod q
    let t = Instant::now();
    let f19 = scan_cyclic_u32(&w17, p17, q as u32, s as u32, 256);
    let dt_merge = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (tw1, tw2) = teeth(q);
    let (w19, p19) = thin_positions(&w17, p17, q, tw1, tw2);
    let f19c = max_cyclic_gap_u32(&w19, p19);
    let dt_con = t.elapsed().as_secs_f64();
    report_rung(17, 19, f19, Some(f19c), dt_merge, Some(dt_con), 25);

    // 19 -> 23
    let q = 23u64;
    let s = (2 * gearsuite::slot::tooth_offset(q)) % q;
    let t = Instant::now();
    let f23 = scan_cyclic_u32(&w19, p19, q as u32, s as u32, 256);
    let dt_merge = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (tw1, tw2) = teeth(q);
    let (w23, p23) = thin_positions(&w19, p19, q, tw1, tw2);
    let f23c = max_cyclic_gap_u32(&w23, p23);
    let dt_con = t.elapsed().as_secs_f64();
    report_rung(19, 23, f23, Some(f23c), dt_merge, Some(dt_con), 34);

    // 23 -> 29
    let q = 29u64;
    let s = (2 * gearsuite::slot::tooth_offset(q)) % q;
    let t = Instant::now();
    let f29 = scan_cyclic_u32(&w23, p23, q as u32, s as u32, 256);
    let dt_merge = t.elapsed().as_secs_f64();
    drop(w17);
    drop(w19);
    let t = Instant::now();
    let w29 = build_word29();
    let dt_con = t.elapsed().as_secs_f64();
    report_rung(23, 29, f29, Some(43), dt_merge, Some(dt_con), 43);

    // 29 -> 31 (merge only in RAM; construction via `construct 31`)
    let q = 31u32;
    let s = ((2 * gearsuite::slot::tooth_offset(q as u64)) % q as u64) as u32;
    let t = Instant::now();
    let mut sc = ChainScan::new(q, s);
    let mut x = w29.first;
    sc.push_x(x);
    for &g in &w29.gaps[..w29.gaps.len() - 1] {
        x += g as u64;
        sc.push_x(x);
    }
    // wrap margin
    x = w29.first + w29.period;
    sc.push_x(x);
    let mut fed = 1usize;
    for &g in &w29.gaps[..w29.gaps.len() - 1] {
        x += g as u64;
        sc.push_x(x);
        fed += 1;
        if fed > 256 {
            break;
        }
    }
    let f31 = sc.best;
    let dt_merge = t.elapsed().as_secs_f64();
    report_rung(29, 31, f31, None, dt_merge, None, 58);
    println!("  (construction at 31: run `h2ladder construct 31` — segmented sieve over P = 33,426,748,355)");
}

fn report_rung(
    y: u64,
    q: u64,
    f_merge: u64,
    f_construct: Option<u64>,
    dt_merge: f64,
    dt_con: Option<f64>,
    expect_slot: u64,
) {
    let ok = f_merge == expect_slot;
    let con = match f_construct {
        Some(c) => format!(
            "construction {} ({}) in {:.2}s",
            c,
            if c == f_merge { "agrees" } else { "DISAGREES" },
            dt_con.unwrap_or(0.0)
        ),
        None => "construction not run here".to_string(),
    };
    println!(
        "  {} -> {}: merge F_slot = {} (adjacent {}) in {:.4}s; {}; corpus {} => {}",
        y,
        q,
        f_merge,
        3 * f_merge,
        dt_merge,
        con,
        3 * expect_slot,
        if ok { "MATCH" } else { "MISMATCH  <-- HEADLINE" }
    );
    assert!(ok, "twin rung {} -> {} gave {} expected {}", y, q, f_merge, expect_slot);
}

/// Segmented-sieve construction of F_slot(y): the honest from-scratch cost.
fn construct(y: u64) {
    let gears = primes_between(5, y);
    let p: u64 = gears.iter().product();
    let nthreads = threads();
    println!(
        "CONSTRUCTION machine {} by segmented sieve: period {} ({} threads)",
        y, p, nthreads
    );
    let t0 = Instant::now();
    let nblocks = (nthreads * 8) as u64;
    let block = (p + nblocks - 1) / nblocks;
    let next = AtomicU64::new(0);
    // per block: (index, first opening, last opening, max internal gap)
    let results: Mutex<Vec<(u64, u64, u64, u64)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| {
                const CHUNK: usize = 1 << 22;
                let mut buf = vec![true; CHUNK];
                loop {
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    if b >= nblocks {
                        break;
                    }
                    let lo = b * block;
                    let hi = (lo + block).min(p);
                    if lo >= hi {
                        continue;
                    }
                    let mut first = u64::MAX;
                    let mut last = 0u64;
                    let mut prev = u64::MAX;
                    let mut maxg = 0u64;
                    let mut start = lo;
                    while start < hi {
                        let end = (start + CHUNK as u64).min(hi);
                        let n = (end - start) as usize;
                        buf[..n].iter_mut().for_each(|v| *v = true);
                        for &q in &gears {
                            let (a, bb) = teeth(q);
                            for t in [a, bb] {
                                let mut k = t + ((start + q - 1 - t) / q) * q;
                                if k < start {
                                    k += q;
                                }
                                while k < end {
                                    buf[(k - start) as usize] = false;
                                    k += q;
                                }
                            }
                        }
                        for i in 0..n {
                            if buf[i] {
                                let x = start + i as u64;
                                if first == u64::MAX {
                                    first = x;
                                } else {
                                    let g = x - prev;
                                    if g > maxg {
                                        maxg = g;
                                    }
                                }
                                prev = x;
                                last = x;
                            }
                        }
                        start = end;
                    }
                    results.lock().unwrap().push((b, first, last, maxg));
                }
            });
        }
    });
    let mut rs = results.into_inner().unwrap();
    rs.sort();
    let mut maxg = 0u64;
    let mut prev_last = u64::MAX;
    let mut global_first = u64::MAX;
    let mut global_last = 0u64;
    for &(_, first, last, mg) in &rs {
        if first == u64::MAX {
            continue;
        }
        if mg > maxg {
            maxg = mg;
        }
        if prev_last != u64::MAX {
            let g = first - prev_last;
            if g > maxg {
                maxg = g;
            }
        } else {
            global_first = first;
        }
        prev_last = last;
        global_last = last;
    }
    let wrap = global_first + p - global_last;
    if wrap > maxg {
        maxg = wrap;
    }
    println!(
        "  F_slot({}) = {} (adjacent {}) by construction, {:.2}s",
        y,
        maxg,
        3 * maxg,
        t0.elapsed().as_secs_f64()
    );
}

/// Streamed deep rungs. Filters are the gears above 29 applied lap-wise to
/// the RAM word at 29; the scanned word (level 31 / 37 / 41) is never
/// materialised. Parallel over top-level lap tuples, with junction stitching.
struct UnitRes {
    idx: usize,
    count: u64,
    max_plain: u64,
    scan_best: u64,
    head: Vec<u64>,
    tail: Vec<u64>, // last H survivors, in order
}

const H_MARGIN: usize = 1024;

fn twin_deep(target: u64) {
    let nthreads = threads();
    println!(
        "STREAMED TWIN RUNG -> {} ({} threads): scan the level-{} word generated lap-wise from the RAM word at 29",
        target,
        nthreads,
        match target {
            37 => 31,
            41 => 37,
            43 => 41,
            _ => panic!("target must be 37, 41 or 43"),
        }
    );
    let w29 = build_word29();
    let p29 = w29.period;
    let p31 = p29 * 31;
    let p37 = p31 * 37;
    let p41 = p37 * 41;
    let (t31a, t31b) = teeth(31);
    let (t37a, t37b) = teeth(37);
    let (t41a, t41b) = teeth(41);
    let su = |q: u64| ((2 * gearsuite::slot::tooth_offset(q)) % q) as u32;

    // units: contiguous x-ranges in scan order
    let (units, scanned_period, scan_q, scan_s): (Vec<u64>, u64, u32, u32) = match target {
        37 => ((0..31).collect(), p31, 37, su(37)),
        41 => ((0..37u64 * 31).collect(), p37, 41, su(41)),
        43 => ((0..41u64 * 37).collect(), p41, 43, su(43)),
        _ => unreachable!(),
    };
    // precompute gap residue tables for incremental residues
    let tbl = |q: u64| -> Vec<u32> { (0..256).map(|g| (g % q) as u32).collect() };
    let tbl31 = tbl(31);
    let tbl37 = tbl(37);
    let tbl41 = tbl(41);
    let tblscan = tbl(scan_q as u64);

    let t0 = Instant::now();
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<UnitRes>> = Mutex::new(Vec::new());
    let gaps = &w29.gaps;
    let nunits = units.len();
    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| loop {
                let ui = next.fetch_add(1, Ordering::Relaxed);
                if ui >= nunits {
                    break;
                }
                // absolute offset and inner lap count for this unit
                let (off, inner_laps) = match target {
                    37 => (units[ui] * p29, 1u64),
                    41 => {
                        let l37 = units[ui] / 31;
                        let l31 = units[ui] % 31;
                        (l37 * p31 + l31 * p29, 1u64)
                    }
                    43 => {
                        let l41 = units[ui] / 37;
                        let l37 = units[ui] % 37;
                        (l41 * p37 + l37 * p31, 31u64)
                    }
                    _ => unreachable!(),
                };
                let mut sc = ChainScan::new(scan_q, scan_s);
                let mut count = 0u64;
                let mut max_plain = 0u64;
                let mut prev = u64::MAX;
                let mut head: Vec<u64> = Vec::with_capacity(H_MARGIN);
                let mut tail: Vec<u64> = vec![0; H_MARGIN];
                let mut tlen = 0usize;
                let mut tpos = 0usize;

                let mut x = off + w29.first;
                let mut r31 = (x % 31) as u32;
                let mut r37 = (x % 37) as u32;
                let mut r41 = (x % 41) as u32;
                let mut rs = (x % scan_q as u64) as u32;
                let total = inner_laps as usize * gaps.len();
                let mut gi = 0usize;
                for _ in 0..total {
                    // filter chain depends on target
                    let alive = match target {
                        37 => r31 != t31a as u32 && r31 != t31b as u32,
                        41 => {
                            (r31 != t31a as u32 && r31 != t31b as u32)
                                && (r37 != t37a as u32 && r37 != t37b as u32)
                        }
                        43 => {
                            (r31 != t31a as u32 && r31 != t31b as u32)
                                && (r37 != t37a as u32 && r37 != t37b as u32)
                                && (r41 != t41a as u32 && r41 != t41b as u32)
                        }
                        _ => unreachable!(),
                    };
                    if alive {
                        count += 1;
                        if prev != u64::MAX {
                            let g = x - prev;
                            if g > max_plain {
                                max_plain = g;
                            }
                        }
                        prev = x;
                        sc.push(x, rs);
                        if head.len() < H_MARGIN {
                            head.push(x);
                        }
                        tail[tpos] = x;
                        tpos = (tpos + 1) % H_MARGIN;
                        if tlen < H_MARGIN {
                            tlen += 1;
                        }
                    }
                    // advance
                    let g = gaps[gi] as u32;
                    gi += 1;
                    if gi == gaps.len() {
                        gi = 0;
                    }
                    x += g as u64;
                    r31 += tbl31[g as usize];
                    if r31 >= 31 {
                        r31 -= 31;
                    }
                    r37 += tbl37[g as usize];
                    if r37 >= 37 {
                        r37 -= 37;
                    }
                    r41 += tbl41[g as usize];
                    if r41 >= 41 {
                        r41 -= 41;
                    }
                    rs += tblscan[g as usize];
                    if rs >= scan_q {
                        rs -= scan_q;
                    }
                }
                // unroll tail ring buffer
                let mut tail_sorted = Vec::with_capacity(tlen);
                for i in 0..tlen {
                    tail_sorted.push(tail[(tpos + H_MARGIN - tlen + i) % H_MARGIN]);
                }
                results.lock().unwrap().push(UnitRes {
                    idx: ui,
                    count,
                    max_plain,
                    scan_best: sc.best,
                    head,
                    tail: tail_sorted,
                });
            });
        }
    });
    let mut rs = results.into_inner().unwrap();
    rs.sort_by_key(|r| r.idx);
    let mut total_count = 0u64;
    let mut f_prev_level = 0u64; // max gap of the scanned word = construction value of level-\ell
    let mut f_merge = 0u64;
    for r in &rs {
        total_count += r.count;
        f_prev_level = f_prev_level.max(r.max_plain);
        f_merge = f_merge.max(r.scan_best);
    }
    // junction stitching (consecutive units + wrap)
    for i in 0..rs.len() {
        let a = &rs[i];
        let b = &rs[(i + 1) % rs.len()];
        let shift = if i + 1 == rs.len() { scanned_period } else { 0 };
        let mut window: Vec<u64> = Vec::with_capacity(a.tail.len() + b.head.len());
        window.extend_from_slice(&a.tail);
        window.extend(b.head.iter().map(|&x| x + shift));
        let mut sc = ChainScan::new(scan_q, scan_s);
        let mut prev = u64::MAX;
        for &x in &window {
            if prev != u64::MAX {
                let g = x - prev;
                f_prev_level = f_prev_level.max(g);
            }
            prev = x;
            sc.push_x(x);
        }
        f_merge = f_merge.max(sc.best);
    }
    let dt = t0.elapsed().as_secs_f64();
    let (expect, prev_expect, atag) = match target {
        37 => (88u64, 58u64, "F(2,37)=264"),
        41 => (91, 88, "F(2,41)=273"),
        43 => (103, 91, "F(2,43)=309"),
        _ => unreachable!(),
    };
    println!(
        "  scanned word: {} openings (level-{} construction check: F_slot = {}, corpus {} => {})",
        total_count,
        match target {
            37 => 31,
            41 => 37,
            _ => 41,
        },
        f_prev_level,
        prev_expect,
        if f_prev_level == prev_expect { "MATCH" } else { "MISMATCH <-- HEADLINE" }
    );
    println!(
        "  MERGE RESULT: F_slot(machine + {}) = {} (adjacent {}), in {:.1}s; corpus {} => {}",
        target,
        f_merge,
        3 * f_merge,
        dt,
        atag,
        if f_merge == expect { "MATCH" } else { "MISMATCH <-- HEADLINE" }
    );
    assert_eq!(f_prev_level, prev_expect, "scanned-word max gap mismatch");
    assert_eq!(f_merge, expect, "merge value mismatch at target {}", target);
}

// ---------------------------------------------------------------------------
// h_2 frame: the family ladder.
// ---------------------------------------------------------------------------

const H2_GEARS: [u64; 6] = [3, 5, 7, 11, 13, 17];
const P17H: u64 = 255_255; // 3*5*7*11*13*17

/// Sieve the level-17 word of difference class c: positions n in [0, P17H)
/// with n != 0 and n != -c (mod q) for every gear q <= 17.
fn sieve_h2_word17(c: u32, buf: &mut [bool], out: &mut Vec<u32>) {
    buf.iter_mut().for_each(|v| *v = true);
    for &q in &H2_GEARS {
        let t2 = ((q - (c as u64 % q)) % q) as usize;
        let mut k = 0usize;
        while k < buf.len() {
            buf[k] = false;
            k += q as usize;
        }
        let mut k = t2;
        while k < buf.len() {
            buf[k] = false;
            k += q as usize;
        }
    }
    out.clear();
    for (i, &al) in buf.iter().enumerate() {
        if al {
            out.push(i as u32);
        }
    }
}

/// Exhaustive h_2(17) over all difference classes (the family base; also the
/// merge run's own verification that the base level is right).
fn h2_17() {
    let nthreads = threads();
    println!(
        "EXHAUSTIVE h_2(17): all {} difference classes mod {} ({} threads)",
        P17H / 2 + 1,
        P17H,
        nthreads
    );
    let t0 = Instant::now();
    let next = AtomicU64::new(0);
    let best = AtomicU64::new(0);
    let arg = Mutex::new(0u64);
    let ncls = P17H / 2 + 1;
    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| {
                let mut buf = vec![true; P17H as usize];
                let mut word = Vec::with_capacity(96_000);
                loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c > P17H / 2 {
                        break;
                    }
                    sieve_h2_word17(c as u32, &mut buf, &mut word);
                    if word.len() < 2 {
                        continue;
                    }
                    let f = max_cyclic_gap_u32(&word, P17H);
                    if f > best.load(Ordering::Relaxed) {
                        let old = best.fetch_max(f, Ordering::Relaxed);
                        if f > old {
                            *arg.lock().unwrap() = c;
                        }
                    }
                }
            });
        }
    });
    let f = best.load(Ordering::Relaxed);
    let e = *arg.lock().unwrap();
    println!(
        "  max F_e(17) = {} at e = {} => h_2(17) = {}; ZM Table 1: 192 => {} ({:.1}s, {} classes)",
        f,
        e,
        2 * f,
        if 2 * f == 192 { "MATCH" } else { "MISMATCH <-- HEADLINE" },
        t0.elapsed().as_secs_f64(),
        ncls
    );
    assert_eq!(2 * f, 192);
}

/// EXACT h_2(19) by the merge law: for every level-17 class word and every
/// residue of e mod 19, F_e(19) is read off the level-17 word by the chain
/// condition. The level-19 period (4,849,845) is never constructed.
fn h2_19() {
    let nthreads = threads();
    println!(
        "EXACT h_2(19) BY MERGE: {} level-17 classes x 19 residues ({} threads)",
        P17H / 2 + 1,
        nthreads
    );
    let t0 = Instant::now();
    let next = AtomicU64::new(0);
    let best = AtomicU64::new(0); // halved F at 19
    let best17 = AtomicU64::new(0);
    let arg = Mutex::new((0u64, 0u32)); // (c, s)
    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| {
                let mut buf = vec![true; P17H as usize];
                let mut word = Vec::with_capacity(96_000);
                loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c > P17H / 2 {
                        break;
                    }
                    sieve_h2_word17(c as u32, &mut buf, &mut word);
                    if word.len() < 2 {
                        continue;
                    }
                    best17.fetch_max(max_cyclic_gap_u32(&word, P17H), Ordering::Relaxed);
                    // one scanner per residue class of e mod 19 up to sign:
                    // scanners s and 19-s accept exactly the same two-element
                    // tooth sets {c, c+s}, so s = 0..=9 covers all 19 residues
                    let mut scans: Vec<ChainScan> = (0..=9).map(|s| ChainScan::new(19, s)).collect();
                    let feed = |scans: &mut Vec<ChainScan>, x: u64| {
                        let r = (x % 19) as u32;
                        for sc in scans.iter_mut() {
                            sc.push(x, r);
                        }
                    };
                    for &x in &word {
                        feed(&mut scans, x as u64);
                    }
                    for &x in word.iter().take(96) {
                        feed(&mut scans, x as u64 + P17H);
                    }
                    for (s, sc) in scans.iter().enumerate() {
                        let f = sc.best;
                        if f > best.load(Ordering::Relaxed) {
                            let old = best.fetch_max(f, Ordering::Relaxed);
                            if f > old {
                                *arg.lock().unwrap() = (c, s as u32);
                            }
                        }
                    }
                }
            });
        }
    });
    let f = best.load(Ordering::Relaxed);
    let f17 = best17.load(Ordering::Relaxed);
    let (c, s) = *arg.lock().unwrap();
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "  base check: max F_e(17) = {} => h_2(17) = {} (ZM 192: {})",
        f17,
        2 * f17,
        if 2 * f17 == 192 { "MATCH" } else { "MISMATCH <-- HEADLINE" }
    );
    // argmax difference by CRT (s is e mod 19 up to sign)
    let e1 = crt2(c, P17H, s as u64, 19);
    let e2 = crt2(c, P17H, (19 - s as u64) % 19, 19);
    println!(
        "  max F_e(19) = {} at class (e = {} mod {}, e = +-{} mod 19) => e in {{{}, {}}} (d = 2e)",
        f, c, P17H, s, e1.min(e2), e1.max(e2)
    );
    println!(
        "  h_2(19) = {}; ZM Table 1: 258 => {}  ({:.1}s total merge run)",
        2 * f,
        if 2 * f == 258 { "MATCH" } else { "MISMATCH <-- HEADLINE" },
        dt
    );
    assert_eq!(2 * f, 258, "h_2(19) by merge must equal ZM's 258");
}

fn crt2(a: u64, m: u64, b: u64, n: u64) -> u64 {
    // smallest x >= 0 with x = a mod m, x = b mod n (m, n coprime, small n)
    let mut x = a;
    while x % n != b {
        x += m;
    }
    x
}

/// Validate merge against from-scratch construction at 19 on K classes, and
/// time both paths.
fn sample19(k: u64) {
    let p19: u64 = 4_849_845;
    let gears19: Vec<u64> = vec![3, 5, 7, 11, 13, 17, 19];
    println!(
        "SAMPLED VALIDATION AT 19: {} random classes, merge vs from-scratch sieve of P = {}",
        k, p19
    );
    let mut buf17 = vec![true; P17H as usize];
    let mut word17 = Vec::with_capacity(96_000);
    let mut buf19 = vec![true; p19 as usize];
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut t_merge = 0.0f64;
    let mut t_con = 0.0f64;
    let mut checked = 0;
    for _ in 0..k {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let c = (rng >> 16) % (P17H / 2 + 1);
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let s = (rng >> 16) % 19;
        let e = crt2(c, P17H, s, 19); // a representative difference with e mod 19 = +s

        let t = Instant::now();
        sieve_h2_word17(c as u32, &mut buf17, &mut word17);
        let f_merge = scan_cyclic_u32(&word17, P17H, 19, s as u32, 96);
        t_merge += t.elapsed().as_secs_f64();

        let t = Instant::now();
        buf19.iter_mut().for_each(|v| *v = true);
        for &q in &gears19 {
            let t2 = ((q - (e % q)) % q) as usize;
            let mut i = 0usize;
            while i < buf19.len() {
                buf19[i] = false;
                i += q as usize;
            }
            let mut i = t2;
            while i < buf19.len() {
                buf19[i] = false;
                i += q as usize;
            }
        }
        let mut prev = u64::MAX;
        let mut first = 0u64;
        let mut maxg = 0u64;
        let mut last = 0u64;
        for (i, &al) in buf19.iter().enumerate() {
            if al {
                let x = i as u64;
                if prev == u64::MAX {
                    first = x;
                } else {
                    let g = x - prev;
                    if g > maxg {
                        maxg = g;
                    }
                }
                prev = x;
                last = x;
            }
        }
        let f_direct = maxg.max(first + p19 - last);
        t_con += t.elapsed().as_secs_f64();

        assert_eq!(
            f_merge, f_direct,
            "MERGE LAW WRONG at class (c={}, s={}, e={}): merge {} vs construction {}  <-- HEADLINE",
            c, s, e, f_merge, f_direct
        );
        checked += 1;
    }
    println!(
        "  {} classes checked: merge == construction on every one (exact agreement)",
        checked
    );
    println!(
        "  timing per class: merge (17-sieve + scan) {:.2} ms vs from-scratch 19-sieve {:.2} ms",
        1000.0 * t_merge / k as f64,
        1000.0 * t_con / k as f64
    );
    println!(
        "  full-scan extrapolation at 19: {} classes x {:.2} ms = {:.0} s from scratch; the merge run reuses each 17-sieve for all 19 residues (measured total in `h2_19`)",
        2_424_922u64,
        1000.0 * t_con / k as f64,
        2_424_922f64 * (t_con / k as f64)
    );
}

/// EXACT h_2(23) by the merge law: for every level-19 class word (generated
/// lap-wise from its level-17 parent, never sieved at period 4,849,845 x 23)
/// and every residue of e mod 23, F_e(23) via the chain condition.
///
/// `prune = false` (the default, exact with no caveats) scans every word.
/// `prune = true` skips words with max(F2, best window of gaps >= 22 plus
/// flanks) <= running best. CAUTION: that threshold is borrowed from the
/// deletion-spacing lemma, whose premise (old gaps >= 3, adjacent teeth)
/// does NOT transfer to the halved frame — classes with 3 | e have gaps of 1
/// and an interior gap can equal the tooth separation s, so the pruned mode
/// is a heuristic accelerator, not exact-safe. It is kept only to document
/// the measured speed difference; the reported h_2(23) comes from the full
/// scan.
fn h2_23(prune: bool) {
    let nthreads = threads();
    let p19 = P17H * 19;
    println!(
        "EXACT h_2(23) BY MERGE: {} level-17 classes x 19 x 23 residues ({} threads)",
        P17H / 2 + 1,
        nthreads
    );
    let t0 = Instant::now();
    let next = AtomicU64::new(0);
    let best = AtomicU64::new(0);
    let arg = Mutex::new((0u64, 0u32, 0u32));
    let pruned = AtomicU64::new(0);
    let scanned = AtomicU64::new(0);
    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| {
                let mut buf = vec![true; P17H as usize];
                let mut word17: Vec<u32> = Vec::with_capacity(96_000);
                let mut word19: Vec<u32> = Vec::with_capacity(1_800_000);
                loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c > P17H / 2 {
                        break;
                    }
                    sieve_h2_word17(c as u32, &mut buf, &mut word17);
                    if word17.len() < 2 {
                        continue;
                    }
                    let rbase: Vec<u32> =
                        word17.iter().map(|&x| x % 19).collect();
                    for s19 in 0..19u32 {
                        // teeth of gear 19 for e = s19 mod 19: {0, 19 - s19}
                        let t1 = 0u32;
                        let t2 = ((19 - s19) % 19) as u32;
                        word19.clear();
                        for l in 0..19u64 {
                            let off = l * P17H;
                            let roff = (off % 19) as u32;
                            for (i, &x) in word17.iter().enumerate() {
                                let mut r = rbase[i] + roff;
                                if r >= 19 {
                                    r -= 19;
                                }
                                if r != t1 && r != t2 {
                                    word19.push((x as u64 + off) as u32);
                                }
                            }
                        }
                        if word19.len() < 2 {
                            continue;
                        }
                        if prune {
                            // heuristic bound, NOT exact-safe (see fn docs)
                            let b = best.load(Ordering::Relaxed);
                            let ub = upper_bound_23(&word19, p19);
                            if ub <= b {
                                pruned.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                        }
                        scanned.fetch_add(1, Ordering::Relaxed);
                        // s and 23-s are the same scanner (see h2_19): 0..=11
                        let mut scans: Vec<ChainScan> =
                            (0..=11).map(|s| ChainScan::new(23, s)).collect();
                        let feed = |scans: &mut Vec<ChainScan>, x: u64| {
                            let r = (x % 23) as u32;
                            for sc in scans.iter_mut() {
                                sc.push(x, r);
                            }
                        };
                        for &x in &word19 {
                            feed(&mut scans, x as u64);
                        }
                        for &x in word19.iter().take(96) {
                            feed(&mut scans, x as u64 + p19);
                        }
                        for (s23, sc) in scans.iter().enumerate() {
                            let f = sc.best;
                            if f > best.load(Ordering::Relaxed) {
                                let old = best.fetch_max(f, Ordering::Relaxed);
                                if f > old {
                                    *arg.lock().unwrap() = (c, s19, s23 as u32);
                                }
                            }
                        }
                    }
                    if c % 10000 == 0 {
                        eprintln!(
                            "  ... c = {}/{}, best halved F = {}, {:.0}s",
                            c,
                            P17H / 2,
                            best.load(Ordering::Relaxed),
                            t0.elapsed().as_secs_f64()
                        );
                    }
                }
            });
        }
    });
    let f = best.load(Ordering::Relaxed);
    let (c, s19, s23) = *arg.lock().unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let e19 = crt2(c, P17H, s19 as u64, 19);
    let e = crt2(e19, p19, s23 as u64, 23);
    println!(
        "  max F_e(23) = {} at class (e = {} mod {}, +-{} mod 19, +-{} mod 23); one representative e = {}",
        f, c, P17H, s19, s23, e
    );
    println!(
        "  h_2(23) = {}; ZM Table 1: 366 => {}  ({:.1}s; {} words scanned, {} pruned by the exact bound)",
        2 * f,
        if 2 * f == 366 { "MATCH" } else { "MISMATCH <-- HEADLINE" },
        dt,
        scanned.load(Ordering::Relaxed),
        pruned.load(Ordering::Relaxed)
    );
    if pruned.load(Ordering::Relaxed) > 0 {
        println!("  NOTE: pruned mode is a heuristic accelerator, NOT exact-safe; the exact run is the unpruned one.");
    }
    assert_eq!(2 * f, 366, "h_2(23) by merge must equal ZM's 366");
}

/// Exact-safe upper bound on F_e(M + 23) from the level-19 word: every chain's
/// interior gaps are >= 22 (deletion-spacing), so any merged span lies inside
/// flank + (maximal run of gaps >= 22) + flank; k <= 1 spans are <= F2.
fn upper_bound_23(word: &[u32], p: u64) -> u64 {
    let n = word.len();
    let gap = |i: usize| -> u64 {
        if i + 1 < n {
            (word[i + 1] - word[i]) as u64
        } else {
            p - word[n - 1] as u64 + word[0] as u64
        }
    };
    let mut best = 0u64;
    let mut prev_g = gap(n - 1);
    let mut i = 0usize;
    while i < n {
        let g = gap(i);
        // F2 candidate
        let f2 = prev_g + g;
        if f2 > best {
            best = f2;
        }
        if g >= 22 {
            // maximal run of gaps >= 22 starting at i
            let flank_l = prev_g;
            let mut sum = 0u64;
            let mut j = i;
            while j < i + n {
                let gj = gap(j % n);
                if gj >= 22 {
                    sum += gj;
                    j += 1;
                } else {
                    break;
                }
            }
            let flank_r = gap(j % n);
            let cand = flank_l + sum + flank_r;
            if cand > best {
                best = cand;
            }
            prev_g = gap((j - 1) % n);
            i = j; // skip past the run
        } else {
            prev_g = g;
            i += 1;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Operation counts: the machine-independent cost of each path.
//
// Both code paths are deterministic, so their elementary-operation counts are
// closed-form; this mode prints them per rung and verifies the closed forms
// against instrumented counters on the rungs small enough to run inline.
//
// Definitions (one "op" = one elementary visit):
//   merge path   = generation visits (walking the old word's laps, applying
//                  deletions) + scanner pushes (chain-condition checks; one
//                  push = one letter fed to one scanner) + base-word sieve
//                  strikes/cells where the path sieves a base word.
//   construction = sieve strikes (sum over gears of teeth-per-gear * P/q)
//                  + P cells scanned to read off the gaps.
// ---------------------------------------------------------------------------

fn ops() {
    let a_of = |y: u128| -> u128 {
        primes_between(5, y as u64).iter().map(|&q| q as u128 - 2).product()
    };
    let p_of = |y: u128| -> u128 {
        primes_between(5, y as u64).iter().map(|&q| q as u128).product()
    };
    let strikes_slot = |y: u128| -> u128 {
        let p = p_of(y);
        primes_between(5, y as u64).iter().map(|&q| 2 * p / q as u128).sum()
    };

    println!("OPERATION COUNTS (machine-independent; one op = one elementary visit)");
    println!("\nA. TWIN SLOT LADDER, per rung y -> q'");
    println!("{:>10} {:>22} {:>22} {:>22} {:>22} {:>9}", "rung", "merge gen visits", "merge scanner pushes", "constr strikes", "constr cells = P", "ratio");
    let rungs: [(u128, u128); 7] =
        [(17, 19), (19, 23), (23, 29), (29, 31), (31, 37), (37, 41), (41, 43)];
    for (prev, qp) in rungs {
        let (gen, pushes): (u128, u128) = match qp {
            19 => (strikes_slot(17) + p_of(17), a_of(17)), // base sieve counted here
            23 => (19 * a_of(17), a_of(19)),
            29 => (23 * a_of(19), a_of(23)),
            31 => (29 * a_of(23), a_of(29)),
            37 => (31 * a_of(29), a_of(31)),
            41 => (37 * 31 * a_of(29), a_of(37)),
            43 => (41 * 37 * 31 * a_of(29), a_of(41)),
            _ => unreachable!(),
        };
        let strikes = {
            let p = p_of(qp);
            primes_between(5, qp as u64).iter().map(|&q| 2 * p / q as u128).sum::<u128>()
        };
        let cells = p_of(qp);
        let merge = gen + pushes;
        let constr = strikes + cells;
        println!(
            "{:>4} -> {:>2} {:>22} {:>22} {:>22} {:>22} {:>9.1}",
            prev, qp, gen, pushes, strikes, cells,
            constr as f64 / merge as f64
        );
    }
    println!("  (streamed rungs 37/41/43 share a fixed prologue: sieve P(23) + 29*A(23) visits to build word29 = {} ops)",
        strikes_slot(23) + p_of(23) + 29 * a_of(23));

    // B. h_2 family rungs. Sum over classes; per class c the level-17 word has
    // A_c = prod over q<=17 of (q - k_q(c)), k_q = 1 if q | c else 2, and the
    // sieve applies sum_q k_q * P17/q strikes + P17 cells.
    let h2p17: u128 = 255_255;
    let gears: [u128; 6] = [3, 5, 7, 11, 13, 17];
    let ncls17: u128 = h2p17 / 2 + 1;
    let mut sum_a: u128 = 0;
    let mut sieve17_strikes: u128 = 0;
    for c in 0..=(h2p17 / 2) {
        let mut a: u128 = 1;
        let mut s: u128 = 0;
        for &q in &gears {
            let k = if c % q == 0 { 1 } else { 2 };
            a *= q - k;
            s += k * (h2p17 / q);
        }
        sum_a += a;
        sieve17_strikes += s;
    }
    let sieve17 = sieve17_strikes + ncls17 * h2p17; // strikes + cells over all classes
    let p19h: u128 = h2p17 * 19;
    let p23h: u128 = p19h * 23;
    // merge 17->19: sieve all 17-words + feed each letter to 10 scanners
    let m19 = sieve17 + 10 * sum_a;
    // construction 19: every class e = 1..P19/2 sieved at period P19
    let n19: u128 = p19h / 2; // 2,424,922
    let c19: u128 = {
        let gears19: [u128; 7] = [3, 5, 7, 11, 13, 17, 19];
        let strikes: u128 = gears19
            .iter()
            .map(|&q| (p19h / q) * (2 * n19 - n19 / q))
            .sum();
        strikes + n19 * p19h
    };
    // merge 19->23: sieve 17-words + generate 19 laps for each of 19 residues
    // (19*19*A_c visits) + scan: sum over r19 of |word19| = A_c*(18 + 18*17)
    // letters, each fed to 12 scanners
    let m23 = sieve17 + 361 * sum_a + 12 * 324 * sum_a;
    let n23: u128 = p23h / 2; // 55,773,217
    let c23: u128 = {
        let gears23: [u128; 8] = [3, 5, 7, 11, 13, 17, 19, 23];
        let strikes: u128 = gears23
            .iter()
            .map(|&q| (p23h / q) * (2 * n23 - n23 / q))
            .sum();
        strikes + n23 * p23h
    };
    println!("\nB. h_2 FAMILY LADDER (sums over all difference classes)");
    println!("  sum over classes of A_c(17) = {} letters; aggregate 17-sieve = {} ops", sum_a, sieve17);
    println!("  17 -> 19: merge {} ops vs construction {} ops  (ratio {:.1})", m19, c19, c19 as f64 / m19 as f64);
    println!("  19 -> 23: merge {} ops vs construction {} ops  (ratio {:.1})", m23, c23, c23 as f64 / m23 as f64);
    println!("  23 -> 29: merge needs every level-23 word: gen 19*19*23*23*A_c + scans; ~{:.2e} ops - not run", (361u128 * 529 * sum_a * 21) as f64);

    // C. instrumented verification of the closed forms (small, runs inline)
    println!("\nC. INSTRUMENTED VERIFICATION");
    let (w17, p17) = slot_openings_u32(17);
    println!(
        "  slot sieve 17: survivors counted {} vs closed form A(17) = {} => {}",
        w17.len(),
        a_of(17),
        if w17.len() as u128 == a_of(17) { "MATCH" } else { "MISMATCH" }
    );
    // count pushes in a real 17 -> 19 scan
    let s = ((2 * gearsuite::slot::tooth_offset(19)) % 19) as u32;
    let mut sc = ChainScan::new(19, s);
    let mut pushes: u128 = 0;
    for &x in &w17 {
        sc.push_x(x as u64);
        pushes += 1;
    }
    for &x in w17.iter().take(256) {
        sc.push_x(x as u64 + p17);
        pushes += 1;
    }
    println!(
        "  17 -> 19 scan: pushes counted {} = A(17) + 256 margin; F = {} (must be 25)",
        pushes, sc.best
    );
    let mut h2buf = vec![true; h2p17 as usize];
    let mut h2word = Vec::new();
    let mut letters: u128 = 0;
    for c in 0..=(h2p17 / 2) as u32 {
        sieve_h2_word17(c, &mut h2buf, &mut h2word);
        letters += h2word.len() as u128;
    }
    println!(
        "  h_2 base: sum of word lengths counted {} vs closed form {} => {}",
        letters,
        sum_a,
        if letters == sum_a { "MATCH" } else { "MISMATCH" }
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("twin") => twin_ladder(),
        Some("construct") => construct(args[2].parse().unwrap()),
        Some("twin37") => twin_deep(37),
        Some("twin41") => twin_deep(41),
        Some("twin43") => twin_deep(43),
        Some("h2_17") => h2_17(),
        Some("h2_19") => h2_19(),
        Some("sample19") => sample19(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60)),
        Some("h2_23") => h2_23(args.get(2).map(|s| s == "pruned").unwrap_or(false)),
        Some("ops") => ops(),
        _ => {
            eprintln!("usage: h2ladder twin | construct <y> | twin37 | twin41 | twin43 | h2_17 | h2_19 | sample19 <K> | h2_23");
            std::process::exit(2);
        }
    }
}
